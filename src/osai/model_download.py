"""Verified, on-demand downloads from the official osCode model catalogue."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ssl
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import certifi

from .errors import ConfigurationError, ModelDownloadError, VerificationError
from .io import OutputLock, atomic_json

MODEL_REPOSITORY = "https://github.com/OmerDesignX/osCode-Models"
_RAW_REPOSITORY = f"{MODEL_REPOSITORY}/raw/refs/heads/main"
_RAW_TEXT_REPOSITORY = (
    "https://raw.githubusercontent.com/OmerDesignX/osCode-Models/main"
)
MODEL_MANIFEST = "OSCODE_MODEL.json"
_ALLOWED_DOWNLOAD_HOSTS = {"github.com", "raw.githubusercontent.com"}
_MAX_CATALOG_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


@dataclass(frozen=True, slots=True)
class ModelVariant:
    runtime: str
    tier: str
    repository_path: str
    folder: str
    bytes: int
    shards: int

    @property
    def format_directory(self) -> str:
        return "GGUF" if self.runtime == "llama.cpp" else "MLX"

    def destination(self, root: Path) -> Path:
        return root / self.format_directory / self.folder

    def primary_path(self, root: Path) -> Path:
        destination = self.destination(root)
        if self.runtime == "llama.cpp":
            return destination / Path(self.repository_path).name
        return destination


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    percent: int
    file: str
    bytes_received: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class ModelDownload:
    model: Path
    destination: Path
    manifest: Path
    runtime: str
    tier: str
    downloaded: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("model", "destination", "manifest"):
            payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    files: int
    bytes: int


MODEL_VARIANTS = (
    ModelVariant(
        "llama.cpp",
        "small",
        "GGUF/osCode-GGUF-Small-Q4_K_M-00001-of-00002.gguf",
        "small",
        2_708_804_288,
        2,
    ),
    ModelVariant(
        "llama.cpp",
        "medium",
        "GGUF/osCode-GGUF-Medium-Q6_K-00001-of-00002.gguf",
        "medium",
        3_464_055_456,
        2,
    ),
    ModelVariant(
        "llama.cpp",
        "large",
        "GGUF/osCode-GGUF-Large-Q8_0-00001-of-00003.gguf",
        "large",
        4_482_403_136,
        3,
    ),
    ModelVariant(
        "mlx",
        "small",
        "MLX/osCode-MLX-Small-Q5",
        "osCode-MLX-Small-Q5",
        2_912_931_406,
        21,
    ),
    ModelVariant(
        "mlx",
        "medium",
        "MLX/osCode-MLX-Medium-Q6",
        "osCode-MLX-Medium-Q6",
        3_701_329_697,
        27,
    ),
    ModelVariant(
        "mlx",
        "large",
        "MLX/osCode-MLX-Large-Q8",
        "osCode-MLX-Large-Q8",
        4_489_728_089,
        34,
    ),
)


class ConsoleProgress:
    """A dependency-free terminal progress bar written to stderr."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stderr
        self._started = False
        self._last: tuple[int, str] | None = None

    def __call__(self, progress: DownloadProgress) -> None:
        current = (progress.percent, progress.file)
        if current == self._last:
            return
        self._last = current
        width = 24
        filled = min(width, max(0, progress.percent * width // 100))
        bar = "#" * filled + "-" * (width - filled)
        received = _human_bytes(progress.bytes_received)
        total = _human_bytes(progress.total_bytes)
        name = progress.file[:48]
        self.stream.write(
            f"\r[{bar}] {progress.percent:3d}% {received}/{total} {name:<48}"
        )
        self.stream.flush()
        self._started = True
        if progress.percent >= 100:
            self.stream.write("\n")
            self.stream.flush()
            self._started = False

    def finish_error(self) -> None:
        if self._started:
            self.stream.write("\n")
            self.stream.flush()
            self._started = False


def variant_for(runtime: str, tier: str) -> ModelVariant:
    normalized_runtime = "llama.cpp" if runtime == "llamacpp" else runtime.lower()
    normalized_tier = tier.lower()
    for variant in MODEL_VARIANTS:
        if variant.runtime == normalized_runtime and variant.tier == normalized_tier:
            return variant
    raise ConfigurationError(
        "official model selection requires runtime mlx or llama.cpp and tier "
        "small, medium, or large"
    )


def files_for_variant(variant: ModelVariant) -> tuple[str, ...]:
    if variant.runtime == "llama.cpp":
        match = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", variant.repository_path)
        if match is None or int(match.group(2)) != variant.shards:
            raise VerificationError("the built-in GGUF model catalogue is invalid")
        return tuple(
            f"{match.group(1)}-{index:05d}-of-{match.group(2)}.gguf"
            for index in range(1, variant.shards + 1)
        )
    prefix = variant.repository_path
    return (
        f"{prefix}/config.json",
        f"{prefix}/chat_template.jinja",
        f"{prefix}/model.safetensors.index.json",
        f"{prefix}/tokenizer.json",
        f"{prefix}/tokenizer_config.json",
        f"{prefix}/README.md",
        *(
            f"{prefix}/model-{index:05d}-of-{variant.shards:05d}.safetensors"
            for index in range(1, variant.shards + 1)
        ),
    )


def ensure_official_model(
    root: str | Path,
    *,
    runtime: str,
    tier: str,
    allow_download: bool = True,
    progress: Callable[[DownloadProgress], None] | None = None,
) -> ModelDownload:
    model_root = Path(root).expanduser().resolve()
    variant = variant_for(runtime, tier)
    installed = _installed_download(model_root, variant)
    if installed is not None:
        return installed
    if _offline_requested():
        raise ModelDownloadError(
            f"the {tier} {runtime} model is not downloaded and OSAI_OFFLINE is enabled"
        )
    if not allow_download:
        raise ModelDownloadError(
            f"the {tier} {runtime} model is not downloaded; rerun without "
            "--no-download-model"
        )
    return download_model_variant(model_root, variant, progress=progress)


def download_model_variant(
    root: str | Path,
    variant: ModelVariant,
    *,
    progress: Callable[[DownloadProgress], None] | None = None,
) -> ModelDownload:
    model_root = Path(root).expanduser().resolve()
    downloads = model_root / ".downloads"
    destination = variant.destination(model_root)
    callback = progress or (lambda _progress: None)
    model_root.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)
    with OutputLock(downloads):
        installed = _installed_download(model_root, variant)
        if installed is not None:
            return installed
        if destination.exists():
            raise ModelDownloadError(
                f"an incomplete or unverified model directory already exists: {destination}; "
                "move or remove that directory before retrying"
            )
        staging = downloads / f"{variant.runtime}-{variant.tier}-{uuid.uuid4().hex}"
        activated = False
        try:
            staging.mkdir(parents=True)
            callback(DownloadProgress(0, "Checking model catalogue", 0, variant.bytes))
            release = _release_catalog()
            checksum_map = _checksum_catalog()
            published = _published_variant(release, variant)
            remote_files = _published_files(published, variant)
            _check_disk_budget(model_root, variant.bytes)
            received = 0
            manifest_files: list[dict[str, Any]] = []
            for repository_path in remote_files:
                relative = _local_relative_path(repository_path, variant)
                target = _safe_destination(staging, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                callback(
                    _progress(received, variant.bytes, Path(repository_path).name)
                )

                def on_chunk(count: int, *, name=Path(repository_path).name) -> None:
                    nonlocal received
                    received += count
                    callback(_progress(received, variant.bytes, name))

                size, digest = _download_file(repository_path, target, on_chunk)
                expected = checksum_map.get(repository_path)
                if expected is None or digest != expected:
                    raise VerificationError(
                        f"SHA-256 verification failed for {Path(repository_path).name}"
                    )
                manifest_files.append(
                    {"path": relative.as_posix(), "bytes": size, "sha256": digest}
                )
            manifest = staging / MODEL_MANIFEST
            atomic_json(
                manifest,
                {
                    "schema_version": 1,
                    "release": str(release["release"]),
                    "repository": MODEL_REPOSITORY,
                    "runtime": variant.runtime,
                    "tier": variant.tier,
                    "repository_path": variant.repository_path,
                    "published_bytes": variant.bytes,
                    "files": manifest_files,
                },
            )
            callback(
                DownloadProgress(
                    99, "Verifying downloaded files", received, variant.bytes
                )
            )
            verify_download_manifest(manifest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
            activated = True
            callback(DownloadProgress(100, "Ready", variant.bytes, variant.bytes))
            return ModelDownload(
                model=variant.primary_path(model_root),
                destination=destination,
                manifest=destination / MODEL_MANIFEST,
                runtime=variant.runtime,
                tier=variant.tier,
                downloaded=True,
            )
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ModelDownloadError(f"model download failed: {exc}") from exc
        except OSError as exc:
            raise ModelDownloadError(f"cannot store the downloaded model: {exc}") from exc
        finally:
            if not activated:
                shutil.rmtree(staging, ignore_errors=True)


def verify_download_manifest(manifest: str | Path) -> ManifestVerification:
    manifest_path = Path(manifest).expanduser().resolve()
    root = manifest_path.parent
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(
            f"invalid downloaded-model manifest {manifest_path}: {exc}"
        ) from exc
    if payload.get("schema_version") != 1 or payload.get("repository") != MODEL_REPOSITORY:
        raise VerificationError(f"unsupported downloaded-model manifest: {manifest_path}")
    try:
        variant = variant_for(str(payload.get("runtime")), str(payload.get("tier")))
    except ConfigurationError as exc:
        raise VerificationError(
            f"downloaded-model manifest has an unknown model variant: {manifest_path}"
        ) from exc
    if (
        payload.get("repository_path") != variant.repository_path
        or payload.get("published_bytes") != variant.bytes
    ):
        raise VerificationError(
            f"downloaded-model manifest does not match the catalogue: {manifest_path}"
        )
    records = payload.get("files")
    if not isinstance(records, list) or not records or len(records) > 1_000:
        raise VerificationError(
            f"downloaded-model manifest has an invalid file list: {manifest_path}"
        )
    required = {
        _local_relative_path(path, variant).as_posix()
        for path in files_for_variant(variant)
    }
    seen: set[str] = set()
    total = 0
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError(f"invalid file record in {manifest_path}")
        relative_value = record.get("path")
        digest = record.get("sha256")
        expected_size = record.get("bytes")
        if (
            not isinstance(relative_value, str)
            or relative_value in seen
            or not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or not isinstance(expected_size, int)
            or expected_size < 1
        ):
            raise VerificationError(f"invalid file record in {manifest_path}")
        target = _safe_destination(root, Path(relative_value))
        if not target.is_file() or target.stat().st_size != expected_size:
            raise VerificationError(f"downloaded model file is missing or truncated: {target}")
        with target.open("rb") as handle:
            prefix = handle.read(len(_LFS_PREFIX))
            if prefix == _LFS_PREFIX:
                raise VerificationError(f"downloaded file is still a Git LFS pointer: {target}")
            handle.seek(0)
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise VerificationError(f"SHA-256 mismatch for downloaded model file: {target}")
        seen.add(relative_value)
        total += expected_size
    if not required.issubset(seen):
        raise VerificationError(f"downloaded-model manifest is incomplete: {manifest_path}")
    return ManifestVerification(files=len(seen), bytes=total)


def _installed_download(root: Path, variant: ModelVariant) -> ModelDownload | None:
    destination = variant.destination(root)
    manifest = destination / MODEL_MANIFEST
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload["files"]
        if (
            payload.get("schema_version") != 1
            or payload.get("repository") != MODEL_REPOSITORY
            or payload.get("runtime") != variant.runtime
            or payload.get("tier") != variant.tier
            or payload.get("repository_path") != variant.repository_path
            or payload.get("published_bytes") != variant.bytes
            or not isinstance(records, list)
            or not records
        ):
            return None
        required = {
            _local_relative_path(path, variant).as_posix()
            for path in files_for_variant(variant)
        }
        seen: set[str] = set()
        for record in records:
            target = _safe_destination(destination, Path(record["path"]))
            if not target.is_file() or target.stat().st_size != record["bytes"]:
                return None
            seen.add(str(record["path"]))
        if not required.issubset(seen):
            return None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, VerificationError):
        return None
    return ModelDownload(
        model=variant.primary_path(root),
        destination=destination,
        manifest=manifest,
        runtime=variant.runtime,
        tier=variant.tier,
        downloaded=False,
    )


def _release_catalog() -> dict[str, Any]:
    try:
        payload = json.loads(_read_text(f"{_RAW_TEXT_REPOSITORY}/release.json"))
    except json.JSONDecodeError as exc:
        raise VerificationError("the remote osCode release catalogue is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise VerificationError("the remote osCode release catalogue is invalid")
    return payload


def _checksum_catalog() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in _read_text(f"{_RAW_TEXT_REPOSITORY}/SHA256SUMS").splitlines():
        match = re.fullmatch(r"([a-fA-F0-9]{64})\s+\./(.+)", line.strip())
        if match:
            relative = match.group(2).replace("\\", "/")
            if "../" not in relative and not relative.startswith("/"):
                result[relative] = match.group(1).lower()
    if not result:
        raise VerificationError("the remote osCode checksum catalogue is empty")
    return result


def _published_variant(release: dict[str, Any], variant: ModelVariant) -> dict[str, Any]:
    release_name = release.get("release")
    published_variants = release.get("variants")
    if re.fullmatch(r"\d+(?:\.\d+)?", str(release_name or "")) is None or not isinstance(
        published_variants, list
    ):
        raise VerificationError("the remote osCode release catalogue is invalid")
    for published in published_variants:
        if not isinstance(published, dict):
            continue
        runtime = str(published.get("runtime", "")).lower()
        normalized_runtime = "llama.cpp" if runtime == "llama.cpp" else runtime
        if normalized_runtime == variant.runtime and published.get("tier") == variant.tier:
            if (
                published.get("path") != variant.repository_path
                or published.get("bytes") != variant.bytes
            ):
                raise VerificationError(
                    "the published model catalogue does not match this osAi release"
                )
            return published
    raise VerificationError("the selected model is absent from the published catalogue")


def _published_files(published: dict[str, Any], variant: ModelVariant) -> tuple[str, ...]:
    raw_files = published.get("files")
    if raw_files is None:
        return files_for_variant(variant)
    if not isinstance(raw_files, list) or not raw_files or len(raw_files) > 1_000:
        raise VerificationError("the published model file list is invalid")
    files = tuple(
        str(value).replace("\\", "/").removeprefix("./") for value in raw_files
    )
    allowed_prefix = (
        f"{variant.repository_path}/"
        if variant.runtime == "mlx"
        else f"{Path(variant.repository_path).parent.as_posix()}/"
    )
    if len(set(files)) != len(files) or any(
        not path.startswith(allowed_prefix)
        or "../" in path
        or path.endswith("/")
        for path in files
    ):
        raise VerificationError("the published model file list leaves its model folder")
    if not set(files_for_variant(variant)).issubset(files):
        raise VerificationError("the published model file list is incomplete")
    return files


def _read_text(url: str) -> str:
    with _open_response(url) as response:
        _validate_response_url(response)
        payload = response.read(_MAX_CATALOG_BYTES + 1)
    if len(payload) > _MAX_CATALOG_BYTES:
        raise VerificationError("the remote model catalogue is unexpectedly large")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("the remote model catalogue is not UTF-8") from exc


def _download_file(
    repository_path: str,
    destination: Path,
    on_chunk: Callable[[int], None],
) -> tuple[int, str]:
    encoded = "/".join(quote(part, safe="") for part in repository_path.split("/"))
    digest = hashlib.sha256()
    size = 0
    with _open_response(f"{_RAW_REPOSITORY}/{encoded}") as response:
        _validate_response_url(response)
        with destination.open("xb") as output:
            while True:
                chunk = response.read(_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                on_chunk(len(chunk))
    if size < 1:
        raise VerificationError(f"downloaded an empty model file: {repository_path}")
    return size, digest.hexdigest()


def _open_response(url: str) -> BinaryIO:
    request = Request(url, headers={"User-Agent": "osAi-model-downloader/0.1.0"})
    return urlopen(  # noqa: S310 - fixed HTTPS hosts are verified below
        request, timeout=60, context=_TLS_CONTEXT
    )


def _validate_response_url(response: BinaryIO) -> None:
    geturl = getattr(response, "geturl", None)
    final_url = str(geturl()) if callable(geturl) else ""
    parsed = urlparse(final_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host in _ALLOWED_DOWNLOAD_HOSTS or host.endswith(".githubusercontent.com")
    ):
        raise VerificationError(f"model download redirected to an untrusted host: {host}")


def _local_relative_path(repository_path: str, variant: ModelVariant) -> Path:
    if variant.runtime == "llama.cpp":
        return Path(Path(repository_path).name)
    prefix = f"{variant.repository_path}/"
    if not repository_path.startswith(prefix):
        raise VerificationError("the MLX download path leaves its model folder")
    return Path(repository_path.removeprefix(prefix))


def _safe_destination(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise VerificationError("unsafe downloaded-model path")
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise VerificationError("downloaded-model path escapes its directory") from exc
    return destination


def _check_disk_budget(root: Path, model_bytes: int) -> None:
    available = shutil.disk_usage(root).free
    required = model_bytes + 512 * 1024**2
    if available < required:
        raise ModelDownloadError(
            f"not enough free disk space: need {_human_bytes(required)}, "
            f"have {_human_bytes(available)}"
        )


def _progress(received: int, total: int, name: str) -> DownloadProgress:
    percent = min(99, max(0, int(received * 100 / total)))
    return DownloadProgress(percent, name, received, total)


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f}{unit}"
        amount /= 1024
    return f"{amount:.1f}TiB"


def _offline_requested() -> bool:
    return os.environ.get("OSAI_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
