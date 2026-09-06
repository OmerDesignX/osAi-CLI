"""Lossless self-contained bundles for quantized models and residual adapters."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, VerificationError
from .io import atomic_json, sha256_file
from .session import clone_or_copy

FUSION_MANIFEST = "osai_fusion.json"
MLX_EMBEDDED_ADAPTER = "osai_adapter"
MLX_FUSION_KIND = "osai-mlx-quantized-residual-v1"
GGUF_FUSION_KIND = "osai-gguf-quantized-residual-v1"
GGUF_EMBEDDED_ADAPTER = "osai_adapter.gguf"


@dataclass(frozen=True, slots=True)
class MlxFusionBundle:
    path: Path
    manifest: Path
    adapter: Path
    base_files: tuple[Path, ...]
    copy_modes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        result["manifest"] = str(self.manifest)
        result["adapter"] = str(self.adapter)
        result["base_files"] = [str(path) for path in self.base_files]
        return result


@dataclass(frozen=True, slots=True)
class GgufFusionBundle:
    path: Path
    manifest: Path
    model: Path
    shards: tuple[Path, ...]
    adapter: Path
    copy_modes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("path", "manifest", "model", "adapter"):
            result[key] = str(result[key])
        result["shards"] = [str(path) for path in self.shards]
        return result


def create_mlx_fusion_bundle(
    base: str | Path,
    adapter: str | Path,
    destination: str | Path,
) -> MlxFusionBundle:
    """Copy an unchanged quantized MLX model and embed its exact LoRA residual."""

    base_path = Path(base).expanduser().resolve()
    adapter_path = Path(adapter).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    _validate_mlx_sources(base_path, adapter_path, target)

    base_files: list[Path] = []
    modes: list[str] = []
    try:
        target.mkdir(parents=True)
        for source in sorted(base_path.rglob("*")):
            if not source.is_file() or ".git" in source.parts:
                continue
            relative = source.relative_to(base_path)
            copied = target / relative
            modes.append(clone_or_copy(source, copied))
            base_files.append(copied)

        embedded = target / MLX_EMBEDDED_ADAPTER
        for source in sorted(adapter_path.rglob("*")):
            if not source.is_file() or ".git" in source.parts:
                continue
            copied = embedded / source.relative_to(adapter_path)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, copied)

        manifest = target / FUSION_MANIFEST
        atomic_json(
            manifest,
            {
                "schema_version": 1,
                "kind": MLX_FUSION_KIND,
                "format": "mlx",
                "adapter_path": MLX_EMBEDDED_ADAPTER,
                "base_weights_unchanged": True,
                "adapter_residual_embedded": True,
                "requires_full_precision_intermediate": False,
            },
        )
        resolved = resolve_mlx_fusion_adapter(target)
        if resolved != embedded.resolve():
            raise VerificationError("embedded MLX adapter path did not round-trip")
        return MlxFusionBundle(
            path=target,
            manifest=manifest,
            adapter=embedded,
            base_files=tuple(base_files),
            copy_modes=tuple(modes),
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def resolve_mlx_fusion_adapter(model: str | Path) -> Path | None:
    """Return a validated embedded adapter path when an osai manifest is present."""

    root = Path(model).expanduser().resolve()
    manifest = root / FUSION_MANIFEST
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid MLX fusion manifest {manifest}: {exc}") from exc
    if payload.get("schema_version") != 1 or payload.get("kind") != MLX_FUSION_KIND:
        raise VerificationError(f"unsupported MLX fusion manifest: {manifest}")
    relative_value = payload.get("adapter_path")
    if not isinstance(relative_value, str) or not relative_value:
        raise VerificationError(f"MLX fusion manifest has no adapter_path: {manifest}")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise VerificationError(f"unsafe embedded adapter path in {manifest}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise VerificationError(
            f"embedded adapter escapes its model directory: {manifest}"
        ) from exc
    required = (resolved / "adapter_config.json", resolved / "adapters.safetensors")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise VerificationError("missing embedded MLX adapter file(s): " + ", ".join(missing))
    return resolved


def create_gguf_fusion_bundle(
    model: str | Path,
    shards: tuple[Path, ...],
    adapter: str | Path,
    destination: str | Path,
) -> GgufFusionBundle:
    """Copy exact GGUF shards and embed their exact LoRA adapter."""

    model_path = Path(model).expanduser().resolve()
    source_shards = tuple(Path(path).expanduser().resolve() for path in shards)
    adapter_path = Path(adapter).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if model_path not in source_shards:
        raise ConfigurationError("primary GGUF model must be one of its shards")
    missing = [str(path) for path in source_shards if not path.is_file()]
    if missing:
        raise ConfigurationError("missing GGUF fusion shard(s): " + ", ".join(missing))
    if not adapter_path.is_file() or adapter_path.stat().st_size == 0:
        raise ConfigurationError(f"missing non-empty GGUF adapter: {adapter_path}")
    if target.exists():
        raise ConfigurationError(f"refusing to overwrite GGUF fusion destination: {target}")

    modes: list[str] = []
    copied_shards: list[Path] = []
    try:
        model_directory = target / "model"
        for source in source_shards:
            copied = model_directory / source.name
            modes.append(clone_or_copy(source, copied))
            copied_shards.append(copied)
        copied_adapter = target / GGUF_EMBEDDED_ADAPTER
        copied_adapter.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_path, copied_adapter)
        copied_model = model_directory / model_path.name
        manifest = target / FUSION_MANIFEST
        atomic_json(
            manifest,
            {
                "schema_version": 1,
                "kind": GGUF_FUSION_KIND,
                "format": "gguf",
                "model_path": str(copied_model.relative_to(target)),
                "shards": [str(path.relative_to(target)) for path in copied_shards],
                "adapter_path": GGUF_EMBEDDED_ADAPTER,
                "base_weights_unchanged": True,
                "adapter_residual_embedded": True,
                "requires_full_precision_intermediate": False,
            },
        )
        resolved = resolve_gguf_fusion_bundle(target)
        _verify_exact_files(source_shards, resolved.shards)
        _verify_exact_files((adapter_path,), (resolved.adapter,))
        return GgufFusionBundle(
            path=target,
            manifest=manifest,
            model=resolved.model,
            shards=resolved.shards,
            adapter=resolved.adapter,
            copy_modes=tuple(modes),
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def resolve_gguf_fusion_bundle(model: str | Path) -> GgufFusionBundle:
    root = Path(model).expanduser().resolve()
    manifest = root / FUSION_MANIFEST
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid GGUF fusion manifest {manifest}: {exc}") from exc
    if payload.get("schema_version") != 1 or payload.get("kind") != GGUF_FUSION_KIND:
        raise VerificationError(f"unsupported GGUF fusion manifest: {manifest}")
    model_path = _safe_bundle_path(root, payload.get("model_path"), manifest)
    adapter_path = _safe_bundle_path(root, payload.get("adapter_path"), manifest)
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise VerificationError(f"GGUF fusion manifest has no shards: {manifest}")
    shard_paths = tuple(_safe_bundle_path(root, value, manifest) for value in raw_shards)
    required = (*shard_paths, model_path, adapter_path)
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise VerificationError("missing GGUF fusion file(s): " + ", ".join(missing))
    if model_path not in shard_paths:
        raise VerificationError(f"GGUF fusion primary model is not a listed shard: {manifest}")
    return GgufFusionBundle(
        path=root,
        manifest=manifest,
        model=model_path,
        shards=shard_paths,
        adapter=adapter_path,
        copy_modes=(),
    )


def _safe_bundle_path(root: Path, value: object, manifest: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"missing bundle path in {manifest}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise VerificationError(f"unsafe bundle path in {manifest}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise VerificationError(f"bundle path escapes its directory: {manifest}") from exc
    return resolved


def _verify_exact_files(sources: tuple[Path, ...], copies: tuple[Path, ...]) -> None:
    if len(sources) != len(copies):
        raise VerificationError("lossless fusion file count changed")
    for source, copied in zip(sources, copies, strict=True):
        if source.stat().st_size != copied.stat().st_size:
            raise VerificationError(f"lossless fusion changed file size: {source.name}")
        if sha256_file(source) != sha256_file(copied):
            raise VerificationError(f"lossless fusion changed file bytes: {source.name}")


def _validate_mlx_sources(base: Path, adapter: Path, destination: Path) -> None:
    if not base.is_dir():
        raise ConfigurationError(f"MLX fusion base does not exist: {base}")
    if (base / FUSION_MANIFEST).exists():
        raise ConfigurationError("cannot fuse another adapter into an already fused MLX bundle")
    if not adapter.is_dir():
        raise ConfigurationError(f"MLX adapter directory does not exist: {adapter}")
    for name in ("adapter_config.json", "adapters.safetensors"):
        path = adapter / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ConfigurationError(f"missing non-empty MLX adapter file: {path}")
    if destination.exists():
        raise ConfigurationError(f"refusing to overwrite MLX fusion destination: {destination}")
