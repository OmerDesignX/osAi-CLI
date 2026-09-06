"""Stable on-disk layout for one training and publication session."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ModelFormat
from .dataset import DatasetSummary
from .errors import ConfigurationError, VerificationError
from .formats import ModelInspection
from .io import atomic_json, sha256_file


def timestamped_session_path(
    root: str | Path,
    label: str,
    *,
    now: datetime | None = None,
) -> Path:
    """Reserve an unused, readable session directory below the configured root."""
    session_root = Path(root).expanduser().resolve()
    session_root.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M-%S")
    slug = "-".join(part for part in _slug(label).split("-") if part) or "run"
    candidate = session_root / f"{stamp}_{slug}"
    suffix = 2
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = session_root / f"{stamp}_{slug}-{suffix}"
            suffix += 1


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value)


@dataclass(frozen=True, slots=True)
class SessionLayout:
    root: Path

    @classmethod
    def at(cls, root: str | Path) -> SessionLayout:
        return cls(Path(root).expanduser().resolve())

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def work(self) -> Path:
        return self.root / ".internal"

    @property
    def rollouts(self) -> Path:
        return self.root / "rollouts"

    @property
    def base_adapter(self) -> Path:
        return self.root / "outputs" / "base-plus-adapter"

    @property
    def adapters(self) -> Path:
        return self.base_adapter / "adapters"

    @property
    def merged(self) -> Path:
        return self.root / "outputs" / "merged-model"

    @property
    def run_manifest(self) -> Path:
        return self.manifests / "run.json"

    @property
    def progress_manifest(self) -> Path:
        return self.manifests / "progress.json"

    @property
    def deployment_manifest(self) -> Path:
        return self.base_adapter / "deployment.json"

    @property
    def dataset_manifest(self) -> Path:
        return self.manifests / "dataset.json"

    def create(self) -> None:
        for path in (
            self.manifests,
            self.logs,
            self.work,
            self.rollouts,
            self.adapters,
            self.merged,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class BaseBundleResult:
    path: Path | None
    files: tuple[Path, ...]
    link_modes: tuple[str, ...]
    materialized: bool

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path) if self.path else None
        result["files"] = [str(path) for path in self.files]
        return result


def record_dataset(layout: SessionLayout, dataset: DatasetSummary) -> Path:
    files = []
    for split in ("train", "valid", "test"):
        path = dataset.path / f"{split}.jsonl"
        if path.is_file():
            files.append(
                {
                    "split": split,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "schema_version": 1,
        "source": str(dataset.path),
        "schema": dataset.schema,
        "examples": {
            "train": dataset.train_examples,
            "valid": dataset.valid_examples,
            "test": dataset.test_examples,
        },
        "files": files,
    }
    atomic_json(layout.dataset_manifest, payload)
    return layout.dataset_manifest


def publish_base_adapter_bundle(
    layout: SessionLayout,
    base: ModelInspection,
    adapters: dict[str, Path],
    *,
    materialize_base: bool,
) -> BaseBundleResult:
    """Publish a deployable base-plus-adapter bundle without mutating the source."""

    layout.create()
    _validate_adapters(layout, adapters)
    base_result = (
        _materialize_base(layout, base)
        if materialize_base
        else BaseBundleResult(None, (), (), False)
    )
    payload = {
        "schema_version": 1,
        "kind": "base-plus-lora-adapter",
        "format": base.format.value,
        "base_source": str(base.path),
        "base": base_result.as_dict(),
        "adapters": {
            name: str(path.resolve().relative_to(layout.base_adapter))
            for name, path in sorted(adapters.items())
        },
        "quantization": asdict(base.quantization),
        "architecture": base.architecture,
        "context_length": base.context_length,
        "base_files_unchanged": True,
    }
    atomic_json(layout.deployment_manifest, payload)
    return base_result


def _validate_adapters(layout: SessionLayout, adapters: dict[str, Path]) -> None:
    if not adapters:
        raise VerificationError("a base-plus-adapter bundle needs at least one adapter")
    for name, raw in adapters.items():
        path = raw.expanduser().resolve()
        if not path.exists():
            raise VerificationError(f"missing {name} adapter: {path}")
        try:
            path.relative_to(layout.base_adapter)
        except ValueError as exc:
            raise ConfigurationError(
                f"published adapter must be inside {layout.base_adapter}: {path}"
            ) from exc


def _materialize_base(layout: SessionLayout, base: ModelInspection) -> BaseBundleResult:
    destination = layout.base_adapter / "base"
    files: list[Path] = []
    modes: list[str] = []
    if base.format is ModelFormat.MLX:
        model_root = destination / "model"
        for source in sorted(base.path.rglob("*")):
            if not source.is_file() or ".git" in source.parts:
                continue
            relative = source.relative_to(base.path)
            target = model_root / relative
            modes.append(clone_or_copy(source, target))
            files.append(target)
        published_path = model_root
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for source in base.shards:
            target = destination / source.name
            modes.append(clone_or_copy(source, target))
            files.append(target)
        published_path = destination / base.path.name
    return BaseBundleResult(published_path, tuple(files), tuple(modes), True)


def clone_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise VerificationError(
                f"refusing to replace existing published base path: {destination}"
            )
        if source.samefile(destination):
            destination.unlink()
        elif (
            destination.stat().st_size == source.stat().st_size
            and sha256_file(destination) == sha256_file(source)
        ):
            return "existing"
        else:
            raise VerificationError(
                f"refusing to replace existing published base file: {destination}"
            )
    clone_command: list[str] | None = None
    if sys.platform == "darwin":
        clone_command = ["/bin/cp", "-c", "-p", str(source), str(destination)]
    elif sys.platform.startswith("linux") and (cp := shutil.which("cp")):
        clone_command = [
            cp,
            "--reflink=always",
            "--preserve=mode,timestamps",
            "--",
            str(source),
            str(destination),
        ]
    if clone_command is not None:
        completed = subprocess.run(
            clone_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            return "copy-on-write-clone"
        destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)
    return "copy"
