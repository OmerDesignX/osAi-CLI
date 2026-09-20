"""Official downloadable and custom local model catalogue."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, ModelFormatError
from .formats import inspect_gguf, inspect_mlx
from .hardware import Engine, HardwareReport, detect_hardware, select_engine
from .model_download import (
    DEFAULT_MODEL_VERSION,
    MODEL_VERSIONS,
    installed_model_path,
    variant_for,
)
from .paths import project_root


class ModelTier(str, Enum):
    XSMALL = "xsmall"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    name: str
    source: str
    tier: str | None
    version: str | None
    mlx: Path | None
    gguf: Path | None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["mlx"] = str(self.mlx) if self.mlx else None
        result["gguf"] = str(self.gguf) if self.gguf else None
        result["mlx_materialized"] = _mlx_materialized(self.mlx) if self.mlx else False
        result["gguf_materialized"] = _gguf_materialized(self.gguf) if self.gguf else False
        return result


@dataclass(frozen=True, slots=True)
class ModelSelection:
    entry: CatalogEntry
    engine: Engine
    model: Path
    companion_mlx: Path | None
    hardware: HardwareReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.as_dict(),
            "engine": self.engine.value,
            "model": str(self.model),
            "companion_mlx": str(self.companion_mlx) if self.companion_mlx else None,
            "hardware": self.hardware.as_dict(),
        }


def bundled_root() -> Path:
    return project_root() / "osCode-Models"


def custom_root() -> Path:
    return project_root() / "models" / "custom"


def bundled_entry(
    tier: str | ModelTier,
    root: Path | None = None,
    version: str = DEFAULT_MODEL_VERSION,
) -> CatalogEntry:
    try:
        value = tier if isinstance(tier, ModelTier) else ModelTier(tier)
    except ValueError as exc:
        raise ConfigurationError("tier must be xsmall, small, medium, or large") from exc
    if version not in MODEL_VERSIONS or (version == "v1" and value is ModelTier.XSMALL):
        raise ConfigurationError("xsmall is available only for v2; version must be v1 or v2")
    model_root = (root or bundled_root()).expanduser().resolve()
    mlx = variant_for("mlx", value.value, version)
    gguf = variant_for("llama.cpp", value.value, version)
    return CatalogEntry(
        name=f"oscode-{value.value}",
        source="official",
        tier=value.value,
        version=version,
        mlx=installed_model_path(model_root, mlx),
        gguf=installed_model_path(model_root, gguf),
    )


def custom_entry(name: str, root: Path | None = None) -> CatalogEntry:
    if not _SAFE_NAME.fullmatch(name):
        raise ConfigurationError(
            "custom model name must contain only letters, digits, dots, underscores, or hyphens"
        )
    location = (root or custom_root()).expanduser().resolve()
    folder = (location / name).resolve()
    try:
        folder.relative_to(location)
    except ValueError as exc:
        raise ConfigurationError("custom model folder must remain inside the custom root") from exc
    if not folder.is_dir():
        raise ModelFormatError(f"custom model folder does not exist: {folder}")
    mlx = _contained_path(folder / "mlx", folder, "custom MLX folder")
    gguf_dir = _contained_path(folder / "gguf", folder, "custom GGUF folder")
    gguf = _choose_gguf(gguf_dir) if gguf_dir.is_dir() else None
    return CatalogEntry(
        name=name,
        source="custom",
        tier=None,
        version=None,
        mlx=mlx if (mlx / "config.json").is_file() else None,
        gguf=gguf,
    )


def list_catalog(
    *,
    bundled: Path | None = None,
    custom: Path | None = None,
    version: str = DEFAULT_MODEL_VERSION,
) -> tuple[CatalogEntry, ...]:
    entries = [
        bundled_entry(tier, bundled, version)
        for tier in ModelTier
        if version == "v2" or tier is not ModelTier.XSMALL
    ]
    location = (custom or custom_root()).expanduser().resolve()
    if location.is_dir():
        for folder in sorted(location.iterdir(), key=lambda item: item.name.lower()):
            if folder.is_dir() and _SAFE_NAME.fullmatch(folder.name):
                entries.append(custom_entry(folder.name, location))
    return tuple(entries)


def resolve_model(
    *,
    tier: str | ModelTier | None = None,
    custom: str | None = None,
    engine: str | Engine = Engine.AUTO,
    bundled: Path | None = None,
    custom_models: Path | None = None,
    hardware: HardwareReport | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> ModelSelection:
    if (tier is None) == (custom is None):
        raise ConfigurationError("select exactly one bundled tier or custom model")
    entry = (
        bundled_entry(tier, bundled, model_version)
        if tier is not None
        else custom_entry(str(custom), custom_models)
    )
    report = hardware or detect_hardware()
    try:
        requested = engine if isinstance(engine, Engine) else Engine(engine)
    except ValueError as exc:
        raise ConfigurationError("engine must be auto, mlx, or llama.cpp") from exc
    selected = select_engine(requested, report)
    preferred = entry.mlx if selected is Engine.MLX else entry.gguf
    if preferred is None and requested is Engine.AUTO:
        selected = Engine.LLAMA_CPP if selected is Engine.MLX else Engine.MLX
        preferred = entry.mlx if selected is Engine.MLX else entry.gguf
    if preferred is None:
        raise ModelFormatError(f"{entry.name} has no local {selected.value} model")
    try:
        _inspect_selection(selected, preferred)
    except ModelFormatError:
        alternate_engine = Engine.LLAMA_CPP if selected is Engine.MLX else Engine.MLX
        alternate = entry.gguf if alternate_engine is Engine.LLAMA_CPP else entry.mlx
        if requested is not Engine.AUTO or alternate is None:
            raise
        _inspect_selection(alternate_engine, alternate)
        selected, preferred = alternate_engine, alternate
    return ModelSelection(
        entry=entry,
        engine=selected,
        model=preferred,
        companion_mlx=(
            entry.mlx
            if selected is Engine.LLAMA_CPP
            and entry.mlx is not None
            and _mlx_materialized(entry.mlx)
            else None
        ),
        hardware=report,
    )


def _choose_gguf(folder: Path) -> Path | None:
    candidates = [
        _contained_path(path, folder, "custom GGUF file") for path in sorted(folder.glob("*.gguf"))
    ]
    first_shards = [path for path in candidates if "-00001-of-" in path.name]
    choices = first_shards or candidates
    if not choices:
        return None
    if len(choices) > 1:
        raise ModelFormatError(
            f"custom GGUF folder is ambiguous; keep one model or split model in {folder}"
        )
    return choices[0]


def _contained_path(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"{label} must remain inside {root}") from exc
    return resolved


def _inspect_selection(engine: Engine, path: Path) -> None:
    if engine is Engine.MLX:
        inspect_mlx(path)
    else:
        inspect_gguf(path)


def _mlx_materialized(path: Path) -> bool:
    try:
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = {path / name for name in index["weight_map"].values()}
        return bool(shards) and all(_materialized_file(shard) for shard in shards)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _gguf_materialized(path: Path) -> bool:
    try:
        from .formats import discover_gguf_shards

        return all(_materialized_file(shard) for shard in discover_gguf_shards(path))
    except (OSError, ModelFormatError):
        return False


def _materialized_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return not handle.read(42).startswith(b"version https://git-lfs")
    except OSError:
        return False
