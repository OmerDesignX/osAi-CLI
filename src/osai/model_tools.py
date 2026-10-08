"""Find an osAi base and adapter bundle and export its fused model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .errors import ConfigurationError
from .fusion import resolve_gguf_fusion_bundle, resolve_mlx_fusion_adapter
from .merged_export import export_gguf_weights, export_mlx_weights


def find_model_bundle(source: Path) -> Path:
    root = source.expanduser().resolve()
    candidates = (
        root,
        root / "gguf",
        root / "mlx",
        root / "merged-model" / "gguf",
        root / "merged-model" / "mlx",
        root / "checkpoint" / "merged-model" / "gguf",
        root / "checkpoint" / "merged-model" / "mlx",
        root / "outputs" / "gguf",
        root / "outputs" / "mlx",
        root / "outputs" / "merged-model" / "gguf",
        root / "outputs" / "merged-model" / "mlx",
        root / "outputs" / "checkpoint" / "merged-model" / "gguf",
        root / "outputs" / "checkpoint" / "merged-model" / "mlx",
    )
    for candidate in candidates:
        if (candidate / "osai_fusion.json").is_file():
            return candidate
    raise ConfigurationError("Choose a model folder or osAi session containing osai_fusion.json")


def export_bundle(source: Path, output: Path, *, python: str | Path | None = None) -> Path:
    bundle = find_model_bundle(source)
    manifest = json.loads((bundle / "osai_fusion.json").read_text(encoding="utf-8"))
    kind = manifest.get("format")
    target_dir = output.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="osai-export-") as temporary:
        work = Path(temporary)
        if kind == "gguf":
            resolved = resolve_gguf_fusion_bundle(bundle)
            return export_gguf_weights(
                resolved.model,
                resolved.shards,
                resolved.adapter,
                target_dir / "merged.gguf",
                work=work,
                log=work / "export-merged.log",
            )
        if kind == "mlx":
            adapter = resolve_mlx_fusion_adapter(bundle)
            if adapter is None:
                raise ConfigurationError("MLX bundle has no adapter")
            config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
            return export_mlx_weights(
                bundle,
                adapter,
                target_dir / "merged",
                work=work,
                log=work / "export-merged.log",
                python=python,
                multimodal="vision_config" in config,
            )
    raise ConfigurationError(f"unsupported bundle format: {kind}")
