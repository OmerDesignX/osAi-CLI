"""Publish lossless quantized-residual bundles beside adapter deployments."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backends.mlx import MlxBackend
from .backends.mlx_vlm import MlxVlmBackend
from .config import ModelFormat
from .errors import ConfigurationError, DependencyError, TrainingError, VerificationError
from .formats import ModelInspection, inspect_model
from .fusion import (
    create_gguf_fusion_bundle,
    create_mlx_fusion_bundle,
    resolve_gguf_fusion_bundle,
)
from .hardware import Accelerator, select_llama_accelerator
from .merged_export import export_gguf_weights, export_mlx_weights
from .offline import offline_environment
from .paths import llama_binary
from .process import run_logged
from .session import SessionLayout


@dataclass(frozen=True, slots=True)
class MergedModelResult:
    path: Path
    format: str
    quantization: dict[str, Any]
    size_bytes: int
    validation_log: Path
    temporary_full_precision_intermediate: bool
    merge_strategy: str
    temporary_quantized_unified_base: bool

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        result["validation_log"] = str(self.validation_log)
        return result


def merge_mlx_model(
    base: ModelInspection,
    adapter_dir: Path,
    layout: SessionLayout,
    *,
    python: str | Path | None = None,
    accelerator: str | Accelerator = Accelerator.AUTO,
    multimodal: bool = False,
) -> MergedModelResult:
    if base.format is not ModelFormat.MLX:
        raise ConfigurationError("MLX merge requires an MLX base")
    layout.create()
    destination = layout.merged / "mlx"
    stage = Path(tempfile.mkdtemp(prefix="mlx-merge-", dir=layout.work)) / "model"
    backend = (
        MlxVlmBackend(python=python, accelerator=accelerator)
        if multimodal
        else MlxBackend(python=python, accelerator=accelerator)
    )
    validation_log = layout.logs / "validate-merged-mlx.log"
    try:
        backend.preflight()
        if destination.exists():
            shutil.copy2(adapter_dir / "adapters.safetensors", destination / "osai_adapter" / "adapters.safetensors")
            shutil.copy2(adapter_dir / "adapter_config.json", destination / "osai_adapter" / "adapter_config.json")
            bundle = destination
        else:
            create_mlx_fusion_bundle(base.path, adapter_dir, stage)
            bundle = stage
        merged = inspect_model(bundle, ModelFormat.MLX)
        _verify_merged_metadata(base, merged)
        backend.verify_fusion(base.path, adapter_dir, bundle, log_path=validation_log)
        export_mlx_weights(base.path, adapter_dir, bundle / "merged", work=layout.work,
                           log=layout.logs / "export-merged-mlx.log", python=backend.python,
                           multimodal=multimodal)
        inspect_model(bundle / "merged", ModelFormat.MLX)
        if bundle == stage:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, destination)
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)
    published = inspect_model(destination, ModelFormat.MLX)
    return MergedModelResult(
        path=destination,
        format=ModelFormat.MLX.value,
        quantization=asdict(published.quantization),
        size_bytes=sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
        validation_log=validation_log,
        temporary_full_precision_intermediate=False,
        merge_strategy="fused-mlx-weights-and-adapter",
        temporary_quantized_unified_base=False,
    )


def merge_gguf_model(
    base: ModelInspection,
    adapter: Path,
    layout: SessionLayout,
    *,
    accelerator: str | Accelerator = Accelerator.AUTO,
    threads: int = 2,
) -> MergedModelResult:
    if base.format is not ModelFormat.GGUF:
        raise ConfigurationError("GGUF merge requires a GGUF base")
    if threads < 1:
        raise ConfigurationError("merge threads must be at least 1")
    completion = llama_binary("llama-cli")
    if completion is None:
        raise DependencyError(
            "missing llama-completion binary; run `osai build-llama`"
        )
    layout.create()
    destination = layout.merged / "gguf"
    stage_root = Path(tempfile.mkdtemp(prefix="gguf-merge-", dir=layout.work))
    stage = stage_root / "gguf"
    validation_log = layout.logs / "validate-merged-gguf.log"
    try:
        if destination.exists():
            bundle = resolve_gguf_fusion_bundle(destination)
            export_target = destination
        else:
            bundle = create_gguf_fusion_bundle(base.path, base.shards, adapter, stage)
            export_target = stage
        merged = inspect_model(bundle.model, ModelFormat.GGUF)
        _verify_merged_metadata(base, merged)
        export_gguf_weights(bundle.model, bundle.shards, adapter,
                            export_target / "merged.gguf", work=layout.work,
                            log=layout.logs / "export-merged-gguf.log", threads=threads)
        if destination.exists():
            pending_adapter = bundle.adapter.with_name("osai_adapter.gguf.pending")
            shutil.copy2(adapter, pending_adapter)
            os.replace(pending_adapter, bundle.adapter)
        _validate_gguf_load(
            completion,
            export_target / "merged.gguf",
            validation_log,
            accelerator=accelerator,
            threads=threads,
        )
        if export_target == stage:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, destination)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    published_bundle = resolve_gguf_fusion_bundle(destination)
    published = inspect_model(destination / "merged.gguf", ModelFormat.GGUF)
    return MergedModelResult(
        path=destination,
        format=ModelFormat.GGUF.value,
        quantization=asdict(published.quantization),
        size_bytes=sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
        validation_log=validation_log,
        temporary_full_precision_intermediate=False,
        merge_strategy="quantized-tensor-fusion-and-adapter",
        temporary_quantized_unified_base=False,
    )


def _validate_gguf_load(
    binary: Path,
    model: Path,
    log_path: Path,
    *,
    accelerator: str | Accelerator,
    adapter: Path | None = None,
    threads: int = 2,
) -> None:
    selected = select_llama_accelerator(accelerator)
    common = [
        str(binary),
        "-m",
        str(model),
        "-p",
        "Hello",
        "-n",
        "1",
        "-c",
        "32",
        "--temp",
        "0",
        "--offline",
        "--simple-io",
        "--no-display-prompt",
        "-no-cnv",
        "--log-colors",
        "off",
        "-t",
        str(threads),
    ]
    if adapter is not None:
        common.extend(["--lora", str(adapter)])
    command = [*common, "-ngl", "auto"]
    try:
        if selected is Accelerator.CPU:
            command = [
                *common,
                "-dev",
                "none",
                "-ngl",
                "0",
                "-fit",
                "off",
                "--no-op-offload",
            ]
        run_logged(command, log_path=log_path, env=offline_environment())
    except TrainingError:
        if selected is Accelerator.CPU:
            raise
        run_logged(
            [
                *common,
                "-dev",
                "none",
                "-ngl",
                "0",
                "-fit",
                "off",
                "--no-op-offload",
            ],
            log_path=log_path,
            env=offline_environment(),
        )


def _verify_merged_metadata(base: ModelInspection, merged: ModelInspection) -> None:
    if base.format is not merged.format:
        raise VerificationError("merged model format differs from its base")
    if base.architecture != merged.architecture:
        raise VerificationError("merged model architecture differs from its base")
    if base.quantization != merged.quantization:
        raise VerificationError(
            f"merged quantization changed: base={base.quantization}, merged={merged.quantization}"
        )
    for name in ("block_count", "embedding_length", "context_length"):
        left = getattr(base, name)
        right = getattr(merged, name)
        if left is not None and right is not None and left != right:
            raise VerificationError(f"merged model {name} differs: base={left}, merged={right}")


def _verify_gguf_tensor_types(base: dict[str, int], merged: dict[str, int]) -> None:
    if base.keys() != merged.keys():
        missing = sorted(base.keys() - merged.keys())
        unexpected = sorted(merged.keys() - base.keys())
        raise VerificationError(
            "merged GGUF tensor set changed: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    changed = sorted(name for name in base if base[name] != merged[name])
    if changed:
        details = ", ".join(
            f"{name} ({base[name]} -> {merged[name]})" for name in changed[:5]
        )
        raise VerificationError(f"merged GGUF tensor types changed: {details}")


def _check_gguf_disk_budget(base: ModelInspection, output: Path) -> None:
    available = shutil.disk_usage(output).free
    required = base.size_bytes + 512 * 1024**2
    if available < required:
        raise ConfigurationError(
            "GGUF fusion needs space for an exact quantized base copy; "
            f"need about {required / 2**30:.1f} GiB, available {available / 2**30:.1f} GiB"
        )


def _require_new_destination(path: Path) -> None:
    if path.exists():
        raise ConfigurationError(f"refusing to overwrite an existing merged model: {path}")
