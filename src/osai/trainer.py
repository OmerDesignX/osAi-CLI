"""End-to-end quantization-preserving training orchestration."""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends.mlx import MlxBackend
from .config import ModelFormat, TrainingConfig
from .dataset import validate_dataset
from .errors import ConfigurationError, TrainingError, VerificationError
from .formats import ModelInspection, assert_compatible, inspect_model
from .gguf_adapter import GgufAdapterResult, convert_mlx_adapter
from .hardware import Accelerator
from .io import FileFingerprint, OutputLock, atomic_json, fingerprint
from .merge import MergedModelResult, merge_gguf_model, merge_mlx_model
from .session import SessionLayout, publish_base_adapter_bundle, record_dataset
from .system import doctor, physical_memory_bytes


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output: Path
    mlx_adapter: Path
    gguf_adapter: Path | None
    manifest: Path
    deployment_manifest: Path
    merged_model: Path | None
    losses: tuple[float, ...]
    test_loss: float | None


def train(
    config: TrainingConfig,
    *,
    python: str | Path | None = None,
    mlx_accelerator: str | Accelerator = Accelerator.AUTO,
    llama_accelerator: str | Accelerator = Accelerator.AUTO,
) -> TrainingResult:
    config.validate()
    _validate_paths(config)
    layout = SessionLayout.at(config.output)
    layout.create()
    manifest_path = layout.run_manifest
    started = _now()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "preflight",
        "started_at": started,
        "config": config.as_dict(),
        "system": doctor().as_dict(),
    }
    atomic_json(manifest_path, manifest)

    with OutputLock(config.output):
        try:
            training_inspection = inspect_model(config.training_model, ModelFormat.MLX)
            target_inspection = inspect_model(config.model, config.effective_format)
            if config.effective_format is ModelFormat.GGUF:
                assert_compatible(training_inspection, target_inspection)
            dataset = validate_dataset(config.data)
            record_dataset(layout, dataset)
            _memory_guard(training_inspection)

            before = _fingerprint_unique_shards(
                training_inspection,
                target_inspection,
                full_hash=config.strict_base_hash,
            )
            manifest.update(
                {
                    "status": "training",
                    "training_model": training_inspection.as_dict(),
                    "target_model": target_inspection.as_dict(),
                    "dataset": asdict(dataset),
                    "base_fingerprints_before": [asdict(item) for item in before],
                }
            )
            manifest["dataset"]["path"] = str(dataset.path)
            atomic_json(manifest_path, manifest)

            mlx_result = MlxBackend(python=python, accelerator=mlx_accelerator).train(config)
            gguf_result: GgufAdapterResult | None = None
            if config.effective_format is ModelFormat.GGUF:
                gguf_result = convert_mlx_adapter(
                    mlx_result.adapter_dir,
                    layout.adapters / "gguf" / "adapter.gguf",
                    base_gguf=target_inspection,
                )

            post_training = inspect_model(config.training_model, ModelFormat.MLX)
            post_target = inspect_model(config.model, config.effective_format)
            _assert_quantization_preserved(training_inspection, post_training)
            _assert_quantization_preserved(target_inspection, post_target)
            after = _fingerprint_unique_shards(
                post_training,
                post_target,
                full_hash=config.strict_base_hash,
            )
            _assert_fingerprints(before, after)

            adapter_bytes = mlx_result.adapter_file.stat().st_size
            if gguf_result is not None:
                adapter_bytes += gguf_result.size_bytes
            if adapter_bytes >= training_inspection.size_bytes // 4:
                raise VerificationError(
                    "adapter files unexpectedly exceed 25% of the quantized base size"
                )

            adapters = {"mlx": mlx_result.adapter_dir}
            if gguf_result is not None:
                adapters["gguf"] = gguf_result.path
            manifest["status"] = "publishing"
            atomic_json(manifest_path, manifest)
            base_bundle = publish_base_adapter_bundle(
                layout,
                target_inspection,
                adapters,
                materialize_base=config.materialize_base,
            )
            merged: MergedModelResult | None = None
            if config.merge_model:
                try:
                    if config.effective_format is ModelFormat.MLX:
                        merged = merge_mlx_model(
                            target_inspection,
                            mlx_result.adapter_dir,
                            layout,
                            python=python,
                            accelerator=mlx_accelerator,
                        )
                    else:
                        assert gguf_result is not None
                        merged = merge_gguf_model(
                            target_inspection,
                            gguf_result.path,
                            layout,
                            accelerator=llama_accelerator,
                        )
                except TrainingError as exc:
                    raise VerificationError(
                        f"training completed but standalone model merge failed: {exc}"
                    ) from exc

            manifest.update(
                {
                    "status": "completed",
                    "completed_at": _now(),
                    "mlx_adapter": {
                        "path": str(mlx_result.adapter_file),
                        "size_bytes": mlx_result.adapter_file.stat().st_size,
                    },
                    "gguf_adapter": asdict(gguf_result) if gguf_result else None,
                    "base_plus_adapter": {
                        "path": str(layout.base_adapter),
                        "deployment_manifest": str(layout.deployment_manifest),
                        "base": base_bundle.as_dict(),
                    },
                    "merged_model": merged.as_dict() if merged else None,
                    "training": {
                        "elapsed_seconds": mlx_result.elapsed_seconds,
                        "reported_losses": list(mlx_result.losses),
                        "test_loss": mlx_result.test_loss,
                    },
                    "base_fingerprints_after": [asdict(item) for item in after],
                    "invariants": {
                        "base_files_unchanged": True,
                        "base_quantization_unchanged": True,
                        "full_precision_base_created": False,
                        "temporary_full_precision_merge": bool(
                            merged and merged.temporary_full_precision_intermediate
                        ),
                        "merge_strategy": merged.merge_strategy if merged else None,
                        "temporary_merge_files_removed": True,
                        "adapter_fraction_of_base": adapter_bytes / training_inspection.size_bytes,
                    },
                }
            )
            if manifest["gguf_adapter"] is not None:
                manifest["gguf_adapter"]["path"] = str(gguf_result.path)
            atomic_json(manifest_path, manifest)
            return TrainingResult(
                output=config.output,
                mlx_adapter=mlx_result.adapter_file,
                gguf_adapter=gguf_result.path if gguf_result else None,
                manifest=manifest_path,
                deployment_manifest=layout.deployment_manifest,
                merged_model=merged.path if merged else None,
                losses=mlx_result.losses,
                test_loss=mlx_result.test_loss,
            )
        except BaseException as exc:
            manifest.update(
                {
                    "status": "failed",
                    "completed_at": _now(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            if not isinstance(exc, KeyboardInterrupt):
                manifest["error"]["traceback"] = traceback.format_exc()
            atomic_json(manifest_path, manifest)
            raise


def _validate_paths(config: TrainingConfig) -> None:
    model = config.model.expanduser().resolve()
    output = config.output.expanduser().resolve()
    data = config.data.expanduser().resolve()
    for label, protected in (("model", model), ("dataset", data)):
        if output == protected or _is_relative_to(output, protected):
            raise ConfigurationError(f"output must not be inside the {label} path: {protected}")


def _memory_guard(model: ModelInspection) -> None:
    memory = physical_memory_bytes()
    if memory is None:
        return
    # Leaves room for OS, activations, Metal buffers, and the small optimizer state.
    if model.size_bytes > memory * 0.55:
        raise ConfigurationError(
            f"quantized model is {model.size_bytes / 2**30:.2f} GiB on a "
            f"{memory / 2**30:.2f} GiB host; use a smaller quantized tier"
        )


def _assert_quantization_preserved(before: ModelInspection, after: ModelInspection) -> None:
    if before.quantization != after.quantization:
        raise VerificationError(
            f"base quantization changed: before={before.quantization}, after={after.quantization}"
        )
    if before.size_bytes != after.size_bytes:
        raise VerificationError(
            f"base byte size changed: before={before.size_bytes}, after={after.size_bytes}"
        )


def _assert_fingerprints(
    before: tuple[FileFingerprint, ...], after: tuple[FileFingerprint, ...]
) -> None:
    if len(before) != len(after):
        raise VerificationError("base shard count changed during training")
    for left, right in zip(before, after, strict=True):
        if left != right:
            raise VerificationError(f"base model file changed during training: {left.path}")


def _fingerprint_unique_shards(
    *models: ModelInspection, full_hash: bool
) -> tuple[FileFingerprint, ...]:
    unique: dict[Path, None] = {}
    for model in models:
        for shard in model.shards:
            unique[shard.resolve()] = None
    return tuple(fingerprint(path, full_hash=full_hash) for path in unique)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
