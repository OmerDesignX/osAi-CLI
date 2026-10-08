"""Write real, standalone fused weights from a base and its LoRA adapter."""

from __future__ import annotations

import os
import shutil
import string
import sys
import tempfile
import uuid
from pathlib import Path

from .config import ModelFormat
from .errors import ConfigurationError, DependencyError, VerificationError
from .formats import gguf_tensor_types, inspect_model
from .paths import llama_binary
from .process import run_logged


def export_gguf_weights(
    model: Path,
    shards: tuple[Path, ...],
    adapter: Path,
    destination: Path,
    *,
    work: Path,
    log: Path,
    threads: int = 2,
) -> Path:
    """Fuse LoRA into the original GGUF tensor types, replacing one output atomically."""
    exporter = llama_binary("llama-export-lora")
    if exporter is None:
        raise DependencyError("llama-export-lora is required to create a standalone merged GGUF")
    splitter = llama_binary("llama-gguf-split") if len(shards) > 1 else None
    if len(shards) > 1 and splitter is None:
        raise DependencyError("llama-gguf-split is required for a sharded GGUF base")
    for source in (*shards, adapter):
        if not source.is_file() or not source.stat().st_size:
            raise ConfigurationError(f"missing model or adapter file: {source}")
    work.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root: Path | None = None
    if splitter is not None:
        required = sum(shard.stat().st_size for shard in shards)
        candidate_roots = [work, Path(tempfile.gettempdir())]
        if os.name == "nt":
            candidate_roots.extend(Path(f"{letter}:/") for letter in string.ascii_uppercase)
        choices = [
            (shutil.disk_usage(root).free, root) for root in candidate_roots if root.exists()
        ]
        # The destination may be on the same volume. Reserve space for both
        # the temporary unified base and the final merged model there.
        suitable = [
            (free, root)
            for free, root in choices
            if free > required * (2 if root.drive == destination.drive else 1) + 256 * 1024**2
        ]
        if not suitable:
            raise ConfigurationError(
                "not enough free space for a temporary unified GGUF and merged weights"
            )
        temporary_root = Path(tempfile.mkdtemp(prefix="osai-unified-", dir=max(suitable)[1]))
    unified = (temporary_root or work) / f"unified-{uuid.uuid4().hex}.gguf"
    pending = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
    try:
        if splitter is not None:
            run_logged([str(splitter), "--merge", str(model), str(unified)], log_path=log)
            merge_base = unified
        else:
            merge_base = model
        run_logged(
            [
                str(exporter),
                "-m",
                str(merge_base),
                "--lora",
                str(adapter),
                "-o",
                str(pending),
                "--keep-quantized",
                "-t",
                str(max(1, threads)),
            ],
            log_path=log,
        )
        if not pending.is_file() or pending.stat().st_size == 0:
            raise VerificationError("GGUF exporter did not write merged weights")
        base_info = inspect_model(model, ModelFormat.GGUF)
        merged_info = inspect_model(pending, ModelFormat.GGUF)
        if (
            merged_info.architecture != base_info.architecture
            or merged_info.block_count != base_info.block_count
            or merged_info.context_length != base_info.context_length
            or gguf_tensor_types(merge_base) != gguf_tensor_types(pending)
        ):
            raise VerificationError("merged GGUF model metadata differs from its base")
        os.replace(pending, destination)
        return destination
    finally:
        pending.unlink(missing_ok=True)
        unified.unlink(missing_ok=True)
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)


def export_mlx_weights(
    model: Path,
    adapter: Path,
    destination: Path,
    *,
    work: Path,
    log: Path,
    python: str | Path | None = None,
    multimodal: bool = False,
) -> Path:
    """Fuse MLX LoRA layers and publish a standalone MLX model directory."""
    if not model.is_dir() or not adapter.is_dir():
        raise ConfigurationError("MLX export requires model and adapter directories")
    if not (adapter / "adapters.safetensors").is_file():
        raise ConfigurationError("MLX adapter weights are missing")
    stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
    stage.parent.mkdir(parents=True, exist_ok=True)
    try:
        if multimodal:
            command = [
                str(python or sys.executable),
                "-m",
                "osai.mlx_vlm_fuse",
                "--model",
                str(model),
                "--adapter",
                str(adapter),
                "--output",
                str(stage),
            ]
        else:
            command = [
                str(python or sys.executable),
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                str(model),
                "--adapter-path",
                str(adapter),
                "--save-path",
                str(stage),
            ]
        run_logged(command, log_path=log)
        if not any(stage.glob("*.safetensors")):
            raise VerificationError("MLX fuse did not write standalone weights")
        # Remove an older fused directory only after the new one is verified.
        previous = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.previous")
        if destination.exists():
            os.replace(destination, previous)
        try:
            os.replace(stage, destination)
        except BaseException:
            if previous.exists():
                os.replace(previous, destination)
            raise
        shutil.rmtree(previous, ignore_errors=True)
        return destination
    finally:
        shutil.rmtree(stage, ignore_errors=True)
