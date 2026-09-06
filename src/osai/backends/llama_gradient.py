"""Exact gradient LoRA training against a frozen, packed GGUF base."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import DEFAULT_TARGETS, ModelFormat
from ..dataset import validate_dataset
from ..errors import ConfigurationError, DependencyError, TrainingError, VerificationError
from ..formats import inspect_model
from ..hardware import Accelerator, select_llama_accelerator
from ..io import OutputLock, atomic_json, fingerprint
from ..multi_gpu import llama_device_arguments
from ..offline import offline_environment
from ..paths import llama_binary
from ..process import run_logged
from ..session import SessionLayout, record_dataset
from ..system import doctor, physical_memory_bytes
from .llama_utils import (
    _evaluate_loss,
    _import_gguf,
    _import_numpy,
    _initialize_parameters,
    _now,
    _validate_output,
    _verify_adapter,
    _write_adapter,
    _write_corpus,
)

_EPOCH_LOSS_RE = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.eE+-]+)")
_PROGRESS_RE = re.compile(r"data=\d+/(\d+)")
_SUPERVISED_STEP_RE = re.compile(r"supervised optimizer step labels=\d+")
_CHECKPOINT_RE = re.compile(
    r"checkpoint epoch=(\d+)\s+best_train_loss=([0-9.eE+-]+)"
)
_SUPERVISED_EVAL_RE = re.compile(r"eval_loss=([0-9.eE+-]+)")
_SUPPORTED_TARGETS = frozenset(DEFAULT_TARGETS)


@dataclass(frozen=True, slots=True)
class LlamaGradientOptions:
    """Settings for reverse-mode LoRA optimization inside llama.cpp."""

    epochs: int = 1
    rank: int = 1
    scale: float = 4.0
    num_layers: int = 1
    context: int = 32
    batch_size: int = 8
    learning_rate: float = 1e-5
    optimizer: str = "sgd"
    threads: int = 2
    seed: int = 0
    target_modules: tuple[str, ...] = ("mlp.down_proj",)
    mask_prompt: bool = True
    strict_base_hash: bool = True
    multi_gpu: str = "auto"
    devices: tuple[str, ...] = ()
    split_mode: str = "layer"
    tensor_split: tuple[float, ...] = ()
    main_gpu: int = 0

    def validate(self) -> None:
        integers = {
            "epochs": self.epochs,
            "rank": self.rank,
            "num_layers": self.num_layers,
            "context": self.context,
            "batch_size": self.batch_size,
            "threads": self.threads,
        }
        invalid = [name for name, value in integers.items() if value < 1]
        if invalid:
            raise ConfigurationError(f"values must be at least 1: {', '.join(invalid)}")
        if self.context < 32:
            raise ConfigurationError("llama.cpp gradient context must be at least 32")
        if self.context % self.batch_size:
            raise ConfigurationError("gradient batch size must divide the context size")
        if self.scale <= 0 or self.learning_rate <= 0:
            raise ConfigurationError("scale and learning rate must be positive")
        if self.optimizer not in {"sgd", "adamw"}:
            raise ConfigurationError("llama.cpp gradient optimizer must be sgd or adamw")
        unknown = sorted(set(self.target_modules) - _SUPPORTED_TARGETS)
        if unknown:
            raise ConfigurationError(
                "unsupported llama.cpp LoRA target module(s): " + ", ".join(unknown)
            )
        llama_device_arguments(Accelerator.CPU, self)


@dataclass(frozen=True, slots=True)
class LlamaGradientResult:
    output: Path
    adapter: Path
    manifest: Path
    losses: tuple[float, ...]
    test_loss: float | None
    accelerator: str
    optimizer_steps: int
    loss_reducing_steps: int


def train_gradient_gguf(
    model: str | Path,
    data: str | Path,
    output: str | Path,
    *,
    options: LlamaGradientOptions | None = None,
    accelerator: str | Accelerator = Accelerator.AUTO,
) -> LlamaGradientResult:
    """Train only F32 LoRA tensors with exact backprop through a packed GGUF base."""

    settings = options or LlamaGradientOptions()
    settings.validate()
    _validate_memory_budget(settings, physical_memory_bytes())
    base = inspect_model(model, ModelFormat.GGUF)
    dataset = validate_dataset(data)
    destination = Path(output).expanduser().resolve()
    _validate_output(destination, base.path, dataset.path)
    layout = SessionLayout.at(destination)
    layout.create()
    record_dataset(layout, dataset)
    manifest_path = layout.run_manifest

    requested_accelerator = select_llama_accelerator(accelerator)
    training_accelerator = requested_accelerator
    fallback_from: str | None = None
    fallback_reason: str | None = None

    binary = llama_binary("llama-finetune")
    if binary is None:
        raise DependencyError("llama-finetune is not built; run `osai build-llama`")

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "preflight",
        "started_at": _now(),
        "backend": "llama.cpp-backprop",
        "method": "exact reverse-mode gradients over LoRA adapters",
        "local_only": True,
        "base_storage": "packed GGUF; frozen and never fully dequantized",
        "model": base.as_dict(),
        "dataset": asdict(dataset),
        "options": asdict(settings),
        "accelerator_requested": requested_accelerator.value,
        "accelerator": training_accelerator.value,
        "accelerator_fallback_from": fallback_from,
        "accelerator_fallback_reason": fallback_reason,
        "system": doctor().as_dict(),
    }
    manifest["dataset"]["path"] = str(dataset.path)
    manifest["options"]["target_modules"] = list(settings.target_modules)
    atomic_json(manifest_path, manifest)

    with OutputLock(destination):
        try:
            before = tuple(
                fingerprint(shard, full_hash=settings.strict_base_hash) for shard in base.shards
            )
            manifest["base_fingerprints_before"] = [asdict(item) for item in before]
            manifest["status"] = "training"
            atomic_json(manifest_path, manifest)

            np = _import_numpy()
            parameters = _initialize_parameters(base, settings, np)
            internal = layout.work
            initial_adapter = internal / "adapter-initial.gguf"
            trained_adapter = internal / "adapter-trained.gguf"
            trained_adapter.unlink(missing_ok=True)
            final_adapter = layout.adapters / "gguf" / "adapter.gguf"
            final_adapter.parent.mkdir(parents=True, exist_ok=True)
            _write_adapter(
                initial_adapter,
                base.architecture,
                parameters,
                settings.scale * settings.rank,
                np,
            )
            initial_digest = _adapter_tensor_digest(initial_adapter, np)

            # Hybrid llama.cpp models round small contexts up to 256 tokens. A
            # slightly larger corpus prevents the upstream dataset constructor
            # from receiving zero examples without inflating per-step memory.
            corpus_context = max(settings.context, 128)
            structured_separator = "\n<|osai_record_end|>\n"
            train_corpus = _write_corpus(
                dataset.path / "train.jsonl",
                internal / "train.txt",
                corpus_context,
                repeat_to_minimum=not settings.mask_prompt,
                record_separator=structured_separator if settings.mask_prompt else "\n\n",
            )
            test_source = dataset.path / "test.jsonl"
            test_corpus = (
                _write_corpus(
                    test_source,
                    internal / "test.txt",
                    corpus_context,
                    repeat_to_minimum=not settings.mask_prompt,
                    record_separator=(
                        structured_separator if settings.mask_prompt else "\n\n"
                    ),
                )
                if test_source.is_file()
                else None
            )

            evaluation_accelerator = requested_accelerator
            initial_loss = math.nan
            perplexity_binary = None
            if not settings.mask_prompt:
                perplexity_binary = llama_binary("llama-perplexity")
                if perplexity_binary is None:
                    raise DependencyError(
                        "llama-perplexity is not built; run `osai build-llama`"
                    )
                initial_loss, evaluation_accelerator = _evaluate_loss(
                    perplexity_binary,
                    base.path,
                    initial_adapter,
                    train_corpus,
                    settings.context,
                    requested_accelerator,
                    layout.logs / "evaluate-before.log",
                    settings,
                )

            log_path = layout.logs / "train.log"
            log_path.unlink(missing_ok=True)
            successful_offset = 0
            command = _gradient_command(
                binary, base.path, initial_adapter, train_corpus, trained_adapter,
                settings, training_accelerator,
            )
            training_env = offline_environment()
            training_env["OSAI_MASK_PROMPT"] = "1" if settings.mask_prompt else "0"
            try:
                run_logged(command, log_path=log_path, env=training_env)
            except TrainingError:
                if training_accelerator is Accelerator.CPU:
                    raise
                fallback_from = training_accelerator.value
                fallback_reason = "accelerator rejected the backward graph"
                training_accelerator = Accelerator.CPU
                successful_offset = log_path.stat().st_size if log_path.exists() else 0
                print(f"osai: {fallback_from} backprop failed; retrying with CPU")
                command = _gradient_command(
                    binary, base.path, initial_adapter, train_corpus, trained_adapter,
                    settings, training_accelerator,
                )
                run_logged(command, log_path=log_path, env=training_env)

            if not trained_adapter.is_file():
                raise TrainingError(f"llama.cpp did not save a trained adapter; see {log_path}")
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(successful_offset)
                successful_log = handle.read()
            epoch_losses = _parse_epoch_losses(successful_log, log_path)
            optimizer_steps = _parse_optimizer_steps(
                successful_log, settings.epochs, settings.mask_prompt
            )
            best_epoch, best_supervised_loss = _parse_best_checkpoint(
                successful_log, epoch_losses
            )
            if settings.mask_prompt:
                initial_loss = epoch_losses[0]

            final_digest = _adapter_tensor_digest(trained_adapter, np)
            if final_digest == initial_digest:
                raise VerificationError(
                    "gradient optimizer completed without changing LoRA tensors"
                )
            os.replace(trained_adapter, final_adapter)
            _verify_adapter(final_adapter, base.architecture)

            if settings.mask_prompt:
                final_loss = _evaluate_supervised_loss(
                    binary,
                    base.path,
                    final_adapter,
                    train_corpus,
                    settings,
                    layout.logs / "evaluate-after.log",
                )
                test_loss = (
                    _evaluate_supervised_loss(
                        binary,
                        base.path,
                        final_adapter,
                        test_corpus,
                        settings,
                        layout.logs / "evaluate-test.log",
                    )
                    if test_corpus
                    else None
                )
                evaluation_accelerator = Accelerator.CPU
            else:
                assert perplexity_binary is not None
                final_loss, evaluation_accelerator = _evaluate_loss(
                    perplexity_binary,
                    base.path,
                    final_adapter,
                    train_corpus,
                    settings.context,
                    evaluation_accelerator,
                    layout.logs / "evaluate-after.log",
                    settings,
                )
                test_loss = None
                if test_corpus:
                    test_loss, evaluation_accelerator = _evaluate_loss(
                        perplexity_binary,
                        base.path,
                        final_adapter,
                        test_corpus,
                        settings.context,
                        evaluation_accelerator,
                        layout.logs / "evaluate-test.log",
                        settings,
                    )

            after_inspection = inspect_model(base.path, ModelFormat.GGUF)
            after = tuple(
                fingerprint(shard, full_hash=settings.strict_base_hash)
                for shard in after_inspection.shards
            )
            if before != after:
                raise VerificationError("quantized GGUF base changed during gradient training")
            if base.quantization != after_inspection.quantization:
                raise VerificationError("GGUF quantization metadata changed during training")

            if settings.mask_prompt:
                loss_reducing_steps = sum(
                    current < previous
                    for previous, current in zip(
                        epoch_losses, epoch_losses[1:], strict=False
                    )
                )
            else:
                loss_reducing_steps = int(final_loss < initial_loss)
            manifest.update(
                {
                    "status": "completed",
                    "completed_at": _now(),
                    "accelerator": training_accelerator.value,
                    "accelerator_fallback_from": fallback_from,
                    "accelerator_fallback_reason": fallback_reason,
                    "evaluation_accelerator": evaluation_accelerator.value,
                    "training": {
                        "epoch_losses": list(epoch_losses),
                        "best_supervised_epoch": best_epoch,
                        "best_supervised_loss": best_supervised_loss,
                        "initial_train_loss": initial_loss,
                        "final_train_loss": final_loss,
                        "test_loss": test_loss,
                        "optimizer_steps": optimizer_steps,
                        "loss_reducing_steps": loss_reducing_steps,
                    },
                    "adapter": {
                        "path": str(final_adapter),
                        "size_bytes": final_adapter.stat().st_size,
                        "initial_tensor_digest": initial_digest,
                        "final_tensor_digest": final_digest,
                    },
                    "base_fingerprints_after": [asdict(item) for item in after],
                    "invariants": {
                        "base_files_unchanged": True,
                        "base_quantization_unchanged": True,
                        "full_precision_base_created": False,
                        "only_lora_tensors_trainable": True,
                        "adapter_tensors_changed": True,
                    },
                }
            )
            atomic_json(manifest_path, manifest)
            return LlamaGradientResult(
                destination,
                final_adapter,
                manifest_path,
                epoch_losses,
                test_loss,
                training_accelerator.value,
                optimizer_steps,
                loss_reducing_steps,
            )
        except BaseException as exc:
            manifest.update(
                {
                    "status": "failed",
                    "completed_at": _now(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            atomic_json(manifest_path, manifest)
            raise


def _gradient_command(
    binary: Path,
    model: Path,
    adapter: Path,
    corpus: Path,
    output: Path,
    settings: LlamaGradientOptions,
    accelerator: Accelerator,
) -> list[str]:
    command = [
        str(binary),
        "-m", str(model),
        "--lora", str(adapter),
        "-f", str(corpus),
        "-o", str(output),
        "-c", str(settings.context),
        "-b", str(settings.batch_size),
        "-ub", str(settings.batch_size),
        "-epochs", str(settings.epochs),
        "-val-split", "0",
        "-lr", format(settings.learning_rate, ".17g"),
        "-opt", settings.optimizer,
        "-t", str(settings.threads),
        "-tb", str(settings.threads),
        "--no-repack",
        "--log-colors", "off",
    ]
    command.extend(llama_device_arguments(accelerator, settings))
    return command


def _evaluate_supervised_loss(
    binary: Path,
    model: Path,
    adapter: Path,
    corpus: Path,
    settings: LlamaGradientOptions,
    log_path: Path,
    accelerator: Accelerator = Accelerator.CPU,
    *,
    example_weights: tuple[float, ...] | None = None,
) -> float:
    """Evaluate the same sparse assistant labels used by native backprop."""

    log_path.unlink(missing_ok=True)
    command = _gradient_command(
        binary,
        model,
        adapter,
        corpus,
        log_path.with_suffix(".unused.gguf"),
        settings,
        accelerator,
    )
    environment = offline_environment()
    environment["OSAI_MASK_PROMPT"] = "1"
    environment["OSAI_EVAL_ONLY"] = "1"
    if example_weights is not None:
        environment["OSAI_EXAMPLE_WEIGHTS"] = ",".join(
            format(value, ".17g") for value in example_weights
        )
    try:
        run_logged(command, log_path=log_path, env=environment)
    except TrainingError:
        if accelerator is Accelerator.CPU:
            raise
        command = _gradient_command(
            binary, model, adapter, corpus, log_path.with_suffix(".unused.gguf"),
            settings, Accelerator.CPU,
        )
        run_logged(command, log_path=log_path, env=environment)
    output = log_path.read_text(encoding="utf-8", errors="replace")
    return _parse_supervised_loss(output, log_path)


def _parse_supervised_loss(output: str, log_path: Path) -> float:
    matches = _SUPERVISED_EVAL_RE.findall(output)
    if not matches:
        raise TrainingError(f"llama.cpp did not report supervised evaluation loss; see {log_path}")
    loss = float(matches[-1])
    if not math.isfinite(loss):
        raise TrainingError(f"llama.cpp reported non-finite supervised loss; see {log_path}")
    return loss


def _adapter_tensor_digest(path: Path, np) -> str:
    reader = _import_gguf().GGUFReader(path)
    if not reader.tensors:
        raise VerificationError(f"adapter contains no tensors: {path}")
    digest = hashlib.sha256()
    for tensor in sorted(reader.tensors, key=lambda value: value.name):
        values = np.asarray(tensor.data, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise VerificationError(f"adapter contains non-finite values: {tensor.name}")
        digest.update(tensor.name.encode("utf-8"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _parse_epoch_losses(output: str, log_path: Path) -> tuple[float, ...]:
    matches = _EPOCH_LOSS_RE.findall(output)
    if not matches:
        raise TrainingError(f"llama.cpp did not report an epoch loss; see {log_path}")
    losses = tuple(float(value) for _, value in matches)
    if not all(math.isfinite(value) for value in losses):
        raise TrainingError(f"llama.cpp reported a non-finite training loss; see {log_path}")
    return losses


def _parse_optimizer_steps(output: str, epochs: int, mask_prompt: bool = False) -> int:
    if mask_prompt:
        return len(_SUPERVISED_STEP_RE.findall(output))
    totals = [int(value) for value in _PROGRESS_RE.findall(output)]
    return max(totals, default=1) * epochs


def _parse_best_checkpoint(
    output: str, epoch_losses: tuple[float, ...]
) -> tuple[int, float]:
    matches = _CHECKPOINT_RE.findall(output)
    if matches:
        epoch, loss = matches[-1]
        return int(epoch), float(loss)
    best_index = min(range(len(epoch_losses)), key=epoch_losses.__getitem__)
    return best_index + 1, epoch_losses[best_index]


def _validate_memory_budget(
    settings: LlamaGradientOptions, physical_memory: int | None
) -> None:
    if physical_memory is None or physical_memory > 10 * 1024**3:
        return
    unsafe: list[str] = []
    if len(settings.target_modules) > 1:
        unsafe.append("more than one target module")
    if settings.num_layers > 1:
        unsafe.append("more than one model layer")
    if settings.context > 64:
        unsafe.append("context above 64")
    if settings.batch_size > 8:
        unsafe.append("microbatch above 8")
    if unsafe:
        raise ConfigurationError(
            "the low-memory GGUF backprop safety limit rejects "
            + ", ".join(unsafe)
            + "; use one target, one layer, context <= 64, and microbatch <= 8"
        )
