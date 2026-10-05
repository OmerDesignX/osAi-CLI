"""Exact gradient LoRA training against a frozen, packed GGUF base."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from contextlib import nullcontext, suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..checkpoints import CheckpointPublisher
from ..config import DEFAULT_TARGETS, ModelFormat
from ..dataset import require_text_training, validate_dataset
from ..errors import (
    ConfigurationError,
    DependencyError,
    TrainingError,
    TrainingStopped,
    VerificationError,
)
from ..formats import inspect_model
from ..fusion import resolve_gguf_fusion_bundle
from ..hardware import Accelerator, select_llama_accelerator
from ..io import OutputLock, atomic_json, fingerprint, sha256_file
from ..multi_gpu import available_llama_devices, llama_device_arguments
from ..offline import offline_environment
from ..paths import llama_binary
from ..process import run_logged
from ..resume_state import extract_adapter
from ..session import SessionLayout, record_dataset
from ..system import doctor, physical_memory_bytes
from .llama_cpp import ensure_runtime_accelerator
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
_CHECKPOINT_RE = re.compile(r"checkpoint epoch=(\d+)\s+best_train_loss=([0-9.eE+-]+)")
_SUPERVISED_EVAL_RE = re.compile(r"eval_loss=([0-9.eE+-]+)")
_SUPERVISED_LABELS_RE = re.compile(r"assistant-only loss enabled for (\d+) labels")
_ELAPSED_RE = re.compile(r"\bt=(\d+):(\d+):(\d+)")
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
    device_speeds: tuple[float, ...] = ()
    split_mode: str = "layer"
    tensor_split: tuple[float, ...] = ()
    main_gpu: int = 0
    auto_settings: bool = False
    calibration_pilot: bool = False

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
        if self.device_speeds and len(self.device_speeds) != len(self.devices):
            raise ConfigurationError("device speeds must match the selected GPUs")
        if any(not math.isfinite(value) or value <= 0 for value in self.device_speeds):
            raise ConfigurationError("device speeds must be finite and positive")
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
    initial_test_loss: float | None = None
    device_speeds: tuple[float, ...] = ()


def train_gradient_gguf(
    model: str | Path,
    data: str | Path,
    output: str | Path,
    *,
    options: LlamaGradientOptions | None = None,
    accelerator: str | Accelerator = Accelerator.AUTO,
    resume: bool = False,
) -> LlamaGradientResult:
    """Train only F32 LoRA tensors with exact backprop through a packed GGUF base."""

    settings = options or LlamaGradientOptions()
    settings.validate()
    _validate_memory_budget(settings, physical_memory_bytes())
    base = inspect_model(model, ModelFormat.GGUF)
    _validate_hybrid_training_path(base, settings)
    ensure_runtime_accelerator(accelerator)
    dataset = validate_dataset(data)
    require_text_training(dataset)
    destination = Path(output).expanduser().resolve()
    _validate_output(destination, base.path, dataset.path)
    layout = SessionLayout.at(destination)
    layout.create()
    resume_snapshot = layout.root / "outputs" / "checkpoint" / "adapter" / "last.gguf.resume"
    resume_input = layout.work / "resume-input.json"
    previous_resume: dict[str, Any] | None = None
    if resume:
        if not resume_snapshot.is_file() or not resume_input.is_file():
            raise ConfigurationError(
                "this session has no exact resume checkpoint; an older adapter alone cannot resume"
            )
        try:
            previous_resume = json.loads(resume_input.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VerificationError("the saved resume settings are unreadable") from exc
    record_dataset(layout, dataset)
    manifest_path = layout.run_manifest

    requested_accelerator = select_llama_accelerator(accelerator)
    if requested_accelerator is not Accelerator.CPU:
        discovered = available_llama_devices(
            llama_binary("llama-completion"),
            requested_accelerator,
        )
        if settings.devices and any(device not in discovered for device in settings.devices):
            raise ConfigurationError(
                "a selected GPU is unavailable or is an integrated adapter while "
                "dedicated GPUs are present"
            )
        if not settings.devices and discovered:
            settings = replace(settings, devices=discovered)
            print(
                f"osai: selected {requested_accelerator.value} device(s): "
                f"{', '.join(settings.devices)}"
            )
        if not settings.devices:
            raise ConfigurationError("the selected GPU backend reports no usable devices")
    if settings.multi_gpu == "on" and len(settings.devices) < 2:
        raise ConfigurationError(
            "multi-GPU was required, but fewer than two usable devices were found"
        )
    if settings.multi_gpu != "off" and len(settings.devices) > 1 and settings.split_mode != "layer":
        raise ConfigurationError("model-sharded GGUF training currently requires layer splitting")
    requested_options = json.loads(json.dumps(asdict(settings)))
    if resume:
        assert previous_resume is not None
        if previous_resume.get("requested_options") != requested_options:
            raise VerificationError(
                "the requested training settings differ from the stopped session"
            )
        saved_options = previous_resume.get("options")
        if not isinstance(saved_options, dict):
            raise VerificationError("the saved resume settings are incomplete")
        # An automatic memory retry can lower these two values after launch.
        # Restore the measured effective values without accepting any other change.
        try:
            settings = replace(
                settings,
                context=int(saved_options["context"]),
                batch_size=int(saved_options["batch_size"]),
            )
            settings.validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError("the saved effective context is invalid") from exc
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
            source = Path(model).expanduser().resolve()
            fused = (
                resolve_gguf_fusion_bundle(source)
                if source.is_dir() and (source / "osai_fusion.json").is_file()
                else None
            )
            if fused is not None:
                reader = _import_gguf().GGUFReader(fused.adapter)
                resumed = {
                    str(tensor.name): np.asarray(tensor.data, dtype=np.float32).copy()
                    for tensor in reader.tensors
                }
                if set(resumed) != set(parameters) or any(
                    resumed[name].shape != parameters[name].shape for name in parameters
                ):
                    raise ConfigurationError(
                        "the fused custom model's LoRA shape differs from the selected "
                        "rank, layers, or targets; use Auto settings to resume it"
                    )
                parameters = resumed
            _verify_hybrid_adapter_tensors(base, parameters)
            manifest["options"]["selected_base_tensors"] = sorted(
                {name.removesuffix(".lora_a").removesuffix(".lora_b") for name in parameters}
            )
            atomic_json(manifest_path, manifest)
            internal = layout.work
            initial_adapter = internal / "adapter-initial.gguf"
            trained_adapter = internal / "adapter-trained.gguf"
            trained_adapter.unlink(missing_ok=True)
            final_adapter = layout.adapters / "gguf" / "adapter.gguf"
            final_adapter.parent.mkdir(parents=True, exist_ok=True)
            if resume:
                extract_adapter(resume_snapshot, initial_adapter)
            elif fused is not None:
                shutil.copy2(fused.adapter, initial_adapter)
            else:
                _write_adapter(
                    initial_adapter,
                    base.architecture,
                    parameters,
                    settings.scale * settings.rank,
                    np,
                )
            initial_digest = _adapter_tensor_digest(initial_adapter, np)
            if resume and previous_resume is not None:
                saved_digest = previous_resume.get("initial_digest")
                if not isinstance(saved_digest, str) or not saved_digest:
                    raise VerificationError("the saved initial adapter digest is missing")
                initial_digest = saved_digest

            # Hybrid llama.cpp models round small contexts up to 256 tokens. A
            # slightly larger corpus prevents the upstream dataset constructor
            # from receiving zero examples without inflating per-step memory.
            corpus_context = max(settings.context, 128)
            structured_separator = "\n<|osai_record_end|>\n"
            train_corpus = internal / "train.txt"
            if resume:
                if not train_corpus.is_file():
                    raise VerificationError("the original prepared training corpus is missing")
            else:
                train_corpus = _write_corpus(
                    dataset.path / "train.jsonl",
                    train_corpus,
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
                    record_separator=(structured_separator if settings.mask_prompt else "\n\n"),
                )
                if test_source.is_file() and not resume
                else None
            )

            resume_signature = {
                "schema_version": 1,
                "model": [asdict(item) for item in before],
                "dataset": str(dataset.path),
                "corpus_sha256": sha256_file(train_corpus),
                "options": asdict(settings),
                "requested_options": requested_options,
                "accelerator": training_accelerator.value,
                "initial_digest": initial_digest,
            }
            if resume:
                assert previous_resume is not None
                if previous_resume != json.loads(json.dumps(resume_signature)):
                    raise VerificationError(
                        "model, data, or training settings differ from the exact resume checkpoint"
                    )
            else:
                atomic_json(resume_input, resume_signature)

            sharded_devices = (
                settings.devices
                if training_accelerator is not Accelerator.CPU
                and settings.multi_gpu != "off"
                and len(settings.devices) > 1
                else ()
            )
            evaluation_accelerator = Accelerator.CPU if sharded_devices else requested_accelerator
            initial_loss = math.nan
            initial_test_loss = None
            perplexity_binary = None
            if not settings.mask_prompt and not settings.calibration_pilot:
                perplexity_binary = llama_binary("llama-perplexity")
                if perplexity_binary is None:
                    raise DependencyError("llama-perplexity is not built; run `osai build-llama`")
                initial_loss, evaluation_accelerator = _evaluate_loss(
                    perplexity_binary,
                    base.path,
                    initial_adapter,
                    train_corpus,
                    settings.context,
                    evaluation_accelerator,
                    layout.logs / "evaluate-before.log",
                    settings,
                )
            if settings.calibration_pilot and test_corpus is not None:
                evaluation_settings = (
                    replace(
                        settings,
                        multi_gpu="off",
                        devices=(settings.devices[0],),
                        tensor_split=(),
                        main_gpu=0,
                    )
                    if settings.devices and training_accelerator is not Accelerator.CPU
                    else settings
                )
                initial_test_loss, evaluation_accelerator = _run_supervised_evaluation(
                    binary,
                    base.path,
                    initial_adapter,
                    test_corpus,
                    evaluation_settings,
                    layout.logs / "evaluate-pilot-before.log",
                    training_accelerator,
                )
                if evaluation_accelerator is not training_accelerator:
                    raise TrainingError("GPU pilot evaluation fell back to CPU")

            log_path = layout.logs / "train.log"
            if not resume:
                log_path.unlink(missing_ok=True)
            successful_offset = 0
            training_env = offline_environment()
            if training_accelerator is Accelerator.CUDA:
                training_env["OSAI_GPU_TIED_EMBEDDINGS"] = "1"
            training_env["OSAI_MASK_PROMPT"] = "1" if settings.mask_prompt else "0"
            training_env["OSAI_MAX_SEQ_LENGTH"] = str(settings.context)
            checkpoint_dir = layout.root / "outputs" / "checkpoint"
            (checkpoint_dir / "adapter").mkdir(parents=True, exist_ok=True)
            training_env["OSAI_CHECKPOINT_REQUEST"] = os.environ.get(
                "OSAI_CHECKPOINT_REQUEST", str(layout.root / "checkpoint.request")
            )
            training_env["OSAI_CHECKPOINT_OUTPUT"] = str(checkpoint_dir / "adapter" / "last.gguf")
            training_env["OSAI_CHECKPOINT_ACK"] = str(checkpoint_dir / "last.ack")
            training_env["OSAI_CHECKPOINT_INTERVAL_SECONDS"] = "300"
            training_env["OSAI_STOP_REQUEST"] = os.environ.get(
                "OSAI_STOP_REQUEST", str(layout.root / "stop.request")
            )
            if resume:
                training_env["OSAI_RESUME_STATE"] = str(resume_snapshot)
                Path(training_env["OSAI_STOP_REQUEST"]).unlink(missing_ok=True)
            if sharded_devices:
                manifest["model_sharded_training"] = {
                    "method": "one optimizer step over layer shards",
                    "devices": list(sharded_devices),
                    "split_mode": settings.split_mode,
                    "adapter_rank": settings.rank,
                }
                atomic_json(manifest_path, manifest)
                print(
                    "osai: one shared GGUF trainer on " + ", ".join(sharded_devices),
                    flush=True,
                )
            while True:
                attempt_offset = log_path.stat().st_size if log_path.exists() else 0
                command = _gradient_command(
                    binary,
                    base.path,
                    initial_adapter,
                    train_corpus,
                    trained_adapter,
                    settings,
                    training_accelerator,
                )
                try:
                    publisher = (
                        nullcontext()
                        if settings.calibration_pilot
                        else CheckpointPublisher(
                            kind="gguf",
                            model=base.path,
                            shards=base.shards,
                            latest=checkpoint_dir / "adapter" / "last.gguf",
                        )
                    )
                    with publisher:
                        run_logged(command, log_path=log_path, env=training_env)
                    successful_offset = attempt_offset
                    break
                except TrainingError as exc:
                    if (
                        "status 75" in str(exc)
                        and Path(training_env["OSAI_STOP_REQUEST"]).is_file()
                        and resume_snapshot.is_file()
                    ):
                        raise TrainingStopped(str(resume_snapshot)) from exc
                    if (
                        settings.auto_settings
                        and not resume
                        and training_accelerator is not Accelerator.CPU
                        and settings.context > 256
                        and _memory_failure(log_path, attempt_offset)
                    ):
                        settings = _lower_auto_context(settings, training_env, manifest)
                        resume_signature["options"] = asdict(settings)
                        atomic_json(resume_input, resume_signature)
                        atomic_json(manifest_path, manifest)
                        trained_adapter.unlink(missing_ok=True)
                        successful_offset = log_path.stat().st_size
                        continue
                    if settings.multi_gpu == "on":
                        raise TrainingError(
                            "required multi-GPU training failed; see the native training log "
                            f"at {log_path}"
                        ) from exc
                    raise

            if not trained_adapter.is_file():
                raise TrainingError(f"llama.cpp did not save a trained adapter; see {log_path}")
            epoch_losses, optimizer_steps, _labels, _speed, checkpoint = (
                _gradient_worker_log_summary(
                    log_path,
                    settings.epochs,
                    settings.mask_prompt,
                    0,
                    start_offset=successful_offset,
                )
            )
            best_epoch, best_supervised_loss = checkpoint or _parse_best_checkpoint(
                "", epoch_losses
            )
            if settings.mask_prompt or settings.calibration_pilot:
                initial_loss = epoch_losses[0]

            final_digest = _adapter_tensor_digest(trained_adapter, np)
            if final_digest == initial_digest:
                raise VerificationError(
                    "gradient optimizer completed without changing LoRA tensors"
                )
            os.replace(trained_adapter, final_adapter)
            _verify_adapter(final_adapter, base.architecture)

            if settings.calibration_pilot:
                final_loss = epoch_losses[-1]
                if test_corpus is not None:
                    test_loss, evaluation_accelerator = _run_supervised_evaluation(
                        binary,
                        base.path,
                        final_adapter,
                        test_corpus,
                        evaluation_settings,
                        layout.logs / "evaluate-pilot-after.log",
                        training_accelerator,
                    )
                    if evaluation_accelerator is not training_accelerator:
                        raise TrainingError("GPU pilot evaluation fell back to CPU")
                else:
                    test_loss = None
                evaluation_accelerator = training_accelerator
            elif settings.mask_prompt:
                evaluation_settings = (
                    replace(
                        settings,
                        multi_gpu="off",
                        devices=(settings.devices[0],),
                        tensor_split=(),
                        main_gpu=0,
                    )
                    if settings.devices and training_accelerator is not Accelerator.CPU
                    else settings
                )
                final_loss, evaluation_accelerator = _run_supervised_evaluation(
                    binary,
                    base.path,
                    final_adapter,
                    train_corpus,
                    evaluation_settings,
                    layout.logs / "evaluate-after.log",
                    training_accelerator,
                )
                if test_corpus:
                    test_loss, evaluation_accelerator = _run_supervised_evaluation(
                        binary,
                        base.path,
                        final_adapter,
                        test_corpus,
                        evaluation_settings,
                        layout.logs / "evaluate-test.log",
                        evaluation_accelerator,
                    )
                else:
                    test_loss = None
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

            if settings.mask_prompt or settings.calibration_pilot:
                loss_reducing_steps = sum(
                    current < previous
                    for previous, current in zip(epoch_losses, epoch_losses[1:], strict=False)
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
                        "initial_test_loss": initial_test_loss,
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
            # The original JSONL remains at its recorded path. Large runs can
            # otherwise leave several gigabytes of duplicate text per session.
            for prepared in (train_corpus, test_corpus):
                if prepared is not None:
                    with suppress(OSError):
                        prepared.unlink()
            return LlamaGradientResult(
                destination,
                final_adapter,
                manifest_path,
                epoch_losses,
                test_loss,
                training_accelerator.value,
                optimizer_steps,
                loss_reducing_steps,
                initial_test_loss,
                (),
            )
        except BaseException as exc:
            manifest.update(
                {
                    "status": "stopped" if isinstance(exc, TrainingStopped) else "failed",
                    "completed_at": _now(),
                    "resume_checkpoint": (
                        str(resume_snapshot) if isinstance(exc, TrainingStopped) else None
                    ),
                    "error": None
                    if isinstance(exc, TrainingStopped)
                    else {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
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
        "-m",
        str(model),
        "--lora",
        str(adapter),
        "-f",
        str(corpus),
        "-o",
        str(output),
        "-c",
        str(settings.context),
        "-b",
        str(settings.batch_size),
        "-ub",
        str(settings.batch_size),
        "-epochs",
        str(settings.epochs),
        "-val-split",
        "0",
        "-lr",
        format(settings.learning_rate, ".17g"),
        "-opt",
        settings.optimizer,
        "-t",
        str(settings.threads),
        "-tb",
        str(settings.threads),
        "--no-repack",
        "--log-colors",
        "off",
    ]
    command.extend(llama_device_arguments(accelerator, settings))
    return command


def _gradient_worker_log_summary(
    log: Path,
    epochs: int,
    mask_prompt: bool,
    elapsed_seconds: float,
    *,
    start_offset: int = 0,
) -> tuple[tuple[float, ...], int, int | None, float, tuple[int, float] | None]:
    """Read native metrics without loading a potentially multi-gigabyte log."""

    losses: list[float] = []
    supervised_steps = 0
    progress_total = 1
    trained_labels: int | None = None
    native_seconds = 0
    checkpoint: tuple[int, float] | None = None
    with log.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(start_offset)
        for line in handle:
            losses.extend(float(value) for _epoch, value in _EPOCH_LOSS_RE.findall(line))
            supervised_steps += len(_SUPERVISED_STEP_RE.findall(line))
            for value in _PROGRESS_RE.findall(line):
                progress_total = max(progress_total, int(value))
            for value in _SUPERVISED_LABELS_RE.findall(line):
                trained_labels = int(value)
            for hours, minutes, seconds in _ELAPSED_RE.findall(line):
                native_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            for epoch, loss in _CHECKPOINT_RE.findall(line):
                checkpoint = int(epoch), float(loss)
    if not losses:
        raise TrainingError(f"llama.cpp did not report an epoch loss; see {log}")
    if len(losses) != epochs:
        raise TrainingError(f"parallel worker reported the wrong epoch count; see {log}")
    if not all(math.isfinite(value) for value in losses):
        raise TrainingError(f"llama.cpp reported a non-finite training loss; see {log}")
    steps = supervised_steps if mask_prompt else progress_total * epochs
    # The process duration also includes model loading and dataset tokenization.
    # Use the native optimizer timer for worker throughput whenever it exists.
    duration = native_seconds or elapsed_seconds or 1
    return tuple(losses), steps, trained_labels, progress_total * epochs / duration, checkpoint


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

    loss, _ = _run_supervised_evaluation(
        binary,
        model,
        adapter,
        corpus,
        settings,
        log_path,
        accelerator,
        example_weights=example_weights,
    )
    return loss


def _run_supervised_evaluation(
    binary: Path,
    model: Path,
    adapter: Path,
    corpus: Path,
    settings: LlamaGradientOptions,
    log_path: Path,
    accelerator: Accelerator,
    *,
    example_weights: tuple[float, ...] | None = None,
) -> tuple[float, Accelerator]:
    """Run supervised evaluation and report the accelerator actually used."""

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
    environment["OSAI_MAX_SEQ_LENGTH"] = str(settings.context)
    environment["OSAI_EVAL_ONLY"] = "1"
    if example_weights is not None:
        environment["OSAI_EXAMPLE_WEIGHTS"] = ",".join(
            format(value, ".17g") for value in example_weights
        )
    try:
        run_logged(command, log_path=log_path, env=environment)
    except TrainingError:
        if accelerator is Accelerator.CPU or not _retryable_accelerator_failure(log_path):
            raise
        print(f"osai: {accelerator.value} evaluation failed; retrying with CPU")
        accelerator = Accelerator.CPU
        command = _gradient_command(
            binary,
            model,
            adapter,
            corpus,
            log_path.with_suffix(".unused.gguf"),
            settings,
            Accelerator.CPU,
        )
        run_logged(command, log_path=log_path, env=environment)
    output = log_path.read_text(encoding="utf-8", errors="replace")
    return _parse_supervised_loss(output, log_path), accelerator


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


def _parse_best_checkpoint(output: str, epoch_losses: tuple[float, ...]) -> tuple[int, float]:
    matches = _CHECKPOINT_RE.findall(output)
    if matches:
        epoch, loss = matches[-1]
        return int(epoch), float(loss)
    best_index = min(range(len(epoch_losses)), key=epoch_losses.__getitem__)
    return best_index + 1, epoch_losses[best_index]


def _validate_memory_budget(settings: LlamaGradientOptions, physical_memory: int | None) -> None:
    if physical_memory is None or physical_memory > 10 * 1024**3:
        return
    unsafe: list[str] = []
    if len(settings.target_modules) > 1:
        unsafe.append("more than one target module")
    if settings.num_layers > 1:
        unsafe.append("more than one model layer")
    if settings.context > 256:
        unsafe.append("context above 256")
    if settings.batch_size > 8:
        unsafe.append("microbatch above 8")
    if unsafe:
        raise ConfigurationError(
            "the low-memory GGUF backprop safety limit rejects "
            + ", ".join(unsafe)
            + "; use one target, one layer, context <= 256, and microbatch <= 8"
        )


def _validate_hybrid_training_path(base, settings: LlamaGradientOptions) -> None:
    if base.architecture.lower() not in {"qwen35", "qwen35moe"}:
        return
    safe_targets = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    if (
        settings.num_layers != 1
        or not settings.target_modules
        or not set(settings.target_modules) <= safe_targets
    ):
        raise ConfigurationError(
            "Qwen3.5 GGUF training can adapt only the final block's MLP projections "
            "because llama.cpp cannot backpropagate through GATED_DELTA_NET. "
            "Use --num-layers 1 and an mlp.* target, or enable --auto-settings."
        )


def _verify_hybrid_adapter_tensors(base, parameters: dict) -> None:
    if base.architecture.lower() not in {"qwen35", "qwen35moe"}:
        return
    final_block = (base.block_count or 0) - 1
    if final_block < 0 or any(
        not name.startswith(f"blk.{final_block}.ffn_") for name in parameters
    ):
        raise ConfigurationError(
            "Qwen3.5 GGUF training requires compatible MLP projections in the final model block"
        )


def _lower_auto_context(
    settings: LlamaGradientOptions,
    environment: dict[str, str],
    manifest: dict[str, Any],
) -> LlamaGradientOptions:
    context = max(256, settings.context // 2)
    batch_size = min(settings.batch_size, context)
    while context % batch_size:
        batch_size -= 1
    lowered = replace(settings, context=context, batch_size=batch_size)
    environment["OSAI_MAX_SEQ_LENGTH"] = str(context)
    manifest["options"]["context"] = context
    manifest["options"]["batch_size"] = batch_size
    manifest.setdefault("auto_context_retries", []).append(
        {
            "from": settings.context,
            "to": context,
            "reason": "native device memory allocation failed",
        }
    )
    print(
        f"osai: auto retry engine=llama.cpp "
        f"attempt={len(manifest['auto_context_retries']) + 1} "
        f"context={settings.context}->{context} "
        f"batch={settings.batch_size}->{batch_size} reason=oom; "
        "overlapping windows preserve assistant labels"
    )
    return lowered


def _memory_failure(log_path: Path, offset: int = 0) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open("rb") as handle:
        handle.seek(max(offset, log_path.stat().st_size - 16_384))
        tail = handle.read().decode("utf-8", errors="replace").lower()
    return any(
        marker in tail
        for marker in (
            "out of device memory",
            "cuda out of memory",
            "failed to allocate gpu",
            "failed to allocate buffer",
            "failed to allocate cuda",
            "out of memory",
        )
    )


def _retryable_accelerator_failure(log_path: Path, offset: int = 0) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open("rb") as handle:
        handle.seek(max(offset, log_path.stat().st_size - 16_384))
        tail = handle.read().decode("utf-8", errors="replace").lower()
    return (
        any(
            marker in tail
            for marker in (
                "out of device memory",
                "cuda out of memory",
                "cuda error",
                "vulkan error",
                "failed to allocate gpu",
                "failed to allocate buffer",
                "device lost",
                "no usable gpu found",
                # A native GPU backend can terminate before it emits a driver
                # diagnostic. Retry once on CPU for these access-violation codes.
                "exit=3221225477",
                "exit=139",
                "exit=-11",
            )
        )
        and "unsupported ggml op for backward pass" not in tail
    )
