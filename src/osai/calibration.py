"""Bounded, local training pilots for automatic learning-rate selection.

This calibration changes only a private temporary sample. The selected model
and dataset are never written to, and every candidate starts from the same
base weights and examples. A short pilot is evidence, not a guarantee that a
long heterogeneous run will decrease monotonically.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .auto_benchmark import BenchmarkResult
from .auto_settings import AutoTrainingSettings
from .backends.llama_gradient import LlamaGradientOptions, train_gradient_gguf
from .backends.mlx import MlxBackend
from .config import ModelFormat, TrainingConfig
from .dataset import normalize_sft_example, normalized_record_text
from .dataset_source import _source_rows, dataset_files
from .errors import ConfigurationError, OsAiError
from .formats import ModelInspection
from .hardware import Engine

_MAX_TRAIN_ROWS = 4
_MAX_TEST_ROWS = 2
_PILOT_EPOCHS = 3
_MIN_IMPROVEMENT_PERCENT = 0.2


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    settings: AutoTrainingSettings
    learning_rate: float
    optimizer: str
    sample_rows: int
    source_rows: int
    first_loss: float
    last_loss: float
    improvement_percent: float
    engine: str
    accelerator: str
    devices: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "settings": self.settings.as_dict(),
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "sample_rows": self.sample_rows,
            "source_rows": self.source_rows,
            "first_loss": self.first_loss,
            "last_loss": self.last_loss,
            "improvement_percent": self.improvement_percent,
            "engine": self.engine,
            "accelerator": self.accelerator,
            "devices": list(self.devices),
        }


def loss_trend(losses: tuple[float, ...]) -> tuple[float, float, float] | None:
    """Compare early and late robust medians; reject non-finite pilot runs."""
    if len(losses) < 3 or not all(math.isfinite(value) and value >= 0 for value in losses):
        return None
    span = max(1, len(losses) // 3)
    first = statistics.median(losses[:span])
    last = statistics.median(losses[-span:])
    if first <= 0 or max(losses) > max(first * 3, first + 2):
        return None
    improvement = 100 * (first - last) / first
    return first, last, improvement


def candidate_rates(
    settings: AutoTrainingSettings, typical_chars: int, scale: float = 4.0
) -> tuple[float, ...]:
    """Use model, batch, context and sampled row length to bound a pilot sweep."""
    context_pressure = min(1.0, math.sqrt(4 * settings.max_seq_length / max(typical_chars, 1)))
    rank_pressure = min(1.0, math.sqrt(8 / max(settings.rank, 1)))
    batch_pressure = min(1.0, math.sqrt(2 / max(settings.batch_size, 1)))
    model_pressure = min(1.0, math.sqrt(3 * 1024**3 / max(settings.model_size_bytes, 1)))
    scale_pressure = min(1.0, math.sqrt(4 / max(scale, 0.001)))
    anchor = max(
        2e-6,
        min(
            1e-5,
            1e-5
            * context_pressure
            * rank_pressure
            * batch_pressure
            * model_pressure
            * scale_pressure,
        ),
    )
    return tuple(float(f"{anchor / divisor:.3g}") for divisor in (1, 2.5, 6))


def _pilot_record(record: dict, context: int) -> dict:
    """Keep one short supervised exchange for a bounded calibration trial."""
    messages = record.get("messages", [])
    assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "assistant"
            and isinstance(messages[index].get("content"), str)
            and messages[index]["content"].strip()
        ),
        None,
    )
    if assistant_index is None:
        raise ConfigurationError("Calibration needs assistant text in its sampled records")
    prompt = next(
        (
            message["content"]
            for message in reversed(messages[:assistant_index])
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        "",
    )
    answer = messages[assistant_index]["content"]
    prompt_chars = min(160, max(64, context // 4))
    answer_chars = min(256, max(96, context // 2))
    excerpt = []
    if prompt.strip():
        excerpt.append({"role": "user", "content": prompt[-prompt_chars:].strip()})
    excerpt.append({"role": "assistant", "content": answer[:answer_chars].strip()})
    return {"messages": excerpt}


def sample_training_data(
    source: Path,
    destination: Path,
    *,
    context: int = 1024,
    train_rows: int = _MAX_TRAIN_ROWS,
    max_rows_per_file: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[int, int, int]:
    """Sample every file and bound the pilot's labelled token work."""
    files = dataset_files(source)
    rng = random.Random(0)
    training: list[dict] = []
    holdout: list[dict] = []
    lengths: list[int] = []
    counts = {"train": 0, "valid": 0, "test": 0}
    for file_index, (split, item) in enumerate(files, 1):
        if progress is not None:
            progress(file_index, len(files), item.name)
        for line, row in enumerate(_source_rows(item), 1):
            if max_rows_per_file is not None and line > max_rows_per_file:
                break
            example = normalize_sft_example(row, item, line)
            if set(example.modalities) - {"text"}:
                raise ConfigurationError(
                    "Calibration currently needs text training records; use manual settings "
                    "for a multimodal dataset"
                )
            record = example.record
            counts[split] += 1
            if split == "train":
                limit = train_rows + (0 if counts["valid"] or counts["test"] else _MAX_TEST_ROWS)
                target = training
                index = counts["train"]
                length = len(normalized_record_text(record))
                if len(lengths) < 128:
                    lengths.append(length)
                else:
                    replacement = rng.randrange(index)
                    if replacement < len(lengths):
                        lengths[replacement] = length
            else:
                limit = _MAX_TEST_ROWS
                target = holdout
                index = counts["valid"] + counts["test"]
            if len(target) < limit:
                target.append(record)
            else:
                replacement = rng.randrange(index)
                if replacement < limit:
                    target[replacement] = record
    if counts["train"] < 4:
        raise ConfigurationError("Calibration needs at least four valid training records")
    if not holdout and len(training) > train_rows:
        holdout = training[-min(_MAX_TEST_ROWS, len(training) - train_rows) :]
        training = training[: -len(holdout)]
    if holdout and len(training) > train_rows:
        training = training[:train_rows]
    destination.mkdir(parents=True, exist_ok=False)
    for name, records in (("train", training), ("test", holdout)):
        if not records:
            continue
        with (destination / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(
                    json.dumps(
                        _pilot_record(record, context), ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                )
    return len(training), counts["train"], int(statistics.median(lengths)) if lengths else 0


def calibrate_training(
    model_source: Path,
    model: ModelInspection,
    data_source: Path,
    benchmark: BenchmarkResult,
    *,
    engine: Engine,
    multi_gpu: str = "auto",
    optimizer: str = "auto",
    scale: float = 4.0,
    dropout: float = 0.0,
    mask_prompt: bool = True,
    seed: int = 0,
    grad_checkpoint: bool = True,
    grad_accumulation_steps: int = 1,
    split_mode: str = "layer",
    tensor_split: tuple[float, ...] = (),
    main_gpu: int = 0,
    distributed_workers: int = 0,
    require_full_context: bool = False,
    progress: Callable[[str], None] | None = None,
) -> CalibrationResult:
    """Select the fastest cautious rate with a measured short-run loss decrease."""
    if optimizer not in {"auto", "sgd", "adamw"}:
        raise ConfigurationError("Calibration optimizer must be auto, sgd, or adamw")
    if scale <= 0 or not 0 <= dropout < 1 or grad_accumulation_steps < 1:
        raise ConfigurationError(
            "Calibration needs positive scale and accumulation, and valid dropout"
        )
    selected_optimizer = (
        ("adamw" if engine is Engine.MLX else "sgd") if optimizer == "auto" else optimizer
    )
    notify = progress or (
        lambda message: print(f"osai: calibration {message}", file=sys.stderr, flush=True)
    )
    parent = os.environ.get("OSAI_CALIBRATION_PARENT")
    if parent and not Path(parent).is_dir():
        raise ConfigurationError("Calibration temporary directory is unavailable")
    with tempfile.TemporaryDirectory(prefix="osai-calibration-", dir=parent) as temporary:
        root = Path(temporary)
        notify("phase=sampling detail=Reading selected training files")
        sample = root / "sample"
        rows, source_rows, typical_chars = sample_training_data(
            data_source,
            sample,
            context=benchmark.settings.max_seq_length,
            train_rows=max(_MAX_TRAIN_ROWS, min(8, len(benchmark.devices) * 2)),
            max_rows_per_file=256 if os.environ.get("OSAI_CALIBRATION_QUICK") == "1" else None,
            progress=lambda index, total, name: notify(
                f"phase=sampling detail=Reading file {index} of {total}: {name}"
            ),
        )
        notify(
            f"phase=pilot detail=Testing {rows} short excerpts from {source_rows} inspected rows"
        )
        settings = benchmark.settings
        required_context = settings.max_seq_length
        trials: list[str] = []
        for index, rate in enumerate(candidate_rates(settings, typical_chars, scale), 1):
            notify(f"phase=pilot detail=Trial {index} of 3 at learning rate {rate:.2e}")
            output = root / f"trial-{index}"
            try:
                if engine is Engine.LLAMA_CPP:
                    native = train_gradient_gguf(
                        model_source,
                        sample,
                        output,
                        options=LlamaGradientOptions(
                            epochs=_PILOT_EPOCHS,
                            rank=settings.rank,
                            scale=scale,
                            num_layers=settings.num_layers,
                            context=settings.max_seq_length,
                            batch_size=settings.gguf_batch_size,
                            learning_rate=rate,
                            optimizer=selected_optimizer,
                            threads=settings.gguf_threads,
                            target_modules=settings.target_modules,
                            mask_prompt=mask_prompt,
                            seed=seed,
                            strict_base_hash=False,
                            multi_gpu=multi_gpu,
                            devices=benchmark.devices,
                            split_mode=split_mode,
                            tensor_split=tensor_split,
                            main_gpu=main_gpu,
                            auto_settings=not require_full_context,
                            calibration_pilot=True,
                        ),
                        accelerator=benchmark.accelerator,
                    )
                    losses = native.losses
                    manifest = json.loads(native.manifest.read_text(encoding="utf-8"))
                    if (
                        benchmark.accelerator != "cpu"
                        and native.accelerator != benchmark.accelerator
                    ):
                        raise ConfigurationError(
                            "GPU calibration fell back to CPU; repair the GPU backend before "
                            "training, or explicitly select CPU"
                        )
                    if len(benchmark.devices) > 1 and multi_gpu != "off":
                        used_devices = manifest.get("parallel_training", {}).get("devices", [])
                        if len(used_devices) != len(benchmark.devices):
                            raise ConfigurationError(
                                "Calibration did not use every selected GPU; check the "
                                "multi-GPU backend before training"
                            )
                    used_context = manifest.get("options", {}).get("context")
                    if isinstance(used_context, int) and used_context > 0:
                        settings = replace(settings, max_seq_length=used_context)
                    used_batch = manifest.get("options", {}).get("batch_size")
                    if isinstance(used_batch, int) and used_batch > 0:
                        settings = replace(settings, gguf_batch_size=used_batch)
                else:
                    config = TrainingConfig(
                        model=model_source,
                        format=ModelFormat.MLX,
                        data=sample,
                        output=output,
                        iterations=_PILOT_EPOCHS,
                        batch_size=settings.batch_size,
                        max_seq_length=settings.max_seq_length,
                        num_layers=settings.num_layers,
                        rank=settings.rank,
                        scale=scale,
                        dropout=dropout,
                        learning_rate=rate,
                        seed=seed,
                        grad_checkpoint=grad_checkpoint,
                        grad_accumulation_steps=grad_accumulation_steps,
                        mask_prompt=mask_prompt,
                        optimizer=selected_optimizer,
                        target_modules=settings.target_modules,
                        strict_base_hash=False,
                        merge_model=False,
                        materialize_base=False,
                        save_every=10_000,
                        steps_per_report=1,
                        val_batches=0,
                        multi_gpu=multi_gpu,
                        devices=benchmark.devices,
                        distributed_workers=distributed_workers,
                        auto_settings=not require_full_context,
                    )
                    losses = MlxBackend(accelerator=benchmark.accelerator).train(config).losses
                    used = json.loads(
                        (output / ".internal" / "mlx_lora_config.yaml").read_text(encoding="utf-8")
                    )
                    settings = replace(
                        settings,
                        max_seq_length=int(used["max_seq_length"]),
                        batch_size=int(used["batch_size"]),
                    )
                if require_full_context and settings.max_seq_length < required_context:
                    raise ConfigurationError(
                        "Full context does not fit this hardware during training; choose Windowing"
                    )
                trend = loss_trend(tuple(losses))
                if trend is not None and trend[2] >= _MIN_IMPROVEMENT_PERCENT:
                    first, last, improvement = trend
                    notify(f"phase=complete detail=Pilot loss fell {improvement:.1f}%")
                    return CalibrationResult(
                        settings,
                        rate,
                        selected_optimizer,
                        rows,
                        source_rows,
                        first,
                        last,
                        improvement,
                        engine.value,
                        benchmark.accelerator,
                        benchmark.devices,
                    )
                trials.append(f"{rate:.2e}: no reliable downward trend")
            except ConfigurationError:
                raise
            except OsAiError as exc:
                if require_full_context and settings.max_seq_length < required_context:
                    raise
                trials.append(f"{rate:.2e}: {exc}")
        guidance = (
            "Full context calibration could not finish at the required context; "
            "choose Windowing if the model or device cannot fit it. "
            if require_full_context
            and all("no reliable downward trend" not in trial for trial in trials)
            else "Calibration could not verify a decreasing pilot loss. Review the dataset "
            "and model, or turn off hardware fitting for manual settings. "
        )
        raise ConfigurationError(guidance + "; ".join(trials)[-700:])
