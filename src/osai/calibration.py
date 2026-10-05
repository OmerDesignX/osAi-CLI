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
from .backends.llama_gradient import LlamaGradientOptions, _memory_failure, train_gradient_gguf
from .backends.mlx import MlxBackend
from .config import ModelFormat, TrainingConfig
from .dataset import normalize_sft_example, normalized_record_text
from .dataset_source import _source_rows, dataset_files
from .errors import ConfigurationError, OsAiError
from .formats import ModelInspection
from .hardware import Engine
from .multi_gpu import llama_device_free_bytes
from .paths import llama_binary

_MAX_TRAIN_ROWS = 4
_MAX_TEST_ROWS = 2
_PILOT_EPOCHS = 3
_MAX_INSPECTED_ROWS_PER_FILE = 256


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
    device_speeds: tuple[float, ...] = ()

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
            "device_speeds": list(self.device_speeds),
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


def _pilot_microbatch(
    settings: AutoTrainingSettings, model: ModelInspection, benchmark: BenchmarkResult
) -> int:
    """Estimate a candidate from available memory; the pilot verifies it."""
    available = settings.memory_budget_bytes
    if benchmark.devices:
        from .hardware import Accelerator

        free = llama_device_free_bytes(
            llama_binary("llama-completion"), Accelerator(benchmark.accelerator)
        )
        if not all(device in free for device in benchmark.devices):
            return settings.gguf_batch_size
        available = min(free[device] for device in benchmark.devices)
    model_share = model.size_bytes / max(1, len(benchmark.devices))
    headroom = max(0, available - model_share)
    context_bound = max(1, math.isqrt(settings.max_seq_length))
    # Packed model bytes per block are not a measure of activation bytes per token.
    # The native pilot verifies this memory-bounded candidate and halves it on OOM.
    memory_fraction = min(1.0, headroom / max(model.size_bytes, 1))
    budget_batch = max(1, int(context_bound * memory_fraction))
    candidate = 1 << (budget_batch.bit_length() - 1)
    if model.context_length is not None:
        while (
            candidate > 1
            and math.ceil(settings.max_seq_length / candidate) * candidate > model.context_length
        ):
            candidate //= 2
    return max(settings.gguf_batch_size, candidate)


def _pilot_record(record: dict, context: int, excerpt_chars: int, rng: random.Random) -> dict:
    """Keep a varied exchange with a budget derived from source record lengths."""
    messages = record.get("messages", [])
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    if not assistant_indices:
        raise ConfigurationError("Calibration needs assistant text in its sampled records")
    assistant_index = rng.choice(assistant_indices)
    prompt = next(
        (
            message["content"]
            for message in reversed(messages[:assistant_index])
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        "",
    )
    answer = messages[assistant_index]["content"]
    prompt_chars = min(excerpt_chars // 4, context // 4)
    answer_chars = min(excerpt_chars - prompt_chars, context // 2)
    answer_start = rng.randrange(max(1, len(answer) - answer_chars + 1))
    excerpt = []
    if prompt.strip():
        excerpt.append({"role": "user", "content": prompt[-prompt_chars:].strip()})
    excerpt.append(
        {"role": "assistant", "content": answer[answer_start : answer_start + answer_chars].strip()}
    )
    return {"messages": excerpt}


def _sample_source_rows(path: Path, limit: int | None):
    """Read JSONL strata across a large file rather than only its first rows."""
    if limit is None or path.suffix.casefold() not in {".jsonl", ".ndjson"}:
        yield from _source_rows(path)
        return
    size = path.stat().st_size
    if size == 0:
        return
    with path.open("rb") as source:
        if sum(1 for _ in zip(source, range(limit + 1), strict=False)) <= limit:
            yield from _source_rows(path)
            return
    seen: set[int] = set()
    with path.open("rb") as source:
        for slot in range(limit):
            offset = size * slot // limit
            source.seek(max(0, offset - 1))
            if offset and source.read(1) != b"\n":
                source.readline()
            start = source.tell()
            if start >= size or start in seen:
                continue
            seen.add(start)
            raw = source.readline()
            try:
                row = json.loads(raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"invalid JSON near byte {start} in {path}") from exc
            if not isinstance(row, dict):
                raise ConfigurationError(f"expected a JSON object near byte {start} in {path}")
            yield row


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
        for line, row in enumerate(_sample_source_rows(item, max_rows_per_file), 1):
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
    typical_chars = int(statistics.median(lengths)) if lengths else 0
    excerpt_chars = min(max(128, math.isqrt(max(typical_chars, 1)) * 4), max(128, context * 2))
    destination.mkdir(parents=True, exist_ok=False)
    for name, records in (("train", training), ("test", holdout)):
        if not records:
            continue
        with (destination / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(
                    json.dumps(
                        _pilot_record(record, context, excerpt_chars, rng),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    return len(training), counts["train"], typical_chars


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
    total_training_rows: int | None = None,
    training_epochs: int = 1,
    progress: Callable[[str], None] | None = None,
) -> CalibrationResult:
    """Select a measured rate, starting in the middle of the adaptive range."""
    if optimizer not in {"auto", "sgd", "adamw"}:
        raise ConfigurationError("Calibration optimizer must be auto, sgd, or adamw")
    if scale <= 0 or not 0 <= dropout < 1 or grad_accumulation_steps < 1:
        raise ConfigurationError(
            "Calibration needs positive scale and accumulation, and valid dropout"
        )
    selected_optimizer = "adamw" if optimizer == "auto" else optimizer
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
            max_rows_per_file=_MAX_INSPECTED_ROWS_PER_FILE,
            progress=lambda index, total, name: notify(
                f"phase=sampling detail=Reading file {index} of {total}: {name}"
            ),
        )
        notify(
            f"phase=pilot detail=Testing {rows} short excerpts from {source_rows} inspected rows"
        )
        settings = benchmark.settings
        required_context = settings.max_seq_length
        if engine is Engine.LLAMA_CPP:
            settings = replace(
                settings,
                gguf_batch_size=_pilot_microbatch(settings, model, benchmark),
            )
        trials: list[str] = []
        candidates = sorted(set(candidate_rates(settings, typical_chars, scale)))
        middle = len(candidates) // 2
        rates = [candidates[middle], *reversed(candidates[:middle]), *candidates[middle + 1 :]]
        starting_rate = rates[0]
        allow_higher = True
        best: CalibrationResult | None = None
        best_test_loss = math.inf
        for index, rate in enumerate(rates, 1):
            if rate > starting_rate and not allow_higher:
                break
            notify(f"phase=pilot detail=Trial {index} of {len(rates)} at learning rate {rate:.2e}")
            output = root / f"trial-{index}"
            try:
                if engine is Engine.LLAMA_CPP:
                    while True:
                        batch = settings.gguf_batch_size
                        context = math.ceil(required_context / batch) * batch
                        try:
                            native = train_gradient_gguf(
                                model_source,
                                sample,
                                output,
                                options=LlamaGradientOptions(
                                    epochs=_PILOT_EPOCHS,
                                    rank=settings.rank,
                                    scale=scale,
                                    num_layers=settings.num_layers,
                                    context=context,
                                    batch_size=batch,
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
                                    auto_settings=False,
                                    calibration_pilot=True,
                                ),
                                accelerator=benchmark.accelerator,
                            )
                            break
                        except OsAiError:
                            logs = (output / "logs").glob("train*.log")
                            if batch <= 1 or not any(_memory_failure(log) for log in logs):
                                raise
                            settings = replace(settings, gguf_batch_size=max(1, batch // 2))
                            notify(
                                "phase=pilot detail=GPU memory limit; retrying with "
                                f"microbatch {settings.gguf_batch_size}"
                            )
                            output = root / f"trial-{index}-batch-{settings.gguf_batch_size}"
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
                        used_devices = manifest.get("model_sharded_training", {}).get("devices", [])
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
                before_test = (
                    getattr(native, "initial_test_loss", None)
                    if engine is Engine.LLAMA_CPP
                    else None
                )
                after_test = (
                    getattr(native, "test_loss", None) if engine is Engine.LLAMA_CPP else None
                )
                heldout_improved = before_test is None or (
                    after_test is not None
                    and math.isfinite(after_test)
                    and after_test < before_test
                )
                if (
                    trend is None
                    or trend[2] < 0
                    or (
                        before_test is not None
                        and (
                            after_test is None
                            or not math.isfinite(after_test)
                            or after_test > before_test
                        )
                    )
                ):
                    allow_higher = False
                if trend is not None and trend[2] > 0 and heldout_improved:
                    first, last, improvement = trend
                    result = CalibrationResult(
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
                        # Short pilot rows do not measure sustained device throughput.
                        (),
                    )
                    score = after_test if after_test is not None else last
                    if score < best_test_loss:
                        best, best_test_loss = result, score
                    break
                trials.append(f"{rate:.2e}: training or held-out loss did not reliably decline")
                if best is not None:
                    break
            except ConfigurationError:
                raise
            except OsAiError as exc:
                allow_higher = False
                if require_full_context and settings.max_seq_length < required_context:
                    raise
                trials.append(f"{rate:.2e}: {exc}")
                if best is not None:
                    break
        if best is not None:
            if engine is Engine.LLAMA_CPP and total_training_rows is not None:
                if total_training_rows < rows or training_epochs < 1:
                    raise ConfigurationError("invalid full-run exposure for calibration")
                pilot_exposure = rows * _PILOT_EPOCHS
                planned_exposure = total_training_rows * training_epochs
                rate_factor = min(1.0, math.sqrt(pilot_exposure / planned_exposure))
                if rate_factor < 1.0:
                    probe_rate = best.learning_rate
                    best = replace(best, learning_rate=probe_rate * rate_factor)
                    notify(
                        "phase=complete detail=Scaling short-pilot rate "
                        f"{probe_rate:.2e} to sustained rate {best.learning_rate:.2e} "
                        f"for {total_training_rows} training records"
                    )
            notify(
                "phase=complete detail=Pilot loss fell "
                f"{best.improvement_percent:.2f}%; full-run learning rate "
                f"{best.learning_rate:.2e}"
            )
            return best
        guidance = (
            "Full context calibration could not finish at the required context; "
            "choose Windowing if the model or device cannot fit it. "
            if require_full_context and all("command exited" in trial for trial in trials)
            else "Calibration could not verify a decreasing pilot loss. Review the dataset "
            "and model, or turn off hardware fitting for manual settings. "
        )
        raise ConfigurationError(guidance + "; ".join(trials)[-700:])
