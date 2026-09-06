"""Post-fine-tuning alignment on frozen quantized MLX and GGUF bases."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from ..alignment import (
    AlignmentDataset,
    AlignmentType,
    example_as_dict,
    preference_gradients,
    preference_loss,
    reward_advantages,
)
from ..config import ModelFormat, TrainingConfig
from ..errors import ConfigurationError, DependencyError, TrainingError, VerificationError
from ..formats import inspect_model
from ..hardware import Accelerator, select_llama_accelerator
from ..io import atomic_json, fingerprint, sha256_file
from ..offline import offline_environment
from ..paths import llama_binary
from ..process import run_logged
from ..rollouts import RolloutResult, RolloutSettings, collect_live_rollouts
from ..session import SessionLayout
from .llama_gradient import (
    LlamaGradientOptions,
    _evaluate_supervised_loss,
    _gradient_command,
)
from .llama_utils import _verify_adapter, _write_corpus
from .mlx import MlxBackend, _resolve_distributed_workers
from .rollout import generate_gguf_answers, generate_mlx_answers


@dataclass(frozen=True, slots=True)
class AlignmentOptions:
    iterations: int = 10
    learning_rate: float = 1e-5
    beta: float = 0.1
    gamma: float = 0.5
    ppo_clip: float = 0.2
    max_seq_length: int = 128
    batch_size: int = 1
    threads: int = 2
    optimizer: str = "auto"
    rollouts: RolloutSettings = RolloutSettings()

    def validate(self) -> None:
        if min(self.iterations, self.max_seq_length, self.batch_size, self.threads) < 1:
            raise ConfigurationError("alignment integer settings must be at least 1")
        if min(self.learning_rate, self.beta, self.ppo_clip) <= 0:
            raise ConfigurationError("alignment learning rate, beta, and clip must be positive")
        if self.optimizer not in {"auto", "sgd", "adamw"}:
            raise ConfigurationError("alignment optimizer must be auto, sgd, or adamw")
        self.rollouts.validate(AlignmentType.PPO)


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    output: Path
    adapter: Path
    manifest: Path
    losses: tuple[float, ...]
    alignment_type: str
    optimizer: str
    rollouts: Path | None = None
    generated_examples: int = 0


def align_mlx(
    model: Path,
    adapter: Path,
    dataset: AlignmentDataset,
    output: Path,
    config: TrainingConfig,
    options: AlignmentOptions,
    *,
    python: Path | None,
    accelerator: str,
) -> AlignmentResult:
    options.validate()
    if options.optimizer == "auto":
        options = replace(options, optimizer="adamw")
    base = inspect_model(model, ModelFormat.MLX)
    source_adapter = _mlx_adapter_dir(adapter)
    destination = output.resolve()
    layout = SessionLayout.at(destination)
    layout.create()
    adapter_dir = layout.adapters / "mlx"
    backend = MlxBackend(python=python, accelerator=accelerator)
    report = backend.preflight()
    source_dataset = dataset
    rollout_result: RolloutResult | None = None
    if options.rollouts.enabled:
        rollout_result = collect_live_rollouts(
            source_dataset,
            layout.rollouts,
            engine="mlx",
            adapter=source_adapter,
            settings=options.rollouts,
            generate=lambda requests: generate_mlx_answers(
                backend,
                base.path,
                source_adapter,
                requests,
                options.rollouts,
                work=layout.work,
                log_path=layout.logs / "rollouts.log",
            ),
        )
        dataset = rollout_result.dataset
    runner_config = {
        "model": str(base.path),
        "adapter": str(source_adapter),
        "output": str(adapter_dir),
        "examples": [example_as_dict(item) for item in dataset.examples],
        "alignment_type": dataset.alignment_type.value,
        "iterations": options.iterations,
        "learning_rate": options.learning_rate,
        "beta": options.beta,
        "gamma": options.gamma,
        "ppo_clip": options.ppo_clip,
        "optimizer": options.optimizer,
        "max_seq_length": options.max_seq_length,
        "batch_size": options.batch_size,
    }
    atomic_json(layout.work / "mlx_alignment.json", runner_config)
    manifest = _manifest(
        base,
        source_adapter,
        dataset,
        options,
        "mlx-gradient",
        source_dataset=source_dataset,
        rollout=rollout_result,
    )
    atomic_json(layout.run_manifest, manifest)
    before = tuple(fingerprint(path, full_hash=True) for path in base.shards)
    workers = _resolve_distributed_workers(config, report)
    command = [
        str(backend.python), "-m", "osai._offline_runner", "align-mlx",
        str(layout.work / "mlx_alignment.json"),
    ]
    if workers > 1:
        command = [
            str(backend.python), "-m", "mlx._distributed_utils.launch",
            "-n", str(workers), "--backend", "nccl", "--hosts", "127.0.0.1",
            "--python", str(backend.python), "--", *command[1:],
        ]
    try:
        run_logged(command, log_path=layout.logs / "alignment.log", env=backend.environment())
        adapter_file = adapter_dir / "adapters.safetensors"
        result_file = adapter_dir / "alignment_result.json"
        if not adapter_file.is_file() or not result_file.is_file():
            raise TrainingError("MLX alignment completed without its final adapter")
        runner_result = json.loads(result_file.read_text(encoding="utf-8"))
        losses = tuple(runner_result["losses"])
        _verify_common(base, before, source_adapter / "adapters.safetensors", adapter_file)
        manifest.update(
            {
                "status": "completed",
                "completed_at": _now(),
                "adapter": str(adapter_dir),
                "initial_loss": runner_result["initial_loss"],
                "losses": list(losses),
                "distributed_workers": workers,
                "trainable_tensors": runner_result["trainable_tensors"],
                "trainable_parameters": runner_result["trainable_parameters"],
                "invariants": _invariants(),
            }
        )
        atomic_json(layout.run_manifest, manifest)
        return AlignmentResult(
            destination, adapter_dir, layout.run_manifest, losses,
            dataset.alignment_type.value, options.optimizer,
            rollout_result.data_file if rollout_result else None,
            rollout_result.generated_examples if rollout_result else 0,
        )
    except BaseException as exc:
        _fail_manifest(layout.run_manifest, manifest, exc)
        raise


def align_gguf(
    model: Path,
    adapter: Path,
    dataset: AlignmentDataset,
    output: Path,
    config: TrainingConfig,
    options: AlignmentOptions,
    *,
    accelerator: str,
) -> AlignmentResult:
    """Align a GGUF LoRA with native preference backpropagation."""

    options.validate()
    if options.optimizer == "auto":
        options = replace(options, optimizer="sgd")
    base = inspect_model(model, ModelFormat.GGUF)
    source_adapter = adapter.expanduser().resolve()
    _verify_adapter(source_adapter, base.architecture)
    destination = output.resolve()
    layout = SessionLayout.at(destination)
    layout.create()
    final_adapter = layout.adapters / "gguf" / "adapter.gguf"
    final_adapter.parent.mkdir(parents=True, exist_ok=True)
    binary = llama_binary("llama-finetune")
    if binary is None:
        raise DependencyError("llama-finetune is not built; run `osai build-llama`")
    selected = select_llama_accelerator(accelerator)
    eval_options = LlamaGradientOptions(
        epochs=1,
        rank=1,
        scale=1.0,
        context=max(32, options.max_seq_length),
        batch_size=_dividing_batch(max(32, options.max_seq_length), config.gguf_batch_size),
        learning_rate=options.learning_rate,
        optimizer=options.optimizer,
        threads=options.threads,
        target_modules=("mlp.down_proj",),
        multi_gpu=config.multi_gpu,
        devices=config.devices,
        split_mode=config.split_mode,
        tensor_split=config.tensor_split,
        main_gpu=config.main_gpu,
    )
    source_dataset = dataset
    rollout_result: RolloutResult | None = None
    if options.rollouts.enabled:
        selected_holder = [selected]

        def generate(requests):
            responses, used_accelerator = generate_gguf_answers(
                base.path,
                source_adapter,
                requests,
                options.rollouts,
                accelerator=selected_holder[0],
                devices=config,
                log_path=layout.logs / "rollouts.log",
            )
            selected_holder[0] = used_accelerator
            return responses

        rollout_result = collect_live_rollouts(
            source_dataset,
            layout.rollouts,
            engine="llama.cpp",
            adapter=source_adapter,
            settings=options.rollouts,
            generate=generate,
        )
        dataset = rollout_result.dataset
        selected = selected_holder[0]
    manifest = _manifest(
        base,
        source_adapter,
        dataset,
        options,
        "llama.cpp-native-preference-backprop",
        source_dataset=source_dataset,
        rollout=rollout_result,
    )
    atomic_json(layout.run_manifest, manifest)
    before = tuple(fingerprint(path, full_hash=True) for path in base.shards)
    corpora = _alignment_corpora(layout, dataset, eval_options.context)
    advantages = (
        reward_advantages(
            dataset.alignment_type,
            [float(item.reward) for item in dataset.examples if item.reward is not None],
            [item.prompt for item in dataset.examples],
        )
        if dataset.alignment_type
        in {AlignmentType.REINFORCE, AlignmentType.RLOO, AlignmentType.GRPO}
        else None
    )

    def score_one(adapter_path: Path, index: int) -> tuple[float, float]:
        item = dataset.examples[index]
        pc = -_evaluate_supervised_loss(
            binary,
            base.path,
            adapter_path,
            corpora[index][0],
            eval_options,
            layout.logs / f"score-{index}-chosen.log",
            selected,
            example_weights=(1.0,),
        )
        pr = (
            -_evaluate_supervised_loss(
                binary,
                base.path,
                adapter_path,
                corpora[index][1],
                eval_options,
                layout.logs / f"score-{index}-rejected.log",
                selected,
                example_weights=(1.0,),
            )
            if item.rejected is not None
            else 0.0
        )
        return pc, pr

    def score(adapter_path: Path) -> float:
        values = [
            _example_loss(
                dataset.alignment_type,
                dataset.examples[index],
                *score_one(adapter_path, index),
                *references[index],
                options,
                advantage=None if advantages is None else advantages[index],
            )
            for index in range(len(dataset.examples))
        ]
        return sum(values) / len(values)

    references: list[tuple[float, float]] = []
    try:
        for index, item in enumerate(dataset.examples):
            rc = -_evaluate_supervised_loss(
                binary, base.path, source_adapter, corpora[index][0], eval_options,
                layout.logs / f"reference-{index}-chosen.log", selected,
                example_weights=(1.0,),
            )
            rr = (
                -_evaluate_supervised_loss(
                    binary, base.path, source_adapter, corpora[index][1], eval_options,
                    layout.logs / f"reference-{index}-rejected.log", selected,
                    example_weights=(1.0,),
                )
                if item.rejected is not None else 0.0
            )
            references.append((rc, rr))
        current_path = source_adapter
        losses = [score(current_path)]
        training_accelerator = selected
        fallback_reason: str | None = None
        for step in range(options.iterations):
            for index, item in enumerate(dataset.examples):
                pc, pr = score_one(current_path, index)
                rc, rr = references[index]
                dpc, dpr = _example_gradients(
                    dataset.alignment_type,
                    item,
                    pc,
                    pr,
                    rc,
                    rr,
                    options,
                    advantage=None if advantages is None else advantages[index],
                )
                weights = [-dpc]
                if item.rejected is not None:
                    weights.append(-dpr)
                next_path = layout.work / f"alignment-{step}-{index}.gguf"
                training_accelerator = _native_alignment_step(
                    binary,
                    base.path,
                    current_path,
                    _pair_corpus(layout, corpora[index], step, index),
                    next_path,
                    eval_options,
                    weights,
                    training_accelerator,
                    layout.logs / f"backprop-{step}-{index}.log",
                )
                current_path = next_path
            losses.append(score(current_path))
            atomic_json(
                layout.progress_manifest,
                {"step": step + 1, "loss": losses[-1], "optimizer": "backprop"},
            )
        os.replace(current_path, final_adapter)
        _verify_common(base, before, source_adapter, final_adapter)
        manifest.update({
            "status": "completed", "completed_at": _now(),
            "adapter": str(final_adapter), "losses": losses,
            "alignment_accelerator_requested": selected.value,
            "alignment_accelerator": training_accelerator.value,
            "accelerator_fallback_reason": (
                fallback_reason
                or (
                    "accelerator rejected the backward graph"
                    if training_accelerator is not selected
                    else None
                )
            ),
            "invariants": _invariants(),
        })
        atomic_json(layout.run_manifest, manifest)
        return AlignmentResult(
            destination, final_adapter, layout.run_manifest, tuple(losses),
            dataset.alignment_type.value, options.optimizer,
            rollout_result.data_file if rollout_result else None,
            rollout_result.generated_examples if rollout_result else 0,
        )
    except BaseException as exc:
        _fail_manifest(layout.run_manifest, manifest, exc)
        raise


def _example_loss(
    method,
    item,
    pc,
    pr,
    rc,
    rr,
    options: AlignmentOptions,
    *,
    advantage: float | None = None,
) -> float:
    if method is AlignmentType.PPO and item.rejected is not None:
        return 0.5 * (
            preference_loss(
                "ppo", pc, 0.0, rc, 0.0,
                beta=options.beta, clip=options.ppo_clip, reward=1.0,
            )
            + preference_loss(
                "ppo", pr, 0.0, rr, 0.0,
                beta=options.beta, clip=options.ppo_clip, reward=-1.0,
            )
        )
    return preference_loss(
        method,
        pc,
        pr,
        item.old_logprob if item.old_logprob is not None else rc,
        rr,
        beta=options.beta,
        gamma=options.gamma,
        reward=item.reward if advantage is None else advantage,
        clip=options.ppo_clip,
    )


def _example_gradients(
    method,
    item,
    pc,
    pr,
    rc,
    rr,
    options: AlignmentOptions,
    *,
    advantage: float | None = None,
):
    if method is AlignmentType.PPO and item.rejected is not None:
        chosen, _ = preference_gradients(
            "ppo", pc, 0.0, rc, 0.0,
            beta=options.beta, clip=options.ppo_clip, reward=1.0,
        )
        rejected, _ = preference_gradients(
            "ppo", pr, 0.0, rr, 0.0,
            beta=options.beta, clip=options.ppo_clip, reward=-1.0,
        )
        return 0.5 * chosen, 0.5 * rejected
    return preference_gradients(
        method,
        pc,
        pr,
        item.old_logprob if item.old_logprob is not None else rc,
        rr,
        beta=options.beta,
        gamma=options.gamma,
        reward=item.reward if advantage is None else advantage,
        clip=options.ppo_clip,
    )


def _pair_corpus(
    layout: SessionLayout,
    corpora: tuple[Path, Path | None],
    step: int,
    index: int,
) -> Path:
    separator = "\n<|osai_record_end|>\n"
    sections = [corpora[0].read_text(encoding="utf-8").strip()]
    if corpora[1] is not None:
        sections.append(corpora[1].read_text(encoding="utf-8").strip())
    destination = layout.work / f"alignment-pair-{step}-{index}.txt"
    destination.write_text(separator.join(sections) + "\n", encoding="utf-8")
    return destination


def _native_alignment_step(
    binary: Path,
    model: Path,
    adapter: Path,
    corpus: Path,
    output: Path,
    settings: LlamaGradientOptions,
    weights: list[float],
    accelerator: Accelerator,
    log_path: Path,
) -> Accelerator:
    if not weights or not all(math.isfinite(value) for value in weights):
        raise TrainingError("alignment produced invalid preference gradients")
    output.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    environment = offline_environment()
    environment["OSAI_MASK_PROMPT"] = "1"
    environment["OSAI_EXAMPLE_WEIGHTS"] = ",".join(
        format(value, ".17g") for value in weights
    )
    command = _gradient_command(
        binary, model, adapter, corpus, output, settings, accelerator
    )
    try:
        run_logged(command, log_path=log_path, env=environment)
    except TrainingError:
        if accelerator is Accelerator.CPU:
            raise
        output.unlink(missing_ok=True)
        command = _gradient_command(
            binary, model, adapter, corpus, output, settings, Accelerator.CPU
        )
        run_logged(command, log_path=log_path, env=environment)
        accelerator = Accelerator.CPU
    if not output.is_file() or output.stat().st_size == 0:
        raise TrainingError(f"llama.cpp did not save an aligned adapter; see {log_path}")
    return accelerator


def _alignment_corpora(layout: SessionLayout, dataset: AlignmentDataset, context: int):
    result = []
    for index, item in enumerate(dataset.examples):
        chosen_source = layout.work / f"alignment-{index}-chosen.jsonl"
        chosen_source.write_text(
            json.dumps({"prompt": item.prompt, "completion": item.chosen or item.response}) + "\n",
            encoding="utf-8",
        )
        chosen = _write_corpus(
            chosen_source, layout.work / f"alignment-{index}-chosen.txt", context,
            repeat_to_minimum=False, record_separator="\n<|osai_record_end|>\n",
        )
        rejected = None
        if item.rejected is not None:
            rejected_source = layout.work / f"alignment-{index}-rejected.jsonl"
            rejected_source.write_text(
                json.dumps({"prompt": item.prompt, "completion": item.rejected}) + "\n",
                encoding="utf-8",
            )
            rejected = _write_corpus(
                rejected_source, layout.work / f"alignment-{index}-rejected.txt", context,
                repeat_to_minimum=False, record_separator="\n<|osai_record_end|>\n",
            )
        result.append((chosen, rejected))
    return result


def _mlx_adapter_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    root = candidate if candidate.is_dir() else candidate.parent
    for name in ("adapters.safetensors", "adapter_config.json"):
        if not (root / name).is_file():
            raise ConfigurationError(f"MLX adapter directory is missing {name}: {root}")
    return root


def _manifest(
    base,
    adapter,
    dataset,
    options,
    backend,
    *,
    source_dataset=None,
    rollout: RolloutResult | None = None,
):
    dataset_record = dataset.as_dict()
    dataset_record["train_sha256"] = sha256_file(dataset.path / "train.jsonl")
    result = {
        "schema_version": 1, "status": "aligning", "started_at": _now(),
        "stage": "alignment", "backend": backend, "local_only": True,
        "model": base.as_dict(), "input_adapter": str(adapter),
        "dataset": dataset_record, "options": asdict(options),
        "rollouts": (
            rollout.as_dict()
            if rollout is not None
            else {"mode": "static", "generated_examples": 0}
        ),
    }
    if source_dataset is not None and source_dataset.path != dataset.path:
        source_record = source_dataset.as_dict()
        source_record["train_sha256"] = sha256_file(
            source_dataset.path / "train.jsonl"
        )
        result["source_dataset"] = source_record
    return result


def _verify_common(base, before, old_adapter, new_adapter):
    after = tuple(fingerprint(path, full_hash=True) for path in base.shards)
    if before != after:
        raise VerificationError("quantized base changed during alignment")
    if sha256_file(old_adapter) == sha256_file(new_adapter):
        raise VerificationError("alignment optimizer did not change the adapter")
    inspect_model(base.path, base.format)


def _invariants():
    return {
        "base_files_unchanged": True,
        "base_quantization_unchanged": True,
        "full_precision_base_created": False,
        "only_adapter_tensors_trainable": True,
    }


def _fail_manifest(path, manifest, exc):
    manifest.update({
        "status": "failed", "completed_at": _now(),
        "error": {"type": type(exc).__name__, "message": str(exc)},
    })
    atomic_json(path, manifest)


def _dividing_batch(context: int, preferred: int) -> int:
    value = min(context, preferred)
    while context % value:
        value -= 1
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
