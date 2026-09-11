"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import __version__
from .alignment import AlignmentType, load_alignment_dataset
from .auto_settings import select_auto_settings
from .backends.alignment import AlignmentOptions, align_gguf, align_mlx
from .backends.llama_cpp import build_llama_cpp, validate_adapter
from .backends.llama_gradient import LlamaGradientOptions, train_gradient_gguf
from .bundle import verify_model_bundle
from .catalog import ModelTier, bundled_root, list_catalog, resolve_model
from .config import ModelFormat, TrainingConfig
from .dataset import validate_dataset
from .errors import ConfigurationError, DependencyError, OsAiError, TrainingError
from .formats import inspect_model
from .gguf_adapter import convert_mlx_adapter
from .hardware import Accelerator, Engine, detect_hardware, select_engine
from .health import check_sessions
from .io import OutputLock, atomic_json
from .learning_proof import prove_learning
from .merge import MergedModelResult, merge_gguf_model, merge_mlx_model
from .model_download import ConsoleProgress, ModelDownload, ensure_official_model
from .paths import project_root
from .rollouts import RolloutSettings
from .session import (
    BaseBundleResult,
    SessionLayout,
    publish_base_adapter_bundle,
    timestamped_session_path,
)
from .system import doctor
from .trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osai",
        description="osAi quantization-preserving training for MLX and GGUF models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="check host and optional backends")
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    doctor_parser.set_defaults(handler=_doctor)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a GGUF or MLX model")
    inspect_parser.add_argument("model", type=Path)
    inspect_parser.add_argument("--format", choices=["mlx", "gguf"])
    inspect_parser.set_defaults(handler=_inspect)

    train_parser = subparsers.add_parser("train", help="run quantized-base adapter training")
    source = train_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--tier", choices=[tier.value for tier in ModelTier])
    source.add_argument("--custom", dest="custom_model")
    train_parser.add_argument(
        "--engine", choices=[engine.value for engine in Engine], default="auto"
    )
    train_parser.add_argument(
        "--accelerator",
        choices=[accelerator.value for accelerator in Accelerator],
        default="auto",
    )
    train_parser.add_argument(
        "--stage",
        choices=["fine-tuning", "alignment", "fine-tune-align"],
        default="fine-tuning",
        help="run supervised fine-tuning, post-training alignment, or both in order",
    )
    train_parser.add_argument("--data", type=Path)
    train_parser.add_argument("--alignment-data", type=Path)
    train_parser.add_argument(
        "--alignment-type",
        choices=[value.value for value in AlignmentType],
        default=None,
        help=(
            "alignment objective; auto selects DPO for preference pairs and PPO for "
            "reward rows; combined preference runs offer ORPO when omitted"
        ),
    )
    train_parser.add_argument(
        "--adapter", type=Path, help="existing adapter required for alignment-only runs"
    )
    train_parser.add_argument("--alignment-iterations", type=int, default=10)
    train_parser.add_argument("--alignment-learning-rate", type=float)
    train_parser.add_argument("--alignment-beta", type=float, default=0.1)
    train_parser.add_argument("--alignment-gamma", type=float, default=0.5)
    train_parser.add_argument("--ppo-clip", type=float, default=0.2)
    train_parser.add_argument(
        "--live-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "generate fresh answers from the local fine-tuned policy before alignment "
            "(default: enabled)"
        ),
    )
    train_parser.add_argument(
        "--rollouts-per-prompt",
        type=int,
        default=2,
        help="fresh local answers per source prompt (default: 2)",
    )
    train_parser.add_argument(
        "--rollout-max-tokens",
        type=int,
        default=32,
        help="maximum new tokens per local answer (default: 32)",
    )
    train_parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=0.8,
        help="local sampling temperature; 0 is deterministic (default: 0.8)",
    )
    train_parser.add_argument(
        "--rollout-top-p",
        type=float,
        default=0.95,
        help="local nucleus-sampling probability (default: 0.95)",
    )
    train_parser.add_argument(
        "--rollout-seed",
        type=int,
        default=0,
        help="first reproducible local rollout seed (default: 0)",
    )
    train_parser.add_argument(
        "--sessions-root", type=Path, default=project_root() / "sessions"
    )
    train_parser.add_argument("--session-name")
    train_parser.add_argument("--bundled-root", type=Path)
    train_parser.add_argument("--custom-root", type=Path)
    train_parser.add_argument(
        "--download-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download and verify a missing official tier (default: enabled)",
    )
    train_parser.add_argument(
        "--multi-gpu",
        choices=["auto", "on", "off"],
        default=None,
        help="use all visible GPUs automatically, require several GPUs, or use one GPU",
    )
    train_parser.add_argument(
        "--device",
        action="append",
        dest="devices",
        help="llama.cpp device name; repeat to set an ordered multi-GPU device list",
    )
    train_parser.add_argument(
        "--split-mode",
        choices=["none", "layer", "row", "tensor"],
        default=None,
        help="llama.cpp multi-GPU model split",
    )
    train_parser.add_argument(
        "--tensor-split",
        help="comma-separated positive GPU proportions, for example 3,1",
    )
    train_parser.add_argument("--main-gpu", type=int)
    train_parser.add_argument(
        "--distributed-workers",
        type=int,
        default=None,
        help="MLX CUDA workers; 0 detects a safe count from visible GPUs and batch size",
    )
    train_parser.add_argument(
        "--auto-settings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fit memory-sensitive settings to the model, RAM, and engine",
    )
    train_parser.add_argument("--iterations", type=int)
    train_parser.add_argument("--batch-size", type=int, help="MLX training batch size")
    train_parser.add_argument("--rank", type=int)
    train_parser.add_argument("--scale", type=float)
    train_parser.add_argument("--num-layers", type=int)
    train_parser.add_argument("--max-seq-length", type=int)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--dropout", type=float)
    train_parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "resize images before local VLM training; model processor default is "
            "used when omitted"
        ),
    )
    train_parser.add_argument(
        "--video-fps", type=float, help="video frame sampling rate (default: 2)"
    )
    train_parser.add_argument(
        "--video-max-frames", type=int, help="maximum sampled frames per video (default: 32)"
    )
    train_parser.add_argument(
        "--assistant-token-id",
        type=int,
        help="assistant boundary token for completion-only VLM loss",
    )
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        "--grad-accumulation-steps",
        dest="grad_accumulation_steps",
        type=int,
    )
    train_parser.add_argument(
        "--gradient-checkpointing",
        "--grad-checkpoint",
        dest="grad_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="recompute MLX activations during backward to reduce memory (default: enabled)",
    )
    train_parser.add_argument("--save-every", type=int)
    train_parser.add_argument("--steps-per-report", type=int)
    train_parser.add_argument("--steps-per-eval", type=int)
    train_parser.add_argument("--val-batches", type=int)
    train_parser.add_argument(
        "--mask-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="exclude prompt tokens from supervised loss (default: enabled)",
    )
    optimizer = train_parser.add_mutually_exclusive_group()
    optimizer.add_argument(
        "--optimizer",
        choices=["auto", "sgd", "adamw"],
        default="auto",
        help="training and alignment optimizer; auto uses AdamW for MLX and SGD for llama.cpp",
    )
    optimizer.add_argument(
        "--gguf-optimizer",
        choices=["sgd", "adamw"],
        help=argparse.SUPPRESS,
    )
    train_parser.add_argument("--gguf-batch-size", type=int)
    train_parser.add_argument("--gguf-threads", type=int)
    train_parser.add_argument(
        "--target-module",
        action="append",
        choices=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        dest="target_modules",
        help="repeat to select LoRA projections; backprop defaults to mlp.down_proj",
    )
    train_parser.add_argument(
        "--strict-base-hash", action=argparse.BooleanOptionalAction, default=None
    )
    train_parser.add_argument(
        "--merge",
        action=argparse.BooleanOptionalAction,
        dest="merge_model",
        default=None,
        help="publish a standalone lossless quantized-residual bundle (default: enabled)",
    )
    train_parser.add_argument(
        "--materialize-base",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the base in the base-plus-adapter deployment (default: enabled)",
    )
    train_parser.add_argument("--python", type=Path, help="Python executable with MLX installed")
    train_parser.set_defaults(handler=_train)

    models_parser = subparsers.add_parser(
        "models", help="list official downloadable tiers and custom local models"
    )
    models_parser.add_argument("--bundled-root", type=Path)
    models_parser.add_argument("--custom-root", type=Path)
    models_parser.add_argument("--json", action="store_true", dest="as_json")
    models_parser.set_defaults(handler=_models)

    verify_models_parser = subparsers.add_parser(
        "verify-models", help="verify downloaded official models without network access"
    )
    verify_models_parser.add_argument(
        "--root", type=Path, default=project_root() / "osCode-Models"
    )
    verify_models_parser.set_defaults(handler=_verify_models)

    sessions_parser = subparsers.add_parser(
        "check-sessions", help="validate completed local session publications"
    )
    sessions_parser.add_argument(
        "--root", type=Path, default=project_root() / "sessions"
    )
    sessions_parser.add_argument("--require-completed", action="store_true")
    sessions_parser.set_defaults(handler=_check_sessions)

    select_parser = subparsers.add_parser(
        "select", help="resolve a local model, engine, and accelerator"
    )
    selection_source = select_parser.add_mutually_exclusive_group(required=True)
    selection_source.add_argument("--tier", choices=[tier.value for tier in ModelTier])
    selection_source.add_argument("--custom", dest="custom_model")
    select_parser.add_argument(
        "--engine", choices=[engine.value for engine in Engine], default="auto"
    )
    select_parser.add_argument("--bundled-root", type=Path)
    select_parser.add_argument("--custom-root", type=Path)
    select_parser.add_argument(
        "--download-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download and verify a missing official tier (default: enabled)",
    )
    select_parser.set_defaults(handler=_select)

    export_parser = subparsers.add_parser(
        "export-gguf", help="convert an MLX adapter to a llama.cpp GGUF adapter"
    )
    export_parser.add_argument("--adapter", required=True, type=Path)
    export_parser.add_argument("--base", required=True, type=Path)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    export_parser.set_defaults(handler=_export_gguf)

    build_parser = subparsers.add_parser("build-llama", help="build vendored llama.cpp")
    build_parser.add_argument(
        "--log", type=Path, default=project_root() / "build" / "llama-build.log"
    )
    build_parser.add_argument("--jobs", type=int)
    build_parser.add_argument(
        "--accelerator",
        choices=[accelerator.value for accelerator in Accelerator],
        default="auto",
    )
    build_parser.add_argument(
        "--no-cpu-fallback", action="store_false", dest="cpu_fallback", default=True
    )
    build_parser.set_defaults(handler=_build_llama)

    validate_parser = subparsers.add_parser(
        "validate-gguf", help="load a GGUF base and adapter with llama.cpp"
    )
    validate_parser.add_argument("--base", required=True, type=Path)
    validate_parser.add_argument("--adapter", required=True, type=Path)
    validate_parser.add_argument("--log", required=True, type=Path)
    validate_parser.add_argument("--prompt", default="Reply with only: adapter validation passed")
    validate_parser.add_argument("--tokens", type=int, default=8)
    validate_parser.add_argument("--context", type=int, default=128)
    validate_parser.add_argument(
        "--accelerator",
        choices=[accelerator.value for accelerator in Accelerator],
        default="auto",
    )
    validate_parser.set_defaults(handler=_validate_gguf)

    proof_parser = subparsers.add_parser(
        "prove-learning", help="compare deterministic base and trained-adapter output"
    )
    proof_parser.add_argument("--engine", choices=["mlx", "llama.cpp"], required=True)
    proof_parser.add_argument("--model", required=True, type=Path)
    proof_parser.add_argument("--adapter", required=True, type=Path)
    proof_parser.add_argument("--prompt", required=True)
    proof_parser.add_argument("--expected", required=True)
    proof_parser.add_argument("--output", required=True, type=Path)
    proof_parser.add_argument("--max-tokens", type=int, default=16)
    proof_parser.add_argument("--context", type=int, default=128)
    proof_parser.add_argument("--python", type=Path)
    proof_parser.add_argument(
        "--accelerator",
        choices=[accelerator.value for accelerator in Accelerator],
        default="auto",
    )
    proof_parser.set_defaults(handler=_prove_learning)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except OsAiError as exc:
        print(f"osai: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("osai: interrupted", file=sys.stderr)
        return 130


def _doctor(args: argparse.Namespace) -> int:
    report = doctor().as_dict()
    if args.as_json:
        _print_json(report)
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    expected = ModelFormat(args.format) if args.format else None
    _print_json(inspect_model(args.model, expected).as_dict())
    return 0


def _train(args: argparse.Namespace) -> int:
    if args.stage == "fine-tuning":
        payload = _fine_tune(args)
    elif args.stage == "alignment":
        if args.adapter is None:
            raise ConfigurationError("--adapter is required with --stage alignment")
        payload = _alignment_stage(args, adapter=args.adapter)
    else:
        if args.data is None:
            raise ConfigurationError("--data is required with --stage fine-tune-align")
        if args.alignment_data is None:
            raise ConfigurationError(
                "--alignment-data is required with --stage fine-tune-align"
            )
        _choose_combined_alignment(args)
        label = args.session_name or args.tier or args.custom_model or (
            args.config.stem if args.config else "run"
        )
        parent = timestamped_session_path(args.sessions_root, f"{label}-fine-tune-align")
        fine_args = argparse.Namespace(**vars(args))
        fine_args._session_override = parent / "stages" / "fine-tuning"
        fine_args.merge_model = False
        fine_args.materialize_base = False
        fine_payload = _fine_tune(fine_args)
        align_args = argparse.Namespace(**vars(args))
        align_args._session_override = parent
        align_args.engine = fine_payload["engine"]
        payload = _alignment_stage(
            align_args,
            adapter=Path(fine_payload["adapter"]),
            model=Path(fine_payload["model"]),
            engine=Engine(fine_payload["engine"]),
        )
        payload["fine_tuning"] = fine_payload
    _print_json(payload)
    return 0


def _fine_tune(args: argparse.Namespace) -> dict[str, Any]:
    engine = select_engine(args.engine)
    fallback_gguf: Path | None = None
    if args.config:
        if args.data is None:
            raise ConfigurationError("--data is required with --config")
        pending_output = args.sessions_root.expanduser().resolve() / ".pending"
        config = TrainingConfig.from_file(
            args.config,
            data=args.data,
            output=pending_output,
        )
        if args.target_modules:
            config = replace(config, target_modules=tuple(args.target_modules))
        if args.merge_model is not None:
            config = replace(config, merge_model=args.merge_model)
        if args.materialize_base is not None:
            config = replace(config, materialize_base=args.materialize_base)
        if args.engine == Engine.AUTO.value:
            if config.effective_format is ModelFormat.MLX:
                engine = Engine.MLX
            elif config.companion_mlx is None:
                engine = Engine.LLAMA_CPP
        session = getattr(args, "_session_override", None) or timestamped_session_path(
            args.sessions_root, args.session_name or args.config.stem,
        )
        config = replace(config, output=session)
    else:
        if args.data is None:
            raise ConfigurationError("--data is required with --tier or --custom")
        selection = _resolve_cli_model(args)
        engine = selection.engine
        fallback_gguf = selection.entry.gguf
        target_modules = (
            tuple(args.target_modules) if args.target_modules else ("mlp.down_proj",)
        )
        source_name = args.tier or args.custom_model or "run"
        config = TrainingConfig(
            model=selection.model,
            format=ModelFormat.MLX if engine is Engine.MLX else ModelFormat.GGUF,
            companion_mlx=selection.companion_mlx,
            data=args.data.resolve(),
            output=args.sessions_root.expanduser().resolve() / ".pending",
            iterations=1 if args.iterations is None else args.iterations,
            batch_size=1 if args.batch_size is None else args.batch_size,
            rank=2 if args.rank is None else args.rank,
            scale=4.0 if args.scale is None else args.scale,
            num_layers=1 if args.num_layers is None else args.num_layers,
            max_seq_length=64 if args.max_seq_length is None else args.max_seq_length,
            learning_rate=1e-5 if args.learning_rate is None else args.learning_rate,
            strict_base_hash=(
                True if args.strict_base_hash is None else args.strict_base_hash
            ),
            merge_model=True if args.merge_model is None else args.merge_model,
            materialize_base=(
                True if args.materialize_base is None else args.materialize_base
            ),
            gguf_batch_size=(8 if args.gguf_batch_size is None else args.gguf_batch_size),
            gguf_threads=2 if args.gguf_threads is None else args.gguf_threads,
            multi_gpu=args.multi_gpu or "auto",
            devices=tuple(args.devices or ()),
            split_mode=args.split_mode or "layer",
            tensor_split=_parse_tensor_split(args.tensor_split),
            main_gpu=0 if args.main_gpu is None else args.main_gpu,
            distributed_workers=(
                0 if args.distributed_workers is None else args.distributed_workers
            ),
            target_modules=target_modules,
        )
        session = getattr(args, "_session_override", None) or timestamped_session_path(
            args.sessions_root, args.session_name or f"{source_name}-{args.engine}",
        )
        config = replace(config, output=session)
    config = _resolve_training_settings(config, args, engine)
    selected_optimizer = _select_optimizer(args, config, engine)
    dataset_summary = validate_dataset(config.data)
    multimodal = bool(set(dataset_summary.modalities) - {"text"})
    if multimodal:
        # The text-only compact profile uses a deliberately tiny context.  Media
        # processors commonly emit hundreds or thousands of visual/audio tokens,
        # so use a safe baseline unless the user explicitly chose a limit.
        if not args.config and args.max_seq_length is None and config.max_seq_length < 2048:
            config = replace(config, max_seq_length=2048)
        if config.effective_format is ModelFormat.GGUF and config.companion_mlx is None:
            raise ConfigurationError(
                "llama.cpp has no multimodal backward API; GGUF VLM training requires "
                "a matching quantized MLX VLM in the custom model's mlx folder so osAi "
                "can backpropagate through media locally and export the language adapter"
            )
        requested_optimizer = getattr(args, "gguf_optimizer", None) or args.optimizer
        if requested_optimizer == "auto":
            selected_optimizer = "adamw"
        result = train(
            replace(config, optimizer=selected_optimizer),
            python=args.python,
            mlx_accelerator=args.accelerator,
            llama_accelerator=args.accelerator,
        )
        target_format = config.effective_format
        adapter = (
            result.gguf_adapter
            if target_format is ModelFormat.GGUF
            else result.mlx_adapter
        )
        return {
            "status": "completed",
            "engine": engine.value,
            "training_backend": "mlx-vlm",
            "modalities": list(dataset_summary.modalities),
            "output": str(result.output),
            "mlx_adapter": str(result.mlx_adapter),
            "gguf_adapter": str(result.gguf_adapter) if result.gguf_adapter else None,
            "manifest": str(result.manifest),
            "base_plus_adapter": str(result.deployment_manifest.parent),
            "deployment_manifest": str(result.deployment_manifest),
            "merged_model": str(result.merged_model) if result.merged_model else None,
            "reported_losses": result.losses,
            "test_loss": result.test_loss,
            "optimizer": selected_optimizer,
            "auto_settings": _auto_payload(config),
            "model": str(config.model),
            "format": target_format.value,
            "adapter": str(adapter),
        }
    if engine is Engine.MLX and args.accelerator in {
        Accelerator.MPS.value,
        Accelerator.VULKAN.value,
    }:
        raise ConfigurationError(
            "MLX supports Metal on macOS and CUDA or CPU on Linux, not MPS or Vulkan"
        )
    if engine is Engine.MLX:
        try:
            result = train(
                replace(config, optimizer=selected_optimizer),
                python=args.python,
                mlx_accelerator=args.accelerator,
                llama_accelerator=args.accelerator,
            )
        except (DependencyError, TrainingError):
            if args.engine != Engine.AUTO.value:
                raise
            if config.effective_format is ModelFormat.GGUF:
                fallback_gguf = config.model
            if fallback_gguf is None:
                raise
            if args.tier is not None:
                fallback_gguf = _ensure_official_tier(
                    args, Engine.LLAMA_CPP
                ).model
            print("osai: MLX execution failed; retrying the local GGUF with llama.cpp")
            engine = Engine.LLAMA_CPP
            config = replace(config, model=fallback_gguf, format=ModelFormat.GGUF)
            config = _resolve_training_settings(config, args, engine)
            selected_optimizer = _select_optimizer(args, config, engine)
        else:
            payload = {
                "status": "completed",
                "engine": engine.value,
                "output": str(result.output),
                "mlx_adapter": str(result.mlx_adapter),
                "gguf_adapter": str(result.gguf_adapter) if result.gguf_adapter else None,
                "manifest": str(result.manifest),
                "base_plus_adapter": str(result.deployment_manifest.parent),
                "deployment_manifest": str(result.deployment_manifest),
                "merged_model": str(result.merged_model) if result.merged_model else None,
                "reported_losses": result.losses,
                "test_loss": result.test_loss,
                "optimizer": selected_optimizer,
                "auto_settings": _auto_payload(config),
                "model": str(config.training_model),
                "format": "mlx",
                "adapter": str(result.mlx_adapter),
            }
            return payload
    if engine is Engine.LLAMA_CPP:
        if config.effective_format is not ModelFormat.GGUF:
            raise ConfigurationError("llama.cpp quantized training requires a GGUF model")
        base_context = inspect_model(config.model, ModelFormat.GGUF).context_length
        context = max(32, config.max_seq_length)
        if base_context is not None and context > base_context:
            raise ConfigurationError(
                f"max sequence length {context} exceeds the model context {base_context}"
            )
        gradient_learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else (config.learning_rate if args.config else 1e-5)
        )
        native = train_gradient_gguf(
            config.model,
            config.data,
            config.output,
            options=LlamaGradientOptions(
                epochs=config.iterations,
                rank=config.rank,
                scale=config.scale,
                num_layers=config.num_layers,
                context=context,
                batch_size=config.gguf_batch_size,
                learning_rate=gradient_learning_rate,
                optimizer=selected_optimizer,
                threads=config.gguf_threads,
                seed=config.seed,
                target_modules=config.target_modules,
                mask_prompt=config.mask_prompt,
                strict_base_hash=config.strict_base_hash,
                multi_gpu=config.multi_gpu,
                devices=config.devices,
                split_mode=config.split_mode,
                tensor_split=config.tensor_split,
                main_gpu=config.main_gpu,
            ),
            accelerator=args.accelerator,
        )
        base_bundle, merged = _publish_native_session(config, native, args.accelerator)
        payload = {
            "status": "completed",
            "engine": engine.value,
            "output": str(native.output),
            "gguf_adapter": str(native.adapter),
            "manifest": str(native.manifest),
            "base_plus_adapter": str(SessionLayout.at(config.output).base_adapter),
            "deployment_manifest": str(
                SessionLayout.at(config.output).deployment_manifest
            ),
            "materialized_base": str(base_bundle.path) if base_bundle.path else None,
            "merged_model": str(merged.path) if merged else None,
            "reported_losses": native.losses,
            "test_loss": native.test_loss,
            "accelerator": native.accelerator,
            "optimizer_steps": native.optimizer_steps,
            "loss_reducing_steps": native.loss_reducing_steps,
            "optimizer": selected_optimizer,
            "auto_settings": _auto_payload(config),
            "model": str(config.model),
            "format": "gguf",
            "adapter": str(native.adapter),
        }
    return payload


def _alignment_stage(
    args: argparse.Namespace,
    *,
    adapter: Path,
    model: Path | None = None,
    engine: Engine | None = None,
) -> dict[str, Any]:
    if args.alignment_data is None:
        raise ConfigurationError("--alignment-data is required for alignment")
    dataset = load_alignment_dataset(
        args.alignment_data,
        args.alignment_type or AlignmentType.AUTO,
        live_rollouts=args.live_rollouts,
    )
    session = getattr(args, "_session_override", None)
    if args.config:
        pending = session or args.sessions_root.expanduser().resolve() / ".pending"
        config = TrainingConfig.from_file(
            args.config, data=args.alignment_data, output=pending
        )
        if model is not None:
            config = replace(
                config,
                model=model,
                format=ModelFormat.MLX if engine is Engine.MLX else ModelFormat.GGUF,
            )
        if engine is None:
            engine = select_engine(args.engine)
            if args.engine == Engine.AUTO.value:
                if config.effective_format is ModelFormat.MLX:
                    engine = Engine.MLX
                elif config.companion_mlx is None:
                    engine = Engine.LLAMA_CPP
    else:
        if model is None:
            selection = _resolve_cli_model(args)
            model = selection.model
            engine = selection.engine
        assert model is not None and engine is not None
        config = TrainingConfig(
            model=model,
            format=ModelFormat.MLX if engine is Engine.MLX else ModelFormat.GGUF,
            data=args.alignment_data,
            output=session or args.sessions_root.expanduser().resolve() / ".pending",
            batch_size=1 if args.batch_size is None else args.batch_size,
            max_seq_length=64 if args.max_seq_length is None else args.max_seq_length,
            gguf_batch_size=8 if args.gguf_batch_size is None else args.gguf_batch_size,
            gguf_threads=2 if args.gguf_threads is None else args.gguf_threads,
            merge_model=True if args.merge_model is None else args.merge_model,
            materialize_base=(
                True if args.materialize_base is None else args.materialize_base
            ),
            strict_base_hash=(
                True if args.strict_base_hash is None else args.strict_base_hash
            ),
            multi_gpu=args.multi_gpu or "auto",
            devices=tuple(args.devices or ()),
            split_mode=args.split_mode or "layer",
            tensor_split=_parse_tensor_split(args.tensor_split),
            main_gpu=0 if args.main_gpu is None else args.main_gpu,
            distributed_workers=(
                0 if args.distributed_workers is None else args.distributed_workers
            ),
        )
    assert engine is not None
    if session is None:
        label = args.session_name or args.tier or args.custom_model or (
            args.config.stem if args.config else "alignment"
        )
        session = timestamped_session_path(args.sessions_root, f"{label}-alignment")
    config = replace(config, output=session)
    config = _resolve_training_settings(config, args, engine)
    selected_optimizer = _select_optimizer(args, config, engine)
    config = replace(config, optimizer=selected_optimizer)
    options = AlignmentOptions(
        iterations=args.alignment_iterations,
        learning_rate=(
            args.alignment_learning_rate
            if args.alignment_learning_rate is not None
            else (1e-5 if engine is Engine.MLX else 1e-3)
        ),
        beta=args.alignment_beta,
        gamma=args.alignment_gamma,
        ppo_clip=args.ppo_clip,
        max_seq_length=max(32, config.max_seq_length),
        batch_size=config.batch_size,
        threads=config.gguf_threads,
        optimizer=selected_optimizer,
        rollouts=RolloutSettings(
            enabled=args.live_rollouts,
            samples_per_prompt=args.rollouts_per_prompt,
            max_tokens=args.rollout_max_tokens,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            seed=args.rollout_seed,
        ),
    )
    if engine is Engine.MLX:
        if args.accelerator in {Accelerator.MPS.value, Accelerator.VULKAN.value}:
            raise ConfigurationError("MLX alignment supports Metal, CUDA, or CPU")
        result = align_mlx(
            config.training_model,
            adapter,
            dataset,
            session,
            config,
            options,
            python=args.python,
            accelerator=args.accelerator,
        )
        base = inspect_model(config.training_model, ModelFormat.MLX)
        adapters = {"mlx": result.adapter}
    else:
        if config.effective_format is not ModelFormat.GGUF:
            raise ConfigurationError("llama.cpp alignment requires a GGUF base")
        result = align_gguf(
            config.model,
            adapter,
            dataset,
            session,
            config,
            options,
            accelerator=args.accelerator,
        )
        base = inspect_model(config.model, ModelFormat.GGUF)
        adapters = {"gguf": result.adapter}

    layout = SessionLayout.at(session)
    base_bundle = publish_base_adapter_bundle(
        layout, base, adapters, materialize_base=config.materialize_base
    )
    merged = None
    if config.merge_model:
        merged = (
            merge_mlx_model(
                base,
                result.adapter,
                layout,
                python=args.python,
                accelerator=args.accelerator,
            )
            if engine is Engine.MLX
            else merge_gguf_model(
                base,
                result.adapter,
                layout,
                accelerator=args.accelerator,
                threads=config.gguf_threads,
            )
        )
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            "base_plus_adapter": {
                "path": str(layout.base_adapter),
                "deployment_manifest": str(layout.deployment_manifest),
                "base": base_bundle.as_dict(),
            },
            "merged_model": merged.as_dict() if merged else None,
        }
    )
    atomic_json(result.manifest, manifest)
    return {
        "status": "completed",
        "stage": "alignment",
        "engine": engine.value,
        "alignment_type": result.alignment_type,
        "optimizer": result.optimizer,
        "method": "reverse-mode gradients",
        "output": str(session),
        "model": str(base.path),
        "format": base.format.value,
        "adapter": str(result.adapter),
        "manifest": str(result.manifest),
        "reported_losses": result.losses,
        "rollouts": str(result.rollouts) if result.rollouts else None,
        "generated_examples": result.generated_examples,
        "base_plus_adapter": str(layout.base_adapter),
        "merged_model": str(merged.path) if merged else None,
    }


def _publish_native_session(
    config: TrainingConfig,
    native,
    accelerator: str,
) -> tuple[BaseBundleResult, MergedModelResult | None]:
    layout = SessionLayout.at(config.output)
    base = inspect_model(config.model, ModelFormat.GGUF)
    try:
        manifest = json.loads(native.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read native training manifest: {exc}") from exc
    with OutputLock(layout.root):
        try:
            manifest["status"] = "publishing"
            atomic_json(native.manifest, manifest)
            base_bundle = publish_base_adapter_bundle(
                layout,
                base,
                {"gguf": native.adapter},
                materialize_base=config.materialize_base,
            )
            merged = (
                merge_gguf_model(
                    base,
                    native.adapter,
                    layout,
                    accelerator=accelerator,
                    threads=config.gguf_threads,
                )
                if config.merge_model
                else None
            )
            manifest.update(
                {
                    "status": "completed",
                    "base_plus_adapter": {
                        "path": str(layout.base_adapter),
                        "deployment_manifest": str(layout.deployment_manifest),
                        "base": base_bundle.as_dict(),
                    },
                    "merged_model": merged.as_dict() if merged else None,
                    "merge_invariants": {
                        "final_quantization_matches_base": bool(merged),
                        "temporary_full_precision_merge": bool(
                            merged and merged.temporary_full_precision_intermediate
                        ),
                        "merge_strategy": merged.merge_strategy if merged else None,
                        "temporary_merge_files_removed": True,
                    },
                    "auto_settings": _auto_payload(config),
                }
            )
            atomic_json(native.manifest, manifest)
            return base_bundle, merged
        except BaseException as exc:
            manifest.update(
                {
                    "status": "failed",
                    "publication_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            atomic_json(native.manifest, manifest)
            raise


def _resolve_training_settings(
    config: TrainingConfig,
    args: argparse.Namespace,
    engine: Engine,
) -> TrainingConfig:
    if args.auto_settings:
        expected = ModelFormat.MLX if engine is Engine.MLX else ModelFormat.GGUF
        model_path = config.training_model if engine is Engine.MLX else config.model
        selected = select_auto_settings(
            inspect_model(model_path, expected),
            engine=engine,
        )
        config = replace(
            config,
            auto_settings=True,
            auto_profile=selected.profile,
            memory_budget_bytes=selected.memory_budget_bytes,
            batch_size=selected.batch_size,
            max_seq_length=selected.max_seq_length,
            num_layers=selected.num_layers,
            rank=selected.rank,
            gguf_batch_size=selected.gguf_batch_size,
            gguf_threads=selected.gguf_threads,
            target_modules=selected.target_modules,
        )
        print(
            "osai: auto settings "
            f"profile={selected.profile} ram={selected.physical_memory_bytes / 2**30:.1f}GiB "
            f"budget={selected.memory_budget_bytes / 2**30:.1f}GiB "
            f"model={selected.model_size_bytes / 2**30:.1f}GiB "
            f"context={selected.max_seq_length} batch={selected.batch_size} "
            f"gguf_batch={selected.gguf_batch_size} layers={selected.num_layers} "
            f"rank={selected.rank} threads={selected.gguf_threads}",
            file=sys.stderr,
        )

    overrides: dict[str, Any] = {}
    for argument, field_name in (
        ("iterations", "iterations"),
        ("batch_size", "batch_size"),
        ("rank", "rank"),
        ("scale", "scale"),
        ("num_layers", "num_layers"),
        ("max_seq_length", "max_seq_length"),
        ("learning_rate", "learning_rate"),
        ("dropout", "dropout"),
        ("video_fps", "video_fps"),
        ("video_max_frames", "video_max_frames"),
        ("assistant_token_id", "assistant_token_id"),
        ("seed", "seed"),
        ("grad_accumulation_steps", "grad_accumulation_steps"),
        ("save_every", "save_every"),
        ("steps_per_report", "steps_per_report"),
        ("steps_per_eval", "steps_per_eval"),
        ("val_batches", "val_batches"),
        ("gguf_batch_size", "gguf_batch_size"),
        ("gguf_threads", "gguf_threads"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            overrides[field_name] = value
    if args.target_modules:
        overrides["target_modules"] = tuple(args.target_modules)
    if getattr(args, "image_size", None) is not None:
        overrides["image_width"], overrides["image_height"] = args.image_size
    if args.mask_prompt is not None:
        overrides["mask_prompt"] = args.mask_prompt
    if args.grad_checkpoint is not None:
        overrides["grad_checkpoint"] = args.grad_checkpoint
    if args.merge_model is not None:
        overrides["merge_model"] = args.merge_model
    if args.materialize_base is not None:
        overrides["materialize_base"] = args.materialize_base
    if args.strict_base_hash is not None:
        overrides["strict_base_hash"] = args.strict_base_hash
    if args.multi_gpu is not None:
        overrides["multi_gpu"] = args.multi_gpu
    if args.devices is not None:
        overrides["devices"] = tuple(args.devices)
    if args.split_mode is not None:
        overrides["split_mode"] = args.split_mode
    if args.tensor_split is not None:
        overrides["tensor_split"] = _parse_tensor_split(args.tensor_split)
    if args.main_gpu is not None:
        overrides["main_gpu"] = args.main_gpu
    if args.distributed_workers is not None:
        overrides["distributed_workers"] = args.distributed_workers
    return replace(config, **overrides)


def _select_optimizer(
    args: argparse.Namespace,
    config: TrainingConfig,
    engine: Engine,
) -> str:
    """Resolve one clear CLI choice after automatic engine selection."""

    legacy = getattr(args, "gguf_optimizer", None)
    requested = legacy or getattr(args, "optimizer", "auto")
    if requested != "auto":
        return requested
    if args.config and config.optimizer != "auto":
        if engine is Engine.LLAMA_CPP and config.optimizer not in {"sgd", "adamw"}:
            raise ConfigurationError(
                "llama.cpp optimizer from config must be sgd or adamw"
            )
        return config.optimizer
    return "adamw" if engine is Engine.MLX else "sgd"


def _choose_combined_alignment(args: argparse.Namespace) -> None:
    """Resolve an omitted combined-run objective before fine-tuning starts."""

    if args.alignment_type is not None:
        return
    dataset = load_alignment_dataset(args.alignment_data, AlignmentType.AUTO)
    if dataset.schema != "preference":
        args.alignment_type = AlignmentType.AUTO.value
        return
    if not sys.stdin.isatty():
        args.alignment_type = AlignmentType.DPO.value
        print(
            "osai: non-interactive preference run selected DPO; pass "
            "--alignment-type orpo to select ORPO",
            file=sys.stderr,
        )
        return
    prompt = (
        "Use ORPO for this fine-tune-align run? ORPO also reinforces the chosen "
        "response during preference alignment [y/N]: "
    )
    while True:
        try:
            answer = input(prompt).strip().casefold()
        except EOFError:
            answer = ""
        if answer in {"y", "yes"}:
            args.alignment_type = AlignmentType.ORPO.value
            return
        if answer in {"", "n", "no"}:
            args.alignment_type = AlignmentType.DPO.value
            return
        print("Please answer yes or no.", file=sys.stderr)


def _auto_payload(config: TrainingConfig) -> dict[str, Any] | None:
    if not config.auto_settings:
        return None
    return {
        "profile": config.auto_profile,
        "memory_budget_bytes": config.memory_budget_bytes,
        "batch_size": config.batch_size,
        "max_seq_length": config.max_seq_length,
        "num_layers": config.num_layers,
        "rank": config.rank,
        "gguf_batch_size": config.gguf_batch_size,
        "gguf_threads": config.gguf_threads,
        "target_modules": list(config.target_modules),
    }


def _parse_tensor_split(value: str | None) -> tuple[float, ...]:
    if value is None:
        return ()
    try:
        result = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ConfigurationError("--tensor-split must contain comma-separated numbers") from exc
    if not result or any(number <= 0 for number in result):
        raise ConfigurationError("--tensor-split values must be positive")
    return result


def _models(args: argparse.Namespace) -> int:
    entries = [
        entry.as_dict()
        for entry in list_catalog(bundled=args.bundled_root, custom=args.custom_root)
    ]
    if args.as_json:
        print(json.dumps(entries, indent=2, sort_keys=True))
    else:
        for entry in entries:
            print(
                f"{entry['name']}: source={entry['source']} tier={entry['tier'] or '-'} "
                f"mlx={'ready' if entry['mlx_materialized'] else '-'} "
                f"llama.cpp={'ready' if entry['gguf_materialized'] else '-'}"
            )
    return 0


def _verify_models(args: argparse.Namespace) -> int:
    _print_json(verify_model_bundle(args.root).as_dict())
    return 0


def _check_sessions(args: argparse.Namespace) -> int:
    _print_json(
        check_sessions(args.root, require_completed=args.require_completed).as_dict()
    )
    return 0


def _select(args: argparse.Namespace) -> int:
    _print_json(_resolve_cli_model(args).as_dict())
    return 0


def _resolve_cli_model(args: argparse.Namespace):
    hardware = detect_hardware()
    engine = select_engine(args.engine, hardware)
    if args.tier is not None:
        _ensure_official_tier(args, engine)
    return resolve_model(
        tier=args.tier,
        custom=args.custom_model,
        engine=engine,
        bundled=args.bundled_root,
        custom_models=args.custom_root,
        hardware=hardware,
    )


def _ensure_official_tier(
    args: argparse.Namespace, engine: Engine
) -> ModelDownload:
    progress = ConsoleProgress()
    try:
        return ensure_official_model(
            args.bundled_root or bundled_root(),
            runtime=engine.value,
            tier=args.tier,
            allow_download=getattr(args, "download_model", True),
            progress=progress,
        )
    except BaseException:
        progress.finish_error()
        raise


def _export_gguf(args: argparse.Namespace) -> int:
    base = inspect_model(args.base, ModelFormat.GGUF)
    result = convert_mlx_adapter(args.adapter, args.output, base_gguf=base, dtype=args.dtype)
    _print_json(
        {
            "path": str(result.path),
            "architecture": result.architecture,
            "tensor_count": result.tensor_count,
            "size_bytes": result.size_bytes,
            "lora_alpha": result.lora_alpha,
        }
    )
    return 0


def _build_llama(args: argparse.Namespace) -> int:
    result = build_llama_cpp(
        log_path=args.log,
        jobs=args.jobs,
        accelerator=args.accelerator,
        cpu_fallback=args.cpu_fallback,
    )
    _print_json(
        {
            "status": "completed",
            "elapsed_seconds": result.elapsed_seconds,
            "log": str(result.log_path),
            "accelerator": result.accelerator,
            "fallback_from": result.fallback_from,
        }
    )
    return 0


def _validate_gguf(args: argparse.Namespace) -> int:
    inspect_model(args.base, ModelFormat.GGUF)
    inspect_model(args.adapter, ModelFormat.GGUF)
    result = validate_adapter(
        args.base,
        args.adapter,
        log_path=args.log,
        prompt=args.prompt,
        tokens=args.tokens,
        context=args.context,
        accelerator=args.accelerator,
    )
    _print_json(
        {
            "status": "completed",
            "binary": str(result.binary),
            "log": str(result.log_file),
            "elapsed_seconds": result.elapsed_seconds,
            "accelerator": result.accelerator,
        }
    )
    return 0


def _prove_learning(args: argparse.Namespace) -> int:
    result = prove_learning(
        engine=args.engine,
        model=args.model,
        adapter=args.adapter,
        prompt=args.prompt,
        expected=args.expected,
        output=args.output,
        accelerator=args.accelerator,
        python=args.python,
        max_tokens=args.max_tokens,
        context=args.context,
    )
    _print_json(result.as_dict())
    return 0 if result.passed else 1


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
