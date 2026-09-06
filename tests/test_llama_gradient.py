from pathlib import Path

import pytest

from osai.backends.llama_gradient import (
    LlamaGradientOptions,
    _gradient_command,
    _parse_best_checkpoint,
    _parse_epoch_losses,
    _parse_optimizer_steps,
    _parse_supervised_loss,
    _validate_memory_budget,
)
from osai.cli import build_parser
from osai.errors import ConfigurationError
from osai.hardware import Accelerator


def test_gradient_options_reject_non_divisible_batch():
    with pytest.raises(ConfigurationError, match="divide"):
        LlamaGradientOptions(context=32, batch_size=6).validate()


def test_gradient_options_reject_unknown_projection():
    options = LlamaGradientOptions(target_modules=("unsafe.projection",))
    with pytest.raises(ConfigurationError, match="unsupported"):
        options.validate()


def test_low_memory_guard_rejects_wide_backward_graph():
    options = LlamaGradientOptions(
        target_modules=("self_attn.q_proj", "mlp.down_proj")
    )
    with pytest.raises(ConfigurationError, match="low-memory"):
        _validate_memory_budget(options, 8 * 1024**3)


def test_cpu_gradient_command_disables_repack_and_devices():
    command = _gradient_command(
        Path("llama-finetune"),
        Path("base.gguf"),
        Path("initial.gguf"),
        Path("train.txt"),
        Path("trained.gguf"),
        LlamaGradientOptions(),
        Accelerator.CPU,
    )
    assert "--no-repack" in command
    assert command[command.index("-dev") + 1] == "none"
    assert command[command.index("-ngl") + 1] == "0"


def test_metal_gradient_command_preserves_multi_gpu_devices():
    options = LlamaGradientOptions(
        multi_gpu="on", devices=("MTL0", "MTL1"), split_mode="layer"
    )
    command = _gradient_command(
        Path("llama-finetune"),
        Path("base.gguf"),
        Path("initial.gguf"),
        Path("train.txt"),
        Path("trained.gguf"),
        options,
        Accelerator.METAL,
    )
    assert command[command.index("-dev") + 1] == "MTL0,MTL1"
    assert command[command.index("-sm") + 1] == "layer"


def test_gradient_log_parsers():
    output = (
        "train: [done] data=0000064/0000064 loss=0.8\r\n"
        "I epoch=1 train_loss=0.795442714 train_loss_uncertainty=0.1\n"
    )
    assert _parse_epoch_losses(output, Path("train.log")) == (0.795442714,)
    assert _parse_optimizer_steps(output, 1) == 64
    supervised = (
        "supervised optimizer step labels=4\n"
        "I checkpoint epoch=2 best_train_loss=0.25\n"
    )
    assert _parse_optimizer_steps(supervised, 10, mask_prompt=True) == 1
    assert _parse_best_checkpoint(supervised, (0.5, 0.25)) == (2, 0.25)
    assert _parse_supervised_loss("I eval_loss=0.03125", Path("eval.log")) == 0.03125


def test_cli_exposes_gradient_optimizer():
    parser = build_parser()
    common = ["train", "--tier", "small"]
    args = parser.parse_args(common)
    assert args.optimizer == "auto"
    assert args.gguf_optimizer is None
    assert args.merge_model is None


def test_cli_allows_explicit_adamw_gradient_optimizer():
    args = build_parser().parse_args(
        ["train", "--tier", "small", "--optimizer", "adamw"]
    )
    assert args.optimizer == "adamw"


def test_cli_retains_legacy_gguf_optimizer_alias():
    args = build_parser().parse_args(
        ["train", "--tier", "small", "--gguf-optimizer", "adamw"]
    )
    assert args.gguf_optimizer == "adamw"


def test_cli_exposes_default_on_merge_controls():
    parser = build_parser()
    enabled = parser.parse_args(["train", "--tier", "small", "--merge"])
    disabled = parser.parse_args(["train", "--tier", "small", "--no-merge"])
    assert enabled.merge_model is True
    assert disabled.merge_model is False


def test_cli_defaults_training_runs_to_sessions():
    args = build_parser().parse_args(["train", "--tier", "small"])
    assert args.sessions_root.name == "sessions"
    assert args.session_name is None
