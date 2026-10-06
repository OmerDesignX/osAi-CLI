from pathlib import Path

import pytest

from osai.auto_settings import select_auto_settings
from osai.backends.llama_gradient import (
    LlamaGradientOptions,
    _gradient_command,
    _lower_auto_context,
    _memory_failure,
    _parse_best_checkpoint,
    _parse_epoch_losses,
    _parse_optimizer_steps,
    _parse_supervised_loss,
    _retryable_accelerator_failure,
    _validate_hybrid_training_path,
    _validate_memory_budget,
)
from osai.backends.llama_utils import (
    _combine_lora_adapters,
    _import_gguf,
    _write_adapter,
    gguf_adapter_training_shape,
)
from osai.cli import build_parser
from osai.config import ModelFormat
from osai.errors import ConfigurationError
from osai.formats import ModelInspection, QuantizationSpec
from osai.hardware import Accelerator, Engine


def test_gradient_options_reject_non_divisible_batch():
    with pytest.raises(ConfigurationError, match="divide"):
        LlamaGradientOptions(context=32, batch_size=6).validate()


def test_gradient_options_reject_unknown_projection():
    options = LlamaGradientOptions(target_modules=("unsafe.projection",))
    with pytest.raises(ConfigurationError, match="unsupported"):
        options.validate()


def test_low_memory_guard_rejects_wide_backward_graph():
    options = LlamaGradientOptions(target_modules=("self_attn.q_proj", "mlp.down_proj"))
    with pytest.raises(ConfigurationError, match="low-memory"):
        _validate_memory_budget(options, 8 * 1024**3)


def test_low_memory_guard_accepts_the_native_256_token_floor():
    _validate_memory_budget(
        LlamaGradientOptions(context=256, batch_size=8),
        8 * 1024**3,
    )


def test_qwen35_manual_graph_crossing_is_rejected_before_training():
    base = ModelInspection(
        format=ModelFormat.GGUF,
        path=Path("v2.gguf"),
        architecture="qwen35",
        quantization=QuantizationSpec("Q4_K_M"),
        size_bytes=2_500_000_000,
        shards=(Path("v2.gguf"),),
        block_count=28,
    )
    with pytest.raises(ConfigurationError, match="GATED_DELTA_NET"):
        _validate_hybrid_training_path(
            base,
            LlamaGradientOptions(num_layers=4, target_modules=("mlp.down_proj",)),
        )
    auto = select_auto_settings(base, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    _validate_hybrid_training_path(
        base,
        LlamaGradientOptions(num_layers=auto.num_layers, target_modules=auto.target_modules),
    )


def test_cpu_retry_requires_a_gpu_specific_failure(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("unsupported ggml op for backward pass: GATED_DELTA_NET")
    assert not _retryable_accelerator_failure(log)
    log.write_text("CUDA error: out of device memory")
    assert _retryable_accelerator_failure(log)
    log.write_text("[osai] exit=3221225477")
    assert _retryable_accelerator_failure(log)


def test_auto_context_retries_gpu_memory_errors_without_losing_labels(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("CUDA error: out of device memory")
    assert _memory_failure(log)
    with log.open("a") as handle:
        handle.write("\nunsupported ggml op for backward pass")
    assert not _memory_failure(log, len("CUDA error: out of device memory"))
    settings = LlamaGradientOptions(context=1024, batch_size=2, auto_settings=True)
    environment = {"OSAI_MAX_SEQ_LENGTH": "1024"}
    manifest = {"options": {"context": 1024, "batch_size": 2}}

    lowered = _lower_auto_context(settings, environment, manifest)

    assert lowered.context == 512
    assert lowered.batch_size == 2
    assert environment["OSAI_MAX_SEQ_LENGTH"] == "512"
    assert manifest["options"]["context"] == 512
    assert manifest["auto_context_retries"][0]["reason"]


def test_parallel_adapter_combination_preserves_weighted_lora_delta(tmp_path: Path):
    np = pytest.importorskip("numpy")
    first = {
        "blk.0.ffn_down.weight.lora_a": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "blk.0.ffn_down.weight.lora_b": np.array([[1, 2], [3, 4]], dtype=np.float32),
    }
    second = {
        "blk.0.ffn_down.weight.lora_a": np.array([[2, 1, 0], [0, 1, 2]], dtype=np.float32),
        "blk.0.ffn_down.weight.lora_b": np.array([[2, 3], [4, 5]], dtype=np.float32),
    }
    paths = (tmp_path / "one.gguf", tmp_path / "two.gguf")
    for path, tensors in zip(paths, (first, second), strict=True):
        _write_adapter(path, "llama", tensors, 4.0, np)
    assert gguf_adapter_training_shape(paths[0]) == (2, 1, ("mlp.down_proj",), 2.0)
    combined = tmp_path / "combined.gguf"
    _combine_lora_adapters(paths, (1, 3), combined, "llama", 4.0, 2, np)
    tensors = {str(t.name): np.asarray(t.data) for t in _import_gguf().GGUFReader(combined).tensors}
    actual = tensors["blk.0.ffn_down.weight.lora_b"] @ tensors["blk.0.ffn_down.weight.lora_a"]
    expected = 0.25 * (
        2 * first["blk.0.ffn_down.weight.lora_b"] @ first["blk.0.ffn_down.weight.lora_a"]
    ) + 0.75 * (2 * second["blk.0.ffn_down.weight.lora_b"] @ second["blk.0.ffn_down.weight.lora_a"])
    np.testing.assert_allclose(actual, expected)


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


@pytest.mark.parametrize("count", [2, 3, 4, 8])
@pytest.mark.parametrize(
    "accelerator,prefix",
    [
        (Accelerator.METAL, "MTL"),
        (Accelerator.VULKAN, "Vulkan"),
        (Accelerator.CUDA, "CUDA"),
    ],
)
def test_gradient_command_preserves_every_selected_gpu(count, accelerator, prefix):
    devices = tuple(f"{prefix}{index}" for index in range(count))
    options = LlamaGradientOptions(multi_gpu="on", devices=devices, split_mode="layer")
    command = _gradient_command(
        Path("llama-finetune"),
        Path("base.gguf"),
        Path("initial.gguf"),
        Path("train.txt"),
        Path("trained.gguf"),
        options,
        accelerator,
    )
    assert command[command.index("-dev") + 1] == ",".join(devices)
    assert command[command.index("-sm") + 1] == "layer"


def test_gradient_log_parsers():
    output = (
        "train: [done] data=0000064/0000064 loss=0.8\r\n"
        "I epoch=1 train_loss=0.795442714 train_loss_uncertainty=0.1\n"
    )
    assert _parse_epoch_losses(output, Path("train.log")) == (0.795442714,)
    assert _parse_optimizer_steps(output, 1) == 64
    supervised = "supervised optimizer step labels=4\nI checkpoint epoch=2 best_train_loss=0.25\n"
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
    args = build_parser().parse_args(["train", "--tier", "small", "--optimizer", "adamw"])
    assert args.optimizer == "adamw"


def test_cli_retains_legacy_gguf_optimizer_alias():
    args = build_parser().parse_args(["train", "--tier", "small", "--gguf-optimizer", "adamw"])
    assert args.gguf_optimizer == "adamw"


def test_cli_exposes_default_on_merge_controls():
    parser = build_parser()
    enabled = parser.parse_args(["train", "--tier", "small", "--merge"])
    disabled = parser.parse_args(["train", "--tier", "small", "--no-merge"])
    assert enabled.merge_model is True
    assert disabled.merge_model is False


def test_native_trainer_honors_the_selected_sequence_limit():
    source = (
        Path(__file__).parents[1]
        / "vendor"
        / "llama.cpp"
        / "examples"
        / "training"
        / "finetune.cpp"
    ).read_text(encoding="utf-8")
    assert 'std::getenv("OSAI_MAX_SEQ_LENGTH")' in source
    assert "without dropping assistant labels" in source
    assert "start = stop - 1" in source
    assert "weighted alignment requires each complete record" in source
    assert "trimmed prompt context" not in source
    assert "osai_tokenize_training_record" in source
    assert "Plain-text corpora have no prompt/answer boundary" in source


def test_cli_defaults_training_runs_to_sessions():
    args = build_parser().parse_args(["train", "--tier", "small"])
    assert args.sessions_root.name == "sessions"
    assert args.session_name is None
