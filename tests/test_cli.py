from pathlib import Path
from types import SimpleNamespace

from osai.cli import (
    _choose_combined_alignment,
    _resolve_training_settings,
    _select_optimizer,
    build_parser,
)
from osai.config import ModelFormat, TrainingConfig
from osai.formats import ModelInspection, QuantizationSpec
from osai.hardware import Engine


def test_train_auto_settings_and_manual_overrides_parse():
    args = build_parser().parse_args(
        [
            "train",
            "--tier",
            "small",
            "--data",
            "data",
            "--auto-settings",
            "--batch-size",
            "2",
            "--rank",
            "6",
            "--dropout",
            "0.1",
            "--seed",
            "7",
            "--gradient-accumulation-steps",
            "4",
            "--no-gradient-checkpointing",
            "--save-every",
            "5",
            "--steps-per-report",
            "2",
            "--steps-per-eval",
            "3",
            "--val-batches",
            "2",
            "--no-mask-prompt",
        ]
    )
    assert args.auto_settings is True
    assert args.batch_size == 2
    assert args.rank == 6
    assert args.dropout == 0.1
    assert args.seed == 7
    assert args.grad_accumulation_steps == 4
    assert args.grad_checkpoint is False
    assert args.save_every == 5
    assert args.steps_per_report == 2
    assert args.steps_per_eval == 3
    assert args.val_batches == 2
    assert args.mask_prompt is False


def test_manual_training_defaults_are_resolved_after_parsing():
    args = build_parser().parse_args(
        ["train", "--tier", "small", "--data", "data"]
    )
    assert args.auto_settings is False
    assert args.batch_size is None
    assert args.rank is None
    assert args.strict_base_hash is None
    assert args.stage == "fine-tuning"
    assert args.alignment_type is None
    assert args.download_model is True
    assert args.live_rollouts is True
    assert args.rollouts_per_prompt == 2


def test_fine_tune_epochs_and_legacy_iterations_share_one_value():
    parser = build_parser()
    epochs = parser.parse_args(
        ["train", "--tier", "small", "--data", "data", "--epochs", "3"]
    )
    legacy = parser.parse_args(
        ["train", "--tier", "small", "--data", "data", "--iterations", "2"]
    )
    assert epochs.iterations == 3
    assert legacy.iterations == 2


def test_official_model_download_can_be_disabled():
    args = build_parser().parse_args(
        ["select", "--tier", "small", "--no-download-model"]
    )
    assert args.download_model is False


def test_auto_optimizer_is_backend_aware(tmp_path: Path):
    args = build_parser().parse_args(["train", "--tier", "small", "--data", "data"])
    config = TrainingConfig(
        model=tmp_path / "model",
        format=ModelFormat.MLX,
        data=tmp_path / "data",
        output=tmp_path / "out",
    )
    assert _select_optimizer(args, config, Engine.MLX) == "adamw"
    assert _select_optimizer(args, config, Engine.LLAMA_CPP) == "sgd"


def test_manual_optimizer_overrides_backend_default(tmp_path: Path):
    args = build_parser().parse_args(
        ["train", "--tier", "small", "--data", "data", "--optimizer", "sgd"]
    )
    config = TrainingConfig(
        model=tmp_path / "model",
        format=ModelFormat.MLX,
        data=tmp_path / "data",
        output=tmp_path / "out",
    )
    assert _select_optimizer(args, config, Engine.MLX) == "sgd"


def test_alignment_pipeline_options_parse():
    args = build_parser().parse_args(
        [
            "train",
            "--tier",
            "small",
            "--stage",
            "fine-tune-align",
            "--data",
            "fine",
            "--alignment-data",
            "preferences",
            "--alignment-type",
            "ipo",
            "--multi-gpu",
            "on",
            "--device",
            "CUDA0",
            "--device",
            "CUDA1",
            "--no-live-rollouts",
            "--rollouts-per-prompt",
            "4",
        ]
    )
    assert args.stage == "fine-tune-align"
    assert args.alignment_type == "ipo"
    assert args.devices == ["CUDA0", "CUDA1"]
    assert args.live_rollouts is False
    assert args.rollouts_per_prompt == 4


def test_cli_accepts_reinforce_rloo_and_grpo():
    parser = build_parser()
    for method in ("reinforce", "rloo", "grpo"):
        args = parser.parse_args(
            ["train", "--tier", "small", "--alignment-type", method]
        )
        assert args.alignment_type == method


def test_combined_preference_run_offers_orpo_before_training(
    monkeypatch, tmp_path: Path
):
    args = _combined_args(tmp_path)
    monkeypatch.setattr(
        "osai.cli.sys.stdin", SimpleNamespace(isatty=lambda: True)
    )
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    _choose_combined_alignment(args)
    assert args.alignment_type == "orpo"


def test_combined_preference_run_uses_dpo_when_orpo_is_declined(
    monkeypatch, tmp_path: Path
):
    args = _combined_args(tmp_path)
    monkeypatch.setattr(
        "osai.cli.sys.stdin", SimpleNamespace(isatty=lambda: True)
    )
    answers = iter(["maybe", "no"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    _choose_combined_alignment(args)
    assert args.alignment_type == "dpo"


def test_explicit_alignment_type_skips_orpo_prompt(monkeypatch, tmp_path: Path):
    args = _combined_args(tmp_path, alignment_type="simpo")
    monkeypatch.setattr(
        "builtins.input", lambda _: (_ for _ in ()).throw(AssertionError("prompted"))
    )
    _choose_combined_alignment(args)
    assert args.alignment_type == "simpo"


def test_noninteractive_combined_preference_run_defaults_to_dpo(
    monkeypatch, tmp_path: Path, capsys
):
    args = _combined_args(tmp_path)
    monkeypatch.setattr(
        "osai.cli.sys.stdin", SimpleNamespace(isatty=lambda: False)
    )
    _choose_combined_alignment(args)
    assert args.alignment_type == "dpo"
    assert "non-interactive" in capsys.readouterr().err


def _combined_args(tmp_path: Path, alignment_type: str | None = None):
    preference = tmp_path / "preference"
    preference.mkdir()
    (preference / "train.jsonl").write_text(
        '{"prompt":"p","chosen":"yes","rejected":"no"}\n', encoding="utf-8"
    )
    command = [
        "train",
        "--tier",
        "small",
        "--stage",
        "fine-tune-align",
        "--data",
        str(tmp_path / "fine"),
        "--alignment-data",
        str(preference),
    ]
    if alignment_type is not None:
        command.extend(["--alignment-type", alignment_type])
    return build_parser().parse_args(command)


def test_manual_values_override_the_auto_profile(monkeypatch, tmp_path: Path):
    args = build_parser().parse_args(
        [
            "train",
            "--tier",
            "small",
            "--data",
            str(tmp_path),
            "--auto-settings",
            "--batch-size",
            "2",
            "--rank",
            "6",
            "--dropout",
            "0.15",
            "--gradient-accumulation-steps",
            "3",
            "--no-gradient-checkpointing",
            "--no-mask-prompt",
        ]
    )
    inspection = ModelInspection(
        format=ModelFormat.GGUF,
        path=tmp_path / "model.gguf",
        architecture="qwen35",
        quantization=QuantizationSpec("Q4_K_M"),
        size_bytes=2 * 1024**3,
        shards=(tmp_path / "model.gguf",),
        block_count=32,
        context_length=4096,
    )
    monkeypatch.setattr("osai.cli.inspect_model", lambda *_: inspection)
    monkeypatch.setattr(
        "osai.auto_settings.physical_memory_bytes", lambda: 8 * 1024**3
    )
    config = TrainingConfig(
        model=inspection.path,
        format=ModelFormat.GGUF,
        data=tmp_path,
        output=tmp_path / "out",
    )
    resolved = _resolve_training_settings(config, args, Engine.LLAMA_CPP)
    assert resolved.auto_profile == "compact"
    assert resolved.batch_size == 2
    assert resolved.rank == 6
    assert resolved.max_seq_length == 64
    assert resolved.dropout == 0.15
    assert resolved.grad_accumulation_steps == 3
    assert resolved.grad_checkpoint is False
    assert resolved.mask_prompt is False
