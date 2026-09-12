from pathlib import Path

from osai.backends.mlx import resolve_mlx_training_steps
from osai.config import ModelFormat, TrainingConfig
from osai.mlx_safety import truncate_completion_aware


def config(tmp_path: Path, **values) -> TrainingConfig:
    return TrainingConfig(
        model=tmp_path / "model",
        format=ModelFormat.MLX,
        data=tmp_path / "data",
        output=tmp_path / "output",
        **values,
    )


def test_one_epoch_covers_every_training_example(tmp_path: Path):
    run = config(tmp_path, iterations=1, batch_size=1)
    assert resolve_mlx_training_steps(run, 15_011) == 15_011


def test_epochs_use_complete_batches_and_accumulation_windows(tmp_path: Path):
    run = config(
        tmp_path,
        iterations=3,
        batch_size=2,
        grad_accumulation_steps=4,
    )
    # ceil(5 / 2) batches * 3 epochs = 9, rounded to a complete accumulation window.
    assert resolve_mlx_training_steps(run, 5) == 12


def test_completion_aware_truncation_retains_short_answer():
    fitted, offset = truncate_completion_aware(list(range(200)), 180, 64)
    assert fitted == list(range(136, 200))
    assert offset == 44
    assert len(fitted) - offset == 20


def test_completion_aware_truncation_balances_long_prompt_and_answer():
    fitted, offset = truncate_completion_aware(list(range(200)), 120, 64)
    assert fitted == list(range(88, 152))
    assert offset == 32
    assert len(fitted) - offset == 32


def test_unmasked_text_keeps_its_leading_context():
    fitted, offset = truncate_completion_aware(list(range(200)), 0, 64)
    assert fitted == list(range(64))
    assert offset == 0
