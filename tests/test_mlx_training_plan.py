import json
from pathlib import Path
from types import SimpleNamespace

import osai.backends.mlx as mlx_backend
from osai.backends.mlx import MlxBackend, _lower_mlx_memory_settings, resolve_mlx_training_steps
from osai.config import ModelFormat, TrainingConfig
from osai.errors import TrainingError
from osai.mlx_safety import WindowedDataset, truncate_completion_aware
from osai.process import ProcessResult


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


def test_windowed_mlx_data_covers_each_supervised_token_once():
    class Rows:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return (list(range(200)), 120) if index == 0 else (list(range(90)), 0)

    windows = WindowedDataset(Rows(), 32)
    covered = [[], []]
    for index in range(len(windows)):
        row, start, _, _ = windows.windows[index]
        sequence, offset = windows[index]
        assert len(sequence) <= 32
        covered[row].extend(sequence[max(1, offset) :])
        assert windows.itemlen(index) == len(sequence)
    assert covered[0] == list(range(120, 200))
    assert covered[1] == list(range(1, 90))


def test_mlx_memory_retry_reduces_context_then_batch_with_worker_divisibility():
    assert _lower_mlx_memory_settings(1024, 4, 2) == (512, 4)
    assert _lower_mlx_memory_settings(64, 4, 2) == (64, 2)
    assert _lower_mlx_memory_settings(64, 2, 2) is None


def test_mlx_auto_retries_oom_and_publishes_only_successful_adapter(tmp_path, monkeypatch):
    run = config(
        tmp_path,
        auto_settings=True,
        max_seq_length=1024,
        batch_size=1,
    )
    monkeypatch.setattr(
        MlxBackend, "preflight", lambda self: {"accelerator": "metal", "gpu_count": 1}
    )
    monkeypatch.setattr(mlx_backend, "prepare_mlx_dataset", lambda *args: tmp_path / "data")
    monkeypatch.setattr(
        mlx_backend,
        "validate_dataset",
        lambda *args: SimpleNamespace(train_examples=2, schema="chat"),
    )
    contexts = []

    def fake_run_logged(command, *, log_path, env):
        settings = json.loads((tmp_path / "output/.internal/mlx_lora_config.yaml").read_text())
        contexts.append(settings["max_seq_length"])
        with log_path.open("a", encoding="utf-8") as log:
            if len(contexts) == 1:
                log.write("Metal out of memory\n")
                raise TrainingError("command exited with status 1")
            log.write("osai: training plan examples=2 windows=6 epochs=1 batch=1 steps=6\n")
            log.write("Train loss 1.25\n")
        adapter_dir = Path(settings["adapter_path"])
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapters.safetensors").write_bytes(b"adapter")
        (adapter_dir / "adapter_config.json").write_text(json.dumps(settings))
        return ProcessResult(tuple(command), 0, 1.0, log_path)

    monkeypatch.setattr(mlx_backend, "run_logged", fake_run_logged)
    result = MlxBackend().train(run)
    assert contexts == [1024, 512]
    assert result.adapter_file.is_file()
    assert result.training_steps == 6
    assert json.loads((result.adapter_dir / "adapter_config.json").read_text())[
        "adapter_path"
    ] == str(result.adapter_dir.resolve())
