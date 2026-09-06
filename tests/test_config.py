import json
from pathlib import Path

import pytest

from osai.config import ModelFormat, TrainingConfig
from osai.errors import ConfigurationError


def test_config_paths_are_relative_to_config(tmp_path: Path):
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps({"model": "model", "data": "data", "output": "out", "format": "mlx"})
    )
    config = TrainingConfig.from_file(config_path)
    assert config.model == tmp_path / "model"
    assert config.data == tmp_path / "data"
    assert config.output == tmp_path / "out"
    assert config.effective_format is ModelFormat.MLX


def test_config_accepts_cli_session_and_data_paths(tmp_path: Path):
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps({"model": "model", "format": "mlx"}))
    config = TrainingConfig.from_file(
        config_path,
        data=tmp_path / "dataset",
        output=tmp_path / "sessions" / "run",
    )
    assert config.data == tmp_path / "dataset"
    assert config.output == tmp_path / "sessions" / "run"


def test_config_optimizer_defaults_to_auto(tmp_path: Path):
    config = TrainingConfig(
        model=tmp_path / "model",
        data=tmp_path / "data",
        output=tmp_path / "out",
    )
    assert config.optimizer == "auto"


@pytest.mark.parametrize(
    ("field", "value"),
    [("rank", 0), ("iterations", 0), ("dropout", 1.0), ("learning_rate", 0)],
)
def test_invalid_training_values_are_rejected(tmp_path: Path, field: str, value):
    values = {"model": tmp_path / "model", "data": tmp_path / "data", "output": tmp_path / "out"}
    values[field] = value
    with pytest.raises(ConfigurationError):
        TrainingConfig(**values)


def test_gguf_requires_mlx_companion(tmp_path: Path):
    config = TrainingConfig(
        model=tmp_path / "model.gguf",
        format=ModelFormat.GGUF,
        data=tmp_path / "data",
        output=tmp_path / "out",
    )
    with pytest.raises(ConfigurationError, match="companion_mlx"):
        _ = config.training_model
