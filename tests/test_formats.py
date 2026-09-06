import json
from pathlib import Path

import pytest

import osai.formats as formats
from osai.errors import ModelFormatError
from osai.formats import discover_gguf_shards, inspect_gguf, inspect_mlx


def test_discovers_all_gguf_shards(tmp_path: Path):
    first = tmp_path / "model-Q4_K_M-00001-of-00002.gguf"
    second = tmp_path / "model-Q4_K_M-00002-of-00002.gguf"
    first.write_bytes(b"GGUF")
    second.write_bytes(b"GGUF")
    assert discover_gguf_shards(first) == (first, second)


def test_missing_gguf_shard_is_rejected(tmp_path: Path):
    first = tmp_path / "model-Q4_K_M-00001-of-00002.gguf"
    first.write_bytes(b"GGUF")
    with pytest.raises(ModelFormatError, match="missing GGUF shard"):
        discover_gguf_shards(first)


def test_lfs_mlx_shard_is_rejected(tmp_path: Path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_text("version https://git-lfs.github.com/spec/v1\n")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen", "quantization": {"bits": 4, "group_size": 64}})
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard.name}})
    )
    with pytest.raises(ModelFormatError, match="Git LFS pointer"):
        inspect_mlx(tmp_path)


def test_full_precision_mlx_is_rejected(tmp_path: Path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"materialized")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen"}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard.name}})
    )
    with pytest.raises(ModelFormatError, match="not weight-quantized"):
        inspect_mlx(tmp_path)


def test_high_bit_mlx_is_rejected_as_full_precision(tmp_path: Path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"materialized")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen", "quantization": {"bits": 16}})
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard.name}})
    )
    with pytest.raises(ModelFormatError, match="not weight-quantized"):
        inspect_mlx(tmp_path)


def test_spoofed_quantized_gguf_name_cannot_hide_f16_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Field:
        def __init__(self, value):
            self.value = value

        def contents(self):
            return self.value

    class Tensor:
        shape = (16, 16)
        tensor_type = 1

    class Reader:
        tensors = [Tensor()]

        def get_field(self, key):
            values = {
                "general.type": "model",
                "general.file_type": 1,
                "general.architecture": "qwen35",
            }
            return Field(values[key]) if key in values else None

    model = tmp_path / "spoofed-Q4_K_M.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(formats, "_gguf_reader", lambda _path: Reader())

    with pytest.raises(ModelFormatError, match="full precision"):
        inspect_gguf(model)


def test_mlx_context_is_reported(tmp_path: Path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"materialized")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization": {"bits": 5, "group_size": 64},
                "text_config": {
                    "hidden_size": 2560,
                    "num_hidden_layers": 32,
                    "max_position_embeddings": 262144,
                },
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard.name}})
    )

    inspection = inspect_mlx(tmp_path)

    assert inspection.context_length == 262144
