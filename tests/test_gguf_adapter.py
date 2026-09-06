import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from osai.config import ModelFormat
from osai.formats import ModelInspection, QuantizationSpec
from osai.gguf_adapter import convert_mlx_adapter


def test_converts_mlx_matrix_orientation_and_scale(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    save_file(
        {
            "language_model.model.layers.31.self_attn.q_proj.lora_a": np.arange(
                8, dtype=np.float32
            ).reshape(4, 2),
            "language_model.model.layers.31.self_attn.q_proj.lora_b": np.arange(
                12, dtype=np.float32
            ).reshape(2, 6),
        },
        adapter / "adapters.safetensors",
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps({"lora_parameters": {"rank": 2, "scale": 4.0}})
    )
    base_path = tmp_path / "base.gguf"
    base_path.write_bytes(b"GGUF")
    inspection = ModelInspection(
        format=ModelFormat.GGUF,
        path=base_path,
        architecture="qwen35",
        quantization=QuantizationSpec("Q4_K_M"),
        size_bytes=4,
        shards=(base_path,),
        block_count=32,
    )
    result = convert_mlx_adapter(
        adapter, tmp_path / "adapter.gguf", base_gguf=inspection, dtype="f32"
    )
    assert result.lora_alpha == 8.0
    assert result.tensor_count == 2
    assert result.path.read_bytes()[:4] == b"GGUF"

    import gguf

    reader = gguf.GGUFReader(result.path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert set(tensors) == {"blk.31.attn_q.weight.lora_a", "blk.31.attn_q.weight.lora_b"}
    assert tuple(tensors["blk.31.attn_q.weight.lora_a"].shape) == (4, 2)
    assert tuple(tensors["blk.31.attn_q.weight.lora_b"].shape) == (2, 6)
