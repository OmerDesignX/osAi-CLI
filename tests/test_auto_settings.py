from pathlib import Path

import pytest

from osai.auto_settings import select_auto_settings
from osai.config import ModelFormat
from osai.errors import ConfigurationError
from osai.formats import ModelInspection, QuantizationSpec
from osai.hardware import Engine

GIB = 1024**3


def model(*, size_gib: float = 2.5, blocks: int = 32, context: int = 4096):
    return ModelInspection(
        format=ModelFormat.GGUF,
        path=Path("model.gguf"),
        architecture="qwen35",
        quantization=QuantizationSpec("Q4_K_M"),
        size_bytes=int(size_gib * GIB),
        shards=(Path("model.gguf"),),
        block_count=blocks,
        context_length=context,
    )


@pytest.mark.parametrize(
    ("memory_gib", "profile"),
    [(8, "compact"), (16, "balanced"), (32, "performance"), (64, "maximum")],
)
def test_auto_profile_scales_with_host_memory(memory_gib: int, profile: str):
    settings = select_auto_settings(
        model(),
        engine=Engine.MLX,
        memory_bytes=memory_gib * GIB,
        cpu_count=12,
    )
    assert settings.profile == profile
    assert settings.memory_budget_bytes == int(memory_gib * GIB * 0.75)


def test_low_memory_backprop_uses_the_proven_safe_shape():
    settings = select_auto_settings(
        model(),
        engine=Engine.LLAMA_CPP,
        memory_bytes=8 * GIB,
        cpu_count=8,
    )
    assert settings.max_seq_length == 64
    assert settings.gguf_batch_size == 8
    assert settings.num_layers == 1
    assert settings.rank == 1
    assert settings.target_modules == ("mlp.down_proj",)
    assert settings.gguf_threads == 2


def test_auto_profile_obeys_model_and_cpu_limits():
    settings = select_auto_settings(
        model(blocks=2, context=128),
        engine=Engine.MLX,
        memory_bytes=64 * GIB,
        cpu_count=3,
    )
    assert settings.max_seq_length == 128
    assert settings.num_layers == 2
    assert settings.gguf_threads == 3


def test_auto_profile_rejects_a_model_without_safe_headroom():
    with pytest.raises(ConfigurationError, match="55% RAM safety limit"):
        select_auto_settings(
            model(size_gib=5),
            engine=Engine.LLAMA_CPP,
            memory_bytes=8 * GIB,
        )


def test_auto_profile_requires_detectable_memory():
    with pytest.raises(ConfigurationError, match="could not detect physical RAM"):
        select_auto_settings(
            model(),
            engine=Engine.MLX,
            memory_bytes=0,
        )
