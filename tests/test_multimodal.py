import json
from pathlib import Path

import pytest

from osai.dataset import DatasetSummary
from osai.errors import ConfigurationError
from osai.multimodal import inspect_mlx_modalities, require_model_modalities


def _model(tmp_path: Path, keys: list[str], config: dict) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    (model / "processor_config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in keys}})
    )
    return model


def test_reserved_image_tokens_do_not_claim_missing_visual_weights(tmp_path: Path):
    model = _model(
        tmp_path,
        ["language_model.model.layers.0.mlp.down_proj.weight"],
        {
            "model_type": "qwen3_5",
            "vision_config": {},
            "image_token_id": 10,
            "video_token_id": 11,
        },
    )
    capabilities = inspect_mlx_modalities(model)
    assert not capabilities.image
    assert not capabilities.video
    with pytest.raises(ConfigurationError, match="no visual tower weights"):
        require_model_modalities(
            DatasetSummary(Path("data"), "chat", 1, 0, 0, modalities=("text", "image")),
            model,
        )


def test_complete_local_vlm_detects_vision_video_and_audio(tmp_path: Path):
    model = _model(
        tmp_path,
        [
            "language_model.model.layers.0.mlp.down_proj.weight",
            "vision_tower.layers.0.weight",
            "audio_encoder.layers.0.weight",
        ],
        {"model_type": "gemma4", "vision_config": {}, "audio_config": {}},
    )
    capabilities = inspect_mlx_modalities(model)
    assert capabilities.media == {"image", "video", "audio"}


def test_image_checkpoint_does_not_claim_unimplemented_video_route(tmp_path: Path):
    model = _model(
        tmp_path,
        ["vision_tower.layers.0.weight"],
        {"model_type": "idefics3", "vision_config": {}},
    )
    capabilities = inspect_mlx_modalities(model)
    assert capabilities.image
    assert not capabilities.video
    with pytest.raises(ConfigurationError, match="cannot train video"):
        require_model_modalities(
            DatasetSummary(Path("data"), "chat", 1, 0, 0, modalities=("text", "video")),
            model,
        )
