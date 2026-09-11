"""Local multimodal model capability inspection.

Configuration token IDs alone are not proof that a checkpoint contains a usable
vision or audio tower.  osAi inspects the local processor files and indexed
weights before it allows media training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import DatasetSummary
from .errors import ConfigurationError, ModelFormatError

_VISION_WEIGHT = re.compile(
    r"(?:^|\.)(?:visual|vision_model|vision_tower|image_encoder)(?:\.|$)", re.I
)
_AUDIO_WEIGHT = re.compile(
    r"(?:^|\.)(?:audio|audio_tower|audio_encoder|speech_encoder)(?:\.|$)", re.I
)
_PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)

# These are the model routes implemented by the vendored MLX-VLM processor and
# trainer.  A vision tower alone is not proof that its processor accepts video.
_VIDEO_TRAINABLE = frozenset(
    {
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
        "gemma4",
    }
)
_AUDIO_TRAINABLE = frozenset(
    {"gemma4", "minicpmo", "nemotron_h_nano_omni", "phi4mm"}
)


@dataclass(frozen=True, slots=True)
class ModelModalities:
    image: bool
    video: bool
    audio: bool
    processor: bool
    vision_weights: bool
    audio_weights: bool
    reason: str | None = None

    @property
    def media(self) -> frozenset[str]:
        result: set[str] = set()
        if self.image:
            result.add("image")
        if self.video:
            result.add("video")
        if self.audio:
            result.add("audio")
        return frozenset(result)

    def as_dict(self) -> dict[str, bool | str | None]:
        return {
            "image": self.image,
            "video": self.video,
            "audio": self.audio,
            "processor": self.processor,
            "vision_weights": self.vision_weights,
            "audio_weights": self.audio_weights,
            "reason": self.reason,
        }


def inspect_mlx_modalities(model: str | Path) -> ModelModalities:
    """Inspect an MLX checkpoint without loading its tensor payloads."""

    root = Path(model).expanduser().resolve()
    config = _read_object(root / "config.json", required=True)
    keys = _weight_keys(root)
    processor = any(
        (root / name).is_file() and (root / name).stat().st_size > 0
        for name in _PROCESSOR_FILES
    )
    vision_weights = any(_VISION_WEIGHT.search(key) for key in keys)
    audio_weights = any(_AUDIO_WEIGHT.search(key) for key in keys)
    vision_config = any(
        config.get(key) is not None
        for key in ("vision_config", "visual_config", "vision_encoder")
    )
    audio_config = any(
        config.get(key) is not None
        for key in ("audio_config", "audio_encoder", "speech_config")
    )
    model_type = str(config.get("model_type", "")).casefold()
    image = processor and vision_weights and vision_config
    video = image and model_type in _VIDEO_TRAINABLE
    audio = (
        processor
        and audio_weights
        and audio_config
        and model_type in _AUDIO_TRAINABLE
    )
    reasons: list[str] = []
    if not processor:
        reasons.append("no local media processor configuration")
    if not vision_weights:
        reasons.append("no visual tower weights")
    if audio_config and not audio_weights:
        reasons.append("audio is configured but audio encoder weights are missing")
    if vision_weights and model_type not in _VIDEO_TRAINABLE:
        reasons.append(f"{model_type or 'unknown model type'} has no video training route")
    if audio_weights and model_type not in _AUDIO_TRAINABLE:
        reasons.append(f"{model_type or 'unknown model type'} has no audio training route")
    return ModelModalities(
        image=image,
        video=video,
        audio=audio,
        processor=processor,
        vision_weights=vision_weights,
        audio_weights=audio_weights,
        reason="; ".join(reasons) or None,
    )


def require_model_modalities(
    dataset: DatasetSummary,
    model: str | Path,
) -> ModelModalities:
    requested = frozenset(dataset.modalities) - {"text"}
    capabilities = inspect_mlx_modalities(model)
    missing = sorted(requested - capabilities.media)
    if missing:
        detail = f" ({capabilities.reason})" if capabilities.reason else ""
        raise ConfigurationError(
            "the dataset contains "
            + ", ".join(sorted(requested))
            + ", but this MLX checkpoint cannot train "
            + ", ".join(missing)
            + detail
            + "; use a complete local VLM checkpoint containing its media tower, "
            "processor files, and quantized language weights"
        )
    return capabilities


def _read_object(path: Path, *, required: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if required:
            raise ModelFormatError(f"cannot read multimodal model metadata {path}: {exc}") from exc
        return {}
    return value if isinstance(value, dict) else {}


def _weight_keys(root: Path) -> tuple[str, ...]:
    index = _read_object(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if isinstance(weight_map, dict) and weight_map:
        return tuple(str(key) for key in weight_map)
    keys: list[str] = []
    try:
        from safetensors import safe_open

        for shard in sorted(root.glob("*.safetensors")):
            with safe_open(shard, framework="numpy") as handle:
                keys.extend(handle.keys())
    except (ImportError, OSError, ValueError):
        return ()
    return tuple(keys)
