from pathlib import Path

import pytest

from osai.backends.mlx import _resolve_distributed_workers
from osai.config import ModelFormat, TrainingConfig
from osai.errors import ConfigurationError
from osai.hardware import Accelerator
from osai.multi_gpu import llama_device_arguments


def _config(tmp_path: Path, **changes) -> TrainingConfig:
    values = {
        "model": tmp_path / "model",
        "format": ModelFormat.MLX,
        "data": tmp_path / "data",
        "output": tmp_path / "out",
        "batch_size": 4,
        **changes,
    }
    return TrainingConfig(**values)


def test_llama_multi_gpu_maps_native_split_flags(tmp_path: Path):
    settings = _config(
        tmp_path,
        multi_gpu="on",
        devices=("CUDA0", "CUDA1"),
        split_mode="tensor",
        tensor_split=(3.0, 1.0),
        main_gpu=1,
    )
    command = llama_device_arguments(Accelerator.CUDA, settings)
    assert command[command.index("-dev") + 1] == "CUDA0,CUDA1"
    assert command[command.index("-sm") + 1] == "tensor"
    assert command[command.index("-ts") + 1] == "3,1"
    assert command[command.index("-mg") + 1] == "1"


def test_llama_metal_multi_gpu_uses_explicit_physical_devices(tmp_path: Path):
    settings = _config(
        tmp_path,
        multi_gpu="on",
        devices=("MTL0", "MTL1"),
        split_mode="layer",
    )
    command = llama_device_arguments(Accelerator.METAL, settings)
    assert command[command.index("-dev") + 1] == "MTL0,MTL1"
    assert command[command.index("-sm") + 1] == "layer"


def test_llama_single_gpu_disables_splitting(tmp_path: Path):
    settings = _config(tmp_path, multi_gpu="off", devices=("Vulkan0", "Vulkan1"))
    command = llama_device_arguments(Accelerator.VULKAN, settings)
    assert command[command.index("-dev") + 1] == "Vulkan0"
    assert command[command.index("-sm") + 1] == "none"


def test_mlx_auto_uses_divisible_cuda_worker_count(tmp_path: Path):
    settings = _config(tmp_path, multi_gpu="auto", distributed_workers=0)
    assert _resolve_distributed_workers(
        settings, {"accelerator": "cuda", "gpu_count": 3}
    ) == 2


def test_mlx_rejects_required_multi_gpu_on_metal(tmp_path: Path):
    settings = _config(tmp_path, multi_gpu="on")
    with pytest.raises(ConfigurationError, match="Metal"):
        _resolve_distributed_workers(
            settings, {"accelerator": "metal", "gpu_count": 1}
        )
