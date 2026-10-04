from pathlib import Path

import pytest

from osai.backends.mlx import _resolve_distributed_workers
from osai.config import ModelFormat, TrainingConfig
from osai.errors import ConfigurationError
from osai.hardware import Accelerator
from osai.multi_gpu import (
    available_llama_devices,
    llama_device_arguments,
    llama_device_free_bytes,
)


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


def test_vulkan_auto_prefers_nvidia_cards_over_integrated_adapter(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = (
            "Available devices:\n"
            "  Vulkan0: AMD Radeon(TM) Graphics (32700 MiB) [type=integrated]\n"
            "  Vulkan1: NVIDIA GeForce RTX 3060 (12324 MiB) [type=dedicated]\n"
            "  Vulkan2: NVIDIA GeForce RTX 3060 (12329 MiB) [type=dedicated]\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert available_llama_devices(tmp_path / "llama-completion", Accelerator.VULKAN) == (
        "Vulkan1",
        "Vulkan2",
    )


def test_vulkan_auto_keeps_discrete_cards_from_multiple_vendors(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = (
            "Available devices:\n"
            "  Vulkan0: AMD Radeon(TM) Graphics (32700 MiB) [type=integrated]\n"
            "  Vulkan1: AMD Radeon RX 7900 XTX (24576 MiB) [type=dedicated]\n"
            "  Vulkan2: NVIDIA GeForce RTX 3060 (12000 MiB) [type=dedicated]\n"
            "  Vulkan3: Intel Arc A770 Graphics (16384 MiB) [type=dedicated]\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert available_llama_devices(tmp_path / "llama-completion", Accelerator.VULKAN) == (
        "Vulkan1",
        "Vulkan2",
        "Vulkan3",
    )


def test_cuda_auto_discovers_both_native_devices(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = (
            "Available devices:\n"
            "  CUDA0: NVIDIA GeForce RTX 3060 (12000 MiB)\n"
            "  CUDA1: NVIDIA GeForce RTX 3060 (12000 MiB)\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert available_llama_devices(tmp_path / "llama-completion", Accelerator.CUDA) == (
        "CUDA0",
        "CUDA1",
    )


def test_metal_prefers_external_gpu_over_integrated_card(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = (
            "Available devices:\n"
            "  MTL0: Intel Iris Plus Graphics\n"
            "  MTL1: AMD Radeon RX 6800 XT eGPU\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert available_llama_devices(tmp_path / "llama-completion", Accelerator.METAL) == ("MTL1",)
    assert available_llama_devices(
        tmp_path / "llama-completion", Accelerator.METAL, include_integrated=True
    ) == ("MTL1",)


def test_vulkan_required_multi_gpu_never_adds_integrated_card(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = (
            "Available devices:\n"
            "  Vulkan0: AMD Radeon(TM) Graphics (32700 MiB) [type=integrated]\n"
            "  Vulkan1: NVIDIA GeForce RTX 3060 (12000 MiB) [type=dedicated]\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert available_llama_devices(
        tmp_path / "llama-completion", Accelerator.VULKAN, include_integrated=True
    ) == ("Vulkan1",)


def test_vulkan_rejects_integrated_only_inventory(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = (
            "Available devices:\n  Vulkan0: AMD Radeon(TM) Graphics (32700 MiB) [type=integrated]\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert available_llama_devices(tmp_path / "llama-completion", Accelerator.VULKAN) == ()


def test_vulkan_requires_authoritative_device_type(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = "Available devices:\n  Vulkan0: AMD Radeon 780M Graphics (32700 MiB)\n"

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    with pytest.raises(ConfigurationError, match="cannot identify dedicated GPUs"):
        available_llama_devices(tmp_path / "llama-completion", Accelerator.VULKAN)


def test_native_device_inventory_reports_free_gpu_memory(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = "Available devices:\n  MTL1: AMD Radeon eGPU (8192 MiB, 6144 MiB free)\n"

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert llama_device_free_bytes(tmp_path / "llama-completion", Accelerator.METAL) == {
        "MTL1": 6144 * 1024**2
    }


def test_typed_vulkan_inventory_retains_free_memory(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = (
            "Available devices:\n"
            "  Vulkan0: NVIDIA GeForce RTX 3060 (12287 MiB, 8192 MiB free) [type=dedicated]\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert llama_device_free_bytes(tmp_path / "llama-completion", Accelerator.VULKAN) == {
        "Vulkan0": 8192 * 1024**2
    }


def test_mlx_auto_uses_divisible_cuda_worker_count(tmp_path: Path):
    settings = _config(tmp_path, multi_gpu="auto", distributed_workers=0)
    assert _resolve_distributed_workers(settings, {"accelerator": "cuda", "gpu_count": 3}) == 2


def test_mlx_rejects_required_multi_gpu_on_metal(tmp_path: Path):
    settings = _config(tmp_path, multi_gpu="on")
    with pytest.raises(ConfigurationError, match="Metal"):
        _resolve_distributed_workers(settings, {"accelerator": "metal", "gpu_count": 1})
