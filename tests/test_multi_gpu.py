from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("count", [3, 4, 8])
@pytest.mark.parametrize(
    "accelerator,prefix",
    [(Accelerator.CUDA, "CUDA"), (Accelerator.VULKAN, "Vulkan"), (Accelerator.METAL, "MTL")],
)
def test_all_native_gpus_reach_the_model_sharding_command(
    monkeypatch, tmp_path, count, accelerator, prefix
):
    expected = tuple(f"{prefix}{index}" for index in range(count))
    report = "\n".join(
        f"  {device}: Discrete GPU (8192 MiB, 6144 MiB free) [type=dedicated]"
        for device in expected
    )
    monkeypatch.setattr(
        "osai.multi_gpu.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=report, stderr=report),
    )
    devices = available_llama_devices(tmp_path / "llama-completion", accelerator)
    assert devices == expected
    command = llama_device_arguments(
        accelerator,
        _config(tmp_path, devices=devices, tensor_split=(1.0,) * count, main_gpu=count - 1),
    )
    assert command[command.index("-dev") + 1] == ",".join(expected)
    assert command[command.index("-ts") + 1] == ",".join("1" for _ in expected)
    assert command[command.index("-sm") + 1] == "layer"
    assert command[command.index("-mg") + 1] == str(count - 1)


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1.0, 1.0, 1.0, 1.0)])
def test_manual_split_cannot_silently_omit_or_add_gpu_weights(tmp_path, weights):
    settings = _config(tmp_path, devices=("CUDA0", "CUDA1", "CUDA2"), tensor_split=weights)
    with pytest.raises(ConfigurationError, match="one weight for every selected GPU"):
        llama_device_arguments(Accelerator.CUDA, settings)


def test_device_list_cannot_count_one_gpu_twice(tmp_path):
    settings = _config(tmp_path, devices=("CUDA0", "CUDA1", "CUDA1"))
    with pytest.raises(ConfigurationError, match="only once"):
        llama_device_arguments(Accelerator.CUDA, settings)


@pytest.mark.parametrize("index", [3, 4])
def test_main_gpu_is_an_index_into_the_selected_devices(tmp_path, index):
    settings = _config(tmp_path, devices=("CUDA0", "CUDA1", "CUDA2"), main_gpu=index)
    with pytest.raises(ConfigurationError, match="index the selected GPU list"):
        llama_device_arguments(Accelerator.CUDA, settings)


@pytest.mark.parametrize("weight", [float("nan"), float("inf")])
def test_nonfinite_gpu_split_is_rejected(tmp_path, weight):
    settings = _config(tmp_path, devices=("CUDA0", "CUDA1", "CUDA2"), tensor_split=(1, 1, weight))
    with pytest.raises(ConfigurationError, match="finite and positive"):
        llama_device_arguments(Accelerator.CUDA, settings)


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


def test_intel_mac_pro_uses_every_discrete_metal_gpu(monkeypatch, tmp_path: Path):
    class Completed:
        returncode = 0
        stdout = (
            "Available devices:\n"
            "  MTL0: AMD FirePro D700 (6144 MiB, 5120 MiB free)\n"
            "  MTL1: AMD FirePro D700 (6144 MiB, 5120 MiB free)\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    devices = available_llama_devices(tmp_path / "llama-completion", Accelerator.METAL)
    assert devices == ("MTL0", "MTL1")
    command = llama_device_arguments(
        Accelerator.METAL,
        _config(tmp_path, multi_gpu="auto", devices=devices, split_mode="layer"),
    )
    assert command[command.index("-dev") + 1] == "MTL0,MTL1"
    assert command[command.index("-sm") + 1] == "layer"


def test_apple_silicon_uses_its_single_unified_metal_device_without_splitting(
    monkeypatch, tmp_path: Path
):
    class Completed:
        returncode = 0
        stdout = "Available devices:\n  MTL0: Apple M1\n"

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    devices = available_llama_devices(tmp_path / "llama-completion", Accelerator.METAL)
    assert devices == ("MTL0",)
    command = llama_device_arguments(
        Accelerator.METAL,
        _config(tmp_path, multi_gpu="auto", devices=devices, split_mode="layer"),
    )
    assert command[command.index("-dev") + 1] == "MTL0"
    assert command[command.index("-sm") + 1] == "none"


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
            "  MTL2: AMD Radeon RX 5700 XT eGPU\n"
        )

    monkeypatch.setattr("osai.multi_gpu.subprocess.run", lambda *_args, **_kwargs: Completed())
    assert available_llama_devices(tmp_path / "llama-completion", Accelerator.METAL) == (
        "MTL1",
        "MTL2",
    )
    assert available_llama_devices(
        tmp_path / "llama-completion", Accelerator.METAL, include_integrated=True
    ) == ("MTL1", "MTL2")


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
