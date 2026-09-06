import pytest

from osai.backends.mlx import MlxBackend
from osai.errors import ConfigurationError, DependencyError
from osai.hardware import (
    Accelerator,
    Engine,
    HardwareReport,
    detect_hardware,
    macos_version_at_least,
    select_engine,
    select_llama_accelerator,
)


def report(*, engine="mlx", accelerator="metal"):
    return HardwareReport(
        os="Darwin",
        architecture="arm64",
        metal=accelerator == "metal",
        mps=False,
        cuda=accelerator == "cuda",
        vulkan=accelerator == "vulkan",
        cpu=True,
        recommended_engine=engine,
        recommended_llama_accelerator=accelerator,
    )


def test_auto_engine_prefers_mlx_when_reported():
    assert select_engine("auto", report()) is Engine.MLX


def test_auto_engine_uses_llama_elsewhere():
    assert select_engine("auto", report(engine="llama.cpp", accelerator="cpu")) is Engine.LLAMA_CPP


def test_llama_gpu_priority_result_is_used():
    assert select_llama_accelerator("auto", report(accelerator="cuda")) is Accelerator.CUDA


def test_mps_is_not_mislabeled_as_a_llama_backend():
    with pytest.raises(ConfigurationError, match="PyTorch API"):
        select_llama_accelerator("mps", report())


def test_mlx_accepts_linux_accelerators_but_not_vulkan():
    assert MlxBackend(accelerator="cuda").accelerator is Accelerator.CUDA
    assert MlxBackend(accelerator="cpu").accelerator is Accelerator.CPU
    with pytest.raises(ConfigurationError, match="MPS or Vulkan"):
        MlxBackend(accelerator="vulkan")


def test_mlx_macos_version_floor(monkeypatch):
    monkeypatch.setattr("osai.hardware.platform.system", lambda: "Darwin")
    monkeypatch.setattr("osai.hardware.platform.mac_ver", lambda: ("13.6.9", (), ""))
    monkeypatch.setattr("osai.backends.mlx.platform.machine", lambda: "arm64")
    assert macos_version_at_least(14) is False
    with pytest.raises(DependencyError, match="requires macOS 14"):
        MlxBackend().preflight()


def test_intel_macos_detects_llama_metal_without_selecting_mlx(monkeypatch):
    monkeypatch.setattr("osai.hardware.platform.system", lambda: "Darwin")
    monkeypatch.setattr("osai.hardware.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("osai.hardware.platform.mac_ver", lambda: ("13.6.9", (), ""))
    monkeypatch.setattr("osai.hardware._mps_available", lambda: False)
    monkeypatch.setattr("osai.hardware._module_importable", lambda _name: False)
    report = detect_hardware()
    assert report.metal is True
    assert report.recommended_engine == "llama.cpp"
    assert report.recommended_llama_accelerator == "metal"
