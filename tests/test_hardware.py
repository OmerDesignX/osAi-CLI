import os
from types import SimpleNamespace

import pytest

import osai.hardware as hardware
from osai.backends import llama_cpp
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
from osai.paths import llama_binary


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


def test_cpu_build_does_not_claim_gpu_training():
    detected = report(accelerator="cuda")
    detected = detected.__class__(**{**detected.as_dict(), "compiled_llama_accelerator": "cpu"})
    with pytest.raises(ConfigurationError, match="built for cpu"):
        select_llama_accelerator("cuda", detected)
    assert select_llama_accelerator("cuda", detected, for_build=True) is Accelerator.CUDA


def test_cuda_build_targets_every_installed_gpu_architecture(monkeypatch):
    monkeypatch.setattr(
        llama_cpp.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(
        llama_cpp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="8.6\n8.9\n8.6\n"),
    )
    assert llama_cpp._cuda_architectures() == ("86", "89")


def test_combined_cuda_vulkan_build_accepts_both_backends():
    detected = report(accelerator="cuda")
    detected = detected.__class__(
        **{**detected.as_dict(), "vulkan": True, "compiled_llama_accelerator": "cuda+vulkan"}
    )
    assert select_llama_accelerator("cuda", detected) is Accelerator.CUDA
    assert select_llama_accelerator("vulkan", detected) is Accelerator.VULKAN


def test_linux_vulkan_build_accepts_system_glslc_without_sdk_variable(monkeypatch):
    monkeypatch.setattr(hardware, "os", SimpleNamespace(name="posix", environ={}))
    monkeypatch.setattr(
        hardware.shutil, "which", lambda name: "/usr/bin/glslc" if name == "glslc" else None
    )
    assert hardware._vulkan_build_available()


def test_writable_native_cache_takes_precedence_only_when_complete(monkeypatch, tmp_path):
    packaged = tmp_path / "source" / "vendor" / "llama.cpp" / "build" / "bin"
    cached = tmp_path / "cache" / "bin"
    packaged.mkdir(parents=True)
    cached.mkdir(parents=True)
    suffix = ".exe" if os.name == "nt" else ""
    name = f"llama-finetune{suffix}"
    (packaged / name).write_bytes(b"package")
    (cached / name).write_bytes(b"cached")
    monkeypatch.setattr("osai.paths.project_root", lambda: tmp_path / "source")
    monkeypatch.setattr("osai.paths.llama_runtime_build", lambda: tmp_path / "cache")
    assert llama_binary("llama-finetune") == packaged / name
    (tmp_path / "cache" / "OSAI_BUILD.json").write_text('{"llamaAccelerator":"cuda"}')
    assert llama_binary("llama-finetune") == cached / name


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
