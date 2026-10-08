import os
from types import SimpleNamespace

import pytest

import osai.hardware as hardware
from osai.backends import llama_cpp
from osai.backends.mlx import MlxBackend
from osai.errors import ConfigurationError, DependencyError, TrainingError
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


def test_auto_runtime_does_not_silently_train_on_cpu_with_nvidia_gpu(monkeypatch):
    detected = report(accelerator="cuda")
    detected = detected.__class__(**{**detected.as_dict(), "compiled_llama_accelerator": "cpu"})
    monkeypatch.setattr(llama_cpp, "detect_hardware", lambda: detected)
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda _name: None)
    monkeypatch.setattr(llama_cpp, "_vulkan_build_available", lambda: False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    with pytest.raises(DependencyError, match="GPU was detected"):
        llama_cpp.ensure_runtime_accelerator("auto")


def test_auto_runtime_needs_cmake_when_only_cpu_build_is_packaged(monkeypatch):
    detected = report(accelerator="cuda")
    detected = detected.__class__(**{**detected.as_dict(), "compiled_llama_accelerator": "cpu"})
    monkeypatch.setattr(llama_cpp, "detect_hardware", lambda: detected)
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda name: "nvcc" if name == "nvcc" else None)
    monkeypatch.setattr(llama_cpp, "_cmake_executable", lambda: None)
    with pytest.raises(DependencyError, match="CMake is required"):
        llama_cpp.ensure_runtime_accelerator("auto")


def test_runtime_rebuilds_cuda_for_dedicated_amd_vulkan(monkeypatch, tmp_path):
    detected = report(accelerator="cuda")
    detected = detected.__class__(
        **{**detected.as_dict(), "vulkan": True, "compiled_llama_accelerator": "cuda"}
    )
    builds = []
    monkeypatch.setattr(llama_cpp, "detect_hardware", lambda: detected)
    monkeypatch.setattr(llama_cpp, "_vulkan_build_available", lambda: True)
    monkeypatch.setattr(llama_cpp, "dedicated_amd_vulkan_available", lambda: True)
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda name: name)
    monkeypatch.setattr(llama_cpp, "_cmake_executable", lambda: "cmake")
    monkeypatch.setattr(llama_cpp, "llama_runtime_build", lambda: tmp_path)
    monkeypatch.setattr(llama_cpp, "build_llama_cpp", lambda **kwargs: builds.append(kwargs))
    llama_cpp.ensure_runtime_accelerator("auto")
    assert len(builds) == 1
    assert builds[0]["also_vulkan"] is True
    assert builds[0]["require_vulkan"] is True


def test_runtime_keeps_cuda_build_for_integrated_amd(monkeypatch):
    detected = report(accelerator="cuda")
    detected = detected.__class__(
        **{**detected.as_dict(), "vulkan": True, "compiled_llama_accelerator": "cuda"}
    )
    monkeypatch.setattr(llama_cpp, "detect_hardware", lambda: detected)
    monkeypatch.setattr(llama_cpp, "_vulkan_build_available", lambda: True)
    monkeypatch.setattr(llama_cpp, "dedicated_amd_vulkan_available", lambda: False)
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda name: "nvcc" if name == "nvcc" else None)
    monkeypatch.setattr(
        llama_cpp, "build_llama_cpp", lambda **_kwargs: pytest.fail("unexpected rebuild")
    )
    llama_cpp.ensure_runtime_accelerator("auto")


def test_dedicated_amd_probe_excludes_integrated_card(monkeypatch):
    output = """GPU0:
        vendorID = 0x1002
        deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    GPU1:
        vendorID = 0x10de
        deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    """
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: "vulkaninfo")
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output),
    )
    assert hardware.dedicated_amd_vulkan_available() is False


def test_dedicated_amd_probe_finds_discrete_card(monkeypatch):
    output = """GPU0:
        vendorID = 0x1002
        deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    """
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: "vulkaninfo")
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output),
    )
    assert hardware.dedicated_amd_vulkan_available() is True


def test_cpu_runtime_rebuilds_missing_or_stale_trainer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(llama_cpp, "llama_binary", lambda _name: None)
    monkeypatch.setattr(llama_cpp, "llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr(llama_cpp, "build_llama_cpp", lambda **kwargs: calls.append(kwargs))
    llama_cpp.ensure_runtime_accelerator("cpu")
    assert len(calls) == 1
    assert calls[0]["accelerator"] is Accelerator.CPU
    assert calls[0]["build_dir"] == tmp_path / "native"


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


@pytest.mark.parametrize(
    "system,accelerator,native",
    [
        ("Windows", Accelerator.CPU, "ON"),
        ("Windows", Accelerator.VULKAN, "ON"),
        ("Windows", Accelerator.CUDA, "ON"),
        ("Darwin", Accelerator.METAL, "OFF"),
        ("Linux", Accelerator.CPU, "OFF"),
        ("Linux", Accelerator.CUDA, "OFF"),
    ],
)
def test_local_windows_build_detects_cpu_without_changing_gpu_backend(
    monkeypatch, tmp_path, system, accelerator, native
):
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: system)
    monkeypatch.setattr(llama_cpp, "_cuda_architectures", lambda: ())
    commands = []
    monkeypatch.setattr(
        llama_cpp, "run_logged", lambda command, **_kwargs: commands.append(command)
    )
    llama_cpp._configure("cmake", tmp_path, tmp_path / "build", tmp_path / "log", accelerator)
    flags = commands[0]
    assert f"-DGGML_NATIVE={native}" in flags
    for backend, selected in (
        ("METAL", Accelerator.METAL),
        ("CUDA", Accelerator.CUDA),
        ("VULKAN", Accelerator.VULKAN),
    ):
        assert f"-DGGML_{backend}={'ON' if accelerator is selected else 'OFF'}" in flags


def test_explicit_cuda_build_failure_never_switches_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(llama_cpp, "llama_cpp_root", lambda: tmp_path)
    monkeypatch.setattr(llama_cpp, "_cmake_executable", lambda: "cmake")
    monkeypatch.setattr(
        llama_cpp, "select_llama_accelerator", lambda *_args, **_kwargs: Accelerator.CUDA
    )
    monkeypatch.setattr(llama_cpp, "_vulkan_build_available", lambda: True)
    attempts = []

    def fail_configure(_cmake, _root, _build, _log, accelerator, also_vulkan=False):
        attempts.append((accelerator, also_vulkan))
        raise TrainingError("CUDA build failed")

    monkeypatch.setattr(llama_cpp, "_configure", fail_configure)
    with pytest.raises(TrainingError, match="CUDA build failed"):
        llama_cpp.build_llama_cpp(
            log_path=tmp_path / "build.log",
            build_dir=tmp_path / "build",
            accelerator=Accelerator.CUDA,
            cpu_fallback=True,
        )
    assert attempts == [(Accelerator.CUDA, False)]


def test_required_mixed_build_does_not_fall_back_to_cuda_only(monkeypatch, tmp_path):
    monkeypatch.setattr(llama_cpp, "llama_cpp_root", lambda: tmp_path)
    monkeypatch.setattr(llama_cpp, "_cmake_executable", lambda: "cmake")
    monkeypatch.setattr(
        llama_cpp, "select_llama_accelerator", lambda *_args, **_kwargs: Accelerator.CUDA
    )
    attempts = []

    def fail_configure(_cmake, _root, _build, _log, accelerator, also_vulkan=False):
        attempts.append((accelerator, also_vulkan))
        raise TrainingError("combined build failed")

    monkeypatch.setattr(llama_cpp, "_configure", fail_configure)
    with pytest.raises(TrainingError, match="combined build failed"):
        llama_cpp.build_llama_cpp(
            log_path=tmp_path / "build.log",
            build_dir=tmp_path / "build",
            accelerator=Accelerator.CUDA,
            also_vulkan=True,
            require_vulkan=True,
        )
    assert attempts == [(Accelerator.CUDA, True)]


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
    (packaged.parent / "OSAI_BUILD.json").write_text('{"llamaAccelerator":"cuda"}')
    (cached / name).write_bytes(b"cached")
    tokenizer = cached / f"llama-tokenize{suffix}"
    tokenizer.write_bytes(b"cached tokenizer")
    monkeypatch.setattr("osai.paths.project_root", lambda: tmp_path / "source")
    monkeypatch.setattr("osai.paths.llama_runtime_build", lambda: tmp_path / "cache")
    assert llama_binary("llama-finetune") == packaged / name
    marker = tmp_path / "cache" / "OSAI_BUILD.json"
    marker.write_text('{"llamaAccelerator":"cuda"}')
    assert llama_binary("llama-finetune") == cached / name

    source = tmp_path / "source" / "vendor" / "llama.cpp" / "src" / "llama-context.cpp"
    source.parent.mkdir()
    source.write_text("updated native source", encoding="utf-8")
    built_at = 1_700_000_000_000_000_000
    os.utime(marker, ns=(built_at, built_at))
    os.utime(source, ns=(built_at + 1_000_000_000, built_at + 1_000_000_000))
    assert llama_binary("llama-finetune") == packaged / name
    assert hardware._compiled_llama_accelerator() == "cuda"
    assert llama_binary("llama-tokenize") == tokenizer

    os.utime(packaged.parent / "OSAI_BUILD.json", ns=(built_at, built_at))
    assert llama_binary("llama-finetune") is None
    assert hardware._compiled_llama_accelerator() is None


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
