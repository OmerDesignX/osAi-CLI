import hashlib
import importlib.util
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_osai.py"
SPEC = importlib.util.spec_from_file_location("setup_osai", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
setup_osai = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup_osai
SPEC.loader.exec_module(setup_osai)


def plan(**overrides):
    values = {
        "system": "Linux",
        "architecture": "x86_64",
        "macos_major": None,
        "cuda_major": None,
        "vulkan_available": False,
    }
    values.update(overrides)
    return setup_osai.build_plan(**values)


def test_apple_silicon_builds_mlx_and_metal_llama():
    result = plan(system="Darwin", architecture="arm64", macos_major=14)
    assert result.requirements == "requirements.txt"
    assert result.mlx_accelerator == "metal"
    assert result.llama_accelerator == "metal"


def test_monterey_uses_llama_only():
    result = plan(system="Darwin", architecture="arm64", macos_major=12)
    assert result.requirements == "requirements-llama.txt"
    assert result.mlx_accelerator is None
    assert result.llama_accelerator == "metal"


def test_intel_macos_builds_metal_llama_without_mlx():
    result = plan(system="Darwin", architecture="x86_64", macos_major=13)
    assert result.requirements == "requirements-llama.txt"
    assert result.mlx_accelerator is None
    assert result.llama_accelerator == "metal"


@pytest.mark.parametrize("cuda_major", [12, 13])
def test_linux_cuda_selects_matching_requirements(cuda_major: int):
    result = plan(cuda_major=cuda_major, vulkan_available=True)
    assert result.requirements == f"requirements-linux-cuda{cuda_major}.txt"
    assert result.mlx_accelerator == "cuda"
    assert result.llama_accelerator == "cuda"


def test_linux_without_cuda_builds_cpu_mlx_and_vulkan_llama():
    result = plan(vulkan_available=True)
    assert result.requirements == "requirements.txt"
    assert result.mlx_accelerator == "cpu"
    assert result.llama_accelerator == "vulkan"


def test_windows_uses_llama_only():
    result = plan(system="Windows", vulkan_available=True)
    assert result.requirements == "requirements-llama.txt"
    assert result.mlx_accelerator is None
    assert result.llama_accelerator == "vulkan"


def test_explicit_cpu_overrides_detected_cuda_for_llama():
    result = plan(cuda_major=12, llama_accelerator="cpu")
    assert result.llama_accelerator == "cpu"


def test_unavailable_accelerator_is_rejected():
    with pytest.raises(setup_osai.SetupError, match="unavailable"):
        plan(llama_accelerator="cuda")


def test_old_macos_is_rejected():
    with pytest.raises(setup_osai.SetupError, match="macOS 12"):
        plan(system="Darwin", architecture="arm64", macos_major=11)


def test_offline_mode_requires_wheelhouse():
    with pytest.raises(setup_osai.SetupError, match="requires --wheelhouse"):
        setup_osai._wheelhouse(True, None)


def test_current_environment_preserves_the_active_python_path():
    args = SimpleNamespace(current_environment=True)
    assert setup_osai._target_python(args, dry_run=True) == Path(sys.executable).absolute()


def test_setup_selects_wheel_for_current_revision(monkeypatch, tmp_path: Path):
    (tmp_path / "VERSION.txt").write_text("9.9.9\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    old_wheel = dist / "osai-9.9.8-py3-none-any.whl"
    new_wheel = dist / "osai-9.9.9-py3-none-any.whl"
    old_wheel.touch()
    new_wheel.touch()
    monkeypatch.setattr(setup_osai, "PROJECT_ROOT", tmp_path)
    assert setup_osai._install_target(None) == new_wheel


def test_gpu_setup_does_not_silently_install_cpu_backend(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(setup_osai, "_vulkan_available", lambda: True)
    commands = setup_osai._setup_commands(
        target_python=tmp_path / "python",
        plan=plan(system="Windows", cuda_major=12, vulkan_available=True),
        requirements=tmp_path / "requirements.txt",
        install_target=tmp_path / "osai.whl",
        wheelhouse=None,
        dev=False,
        skip_mlx_build=True,
        skip_llama_build=False,
        jobs=2,
    )
    build = next(command for command, _ in commands if "build-llama" in command)
    assert "--also-vulkan" not in build
    assert "--no-cpu-fallback" in build


def test_mixed_cuda_amd_setup_requires_both_native_backends(tmp_path: Path):
    commands = setup_osai._setup_commands(
        target_python=tmp_path / "python",
        plan=plan(system="Windows", cuda_major=12, vulkan_available=True),
        requirements=tmp_path / "requirements.txt",
        install_target=tmp_path / "osai.whl",
        wheelhouse=None,
        dev=False,
        skip_mlx_build=True,
        skip_llama_build=False,
        jobs=2,
        also_vulkan=True,
    )
    build = next(command for command, _ in commands if "build-llama" in command)
    assert "--also-vulkan" in build
    assert "--require-vulkan" in build
    assert "--no-cpu-fallback" in build


def test_only_dedicated_amd_vulkan_gpu_triggers_mixed_build():
    summary = """GPU0:
        vendorID = 0x1002
        deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
GPU1:
        vendorID = 0x10de
        deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
"""
    assert not setup_osai._vulkan_summary_has_dedicated_amd(summary)
    assert setup_osai._vulkan_summary_has_dedicated_amd(
        summary + "GPU2:\n vendorID = 0x1002\n deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
    )


def test_windows_cuda_setup_skips_vulkan_sdk_download(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(setup_osai.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_osai, "_cuda_major", lambda: 12)
    monkeypatch.delenv("VULKAN_SDK", raising=False)
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    (tmp_path / "System32").mkdir()
    (tmp_path / "System32" / "vulkan-1.dll").touch()
    monkeypatch.setattr(
        setup_osai,
        "_install_windows_vulkan_sdk",
        lambda: pytest.fail("CUDA setup must not download the Vulkan SDK"),
    )
    setup_osai._discover_local_sdks(install_missing=True)


def test_explicit_vulkan_setup_can_download_sdk_with_cuda_present(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(setup_osai.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_osai, "_cuda_major", lambda: 12)
    monkeypatch.delenv("VULKAN_SDK", raising=False)
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setattr(setup_osai, "_find_windows_vulkan_sdk", lambda _: None)
    (tmp_path / "System32").mkdir()
    (tmp_path / "System32" / "vulkan-1.dll").touch()
    sdk = tmp_path / "VulkanSDK"
    monkeypatch.setattr(setup_osai, "_install_windows_vulkan_sdk", lambda: sdk)
    setup_osai._discover_local_sdks(install_missing=True, prefer_vulkan=True)
    assert setup_osai.os.environ["VULKAN_SDK"] == str(sdk)


def test_windows_setup_reuses_registered_sdk_from_older_install(monkeypatch, tmp_path: Path):
    sdk = tmp_path / "older-install" / "vulkan-sdk"
    for relative in ("Bin/glslc.exe", "Include/vulkan/vulkan.h", "Lib/vulkan-1.lib"):
        file = sdk / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    monkeypatch.setattr(setup_osai.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_osai, "_cuda_major", lambda: None)
    monkeypatch.setattr(setup_osai, "_windows_registered_vulkan_sdk_locations", lambda: [sdk])
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "Program Files"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setenv("VULKAN_SDK", str(tmp_path / "removed-sdk"))
    (tmp_path / "System32").mkdir()
    (tmp_path / "System32" / "vulkan-1.dll").touch()
    monkeypatch.setattr(
        setup_osai,
        "_install_windows_vulkan_sdk",
        lambda: pytest.fail("an installed SDK should be reused"),
    )
    setup_osai._discover_local_sdks(install_missing=True)
    assert setup_osai.os.environ["VULKAN_SDK"] == str(sdk)


def test_windows_vulkan_download_uses_shared_cache_once(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    sdk = setup_osai._shared_windows_vulkan_sdk()
    installer = sdk.parent / f"vulkansdk-{setup_osai.WINDOWS_VULKAN_VERSION}.exe"
    installer.parent.mkdir(parents=True)
    installer.touch()
    monkeypatch.setattr(setup_osai, "_file_sha256", lambda _: setup_osai.WINDOWS_VULKAN_SHA256)
    monkeypatch.setattr(
        setup_osai.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=4 * 1024**3),
    )
    calls = []

    def install(command, **_kwargs):
        calls.append(command)
        for relative in ("Bin/glslc.exe", "Include/vulkan/vulkan.h", "Lib/vulkan-1.lib"):
            file = sdk / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()

    monkeypatch.setattr(setup_osai.subprocess, "run", install)
    assert setup_osai._install_windows_vulkan_sdk() == sdk
    assert setup_osai._install_windows_vulkan_sdk() == sdk
    assert len(calls) == 1
    assert calls[0][2] == str(sdk)


def test_portable_compiler_exposes_windows_10_file_apis(monkeypatch, tmp_path):
    binary = tmp_path / "build" / "osai-tools" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"
    binary.mkdir(parents=True)
    (binary / "clang++.exe").touch()
    monkeypatch.setattr(setup_osai, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("OSAI_VSDEVCMD", raising=False)
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.setenv("CFLAGS", "-O2")
    monkeypatch.setenv("CXXFLAGS", "-fno-omit-frame-pointer")
    environment = setup_osai._windows_compiler_environment(
        plan(system="Windows", vulkan_available=True), tmp_path / "Scripts" / "python.exe"
    )
    assert environment["CFLAGS"] == "-O2 -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00"
    assert environment["CXXFLAGS"] == (
        "-fno-omit-frame-pointer -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00"
    )


def windows_sdk(monkeypatch, tmp_path):
    compiler = tmp_path / "Bin" / "glslc.exe"
    compiler.parent.mkdir()
    compiler.touch()
    installer = tmp_path / "Helpers" / "VC_redist.X64.exe"
    installer.parent.mkdir()
    installer.touch()
    monkeypatch.setattr(setup_osai, "_windows_vc_redist_installer", lambda: installer)
    monkeypatch.setattr(setup_osai, "_file_sha256", lambda _: setup_osai.WINDOWS_VC_REDIST_SHA256)
    return {"VULKAN_SDK": str(tmp_path), "PATH": "build-tools"}, compiler, installer


def test_working_shader_compiler_needs_no_runtime_install(monkeypatch, tmp_path):
    environment, compiler, _ = windows_sdk(monkeypatch, tmp_path)

    def probe(selected, selected_environment):
        assert selected == compiler
        assert selected_environment == environment
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(setup_osai, "_probe_windows_glslc", probe)
    monkeypatch.setattr(
        setup_osai,
        "_windows_vc_redist_installer",
        lambda: pytest.fail("unnecessary runtime install"),
    )
    setup_osai._ensure_windows_vulkan_runtime(environment)


@pytest.mark.parametrize(
    "missing_dll_code",
    [-1073741515, 3221225781, -1073741511, 3221225785, -1073741819, 3221225477],
)
@pytest.mark.parametrize("installer_code", [0, 1638, 3010])
def test_missing_shader_dll_installs_verified_runtime(
    monkeypatch, tmp_path, missing_dll_code, installer_code
):
    environment, _, installer = windows_sdk(monkeypatch, tmp_path)
    results = iter([missing_dll_code, 0])
    monkeypatch.setattr(
        setup_osai, "_probe_windows_glslc", lambda *_: SimpleNamespace(returncode=next(results))
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["env"] == environment
        return SimpleNamespace(returncode=installer_code)

    monkeypatch.setattr(setup_osai.subprocess, "run", run)
    setup_osai._ensure_windows_vulkan_runtime(environment)
    assert commands == [[str(installer), "/install", "/quiet", "/norestart"]]


def test_runtime_installer_checksum_is_required(monkeypatch, tmp_path):
    environment, _, _ = windows_sdk(monkeypatch, tmp_path)
    monkeypatch.setattr(
        setup_osai, "_probe_windows_glslc", lambda *_: SimpleNamespace(returncode=-1073741515)
    )
    monkeypatch.setattr(setup_osai, "_file_sha256", lambda _: "tampered")
    monkeypatch.setattr(
        setup_osai.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("unverified installer")
    )
    with pytest.raises(setup_osai.SetupError, match="SHA-256 verification"):
        setup_osai._ensure_windows_vulkan_runtime(environment)


def test_other_shader_errors_do_not_install_runtime(monkeypatch, tmp_path):
    environment, _, _ = windows_sdk(monkeypatch, tmp_path)
    monkeypatch.setattr(
        setup_osai,
        "_probe_windows_glslc",
        lambda *_: SimpleNamespace(returncode=1, stdout="", stderr="compiler error"),
    )
    monkeypatch.setattr(
        setup_osai,
        "_windows_vc_redist_installer",
        lambda: pytest.fail("unnecessary runtime install"),
    )
    with pytest.raises(setup_osai.SetupError, match="compiler error"):
        setup_osai._ensure_windows_vulkan_runtime(environment)


def test_runtime_installer_failure_reports_admin_prompt(monkeypatch, tmp_path):
    environment, _, _ = windows_sdk(monkeypatch, tmp_path)
    monkeypatch.setattr(
        setup_osai, "_probe_windows_glslc", lambda *_: SimpleNamespace(returncode=-1073741515)
    )
    monkeypatch.setattr(
        setup_osai.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1602)
    )
    with pytest.raises(setup_osai.SetupError, match="approve the Windows administrator prompt"):
        setup_osai._ensure_windows_vulkan_runtime(environment)


def test_runtime_install_must_make_compiler_executable(monkeypatch, tmp_path):
    environment, _, _ = windows_sdk(monkeypatch, tmp_path)
    monkeypatch.setattr(
        setup_osai, "_probe_windows_glslc", lambda *_: SimpleNamespace(returncode=-1073741515)
    )
    monkeypatch.setattr(
        setup_osai.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
    )
    with pytest.raises(setup_osai.SetupError, match="still cannot start"):
        setup_osai._ensure_windows_vulkan_runtime(environment)


def test_shader_probe_timeout_is_reported(monkeypatch, tmp_path):
    environment, _, _ = windows_sdk(monkeypatch, tmp_path)

    def probe(*_):
        raise subprocess.TimeoutExpired("glslc", 30)

    monkeypatch.setattr(setup_osai, "_probe_windows_glslc", probe)
    with pytest.raises(setup_osai.SetupError, match="timed out"):
        setup_osai._ensure_windows_vulkan_runtime(environment)


@pytest.mark.parametrize("failure", [None, "checksum", "redirect"])
def test_runtime_download_requires_verified_microsoft_payload(monkeypatch, tmp_path, failure):
    payload = b"verified Microsoft runtime payload"
    monkeypatch.setattr(setup_osai, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        setup_osai,
        "WINDOWS_VC_REDIST_SHA256",
        hashlib.sha256(payload if failure != "checksum" else b"expected").hexdigest(),
    )

    class Response(io.BytesIO):
        url = (
            "https://untrusted.example/runtime.exe"
            if failure == "redirect"
            else setup_osai.WINDOWS_VC_REDIST_URL
        )

    def download(request, **_kwargs):
        assert request.full_url == setup_osai.WINDOWS_VC_REDIST_URL
        return Response(payload)

    monkeypatch.setattr(setup_osai.urllib.request, "urlopen", download)
    destination = tmp_path / "build" / "osai-tools" / "VC_redist.x64.exe"
    if failure:
        with pytest.raises(setup_osai.SetupError):
            setup_osai._windows_vc_redist_installer()
        assert not destination.exists()
    else:
        assert setup_osai._windows_vc_redist_installer().read_bytes() == payload
        monkeypatch.setattr(
            setup_osai.urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: pytest.fail("verified cache must be reused"),
        )
        assert setup_osai._windows_vc_redist_installer() == destination
    assert not destination.with_suffix(".part").exists()
