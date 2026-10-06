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
    assert setup_osai._target_python(args, dry_run=True) == Path(
        sys.executable
    ).absolute()


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
    assert "--also-vulkan" in build
    assert "--no-cpu-fallback" in build


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
    monkeypatch.setattr(
        setup_osai, "_file_sha256", lambda _: setup_osai.WINDOWS_VC_REDIST_SHA256
    )
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
