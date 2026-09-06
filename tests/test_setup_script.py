import importlib.util
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
