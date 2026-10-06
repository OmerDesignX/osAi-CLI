from pathlib import Path

import pytest

from osai import paths
from osai.backends import llama_cpp
from osai.errors import DependencyError


def test_managed_install_uses_short_native_build_directory(monkeypatch, tmp_path: Path):
    install = tmp_path / "installations" / "1234567890"
    executable = install / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (install / "source").mkdir()
    monkeypatch.delenv("OSAI_LLAMA_BUILD_DIR", raising=False)
    monkeypatch.setattr(paths.sys, "executable", str(executable))
    assert paths.llama_runtime_build() == install / "native"


def test_cmake_uses_bundled_binary_when_shell_path_lacks_it(monkeypatch, tmp_path: Path):
    cmake = tmp_path / "native-tools" / "cmake" / "bin" / "cmake.exe"
    cmake.parent.mkdir(parents=True)
    cmake.touch()
    monkeypatch.setenv("OSAI_CMAKE", str(cmake))
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda _: None)
    assert llama_cpp._cmake_executable() == str(cmake)


def test_stale_vulkan_ninja_cache_uses_current_environment(monkeypatch, tmp_path: Path):
    cache = (
        tmp_path
        / "ggml"
        / "src"
        / "ggml-vulkan"
        / "vulkan-shaders-gen-prefix"
        / "src"
        / "vulkan-shaders-gen-build"
        / "CMakeCache.txt"
    )
    cache.parent.mkdir(parents=True)
    cache.write_text(
        "CMAKE_MAKE_PROGRAM:FILEPATH=C:/removed-app-build/ninja.exe\n",
        encoding="utf-8",
    )
    ninja = tmp_path / "tools" / "ninja.exe"
    ninja.parent.mkdir()
    ninja.touch()
    monkeypatch.setattr(
        llama_cpp.shutil,
        "which",
        lambda value: str(ninja) if value == "ninja" else None,
    )
    llama_cpp._repair_stale_ninja_cache(tmp_path)
    assert cache.read_text(encoding="utf-8") == (
        f"CMAKE_MAKE_PROGRAM:FILEPATH={ninja.as_posix()}\n"
    )


def test_portable_build_packages_runtime_dlls_for_normal_app_launch(tmp_path):
    toolchain = tmp_path / "llvm-mingw-20260616-ucrt-x86_64" / "bin"
    toolchain.mkdir(parents=True)
    runtime_names = ("libc++.dll", "libunwind.dll", "libomp.dll", "libwinpthread-1.dll")
    for name in runtime_names:
        (toolchain / name).write_bytes(name.encode())
    build = tmp_path / "native"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_CXX_COMPILER:FILEPATH={(toolchain / 'clang++.exe').as_posix()}\n",
        encoding="utf-8",
    )
    llama_cpp._copy_windows_portable_runtime_dlls(build)
    for name in runtime_names:
        assert (build / "bin" / name).read_bytes() == name.encode()


def test_missing_portable_runtime_is_reported_before_build_is_marked_ready(tmp_path):
    toolchain = tmp_path / "llvm-mingw-test" / "bin"
    toolchain.mkdir(parents=True)
    build = tmp_path / "native"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_CXX_COMPILER:FILEPATH={(toolchain / 'clang++.exe').as_posix()}\n",
        encoding="utf-8",
    )
    with pytest.raises(DependencyError, match="portable C\\+\\+ runtime is missing"):
        llama_cpp._copy_windows_portable_runtime_dlls(build)


def test_microsoft_compiler_build_does_not_require_portable_runtime(tmp_path):
    build = tmp_path / "native"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        "CMAKE_CXX_COMPILER:FILEPATH=C:/VisualStudio/bin/cl.exe\n", encoding="utf-8"
    )
    llama_cpp._copy_windows_portable_runtime_dlls(build)
    assert not (build / "bin").exists()
