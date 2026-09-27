from pathlib import Path

from osai import paths
from osai.backends import llama_cpp


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
