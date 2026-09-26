import os
from pathlib import Path

from osai.paths import llama_binary, project_root


def test_installed_cli_finds_downloaded_source_and_native_binary(monkeypatch, tmp_path: Path):
    installation = tmp_path / "installations" / "20260926"
    executable = (
        installation
        / ".venv"
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )
    source = installation / "source" / "osAi-CLI-main"
    (source / "vendor" / "llama.cpp" / "build" / "bin").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (source / "vendor" / "llama.cpp" / "CMakeLists.txt").write_text("", encoding="utf-8")
    binary = (
        source
        / "vendor"
        / "llama.cpp"
        / "build"
        / "bin"
        / ("llama-completion.exe" if os.name == "nt" else "llama-completion")
    )
    binary.write_bytes(b"native")
    monkeypatch.delenv("OSAI_ROOT", raising=False)
    monkeypatch.setenv("OSAI_LLAMA_BUILD_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        "osai.paths.__file__",
        str(installation / ".venv" / "Lib" / "site-packages" / "osai" / "paths.py"),
    )
    monkeypatch.setattr("osai.paths.sys.executable", str(executable))
    monkeypatch.chdir(tmp_path)

    assert project_root() == source
    assert llama_binary("llama-completion") == binary
