"""The installation probe must not load training backends."""

import subprocess
import sys

import osai


def test_version_entry_avoids_trainer_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from osai.entry import main; "
            "code = main(['--version']); "
            "assert 'osai.cli' not in sys.modules; raise SystemExit(code)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"osai {osai.__version__}"


def test_module_entry_keeps_regular_cli_commands():
    result = subprocess.run(
        [sys.executable, "-m", "osai", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "quantization-preserving training" in result.stdout
