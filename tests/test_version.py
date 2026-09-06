import re
from pathlib import Path

import osai

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_release_version_is_synchronized():
    root = Path(__file__).resolve().parents[1]
    version = (root / "VERSION.txt").read_text(encoding="utf-8").strip()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version)
    assert project["project"]["version"] == version
    assert osai.__version__ == version
