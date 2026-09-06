from pathlib import Path

import pytest

from osai.errors import TrainingError
from osai.io import OutputLock, atomic_json


def test_atomic_json_and_lock(tmp_path: Path):
    target = tmp_path / "nested" / "value.json"
    atomic_json(target, {"ok": True})
    assert target.read_text().endswith("\n")
    with (
        OutputLock(tmp_path),
        pytest.raises(TrainingError, match="locked"),
        OutputLock(tmp_path),
    ):
        pass
    assert not (tmp_path / ".osai.lock").exists()
