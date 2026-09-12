from pathlib import Path

import pytest

from osai import io
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


def test_output_lock_reclaims_legacy_lock_from_dead_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    lock = tmp_path / ".osai.lock"
    lock.write_text("pid=78982\n", encoding="utf-8")
    monkeypatch.setattr(io, "_process_is_running", lambda pid: False)

    with OutputLock(tmp_path):
        contents = lock.read_text(encoding="utf-8")
        assert f"pid={io.os.getpid()}" in contents
        assert "token=" in contents

    assert not lock.exists()


def test_output_lock_does_not_reclaim_live_process_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    lock = tmp_path / ".osai.lock"
    lock.write_text("pid=42\n", encoding="utf-8")
    monkeypatch.setattr(io, "_process_is_running", lambda pid: True)

    with pytest.raises(TrainingError, match="another active run"), OutputLock(tmp_path):
        pass

    assert lock.read_text(encoding="utf-8") == "pid=42\n"
