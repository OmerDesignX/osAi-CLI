"""Subprocess output remains Unicode when piped into the desktop app."""

from __future__ import annotations

import io
import sys

from osai.process import run_logged


class _Pipe:
    encoding = "cp1252"

    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def isatty(self) -> bool:
        return False


def test_piped_native_metrics_remain_utf8(monkeypatch, tmp_path):
    pipe = _Pipe()
    monkeypatch.setattr(sys, "stdout", pipe)
    run_logged(
        [sys.executable, "-c", "print('loss=0.51±0.06 acc=84.38±1.40%')"],
        log_path=tmp_path / "train.log",
    )
    assert "loss=0.51±0.06 acc=84.38±1.40%" in pipe.buffer.getvalue().decode("utf-8")
