"""Observable subprocess execution without invoking a shell."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .errors import TrainingError


@dataclass(frozen=True, slots=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    log_path: Path


def run_logged(
    command: Sequence[str | os.PathLike[str]],
    *,
    log_path: str | Path,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> ProcessResult:
    argv = tuple(os.fspath(part) for part in command)
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with destination.open("a", encoding="utf-8", buffering=1) as log:
        _write_command(log, argv)
        try:
            process = subprocess.Popen(
                argv,
                cwd=os.fspath(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise TrainingError(f"could not start {argv[0]}: {exc}") from exc

        assert process.stdout is not None
        try:
            while True:
                if timeout is not None and time.monotonic() - started > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise TrainingError(f"command timed out after {timeout:.0f}s: {argv[0]}")
                line = process.stdout.readline()
                if line:
                    log.write(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                if process.poll() is not None:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise

        returncode = process.wait()
        elapsed = time.monotonic() - started
        log.write(f"\n[osai] exit={returncode} elapsed_seconds={elapsed:.3f}\n")
    result = ProcessResult(argv, returncode, elapsed, destination)
    if returncode != 0:
        raise TrainingError(f"command exited with status {returncode}; see log: {destination}")
    return result


def _write_command(log: TextIO, argv: tuple[str, ...]) -> None:
    # repr preserves argument boundaries and cannot execute because no shell is used.
    log.write("[osai] command=" + repr(argv) + "\n")
