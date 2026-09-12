"""Crash-safe local I/O, locking, and integrity helpers."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import TrainingError


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise


def atomic_text(path: str | Path, value: str) -> None:
    """Replace a UTF-8 text file only after its contents reach stable storage."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    path: str
    size: int
    modified_ns: int
    sha256: str | None = None


def fingerprint(path: str | Path, *, full_hash: bool = False) -> FileFingerprint:
    file_path = Path(path)
    stat = file_path.stat()
    digest = sha256_file(file_path) if full_hash else None
    return FileFingerprint(
        path=str(file_path.resolve()),
        size=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        sha256=digest,
    )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OutputLock:
    """Cross-platform single-writer lock with safe crash recovery."""

    _MALFORMED_LOCK_GRACE_SECONDS = 30.0
    _CREATE_ATTEMPTS = 8

    def __init__(self, output: str | Path):
        self.output = Path(output)
        self.path = self.output / ".osai.lock"
        self._descriptor: int | None = None
        self._payload: bytes | None = None

    def __enter__(self) -> OutputLock:
        self.output.mkdir(parents=True, exist_ok=True)
        self._payload = (
            f"version=1\npid={os.getpid()}\ncreated_ns={time.time_ns()}\n"
            f"token={secrets.token_hex(16)}\n"
        ).encode("ascii")

        for _ in range(self._CREATE_ATTEMPTS):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                if self._reclaim_stale_lock():
                    continue
                raise TrainingError(f"output is locked by another active run: {self.path}") from exc

            try:
                os.write(descriptor, self._payload)
                os.fsync(descriptor)
            except BaseException:
                os.close(descriptor)
                with suppress(FileNotFoundError):
                    self.path.unlink()
                raise
            self._descriptor = descriptor
            return self

        raise TrainingError(f"could not acquire the output lock: {self.path}")

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        try:
            if self._payload is not None and self.path.read_bytes() == self._payload:
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._payload = None

    def _reclaim_stale_lock(self) -> bool:
        """Remove a lock only when the observed owner can no longer be running."""

        try:
            descriptor = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return True

        try:
            observed = os.fstat(descriptor)
            payload = os.read(descriptor, 16 * 1024).decode("utf-8", errors="replace")
        finally:
            os.close(descriptor)

        owner_pid = _lock_owner_pid(payload)
        if owner_pid is not None:
            stale = not _process_is_running(owner_pid)
        else:
            age_seconds = max(0.0, time.time() - observed.st_mtime)
            stale = age_seconds >= self._MALFORMED_LOCK_GRACE_SECONDS

        if not stale:
            return False

        try:
            current = self.path.stat()
        except FileNotFoundError:
            return True

        observed_identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        )
        if current_identity != observed_identity:
            return True

        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise TrainingError(f"cannot recover stale output lock: {self.path}") from exc
        return True


def _lock_owner_pid(payload: str) -> int | None:
    for line in payload.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "pid":
            try:
                pid = int(value.strip())
            except ValueError:
                return None
            return pid if pid > 0 else None
    return None


def _process_is_running(pid: int) -> bool:
    """Check process liveness without sending a terminating signal."""

    if os.name == "nt":
        return _windows_process_is_running(pid)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _windows_process_is_running(pid: int) -> bool:
    """Use a read-only Windows process handle; ``os.kill(pid, 0)`` is unsafe there."""

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return ctypes.get_last_error() == error_access_denied

    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)
