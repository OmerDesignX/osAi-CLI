"""Read-only validation for published training sessions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import VerificationError


@dataclass(frozen=True, slots=True)
class SessionHealth:
    root: Path
    sessions: int
    completed: int
    failed: int
    active: int
    published_models: int

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["root"] = str(self.root)
        return result


def check_sessions(root: str | Path, *, require_completed: bool = False) -> SessionHealth:
    session_root = Path(root).expanduser().resolve()
    if not session_root.is_dir():
        raise VerificationError(f"session root does not exist: {session_root}")

    manifests = sorted(session_root.glob("*/manifests/run.json"))
    completed = failed = active = published = 0
    for manifest_path in manifests:
        session = manifest_path.parent.parent
        manifest = _read_json(manifest_path)
        status = manifest.get("status")
        if status == "completed":
            completed += 1
            published += _check_completed_session(session, manifest)
        elif status == "failed":
            failed += 1
        else:
            active += 1
        _check_staging_cleanup(session)

    if require_completed and completed == 0:
        raise VerificationError(f"no completed sessions found under {session_root}")
    return SessionHealth(
        root=session_root,
        sessions=len(manifests),
        completed=completed,
        failed=failed,
        active=active,
        published_models=published,
    )


def _check_completed_session(session: Path, manifest: dict[str, Any]) -> int:
    deployment = manifest.get("base_plus_adapter")
    if not isinstance(deployment, dict):
        raise VerificationError(f"completed session lacks base_plus_adapter: {session}")
    deployment_path = _contained_path(
        session, deployment.get("deployment_manifest"), "deployment manifest"
    )
    deployment_data = _read_json(deployment_path)
    adapters = deployment_data.get("adapters")
    if not isinstance(adapters, dict) or not adapters:
        raise VerificationError(f"deployment has no adapters: {deployment_path}")
    for name, relative in adapters.items():
        if not isinstance(relative, str):
            raise VerificationError(f"invalid {name} adapter path: {deployment_path}")
        adapter = (deployment_path.parent / relative).resolve()
        if not adapter.is_relative_to(deployment_path.parent.resolve()) or not adapter.exists():
            raise VerificationError(f"missing or unsafe {name} adapter: {adapter}")

    merged = manifest.get("merged_model")
    if merged is None:
        return 0
    if not isinstance(merged, dict):
        raise VerificationError(f"invalid merged model record: {session}")
    merged_path = _contained_path(session, merged.get("path"), "merged model")
    if merged_path.is_file() and merged_path.stat().st_size == 0:
        raise VerificationError(f"merged model is empty: {merged_path}")
    if merged_path.is_dir() and not any(path.is_file() for path in merged_path.rglob("*")):
        raise VerificationError(f"merged model directory is empty: {merged_path}")
    return 1


def _check_staging_cleanup(session: Path) -> None:
    internal = session / ".internal"
    if not internal.is_dir():
        return
    leftovers = [
        path
        for pattern in ("mlx-merge-*", "gguf-merge-*")
        for path in internal.glob(pattern)
    ]
    if leftovers:
        raise VerificationError(f"temporary merge directory remains: {leftovers[0]}")


def _contained_path(session: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise VerificationError(f"completed session lacks {label}: {session}")
    path = Path(raw).expanduser().resolve()
    if not path.is_relative_to(session.resolve()):
        raise VerificationError(f"{label} escapes session directory: {path}")
    if not path.exists():
        raise VerificationError(f"missing {label}: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"JSON manifest must contain an object: {path}")
    return value
