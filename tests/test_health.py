import json
from pathlib import Path

import pytest

from osai.errors import VerificationError
from osai.health import check_sessions


def test_check_sessions_validates_completed_publication(tmp_path: Path):
    session = tmp_path / "sessions" / "one"
    deployment = session / "outputs" / "base-plus-adapter" / "deployment.json"
    adapter = deployment.parent / "adapters" / "gguf" / "adapter.gguf"
    merged = session / "outputs" / "merged-model" / "gguf" / "merged.gguf"
    manifest = session / "manifests" / "run.json"
    adapter.parent.mkdir(parents=True)
    merged.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter")
    merged.write_bytes(b"merged")
    deployment.write_text(json.dumps({"adapters": {"gguf": "adapters/gguf/adapter.gguf"}}))
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "base_plus_adapter": {"deployment_manifest": str(deployment)},
                "merged_model": {"path": str(merged)},
            }
        )
    )

    report = check_sessions(tmp_path / "sessions", require_completed=True)

    assert report.completed == 1
    assert report.published_models == 1


def test_check_sessions_rejects_missing_adapter(tmp_path: Path):
    session = tmp_path / "sessions" / "one"
    deployment = session / "outputs" / "base-plus-adapter" / "deployment.json"
    manifest = session / "manifests" / "run.json"
    deployment.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    deployment.write_text(json.dumps({"adapters": {"gguf": "adapters/gguf/missing.gguf"}}))
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "base_plus_adapter": {"deployment_manifest": str(deployment)},
                "merged_model": None,
            }
        )
    )

    with pytest.raises(VerificationError, match="missing or unsafe"):
        check_sessions(tmp_path / "sessions")
