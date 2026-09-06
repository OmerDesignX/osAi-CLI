import json
import os
from datetime import datetime
from pathlib import Path

from osai.config import ModelFormat
from osai.formats import ModelInspection, QuantizationSpec
from osai.session import (
    SessionLayout,
    publish_base_adapter_bundle,
    timestamped_session_path,
)


def test_timestamped_session_path_is_readable_and_unique(tmp_path: Path):
    now = datetime(2026, 9, 5, 14, 30, 45)
    first = timestamped_session_path(tmp_path / "sessions", "Small / MLX", now=now)
    assert first.name == "2026-09-05_14-30-45_small-mlx"
    assert first.is_dir()
    second = timestamped_session_path(tmp_path / "sessions", "Small / MLX", now=now)
    assert second.name == "2026-09-05_14-30-45_small-mlx-2"


def test_session_publishes_materialized_base_and_adapter_manifest(tmp_path: Path):
    source = tmp_path / "source" / "model-Q4_K_M.gguf"
    source.parent.mkdir()
    source.write_bytes(b"GGUF-packed-base")
    layout = SessionLayout.at(tmp_path / "session")
    layout.create()
    assert layout.rollouts.is_dir()
    adapter = layout.adapters / "gguf" / "adapter.gguf"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"GGUF-adapter")
    base = ModelInspection(
        format=ModelFormat.GGUF,
        path=source,
        architecture="test",
        quantization=QuantizationSpec("Q4_K_M"),
        size_bytes=source.stat().st_size,
        shards=(source,),
    )

    result = publish_base_adapter_bundle(
        layout,
        base,
        {"gguf": adapter},
        materialize_base=True,
    )

    assert result.materialized
    assert result.path is not None and result.path.read_bytes() == source.read_bytes()
    assert not os.path.samefile(result.path, source)
    assert result.link_modes[0] in {"copy-on-write-clone", "copy"}
    manifest = json.loads(layout.deployment_manifest.read_text())
    assert manifest["kind"] == "base-plus-lora-adapter"
    assert manifest["adapters"]["gguf"] == "adapters/gguf/adapter.gguf"
