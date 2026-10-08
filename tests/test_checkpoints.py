"""Checkpoint requests keep one complete, replaceable adapter snapshot."""

from __future__ import annotations

import os
import json
import shutil
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from osai.backends.llama_gradient import _gradient_worker_log_summary
from osai.checkpoints import CheckpointPublisher, LatestCheckpoint


def test_worker_summary_streams_native_metrics_and_skips_failed_attempt(tmp_path: Path):
    log = tmp_path / "worker.log"
    with log.open("w", encoding="utf-8") as handle:
        handle.write("assistant-only loss enabled for 120 labels in 2 examples\n")
        for index in range(10000):
            handle.write(f"train: data={index + 1:05d}/10000 t=00:00:20\n")
        handle.write("supervised optimizer step labels=64\n")
        handle.write("supervised optimizer step labels=56\n")
        handle.write("epoch=1 train_loss=0.5\n")
    assert _gradient_worker_log_summary(log, 1, True, 25.0) == (
        (0.5,),
        2,
        120,
        500.0,
        None,
    )
    with log.open("a", encoding="utf-8") as handle:
        previous = handle.tell()
        handle.write("epoch=1 train_loss=0.25\ncheckpoint epoch=1 best_train_loss=0.25\n")
    assert _gradient_worker_log_summary(log, 1, False, 1.0, start_offset=previous) == (
        (0.25,),
        1,
        None,
        1.0,
        (1, 0.25),
    )


def test_latest_checkpoint_replaces_adapter_and_ack(tmp_path: Path) -> None:
    request = tmp_path / "checkpoint.request"
    request.write_text("save-one\n", encoding="utf-8")
    output = tmp_path / "outputs" / "checkpoint" / "adapter" / "adapters.safetensors"
    environment = {
        "OSAI_CHECKPOINT_OUTPUT": str(output),
        "OSAI_CHECKPOINT_REQUEST": str(request),
        "OSAI_CHECKPOINT_ACK": str(tmp_path / "last.ack"),
    }
    with patch.dict(os.environ, environment):
        latest = LatestCheckpoint()
        adapter = tmp_path / "attempt" / "adapters.safetensors"
        adapter.parent.mkdir()
        (adapter.parent / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert latest.due(1, 10) == (True, "save-one")
        latest.save(adapter, lambda destination: destination.write_bytes(b"first"), "save-one")
        assert output.read_bytes() == b"first"
        assert latest.ack.read_text(encoding="utf-8").strip() == "save-one"
        assert latest.due(2, 10) == (False, "")
        latest.save(adapter, lambda destination: destination.write_bytes(b"second"))
        assert output.read_bytes() == b"second"
        assert sorted(item.name for item in output.parent.iterdir()) == [
            "adapter_config.json",
            "adapters.safetensors",
        ]


def test_checkpoint_model_updates_embedded_adapter(tmp_path: Path) -> None:
    model = tmp_path / "base.gguf"
    model.write_bytes(b"base")
    latest = tmp_path / ".internal" / "checkpoint" / "adapter" / "last.gguf"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"first")

    def create_bundle(_model: Path, _shards: tuple[Path, ...], adapter: Path, stage: Path):
        stage.mkdir(parents=True)
        (stage / "model").mkdir()
        shutil.copyfile(model, stage / "model" / model.name)
        shutil.copyfile(adapter, stage / "osai_adapter.gguf")
        (stage / "osai_fusion.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(model=stage / "model" / model.name,
                               shards=(stage / "model" / model.name,),
                               adapter=stage / "osai_adapter.gguf")

    def export(_model, _shards, adapter, destination, **_kwargs):
        destination.write_bytes(b"merged-" + adapter.read_bytes())
        return destination

    def resolve(directory):
        return SimpleNamespace(model=directory / "model" / model.name,
                               shards=(directory / "model" / model.name,),
                               adapter=directory / "osai_adapter.gguf")

    publisher = CheckpointPublisher(kind="gguf", model=model, shards=(model,), latest=latest)
    with patch("osai.fusion.create_gguf_fusion_bundle", side_effect=create_bundle), \
         patch("osai.fusion.resolve_gguf_fusion_bundle", side_effect=resolve), \
         patch("osai.merged_export.export_gguf_weights", side_effect=export):
        publisher._publish_if_new()
        published = publisher.destination / "osai_adapter.gguf"
        assert published.read_bytes() == b"first"
        assert (publisher.destination / "merged.gguf").read_bytes() == b"merged-first"
        replacement = latest.with_name("replacement.gguf")
        replacement.write_bytes(b"second")
        os.replace(replacement, latest)
        publisher._publish_if_new()
        assert published.read_bytes() == b"second"
        assert (publisher.destination / "merged.gguf").read_bytes() == b"merged-second"
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        (latest.parent.parent / "last.ack").write_text("manual-save\n", encoding="utf-8")
        (tmp_path / "checkpoint-export.json").write_text(
            json.dumps({"generation": "manual-save", "directory": str(chosen)}),
            encoding="utf-8",
        )
        replacement.write_bytes(b"third")
        os.replace(replacement, latest)
        publisher._publish_if_new()
        assert (chosen / "gguf" / "osai_adapter.gguf").read_bytes() == b"third"
        assert (chosen / "gguf" / "merged.gguf").read_bytes() == b"merged-third"
        assert not (tmp_path / "checkpoint-export.json").exists()
        assert not list(publisher.destination.parent.glob("*.pending"))


def test_checkpoint_model_waits_for_newer_adapter_before_acknowledging(
    tmp_path: Path, capsys
) -> None:
    model = tmp_path / "base.gguf"
    model.write_bytes(b"base")
    latest = tmp_path / ".internal" / "checkpoint" / "adapter" / "last.gguf"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"first")
    publisher = CheckpointPublisher(kind="gguf", model=model, shards=(model,), latest=latest)

    def publish() -> None:
        replacement = latest.with_name("replacement.gguf")
        replacement.write_bytes(b"newer-adapter")
        os.replace(replacement, latest)

    with patch.object(publisher, "_publish", side_effect=publish):
        publisher._publish_if_new()
    assert publisher.signature is None
    assert "checkpoint model ready" not in capsys.readouterr().out
