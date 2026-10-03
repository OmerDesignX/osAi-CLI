"""Checkpoint requests keep one complete, replaceable adapter snapshot."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from osai.backends.llama_gradient import LlamaGradientOptions, _run_parallel_gradient
from osai.checkpoints import CheckpointPublisher, LatestCheckpoint
from osai.errors import TrainingError
from osai.hardware import Accelerator


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
    latest = tmp_path / "outputs" / "checkpoint" / "adapter" / "last.gguf"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"first")

    def create_bundle(_model: Path, _shards: tuple[Path, ...], adapter: Path, stage: Path):
        stage.mkdir(parents=True)
        shutil.copyfile(adapter, stage / "osai_adapter.gguf")
        (stage / "osai_fusion.json").write_text("{}", encoding="utf-8")

    publisher = CheckpointPublisher(kind="gguf", model=model, shards=(model,), latest=latest)
    with patch("osai.fusion.create_gguf_fusion_bundle", side_effect=create_bundle):
        publisher._publish_if_new()
        published = publisher.destination / "osai_adapter.gguf"
        assert published.read_bytes() == b"first"
        replacement = latest.with_name("replacement.gguf")
        replacement.write_bytes(b"second")
        os.replace(replacement, latest)
        publisher._publish_if_new()
        assert published.read_bytes() == b"second"
        assert not list(publisher.destination.parent.glob("*.pending"))


def test_parallel_checkpoint_waits_for_both_gpu_snapshots(tmp_path: Path) -> None:
    work = tmp_path / ".internal"
    logs = tmp_path / "logs"
    work.mkdir()
    logs.mkdir()
    request = tmp_path / "checkpoint.request"
    request.write_text("save-both\n", encoding="utf-8")
    combines: list[tuple[bytes, ...]] = []

    def fake_run(command, *, log_path, env, output_prefix, **_kwargs):
        snapshot = Path(env["OSAI_CHECKPOINT_OUTPUT"])
        snapshot.write_bytes(output_prefix.encode())
        Path(log_path).write_text("epoch=1 train_loss=0.5\ndata=1/2\n", encoding="utf-8")
        Path(env["OSAI_CHECKPOINT_ACK"]).write_text("save-both\n", encoding="utf-8")
        time.sleep(0.8)
        Path(command[1]).write_bytes(output_prefix.encode())

    def fake_combine(adapters, _weights, destination, *_args):
        values = tuple(path.read_bytes() for path in adapters)
        combines.append(values)
        destination.write_bytes(b"|".join(values))

    settings = LlamaGradientOptions(epochs=1, mask_prompt=False, devices=("CUDA0", "CUDA1"))

    def fake_command(_binary, _model, _adapter, _corpus, output, *_rest):
        return ["fake", str(output)]

    with (
        patch(
            "osai.backends.llama_gradient._write_corpus",
            side_effect=lambda _, path, *_args, **_kwargs: path,
        ),
        patch(
            "osai.backends.llama_gradient._gradient_command",
            side_effect=fake_command,
        ),
        patch("osai.backends.llama_gradient.run_logged", side_effect=fake_run),
        patch(
            "osai.backends.llama_gradient._combine_lora_adapters",
            side_effect=fake_combine,
        ),
        patch("osai.backends.llama_gradient._verify_adapter"),
        patch(
            "osai.backends.llama_gradient._adapter_tensor_digest",
            return_value="trained",
        ),
        patch("osai.backends.llama_gradient.CheckpointPublisher"),
    ):
        losses, steps = _run_parallel_gradient(
            Path("fake"),
            tmp_path / "base.gguf",
            "test",
            (tmp_path / "base.gguf",),
            tmp_path / "train.jsonl",
            tmp_path / "initial.gguf",
            work / "final.gguf",
            work,
            logs,
            settings,
            Accelerator.CUDA,
            {"OSAI_CHECKPOINT_REQUEST": str(request)},
            2,
            "initial",
            object(),
        )
    assert losses == (0.5,)
    assert steps == 4
    assert len(combines) == 2
    assert (tmp_path / "outputs" / "checkpoint" / "last.ack").read_text().strip() == "save-both"
    assert (work / "final.gguf").read_bytes() == b"[CUDA0] |[CUDA1] "


def test_parallel_failure_stops_other_gpu_and_reports_the_failed_device(tmp_path: Path) -> None:
    work = tmp_path / ".internal"
    logs = tmp_path / "logs"
    work.mkdir()
    logs.mkdir()

    def fake_run(_command, *, output_prefix, cancel_event, **_kwargs):
        if output_prefix == "[CUDA1] ":
            time.sleep(0.1)
            raise TrainingError("non-finite supervised loss")
        assert cancel_event.wait(timeout=3)

    with (
        patch(
            "osai.backends.llama_gradient._write_corpus",
            side_effect=lambda _, path, *_args, **_kwargs: path,
        ),
        patch(
            "osai.backends.llama_gradient._gradient_command",
            return_value=["fake"],
        ),
        patch("osai.backends.llama_gradient.run_logged", side_effect=fake_run),
        pytest.raises(TrainingError, match="failed on CUDA1"),
    ):
        _run_parallel_gradient(
            Path("fake"),
            tmp_path / "base.gguf",
            "test",
            (tmp_path / "base.gguf",),
            tmp_path / "train.jsonl",
            tmp_path / "initial.gguf",
            work / "final.gguf",
            work,
            logs,
            LlamaGradientOptions(
                epochs=1,
                mask_prompt=False,
                devices=("CUDA0", "CUDA1"),
                calibration_pilot=True,
            ),
            Accelerator.CUDA,
            {"OSAI_CHECKPOINT_REQUEST": str(tmp_path / "checkpoint.request")},
            2,
            "initial",
            object(),
        )
