"""Checkpoint requests keep one complete, replaceable adapter snapshot."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from osai.backends.llama_gradient import (
    LlamaGradientOptions,
    _gradient_worker_log_summary,
    _run_parallel_gradient,
    _weighted_record_shards,
)
from osai.checkpoints import CheckpointPublisher, LatestCheckpoint
from osai.errors import TrainingError
from osai.hardware import Accelerator


def test_weighted_gpu_shards_keep_all_records_and_follow_measured_speeds(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    lengths = [100 + index * 17 for index in range(30)]
    source.write_bytes(b"".join(b"x" * length + b"\n" for length in lengths))
    shards = _weighted_record_shards(source, (2.0, 1.0), len(lengths))
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == set(range(len(lengths)))
    workloads = [sum(lengths[index] + 1 for index in shard) for shard in shards]
    assert 1.5 < workloads[0] / workloads[1] < 2.5
    two_rows = tmp_path / "two.jsonl"
    two_rows.write_bytes(b"x" * 200 + b"\n" + b"y" * 20 + b"\n")
    assert _weighted_record_shards(two_rows, (100.0, 1.0), 2) == (
        frozenset({0}),
        frozenset({1}),
    )


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
        400.0,
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
    (tmp_path / "train.jsonl").write_bytes(b'{"text":"a"}\n{"text":"b"}\n')
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

    settings = LlamaGradientOptions(
        epochs=1, mask_prompt=False, devices=("CUDA0", "CUDA1"), threads=12
    )
    worker_threads: list[int] = []

    def fake_command(_binary, _model, _adapter, _corpus, output, worker, *_rest):
        worker_threads.append(worker.threads)
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
        patch("osai.backends.llama_gradient.os.cpu_count", return_value=12),
    ):
        losses, steps, speeds = _run_parallel_gradient(
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
    assert speeds == (2.0, 2.0)
    assert len(combines) == 2
    assert worker_threads == [6, 6]
    assert (tmp_path / "outputs" / "checkpoint" / "last.ack").read_text().strip() == "save-both"
    assert (work / "final.gguf").read_bytes() == b"[CUDA0] |[CUDA1] "


def test_parallel_failure_stops_other_gpu_and_reports_the_failed_device(tmp_path: Path) -> None:
    work = tmp_path / ".internal"
    logs = tmp_path / "logs"
    work.mkdir()
    logs.mkdir()
    (tmp_path / "train.jsonl").write_bytes(b'{"text":"a"}\n{"text":"b"}\n')

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
