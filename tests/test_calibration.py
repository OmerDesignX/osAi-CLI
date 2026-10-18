import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from osai.auto_benchmark import BenchmarkResult
from osai.auto_settings import select_auto_settings
from osai.calibration import (
    _pilot_microbatch,
    _sample_evaluation_context,
    _sample_source_rows,
    calibrate_training,
    candidate_rates,
    loss_trend,
    sample_training_data,
)
from osai.config import ModelFormat
from osai.errors import ConfigurationError, TrainingError
from osai.formats import ModelInspection, QuantizationSpec
from osai.hardware import Engine


def _model(tmp_path: Path) -> ModelInspection:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    return ModelInspection(
        format=ModelFormat.GGUF,
        path=path,
        architecture="qwen35",
        quantization=QuantizationSpec("Q4_K_M"),
        size_bytes=2 * 1024**3,
        shards=(path,),
        block_count=32,
        context_length=4096,
    )


def _data(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for part in range(2):
        with (source / f"part-{part}.jsonl").open("w", encoding="utf-8") as output:
            for index in range(14):
                output.write(
                    json.dumps({"prompt": f"Question {part}-{index}", "completion": "Answer"})
                    + "\n"
                )
    return source


def test_calibration_samples_all_selected_files_without_altering_them(tmp_path: Path):
    source = _data(tmp_path)
    rows, total, typical = sample_training_data(source, tmp_path / "pilot")
    assert rows == 4
    assert total == 28
    assert typical > 0
    assert len((tmp_path / "pilot" / "test.jsonl").read_text().splitlines()) == 2
    sample = json.loads((tmp_path / "pilot" / "train.jsonl").read_text().splitlines()[0])
    assert sample["messages"][-1]["content"] == "Answer"
    assert len((source / "part-0.jsonl").read_text().splitlines()) == 14


def test_calibration_scores_the_sample_in_a_bounded_context(tmp_path: Path):
    source = _data(tmp_path)
    sample = tmp_path / "pilot"
    sample_training_data(source, sample, context=61_588)

    evaluation_context = _sample_evaluation_context(sample, 61_588, 8)

    assert 256 <= evaluation_context < 61_588
    assert evaluation_context % 8 == 0
    largest_record_bytes = max(
        len(line.encode("utf-8"))
        for path in sample.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    assert evaluation_context >= largest_record_bytes


def test_quick_calibration_reads_each_file_with_a_bounded_sample(tmp_path: Path):
    source = _data(tmp_path)
    visited = []
    rows, inspected, _ = sample_training_data(
        source,
        tmp_path / "quick-pilot",
        max_rows_per_file=8,
        progress=lambda index, total, name: visited.append((index, total, name)),
    )
    assert rows == 4
    assert inspected == 16
    assert visited == [(1, 2, "part-0.jsonl"), (2, 2, "part-1.jsonl")]


def test_quick_calibration_samples_late_records(tmp_path: Path):
    source = tmp_path / "large.jsonl"
    source.write_text(
        "".join(
            json.dumps({"prompt": str(index), "completion": "A"}) + "\n" for index in range(1000)
        ),
        encoding="utf-8",
    )
    indices = [int(row["prompt"]) for row in _sample_source_rows(source, 16)]
    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] > 900


@pytest.mark.parametrize("count", [100, 256])
def test_quick_calibration_reads_every_row_within_sample_limit(tmp_path: Path, count: int):
    source = tmp_path / "train.jsonl"
    lines = [json.dumps({"prompt": str(index), "completion": "A"}) + "\n" for index in range(count)]
    source.write_text("".join(lines), encoding="utf-8")
    indices = [int(row["prompt"]) for row in _sample_source_rows(source, 256)]
    assert indices == list(range(count))


def test_full_dataset_calibration_reduces_short_pilot_rate(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cuda", ("CUDA0", "CUDA1"), 0.1)
    source = _data(tmp_path)
    tried = []
    progress = []

    def pilot(_model, _sample, output, *, options, accelerator):
        assert options.evaluation_context is not None
        assert options.evaluation_context < options.context
        tried.append(options.learning_rate)
        manifest = output.parent / "manifest.json"
        manifest.write_text(json.dumps({"options": {"context": options.context}}))
        return SimpleNamespace(
            losses=(0.5, 0.4, 0.3),
            manifest=manifest,
            accelerator="cuda",
            initial_test_loss=0.5,
            test_loss=0.3,
        )

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", pilot)
    result = calibrate_training(
        model.path,
        model,
        source,
        benchmark,
        engine=Engine.LLAMA_CPP,
        multi_gpu="off",
        total_training_rows=31_580,
        training_epochs=1,
        progress=progress.append,
    )
    assert result.learning_rate < tried[0] / 10
    assert result.learning_rate > 0
    assert result.improvement_percent == pytest.approx(40)
    assert any("Scaling short-pilot rate" in message for message in progress)


def test_pilot_microbatch_uses_each_gpu_memory_and_keeps_safe_fallback(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cuda", ("CUDA0", "CUDA1"), 0.1)
    gib = 1024**3
    monkeypatch.setattr(
        "osai.calibration.llama_device_free_bytes",
        lambda *_: {"CUDA0": 8 * gib, "CUDA1": 6 * gib},
    )
    assert _pilot_microbatch(settings, model, benchmark) > settings.gguf_batch_size
    monkeypatch.setattr("osai.calibration.llama_device_free_bytes", lambda *_: {"CUDA0": 8 * gib})
    assert _pilot_microbatch(settings, model, benchmark) == settings.gguf_batch_size


def test_pilot_microbatch_uses_full_context_headroom_without_model_byte_proxy(monkeypatch):
    gib = 1024**3
    settings = SimpleNamespace(
        memory_budget_bytes=64 * gib, max_seq_length=61588, gguf_batch_size=1
    )
    model = SimpleNamespace(size_bytes=int(2.5 * gib), context_length=262144)
    benchmark = SimpleNamespace(devices=("CUDA0", "CUDA1"), accelerator="cuda")
    monkeypatch.setattr(
        "osai.calibration.llama_device_free_bytes",
        lambda *_: {"CUDA0": 12 * gib, "CUDA1": 12 * gib},
    )
    assert _pilot_microbatch(settings, model, benchmark) == 128


def test_metal_pilot_starts_with_a_conservative_microbatch(monkeypatch):
    gib = 1024**3
    settings = SimpleNamespace(
        memory_budget_bytes=64 * gib, max_seq_length=61588, gguf_batch_size=1
    )
    model = SimpleNamespace(size_bytes=int(2.5 * gib), context_length=262144)
    benchmark = SimpleNamespace(devices=("MTL0", "MTL1"), accelerator="metal")
    monkeypatch.setattr(
        "osai.calibration.llama_device_free_bytes",
        lambda *_: {"MTL0": 4 * gib, "MTL1": 8 * gib},
    )

    assert _pilot_microbatch(settings, model, benchmark) == 64


def test_metal_calibration_retries_nonfinite_loss_without_reducing_context(
    monkeypatch, tmp_path: Path
):
    gib = 1024**3
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * gib)
    settings = replace(settings, max_seq_length=4096, gguf_batch_size=1)
    benchmark = BenchmarkResult(
        settings,
        "llama.cpp",
        "metal",
        ("MTL0", "MTL1"),
        0.1,
        tensor_split=(3584.0, 7372.8),
        main_gpu=1,
    )
    requested = []
    progress = []
    monkeypatch.setattr(
        "osai.calibration.llama_device_free_bytes",
        lambda *_: {"MTL0": 4 * gib, "MTL1": 8 * gib},
    )

    def pilot(_model, _sample, output, *, options, accelerator):
        requested.append(options)
        if len(requested) == 1:
            logs = output / "logs"
            logs.mkdir(parents=True)
            (logs / "train.log").write_text(
                "osai: non-finite supervised loss; stopping before another checkpoint\n",
                encoding="utf-8",
            )
            raise TrainingError("native Metal training failed")
        manifest = output.parent / "metal-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "options": {
                        "context": options.context,
                        "batch_size": options.batch_size,
                    },
                    "model_sharded_training": {"devices": ["MTL0", "MTL1"]},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            losses=(0.5, 0.4, 0.3),
            manifest=manifest,
            accelerator=accelerator,
            initial_test_loss=0.5,
            test_loss=0.3,
        )

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", pilot)
    result = calibrate_training(
        model.path,
        model,
        _data(tmp_path),
        benchmark,
        engine=Engine.LLAMA_CPP,
        multi_gpu="on",
        require_full_context=True,
        progress=progress.append,
    )

    assert [options.batch_size for options in requested] == [64, 32]
    assert all(options.context >= 4096 for options in requested)
    assert all(options.devices == ("MTL0", "MTL1") for options in requested)
    assert result.settings.max_seq_length >= 4096
    assert any("Metal numerical limit" in message for message in progress)
    assert any("full training pilot remains" in message for message in progress)


def test_calibration_bounds_long_examples_without_rewriting_training_data(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        "".join(
            json.dumps({"prompt": "p" * 20_000, "completion": "a" * 20_000}) + "\n"
            for _ in range(6)
        ),
        encoding="utf-8",
    )
    rows, total, typical = sample_training_data(
        source, tmp_path / "pilot", context=256, train_rows=4
    )
    sample = json.loads((tmp_path / "pilot" / "train.jsonl").read_text().splitlines()[0])
    assert (rows, total) == (4, 6)
    assert typical > 40_000
    assert len(sample["messages"][0]["content"]) <= 128
    assert 64 < len(sample["messages"][1]["content"]) <= 384
    assert len(source.read_text().splitlines()[0]) > 40_000


def test_loss_trend_rejects_spikes_and_chooses_measured_decrease(monkeypatch, tmp_path: Path):
    assert loss_trend((0.4, 0.45, 7.0)) is None
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)
    rates = sorted(candidate_rates(settings, 100))
    tried = []

    def pilot(_model, _sample, output, *, options, accelerator):
        assert options.calibration_pilot
        tried.append((options.learning_rate, accelerator))
        manifest = output.parent / f"manifest-{len(tried)}.json"
        manifest.write_text(json.dumps({"options": {"context": options.context}}))
        return SimpleNamespace(
            losses=(0.5, 0.6, 0.7) if len(tried) == 1 else (0.5, 0.4, 0.3),
            manifest=manifest,
            accelerator="cpu",
        )

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", pilot)
    parent = tmp_path / "calibration-parent"
    parent.mkdir()
    monkeypatch.setenv("OSAI_CALIBRATION_PARENT", str(parent))
    result = calibrate_training(
        model.path, model, _data(tmp_path), benchmark, engine=Engine.LLAMA_CPP
    )
    assert [rate for rate, _ in tried] == [rates[1], rates[0]]
    assert result.learning_rate == rates[0]
    assert result.improvement_percent == pytest.approx(40)
    assert len(tried) == 2
    assert list(parent.iterdir()) == []


def test_calibration_refuses_to_claim_success_without_a_decrease(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)
    attempted = []

    def rising(_model, _sample, output, *, options, accelerator):
        attempted.append(options.learning_rate)
        manifest = output.parent / "rising-manifest.json"
        manifest.write_text(json.dumps({"options": {"context": options.context}}))
        return SimpleNamespace(losses=(0.5, 0.6, 0.7), manifest=manifest, accelerator="cpu")

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", rising)
    with pytest.raises(ConfigurationError, match="could not verify"):
        calibrate_training(model.path, model, _data(tmp_path), benchmark, engine=Engine.LLAMA_CPP)
    assert len(attempted) == 2
    assert attempted[0] > attempted[1]


def test_calibration_rejects_overfitting_pilot(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)
    attempts = []

    def pilot(_model, _sample, output, *, options, accelerator):
        attempts.append(options.learning_rate)
        manifest = output.parent / f"holdout-{len(attempts)}.json"
        manifest.write_text(json.dumps({"options": {"context": options.context}}))
        return SimpleNamespace(
            losses=(0.5, 0.4, 0.3),
            initial_test_loss=0.5,
            test_loss=0.7 if len(attempts) == 1 else 0.4,
            manifest=manifest,
            accelerator="cpu",
        )

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", pilot)
    result = calibrate_training(
        model.path, model, _data(tmp_path), benchmark, engine=Engine.LLAMA_CPP
    )
    assert len(attempts) == 2
    assert result.learning_rate == attempts[1]


def test_calibration_keeps_cautious_rate_after_small_verified_decline(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)
    attempts = []

    def pilot(_model, _sample, output, *, options, accelerator):
        attempts.append(options.learning_rate)
        manifest = output.parent / f"faster-{len(attempts)}.json"
        manifest.write_text(json.dumps({"options": {"context": options.context}}))
        return SimpleNamespace(
            losses=(0.5, 0.4999, 0.4998) if len(attempts) == 1 else (0.5, 0.499, 0.498),
            initial_test_loss=0.5,
            test_loss=0.499 if len(attempts) == 1 else 0.498,
            manifest=manifest,
            accelerator="cpu",
        )

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", pilot)
    result = calibrate_training(
        model.path, model, _data(tmp_path), benchmark, engine=Engine.LLAMA_CPP
    )
    assert len(attempts) == 1
    assert attempts[0] == sorted(candidate_rates(settings, 100))[1]
    assert result.learning_rate == attempts[0]


def test_calibration_tries_faster_rate_only_after_flat_safe_pilots(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)
    attempted = []

    def pilot(_model, _sample, output, *, options, accelerator):
        attempted.append(options.learning_rate)
        manifest = output.parent / f"flat-{len(attempted)}.json"
        manifest.write_text(json.dumps({"options": {"context": options.context}}))
        improved = len(attempted) == 3
        return SimpleNamespace(
            losses=(0.5, 0.45, 0.4) if improved else (0.5, 0.5, 0.5),
            initial_test_loss=0.5,
            test_loss=0.4 if improved else 0.5,
            manifest=manifest,
            accelerator="cpu",
        )

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", pilot)
    result = calibrate_training(
        model.path, model, _data(tmp_path), benchmark, engine=Engine.LLAMA_CPP
    )
    low, middle, high = sorted(candidate_rates(settings, 100))
    assert attempted == [middle, low, high]
    assert result.learning_rate == high


def test_full_context_calibration_rejects_memory_fallback(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)

    def lowered(_model, _sample, output, *, options, accelerator):
        manifest = output.parent / "lowered-manifest.json"
        manifest.write_text(json.dumps({"options": {"context": options.context // 2}}))
        return SimpleNamespace(losses=(0.5, 0.4, 0.3), manifest=manifest, accelerator="cpu")

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", lowered)
    with pytest.raises(ConfigurationError, match="choose Windowing"):
        calibrate_training(
            model.path,
            model,
            _data(tmp_path),
            benchmark,
            engine=Engine.LLAMA_CPP,
            require_full_context=True,
        )


def test_multi_gpu_calibration_rejects_single_device_pilot(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cuda", ("CUDA0", "CUDA1"), 0.1)

    def one_device(_model, _sample, output, *, options, accelerator):
        manifest = output.parent / "one-device-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "options": {"context": options.context},
                    "parallel_training": {"devices": ["CUDA0"]},
                }
            )
        )
        return SimpleNamespace(losses=(0.5, 0.4, 0.3), manifest=manifest, accelerator="cuda")

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", one_device)
    with pytest.raises(ConfigurationError, match="every selected GPU"):
        calibrate_training(
            model.path,
            model,
            _data(tmp_path),
            benchmark,
            engine=Engine.LLAMA_CPP,
            multi_gpu="on",
        )


@pytest.mark.parametrize(
    "used_devices",
    [
        ["CUDA0", "CUDA1"],
        ["CUDA0", "CUDA1", "CUDA1"],
        ["CUDA0", "CUDA1", "CUDA3"],
        ["CUDA1", "CUDA0", "CUDA2"],
        ["CUDA0", "CUDA1", "CUDA2"],
    ],
)
def test_three_gpu_calibration_verifies_the_exact_sharding_devices(
    monkeypatch, tmp_path, used_devices
):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    devices = ("CUDA0", "CUDA1", "CUDA2")
    benchmark = BenchmarkResult(settings, "llama.cpp", "cuda", devices, 0.1)
    requested = []

    def pilot(_model, _sample, output, *, options, accelerator):
        requested.append(options.devices)
        manifest = output.parent / f"sharded-{len(requested)}.json"
        manifest.write_text(json.dumps({
            "options": {"context": options.context},
            "model_sharded_training": {"devices": used_devices},
        }))
        return SimpleNamespace(
            losses=(0.5, 0.4, 0.3), manifest=manifest, accelerator=accelerator,
            initial_test_loss=0.5, test_loss=0.3,
        )

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", pilot)
    if tuple(used_devices) != devices:
        with pytest.raises(ConfigurationError, match="every selected GPU"):
            calibrate_training(
                model.path, model, _data(tmp_path), benchmark,
                engine=Engine.LLAMA_CPP, multi_gpu="on",
            )
    else:
        result = calibrate_training(
            model.path, model, _data(tmp_path), benchmark,
            engine=Engine.LLAMA_CPP, multi_gpu="on",
        )
        assert result.devices == devices
    assert requested and all(selected == devices for selected in requested)
