import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osai.auto_benchmark import BenchmarkResult
from osai.auto_settings import select_auto_settings
from osai.calibration import (
    _pilot_microbatch,
    _sample_source_rows,
    calibrate_training,
    candidate_rates,
    loss_trend,
    sample_training_data,
)
from osai.config import ModelFormat
from osai.errors import ConfigurationError
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
    assert len(sample["messages"][0]["content"]) <= 64
    assert len(sample["messages"][1]["content"]) <= 128
    assert len(source.read_text().splitlines()[0]) > 40_000


def test_loss_trend_rejects_spikes_and_chooses_measured_decrease(monkeypatch, tmp_path: Path):
    assert loss_trend((0.4, 0.45, 7.0)) is None
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)
    rates = candidate_rates(settings, 100)
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
    assert result.learning_rate == rates[1]
    assert result.improvement_percent == pytest.approx(40)
    assert len(tried) == 2
    assert list(parent.iterdir()) == []


def test_calibration_refuses_to_claim_success_without_a_decrease(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    benchmark = BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)

    def rising(_model, _sample, output, *, options, accelerator):
        manifest = output.parent / "rising-manifest.json"
        manifest.write_text(json.dumps({"options": {"context": options.context}}))
        return SimpleNamespace(losses=(0.5, 0.6, 0.7), manifest=manifest, accelerator="cpu")

    monkeypatch.setattr("osai.calibration.train_gradient_gguf", rising)
    with pytest.raises(ConfigurationError, match="could not verify"):
        calibrate_training(model.path, model, _data(tmp_path), benchmark, engine=Engine.LLAMA_CPP)


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


def test_calibration_tests_faster_rate_after_small_verified_decline(monkeypatch, tmp_path: Path):
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
    assert len(attempts) == 2
    assert attempts[1] > attempts[0]
    assert result.learning_rate == attempts[1]


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
