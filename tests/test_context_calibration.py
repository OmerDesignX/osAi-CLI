"""Automatic fitting must size context from the selected data first."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from osai import cli
from osai.auto_benchmark import BenchmarkResult
from osai.auto_settings import select_auto_settings
from osai.config import ModelFormat
from osai.errors import ConfigurationError
from osai.formats import ModelInspection, QuantizationSpec
from osai.hardware import Engine


def test_auto_train_defaults_to_full_context_and_allows_windowing(monkeypatch):
    selected = []

    def train(args):
        selected.append(args.full_content_context)
        return {}

    monkeypatch.setattr(cli, "_fine_tune", train)
    monkeypatch.setattr(cli, "_print_json", lambda _: None)
    for options in (
        ["--auto-settings"],
        ["--auto-settings", "--no-full-content-context"],
        ["--no-auto-settings"],
    ):
        args = cli.build_parser().parse_args(["train", "--tier", "small", *options])
        cli._train(args)
    assert selected == [True, False, False]


def test_calibration_scans_and_checks_model_limit_before_hardware_pilot(
    monkeypatch, tmp_path: Path, capsys
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    model = ModelInspection(
        format=ModelFormat.GGUF,
        path=model_path,
        architecture="qwen35",
        quantization=QuantizationSpec("Q4_K_M"),
        size_bytes=2 * 1024**3,
        shards=(model_path,),
        block_count=32,
        context_length=4096,
    )
    settings = select_auto_settings(model, engine=Engine.LLAMA_CPP, memory_bytes=64 * 1024**3)
    events = []
    monkeypatch.setattr(
        cli,
        "_resolve_cli_model",
        lambda _: SimpleNamespace(engine=Engine.LLAMA_CPP, model=model_path, companion_mlx=None),
    )
    monkeypatch.setattr(cli, "inspect_model", lambda *_: model)

    def scan(*_, **__):
        events.append("scan")
        return {"context": 3072, "largest_tokens": 3008}

    def benchmark(*_, **__):
        events.append("hardware")
        return BenchmarkResult(settings, "llama.cpp", "cpu", (), 0.1)

    def pilot(_path, _model, _data, benchmark_result, **kwargs):
        events.append("pilot")
        print("native pilot progress")
        assert benchmark_result.settings.max_seq_length == 3072
        assert kwargs["require_full_context"] is True
        return SimpleNamespace(as_dict=lambda: {"context": 3072})

    monkeypatch.setattr(cli, "largest_training_context", scan)
    monkeypatch.setattr(cli, "benchmark_auto_settings", benchmark)
    monkeypatch.setattr(cli, "calibrate_training", pilot)
    payloads = []
    monkeypatch.setattr(cli, "_print_json", payloads.append)
    args = cli.build_parser().parse_args(
        ["calibrate", "--custom", str(model_path), "--data", str(tmp_path / "data.jsonl")]
    )
    cli._calibrate(args)
    assert events == ["scan", "hardware", "pilot"]
    assert payloads[-1]["required_context"] == 3072
    output = capsys.readouterr()
    assert "native pilot progress" in output.err
    assert "native pilot progress" not in output.out

    events.clear()
    monkeypatch.setattr(
        cli,
        "largest_training_context",
        lambda *_, **__: {"context": 8192, "largest_tokens": 8128},
    )
    with pytest.raises(ConfigurationError, match="choose Windowing"):
        cli._calibrate(args)
    assert events == []


def test_training_reuses_verified_full_context_without_a_second_token_scan(
    monkeypatch, tmp_path: Path
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text":"example"}\n', encoding="utf-8")
    model = ModelInspection(
        format=ModelFormat.GGUF,
        path=model_path,
        architecture="qwen35",
        quantization=QuantizationSpec("Q4_K_M"),
        size_bytes=2 * 1024**3,
        shards=(model_path,),
        block_count=32,
        context_length=4096,
    )
    selection = SimpleNamespace(
        engine=Engine.LLAMA_CPP,
        model=model_path,
        companion_mlx=None,
        entry=SimpleNamespace(gguf=model_path),
    )
    monkeypatch.setattr(cli, "_resolve_cli_model", lambda _: selection)
    monkeypatch.setattr(cli, "inspect_model", lambda *_: model)
    monkeypatch.setattr(
        cli,
        "largest_training_context",
        lambda *_args, **_kwargs: pytest.fail("the dataset was scanned twice"),
    )
    measured = []

    def resolve(config, _args, _engine, required_context=None):
        measured.append(required_context)
        return config

    monkeypatch.setattr(cli, "_resolve_training_settings", resolve)
    monkeypatch.setattr(cli, "prepare_dataset_source", lambda source, _destination: source)

    class ReachedTrainingData(Exception):
        pass

    monkeypatch.setattr(
        cli, "validate_dataset", lambda _source: (_ for _ in ()).throw(ReachedTrainingData)
    )
    args = cli.build_parser().parse_args(
        [
            "train",
            "--custom",
            str(model_path),
            "--data",
            str(data_path),
            "--full-content-context",
            "--calibrated-context",
            "3072",
            "--max-seq-length",
            "3072",
            "--no-auto-settings",
        ]
    )
    with pytest.raises(ReachedTrainingData):
        cli._fine_tune(args)
    assert measured == [3072]

    args.max_seq_length = 2048
    with pytest.raises(ConfigurationError, match="must cover"):
        cli._fine_tune(args)
