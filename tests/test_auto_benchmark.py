from pathlib import Path

import pytest

from osai.auto_benchmark import benchmark_auto_settings
from osai.config import ModelFormat
from osai.errors import ConfigurationError
from osai.formats import ModelInspection, QuantizationSpec
from osai.hardware import Accelerator, Engine


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


def test_benchmark_downgrades_failed_profile_and_reuses_cache(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CPU)
    calls = []

    def probe(_binary, _model, settings, _accelerator, _device, _adapter):
        calls.append(settings.profile)
        if settings.profile == "maximum":
            raise RuntimeError("out of memory")

    monkeypatch.setattr("osai.auto_benchmark._probe_llama", probe)
    selected = benchmark_auto_settings(model, engine=Engine.LLAMA_CPP)
    assert selected.settings.profile == "performance"
    assert calls == ["maximum", "performance"]

    cached = benchmark_auto_settings(model, engine=Engine.LLAMA_CPP)
    assert cached.cached is True
    assert cached.settings == selected.settings
    assert calls == ["maximum", "performance"]


def test_benchmark_rebuilds_stale_trainer_before_probing(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    rebuilt = []
    monkeypatch.setattr(
        "osai.auto_benchmark.llama_binary",
        lambda name: None if name == "llama-finetune" and not rebuilt else binary,
    )
    monkeypatch.setattr("osai.backends.llama_cpp.ensure_runtime_accelerator", rebuilt.append)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CPU)
    monkeypatch.setattr("osai.auto_benchmark._probe_llama", lambda *_args: None)

    benchmark_auto_settings(model, engine=Engine.LLAMA_CPP, accelerator="cpu", force=True)
    assert rebuilt == ["cpu"]


def test_benchmark_probes_cuda_model_shards_together(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CUDA)
    monkeypatch.setattr(
        "osai.auto_benchmark.available_llama_devices",
        lambda _binary, _accelerator, **_kwargs: ("CUDA0", "CUDA1"),
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, _settings, _accelerator, device, _adapter: probed.append(device),
    )

    selected = benchmark_auto_settings(
        model, engine=Engine.LLAMA_CPP, accelerator="cuda", multi_gpu="on", force=True
    )

    assert selected.devices == ("CUDA0", "CUDA1")
    assert probed == ["CUDA0,CUDA1"]


def test_benchmark_rejects_integrated_vulkan_device_when_discrete_exists(
    monkeypatch, tmp_path: Path
):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr(
        "osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.VULKAN
    )
    monkeypatch.setattr(
        "osai.auto_benchmark.available_llama_devices",
        lambda _binary, _accelerator: ("Vulkan1", "Vulkan2"),
    )

    with pytest.raises(ConfigurationError, match="unavailable"):
        benchmark_auto_settings(
            model,
            engine=Engine.LLAMA_CPP,
            accelerator="vulkan",
            multi_gpu="on",
            devices=("Vulkan0", "Vulkan1"),
        )


def test_benchmark_reserves_gpu_memory_for_backward_pass(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CUDA)
    monkeypatch.setattr(
        "osai.auto_benchmark.available_llama_devices",
        lambda _binary, _accelerator, **_kwargs: ("CUDA0",),
    )
    monkeypatch.setattr(
        "osai.auto_benchmark.llama_device_free_bytes",
        lambda _binary, _accelerator: {"CUDA0": 6 * 1024**3},
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, settings, _accelerator, _device, _adapter: probed.append(
            settings.profile
        ),
    )

    selected = benchmark_auto_settings(model, engine=Engine.LLAMA_CPP, force=True)

    assert selected.settings.profile == "balanced"
    assert selected.settings.max_seq_length == 512
    assert probed == ["balanced"]


def test_two_12_gib_gpus_select_more_context_with_model_sharding(monkeypatch, tmp_path):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CUDA)
    monkeypatch.setattr(
        "osai.auto_benchmark.available_llama_devices",
        lambda _binary, _accelerator, **_kwargs: ("CUDA0", "CUDA1"),
    )
    monkeypatch.setattr(
        "osai.auto_benchmark.llama_device_free_bytes",
        lambda _binary, _accelerator: {
            "CUDA0": 11 * 1024**3,
            "CUDA1": 11 * 1024**3,
        },
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, settings, _accelerator, device, _adapter: probed.append(
            (settings.max_seq_length, device)
        ),
    )

    selected = benchmark_auto_settings(model, engine=Engine.LLAMA_CPP, multi_gpu="on")

    assert selected.settings.profile == "maximum"
    assert selected.settings.max_seq_length == 2048
    assert selected.settings.gguf_batch_size == 1
    assert probed == [(2048, "CUDA0,CUDA1")]


def test_two_3_gib_gpus_can_fit_a_sharded_compact_model(monkeypatch, tmp_path):
    model = _model(tmp_path)
    model = ModelInspection(
        format=model.format,
        path=model.path,
        architecture=model.architecture,
        quantization=model.quantization,
        size_bytes=int(2.5 * 1024**3),
        shards=model.shards,
        block_count=model.block_count,
        context_length=model.context_length,
    )
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CUDA)
    monkeypatch.setattr(
        "osai.auto_benchmark.available_llama_devices",
        lambda _binary, _accelerator, **_kwargs: ("CUDA0", "CUDA1"),
    )
    monkeypatch.setattr(
        "osai.auto_benchmark.llama_device_free_bytes",
        lambda _binary, _accelerator: {
            "CUDA0": 3 * 1024**3,
            "CUDA1": 3 * 1024**3,
        },
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, _settings, _accelerator, device, _adapter: probed.append(device),
    )

    selected = benchmark_auto_settings(model, engine=Engine.LLAMA_CPP, multi_gpu="on")

    assert selected.settings.profile == "compact"
    assert probed == ["CUDA0,CUDA1"]


def test_full_context_is_probed_across_gpu_shards_with_smaller_memory_settings(
    monkeypatch, tmp_path: Path
):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CUDA)
    monkeypatch.setattr(
        "osai.auto_benchmark.available_llama_devices",
        lambda _binary, _accelerator, **_kwargs: ("CUDA0", "CUDA1"),
    )
    monkeypatch.setattr(
        "osai.auto_benchmark.llama_device_free_bytes",
        lambda _binary, _accelerator: {"CUDA0": 11 * 1024**3, "CUDA1": 11 * 1024**3},
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, settings, _accelerator, device, _adapter: probed.append(
            (settings, device)
        ),
    )
    result = benchmark_auto_settings(
        model, engine=Engine.LLAMA_CPP, multi_gpu="on", required_context=4096
    )
    assert result.settings.max_seq_length == 4096
    assert result.settings.batch_size == 2
    assert result.settings.gguf_batch_size == 1
    assert result.settings.rank <= 8
    assert [device for _, device in probed] == ["CUDA0,CUDA1"]
    assert all(settings.max_seq_length == 4096 for settings, _ in probed)
    cached = benchmark_auto_settings(
        model, engine=Engine.LLAMA_CPP, multi_gpu="on", required_context=4096
    )
    assert cached.cached is True
    assert cached.settings == result.settings
    assert len(probed) == 1


def test_successful_benchmark_survives_read_only_cache(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CPU)
    monkeypatch.setattr("osai.auto_benchmark._probe_llama", lambda *_args: None)

    def deny_cache_write(*_args):
        raise PermissionError("cache is read-only")

    monkeypatch.setattr("osai.auto_benchmark.os.replace", deny_cache_write)

    selected = benchmark_auto_settings(model, engine=Engine.LLAMA_CPP, force=True)

    assert selected.settings.profile == "maximum"
