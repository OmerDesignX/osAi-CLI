from pathlib import Path

import pytest

from osai.auto_benchmark import _native_failure, _probe_timeout, benchmark_auto_settings
from osai.config import ModelFormat
from osai.errors import ConfigurationError
from osai.formats import ModelInspection, QuantizationSpec
from osai.hardware import Accelerator, Engine
from osai.multi_gpu import LlamaDeviceInventory


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


def _inventory(devices, free=None, total=None):
    free = free or {}
    return LlamaDeviceInventory(tuple(devices), free, total or free)


def test_large_full_context_probes_allow_slow_discrete_metal_allocation():
    assert _probe_timeout(2048) == 180
    assert _probe_timeout(61_588) == 600
    assert _probe_timeout(1_000_000) == 900


def test_benchmark_downgrades_failed_profile_and_reuses_cache(monkeypatch, tmp_path: Path):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CPU)
    calls = []

    def probe(
        _binary,
        _model,
        settings,
        _accelerator,
        _device,
        _adapter,
        _tensor_split,
        _main_gpu,
    ):
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


@pytest.mark.parametrize("count", [2, 3, 4, 8])
def test_benchmark_probes_all_cuda_model_shards_together(monkeypatch, tmp_path: Path, count):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.CUDA)
    devices = tuple(f"CUDA{index}" for index in range(count))
    monkeypatch.setattr(
        "osai.auto_benchmark.llama_device_inventory_details",
        lambda _binary, _accelerator, **_kwargs: _inventory(devices),
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, _settings, _accelerator, device, _adapter, _split, _main: (
            probed.append(device)
        ),
    )

    selected = benchmark_auto_settings(
        model, engine=Engine.LLAMA_CPP, accelerator="cuda", multi_gpu="on", force=True
    )

    assert selected.devices == devices
    assert probed == [",".join(devices)]


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
        "osai.auto_benchmark.llama_device_inventory_details",
        lambda _binary, _accelerator, **_kwargs: _inventory(("Vulkan1", "Vulkan2")),
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
        "osai.auto_benchmark.llama_device_inventory_details",
        lambda _binary, _accelerator, **_kwargs: _inventory(("CUDA0",), {"CUDA0": 6 * 1024**3}),
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, settings, _accelerator, _device, _adapter, _split, _main: (
            probed.append(settings.profile)
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
        "osai.auto_benchmark.llama_device_inventory_details",
        lambda _binary, _accelerator, **_kwargs: _inventory(
            ("CUDA0", "CUDA1"),
            {"CUDA0": 11 * 1024**3, "CUDA1": 11 * 1024**3},
        ),
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, settings, _accelerator, device, _adapter, _split, _main: (
            probed.append((settings.max_seq_length, device))
        ),
    )

    selected = benchmark_auto_settings(model, engine=Engine.LLAMA_CPP, multi_gpu="on")

    assert selected.settings.profile == "maximum"
    assert selected.settings.max_seq_length == 2048
    assert selected.settings.gguf_batch_size == 1
    assert probed == [(2048, "CUDA0,CUDA1")]


def test_asymmetric_metal_gpus_use_vram_weighted_probe_and_largest_main(monkeypatch, tmp_path):
    model = _model(tmp_path)
    binary = tmp_path / "llama-completion"
    binary.write_bytes(b"binary")
    monkeypatch.setattr("osai.auto_settings.physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr("osai.auto_benchmark.llama_binary", lambda _name: binary)
    monkeypatch.setattr("osai.auto_benchmark.llama_runtime_build", lambda: tmp_path / "native")
    monkeypatch.setattr("osai.auto_benchmark.select_llama_accelerator", lambda _: Accelerator.METAL)
    monkeypatch.setattr(
        "osai.auto_benchmark.llama_device_inventory_details",
        lambda _binary, _accelerator, **_kwargs: _inventory(
            ("MTL0", "MTL1"),
            {"MTL0": int(3.5 * 1024**3), "MTL1": 3 * 1024**3},
            {"MTL0": 4 * 1024**3, "MTL1": 8 * 1024**3},
        ),
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, _settings, _accelerator, device, _adapter, split, main: (
            probed.append((device, split, main))
        ),
    )

    selected = benchmark_auto_settings(
        model,
        engine=Engine.LLAMA_CPP,
        accelerator="metal",
        multi_gpu="on",
        force=True,
    )

    assert selected.devices == ("MTL0", "MTL1")
    assert selected.main_gpu == 1
    assert selected.tensor_split[0] > selected.tensor_split[1]
    assert probed == [("MTL0,MTL1", selected.tensor_split, 1)]


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
        "osai.auto_benchmark.llama_device_inventory_details",
        lambda _binary, _accelerator, **_kwargs: _inventory(
            ("CUDA0", "CUDA1"),
            {"CUDA0": 3 * 1024**3, "CUDA1": 3 * 1024**3},
        ),
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, _settings, _accelerator, device, _adapter, _split, _main: (
            probed.append(device)
        ),
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
        "osai.auto_benchmark.llama_device_inventory_details",
        lambda _binary, _accelerator, **_kwargs: _inventory(
            ("CUDA0", "CUDA1"),
            {"CUDA0": 11 * 1024**3, "CUDA1": 11 * 1024**3},
        ),
    )
    probed = []
    monkeypatch.setattr(
        "osai.auto_benchmark._probe_llama",
        lambda _binary, _model, settings, _accelerator, device, _adapter, _split, _main: (
            probed.append((settings, device))
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


def test_native_failure_preserves_assertion_and_termination_signal():
    output = "\n".join(
        [
            "apply_selection_policy_once: prefer use of removable GPUs",
            "llama backend init",
            "GGML_ASSERT(buf_src) failed at ggml-metal-device.m:2328",
        ]
    )

    detail = _native_failure(output, returncode=-6)

    assert "SIGABRT" in detail
    assert "GGML_ASSERT(buf_src)" in detail
