"""Shared helpers for gradient training against packed GGUF models."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ..dataset import normalize_sft_example, normalized_record_text
from ..errors import ConfigurationError, DependencyError, TrainingError, VerificationError
from ..formats import ModelInspection
from ..hardware import Accelerator
from ..multi_gpu import DeviceSettings, llama_device_arguments
from ..offline import offline_environment
from ..paths import llama_cpp_root
from ..process import run_logged

_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)")
_LOSS_ROW_RE = re.compile(
    r"^\s*\d+\s+[0-9.eE+-]+\s+([0-9.eE+-]+)\s+[0-9.eE+-]+\s*$", re.MULTILINE
)
_MODULE_NAMES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


class AdapterSettings(DeviceSettings, Protocol):
    rank: int
    num_layers: int
    seed: int
    target_modules: tuple[str, ...]


def _initialize_parameters(base: ModelInspection, settings: AdapterSettings, np):
    if base.block_count is None:
        raise VerificationError("GGUF block count is required to select LoRA layers")
    if settings.num_layers > base.block_count:
        raise ConfigurationError("num_layers exceeds the GGUF block count")
    wanted = {
        f"blk.{block}.{_MODULE_NAMES[module]}.weight"
        for block in range(base.block_count - settings.num_layers, base.block_count)
        for module in settings.target_modules
    }
    shapes: dict[str, tuple[int, int]] = {}
    gguf = _import_gguf()
    for shard in base.shards:
        reader = gguf.GGUFReader(shard)
        for tensor in reader.tensors:
            if tensor.name in wanted:
                dimensions = tuple(int(value) for value in tensor.shape)
                if len(dimensions) != 2:
                    raise VerificationError(f"LoRA target is not a matrix: {tensor.name}")
                shapes[tensor.name] = (dimensions[0], dimensions[1])
    missing = sorted(wanted - set(shapes))
    if missing:
        raise VerificationError("GGUF is missing requested LoRA tensors: " + ", ".join(missing))
    rng = np.random.default_rng(settings.seed)
    parameters = {}
    for name, (input_size, output_size) in sorted(shapes.items()):
        parameters[f"{name}.lora_a"] = rng.normal(
            0.0, 1.0 / math.sqrt(input_size), size=(settings.rank, input_size)
        ).astype(np.float32)
        parameters[f"{name}.lora_b"] = np.zeros(
            (output_size, settings.rank), dtype=np.float32
        )
    return parameters


def _write_adapter(path: Path, architecture: str, parameters, alpha: float, np) -> None:
    gguf = _import_gguf()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        writer = gguf.GGUFWriter(temporary, architecture)
        writer.add_name(path.stem)
        writer.add_type(gguf.GGUFType.ADAPTER)
        writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
        writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
        for name, value in sorted(parameters.items()):
            writer.add_tensor(name, np.asarray(value, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _evaluate_loss(
    binary: Path,
    model: Path,
    adapter: Path,
    corpus: Path,
    context: int,
    accelerator: Accelerator,
    log_path: Path,
    device_settings: DeviceSettings,
) -> tuple[float, Accelerator]:
    offset = log_path.stat().st_size if log_path.exists() else 0
    command = [
        str(binary),
        "-m",
        str(model),
        "--lora",
        str(adapter),
        "-f",
        str(corpus),
        "-c",
        str(context),
        "-b",
        str(max(64, context * 2)),
        "-ub",
        str(max(64, context)),
        "--chunks",
        "1",
        "--ppl-output-type",
        "1",
        "--log-colors",
        "off",
    ]
    command.extend(llama_device_arguments(accelerator, device_settings))
    try:
        run_logged(command, log_path=log_path, env=offline_environment())
    except TrainingError:
        if accelerator is Accelerator.CPU:
            raise
        previous = accelerator
        accelerator = Accelerator.CPU
        gpu_arguments = llama_device_arguments(previous, device_settings)
        del command[-len(gpu_arguments) :]
        command.extend(llama_device_arguments(accelerator, device_settings))
        print(f"osai: {previous.value} execution failed; retrying with CPU")
        run_logged(command, log_path=log_path, env=offline_environment())
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        output = handle.read()
    return _parse_loss(output, log_path), accelerator


def _parse_loss(output: str, log_path: Path) -> float:
    loss_matches = _LOSS_ROW_RE.findall(output)
    if loss_matches:
        loss = float(loss_matches[-1])
    else:
        matches = _PPL_RE.findall(output)
        if not matches:
            raise TrainingError(f"llama.cpp did not report perplexity; see {log_path}")
        perplexity = float(matches[-1])
        if not math.isfinite(perplexity) or perplexity <= 0:
            raise TrainingError(f"llama.cpp reported invalid perplexity: {perplexity}")
        loss = math.log(perplexity)
    if not math.isfinite(loss):
        raise TrainingError(f"llama.cpp reported a non-finite loss; see {log_path}")
    return loss


def _write_corpus(
    source: Path,
    destination: Path,
    context: int,
    *,
    repeat_to_minimum: bool = True,
    record_separator: str = "\n\n",
) -> Path:
    sections: list[str] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"invalid JSON at {source}:{line_number}: {exc.msg}"
                ) from exc
            normalized = normalize_sft_example(record, source, line_number)
            sections.append(normalized_record_text(normalized.record))
    corpus = record_separator.join(sections).strip() + "\n"
    minimum_characters = context * 16
    if repeat_to_minimum and len(corpus) < minimum_characters:
        corpus = (corpus * (minimum_characters // max(1, len(corpus)) + 1))[
            : minimum_characters * 2
        ]
    destination.write_text(corpus, encoding="utf-8")
    return destination


def _verify_adapter(path: Path, architecture: str) -> None:
    gguf = _import_gguf()
    reader = gguf.GGUFReader(path)
    if reader.get_field("general.type").contents() != "adapter":
        raise VerificationError("generated GGUF does not identify as an adapter")
    if reader.get_field("adapter.type").contents() != "lora":
        raise VerificationError("generated GGUF is not a LoRA adapter")
    if reader.get_field("general.architecture").contents() != architecture:
        raise VerificationError("generated adapter architecture differs from its GGUF base")
    if not reader.tensors or len(reader.tensors) % 2:
        raise VerificationError("generated adapter has incomplete LoRA tensor pairs")


def _validate_output(output: Path, model: Path, data: Path) -> None:
    for label, protected in (("model", model.resolve()), ("dataset", data.resolve())):
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise ConfigurationError(f"output must not be inside the {label} path: {protected}")


def _import_gguf():
    path = llama_cpp_root() / "gguf-py"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        import gguf
    except ImportError as exc:
        raise DependencyError("GGUF training requires numpy and vendored gguf-py") from exc
    return gguf


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyError("GGUF training requires numpy") from exc
    return np


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
