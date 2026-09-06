"""QLoRA backend using the vendored MLX LM implementation."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import TrainingConfig
from ..errors import ConfigurationError, DependencyError, TrainingError
from ..hardware import Accelerator, macos_version_at_least
from ..io import atomic_json
from ..offline import offline_environment
from ..paths import mlx_lm_root
from ..process import run_logged
from ..session import SessionLayout

_LOSS_RE = re.compile(r"(?:Train|Val) loss\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_TABLE_LOSS_RE = re.compile(r"^\s*\d+\s+([0-9]+(?:\.[0-9]+)?)\s", re.MULTILINE)
_TEST_LOSS_RE = re.compile(r"Test loss\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True, slots=True)
class MlxTrainingResult:
    adapter_dir: Path
    adapter_file: Path
    log_file: Path
    elapsed_seconds: float
    losses: tuple[float, ...]
    test_loss: float | None = None


def _resolve_distributed_workers(
    config: TrainingConfig, report: dict[str, str | bool | int]
) -> int:
    """Choose safe local MLX data-parallel workers.

    MLX exposes NCCL multi-GPU on Linux. Apple silicon presents its unified GPU
    as one device, so launching several Metal workers would only duplicate the
    model on the same physical GPU.
    """

    accelerator = str(report.get("accelerator", "cpu"))
    available = int(report.get("gpu_count", 0))
    if config.multi_gpu == "off":
        return 1
    if accelerator != "cuda":
        if config.multi_gpu == "on":
            raise ConfigurationError(
                "MLX multi-GPU requires Linux CUDA/NCCL; Metal exposes one unified GPU"
            )
        return 1
    requested = config.distributed_workers or available
    if config.multi_gpu == "on" and (available < 2 or requested < 2):
        raise ConfigurationError("MLX multi-GPU was required but fewer than two GPUs exist")
    workers = min(requested, available)
    while workers > 1 and config.batch_size % workers:
        workers -= 1
    if config.multi_gpu == "on" and workers < 2:
        raise ConfigurationError(
            "MLX global batch size must be divisible by at least two CUDA workers"
        )
    return max(1, workers)


class MlxBackend:
    def __init__(
        self,
        python: str | Path | None = None,
        accelerator: str | Accelerator = Accelerator.AUTO,
    ):
        self.python = Path(python) if python is not None else Path(sys.executable)
        try:
            self.accelerator = (
                accelerator if isinstance(accelerator, Accelerator) else Accelerator(accelerator)
            )
        except ValueError as exc:
            raise ConfigurationError("MLX accelerator must be auto, metal, cuda, or cpu") from exc
        if self.accelerator in {Accelerator.MPS, Accelerator.VULKAN}:
            raise ConfigurationError("MLX does not use the MPS or Vulkan backend")

    def environment(self) -> dict[str, str]:
        env = offline_environment(os.environ)
        vendor = mlx_lm_root()
        python_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(vendor) + (os.pathsep + python_path if python_path else "")
        env.setdefault("TOKENIZERS_PARALLELISM", "true")
        env["OSAI_MLX_ACCELERATOR"] = self.accelerator.value
        return env

    def preflight(self) -> dict[str, str | bool | int]:
        system = platform.system()
        machine = platform.machine().lower()
        if system == "Darwin" and machine not in {"arm64", "aarch64"}:
            raise DependencyError(
                "MLX on macOS requires Apple silicon; use llama.cpp on Intel Macs"
            )
        if system == "Darwin" and not macos_version_at_least(14):
            raise DependencyError(
                "the bundled MLX revision requires macOS 14 or newer; use llama.cpp on "
                "macOS Monterey or Ventura"
            )
        if system not in {"Darwin", "Linux"}:
            raise DependencyError("MLX training is supported on Apple-silicon macOS and Linux")
        if not mlx_lm_root().is_dir():
            raise DependencyError(f"vendored MLX LM source not found: {mlx_lm_root()}")
        try:
            completed = subprocess.run(
                [str(self.python), "-m", "osai._offline_runner", "probe-mlx"],
                env=self.environment(),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except OSError as exc:
            raise DependencyError(f"cannot run Python at {self.python}: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise DependencyError(
                "MLX runtime is not usable in this Python environment. Install the project's "
                f"mlx extra in a clean virtual environment. Detail: {detail}"
            )
        try:
            report = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise DependencyError("MLX probe returned invalid diagnostics") from exc
        if not report.get("usable"):
            detail = report.get("reason") or "requested MLX backend is unavailable"
            raise DependencyError(str(detail))
        return report

    def train(self, config: TrainingConfig) -> MlxTrainingResult:
        report = self.preflight()
        layout = SessionLayout.at(config.output)
        layout.create()
        internal = layout.work
        adapter_dir = layout.adapters / "mlx"
        log_path = layout.logs / "train.log"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        log_path.unlink(missing_ok=True)

        mlx_config = {
            "model": str(config.training_model.resolve()),
            "train": True,
            "test": False,
            "fine_tune_type": "lora",
            "data": str(config.data.resolve()),
            "optimizer": "adamw" if config.optimizer == "auto" else config.optimizer,
            "batch_size": config.batch_size,
            "iters": config.iterations,
            "val_batches": config.val_batches,
            "learning_rate": config.learning_rate,
            "num_layers": config.num_layers,
            "max_seq_length": config.max_seq_length,
            "grad_checkpoint": config.grad_checkpoint,
            "grad_accumulation_steps": config.grad_accumulation_steps,
            "mask_prompt": config.mask_prompt,
            "adapter_path": str(adapter_dir.resolve()),
            "save_every": min(config.save_every, config.iterations),
            "steps_per_report": config.steps_per_report,
            "steps_per_eval": config.steps_per_eval,
            "seed": config.seed,
            "lora_parameters": {
                "rank": config.rank,
                "scale": config.scale,
                "dropout": config.dropout,
                "keys": list(config.target_modules),
            },
        }
        config_path = internal / "mlx_lora_config.yaml"
        # JSON is a strict subset of YAML and avoids another serialization surface.
        atomic_json(config_path, mlx_config)
        command = [
                str(self.python),
                "-m",
                "osai._offline_runner",
                "mlx_lm",
                "lora",
                "--config",
                str(config_path),
            ]
        workers = _resolve_distributed_workers(config, report)
        if workers > 1:
            command = [
                str(self.python),
                "-m",
                "mlx._distributed_utils.launch",
                "-n",
                str(workers),
                "--backend",
                "nccl",
                "--hosts",
                "127.0.0.1",
                "--python",
                str(self.python),
                "--",
                *command[1:],
            ]
        result = run_logged(
            command,
            log_path=log_path,
            env=self.environment(),
        )
        adapter_file = adapter_dir / "adapters.safetensors"
        adapter_config = adapter_dir / "adapter_config.json"
        if not adapter_file.is_file() or adapter_file.stat().st_size == 0:
            raise TrainingError(f"MLX LM completed without a non-empty adapter: {adapter_file}")
        if not adapter_config.is_file():
            raise TrainingError(f"MLX LM did not write adapter metadata: {adapter_config}")
        log_text = _ANSI_RE.sub("", log_path.read_text(encoding="utf-8"))
        matches = list(_LOSS_RE.finditer(log_text)) or list(_TABLE_LOSS_RE.finditer(log_text))
        losses = tuple(float(match.group(1)) for match in matches)
        if not losses:
            raise TrainingError(
                f"training completed without a finite reported loss; see {log_path}"
            )
        if not all(math.isfinite(loss) for loss in losses):
            raise TrainingError(f"training reported a non-finite loss; see {log_path}")
        test_loss = None
        if (config.data / "test.jsonl").is_file():
            test_loss = self.evaluate(config, adapter_dir)
        return MlxTrainingResult(
            adapter_dir=adapter_dir,
            adapter_file=adapter_file,
            log_file=log_path,
            elapsed_seconds=result.elapsed_seconds,
            losses=losses,
            test_loss=test_loss,
        )

    def evaluate(self, config: TrainingConfig, adapter_dir: Path) -> float:
        layout = SessionLayout.at(config.output)
        internal = layout.work
        eval_config = {
            "model": str(config.training_model.resolve()),
            "train": False,
            "test": True,
            "data": str(config.data.resolve()),
            "batch_size": config.batch_size,
            "test_batches": config.val_batches,
            "max_seq_length": config.max_seq_length,
            "adapter_path": str(adapter_dir.resolve()),
            "mask_prompt": config.mask_prompt,
            "seed": config.seed,
        }
        config_path = internal / "mlx_eval_config.yaml"
        atomic_json(config_path, eval_config)
        log_path = layout.logs / "evaluate.log"
        log_path.unlink(missing_ok=True)
        run_logged(
            [
                str(self.python),
                "-m",
                "osai._offline_runner",
                "mlx_lm",
                "lora",
                "--config",
                str(config_path),
            ],
            log_path=log_path,
            env=self.environment(),
        )
        match = _TEST_LOSS_RE.search(log_path.read_text(encoding="utf-8"))
        if match is None:
            raise TrainingError(f"evaluation completed without a finite test loss; see {log_path}")
        loss = float(match.group(1))
        if not math.isfinite(loss):
            raise TrainingError(f"evaluation reported a non-finite test loss; see {log_path}")
        return loss

    def verify_fusion(
        self,
        base: Path,
        adapter_dir: Path,
        merged: Path,
        *,
        log_path: Path,
    ) -> None:
        self.preflight()
        run_logged(
            [
                str(self.python),
                "-m",
                "osai._offline_runner",
                "verify-mlx-fusion",
                str(base.resolve()),
                str(adapter_dir.resolve()),
                str(merged.resolve()),
            ],
            log_path=log_path,
            env=self.environment(),
        )

    def validate_model(self, model: Path, *, log_path: Path) -> None:
        self.preflight()
        run_logged(
            [
                str(self.python),
                "-m",
                "osai._offline_runner",
                "mlx_lm",
                "generate",
                "--model",
                str(model.resolve()),
                "--prompt",
                "Hello",
                "--max-tokens",
                "1",
                "--temp",
                "0",
            ],
            log_path=log_path,
            env=self.environment(),
        )
