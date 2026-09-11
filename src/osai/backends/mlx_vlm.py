"""Quantized-language LoRA training conditioned on local visual/audio inputs."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

from ..config import TrainingConfig
from ..dataset import prepare_mlx_vlm_dataset
from ..errors import DependencyError, TrainingError
from ..io import atomic_json
from ..paths import mlx_vlm_root
from ..process import run_logged
from ..session import SessionLayout
from .mlx import MlxBackend, MlxTrainingResult, _resolve_distributed_workers

_LOSS_RE = re.compile(r"(?:Train|Val) loss\s+([0-9]+(?:\.[0-9]+)?)", re.I)
_TEST_RE = re.compile(r"Test loss\s+([0-9]+(?:\.[0-9]+)?)", re.I)
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class MlxVlmBackend(MlxBackend):
    """MLX-VLM backprop with frozen quantized base and frozen media towers."""

    def environment(self) -> dict[str, str]:
        env = super().environment()
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(mlx_vlm_root()) + (
            os.pathsep + current if current else ""
        )
        return env

    def preflight(self) -> dict[str, str | bool | int]:
        report = super().preflight()
        if not mlx_vlm_root().is_dir():
            raise DependencyError(f"vendored MLX-VLM source not found: {mlx_vlm_root()}")
        completed = subprocess.run(
            [str(self.python), "-m", "osai._offline_runner", "probe-mlx-vlm"],
            env=self.environment(),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise DependencyError(
                "MLX-VLM media dependencies are unavailable in this local Python "
                f"environment. Re-run osAi setup. Detail: {detail}"
            )
        return report

    def train(self, config: TrainingConfig) -> MlxTrainingResult:
        report = self.preflight()
        layout = SessionLayout.at(config.output)
        layout.create()
        data_path = prepare_mlx_vlm_dataset(
            config.data,
            layout.work / "mlx-vlm-dataset",
            video_fps=config.video_fps,
            video_max_frames=config.video_max_frames,
        )
        adapter_dir = layout.adapters / "mlx"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(config.training_model.resolve()),
            "data": str(data_path),
            "adapter_path": str(adapter_dir.resolve()),
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
            "assistant_token_id": config.assistant_token_id,
            "save_every": min(config.save_every, config.iterations),
            "steps_per_report": config.steps_per_report,
            "steps_per_eval": config.steps_per_eval,
            "seed": config.seed,
            "rank": config.rank,
            "scale": config.scale,
            "dropout": config.dropout,
            "target_modules": list(config.target_modules),
            "image_resize_shape": (
                [config.image_width, config.image_height]
                if config.image_width is not None
                else None
            ),
        }
        config_path = layout.work / "mlx_vlm_config.json"
        atomic_json(config_path, payload)
        command = [
            str(self.python),
            "-m",
            "osai._offline_runner",
            "train-mlx-vlm",
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
        log_path = layout.logs / "train.log"
        log_path.unlink(missing_ok=True)
        result = run_logged(command, log_path=log_path, env=self.environment())
        adapter_file = adapter_dir / "adapters.safetensors"
        adapter_config = adapter_dir / "adapter_config.json"
        if not adapter_file.is_file() or adapter_file.stat().st_size == 0:
            raise TrainingError(f"MLX-VLM completed without an adapter: {adapter_file}")
        if not adapter_config.is_file():
            raise TrainingError(f"MLX-VLM did not write adapter metadata: {adapter_config}")
        log = _ANSI_RE.sub("", log_path.read_text(encoding="utf-8"))
        losses = tuple(float(match.group(1)) for match in _LOSS_RE.finditer(log))
        if not losses or not all(math.isfinite(loss) for loss in losses):
            raise TrainingError(f"VLM training did not report finite losses; see {log_path}")
        test_match = _TEST_RE.search(log)
        test_loss = float(test_match.group(1)) if test_match else None
        return MlxTrainingResult(
            adapter_dir=adapter_dir,
            adapter_file=adapter_file,
            log_file=log_path,
            elapsed_seconds=result.elapsed_seconds,
            losses=losses,
            test_loss=test_loss,
        )

    def verify_fusion(
        self,
        base: Path,
        adapter_dir: Path,
        merged: Path,
        *,
        log_path: Path,
    ) -> None:
        adapter_config = json.loads(
            (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        rank = int(adapter_config["rank"])
        payload = {
            "model": str(merged.resolve()),
            "rank": rank,
            "scale": float(adapter_config["alpha"]) / rank,
            "dropout": float(adapter_config.get("dropout", 0.0)),
            "num_layers": int(adapter_config.get("num_layers", 1)),
            "target_modules": adapter_config.get("keys") or ["mlp.down_proj"],
        }
        config_path = log_path.parent / "verify-mlx-vlm-fusion.json"
        atomic_json(config_path, payload)
        run_logged(
            [
                str(self.python),
                "-m",
                "osai._offline_runner",
                "verify-mlx-vlm-fusion",
                str(config_path),
            ],
            log_path=log_path,
            env=self.environment(),
        )
