"""A single replaceable adapter snapshot for a running MLX trainer."""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path


class LatestCheckpoint:
    def __init__(self) -> None:
        self.output = Path(os.environ["OSAI_CHECKPOINT_OUTPUT"])
        self.request = Path(os.environ["OSAI_CHECKPOINT_REQUEST"])
        self.ack = Path(os.environ["OSAI_CHECKPOINT_ACK"])
        try:
            self.generation = self.ack.read_text(encoding="utf-8").strip()
        except OSError:
            self.generation = ""
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def _pending_request(self) -> str:
        try:
            token = (
                self.request.read_text(encoding="utf-8").strip() if self.request.is_file() else ""
            )
        except OSError:
            token = ""
        if len(token) > 128:
            token = ""
        return token if token != self.generation else ""

    def due(self, step: int, every: int) -> tuple[bool, str]:
        token = self._pending_request()
        return step % every == 0 or bool(token), token

    def save(self, adapter: Path, writer: Callable[[Path], None], generation: str = "") -> None:
        generation = generation or self._pending_request()
        adapter.parent.mkdir(parents=True, exist_ok=True)
        pending = adapter.with_name(adapter.stem + ".pending" + adapter.suffix)
        writer(pending)
        os.replace(pending, adapter)
        latest_pending = self.output.with_name(self.output.name + ".pending")
        shutil.copyfile(adapter, latest_pending)
        os.replace(latest_pending, self.output)
        config = adapter.parent / "adapter_config.json"
        if config.is_file():
            config_pending = self.output.parent / "adapter_config.json.pending"
            shutil.copyfile(config, config_pending)
            os.replace(config_pending, self.output.parent / "adapter_config.json")
        if generation:
            ack_pending = self.ack.with_name(self.ack.name + ".pending")
            ack_pending.write_text(generation + "\n", encoding="utf-8")
            os.replace(ack_pending, self.ack)
            self.generation = generation
        print(
            f"osai: checkpoint saved path={self.output} generation={generation or 'auto'}",
            flush=True,
        )


class CheckpointPublisher:
    """Keep one reusable fusion bundle in step with the latest adapter."""

    def __init__(
        self,
        *,
        kind: str,
        model: Path,
        latest: Path,
        shards: tuple[Path, ...] = (),
    ) -> None:
        if kind not in {"gguf", "mlx"}:
            raise ValueError(f"unsupported checkpoint model format: {kind}")
        self.kind = kind
        self.model = model
        self.latest = latest
        self.shards = shards
        self.destination = latest.parent.parent / "merged-model" / kind
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="osai-checkpoint-publisher")
        self.signature: tuple[int, int] | None = None

    def __enter__(self) -> CheckpointPublisher:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_event.set()
        self.thread.join()

    def _run(self) -> None:
        while True:
            self._publish_if_new()
            if self.stop_event.wait(1):
                self._publish_if_new()
                return

    def _publish_if_new(self) -> None:
        try:
            stat = self.latest.stat()
        except OSError:
            return
        signature = (stat.st_size, stat.st_mtime_ns)
        if not stat.st_size or signature == self.signature:
            return
        if self.kind == "mlx" and not (self.latest.parent / "adapter_config.json").is_file():
            return
        try:
            self._publish()
            print(f"osai: checkpoint model ready path={self.destination}", flush=True)
        except Exception as exc:
            print(f"osai: checkpoint model failed reason={exc}", flush=True)
        self.signature = signature

    def _publish(self) -> None:
        from .fusion import (
            GGUF_EMBEDDED_ADAPTER,
            MLX_EMBEDDED_ADAPTER,
            create_gguf_fusion_bundle,
            create_mlx_fusion_bundle,
        )

        if not self.destination.exists():
            stage = self.destination.parent / f".{self.kind}-pending"
            if stage.exists():
                shutil.rmtree(stage)
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            base_bytes = (
                sum(shard.stat().st_size for shard in self.shards)
                if self.kind == "gguf"
                else sum(item.stat().st_size for item in self.model.rglob("*") if item.is_file())
            )
            needed = base_bytes + self.latest.stat().st_size * 3 + 1024**3
            if shutil.disk_usage(stage.parent).free < needed:
                raise OSError("not enough free disk space for the checkpoint model")
            if self.kind == "gguf":
                create_gguf_fusion_bundle(self.model, self.shards, self.latest, stage)
            else:
                create_mlx_fusion_bundle(self.model, self.latest.parent, stage)
            os.replace(stage, self.destination)
            return

        target = (
            self.destination / GGUF_EMBEDDED_ADAPTER
            if self.kind == "gguf"
            else self.destination / MLX_EMBEDDED_ADAPTER / "adapters.safetensors"
        )
        if not target.is_file():
            raise OSError(f"checkpoint model is incomplete: {target}")
        pending = target.with_name(target.name + ".pending")
        shutil.copyfile(self.latest, pending)
        os.replace(pending, target)
        if self.kind == "mlx":
            config = self.latest.parent / "adapter_config.json"
            target_config = target.parent / "adapter_config.json"
            if config.is_file():
                pending_config = target_config.with_name(target_config.name + ".pending")
                shutil.copyfile(config, pending_config)
                os.replace(pending_config, target_config)
