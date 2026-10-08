"""A single replaceable adapter snapshot for a running MLX trainer."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
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
        multimodal: bool = False,
    ) -> None:
        if kind not in {"gguf", "mlx"}:
            raise ValueError(f"unsupported checkpoint model format: {kind}")
        self.kind = kind
        self.model = model
        self.latest = latest
        self.shards = shards
        self.multimodal = multimodal
        self.root = latest.parents[3]
        self.work = self.root / ".internal"
        self.destination = self.root / "outputs" / kind
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="osai-checkpoint-publisher")
        self.signature: tuple[int, int] | None = None
        self.failed_signature: tuple[int, int] | None = None
        self.failed_at = 0.0

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
        if signature == self.failed_signature and time.monotonic() - self.failed_at < 30:
            return
        if self.kind == "mlx" and not (self.latest.parent / "adapter_config.json").is_file():
            return
        try:
            self._publish()
            # A newer adapter may have been written while fusion was running.
            # Publish that one before acknowledging a checkpoint request.
            current = self.latest.stat()
            if (current.st_size, current.st_mtime_ns) != signature:
                return
            generation = self._generation()
            exported = self._manual_export(generation)
            print(
                f"osai: checkpoint model ready path={exported or self.destination} "
                f"generation={generation or 'auto'}",
                flush=True,
            )
        except Exception as exc:
            self.failed_signature = signature
            self.failed_at = time.monotonic()
            print(f"osai: checkpoint model failed reason={exc}", flush=True)
            return
        self.signature = signature
        self.failed_signature = None

    def _generation(self) -> str:
        try:
            return (self.latest.parent.parent / "last.ack").read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _publish(self) -> None:
        from .merged_export import export_gguf_weights, export_mlx_weights

        self.work.mkdir(parents=True, exist_ok=True)
        snapshot = self.work / f"checkpoint-export-{uuid.uuid4().hex}{self.latest.suffix}"
        shutil.copy2(self.latest, snapshot)
        snapshot_dir = snapshot.parent / f"{snapshot.stem}-adapter"
        if self.kind == "mlx":
            snapshot_dir.mkdir()
            shutil.copy2(snapshot, snapshot_dir / "adapters.safetensors")
            shutil.copy2(
                self.latest.parent / "adapter_config.json", snapshot_dir / "adapter_config.json"
            )
        try:
            self._publish_snapshot(snapshot, snapshot_dir, export_gguf_weights, export_mlx_weights)
        finally:
            snapshot.unlink(missing_ok=True)
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    def _publish_snapshot(
        self, snapshot, snapshot_dir, export_gguf_weights, export_mlx_weights
    ) -> None:
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
            base_copy_bytes = (
                base_bytes if os.name == "nt" and self.model.drive != self.destination.drive else 0
            )
            needed = base_bytes + base_copy_bytes + self.latest.stat().st_size * 3 + 128 * 1024**2
            if shutil.disk_usage(stage.parent).free < needed:
                raise OSError("not enough free disk space for the checkpoint model")
            if self.kind == "gguf":
                bundle = create_gguf_fusion_bundle(self.model, self.shards, snapshot, stage)
                export_gguf_weights(
                    bundle.model,
                    bundle.shards,
                    snapshot,
                    stage / "merged.gguf",
                    work=self.work,
                    log=self.root / "logs" / "checkpoint-merge.log",
                )
            else:
                create_mlx_fusion_bundle(self.model, snapshot_dir, stage)
                export_mlx_weights(
                    self.model,
                    snapshot_dir,
                    stage / "merged",
                    work=self.work,
                    log=self.root / "logs" / "checkpoint-merge.log",
                    multimodal=self.multimodal,
                )
            os.replace(stage, self.destination)
            return

        target = (
            self.destination / GGUF_EMBEDDED_ADAPTER
            if self.kind == "gguf"
            else self.destination / MLX_EMBEDDED_ADAPTER / "adapters.safetensors"
        )
        if not target.is_file():
            raise OSError(f"checkpoint model is incomplete: {target}")
        if self.kind == "gguf":
            from .fusion import resolve_gguf_fusion_bundle

            bundle = resolve_gguf_fusion_bundle(self.destination)
            export_gguf_weights(
                bundle.model,
                bundle.shards,
                snapshot,
                self.destination / "merged.gguf",
                work=self.work,
                log=self.root / "logs" / "checkpoint-merge.log",
            )
        else:
            export_mlx_weights(
                self.model,
                snapshot_dir,
                self.destination / "merged",
                work=self.work,
                log=self.root / "logs" / "checkpoint-merge.log",
                multimodal=self.multimodal,
            )
        pending = target.with_name(target.name + ".pending")
        shutil.copyfile(snapshot, pending)
        os.replace(pending, target)
        if self.kind == "mlx":
            config = snapshot_dir / "adapter_config.json"
            target_config = target.parent / "adapter_config.json"
            if config.is_file():
                pending_config = target_config.with_name(target_config.name + ".pending")
                shutil.copyfile(config, pending_config)
                os.replace(pending_config, target_config)

    def _manual_export(self, generation: str) -> Path | None:
        request = self.root / "checkpoint-export.json"
        try:
            payload = json.loads(request.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not generation or payload.get("generation") != generation:
            return None
        selected = payload.get("directory")
        if not isinstance(selected, str) or not selected:
            raise ValueError("checkpoint destination is missing")
        target = Path(selected).expanduser().resolve() / self.kind
        if target == self.destination.resolve():
            request.unlink(missing_ok=True)
            return target
        if target.exists() and not (target / "osai_fusion.json").is_file():
            raise OSError(f"refusing to replace an unrelated model folder: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.with_name(f".{target.name}.{uuid.uuid4().hex}.pending")
        backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.previous")
        try:
            shutil.copytree(self.destination, stage)
            if target.exists():
                os.replace(target, backup)
            try:
                os.replace(stage, target)
            except BaseException:
                if backup.exists():
                    os.replace(backup, target)
                raise
            request.unlink(missing_ok=True)
            return target
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
