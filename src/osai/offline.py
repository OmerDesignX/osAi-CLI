"""Strict local-only execution policy for all model runtimes."""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping

from .errors import DependencyError

OFFLINE_ENVIRONMENT = {
    "ANONYMIZED_TELEMETRY": "False",
    "CLEARML_OFFLINE_MODE": "1",
    "COMET_DISABLE_AUTO_LOGGING": "1",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
}


def offline_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment that disables known download and telemetry paths."""

    result = dict(os.environ if base is None else base)
    result.update(OFFLINE_ENVIRONMENT)
    return result


def install_network_guard() -> None:
    """Deny IPv4/IPv6 sockets in this process while retaining local Unix IPC.

    Environment flags stop supported libraries before they try the network.  The
    socket guard is a second boundary for unexpected Python dependency behavior.
    """

    if getattr(socket, "_osai_offline", False):
        return

    original_socket = socket.socket

    class LocalOnlySocket(original_socket):
        def connect(self, address):  # noqa: ANN001
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                raise DependencyError(
                    "network access is disabled by osai local-only mode"
                )
            return super().connect(address)

        def connect_ex(self, address):  # noqa: ANN001
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                raise DependencyError(
                    "network access is disabled by osai local-only mode"
                )
            return super().connect_ex(address)

        def sendto(self, data, *args):  # noqa: ANN001
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                raise DependencyError(
                    "network access is disabled by osai local-only mode"
                )
            return super().sendto(data, *args)

    def deny_connection(*_args, **_kwargs):
        raise DependencyError("network access is disabled by osai local-only mode")

    socket.socket = LocalOnlySocket
    socket.create_connection = deny_connection
    socket._osai_offline = True
