import os
import subprocess
import sys

from osai.offline import offline_environment


def test_offline_environment_overrides_network_library_defaults():
    env = offline_environment({"HF_HUB_OFFLINE": "0", "WANDB_DISABLED": "false"})
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert env["WANDB_DISABLED"] == "true"


def test_network_guard_denies_ip_sockets():
    code = """
import socket
from osai.offline import install_network_guard
install_network_guard()
try:
    socket.create_connection((\"127.0.0.1\", 9), timeout=0.01)
except Exception as exc:
    print(type(exc).__name__, str(exc))
else:
    raise SystemExit(3)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert "network access is disabled" in result.stdout
