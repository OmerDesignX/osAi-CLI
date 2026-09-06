"""Run the allowlisted MLX LM module with network sockets disabled."""

from __future__ import annotations

import json
import os
import platform
import runpy
import sys

from .errors import DependencyError
from .offline import OFFLINE_ENVIRONMENT, install_network_guard


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {
        "mlx_lm",
        "probe-mlx",
        "align-mlx",
        "rollout-mlx",
        "verify-mlx-fusion",
    }:
        print(
            "usage: python -m osai._offline_runner "
            "{probe-mlx|align-mlx CONFIG|rollout-mlx CONFIG|"
            "verify-mlx-fusion BASE ADAPTER MERGED|"
            "mlx_lm [args...]}",
            file=sys.stderr,
        )
        return 2
    os.environ.update(OFFLINE_ENVIRONMENT)
    install_network_guard()

    if sys.argv[1] == "probe-mlx":
        if len(sys.argv) != 2:
            print("probe-mlx accepts no arguments", file=sys.stderr)
            return 2
        report = _configure_mlx()
        import mlx_lm  # noqa: F401

        print(json.dumps(report, sort_keys=True))
        return 0

    if sys.argv[1] == "align-mlx":
        if len(sys.argv) != 3:
            print("align-mlx requires one config path", file=sys.stderr)
            return 2
        report = _configure_mlx()
        if not report["usable"]:
            raise DependencyError(str(report["reason"]))
        from ._mlx_alignment_runner import run

        return run(sys.argv[2])

    if sys.argv[1] == "rollout-mlx":
        if len(sys.argv) != 3:
            print("rollout-mlx requires one config path", file=sys.stderr)
            return 2
        report = _configure_mlx()
        if not report["usable"]:
            raise DependencyError(str(report["reason"]))
        from ._mlx_rollout_runner import run

        return run(sys.argv[2])

    if sys.argv[1] == "verify-mlx-fusion":
        if len(sys.argv) != 5:
            print("verify-mlx-fusion requires BASE ADAPTER MERGED", file=sys.stderr)
            return 2
        report = _configure_mlx()
        if not report["usable"]:
            raise DependencyError(str(report["reason"]))
        from ._mlx_fusion_verifier import run

        return run(sys.argv[2], sys.argv[3], sys.argv[4])

    module = sys.argv[1]
    report = _configure_mlx()
    if not report["usable"]:
        raise DependencyError(str(report["reason"]))
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


def _configure_mlx() -> dict[str, str | bool | int]:
    import mlx.core as mx

    requested = os.environ.get("OSAI_MLX_ACCELERATOR", "auto").lower()
    system = platform.system()
    metal_module = getattr(mx, "metal", None)
    cuda_module = getattr(mx, "cuda", None)
    metal = bool(metal_module and metal_module.is_available())
    cuda = bool(cuda_module and cuda_module.is_available())
    if requested == "auto":
        selected = "metal" if metal else "cuda" if cuda else "cpu"
    else:
        selected = requested
    reason = None
    usable = True
    if selected == "metal" and (system != "Darwin" or not metal):
        usable = False
        reason = "requested MLX Metal backend is unavailable"
    elif selected == "cuda" and (system != "Linux" or not cuda):
        usable = False
        reason = "requested MLX CUDA backend is unavailable"
    elif selected not in {"metal", "cuda", "cpu"}:
        usable = False
        reason = f"unsupported MLX accelerator: {selected}"
    if usable:
        mx.set_default_device(mx.cpu if selected == "cpu" else mx.gpu)
    gpu_count = int(mx.device_count(mx.gpu)) if selected != "cpu" else 0
    return {
        "mlx_version": mx.__version__,
        "platform": system,
        "accelerator": selected,
        "metal_available": metal,
        "cuda_available": cuda,
        "gpu_count": gpu_count,
        "usable": usable,
        "reason": reason or "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
