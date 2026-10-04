"""Small CLI entry point that answers version checks before loading trainers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from . import __version__


def main(argv: Sequence[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if list(arguments) == ["--version"]:
        print(f"osai {__version__}")
        return 0

    from .cli import main as cli_main

    return cli_main(arguments)
