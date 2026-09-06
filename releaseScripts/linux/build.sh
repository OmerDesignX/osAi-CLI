#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script on Linux." >&2
  exit 1
fi

PYTHON_BIN="${OSAI_RELEASE_PYTHON:-python3}"
cd "$ROOT"
exec "$PYTHON_BIN" releaseScripts/common/build_release.py
