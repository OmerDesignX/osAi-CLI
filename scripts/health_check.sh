#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false

"$project_dir/.venv/bin/ruff" check \
    "$project_dir/src" "$project_dir/tests" "$project_dir/scripts" --no-cache
"$project_dir/.venv/bin/pytest" -p no:cacheprovider "$project_dir/tests"
"$project_dir/.venv/bin/python" \
    "$project_dir/scripts/setup_osai.py" --current-environment --dry-run
"$project_dir/.venv/bin/osai" verify-models --root "$project_dir/osCode-Models"
"$project_dir/.venv/bin/osai" check-sessions --root "$project_dir/sessions"
