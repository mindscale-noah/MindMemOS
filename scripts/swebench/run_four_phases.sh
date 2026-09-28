#!/usr/bin/env bash
# Run one SWE-bench Verified experiment phase at a time from the repository root.
# Each invocation uses the same frozen config and split. Never change a config
# after prepare; create a new output directory and config for a new protocol.
#
# Examples (replace the env path with a local file that contains your keys):
#   bash scripts/swebench/run_four_phases.sh prepare  config/eval/swebench_verified.yaml /path/to/credentials.env
#   bash scripts/swebench/run_four_phases.sh prepare-official config/eval/swebench_verified.yaml /path/to/credentials.env
#   bash scripts/swebench/run_four_phases.sh train    config/eval/swebench_verified.yaml /path/to/credentials.env
#   bash scripts/swebench/run_four_phases.sh extract  config/eval/swebench_verified.yaml /path/to/credentials.env
#   bash scripts/swebench/run_four_phases.sh baseline config/eval/swebench_verified.yaml /path/to/credentials.env
#   bash scripts/swebench/run_four_phases.sh test     config/eval/swebench_verified.yaml /path/to/credentials.env
#
# The example YAML contains placeholder model IDs. Copy it to a new config and
# fill in model IDs, dataset path, image map, output path, and memory project IDs
# before using these examples. The env file stays outside Git.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/swebench/run_four_phases.sh {prepare|prepare-official|train|extract|baseline|test} CONFIG_YAML CREDENTIALS_ENV" >&2
  exit 2
fi

phase=$1
config=$2
env_file=$3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)

# Resolve the credentials path before changing directories. Config paths are
# deliberately relative to the repository root, as required by load_config().
if [[ ! -f "$env_file" ]]; then
  echo "Credential file not found: $env_file" >&2
  exit 2
fi
env_file=$(cd "$(dirname "$env_file")" && pwd -P)/$(basename "$env_file")
cd "$repo_root"

python_bin=${SWEBENCH_PYTHON:-"$repo_root/.venv/bin/python"}
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: $python_bin" >&2
  exit 2
fi
if [[ ! -f "$config" ]]; then
  echo "Configuration file not found: $config" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/src/mindmemos_eval:$repo_root/src/mindmemos_sdk${PYTHONPATH:+:$PYTHONPATH}"

# Memory service settings for extract/test. An exported value wins, then the
# local env file, then the Lite API's default address on this machine.
# Leave project keys out of this Git-tracked script: put them in the local env
# file or export them in the shell before running extract/test.
memory_url_from_file=$("$python_bin" - "$env_file" <<'PY'
import sys

from dotenv import dotenv_values

print(dotenv_values(sys.argv[1]).get("SWEBENCH_MEMORY_BASE_URL") or "")
PY
)
export SWEBENCH_MEMORY_BASE_URL="${SWEBENCH_MEMORY_BASE_URL:-${memory_url_from_file:-http://127.0.0.1:8000}}"
export SWEBENCH_PLAN_MEMORY_KEY="${SWEBENCH_PLAN_MEMORY_KEY:-}"
export SWEBENCH_EXPERIENCE_MEMORY_KEY="${SWEBENCH_EXPERIENCE_MEMORY_KEY:-}"

case "$phase" in
  prepare)
    # Freeze a fresh 50/50 split and config. Supply a local image map covering
    # the selected task IDs before train. This never overwrites an image map.
    "$python_bin" -u -m mindmemos_eval.swebench split --config "$config"
    ;;
  prepare-official)
    # Use this instead of prepare only with official SWE-bench images. The
    # config must put image_map_path inside its new output_dir and explicitly
    # enable docker.pull_missing plus docker.restore_base_commit. It writes
    # the image map and grading dataset; never run it on an existing output.
    "$python_bin" - "$config" <<'PY'
import sys
from pathlib import Path

from mindmemos_eval.swebench.config import load_config

config = load_config(Path(sys.argv[1]))
if config.output_dir.exists():
    raise SystemExit(f"Output directory already exists: {config.output_dir}")
if not config.image_map_path.is_relative_to(config.output_dir):
    raise SystemExit("Official image map must be inside the new output directory")
if not config.docker.pull_missing or not config.docker.restore_base_commit:
    raise SystemExit("Official images require pull_missing and restore_base_commit")
PY
    "$python_bin" -u scripts/swebench/run_experiment.py \
      --config "$config" --env-file "$env_file" --mode prepare
    ;;
  train)
    # ① Collect 50 training trajectories without memory retrieval. The runner
    # skips every existing result, including failed results, and stops on an
    # interrupted attempt. The export records all 50 states and file hashes.
    if ! "$python_bin" -u -m mindmemos_eval.swebench train \
      --config "$config" --env-file "$env_file"; then
      "$python_bin" -u scripts/swebench/collect_train.py \
        --config "$config" --mode export || true
      exit 1
    fi
    "$python_bin" -u scripts/swebench/collect_train.py \
      --config "$config" --mode export
    ;;
  extract)
    # ② Write planning memories from parent trajectories and experience
    # memories from child trajectories to two distinct, empty Lite projects.
    # This needs SWEBENCH_MEMORY_BASE_URL and both project API keys in the env.
    # A pending receipt means the remote write is uncertain: reconcile it
    # before invoking this phase again to avoid duplicate memory writes.
    "$python_bin" -u -m mindmemos_eval.swebench extract \
      --config "$config" --env-file "$env_file"
    ;;
  baseline)
    # ③ Run the held-out 50 tasks without memory retrieval, even when the two
    # memory projects already contain extracted training memories. Patches go
    # to baseline/predictions.jsonl; completed does not mean officially solved.
    "$python_bin" -u -m mindmemos_eval.swebench baseline \
      --config "$config" --env-file "$env_file"
    # Optional official scoring after all 50 baseline tasks are complete:
    # "$python_bin" scripts/swebench/grade_baseline.py --config "$config" \
    #   --harness-python outputs/swebench-harness/.venv/bin/python
    ;;
  test)
    # ④ Run the same held-out split with planning and experience memory search
    # before each parent/child model call. This phase requires all 100 role
    # extraction receipts, and reads memories without writing test data back.
    # Patches go to test/predictions.jsonl; grade them separately if needed.
    "$python_bin" -u -m mindmemos_eval.swebench test \
      --config "$config" --env-file "$env_file"
    ;;
  *)
    echo "Unknown phase: $phase (use prepare, prepare-official, train, extract, baseline, or test)" >&2
    exit 2
    ;;
esac
