#!/usr/bin/env bash
# Run one SWE-bench Verified experiment phase at a time from the repository root.
# Each invocation uses the same frozen config and split. Never change a config
# after prepare; create a new output directory and config for a new protocol.
#
# Examples (replace the env path with a local file that contains your keys):
#   bash scripts/swebench/run_four_phases.sh prepare  config/eval/swebench_verified.yaml /path/to/credentials.env
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
  echo "Usage: bash scripts/swebench/run_four_phases.sh {prepare|train|extract|baseline|test} CONFIG_YAML CREDENTIALS_ENV" >&2
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

case "$phase" in
  prepare)
    # Local only: freeze a fresh 50/50 split, config, official image names,
    # grading dataset, and source hashes. Do this once per new output directory.
    # This route expects SWE-bench official images. For a custom local image map,
    # run `python -m mindmemos_eval.swebench split --config "$config"` instead.
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
    echo "Unknown phase: $phase (use prepare, train, extract, baseline, or test)" >&2
    exit 2
    ;;
esac
