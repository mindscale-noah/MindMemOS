#!/usr/bin/env bash
# Run every WildClawBench task one at a time, and after each task, block until
# MindMemOS has fully processed (status=ok/error, not queued/processing) that
# task's async memory writes before starting the next task.
#
# script/run.sh / eval/run_batch.py have no concept of MindMemOS's async add
# pipeline, so running "--category all --parallel 1" alone only guarantees
# task containers don't overlap -- it does NOT guarantee the previous task's
# memory has finished writing before the next task's agent starts recalling.
# This wrapper closes that gap.
#
# Usage:
#   WILDCLAWBENCH_DIR=/Users/chenliang/WildClawBench \
#   MINDMEMOS_PROJECT_ID=proj_wildclawbench_20260706_112221 \
#   bash scripts/wildclawbench/run_serial.sh --category all --model yibu/gpt-4.1-mini --models-config my_api.json

set -euo pipefail

WILDCLAWBENCH_DIR="${WILDCLAWBENCH_DIR:?set WILDCLAWBENCH_DIR to your WildClawBench checkout}"
MINDMEMOS_PROJECT_ID="${MINDMEMOS_PROJECT_ID:?set MINDMEMOS_PROJECT_ID to the wildclawbench project_id from config/mindmemos/api_keys.yaml}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-180}"

category="all"
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --category)
      category="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wait_drain() {
  python3 "$script_dir/wait_drain.py" \
    --project-id "$MINDMEMOS_PROJECT_ID" \
    --qdrant-url "$QDRANT_URL" \
    --timeout "$DRAIN_TIMEOUT"
}

if [[ "$category" == "all" ]]; then
  categories_dir="$WILDCLAWBENCH_DIR/tasks"
  categories=$(find "$categories_dir" -mindepth 1 -maxdepth 1 -type d | sort)
else
  categories="$WILDCLAWBENCH_DIR/tasks/$category"
fi

# A single task failing (run.sh returns non-zero, e.g. the model couldn't
# produce the required results/ dir) is EXPECTED in a benchmark and must NOT
# abort the whole run. `set -e` would do exactly that -- it silently kills the
# wrapper the moment run.sh exits non-zero, so the sweep looks like it
# "mysteriously stopped" mid-category. Guard every fallible per-task command
# with `if` (which suppresses `set -e` for that command), record failures, and
# keep going. Same for wait_drain: a drain timeout should warn and continue,
# not nuke the remaining tasks.
failed_tasks=()
drain_timeouts=()

for category_dir in $categories; do
  for task_file in "$category_dir"/*task_*.md; do
    [[ -e "$task_file" ]] || continue
    rel_task="tasks/$(basename "$category_dir")/$(basename "$task_file")"
    echo "=== running $rel_task ==="
    if (cd "$WILDCLAWBENCH_DIR" && bash script/run.sh openclaw --task "$rel_task" "${extra_args[@]}"); then
      :
    else
      rc=$?
      echo "=== WARNING: $rel_task exited non-zero (rc=$rc); recording and continuing ==="
      failed_tasks+=("$rel_task (rc=$rc)")
    fi
    echo "=== draining MindMemOS async writes for $rel_task ==="
    if wait_drain; then
      :
    else
      rc=$?
      echo "=== WARNING: drain for $rel_task did not complete (rc=$rc); continuing ==="
      drain_timeouts+=("$rel_task (rc=$rc)")
    fi
  done
done

echo
echo "==================== run complete ===================="
if [[ ${#failed_tasks[@]} -eq 0 && ${#drain_timeouts[@]} -eq 0 ]]; then
  echo "All tasks completed and drained cleanly."
else
  if [[ ${#failed_tasks[@]} -gt 0 ]]; then
    echo "Tasks whose run.sh exited non-zero (${#failed_tasks[@]}):"
    printf '  - %s\n' "${failed_tasks[@]}"
    echo "  (often just the model failing the task -- check each score.json)"
  fi
  if [[ ${#drain_timeouts[@]} -gt 0 ]]; then
    echo "Tasks whose memory drain did not finish in time (${#drain_timeouts[@]}):"
    printf '  - %s\n' "${drain_timeouts[@]}"
    echo "  (check MindMemOS API / Kafka health; raise DRAIN_TIMEOUT if needed)"
  fi
fi
