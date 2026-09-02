#!/usr/bin/env bash
# Re-sync the WildClawBench eval image with the CURRENT working-tree source,
# then authenticate it with the given API key -- idempotent and always safe to
# run. You do NOT need to figure out "did the SDK / plugin change?" first: just
# run this before every evaluation and the image is guaranteed to match the code
# you have checked out right now.
#
# What it does, in order:
#   1. Build the OpenClaw plugin from local source (tsc -> dist/).
#   2. Start a temp container from the eval image.
#   3. Reinstall mindmemos_sdk from local source (no network: --no-deps, deps
#      are already baked into the image).
#   4. Overlay the freshly built plugin dist/ into the container (no network).
#   5. Authenticate with the given --api-key and verify via `mindmemos doctor`.
#   6. Switch the plugin to sync add mode (see the step-6 comment below) and
#      raise the SDK CLI HTTP timeout to cover inline memory extraction.
#   7. Tag the current image as a backup, then commit the container over the
#      image tag. The backup tag is removed automatically once the whole
#      script exits successfully -- if anything fails after this point, the
#      backup is left in place so you can roll back:
#        docker rmi "$IMAGE" && docker tag <backup tag printed above> "$IMAGE"
#
# It does NOT touch src/mindmemos (the memory algorithm / API). That code runs
# from source via `make api`, not from the image -- to pick up changes there,
# just restart the API process. See WILDCLAWBENCH_QUICKSTART_ZH.md section 2.
#
# Usage:
#   bash scripts/wildclawbench/sync_image.sh --api-key <current api_key>
#
# Optional env overrides:
#   IMAGE          eval image tag       (default: wildclawbench-mindmemos:v1.3-brave-yibu)
#   MINDMEMOS_REPO repo root            (default: this script's repo root)
#   BASE_URL       MindMemOS API url    (default: http://host.docker.internal:8001)
#   USER_ID        auth user id         (default: wildclawbench)

set -euo pipefail

IMAGE="${IMAGE:-wildclawbench-mindmemos:v1.3-brave-yibu}"
BASE_URL="${BASE_URL:-http://host.docker.internal:8001}"
USER_ID="${USER_ID:-wildclawbench}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINDMEMOS_REPO="${MINDMEMOS_REPO:-$(cd "$script_dir/../.." && pwd)}"

api_key=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-key) api_key="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$api_key" ]]; then
  echo "ERROR: --api-key is required (the api_key printed by wildclawbench_new_key.py)" >&2
  exit 2
fi

sdk_src="$MINDMEMOS_REPO/src/mindmemos_sdk"
plugin_src="$MINDMEMOS_REPO/plugins/openclaw-plugin"
for d in "$sdk_src" "$plugin_src"; do
  [[ -d "$d" ]] || { echo "ERROR: not found: $d" >&2; exit 2; }
done

# Container-internal paths (verified against the MindMemOS WildClawBench images).
CONTAINER_SDK_DST="/workspace/mindmemos_sdk"
CONTAINER_PLUGIN_DIST="/root/.openclaw/extensions/mindmemos-memory/dist"

cname="wildclaw-sync-$$"
cleanup() { docker rm -f "$cname" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> [1/7] building plugin from local source ($plugin_src)"
( cd "$plugin_src" && npm install --silent && npm run build --silent )

echo "==> [2/7] starting temp container from $IMAGE"
docker run -dit --name "$cname" "$IMAGE" sleep infinity >/dev/null

echo "==> [3/7] reinstalling mindmemos_sdk from local source (offline, --no-deps)"
# Wipe the destination first: `docker cp <dir> cname:<existing-dir>` nests the
# copy as <existing-dir>/<dir-name> under GitBash/MSYS path handling, which
# would leave pip installing a stale outer tree. Dest absent -> clean copy.
docker exec "$cname" bash -c "rm -rf $CONTAINER_SDK_DST"
docker cp "$sdk_src" "$cname:$CONTAINER_SDK_DST"
docker exec -e PIP_CONFIG_FILE=/dev/null "$cname" bash -lc "
  python3 -m pip install --no-deps --force-reinstall --no-build-isolation '$CONTAINER_SDK_DST'
"

echo "==> [4/7] overlaying freshly built plugin dist/ into container"
# docker cp's "src/." overlay semantics are unreliable under GitBash/MSYS
# (observed silently nesting dist/dist/ instead of overlaying, leaving the
# plugin loader on a stale dist/index.js). Stream a tarball instead -- the
# container-absolute paths live inside bash -c strings where MSYS never
# translates them -- and verify a new-code marker afterwards so a silent
# mis-copy fails loudly instead of shipping a stale plugin.
docker exec "$cname" bash -c "rm -rf $CONTAINER_PLUGIN_DIST"
( cd "$plugin_src" && tar -cf - dist ) | docker exec -i "$cname" bash -c "tar -xf - -C $(dirname "$CONTAINER_PLUGIN_DIST")"
# The tarball preserves the host file owner (your local user's uid), but
# OpenClaw's plugin loader blocks any plugin file not owned by root as a
# "suspicious ownership" security check -- must chown back to root or the
# plugin silently gets blocked and every task fails at "Model setup failed".
docker exec "$cname" bash -c "chown -R root:root $CONTAINER_PLUGIN_DIST && grep -q mindmemos-logs $CONTAINER_PLUGIN_DIST/index.js && echo 'plugin overlay verified'"

echo "==> [5/7] authenticating and verifying"
docker exec "$cname" mindmemos auth \
  --base-url "$BASE_URL" \
  --api-key "$api_key" \
  --user-id "$USER_ID"
docker exec "$cname" mindmemos doctor

echo "==> [6/7] switching plugin add mode to sync (force-drain every add)"
# Async adds park messages in the schema add buffer and only flush when the
# rule chunker cuts an episode (>=50 messages, no speaker/time cuts) -- under
# the WildClawBench profiles a single task's 12-30 message trajectory NEVER
# reaches that bar, so memories are never generated. Sync mode drains the
# buffer inline with force=True (schema_add.add_sync) on every add: one task
# trajectory = one episode, generated before the HTTP call returns. The SDK
# CLI's 30s HTTP timeout cannot cover the inline extraction (~25-60s of LLM
# calls), so raise it to 600s in ~/.mindmemos/settings.json first.
docker exec -i "$cname" python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/root/.mindmemos/settings.json")
cfg = json.loads(p.read_text(encoding="utf-8"))
net = cfg.setdefault("network", {})
net["timeout_seconds"] = 600
net["max_retries"] = 0
p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("network.timeout_seconds ->", net["timeout_seconds"], "max_retries ->", net["max_retries"])
PY
docker exec "$cname" openclaw config set plugins.entries.mindmemos-memory.config.addMode sync

# Several tasks' warmup steps `pip install` exact pins inside the container
# (e.g. numpy==1.26.4, fastapi). pypi.org is unreachable from this network, so
# warmups fail with "Could not find a version" and the task never starts. Bake
# the Tsinghua PyPI mirror into both pip config locations (user + system) so
# every interpreter in the image (system python, conda envs) picks it up.
docker exec "$cname" bash -c '
  mkdir -p /root/.config/pip
  printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n" > /root/.config/pip/pip.conf
  mkdir -p /etc/pip
  printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n" > /etc/pip.conf
  pip config list 2>/dev/null || true
'

echo "==> [7/7] backing up current $IMAGE, then committing over it"
backup_tag="${IMAGE}-backup-$(date +%Y%m%d%H%M%S)"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker tag "$IMAGE" "$backup_tag"
  echo "    backed up existing image as $backup_tag"
else
  backup_tag=""
  echo "    no existing $IMAGE found, nothing to back up"
fi

docker commit "$cname" "$IMAGE" >/dev/null

# The temp container still references the pre-commit image (now only reachable
# via $backup_tag), so it must be removed before that tag can be deleted.
cleanup

if [[ -n "$backup_tag" ]]; then
  docker rmi "$backup_tag" >/dev/null
  echo "    sync succeeded, removed backup $backup_tag"
fi

echo "OK: $IMAGE is now synced with your current source and authenticated."
echo "    api_key -> project is fixed by the key itself; run wildclawbench_new_key.py to switch projects."
