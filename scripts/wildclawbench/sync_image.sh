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
#   6. Tag the current image as a backup, then commit the container over the
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
#   IMAGE          eval image tag       (default: wildclawbench-mindmemos:v1.3)
#   MINDMEMOS_REPO repo root            (default: this script's repo root)
#   BASE_URL       MindMemOS API url    (default: http://host.docker.internal:8001)
#   USER_ID        auth user id         (default: wildclawbench)

set -euo pipefail

IMAGE="${IMAGE:-wildclawbench-mindmemos:v1.3}"
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

# Container-internal paths (verified against wildclawbench-mindmemos:v1.3).
CONTAINER_SDK_DST="/workspace/mindmemos_sdk"
CONTAINER_PLUGIN_DIST="/root/.openclaw/extensions/mindmemos-memory/dist"

cname="wildclaw-sync-$$"
cleanup() { docker rm -f "$cname" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> [1/6] building plugin from local source ($plugin_src)"
( cd "$plugin_src" && npm install --silent && npm run build --silent )

echo "==> [2/6] starting temp container from $IMAGE"
docker run -dit --name "$cname" "$IMAGE" sleep infinity >/dev/null

echo "==> [3/6] reinstalling mindmemos_sdk from local source (offline, --no-deps)"
docker cp "$sdk_src" "$cname:$CONTAINER_SDK_DST"
docker exec -e PIP_CONFIG_FILE=/dev/null "$cname" bash -lc "
  python3 -m pip install --no-deps --force-reinstall --no-build-isolation '$CONTAINER_SDK_DST'
"

echo "==> [4/6] overlaying freshly built plugin dist/ into container"
docker exec "$cname" mkdir -p "$CONTAINER_PLUGIN_DIST"
docker cp "$plugin_src/dist/." "$cname:$CONTAINER_PLUGIN_DIST/"
# docker cp preserves the host file owner (your local user's uid), but
# OpenClaw's plugin loader blocks any plugin file not owned by root as a
# "suspicious ownership" security check -- must chown back to root or the
# plugin silently gets blocked and every task fails at "Model setup failed".
docker exec "$cname" chown -R root:root "$CONTAINER_PLUGIN_DIST"

echo "==> [5/6] authenticating and verifying"
docker exec "$cname" mindmemos auth \
  --base-url "$BASE_URL" \
  --api-key "$api_key" \
  --user-id "$USER_ID"
docker exec "$cname" mindmemos doctor

echo "==> [6/6] backing up current $IMAGE, then committing over it"
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
