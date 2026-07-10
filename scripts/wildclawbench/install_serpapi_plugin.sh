#!/usr/bin/env bash
# Install/refresh the custom "serpapi" OpenClaw web-search plugin
# (resources/memory/wildclawbench/openclaw-serpapi-plugin) into the
# WildClawBench eval image, enable
# it, and point tools.web.search.provider at it. Idempotent: safe to re-run
# any time the plugin source changes, or if you're rebuilding the image from
# scratch and need search working.
#
# This is independent of scripts/wildclawbench/sync_image.sh, which only
# touches mindmemos_sdk and the mindmemos-memory plugin -- it has no idea
# this plugin exists, and running it will NOT remove or break this one
# (they live in separate ~/.openclaw/extensions/<id>/ directories).
#
# Why this needs its own script instead of just `openclaw plugins install`:
#   1. Freshly copied plugin files are owned by the host user (not root),
#      and OpenClaw's plugin loader silently blocks any plugin file not
#      owned by root as "suspicious ownership" -- must chown after copying.
#   2. The plugin manifest must NOT carry `"activation": {"onStartup": false}`
#      (copied from the original Brave plugin this was derived from) --
#      that flag prevents the gateway from ever auto-loading the plugin, so
#      web_search silently fails with "no provider is available" forever,
#      even though `openclaw plugins list` shows it as enabled. See
#      resources/memory/wildclawbench/openclaw-serpapi-plugin/README.md for the full story.
#   3. `tools.web.search.provider` must be explicitly set to "serpapi", or
#      OpenClaw falls back to auto-detect (and may pick nothing, or warn
#      about an invalid provider id left over from an earlier Brave setup).
#
# Usage:
#   SERPAPI_BASE_URL="https://<private-compatible-endpoint>" \
#     bash scripts/wildclawbench/install_serpapi_plugin.sh
#
# Optional env overrides:
#   IMAGE            eval image tag  (default: wildclawbench-mindmemos:v1.3)
#   MINDMEMOS_REPO   repo root       (default: this script's repo root)
# Required env:
#   SERPAPI_BASE_URL SerpApi-compatible endpoint base URL. Keep this out of git.

set -euo pipefail

IMAGE="${IMAGE:-wildclawbench-mindmemos:v1.3}"
: "${SERPAPI_BASE_URL:?set SERPAPI_BASE_URL to your SerpApi-compatible endpoint base URL}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINDMEMOS_REPO="${MINDMEMOS_REPO:-$(cd "$script_dir/../.." && pwd)}"
plugin_src="$MINDMEMOS_REPO/resources/memory/wildclawbench/openclaw-serpapi-plugin"

[[ -d "$plugin_src" ]] || { echo "ERROR: not found: $plugin_src" >&2; exit 2; }

CONTAINER_INSTALLED_DIR="/root/.openclaw/extensions/serpapi"
CONTAINER_STAGING_DIR="/workspace/openclaw-serpapi-plugin-src"

cname="wildclaw-serpapi-install-$$"
cleanup() { docker rm -f "$cname" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> [1/6] building plugin from local TypeScript source ($plugin_src)"
# dist/ is a git-ignored build artifact (tsc -> dist/), so a fresh clone has
# no dist/ yet; compile it here before staging into the container.
( cd "$plugin_src" && npm install --silent && npm run build --silent )

echo "==> [2/6] starting temp container from $IMAGE"
docker run -dit --name "$cname" "$IMAGE" sleep infinity >/dev/null

echo "==> [3/6] copying plugin source into a staging dir and installing with --force"
# --force matters here: this must work both on a fresh image (nothing
# installed yet) and when re-running after the plugin already exists
# (openclaw plugins install refuses with "delete it first" otherwise).
# Remove previous staging/install dirs first: older builds may have hashed dist
# filenames, and --force does not reliably delete stale files.
docker exec "$cname" rm -rf "$CONTAINER_STAGING_DIR" "$CONTAINER_INSTALLED_DIR"
# Install from a separate staging dir, not directly into extensions/serpapi,
# so `openclaw plugins install` (which also links the openclaw peerDependency)
# is what actually places the files -- copying straight into extensions/
# would make this step a no-op on re-runs and skip that linking.
docker exec "$cname" mkdir -p "$CONTAINER_STAGING_DIR"
# Exclude the local node_modules/ (393M+ after `npm install` above): the
# container has its own global openclaw that `plugins install` links as the
# peerDependency, and copying node_modules in would bloat the committed image.
tar -C "$plugin_src" --exclude=node_modules --exclude=.git -cf - . \
  | docker exec -i "$cname" tar -C "$CONTAINER_STAGING_DIR" -xf -
docker exec "$cname" openclaw plugins install "$CONTAINER_STAGING_DIR" --force

echo "==> [4/6] fixing ownership and enabling the plugin"
# openclaw plugins install copies files preserving the host user's ownership,
# which OpenClaw's loader then blocks as "suspicious ownership" -- must chown
# the actual installed location (not the staging dir) after install.
docker exec "$cname" chown -R root:root "$CONTAINER_INSTALLED_DIR"
docker exec "$cname" openclaw config set plugins.entries.serpapi.enabled true
docker exec "$cname" openclaw config set plugins.entries.serpapi.config.webSearch.apiKey '${BRAVE_API_KEY}'
docker exec "$cname" openclaw config set plugins.entries.serpapi.config.webSearch.baseUrl "$SERPAPI_BASE_URL"
docker exec "$cname" openclaw config set tools.web.search.provider serpapi

echo "==> [5/6] validating config"
docker exec "$cname" openclaw config validate

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

cleanup

if [[ -n "$backup_tag" ]]; then
  docker rmi "$backup_tag" >/dev/null
  echo "    sync succeeded, removed backup $backup_tag"
fi

echo "OK: $IMAGE now has the serpapi web-search plugin installed and enabled."
echo "    Remember: BRAVE_API_KEY in WildClawBench's .env must hold your SerpApi-compatible key"
echo "    SERPAPI_BASE_URL was written into the image config as the SerpApi-compatible endpoint."
echo "    (this is the delivery channel the plugin reads from -- see the plugin's README)."
