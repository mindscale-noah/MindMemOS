#!/usr/bin/env bash
# Install the "brave-yibu" OpenClaw web-search plugin into the standard
# WildClawBench evaluation image tag without overwriting the source image.
#
# Usage:
#   bash scripts/wildclawbench/install_brave_yibu_plugin.sh
#
# Optional env overrides:
#   SOURCE_IMAGE          base eval image tag
#                         (default: wildclawbench-mindmemos:v1.3)
#   TARGET_IMAGE          output eval image tag
#                         (default: wildclawbench-mindmemos:v1.3-brave-yibu)
#   BRAVE_YIBU_BASE_URL   full Brave-compatible endpoint
#                         (default: https://yibuapi.com/brave/v1/web/search)
#   MINDMEMOS_REPO        repo root (default: this script's repo root)
#
# Required at runtime:
#   BRAVE_API_KEY in WildClawBench's .env must hold the yibu key. The image
#   config stores '${BRAVE_API_KEY}' as a placeholder, not the secret itself.

set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-wildclawbench-mindmemos:v1.3}"
TARGET_IMAGE="${TARGET_IMAGE:-wildclawbench-mindmemos:v1.3-brave-yibu}"
BRAVE_YIBU_BASE_URL="${BRAVE_YIBU_BASE_URL:-https://yibuapi.com/brave/v1/web/search}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINDMEMOS_REPO="${MINDMEMOS_REPO:-$(cd "$script_dir/../.." && pwd)}"
plugin_src="$MINDMEMOS_REPO/resources/memory/wildclawbench/openclaw-brave-yibu-plugin"

[[ -d "$plugin_src" ]] || { echo "ERROR: not found: $plugin_src" >&2; exit 2; }

CONTAINER_INSTALLED_DIR="/root/.openclaw/extensions/brave-yibu"
CONTAINER_STAGING_DIR="/workspace/openclaw-brave-yibu-plugin-src"

cname="wildclaw-brave-yibu-install-$$"
cleanup() { docker rm -f "$cname" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> [1/6] building plugin from local TypeScript source ($plugin_src)"
( cd "$plugin_src" && npm install --silent && npm run build --silent )

echo "==> [2/6] starting temp container from $SOURCE_IMAGE"
docker run -dit --name "$cname" "$SOURCE_IMAGE" sleep infinity >/dev/null

echo "==> [3/6] copying plugin source into a staging dir and installing with --force"
docker exec "$cname" rm -rf "$CONTAINER_STAGING_DIR" "$CONTAINER_INSTALLED_DIR"
docker exec "$cname" mkdir -p "$CONTAINER_STAGING_DIR"
tar -C "$plugin_src" --exclude=node_modules --exclude=.git -cf - . \
  | docker exec -i "$cname" tar -C "$CONTAINER_STAGING_DIR" -xf -
docker exec "$cname" openclaw plugins install "$CONTAINER_STAGING_DIR" --force

echo "==> [4/6] fixing ownership and enabling the plugin"
docker exec "$cname" chown -R root:root "$CONTAINER_INSTALLED_DIR"
docker exec "$cname" openclaw config set plugins.entries.brave-yibu.enabled true
docker exec "$cname" openclaw config set plugins.entries.brave-yibu.config.webSearch.apiKey '${BRAVE_API_KEY}'
docker exec "$cname" openclaw config set plugins.entries.brave-yibu.config.webSearch.baseUrl "$BRAVE_YIBU_BASE_URL"
docker exec "$cname" openclaw config set tools.web.search.provider brave-yibu

echo "==> [5/6] validating config"
docker exec "$cname" openclaw config validate

echo "==> [6/6] committing $TARGET_IMAGE"
docker commit "$cname" "$TARGET_IMAGE" >/dev/null
cleanup

echo "OK: created $TARGET_IMAGE with brave-yibu web-search provider."
echo "    Source image was left unchanged: $SOURCE_IMAGE"
