# openclaw-serpapi-plugin

An OpenClaw web-search provider plugin (`id: serpapi`) that calls yibu's
SerpApi-compatible endpoint (`GET https://yibuapi.com/serpapi/search`)
instead of Brave.

## Why this exists

The WildClawBench eval image's `brave` web-search plugin requires a paid
Brave Search API key, which wasn't set up. Rather than pay for Brave, this
plugin reuses the already-provisioned yibu gateway key (the same one used
for agent model inference) to provide equivalent web-search capability for
the `04_Search_Retrieval` task category.

The original `brave` plugin is left untouched and disabled in the eval
image — this is a fully separate, additional plugin, not a patched Brave.

## Provenance

This is a hand-adapted copy of the bundled `@openclaw/brave-plugin` (npm,
version `2026.6.6`, installed from `~/.openclaw/npm/projects/openclaw-brave-plugin-*/node_modules/@openclaw/brave-plugin`
inside the `wildclawbench-mindmemos:v1.3` image). Files were copied
byte-for-byte where unchanged; only identity/endpoint/mapping logic was
edited. See file-by-file notes below.

## What changed vs. the original Brave plugin

| File | Changed? | What |
|---|---|---|
| `package.json` | Yes | `name` changed (no longer claims to be `@openclaw/brave-plugin`); dropped npm-registry-only fields (`install.npmSpec`, `release`) since this is installed as a local plugin dir, not published |
| `openclaw.plugin.json` | Yes | `id: brave` → `serpapi`; `contracts.webSearchProviders` → `["serpapi"]`; **`activation.onStartup` removed entirely** — leaving it as `{"onStartup": false}` (copied verbatim from Brave) silently prevented the gateway from auto-loading this plugin at all, so `web_search` always failed with "no provider is available" even though the plugin looked "enabled" in config. Removing the field matches how `mindmemos-memory`'s manifest has no `activation` key and loads fine. `envVars` kept as `["BRAVE_API_KEY"]` — see note below on why. |
| `dist/index.js` | Yes (cosmetic) | `id`/`name`/`description` strings updated to serpapi; import path unchanged (file names weren't renamed) |
| `dist/web-search-shared.js` | Yes (identity) | `BRAVE_CREDENTIAL_PATH` value, `buildBraveWebSearchProviderBase()`'s `id`/`label`/`envVars`/`credentialPath`/`signupUrl`/`docsUrl`/contract `pluginId` all point at `serpapi` now. Exported function/const **names** were left as-is (`buildBraveWebSearchProviderBase`, `BRAVE_CREDENTIAL_PATH`) because `brave-web-search-provider-6mNi77fe.js` and `web-search-contract-api.js` still import them by those names and weren't touched. |
| `dist/brave-web-search-provider-6mNi77fe.js` | Yes (one hardcoded string) | `mergeScopedSearchConfig(ctx.searchConfig, "brave", ...)` / `resolveProviderWebSearchPluginConfig(ctx.config, "brave")` both hardcoded `"brave"` as the plugin id used to look up `plugins.entries.<id>.config` — this had to become `"serpapi"` or the plugin's own `webSearch` config (apiKey/baseUrl) would never resolve even though the provider identity elsewhere said `serpapi`. Tool description text and the diagnostic flag name (`"brave.http"` → `"serpapi.http"`) were also updated. |
| `dist/web-search-provider.js` | No | Pure re-export, provider-agnostic |
| `dist/web-search-contract-api.js` | No | Only used for contract/schema tests, `createTool: () => null`, never makes a real request |
| `dist/brave-web-search-provider.runtime-D_z2neAi.js` | Yes (core logic, rewritten) | This is the actual HTTP call + response mapping. Brave's original runtime supports LLM-context mode, country/language/freshness validation with Brave-specific codes — none of that exists in SerpApi, so this was rewritten smaller rather than force-mapped: endpoint → `GET {baseUrl}/serpapi/search` with `q`/`api_key`/`num` query params (not Brave's `X-Subscription-Token` header); response mapped from `data.organic_results[]` (`title`/`link`/`snippet`) instead of Brave's `data.web.results[]`. Country/language/freshness/date args from the shared tool schema are accepted but silently ignored — SerpApi has no equivalent. Caching, SSRF host-safety checks, and the exported function name (`executeBraveSearch`) were kept identical so the tool-definition file didn't need touching. |

## Why the env var is still called `BRAVE_API_KEY`

WildClawBench's own `src/utils/docker_utils.py` hardcodes injection of a
container env var literally named `BRAVE_API_KEY` (there's no generic
mechanism to forward a custom-named secret into task containers). Rather
than patch WildClawBench's own source — which would be lost on a fresh
`git clone` / update — this plugin reuses that exact variable name as its
delivery channel. The value it contains is a yibu API key, not a real
Brave key; see the comment above `BRAVE_API_KEY=` in WildClawBench's
`.env` for the same note.

## Installing / rebuilding this into the eval image

Use [scripts/wildclawbench/install_serpapi_plugin.sh](../../scripts/wildclawbench/install_serpapi_plugin.sh).
It is idempotent and independent of [scripts/wildclawbench/sync_image.sh](../../scripts/wildclawbench/sync_image.sh)
(that script only touches `mindmemos_sdk` and the `mindmemos-memory`
plugin — it does not know about this one).
