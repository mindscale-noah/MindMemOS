import { buildSerpapiWebSearchProviderBase } from "./web-search-shared.js";
import { isDiagnosticFlagEnabled } from "openclaw/plugin-sdk/diagnostic-runtime";
import {
  mergeScopedSearchConfig,
  resolveProviderWebSearchPluginConfig,
} from "openclaw/plugin-sdk/provider-web-search-config-contract";
import { isRecord } from "openclaw/plugin-sdk/string-coerce-runtime";

/**
 * SerpApi-compatible web-search provider factory. It builds the agent tool
 * definition and lazy-loads HTTP execution only when a search is run.
 */

let serpapiWebSearchRuntimePromise: Promise<typeof import("./serpapi-web-search-provider.runtime.js")> | undefined;
function loadSerpapiWebSearchRuntime() {
  serpapiWebSearchRuntimePromise ??= import("./serpapi-web-search-provider.runtime.js");
  return serpapiWebSearchRuntimePromise;
}

const SerpapiSearchSchema = {
  type: "object",
  properties: {
    query: {
      type: "string",
      description: "Search query string.",
    },
    count: {
      type: "integer",
      description: "Number of results to return (1-10).",
      minimum: 1,
      maximum: 10,
    },
    country: {
      type: "string",
      description:
        "2-letter country code for region-specific results (e.g., 'DE', 'US', 'ALL'). Default: 'US'.",
    },
    language: {
      type: "string",
      description: "ISO 639-1 language code for results (e.g., 'en', 'de', 'fr').",
    },
    freshness: {
      type: "string",
      description: "Filter by time: 'day' (24h), 'week', 'month', or 'year'.",
    },
    date_after: {
      type: "string",
      description: "Only results published after this date (YYYY-MM-DD).",
    },
    date_before: {
      type: "string",
      description: "Only results published before this date (YYYY-MM-DD).",
    },
    search_lang: {
      type: "string",
      description:
        "Brave language code for search results (e.g., 'en', 'de', 'en-gb', 'zh-hans', 'zh-hant', 'pt-br').",
    },
    ui_lang: {
      type: "string",
      description:
        "Locale code for UI elements in language-region format (e.g., 'en-US', 'de-DE', 'fr-FR', 'tr-TR'). Must include region subtag.",
    },
  },
};

// Legacy "mode" probe carried over verbatim from the Brave provider this was
// derived from. It reads the same `brave` scope the original did and its
// return value is intentionally unused -- SerpApi's minimal runtime has no
// llm-context mode. Kept only to stay behavior-equivalent with the source.
function resolveLegacyBraveSearchMode(searchConfig: any): "llm-context" | "web" {
  return (isRecord(searchConfig?.brave) ? searchConfig.brave : undefined)?.mode === "llm-context"
    ? "llm-context"
    : "web";
}

function createSerpapiToolDefinition(searchConfig: any, config: unknown) {
  void resolveLegacyBraveSearchMode(searchConfig);
  const diagnosticsEnabled = isDiagnosticFlagEnabled("serpapi.http", config);
  return {
    description:
      "Search the web using a SerpApi-compatible provider. Returns titles, URLs, and snippets for fast research.",
    parameters: SerpapiSearchSchema,
    execute: async (args: unknown) => {
      const { executeSerpapiSearch } = await loadSerpapiWebSearchRuntime();
      return await executeSerpapiSearch(args, searchConfig, { diagnosticsEnabled });
    },
  };
}

/** Create the runtime SerpApi-compatible provider descriptor. */
export function createSerpapiWebSearchProvider() {
  return {
    ...buildSerpapiWebSearchProviderBase(),
    createTool: (ctx: any) =>
      createSerpapiToolDefinition(
        mergeScopedSearchConfig(
          ctx.searchConfig,
          "serpapi",
          resolveProviderWebSearchPluginConfig(ctx.config, "serpapi"),
          { mirrorApiKeyToTopLevel: true },
        ),
        ctx.config,
      ),
  };
}
