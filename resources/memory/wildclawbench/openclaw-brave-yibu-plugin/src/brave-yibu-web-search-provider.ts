import { buildBraveYibuWebSearchProviderBase } from "./web-search-shared.js";
import { isDiagnosticFlagEnabled } from "openclaw/plugin-sdk/diagnostic-runtime";
import {
  mergeScopedSearchConfig,
  resolveProviderWebSearchPluginConfig,
} from "openclaw/plugin-sdk/provider-web-search-config-contract";

let braveYibuWebSearchRuntimePromise:
  | Promise<typeof import("./brave-yibu-web-search-provider.runtime.js")>
  | undefined;

function loadBraveYibuWebSearchRuntime() {
  braveYibuWebSearchRuntimePromise ??= import("./brave-yibu-web-search-provider.runtime.js");
  return braveYibuWebSearchRuntimePromise;
}

const BraveYibuSearchSchema = {
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

function createBraveYibuToolDefinition(searchConfig: any, config: unknown) {
  const diagnosticsEnabled = isDiagnosticFlagEnabled("brave-yibu.http", config);
  return {
    description:
      "Search the web using Yibu's Brave-compatible provider. Returns titles, URLs, and snippets for fast research.",
    parameters: BraveYibuSearchSchema,
    execute: async (args: unknown) => {
      const { executeBraveYibuSearch } = await loadBraveYibuWebSearchRuntime();
      return await executeBraveYibuSearch(args, searchConfig, { diagnosticsEnabled });
    },
  };
}

/** Create the runtime Yibu Brave-compatible provider descriptor. */
export function createBraveYibuWebSearchProvider() {
  return {
    ...buildBraveYibuWebSearchProviderBase(),
    createTool: (ctx: any) =>
      createBraveYibuToolDefinition(
        mergeScopedSearchConfig(
          ctx.searchConfig,
          "brave-yibu",
          resolveProviderWebSearchPluginConfig(ctx.config, "brave-yibu"),
          { mirrorApiKeyToTopLevel: true },
        ),
        ctx.config,
      ),
  };
}
