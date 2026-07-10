import { buildBraveWebSearchProviderBase } from "./web-search-shared.js";
import { isDiagnosticFlagEnabled } from "openclaw/plugin-sdk/diagnostic-runtime";
import { mergeScopedSearchConfig, resolveProviderWebSearchPluginConfig } from "openclaw/plugin-sdk/provider-web-search-config-contract";
import { isRecord } from "openclaw/plugin-sdk/string-coerce-runtime";
//#region extensions/brave/src/brave-web-search-provider.ts
/**
* Brave web-search provider factory. It builds the agent tool definition and
* lazy-loads HTTP execution only when a search is run.
*/
let braveWebSearchRuntimePromise;
function loadBraveWebSearchRuntime() {
	braveWebSearchRuntimePromise ??= import("./brave-web-search-provider.runtime-D_z2neAi.js");
	return braveWebSearchRuntimePromise;
}
const BraveSearchSchema = {
	type: "object",
	properties: {
		query: {
			type: "string",
			description: "Search query string."
		},
		count: {
			type: "integer",
			description: "Number of results to return (1-10).",
			minimum: 1,
			maximum: 10
		},
		country: {
			type: "string",
			description: "2-letter country code for region-specific results (e.g., 'DE', 'US', 'ALL'). Default: 'US'."
		},
		language: {
			type: "string",
			description: "ISO 639-1 language code for results (e.g., 'en', 'de', 'fr')."
		},
		freshness: {
			type: "string",
			description: "Filter by time: 'day' (24h), 'week', 'month', or 'year'."
		},
		date_after: {
			type: "string",
			description: "Only results published after this date (YYYY-MM-DD)."
		},
		date_before: {
			type: "string",
			description: "Only results published before this date (YYYY-MM-DD)."
		},
		search_lang: {
			type: "string",
			description: "Brave language code for search results (e.g., 'en', 'de', 'en-gb', 'zh-hans', 'zh-hant', 'pt-br')."
		},
		ui_lang: {
			type: "string",
			description: "Locale code for UI elements in language-region format (e.g., 'en-US', 'de-DE', 'fr-FR', 'tr-TR'). Must include region subtag."
		}
	}
};
function resolveBraveMode(searchConfig) {
	return (isRecord(searchConfig?.brave) ? searchConfig.brave : void 0)?.mode === "llm-context" ? "llm-context" : "web";
}
function createBraveToolDefinition(searchConfig, config) {
	const braveMode = resolveBraveMode(searchConfig);
	const diagnosticsEnabled = isDiagnosticFlagEnabled("serpapi.http", config);
	return {
		description: "Search the web using SerpApi (routed through yibu). Returns titles, URLs, and snippets for fast research.",
		parameters: BraveSearchSchema,
		execute: async (args) => {
			const { executeBraveSearch } = await loadBraveWebSearchRuntime();
			return await executeBraveSearch(args, searchConfig, { diagnosticsEnabled });
		}
	};
}
/** Create the runtime SerpApi (yibu) provider descriptor. */
function createBraveWebSearchProvider() {
	return {
		...buildBraveWebSearchProviderBase(),
		createTool: (ctx) => createBraveToolDefinition(mergeScopedSearchConfig(ctx.searchConfig, "serpapi", resolveProviderWebSearchPluginConfig(ctx.config, "serpapi"), { mirrorApiKeyToTopLevel: true }), ctx.config)
	};
}
//#endregion
export { createBraveWebSearchProvider as t };
