import { createWebSearchProviderContractFields } from "openclaw/plugin-sdk/provider-web-search-config-contract";
import { isRecord } from "openclaw/plugin-sdk/string-coerce-runtime";
//#region extensions/serpapi/web-search-shared.ts
/**
* Shared SerpApi (yibu) provider metadata and credential lookup. Contract
* tests and runtime provider creation both use this lightweight descriptor.
*/
/** Canonical config path for the SerpApi API key. */
const BRAVE_CREDENTIAL_PATH = "plugins.entries.serpapi.config.webSearch.apiKey";
/** Resolve legacy top-level SerpApi credentials from old web-search config. */
function resolveLegacyTopLevelBraveCredential(config) {
	if (!isRecord(config)) return;
	const tools = isRecord(config.tools) ? config.tools : void 0;
	const web = isRecord(tools?.web) ? tools.web : void 0;
	const search = isRecord(web?.search) ? web.search : void 0;
	if (!search || !("apiKey" in search)) return;
	return {
		path: "tools.web.search.apiKey",
		value: search.apiKey
	};
}
function resolveBraveWebSearchPluginConfig(config) {
	if (!isRecord(config)) return;
	const plugins = isRecord(config.plugins) ? config.plugins : void 0;
	const entries = isRecord(plugins?.entries) ? plugins.entries : void 0;
	const entry = isRecord(entries?.serpapi) ? entries.serpapi : void 0;
	const pluginConfig = isRecord(entry?.config) ? entry.config : void 0;
	return isRecord(pluginConfig?.webSearch) ? pluginConfig.webSearch : void 0;
}
/** Resolve SerpApi credentials from current plugin config or legacy fallback. */
function resolveConfiguredBraveCredential(config) {
	return resolveBraveWebSearchPluginConfig(config)?.apiKey ?? resolveLegacyTopLevelBraveCredential(config)?.value;
}
/** Build the common SerpApi provider metadata without the runtime tool executor. */
function buildBraveWebSearchProviderBase() {
	return {
		id: "serpapi",
		label: "SerpApi (yibu)",
		hint: "Google-style organic results via yibu SerpApi gateway",
		onboardingScopes: ["text-inference"],
		credentialLabel: "SerpApi (yibu) API key",
		envVars: ["BRAVE_API_KEY"],
		placeholder: "sk-...",
		signupUrl: "https://yibuapi.com/pricing",
		docsUrl: "https://yibuapi.apifox.cn/394660230e0",
		autoDetectOrder: 10,
		credentialPath: BRAVE_CREDENTIAL_PATH,
		...createWebSearchProviderContractFields({
			credentialPath: BRAVE_CREDENTIAL_PATH,
			searchCredential: { type: "top-level" },
			configuredCredential: { pluginId: "serpapi" }
		}),
		getConfiguredCredentialValue: resolveConfiguredBraveCredential,
		getConfiguredCredentialFallback: resolveLegacyTopLevelBraveCredential
	};
}
//#endregion
export { BRAVE_CREDENTIAL_PATH, buildBraveWebSearchProviderBase, resolveConfiguredBraveCredential, resolveLegacyTopLevelBraveCredential };
