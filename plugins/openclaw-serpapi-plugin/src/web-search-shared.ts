import { createWebSearchProviderContractFields } from "openclaw/plugin-sdk/provider-web-search-config-contract";
import { isRecord } from "openclaw/plugin-sdk/string-coerce-runtime";

/**
 * Shared SerpApi (yibu) provider metadata and credential lookup. Contract
 * tests and runtime provider creation both use this lightweight descriptor.
 */

/** Canonical config path for the SerpApi API key. */
export const SERPAPI_CREDENTIAL_PATH = "plugins.entries.serpapi.config.webSearch.apiKey";

/** Resolve legacy top-level SerpApi credentials from old web-search config. */
export function resolveLegacyTopLevelSerpapiCredential(
  config: unknown,
): { path: string; value: unknown } | undefined {
  if (!isRecord(config)) return;
  const tools = isRecord(config.tools) ? config.tools : undefined;
  const web = isRecord(tools?.web) ? tools.web : undefined;
  const search = isRecord(web?.search) ? web.search : undefined;
  if (!search || !("apiKey" in search)) return;
  return {
    path: "tools.web.search.apiKey",
    value: search.apiKey,
  };
}

function resolveSerpapiWebSearchPluginConfig(config: unknown): Record<string, unknown> | undefined {
  if (!isRecord(config)) return;
  const plugins = isRecord(config.plugins) ? config.plugins : undefined;
  const entries = isRecord(plugins?.entries) ? plugins.entries : undefined;
  const entry = isRecord(entries?.serpapi) ? entries.serpapi : undefined;
  const pluginConfig = isRecord(entry?.config) ? entry.config : undefined;
  return isRecord(pluginConfig?.webSearch) ? pluginConfig.webSearch : undefined;
}

/** Resolve SerpApi credentials from current plugin config or legacy fallback. */
export function resolveConfiguredSerpapiCredential(config: unknown): unknown {
  return (
    resolveSerpapiWebSearchPluginConfig(config)?.apiKey ??
    resolveLegacyTopLevelSerpapiCredential(config)?.value
  );
}

/** Build the common SerpApi provider metadata without the runtime tool executor. */
export function buildSerpapiWebSearchProviderBase() {
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
    credentialPath: SERPAPI_CREDENTIAL_PATH,
    ...createWebSearchProviderContractFields({
      credentialPath: SERPAPI_CREDENTIAL_PATH,
      searchCredential: { type: "top-level" },
      configuredCredential: { pluginId: "serpapi" },
    }),
    getConfiguredCredentialValue: resolveConfiguredSerpapiCredential,
    getConfiguredCredentialFallback: resolveLegacyTopLevelSerpapiCredential,
  };
}
