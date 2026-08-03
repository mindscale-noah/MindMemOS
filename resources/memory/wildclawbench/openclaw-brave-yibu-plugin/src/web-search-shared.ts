import { createWebSearchProviderContractFields } from "openclaw/plugin-sdk/provider-web-search-config-contract";
import { isRecord } from "openclaw/plugin-sdk/string-coerce-runtime";

/** Canonical config path for the Yibu Brave-compatible API key. */
export const BRAVE_YIBU_CREDENTIAL_PATH = "plugins.entries.brave-yibu.config.webSearch.apiKey";

/** Resolve legacy top-level web-search credentials from old config. */
export function resolveLegacyTopLevelBraveYibuCredential(
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

function resolveBraveYibuWebSearchPluginConfig(config: unknown): Record<string, unknown> | undefined {
  if (!isRecord(config)) return;
  const plugins = isRecord(config.plugins) ? config.plugins : undefined;
  const entries = isRecord(plugins?.entries) ? plugins.entries : undefined;
  const entry = isRecord(entries?.["brave-yibu"]) ? entries["brave-yibu"] : undefined;
  const pluginConfig = isRecord(entry?.config) ? entry.config : undefined;
  return isRecord(pluginConfig?.webSearch) ? pluginConfig.webSearch : undefined;
}

/** Resolve Yibu Brave-compatible credentials from current plugin config or legacy fallback. */
export function resolveConfiguredBraveYibuCredential(config: unknown): unknown {
  return (
    resolveBraveYibuWebSearchPluginConfig(config)?.apiKey ??
    resolveLegacyTopLevelBraveYibuCredential(config)?.value
  );
}

/** Build the common Yibu Brave-compatible provider metadata without the runtime executor. */
export function buildBraveYibuWebSearchProviderBase() {
  return {
    id: "brave-yibu",
    label: "Yibu Brave Search",
    hint: "Brave-compatible web results via yibuapi.com",
    onboardingScopes: ["text-inference"],
    credentialLabel: "Yibu Brave-compatible API key",
    envVars: ["BRAVE_API_KEY"],
    placeholder: "sk-...",
    signupUrl: "https://yibuapi.com",
    docsUrl: "https://yibuapi.com",
    autoDetectOrder: 11,
    credentialPath: BRAVE_YIBU_CREDENTIAL_PATH,
    ...createWebSearchProviderContractFields({
      credentialPath: BRAVE_YIBU_CREDENTIAL_PATH,
      searchCredential: { type: "top-level" },
      configuredCredential: { pluginId: "brave-yibu" },
    }),
    getConfiguredCredentialValue: resolveConfiguredBraveYibuCredential,
    getConfiguredCredentialFallback: resolveLegacyTopLevelBraveYibuCredential,
  };
}
