import { assertOkOrThrowProviderError, readProviderJsonResponse } from "openclaw/plugin-sdk/provider-http";
import {
  DEFAULT_SEARCH_COUNT,
  MAX_SEARCH_COUNT,
  buildSearchCacheKey,
  readCachedSearchPayload,
  readConfiguredSecretString,
  readPositiveIntegerParam,
  readProviderEnvValue,
  readStringParam,
  resolveSearchCacheTtlMs,
  resolveSearchCount,
  resolveSearchTimeoutSeconds,
  resolveSiteName,
  withSelfHostedWebSearchEndpoint,
  withTrustedWebSearchEndpoint,
  wrapWebContent,
  writeCachedSearchPayload,
} from "openclaw/plugin-sdk/provider-web-search";
import { createSubsystemLogger } from "openclaw/plugin-sdk/runtime-env";
import {
  assertHttpUrlTargetsPrivateNetwork,
  isBlockedHostnameOrIp,
  isPrivateIpAddress,
  resolvePinnedHostnameWithPolicy,
} from "openclaw/plugin-sdk/ssrf-runtime";

type AnyConfig = Record<string, any> | undefined;
type Diagnostics = { enabled?: boolean } | undefined;
type EndpointMode = "selfHosted" | "strict";

const DEFAULT_BRAVE_YIBU_ENDPOINT = "https://yibuapi.com/brave/v1/web/search";
const braveYibuHttpLogger = createSubsystemLogger("brave-yibu/http");

function logBraveYibuHttp(diagnostics: Diagnostics, event: string, meta: Record<string, unknown>): void {
  if (!diagnostics?.enabled) return;
  braveYibuHttpLogger.info(`brave-yibu http ${event}`, meta);
}

function describeBraveYibuRequestUrl(url: URL): Record<string, unknown> {
  return {
    url: url.toString().replace(/([?&](?:api_key|key|token)=)[^&]*/giu, "$1[REDACTED]"),
    query: url.searchParams.get("q") ?? "",
    params: Object.fromEntries(url.searchParams.entries()),
  };
}

function resolveBraveYibuApiKey(searchConfig: AnyConfig): string | undefined {
  return (
    readConfiguredSecretString(searchConfig?.apiKey, "tools.web.search.apiKey") ??
    readProviderEnvValue(["BRAVE_API_KEY"])
  );
}

function resolveBraveYibuEndpoint(braveYibuConfig: AnyConfig): string {
  return (
    readConfiguredSecretString(
      braveYibuConfig?.baseUrl,
      "plugins.entries.brave-yibu.config.webSearch.baseUrl",
    ) ||
    readProviderEnvValue(["BRAVE_YIBU_BASE_URL"]) ||
    DEFAULT_BRAVE_YIBU_ENDPOINT
  );
}

function buildBraveYibuEndpointUrl(configuredUrl: string): URL {
  const url = new URL(configuredUrl);
  if (!url.pathname || url.pathname === "/") {
    url.pathname = "/brave/v1/web/search";
  }
  url.search = "";
  return url;
}

async function braveYibuEndpointTargetsPrivateNetwork(url: URL): Promise<boolean> {
  if (isBlockedHostnameOrIp(url.hostname)) return true;
  try {
    return (
      await resolvePinnedHostnameWithPolicy(url.hostname, {
        policy: {
          allowPrivateNetwork: true,
          allowRfc2544BenchmarkRange: true,
        },
      })
    ).addresses.every((address: string) => isPrivateIpAddress(address));
  } catch {
    return false;
  }
}

async function validateBraveYibuEndpoint(configuredUrl: string): Promise<EndpointMode> {
  let parsed: URL;
  try {
    parsed = buildBraveYibuEndpointUrl(configuredUrl);
  } catch {
    throw new Error("Yibu Brave endpoint must be a valid http:// or https:// URL.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:")
    throw new Error("Yibu Brave endpoint must use http:// or https://.");
  if (parsed.protocol === "http:") {
    await assertHttpUrlTargetsPrivateNetwork(parsed.toString(), {
      dangerouslyAllowPrivateNetwork: true,
      errorMessage:
        "Yibu Brave HTTP endpoint must target a trusted private or loopback host. Use https:// for public hosts.",
    });
    return "selfHosted";
  }
  return (await braveYibuEndpointTargetsPrivateNetwork(parsed)) ? "selfHosted" : "strict";
}

function missingBraveYibuKeyPayload(): Record<string, unknown> {
  return {
    error: "missing_brave_yibu_api_key",
    message:
      "web_search (brave-yibu) needs a Yibu Brave-compatible API key. Set BRAVE_API_KEY in the Gateway environment, or configure plugins.entries.brave-yibu.config.webSearch.apiKey.",
    docs: "https://yibuapi.com",
  };
}

function optionalStringParam(args: any, name: string): string | undefined {
  const value = args?.[name];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function mapBraveLikeEntry(entry: any): Record<string, unknown> | undefined {
  const title = entry?.title ?? "";
  const url = entry?.url ?? entry?.link ?? "";
  const description = entry?.description ?? entry?.snippet ?? "";
  if (!title && !url && !description) return;
  return {
    title: title ? wrapWebContent(title, "web_search") : "",
    url,
    description: description ? wrapWebContent(description, "web_search") : "",
    siteName: resolveSiteName(url) || entry?.profile?.name || undefined,
  };
}

/** Map Brave-compatible web.results[] into the OpenClaw web-search result shape. */
function mapBraveYibuResults(data: any): Array<Record<string, unknown>> {
  const webResults = Array.isArray(data?.web?.results) ? data.web.results : [];
  const mappedWebResults = webResults.map(mapBraveLikeEntry).filter(Boolean) as Array<Record<string, unknown>>;
  if (mappedWebResults.length > 0) return mappedWebResults;

  const organicResults = Array.isArray(data?.organic_results) ? data.organic_results : [];
  const mappedOrganicResults = organicResults.map(mapBraveLikeEntry).filter(Boolean) as Array<Record<string, unknown>>;
  if (mappedOrganicResults.length > 0) return mappedOrganicResults;

  const discussionResults = Array.isArray(data?.discussions?.results) ? data.discussions.results : [];
  return discussionResults.map(mapBraveLikeEntry).filter(Boolean) as Array<Record<string, unknown>>;
}

type RunParams = {
  endpoint: string;
  endpointMode: EndpointMode;
  query: string;
  count?: number;
  apiKey: string;
  timeoutSeconds: number;
  diagnostics: Diagnostics;
  country?: string;
  language?: string;
  freshness?: string;
  searchLang?: string;
  uiLang?: string;
};

async function runBraveYibuWebSearch(params: RunParams): Promise<Array<Record<string, unknown>>> {
  const url = buildBraveYibuEndpointUrl(params.endpoint);
  url.searchParams.set("q", params.query);
  if (params.count) url.searchParams.set("count", String(params.count));
  if (params.country) url.searchParams.set("country", params.country);
  if (params.language) url.searchParams.set("language", params.language);
  if (params.freshness) url.searchParams.set("freshness", params.freshness);
  if (params.searchLang) url.searchParams.set("search_lang", params.searchLang);
  if (params.uiLang) url.searchParams.set("ui_lang", params.uiLang);

  logBraveYibuHttp(params.diagnostics, "request", describeBraveYibuRequestUrl(url));
  const startedAt = Date.now();
  const data = await (
    params.endpointMode === "selfHosted" ? withSelfHostedWebSearchEndpoint : withTrustedWebSearchEndpoint
  )(
    {
      url: url.toString(),
      timeoutSeconds: params.timeoutSeconds,
      init: {
        method: "GET",
        headers: {
          Accept: "application/json",
          Authorization: `Bearer ${params.apiKey}`,
        },
      },
    },
    async (response: any) => {
      logBraveYibuHttp(params.diagnostics, "response", {
        status: response.status,
        ok: response.ok,
        durationMs: Date.now() - startedAt,
      });
      await assertOkOrThrowProviderError(response, "Yibu Brave search error");
      return readProviderJsonResponse(response, "Yibu Brave search error");
    },
  );
  return mapBraveYibuResults(data);
}

/** Execute one Yibu Brave-compatible search request. */
export async function executeBraveYibuSearch(
  args: any,
  searchConfig: AnyConfig,
  options?: { diagnosticsEnabled?: boolean },
): Promise<Record<string, unknown>> {
  const apiKey = resolveBraveYibuApiKey(searchConfig);
  if (!apiKey) return missingBraveYibuKeyPayload();
  const braveYibuConfig =
    searchConfig?.["brave-yibu"] &&
    typeof searchConfig["brave-yibu"] === "object" &&
    !Array.isArray(searchConfig["brave-yibu"])
      ? searchConfig["brave-yibu"]
      : {};
  const endpoint = resolveBraveYibuEndpoint(braveYibuConfig);
  const endpointMode = await validateBraveYibuEndpoint(endpoint);
  const query = readStringParam(args, "query", { required: true });
  const count =
    readPositiveIntegerParam(args, "count", {
      max: MAX_SEARCH_COUNT,
      message: `count must be an integer from 1 to ${MAX_SEARCH_COUNT}.`,
    }) ??
    searchConfig?.maxResults ??
    undefined;
  const resolvedCount = resolveSearchCount(count, DEFAULT_SEARCH_COUNT);
  const diagnostics: Diagnostics = { enabled: options?.diagnosticsEnabled === true };
  const country = optionalStringParam(args, "country");
  const language = optionalStringParam(args, "language");
  const freshness = optionalStringParam(args, "freshness");
  const searchLang = optionalStringParam(args, "search_lang");
  const uiLang = optionalStringParam(args, "ui_lang");
  const cacheKey = buildSearchCacheKey([
    "brave-yibu",
    endpoint,
    query,
    resolvedCount,
    country ?? "",
    language ?? "",
    freshness ?? "",
    searchLang ?? "",
    uiLang ?? "",
  ]);
  const cached = readCachedSearchPayload(cacheKey);
  if (cached) {
    logBraveYibuHttp(diagnostics, "cache hit", { query, cacheKey });
    return cached;
  }
  logBraveYibuHttp(diagnostics, "cache miss", { query, cacheKey });
  const start = Date.now();
  const timeoutSeconds = resolveSearchTimeoutSeconds(searchConfig);
  const cacheTtlMs = resolveSearchCacheTtlMs(searchConfig);
  const results = await runBraveYibuWebSearch({
    endpoint,
    endpointMode,
    query,
    count: resolvedCount,
    apiKey,
    timeoutSeconds,
    diagnostics,
    country,
    language,
    freshness,
    searchLang,
    uiLang,
  });
  const payload = {
    query,
    provider: "brave-yibu",
    count: results.length,
    tookMs: Date.now() - start,
    externalContent: {
      untrusted: true,
      source: "web_search",
      provider: "brave-yibu",
      wrapped: true,
    },
    results,
  };
  writeCachedSearchPayload(cacheKey, payload, cacheTtlMs);
  logBraveYibuHttp(diagnostics, "cache write", {
    query,
    cacheKey,
    ttlMs: cacheTtlMs,
    count: results.length,
  });
  return payload;
}
