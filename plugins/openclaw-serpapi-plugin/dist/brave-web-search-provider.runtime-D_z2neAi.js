import { normalizeOptionalString } from "openclaw/plugin-sdk/string-coerce-runtime";
import { assertOkOrThrowProviderError, readProviderJsonResponse } from "openclaw/plugin-sdk/provider-http";
import { DEFAULT_SEARCH_COUNT, MAX_SEARCH_COUNT, buildSearchCacheKey, readCachedSearchPayload, readConfiguredSecretString, readPositiveIntegerParam, readProviderEnvValue, readStringParam, resolveSearchCacheTtlMs, resolveSearchCount, resolveSearchTimeoutSeconds, resolveSiteName, withSelfHostedWebSearchEndpoint, withTrustedWebSearchEndpoint, wrapWebContent, writeCachedSearchPayload } from "openclaw/plugin-sdk/provider-web-search";
import { createSubsystemLogger } from "openclaw/plugin-sdk/runtime-env";
import { assertHttpUrlTargetsPrivateNetwork, isBlockedHostnameOrIp, isPrivateIpAddress, resolvePinnedHostnameWithPolicy } from "openclaw/plugin-sdk/ssrf-runtime";
//#region extensions/serpapi/src/serpapi-web-search-provider.runtime.ts
/**
* SerpApi (yibu) Search HTTP runtime. It resolves credentials, enforces
* endpoint safety, applies caching, and maps SerpApi organic_results into the
* OpenClaw web-search result shape. This is a deliberately smaller surface
* than Brave's runtime (no llm-context mode, no Brave-specific country/
* language/freshness validation) because SerpApi's API doesn't have those
* concepts -- extra Brave-only args passed by the shared tool definition are
* simply ignored here.
*/
const DEFAULT_SERPAPI_BASE_URL = "https://yibuapi.com";
const SERPAPI_SEARCH_ENDPOINT_PATH = "/serpapi/search";
const serpapiHttpLogger = createSubsystemLogger("serpapi/http");
function logSerpapiHttp(diagnostics, event, meta) {
	if (!diagnostics?.enabled) return;
	serpapiHttpLogger.info(`serpapi http ${event}`, meta);
}
function describeSerpapiRequestUrl(url) {
	return {
		url: url.toString(),
		query: url.searchParams.get("q") ?? "",
		params: Object.fromEntries(url.searchParams.entries())
	};
}
function resolveSerpapiApiKey(searchConfig) {
	return readConfiguredSecretString(searchConfig?.apiKey, "tools.web.search.apiKey") ?? readProviderEnvValue(["BRAVE_API_KEY"]);
}
function resolveSerpapiBaseUrl(serpapiConfig) {
	return readConfiguredSecretString(serpapiConfig?.baseUrl, "plugins.entries.serpapi.config.webSearch.baseUrl")?.replace(/\/+$/u, "") || DEFAULT_SERPAPI_BASE_URL;
}
function buildSerpapiEndpointUrl(baseUrl) {
	const url = new URL(baseUrl);
	url.pathname = `${url.pathname.replace(/\/+$/u, "")}${SERPAPI_SEARCH_ENDPOINT_PATH}`;
	url.search = "";
	return url;
}
async function serpapiEndpointTargetsPrivateNetwork(url) {
	if (isBlockedHostnameOrIp(url.hostname)) return true;
	try {
		return (await resolvePinnedHostnameWithPolicy(url.hostname, { policy: {
			allowPrivateNetwork: true,
			allowRfc2544BenchmarkRange: true
		} })).addresses.every((address) => isPrivateIpAddress(address));
	} catch {
		return false;
	}
}
async function validateSerpapiBaseUrl(baseUrl) {
	let parsed;
	try {
		parsed = new URL(baseUrl);
	} catch {
		throw new Error("SerpApi base URL must be a valid http:// or https:// URL.");
	}
	if (parsed.protocol !== "http:" && parsed.protocol !== "https:") throw new Error("SerpApi base URL must use http:// or https://.");
	if (parsed.protocol === "http:") {
		await assertHttpUrlTargetsPrivateNetwork(parsed.toString(), {
			dangerouslyAllowPrivateNetwork: true,
			errorMessage: "SerpApi HTTP base URL must target a trusted private or loopback host. Use https:// for public hosts."
		});
		return "selfHosted";
	}
	return await serpapiEndpointTargetsPrivateNetwork(parsed) ? "selfHosted" : "strict";
}
function missingSerpapiKeyPayload() {
	return {
		error: "missing_serpapi_api_key",
		message: "web_search (serpapi) needs a SerpApi (yibu) API key. Set BRAVE_API_KEY in the Gateway environment, or configure plugins.entries.serpapi.config.webSearch.apiKey.",
		docs: "https://yibuapi.apifox.cn/394660230e0"
	};
}
/** Map a SerpApi organic_results[] entry into the OpenClaw web-search result shape. */
function mapSerpapiResults(data) {
	return (Array.isArray(data.organic_results) ? data.organic_results : []).map((entry) => {
		const title = entry.title ?? "";
		const url = entry.link ?? "";
		const description = entry.snippet ?? "";
		return {
			title: title ? wrapWebContent(title, "web_search") : "",
			url,
			description: description ? wrapWebContent(description, "web_search") : "",
			siteName: resolveSiteName(url) || void 0
		};
	});
}
async function runSerpapiWebSearch(params) {
	const url = buildSerpapiEndpointUrl(params.baseUrl);
	url.searchParams.set("q", params.query);
	url.searchParams.set("api_key", params.apiKey);
	if (params.count) url.searchParams.set("num", String(params.count));
	logSerpapiHttp(params.diagnostics, "request", describeSerpapiRequestUrl(url));
	const startedAt = Date.now();
	const data = await (params.endpointMode === "selfHosted" ? withSelfHostedWebSearchEndpoint : withTrustedWebSearchEndpoint)({
		url: url.toString(),
		timeoutSeconds: params.timeoutSeconds,
		init: {
			method: "GET",
			headers: { Accept: "application/json" }
		}
	}, async (response) => {
		logSerpapiHttp(params.diagnostics, "response", {
			status: response.status,
			ok: response.ok,
			durationMs: Date.now() - startedAt
		});
		await assertOkOrThrowProviderError(response, "SerpApi (yibu) error");
		return readProviderJsonResponse(response, "SerpApi (yibu) error");
	});
	return mapSerpapiResults(data);
}
/** Execute one SerpApi (yibu) search request. Brave-only args (country,
* language, freshness, ui_lang, etc.) are accepted but ignored -- SerpApi
* has no equivalent concepts in this minimal integration. */
async function executeBraveSearch(args, searchConfig, options) {
	const apiKey = resolveSerpapiApiKey(searchConfig);
	if (!apiKey) return missingSerpapiKeyPayload();
	const serpapiConfig = searchConfig?.serpapi && typeof searchConfig.serpapi === "object" && !Array.isArray(searchConfig.serpapi) ? searchConfig.serpapi : {};
	const serpapiBaseUrl = resolveSerpapiBaseUrl(serpapiConfig);
	const serpapiEndpointMode = await validateSerpapiBaseUrl(serpapiBaseUrl);
	const query = readStringParam(args, "query", { required: true });
	const count = readPositiveIntegerParam(args, "count", {
		max: MAX_SEARCH_COUNT,
		message: `count must be an integer from 1 to ${MAX_SEARCH_COUNT}.`
	}) ?? searchConfig?.maxResults ?? void 0;
	const resolvedCount = resolveSearchCount(count, DEFAULT_SEARCH_COUNT);
	const diagnostics = { enabled: options?.diagnosticsEnabled === true };
	const cacheKey = buildSearchCacheKey([
		"serpapi",
		serpapiBaseUrl,
		query,
		resolvedCount
	]);
	const cached = readCachedSearchPayload(cacheKey);
	if (cached) {
		logSerpapiHttp(diagnostics, "cache hit", { query, cacheKey });
		return cached;
	}
	logSerpapiHttp(diagnostics, "cache miss", { query, cacheKey });
	const start = Date.now();
	const timeoutSeconds = resolveSearchTimeoutSeconds(searchConfig);
	const cacheTtlMs = resolveSearchCacheTtlMs(searchConfig);
	const results = await runSerpapiWebSearch({
		baseUrl: serpapiBaseUrl,
		endpointMode: serpapiEndpointMode,
		query,
		count: resolvedCount,
		apiKey,
		timeoutSeconds,
		diagnostics
	});
	const payload = {
		query,
		provider: "serpapi",
		count: results.length,
		tookMs: Date.now() - start,
		externalContent: {
			untrusted: true,
			source: "web_search",
			provider: "serpapi",
			wrapped: true
		},
		results
	};
	writeCachedSearchPayload(cacheKey, payload, cacheTtlMs);
	logSerpapiHttp(diagnostics, "cache write", {
		query,
		cacheKey,
		ttlMs: cacheTtlMs,
		count: results.length
	});
	return payload;
}
//#endregion
export { executeBraveSearch };
