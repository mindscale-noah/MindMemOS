import { buildSerpapiWebSearchProviderBase } from "./web-search-shared.js";

/** Create the SerpApi provider descriptor for contract checks. */
export function createSerpapiWebSearchProvider() {
  return {
    ...buildSerpapiWebSearchProviderBase(),
    createTool: () => null,
  };
}
