import { buildBraveYibuWebSearchProviderBase } from "./web-search-shared.js";

/** Create the Yibu Brave-compatible provider descriptor for contract checks. */
export function createBraveYibuWebSearchProvider() {
  return {
    ...buildBraveYibuWebSearchProviderBase(),
    createTool: () => null,
  };
}
