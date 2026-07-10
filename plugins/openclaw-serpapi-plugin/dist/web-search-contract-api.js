import { buildBraveWebSearchProviderBase } from "./web-search-shared.js";
//#region extensions/brave/web-search-contract-api.ts
/** Create the Brave provider descriptor for contract checks. */
function createBraveWebSearchProvider() {
	return {
		...buildBraveWebSearchProviderBase(),
		createTool: () => null
	};
}
//#endregion
export { createBraveWebSearchProvider };
