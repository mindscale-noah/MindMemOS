import { t as createSerpapiWebSearchProvider } from "./brave-web-search-provider-6mNi77fe.js";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
//#region extensions/serpapi/index.ts
/**
* SerpApi (yibu) Search plugin entry. It registers the SerpApi web-search
* provider and keeps runtime HTTP execution lazy.
*/
/** Plugin entry for SerpApi Search. */
var serpapi_default = definePluginEntry({
	id: "serpapi",
	name: "SerpApi (yibu) Plugin",
	description: "Custom SerpApi web-search plugin, routed through yibu",
	register(api) {
		api.registerWebSearchProvider(createSerpapiWebSearchProvider());
	}
});
//#endregion
export { serpapi_default as default };
