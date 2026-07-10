import { createSerpapiWebSearchProvider } from "./serpapi-web-search-provider.js";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

/**
 * SerpApi (yibu) Search plugin entry. It registers the SerpApi web-search
 * provider and keeps runtime HTTP execution lazy.
 */

/** Plugin entry for SerpApi Search. */
const serpapi_default = definePluginEntry({
  id: "serpapi",
  name: "SerpApi (yibu) Plugin",
  description: "Custom SerpApi web-search plugin, routed through yibu",
  register(api: any) {
    api.registerWebSearchProvider(createSerpapiWebSearchProvider());
  },
});

export default serpapi_default;
