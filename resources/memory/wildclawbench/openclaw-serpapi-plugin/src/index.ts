import { createSerpapiWebSearchProvider } from "./serpapi-web-search-provider.js";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

/**
 * SerpApi-compatible Search plugin entry. It registers the SerpApi web-search
 * provider and keeps runtime HTTP execution lazy.
 */

/** Plugin entry for SerpApi Search. */
const serpapi_default = definePluginEntry({
  id: "serpapi",
  name: "SerpApi Plugin",
  description: "Custom SerpApi-compatible web-search plugin",
  register(api: any) {
    api.registerWebSearchProvider(createSerpapiWebSearchProvider());
  },
});

export default serpapi_default;
