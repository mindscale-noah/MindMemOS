import { createBraveYibuWebSearchProvider } from "./brave-yibu-web-search-provider.js";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

/** Plugin entry for Yibu Brave-compatible Search. */
const braveYibuDefault = definePluginEntry({
  id: "brave-yibu",
  name: "Yibu Brave Search Plugin",
  description: "Custom Yibu Brave-compatible web-search plugin",
  register(api: any) {
    api.registerWebSearchProvider(createBraveYibuWebSearchProvider());
  },
});

export default braveYibuDefault;
