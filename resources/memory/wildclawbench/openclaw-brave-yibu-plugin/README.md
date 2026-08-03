# Yibu Brave Search Plugin

This is the OpenClaw web-search provider used by the WildClawBench evaluation
image. It calls Yibu's Brave-compatible API.

Runtime behavior:

- provider id: `brave-yibu`
- default endpoint: `https://yibuapi.com/brave/v1/web/search`
- auth: `Authorization: Bearer <BRAVE_API_KEY>`
- response shape: Brave-compatible `web.results[]`

Typical OpenClaw config inside the eval image:

```json
{
  "plugins": {
    "entries": {
      "brave-yibu": {
        "enabled": true,
        "config": {
          "webSearch": {
            "apiKey": "${BRAVE_API_KEY}",
            "baseUrl": "https://yibuapi.com/brave/v1/web/search"
          }
        }
      }
    }
  },
  "tools": {
    "web": {
      "search": {
        "provider": "brave-yibu"
      }
    }
  }
}
```

Use `scripts/wildclawbench/install_brave_yibu_plugin.sh` to build the standard
search-enabled evaluation image. The default target image is
`wildclawbench-mindmemos:v1.3-brave-yibu`.
