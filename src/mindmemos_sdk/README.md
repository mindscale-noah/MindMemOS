<h1>
  <img src="https://raw.githubusercontent.com/mindscale-noah/MindMemOS/main/assets/mindmemos-logo-small.png" alt="MindMemOS logo" width="40" height="40" align="absmiddle" style="vertical-align: middle;" />
  mindmemos-sdk
</h1>

![MindMemOS Memory For AI Agents](https://raw.githubusercontent.com/mindscale-noah/MindMemOS/main/assets/mindmemos-hero.png)

<p align="center">
  <a href="https://github.com/mindscale-noah/MindMemOS">
    <img src="https://img.shields.io/badge/GitHub-MindMemOS-181717?logo=github&logoColor=white" alt="MindMemOS GitHub">
  </a>
  <a href="https://mindmemos.cn">
    <img src="https://img.shields.io/badge/Website-mindmemos.cn-0A66C2?logo=googlechrome&logoColor=white" alt="MindMemOS Website">
  </a>
  <a href="https://mindmemos.cn/#/api-docs">
    <img src="https://img.shields.io/badge/FastAPI-Docs-009688?logo=fastapi&logoColor=white" alt="MindMemOS FastAPI Docs">
  </a>
  <a href="https://pypi.org/project/mindmemos-sdk/">
    <img src="https://img.shields.io/pypi/v/mindmemos-sdk?color=%2334D058&label=pypi%20sdk" alt="mindmemos-sdk PyPI version">
  </a>
  <a href="https://pypi.org/project/mindmemos-sdk/">
    <img src="https://img.shields.io/pypi/dm/mindmemos-sdk?label=pypi%20downloads" alt="mindmemos-sdk PyPI downloads">
  </a>
</p>

Python SDK and CLI for MindMemOS, a long-term memory system for AI agents and applications.

## Install

```bash
pip install mindmemos-sdk
```

The package also installs the `mindmemos` command.

## Configure

```bash
mindmemos auth
```

You can also pass `base_url`, `api_key`, and `user_id` directly when creating a client. The default `user_id`
is used by operations that require a user, such as `memory.add`. Search is different: `memory.search` only uses
the `user_id` passed to that call. Omit it for a project-wide search, or pass it explicitly for a user-scoped search.

## Python SDK

```python
from mindmemos_sdk import DialogueMessage, MindMemOSClient

with MindMemOSClient(user_id="alice", app_id="my-agent") as client:
    client.memory.add(
        messages=[
            DialogueMessage(role="user", content="I prefer iced Americano."),
        ],
    )

    result = client.memory.search(
        "What coffee does the user prefer?", top_k=5, user_id="alice", token_budget=2000
    )
    for memory in result.memories:
        print(memory.memory)
```

When the project's multimodal mode is enabled, messages accept images (local
path, URL, base64 data, or a stored `blob_id`) and search accepts `query_images`
/ `multimodal`; in off mode image sends are rejected with error codes
`add.images_unsupported` / `search.images_unsupported`. Image hits carry
`metadata.blob_id` and may carry `metadata.image_url` — a permanent
public-mirror URL or short-lived presigned link you can fetch directly, or a
signed `/v1/memory/images/{blob_id}?...&expires=<unix>&signature=<hmac>` proxy
path (resolve against your `base_url`; also fetchable directly with no API key,
until it expires) — and `client.memory.get_image(blob_id, user_id)`
re-downloads by id regardless of shape or expiry:

```python
from mindmemos_sdk import ImageRef

client.memory.add(messages=[DialogueMessage(role="user", content="my desk",
                                            images=[ImageRef(path="/data/desk.png")])])
hits = client.memory.search("desk setup", multimodal=True, user_id="alice")
data, mime_type = client.memory.get_image(blob_id, user_id="alice")
```

## CLI

```bash
mindmemos memory add --content "I prefer iced Americano" --user-id alice
mindmemos memory search "coffee preference" --top-k 5 --user-id alice --token-budget 2000
# multimodal (needs the server's multimodal mode)
mindmemos memory add --content "my desk" --image ./desk.png --user-id alice
mindmemos memory search "desk setup" --multimodal --user-id alice
```
