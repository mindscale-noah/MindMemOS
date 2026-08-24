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
  <a href="https://mindmemos.cn/api-docs">
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

You can also pass `base_url`, `api_key`, and `user_id` directly when creating a client.

## Python SDK

```python
from mindmemos_sdk import DialogueMessage, MindMemOSClient

with MindMemOSClient(user_id="alice", app_id="my-agent") as client:
    client.memory.add(
        messages=[
            DialogueMessage(role="user", content="I prefer iced Americano."),
        ],
    )

    result = client.memory.search("What coffee does the user prefer?", top_k=5)
    for memory in result.memories:
        print(memory.memory)
```

## CLI

```bash
mindmemos memory add --content "Installed pandas successfully" --task "Install pandas" --user-id alice
mindmemos memory search "coffee preference" --top-k 5 --user-id alice
```

## Local Skill management

The SDK keeps immutable Skill versions below `~/.mindmemos`. Source directories
are read only when you explicitly register or publish, and are never tracked as
the runtime source of truth.

```bash
# Import a complete snapshot and create its root UUID version.
mindmemos skill register ./my-skill --alias my-skill -m "Initial import"

# Snapshot a later directory state as an immutable child version.
mindmemos skill publish my-skill --from ./my-skill -m "Improve tool guidance"

# Inspect history, switch only the local active pointer, and export explicitly.
mindmemos skill history my-skill
mindmemos skill switch my-skill --to <version-uuid>
mindmemos skill export my-skill --version <version-uuid> --to ./restored-skill

# Cloud operations preserve version UUIDs and do not upload linked files.
mindmemos skill push my-skill
mindmemos skill pull my-skill
mindmemos skill sync my-skill
mindmemos skill promote my-skill --version <version-uuid>
```

`switch` and `rollback` change only `active_version_id`. `promote` changes only
the cloud `published_head_id`. Push, pull, and sync do not implicitly change the
local active version.

The same control plane is available through `SkillManager`:

```python
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import (
    RegisterLocalRequest,
    SkillCloudClient,
    SkillManager,
)
from mindmemos_sdk.transport import HttpTransport

config = ConfigManager()
settings = config.load_or_default()
manager = SkillManager.from_config_manager(
    config,
    SkillCloudClient(
        HttpTransport(
            base_url=settings.base_url,
            api_key=settings.auth.api_key,
        )
    ),
)
registered = manager.register_local(
    RegisterLocalRequest(source_path="./my-skill", alias="my-skill")
)
active = manager.active_skill_context(registered.skill_id)
```

Backend Skill calls use one SDK DTO surface. API mode uses the MindMemOS HTTP
contract, while Lite mode borrows an already-running transport-neutral Skill
service. In both cases `skills.local` remains the SDK-owned local version
manager:

```python
from mindmemos_sdk import AsyncSkillClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.transport import AsyncHttpTransport

# MindMemOS API mode
skills = AsyncSkillClient.from_http(
    AsyncHttpTransport(base_url=settings.base_url, api_key=settings.auth.api_key),
    config_manager=ConfigManager(),
    owns_transport=True,
)

# MindMemOS Lite mode
skills = AsyncSkillClient.from_lite_runtime(
    runtime,
    project_id="local-project",
    config_manager=ConfigManager(),
)

local_versions = skills.local.list_local()
backend_skills = await skills.list_skills()
```

The Lite runtime/service is borrowed: the SDK neither starts nor closes it.
`mindmemos_lite` is imported only when a Lite factory is used. The runtime must
have been composed with a concrete transport-neutral `SkillService`; the Lite
package's service `Protocol` is not instantiated as an implementation.

Run `mindmemos ui` for the browser UI. Its editor creates browser-only drafts;
publishing creates a new immutable UUID version instead of modifying an existing
version in place. The **Lite → Traces** view accepts a Lite observability
directory (the default runtime location is `.mindmemos/observability`) or a
specific SQLite file. It lists individual spans with server-side name filtering
and pagination. Selecting any span resolves its root span, displays the ancestry
path, and renders the complete trace as a clickable flame chart. SQLite
databases are opened read-only; input and output content is available only when
the Lite runtime wrote it with `observability.capture_content: true`.
