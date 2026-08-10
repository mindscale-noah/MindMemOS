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

Authentication, CLI settings, the local UI, and Python clients now read and
write `~/.mindmemos/config.yaml` exclusively. `settings.json` has no runtime
write path and is recognized only as an automatic one-time v1 migration source.

You can also pass `base_url`, `api_key`, and `user_id` directly when creating a client.

Existing `settings.json` v1 configuration is converted automatically on the
first v2 SDK or Skill configuration load. To preview or apply the same
deterministic migration manually:

```bash
mindmemos config migrate
mindmemos config migrate --apply
```

The apply command writes `config.yaml`, preserves `settings.json`, and creates
`settings.json.v1.bak`. It migrates configuration only: local Skill manifests,
history, caches, outbox files, and `state.db` are not scanned or modified.

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
mindmemos memory add --content "I prefer iced Americano" --user-id alice
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

# Inspect history and export either the deterministic latest version or an explicit version.
mindmemos skill history my-skill
mindmemos skill export my-skill --version <version-uuid> --to ./restored-skill

# Local and cloud bundles contain only canonical SKILL.md.
# Scripts, resources, references, assets, config, logs, and other files stay local.
mindmemos skill push my-skill
mindmemos skill pull my-skill
mindmemos skill sync my-skill
```

Neither edge nor cloud stores an active or published-head pointer. When a
version is omitted, register, publish, export, run, and injection use the same
latest-available selector: `draft/published`, ordered by
`(created_at DESC, version_id DESC)`. Evolution and merge always require explicit
base or parent version IDs.

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
latest = manager.latest_skill_context(registered.skill_id)
```

Backend Skill calls use the MindMemOS HTTP contract, while `skills.local`
remains the SDK-owned local version manager:

```python
from mindmemos_sdk import AsyncSkillClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.transport import AsyncHttpTransport

skills = AsyncSkillClient.from_http(
    AsyncHttpTransport(base_url=settings.base_url, api_key=settings.auth.api_key),
    config_manager=ConfigManager(),
    owns_transport=True,
)

local_versions = skills.local.list_local()
backend_skills = await skills.list_skills()
```

Run `mindmemos ui` for the browser UI. Its editor creates browser-only drafts;
publishing creates a new immutable UUID version instead of modifying an existing
version in place.
