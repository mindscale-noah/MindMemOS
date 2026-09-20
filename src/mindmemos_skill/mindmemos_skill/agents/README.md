# 自定义模型端点

## Python 配置

四个 Agent 统一使用 `api_key`、`base_url` 直接传值：

```python
from mindmemos_skill.agents import Agent

agent = Agent.build("codex", {
    "model": "gpt-5.4-mini",
    "api_key": "your-api-key",
    "base_url": "https://gateway.example.com/v1",
})
```

未提供的字段（或 `None`）在构建 config 时分别读取统一环境变量：

```bash
export MINDMEMOS_AGENT_API_KEY='your-api-key'
export MINDMEMOS_AGENT_BASE_URL='https://gateway.example.com/v1'
```

```python
agent = Agent.build("codex", {"model": "gpt-5.4-mini"})
```

优先级：**config 直接值 > 统一环境变量**。两者都未配置时，CLI 沿用原生默认配置；
`openai_chat` 必须提供 API key。如使用 `.env`，在构建 config 前调用 `load_dotenv()`。
密钥在 config 的 JSON 输出中脱敏。

按下表替换注册名、模型和 URL；模型名以你的网关为准。

| 注册名 | 模型示例 | `base_url` 示例 | 所需协议 |
|---|---|---|---|
| `openai_chat` | `deepseek-v4-flash` | `https://gateway.example.com/v1` | Chat Completions |
| `codex` | `gpt-5.4-mini` | `https://gateway.example.com/v1` | Responses |
| `claude` | `claude-haiku-4-5-20251001` | `https://gateway.example.com` | Anthropic Messages |
| `deepseek_harness` | `deepseek-v4-flash` | `https://gateway.example.com/v1` | Chat Completions |

- `deepseek_harness` 默认 `provider="deepseek-official"`。
- 三个 CLI Agent 可额外配置 `cli_path`，指定可执行文件路径。
- `claude` 从 config 或统一环境变量获得端点或凭据时，会禁用 user/project/local settings，避免其中的 `env` 覆盖本次配置；这些来源中的其他设置也不再加载。

## 直接使用 CLI

以下命令将统一变量转换为 CLI 原生变量，在准备好的任务目录执行，需要已安装对应 CLI。`openai_chat` 直接调用 SDK，没有独立 harness CLI。

### Codex

```bash
export CODEX_API_KEY="$MINDMEMOS_AGENT_API_KEY"
codex exec --json --ephemeral --skip-git-repo-check --sandbox workspace-write \
  --model gpt-5.4-mini \
  -c 'approval_policy="never"' \
  -c 'model_provider="custom"' \
  -c 'model_providers.custom.name="Custom"' \
  -c 'model_providers.custom.base_url="https://gateway.example.com/v1"' \
  -c 'model_providers.custom.env_key="CODEX_API_KEY"' \
  -c 'model_providers.custom.wire_api="responses"' \
  -c 'model_providers.custom.supports_websockets=false' \
  'Reply with ENDPOINT_OK.'
```

### Claude

```bash
ANTHROPIC_AUTH_TOKEN="$MINDMEMOS_AGENT_API_KEY" \
ANTHROPIC_BASE_URL='https://gateway.example.com' \
claude --print --output-format stream-json --verbose \
  --setting-sources '' --no-session-persistence --permission-mode dontAsk \
  --model claude-haiku-4-5-20251001 \
  'Reply with ENDPOINT_OK.'
```

### DeepSeek Harness

创建 `dsh-model.yml`：

```yaml
- id: agent-default-model
  config:
    provider: deepseek-official
    model: deepseek-v4-flash
```

使用独立的临时 `DSH_HOME`，避免加载已有用户配置：

```bash
DEEPSEEK_API_KEY="$MINDMEMOS_AGENT_API_KEY" \
DEEPSEEK_BASE_URL='https://gateway.example.com/v1' \
DSH_HOME="$(mktemp -d)" \
dsh --profile headless --patch ./dsh-model.yml 'Reply with ENDPOINT_OK.'
```
