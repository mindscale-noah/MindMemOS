# 自定义模型端点

## Python 配置

所有 Agent 统一使用 `api_key`、`base_url` 直接传值：

```python
from mindmemos_skill.agents import Agent

agent = Agent.build("codex", {
    "model": "gpt-5.4-mini",
    "api_key": "your-api-key",
    "base_url": "https://gateway.example.com/v1",
})
```

模型调用必须在 config 中提供 `api_key` 和 `base_url`。省略字段或设为 `None` 时，不读取任何环境变量；实际调用前会明确报配置缺失。离线配置校验、Schema 生成和不调用模型的操作可以保留空值。密钥在 config 的 JSON 输出中脱敏。

所有内置 Env、优化器和路由器都使用对应的模型配置。服务 YAML 不展开 `${VAR}`；在 Agent、`optimizer`、`router` 等模型配置块中填写连接值。

CLI 后端将 config 值传给子进程，并过滤继承的原生模型凭据和地址。原生变量只是 CLI 的传输接口，不能作为本项目的配置来源。

按下表替换注册名、模型和 URL；模型名以你的网关为准。

| 注册名 | 模型示例 | `base_url` 示例 | 所需协议 |
|---|---|---|---|
| `openai_chat` | `deepseek-v4-flash` | `https://gateway.example.com/v1` | Chat Completions |
| `codex` | `gpt-5.4-mini` | `https://gateway.example.com/v1` | Responses |
| `claude` | `claude-haiku-4-5-20251001` | `https://gateway.example.com` | Anthropic Messages |
| `deepseek_harness` | `deepseek-v4-flash` | `https://gateway.example.com/v1` | Chat Completions |

- `deepseek_harness` 默认 `provider="deepseek-official"`。
- CLI Agent 可额外配置 `cli_path`，指定可执行文件路径。
- `claude` 调用时禁用 user/project/local settings，避免其中的 `env` 覆盖本次配置；这些来源中的其他设置也不再加载。

## OpenClaw

`openclaw` 与 Codex、Claude 使用相同的 `Agent.build()` 和异步 `run()` 接口。需预先安装 OpenClaw CLI。

```python
from pathlib import Path
from mindmemos_skill.agents import Agent

agent = Agent.build("openclaw", {
    "model": "your-model-id",
    "api_key": "your-api-key",
    "base_url": "https://gateway.example.com/v1",
    "api": "openai-completions",
    "context_window": 200000,
    "max_completion_tokens": 8192,
    "timeout_s": 300,
})
result = await agent.run(system="Follow the task instructions.", user="Summarize the files.", workspace=Path("workspace"))
```

切换模型时创建新的 Agent，修改 `model`，需要时同时修改 `base_url`、`api_key` 和 `api`。`model` 必填，填写端点实际接受的完整模型 ID；不额外添加 OpenClaw provider 前缀。`api` 支持 `openai-completions`、`openai-responses`、`anthropic-messages`。上下文容量和最大输出长度应按模型实际能力设置。

每次执行使用本地 `agent --local --json`、独立临时配置和会话，不改写个人 OpenClaw 设置。凭据通过子进程环境传递，生成的配置只保存环境引用。执行结束或异常时清理临时状态；结果保留最终响应和可用的本次会话轨迹。系统指令会作为任务消息前缀传入，不替换 OpenClaw 自身系统提示词。

工作目录必须已存在，Skill 可放在 `skills/` 或 `.agents/skills/`。工作目录设置不等于文件系统沙箱。`reasoning_effort` 映射为 OpenClaw 的 `--thinking`，需模型支持；`cli_path`、`timeout_s`、`extra_args` 与其他 CLI 后端一致。

服务的 `dev.example.yaml` 和 `dev.yaml` 提供 `native_openclaw_custom` 预设。它不会替换已有算法 Agent；仅允许 Chat Agent 的算法配置仍按原有约束校验。
