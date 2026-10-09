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
- 三个 CLI Agent 可额外配置 `cli_path`，指定可执行文件路径。
- `claude` 调用时禁用 user/project/local settings，避免其中的 `env` 覆盖本次配置；这些来源中的其他设置也不再加载。
