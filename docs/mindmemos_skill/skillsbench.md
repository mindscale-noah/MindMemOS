# SkillsBench 测评环境

SkillsBench 使用[官方 v1.0 任务](https://github.com/benchflow-ai/skillsbench/releases/tag/v1.0)，通过 `mindmemos-eval skill test` 调用 `harbor run`，由 Harbor 管理任务容器、原生 Agent 和官方 verifier。

## 准备数据与配置

依赖 Docker、Docker Compose 和独立安装的 Harbor CLI。Docker daemon 须能访问任务和输出目录。

```bash
uv tool install --python 3.12 'harbor==0.24.0'
.venv/bin/python scripts/download_datasets.py download skill \
  --name skillsbench --data-root data/mindmemos_skill
```

任务目录为 `data/mindmemos_skill/skillsbench/tasks/`。测评配置写入 `config/mindmemos_eval/env_test/dev.yaml`：

```yaml
environment:
  class_name: skillsbench
  config:
    agent: oracle
data:
  class_name: skillsbench
  config:
    benchmark_root: data/mindmemos_skill/skillsbench
    task_ids: [offer-letter-generator]
```

使用 `oracle` 运行参考解和官方 verifier：

```bash
.venv/bin/mindmemos-eval skill test \
  --config config/mindmemos_eval/env_test/dev.yaml --no-skill
```

## 使用 Skills 测评

`oracle` 用于参考解校验。模型测评时，将 `environment.config` 设为：

```yaml
agent: claude-code
model: anthropic/your-model
agent_env: [ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL]
```

在运行终端设置连接环境变量，`model` 填写实际模型名。Codex 使用 `agent: codex`、`model: openai/your-model` 和 `agent_env: [OPENAI_API_KEY, OPENAI_BASE_URL]`。

通过 `--skills` 传入一个或多个完整 Skill 目录，使用 `--no-skill` 运行无 Skills 对照：

```bash
.venv/bin/mindmemos-eval skill test \
  --config config/mindmemos_eval/env_test/dev.yaml \
  --skills /path/to/skills/first /path/to/skills/second
```

所选任务使用同一组 Skills。默认输出位于 `output/mindmemos_eval/test/`，包含报告、轨迹和 Harbor 产物。`errors` 记录执行异常，异常不参与评分均值。

轨迹上传与演进通过 SDK/API 调用，见 [Offline Evolution](offline_evolution.md)。将保存的版本导出为 Skill 目录后，通过 `--skills` 复测。
