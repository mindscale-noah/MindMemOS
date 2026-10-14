# AgentSkillOS 建图与检索评分

从仓库根目录运行 `scripts.skill_eval.agentskillos.evaluate`，通过 MindMemOS SDK 注册完整候选池、调用建图服务，并在同一个图快照上统计 Recall 和 MRR。脚本复用当前 AgentSkillOS 的指标计算函数，与旧项目的核心标签检索评测口径一致。

这是从客户端注册、建图到检索评分的端到端测评。分数反映 Skill 检索质量；AgentSkillOS 任务执行成功率需要另行运行官方任务评测。

## 开始运行

先启动配置了 `skill_manage` 管理 pipeline 的 Skill HTTP 服务，并确认服务端聊天模型、Embedding 配置和鉴权可用。需要 `skill:write` 和 `skill:read` 权限。项目开发环境须已安装 SDK 和 Skill 包。

当前 `config/mindmemos_skill/dev.yaml` 与 `dev.example.yaml` 的建图模型已设为 `deepseek-v4-flash`（`max_completion_tokens=32768`、单次请求 `timeout_s=1200`），Embedding 已设为 `qwen3-embedding-4b`，预期向量维度为 `2560`，`send_dimensions=false`。实际模型由服务端配置决定，启动或重启服务后才会生效。模型 API 地址必须指向提供这些模型的 OpenAI 兼容服务；SDK 的 `--base-url` 指向 Skill 服务，不是模型服务。

明确指定服务地址时，脚本从 `MINDMEMOS_SKILL_API_KEY` 环境变量读取凭据。本地服务若使用无需凭据的鉴权模式，可以不设置此变量。服务地址和数据路径应替换为实际值：

```bash
uv run python -m scripts.skill_eval.agentskillos.evaluate \
  --dataset-root /path/to/agentskillos_eval \
  --base-url http://127.0.0.1:8000 \
  --output output/agentskillos_graph_run1
```

省略 `--base-url` 时，使用 SDK 已保存的 Skill 服务连接和凭据。可用 `--sdk-home /path/to/sdk-settings-directory` 选择保存了 `settings.json` 的目录。使用模块形式 `python -m ...` 从仓库根目录启动。

脚本接受两种数据目录：

| 数据目录 | 必需内容 |
| --- | --- |
| 原始旧项目 `agentskillos_eval` | `tasks.jsonl`、`relevance.json`、`ssl_pool_join.json` 的 `join` 映射、`skill_pool_200/<source_skill_id>/SKILL.md` 及资源 |
| 旧准备脚本生成的目录 | `queries/tasks.jsonl`、`queries/relevance.json`、`skill_pool_manifest.json`、`splits.json`、manifest 中每个 `relative_path` 指向的 Skill 目录 |

任务文件每行包含 `task_id` 和 `instruction_text`。标签文件以任务 ID 为键，使用 `core_gt_ids`；该字段缺失或为 `null` 时回退到 `gt_skill_ids`。`task_type="generic_only"` 的任务跳过评分。其余任务必须有非空且唯一的核心标签，标签 ID 必须属于完整候选池。准备后的数据必须只有一个覆盖全部任务的 `test` split。

注册保留原始 `SKILL.md` 和资源，评分通过“服务版本 ID → 数据源 Skill ID → benchmark member ID”映射，不依赖 Skill 名称恰好与标签相同。私有标签只在客户端评分时使用，检索请求只发送公开 instruction。

## 参数与评分口径

默认使用 `--strategy graph`、`--top-k 50`。也可用 `--strategy vector` 测试同一建图流程后的向量检索；`--include-scenes` 只在向量模式下生效。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--graph-alias` | 不设置 | 为新建的图设置别名；应使用未占用的别名 |
| `--reuse-registrations` | 不设置 | 复用完整的旧 `registrations.json`，跳过重复注册 |
| `--force-extract-scene` | `False` | 请求服务重新抽取 Scene |
| `--http-timeout` | `3600` 秒 | 单次 HTTP 请求超时，包含同步建图请求 |
| `--task-timeout` | `7200` 秒 | 异步建图任务等待上限 |
| `--poll-interval` | `5` 秒 | 异步任务状态查询间隔 |
| `--output` | `output/agentskillos_sdk_时间戳` | 新输出目录；已有目录会被拒绝 |

注册与查询串行执行，并发为 1；服务端建图并发由服务配置控制。默认新注册候选池并新建图，因此会产生服务端 Skill 和图记录，不自动清理。若注册已完成、建图失败，可用 `--reuse-registrations /path/to/registrations.json` 复用完整的精确版本映射，再建一张新图；脚本会拒绝缺少成员、重复版本或空版本 ID。切换模型时同时使用 `--force-extract-scene`，避免复用旧模型的 Scene 缓存。脚本不复用已有图或部分评分结果。

- `Recall@10`、`Recall@20`、`Recall@50`：前 K 名命中的核心标签数，除以该任务完整核心标签数。
- `MRR@10`：前 10 名首次命中核心标签的排名倒数；前 10 名无命中时为 0。
- 同时输出旧口径的 nDCG、Hit、Precision 和 FullCoverage，共 12 个指标。
- 汇总对已成功评分的任务取宏平均。候选不足 K 时按实际返回的结果评分，核心标签分母保持不变。
- 检索异常单独记录，不按正常未命中处理。出现异常时报告 `status="partial"`、评分覆盖率和错误任务，并返回非零退出码。

为完整报告 Recall@50，脚本要求 `--top-k >= 50`。所有查询使用建图返回的 `graph_version_id`，并检查服务实际返回的快照和候选版本范围。

## 查看结果

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 汇总指标、查询数、候选数、跳过任务、错误任务、评分覆盖率、检索参数与图快照 |
| `results.jsonl` | 逐任务核心标签、排名、服务版本 ID 和指标；失败任务记录错误 |
| `registrations.json` | 数据源 Skill ID 到注册版本 ID 的映射，每完成一个注册立即保存 |
| `organization.json` | 建图提交返回值；异步模式包含任务 ID |
| `graph.json` | 发布后的图 ID、固定快照 ID 和全部成员版本 |
| `pool.json` | 本次候选池的源 ID、benchmark member ID 和目录映射 |

建图失败或超时时不会生成最终评分汇总，已保存的注册和任务信息可用于定位服务状态。异步等待超时不会取消服务端任务。实时日志显示注册数量、建图任务状态和已评分／失败查询数量。

本地回归验证：

```bash
uv run python -m pytest tests/scripts/test_agentskillos_sdk_evaluation.py -q
```

这些测试使用离线 HTTP 响应验证 SDK 请求与评分流程。真实模型效果需通过上述命令连接实际 Skill 服务运行后，以 `summary.json` 为准。

## 排查模型与建图失败

Scene 抽取通过 `skill_manage.scene.extract` 和 `skill_manage.scene.model_call` 两层 OTel span 记录耗时、调用次数和 token 用量。服务开启 OTLP 后，可在现有 Grafana/ClickHouse 中按请求 trace ID 查调用；HTTP 请求成功不代表建图已完成，以图发布和评分文件为准。

`scene.extraction.transport_failed` 记录异常类型、可用的 HTTP 状态，以及服务能识别的截断或空响应原因；`scene.extraction.invalid_output` 记录校验错误类型，例如 `scene_entry_missing`、`scene_step_ownership_invalid`、`scene_node_unreachable`，以区分具体结构约束。诊断不保存 Skill 原文、模型回复或供应商错误正文。`completion_truncated` 表示输出预算耗尽，应先用相同文档复测合适的预算，不能把截断输出当作无 Scene。`insufficient_source` 则保留 Skill 节点并缓存空 Scene，抽取 span 标记为成功并记录 `extraction.outcome=insufficient_source`，不计作模型故障。

关系生成的每个批次通过 `skill_manage.build.linking.batch` span 记录耗时。JSON 结构、候选对归属和必需权重校验均位于共享 Agent 的解析重试范围内，按 `parse_max_attempts` 和 `parse_retry_mode` 执行；只重试失败批次，成功批次不重复调用。`graph.relations.invalid_output` 只记录错误分类。严格模式下，重试耗尽仍中止建图，不丢弃错误关系后继续发布。

## 离线复核已保存的排名

`scripts.skill_eval.agentskillos.replay` 保留旧项目的离线评分与基线比较流程。排名文件是 `{任务 ID: [数据源 Skill ID, ...]}`；任务、标签和排名必须覆盖同一组任务。所有指标复用同一计算函数，已有输出文件会被拒绝覆盖。

```bash
uv run python -m scripts.skill_eval.agentskillos.replay \
  --tasks /path/to/tasks.jsonl \
  --relevance /path/to/relevance.json \
  --rankings /path/to/rankings.json \
  --baseline /path/to/baseline.json \
  --protocol legacy_replay \
  --output output/agentskillos_replay.json
```

基线可省略。报告含逐任务和汇总指标；提供基线时还输出指标差值与 top-K 集合重叠率。此入口只重算保存的排名，不调用服务或模型，不能单独证明端到端建图成功。脚本测试位于 `tests/scripts/test_agentskillos_replay.py`。
