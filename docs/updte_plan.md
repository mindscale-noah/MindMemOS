# Skill Judge / Trajectory Judge 接入与 Skill Manage 评测展示计划

## 1. 背景与目标

当前分支已经合入 `algos/analyze` 下的静态 Skill Judge 与 Trajectory Judge 算法，但算法目前只提供纯计算能力，没有持久化、服务编排和 HTTP 接口，也没有接入 Skill Manage 的 Recommend 主流程。

本计划的目标如下：

1. 保持 `SkillJudge` 与 `TrajectoryJudge` 为纯分析算法，不直接访问数据库。
2. 新增独立的评测结果存储，不与 Skill 家族、Skill 版本或 Skill 图混存。
3. Judge 请求优先读取数据库中的 `completed` 结果；命中时直接返回，不调用模型。
4. 未命中时执行原有 Judge 流程，成功后保存结果；失败时保存 `failed` 记录，并保留原始异常。
5. Recommend 检索完成后，可以在调用方明确开启时附带相关评测结果；默认请求和响应保持现有行为。
6. Recommend 内部通过持久化 DataService 读取评测，不发起内部 HTTP 请求。
7. 尽可能保持现有 Manage 用户接口和检索算法不变，只增加可选后处理。
8. 变更、测试和提交均限制在本任务范围内，不提交现有无关未跟踪文件。

## 2. 存储结论

### 2.1 Skill 家族和版本

- `skills_v1` 使用 `(project_id, skill_id)` 标识一个 Skill 家族。
- `skill_versions_v1` 使用 `version_id` 作为主键，同时通过约束和索引保证 `(project_id, version_id)` 唯一。
- Skill 版本是 Skill 家族下不可变的源内容快照，适合引用，不适合附加可重复计算、可失败重试的 Judge 结果。

### 2.2 Skill 图

- `skill_graphs_v1` 使用 `(project_id, graph_id)` 标识图家族。
- `skill_graph_versions_v1` 使用 `(project_id, graph_id, graph_version_id)` 标识图的不可变快照。
- `skill_graph_nodes_v1` 通过 `skill_version_id` 引用 Skill 版本。
- 图版本是组织关系快照，不是 Skill 版本；同一 Skill 版本可以出现在不同图中，不能把 Judge 结果存入图版本。

### 2.3 结论

新增 `skill_evaluations_v1`，由独立 `EvaluationDataService` 管理。Judge 结果只通过 `project_id + skill_version_id`、`project_id + trajectory_id` 等业务外键关联，不向 Skill 或 Graph 表增加 Judge 字段，也不建立数据库外键耦合，避免 Analyze 存储演进影响 Manage 主流程。

## 3. 评测缓存与幂等规则

### 3.1 静态 Skill Judge key

`evaluation_key` 由以下规范化内容计算 SHA-256：

- `judge_type = static_skill`
- `project_id`
- `skill_version_id`
- `skill_hash` 或 `content_hash`
- `algorithm`
- `algorithm_version`
- `prompt_version`
- 影响结果的 Judge 配置

`request_id` 只用于追踪，不进入缓存 key。

### 3.2 Trajectory Judge key

`evaluation_key` 由以下规范化内容计算 SHA-256：

- `judge_type = trajectory`
- `project_id`
- `trajectory_id`
- `attribute`
- `verifier_score`
- `algorithm`
- `algorithm_version`
- `prompt_version`
- 影响结果的 Judge 配置

### 3.3 状态规则

- 只有 `status = completed` 的记录可以命中缓存。
- `failed` 记录保留错误信息和请求上下文，但不命中缓存，后续请求允许重试。
- 相同 `(project_id, evaluation_key)` 使用幂等写入；重复请求更新同一逻辑结果。
- 保存异常不能静默吞掉主分析错误；记录日志后按任务语义处理。
- 长耗时模型调用不得放在数据库事务中。

## 4. 分阶段实施

| 阶段 | 内容 | 主要文件 |
| --- | --- | --- |
| 0 | 冻结接口、key、状态和数据契约 | `docs/updte_plan.md` |
| 1 | 新增评测记录、表模型、写入模型和 DataService | `persistence/db_services/evaluations/*`、`persistence/constants.py` |
| 2 | 注册并装配独立评测仓储 | `persistence/tables.py`、`session.py`、`store.py`、`persistence/__init__.py` |
| 3 | 新增 Analyze Pipeline，统一缓存、执行、保存和失败重试 | `pipelines/analyze.py`、`pipelines/builtin_pipelines/analyze/*` |
| 4 | 接入配置、校验、工厂和 SkillService | `config/config.py`、`config/loader.py`、`config/schema.py`、`api/factory.py`、`api/service.py` |
| 5 | 新增 Analyze 与 Evaluation HTTP 契约和路由 | `api/typing.py`、`api/entries/typing.py`、`api/entries/fastapi.py` |
| 6 | Recommend 可选附带评测，保持默认响应兼容 | `api/typing.py`、`api/service.py`、`api/entries/fastapi.py` |
| 7 | 同步开发配置模板和本地配置 | `config/mindmemos_skill/dev.example.yaml`、`config/mindmemos_skill/dev.yaml` |
| 8 | 补齐持久化、缓存、服务、HTTP 和 Recommend 兼容测试 | `src/mindmemos_skill/tests/*` |
| 9 | 执行静态检查与测试，修复回归 | 全量相关测试范围 |
| 10 | 精确暂存并提交 | 仅本任务文件 |

## 5. 详细修改计划

### 5.1 评测持久化

新增目录：

```text
src/mindmemos_skill/mindmemos_skill/persistence/db_services/evaluations/
```

新增内容：

- `typing.py`
  - `EvaluationRecord`
  - `EvaluationSummary`
  - `EvaluationWrite`
  - 固定字段：`evaluation_id`、`project_id`、`evaluation_key`、`judge_type`、`skill_id`、`skill_version_id`、`trajectory_id`、`algorithm`、`algorithm_version`、`prompt_version`、`status`、`summary`、`score_mean`、`scores`、`success`、`confidence`、`skill_usage_rate`、`metrics`、`findings`、`artifacts`、`metadata`、`error`、`created_at`、`updated_at`
- `tables.py`
  - 表名 `skill_evaluations_v1`
  - 主键 `evaluation_id`
  - 唯一约束 `(project_id, evaluation_key)`
  - 索引 `(project_id, skill_version_id, judge_type, created_at)`
  - 索引 `(project_id, trajectory_id, created_at)`
  - 索引 `(project_id, status, created_at)`
- `service.py`
  - `save_completed(...)`
  - `save_failed(...)`
  - `get_completed(evaluation_key)`
  - `latest_for_versions(skill_version_ids, judge_types=None)`
  - `query(...)`
- `__init__.py`
  - 导出公开类型和 DataService

在 `persistence/tables.py` 中组合新表的 `build_tables()`；在 `SkillPersistenceSession` 中新增 `self.evaluations`；在 `SkillPersistence` 中新增 `evaluations` 属性。评测仓储只拥有自己的表，不依赖 Skill 或 Graph 仓储。

### 5.2 Analyze Pipeline

新增抽象和内置实现：

```text
pipelines/analyze.py
pipelines/builtin_pipelines/analyze/__init__.py
pipelines/builtin_pipelines/analyze/config.py
pipelines/builtin_pipelines/analyze/pipeline.py
```

职责：

1. 接收静态 Skill 或 Trajectory 分析请求。
2. 构造稳定 `evaluation_key`。
3. 查询 `completed` 结果，命中则直接转换为分析结果。
4. 未命中时构造并执行纯 Judge 算法。
5. 成功后将结果规范化并保存为 `completed`。
6. 失败后保存 `failed` 记录，并重新抛出原异常。
7. 保持模型调用在持久化事务之外。
8. 提供按项目查询历史评测结果的能力。

配置模型包含：

- 可选 `skill_judge` 配置
- 可选 `trajectory_judge` 配置
- 至少启用一种 Judge，或允许仅查询已有评测
- 配置使用现有 `AgentConfig` 注册机制解析 Agent 预设

### 5.3 Service 与 Factory 接入

- `ServiceConfig` 新增可选 `analyze_pipeline: ImplementationConfig | None`。
- `config/loader.py` 校验 Analyze Pipeline 必须依赖全局 persistence。
- `config/schema.py` 增加 Analyze Pipeline 的显式分支。
- `api/factory.py` 构造 `AnalyzePipeline` 并注入 `SkillService`。
- `SkillService` 新增可选 `analyze_pipeline` 构造参数和内部引用。
- `SkillService.close()` 按已有 Pipeline 生命周期统一关闭。

### 5.4 新增用户接口

保持 Recommend 原有 URL：

```text
POST /v1/skills/algos/recommend
```

新增 HTTP 路由：

```text
POST /v1/skills/algos/analyze/skill
POST /v1/skills/algos/analyze/trajectory
POST /v1/skills/algos/evaluations/query
```

静态 Skill Judge 请求支持：

- 精确 `skill_version_id`
- 可选 `request_id`
- 可选 `force_refresh`，用于显式跳过 completed 缓存
- 可选 Judge 超时/执行配置由服务端配置决定，不在 HTTP 层传递模型凭据

Trajectory Judge 请求支持：

- `trajectory_id`
- 可选 `request_id`
- 可选 `attribute`
- 可选 `verifier_score`
- 可选 `force_refresh`

Evaluation 查询支持：

- `skill_version_ids`
- 可选 `judge_types`
- 可选分页限制
- 默认只返回 completed 结果

所有 HTTP 请求字段保持 `extra="forbid"`，错误沿用现有异常处理器。

### 5.5 Recommend 可选评测展示

请求契约增加：

```python
include_evaluation: bool = Field(default=False, strict=True)
```

该字段只允许额外的展示开关，不改变：

- 图解析和快照选择
- 检索策略
- Skill 排序
- `scores`
- `score_type`
- `include_scenes`

响应契约增加可选字段：

```python
evaluations: list[RecommendationEvaluation] | None = None
```

行为要求：

- `include_evaluation=False` 时，序列化必须完全省略 `evaluations`，保持旧响应兼容。
- `include_evaluation=True` 时，在同一个 persistence session 内完成 Recommend 和评测读取。
- 按 Recommend 返回的 Skill 顺序组织结果，同一版本按 Judge 类型和更新时间排序。
- 评测读取失败只记录 warning，不阻断 Skill 推荐。
- 不修改 `GraphManagePipeline.recommend`，后处理放在 `SkillService` 内。

### 5.6 配置同步

按项目约定同步：

- `config/mindmemos_skill/dev.example.yaml`
- `config/mindmemos_skill/dev.yaml`

新增 `analyze_pipeline` 时两者字段结构保持一致；示例文件使用公开占位值，本地文件保留本地连接信息。若本地文件不存在，则创建字段结构一致的本文件，敏感值使用本地约定或占位值，并确保不包含不应提交的真实密钥。

### 5.7 测试计划

持久化测试：

- 新增表可注册，表总数为 11。
- 项目隔离生效。
- `(project_id, evaluation_key)` 幂等保存。
- `completed` / `failed` 状态查询规则正确。
- 按多个 Skill 版本批量查询时保持版本归属。

Pipeline 测试：

- completed 命中时不调用 Judge。
- 未命中原流程执行并保存。
- failed 不命中且允许重试。
- `request_id` 变化不影响 key。
- Judge 失败时 failed 记录可查询，原异常可复现。
- 无对应算法配置时返回清晰错误。

Service / HTTP 测试：

- Analyze 请求完成并返回结果。
- Evaluation 查询只返回当前项目数据。
- Recommend 关闭开关时响应不新增字段。
- Recommend 开启开关时返回评测，且 Skill 顺序、分数和图版本不变。
- 评测查询失败时 Recommend 仍正常返回。
- 原有 Recommend 请求字段不传时行为不变。

兼容测试：

- 所有现有 Skill、Graph、Manage 测试继续通过。
- Ruff 通过。
- 单独运行新增测试和相关回归测试。

## 6. 风险与防护

1. **耦合风险**：不向 `skills_*`、`skill_versions_*`、`skill_graphs_*` 增加 Judge 字段；Analyze 仅通过独立服务访问。
2. **响应兼容风险**：默认关闭开关，关闭时不序列化新字段。
3. **缓存污染风险**：`failed` 不命中；算法、Prompt、配置或内容身份变化时 key 变化。
4. **事务风险**：模型调用与数据库事务严格分离。
5. **可用性风险**：Recommend 的评测读取失败只降级为空结果并记录 warning。
6. **数据迁移风险**：`mindmemos_skill` 是未上线独立服务；本次只新增表，不引入 Skill / Graph 表迁移或兼容桥接。
7. **配置风险**：两个配置文件同步更新，且校验失败在构造资源前发生。

## 7. 验收标准

- `docs/updte_plan.md` 与实际实现一致，并记录最终验证结果。
- Judge 算法仍保持纯计算；没有直接访问数据库。
- 新增 `skill_evaluations_v1` 与现有实体表完全分离。
- Analyze Pipeline 能完成“查缓存 -> 未命中执行 -> 保存成功/失败”。
- Recommend 默认响应与变更前完全兼容。
- Recommend 显式开启评测展示后，能返回对应版本的评测结果。
- 相关 Ruff、Pytest 和兼容测试通过。
- 最终提交只包含本任务文件，不包含既有未跟踪文档、`docs_skill_manage/` 或 `gos/`。

## 8. 实施状态

- [x] 评测持久化
- [x] Analyze Pipeline
- [x] Config / Factory / Service 接入
- [x] Analyze 与 Evaluation HTTP 接口
- [x] Recommend 可选评测展示
- [x] 配置同步
- [x] 测试与静态检查
- [x] 精确提交

### 验证记录（2026-10-22）

- 针对性回归：`70 passed, 1 warning`。
- 全量 Skill 测试：`1428 passed, 21 failed, 27 skipped`。
- Ruff：`ruff check src/mindmemos_skill` 通过；本任务变更及新增的 32 个 Python 文件 `ruff format --check` 通过。
- 离线验证：Fake Judge 已覆盖成功、失败、缓存、重试、HTTP 和 Recommend 展示路径。
- 真实 Judge：本轮未验证，原因是当前 LLM API 未开启；该环境限制未阻断离线接入验证。
- 全量测试剩余 21 项失败均为既有环境或未触及模块问题（Windows 符号链接权限、原生异步/环境、Skill Forest prompt/引用/恢复、LibreOffice/PATH 等），Analyze、Evaluation、HTTP、Recommend 和配置相关失败为 0。
- 精确提交：仅纳入本任务相关文件；现有未跟踪文档、`docs_skill_manage/`、`gos/` 等无关内容未纳入变更。
