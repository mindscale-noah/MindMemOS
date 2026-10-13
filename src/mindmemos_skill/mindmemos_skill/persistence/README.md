# Skill 持久化

`create_persistence(database)` 在借用的数据库上注册 Skill 业务表；`ensure_collection(persistence)` 创建当前结构中缺失的表。`create_service()` 初始化服务时也会建表。数据库连接由调用方关闭。

`db_services/skills/` 管理 Skill 身份与版本，`db_services/graphs/` 管理图、快照、节点和边，`db_services/trajectories/` 管理执行轨迹，`db_services/tasks/` 管理任务状态。表名由 `constants.py` 定义。

此服务尚未上线，持久化代码只支持当前数据库格式。如果本地 Skill 数据库结构与代码冲突，停止 Skill 服务，清理该服务的本地数据库并重启，使当前表结构重新建立。清理前确认数据库路径只供 Skill 服务使用；不得清理 `src/mindmemos/` 已部署云服务的数据。

## 统一入口与项目范围

`SkillPersistence` 统一注册业务表并组装数据 service。普通业务操作必须在 `session(scope)` 中执行；没有活动 session 时，访问仓库会抛出异常。调用方应传入已经完成授权的 `Scope`，持久层不负责验证用户身份。

```python
from mindmemos_skill.infra.database import Scope

with persistence.session(Scope(fields={"project_id": authorized_project_id})) as repositories:
    version = await repositories.skills.get(version_id)
    graph = await repositories.graphs.get_graph(graph_reference)
    task = await repositories.tasks.get(task_id)
```

| 入口 | 职责 | 数据范围 |
| --- | --- | --- |
| `skills` | Skill 家族、版本、源内容、提交和激活 | Skill 家族与版本表 |
| `graphs` | 图引用、快照、拓扑、发布 | 图、快照、节点和边；通过同 session 的 `skills` 检查源版本 |
| `trajectories` | 轨迹、统计和消费标记 | 轨迹与消费表；通过同 session 的 `skills` 检查源版本 |
| `tasks` | 项目内任务创建、详情和日志 | 任务与任务日志表，不提供跨项目调度 |
| `audit` | run、span 和 log 查询 | 独立审计数据库的只读表；未配置存储时为 `None` |

仓库使用 `RepositoryDatabase` 限制可访问的表，并保留数据库的完整项目范围。额外 `agent_id`、`user_id` 维度不会自动丢弃；目标表缺少该隔离字段时拒绝操作。业务方法中的 ID、状态、时间和分页参数只能进一步筛选范围。session 退出后，保留的仓库句柄失效。嵌套项目 session 可以使用相同范围，但不能切换项目。

所有业务查询、注册、轨迹上报和图写入方法均不再接收 `project_id` 参数，项目归属从活动 scope 获取。数据库 record 中的 `project_id` 仍是必填归属字段；仓库创建的 record 自动补全该字段，直接导入完整 record 时仍检查它与 scope 是否一致。

可通过 `persistence.project_id`、`repositories.project_id` 或 service 内的 `self.database.project_id` 读取当前项目。这些属性只读，直接从活动 scope 派生，不单独保存项目 ID。没有项目 session、处于无项目范围的维护上下文，或句柄已关闭时，读取会抛出异常。

## 事务与图发布

`persistence.transaction()` 在当前项目 session 内创建一个事务，并使统一入口的业务仓库都使用该事务。跨仓库操作通过同一个数据库连接提交或回滚；不支持嵌套事务。仓库内部的原子操作会加入已有事务；其中发生的异常即使被调用方捕获，也会使该事务整体回滚。

```python
with persistence.session(scope):
    async with persistence.transaction() as repositories:
        family = await repositories.skills.register(**registration)
        await repositories.tasks.create(task_spec)
```

模型调用、Scene 提取和图计算放在事务之外。准备好完整节点、边和构建元数据后，通过 `graphs.publish()` 发布。`GraphWriteRequest` 指定目标图、精确 Skill 版本映射、操作人、提交说明和捕获的图头；项目来自活动 session。发布保留版本绑定、快照完整性和图头并发检查。`graphs.load(graph_id, snapshot_id)` 读取并校验快照，不需要提供操作人或发布参数。`graphs.writer(request)` 为算法提供单次操作的存储适配器。

审计数据库不参与业务数据库事务，因此事务 session 中 `audit` 为 `None`，不能承诺跨数据库原子提交。审计采集器及延迟写入由 `observability/` 管理，不保留请求结束后的仓库句柄。

## 自定义仓库

后端 service 可以自定义所属表的查询与写入逻辑，复用现有数据库的过滤、投影、聚合和事务能力。扩展 `SkillPersistence.build_tables()` 声明新增表，并通过 `session_type` 指定扩展后的 `SkillPersistenceSession`。扩展 session 使用 `repository(tables, read_tables=...)` 为 service 提供访问句柄：`tables` 允许读写，`read_tables` 只允许读取依赖表；事务和项目范围自动保留。表必须预先注册，不能通过查询参数扩大项目范围。

内置 service 直接组装，不为每个仓库增加注册框架。需要按名称选择整个 persistence 实现时，使用现有 `Persistence.register()` 机制，并在应用入口显式导入扩展模块。

## Worker、审计和维护

可信的启动组装代码通过 `persistence.task_worker()` 取得跨项目任务调度能力，再注入 task manager。该能力提供领取、续租、结束和恢复任务等操作，不能在项目 session 中取得。Worker 根据可信任务记录建立项目 session 后执行实际业务。可信调度代码需要查询某个项目的任务时，通过 `worker.scoped(scope).get(task_id)` 或 `.logs(task_id)` 获取该项目的数据；领取、续租与恢复仍属于跨项目调度能力。

可在构造时传入独立 `audit_database`，或在服务启动阶段调用 `attach_audit_database()`。项目 session 的 `audit` 只负责查询；采集与写入仍由 `AlgorithmAuditPersistence` 管理。审计运行在 `with audit.session(scope):` 中调用 `trace_run(...)`；延迟写入捕获该运行的 scope，不依赖之后的活动请求。独立审计查询可通过 `audit.queries(scope)` 取得只读仓库，查询方法不接收项目参数。所有数据库都由创建它们的调用方关闭。

初始化统一使用 `initialize_storage()` 或 `ensure_collection()`，不需要打开项目 session。确实需要无项目限制的测试数据准备或维护时，显式使用 `maintenance()`；它不能在业务 session 中打开，也不得用于请求处理。`create_persistence()` 不再默认提供无范围的仓库访问。

## 版本内容

版本表直接保存 `content`（完整 SKILL.md 文本）、`resources`（相对路径到标准 Base64 的 JSON 映射）、`standard_format`、`name` 和 `description`。`content_hash` 校验原文 UTF-8 字节，`skill_hash` 校验正文与完整资源；写入及读取完整版本时均校验快照，哈希不一致会拒绝操作。版本表不再保存 `bundle` 或 `local_snapshot_hash`。

Python 服务的结构化入口是 `register_skill(RegisterRequest)` 和 `commit_skill(CommitRequest)`；本地文件入口是 `register_from_local(RegisterFromLocalRequest)` 和 `commit_from_local(CommitFromLocalRequest)`。本地入口先授权，再读取标准 Skill，复用结构化写入。注册返回家族、初始版本 ID 和两个哈希；提交返回家族信息与本次新版本，包括未激活的版本。

提交默认使用传入的完整资源集合；`inherit_resources=True` 明确继承父版本资源，此时不能同时提供非空资源。版本保存自己的名称与描述，提交不改写家族展示名；提交元数据写入版本并替换家族元数据，`commit_source` 存入版本的 `origin`。新版本、家族元数据及默认版本指针在同一事务中提交或回滚。

## 激活与版本查询

Skill 家族使用唯一的 `active_version_id` 指针，默认读取和导出选择该版本；未激活提交保留原指针。Python `active_skill(ActiveRequest)` 接收要激活的 `version_id`，从该版本解析所属家族，不创建新版本。可提供 `expected_active_version_id` 拒绝基于过期状态的操作；持久层也会比较操作开始时的指针。激活在同一事务内锁定预期指针、验证目标归属并切换，失败整体回滚。图谱自身的当前快照字段不属于 Skill 版本激活语义。

`resolve_skill_version(VersionResolveRequest)` 支持 `content_hash`、`skill_hash`、精确 `skill_version_id`，以及可选的 `skill_id` 约束。所有已提供条件同时生效；至少提供一种哈希或版本 ID。正文相同但资源不同会匹配相同 `content_hash`；多个版本即使完整 `skill_hash` 相同也有独立身份，必须通过版本 ID 等条件消除歧义。解析排除已删除家族，返回两个哈希、`version` 标签、别名、提交说明及 `is_active`。

`skill_history(HistoryRequest)` 必填 `skill_id`，默认 `limit=None` 返回全部版本。历史按 `(created_at, version_id)` 升序，支持 offset，或使用 `SkillHistoryCursor(snapshot_at, after_created_at, after_version_id)` 续页。稳定游标必须保留第一页的时间截止点，并以最后一条记录的时间和 ID 续页；不能混用 offset。新创建的版本不进入已有时间截止点。自定义持久层 `fields` 原样投影，不强制补齐完整摘要。

家族普通列表按名称、ID 排序；带 `snapshot_at` 的游标列表按创建时间、ID 降序。`limit=None` 不限制返回数量。`diff_skill(DiffRequest)` 接收目标版本和可选起始版本；省略起始版本时使用目标家族的 active 版本，显式版本允许比较同项目不同家族。元信息修改使用 `SetMetaRequest.skill_id`。
