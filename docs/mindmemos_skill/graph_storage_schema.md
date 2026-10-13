# Skill 图存储：当前版本指针与不可变快照

本文件说明 `mindmemos_skill` 当前实现的数据结构。建表定义以
`src/mindmemos_skill/mindmemos_skill/persistence/db_services/graphs/tables.py` 为准。
所有图表以 `project_id` 隔离项目，以 `(project_id, graph_id)` 标识一张逻辑图。
`actor_id` 仅记录创建者，不参与数据隔离。
图的 `graph_id` 与快照的 `graph_version_id` 都以 `Text` 存储完整带前缀的 ID。
`current_version_id` 和节点、边的 `graph_version_id` 同样是 `Text`，直接引用完整
`skgv-…` 值。快照不再单独保存 `public_version_id`。

## 四张核心表

| 表 | 主键 | 职责 |
| --- | --- | --- |
| `skill_graphs_v1` | `(project_id, graph_id)` | 图的稳定身份、展示元数据和 `current_version_id` |
| `skill_graph_versions_v1` | `(project_id, graph_id, graph_version_id)` | 完整不可变图快照的状态、数量、哈希、算法配置和创建信息 |
| `skill_graph_nodes_v1` | `(project_id, graph_id, graph_version_id, node_id)` | Skill/Scene 节点、检索字段和 embedding |
| `skill_graph_edges_v1` | `(project_id, graph_id, graph_version_id, edge_id)` | 同一快照内的有向关系 |

`skill_graphs.current_version_id` 是当前生效版本的唯一指针；`NULL` 表示尚未发布。
它通过包含 `project_id`、`graph_id` 的复合外键指向本图的版本。不要在每个版本上另设
`active` 布尔值。推荐请求省略版本 ID 时固定此指针；显式指定历史 `graph_version_id`
时读取对应不可变快照。新组织成功会原子地插入快照、节点和边，再切换指针。旧快照保留。

图版本的 `status` 为 `staging`、`validating`、`ready` 或 `failed`；目前发布实现只提交
`ready` 快照。`node_count`、`edge_count`、`vector_count`、`snapshot_hash` 和 `ready_at`
用于完整性校验。`metadata` 保存不含密钥的算法配置。`request_hash` 在同一图内唯一，
用于成功请求的幂等重试；`actor_id` 和 `message` 保存该快照的创建信息。
本设计不记录每次重新激活历史快照的事件，也没有发布 revision。

节点类型为 `skill` 或 `scene`。Skill 节点引用同项目的精确 `skill_version_id`；
Scene 节点通过 `owner_node_id` 指向同快照中的所属 Skill 节点。每个快照内同一
`skill_version_id` 至多出现一次。节点包含解析出的能力描述、输入输出、领域标签、
工具词、示例任务、展示片段、扩展元数据，以及可空的向量与输入哈希。
边的起点和终点都必须在同一快照内；关系类型为 `dependency`、`containment`、
`flow`、`semantic`、`workflow` 或 `alternative`。同类型、同端点的边不可重复。

## 发布与并发

构图和模型调用发生在事务外。发布事务锁定图身份行，再次检查 `request_hash`，
插入完整 `ready` 快照及其节点、边，最后更新 `current_version_id`。
失败或中断会回滚整个发布事务。相同请求重试返回已有快照，不会把图头切回旧快照。
不同输入产生新快照；旧版本可以通过显式版本 ID 查询和推荐。

当前组织入口尚不接受 `expected_current_version_id`，因此并发的不同组织请求按取得
图行写锁的顺序成为当前版本。若需要拒绝过期任务覆盖新版本，应在受理时记录预期
当前版本，并在发布事务中比较；若还要识别 `A → B → A` 后的过期任务，可在图身份
行增加递增的图头代数，而不恢复发布历史表。

## 本地数据库边界

这些表属于尚未上线的独立 Skill 服务。初始化只创建当前表结构，不保留旧 revision 表、
旧图头字段或滚动部署桥接。开发数据库与当前结构不兼容时，先备份需要保留的实验产物，
再按项目约定清理该服务本地数据库并重启；不得触碰已部署云服务数据库。

图发布测试覆盖事务回滚、重复请求复用、并发发布及跨项目隔离。
SQLite 发布先写入或锁定图身份，再查询快照，避免读事务升级造成锁冲突。
PostgreSQL 的实际运行验证仍以对应环境中的测试结果为准。
