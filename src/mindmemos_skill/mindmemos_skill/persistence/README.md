# 业务数据服务

本模块以 `db_services/` 下的文件夹为服务边界。每个服务内部定义 `typing.py`、`tables.py`，有业务读写逻辑时再添加 `service.py`；顶层只负责共同类型、注册、工厂与建表入口。业务表使用 SQLAlchemy 定义，通过已有 `infra.database.TableRegistry/TableSpec` 注册；每个 Column 都有英文 `comment`，说明物理含义。

| 数据服务 | 表 | 业务职责 |
| --- | --- | --- |
| `skills/` | `skills` | 中心仓 Skill 身份、名称、项目内唯一 alias、首次注册来源与说明、唯一当前版本指针 |
| `skills/` | `skill_versions` | 本地版本内容、多父谱系和运行时元数据 |
| `trajectories/` | `trajectories` | 一次 rollout attempt 的输入、交互与结果 |
| `remote/` | `remote_skill_bindings` | 本地 Skill 与远端项目中的云端 lineage 关联及同步观察 |
| `remote/` | `remote_version_bindings` | 本地版本与云端版本的映射、云端元数据和原始 bundle |
| `remote/` | `remote_trajectory_bindings` | 本地轨迹与云端 memory/add record 的映射 |
| `remote/` | `remote_operations` | 远端请求的本地去重、执行状态和待重试信息 |
| `skill_management/` | `skill_graphs`、`skill_graph_versions`、`skill_graph_nodes`、`skill_graph_edges`、`skill_graph_revisions` | 图内容快照、节点向量、关系和成功发布记录的表定义 |

不包含审计日志、图操作队列或租约表。图管理目前仅实现记录类型、表约束和注册，尚未实现算法执行、封存、发布或 RLS。没有修改 `mindmemos` 的表、接口或 evolve pipeline，也没有导入 `mindmemos` 运行时代码。

## 建表与使用

物理表名统一使用 `_v1` 后缀，实际名称以 `constants.py` 为准。已有本地库升级时，先停止旧版 Skill 进程，按需执行下文历史字段迁移，再调用
`migrate_table_names_to_v1(database)`（从 `mindmemos_skill.persistence.migrations` 导入），最后调用 `ensure_collection(persistence)` 并启动新版进程。
迁移前须用 `create_persistence(database)` 注册表；该迁移仅重命名已存在的表，保留数据、索引和外键，不修改字段或云服务集合。
新库无需执行迁移。迁移可重复调用，中断后可再次执行；若同一张表的新旧名称同时存在，会在改动前报错，保留两份数据供检查。
旧版进程不能与改名后的本地库混用。该迁移不影响 `src/mindmemos/` 已部署云服务的数据契约。

```python
from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence import (
    SkillVersionRecord,
    create_persistence,
    ensure_collection,
)

with_database = SQLiteDatabase(SQLiteConfig(url="sqlite:///:memory:"))
try:
    services = create_persistence(with_database, include_remote=True)
    ensure_collection(services)
    services.skills.add(SkillVersionRecord(
        project_id="local-project",
        version_id="local-v1",
        skill_id="local-skill",
        name="Example",
        bundle="Example local bundle",
        content_hash="caller-computed-local-hash",
    ))
    assert services.skills.get("local-v1").name == "Example"
finally:
    with_database.close()
```

`create_persistence` 只组装服务并登记表，不连接数据库。`ensure_collection` 显式创建缺失表，可重复调用；它不是 schema migration，不修改已有表。skill_versions 新增必填 project_id，旧库需显式回填项目并迁移，不能直接依赖 ensure_collection。每个实例使用独立 registry。数据库由调用方关闭，数据服务借用它。

已有 infra 后端也可以直接接受 `build_persistence_tables()` 返回的 registry。这个方式与 `create_persistence()` 二选一，不能重复注册同一组表。跨服务原子写入使用 `database.transaction()` 返回的句柄，并将该句柄传给对应数据服务。

服务工厂也支持现有 YAML 后端装配：

```yaml
backends:
  database:
    module: infra
    class_name: sqlite
    config:
      url: sqlite:///./data/skill.db
  persistence:
    module: persistence
    class_name: skill
    config:
      include_remote: true
    dependencies:
      database: database
```

装配后调用 `service.initialize_storage()` 创建缺失表并迁移版本标签。低层 `ensure_collection` 仅建表，不替代服务升级入口。中心仓 API 已实现，自动 evolve/train 仍为占位。

## 核心数据语义

- `skill_versions.version_id` 是本地版本主键，`skill_id` 是本地版本族；`version_label` 是同 project_id、skill_id 内唯一的语义化展示别名；底层读取仍兼容旧空标签。内容、资源和哈希由所属 Skill 实现提供，持久化不猜测 bundle 编码，也不重算哈希。
- 本地父版本保留列表。数据服务检查自身引用、重复父 ID 和时间顺序；不要求已下载完整祖先链。
- `trajectories` 以 `trajectory_id` 标识物理执行，以 `(rollout_id, attempt_no)` 唯一区分重试；JSON 过程和奖励信息与远端摘要独立。
- 两类数据服务的 `add` 为插入语义，重复主键不覆盖历史记录。当前提供最小新增/读取能力，不实现完整版本管理或 rollout 状态机。
- 核心版本 ID 和 rollout ID 需要在当前本地库中唯一。task_id 可由 task_metadata 中的数据集上下文区分。本地 skill_versions.project_id 为必填本地项目身份，通过复合外键保护图节点引用；它不隐式等同于云端 project_id。查询授权与 RLS 尚需业务入口落实。

## Remote 兼容设计

基于当前云端源码：`mindmemos/typing/skill.py`、`infra/db/collections/skill.py`、`pipelines/skill/version_store.py`、`pipelines/skill/evolution.py`。没有连接生产服务验证部署版本。

云端继续维护 `skill_version_v1`、`skill_blob_v1`、`skill_trace_pending_v1`、`skill_trace_summary_v1` 及原有 add_record。新 remote 表只在 `mindmemos_skill` 使用的数据库中建立，是关联和工作队列，不替代云端 collection。

| 云端已有语义 | 新本地存储处理 |
| --- | --- |
| version_id 为云端身份 | `remote_version_bindings.version_id` 原样保存，另用 local_version_id 关联本地 |
| cloud_skill_id 代表单父版本 lineage | 在 deployment + project 下绑定本地 skill_id |
| parent_version_id 只有一个 | 保留原值，不擅自折叠本地多父关系；发布合并版本时调用方必须明确选定云端父版本 |
| 云端 version_label 可空、可重复 | remote 原值保持不变；本地中心仓标签按项目与 Skill 唯一，旧值迁移保存在 local_metadata |
| content_hash 按云端白名单 bundle 计算 | 独立保存云端 content_hash/content，不拿本地哈希顶替 |
| 状态 observed/draft/evaluating/published/superseded/rolled_back | Remote 模型接受相同状态，并可刷新状态观察 |
| origin 为 edge/cloud | Remote 原样保留，本地 provenance 独立 |
| sync 请求 local_version_id 实际是客户端持有的云端版本 ID | `last_checked_version_id` 记录该云端 ID，不发送本地新生成的版本 ID |
| 轨迹通过 memory/add 写入 | 保存 add_record_id 映射，不虚构独立的轨迹上传 API |
| evolve 使用摘要与 consumed_version_id | 消费状态继续由云端维护，不复制到本地轨迹表 |

所有 remote 表均显式保存 `remote_id + project_id`。remote_id 是配置里的稳定部署别名，不存 API key 或授权 header。同一组云端 ID 可以存在于不同项目／部署，不互相覆盖。当前采用每个 remote/project 内本地 family 与云端 lineage 一对一、local version 与 cloud version 一对一的映射；若以后需要同一版本投射到多个云端父链，需要显式扩展这个约束。

Remote 版本表暂时携带原始云端 content，以保存两种 bundle 表达的差异。没有复制 cloud blob 去重表，避免为映射再建立一套云端版本库。内容量增大后可单独评估去重。

外键约束保护远端映射引用的本地版本／轨迹。RemoteDataService.bind_version 进一步检查本地版本所属 family 与云端 family 绑定一致。Remote 服务暂时沿用已有版本记录检查 family 存在性，尚未切换为检查中心仓 skills 主表；直接使用通用 infra CRUD 会绕过这部分业务检查。

## 中心仓 Skill 主表

`skills` 以 `(project_id, skill_id)` 标识中心仓中的 Skill，身份不依赖注册来源或导出路径。
`name` 为可变展示名称；可选 `alias` 按精确字符串在项目内唯一，空值表示没有 alias。
`registration_source` 和 `registration_message` 仅记录首次注册来源及说明，不跟踪外部目录，
也不代替版本的 `commit_message`。业务接口负责限定可修改字段，通用更新接口不维护字段白名单。

`current_version_id` 必填，通过复合外键只能指向同一项目、同一 Skill 的版本。
它是中心仓默认读取、演进和导出的唯一版本选择依据；不在版本上重复保存 is_active。
显式导出历史版本不修改指针，切换指针也不修改已导出的应用目录。
当前被引用的版本不能删除；切换指针保留旧版本内容。

首次登记时，在一个 `database.transaction()` 中先插入初始版本，再插入主记录。
后续先保存新版本，再更新指针；并发提交可在 update 的 filters 中带上预期的 current_version_id，
通过返回的匹配行数判断冲突。写入方显式更新 updated_at。
`skills/service.py` 统一提供中心仓事务与版本数据访问，经 `SkillPersistence.skills` 注入 RepositoryPipeline；Service 只转交业务请求；SDK 数据迁入尚未实现；
历史版本仍可独立保存，不强制所有已有 family 立即建立主记录。

新库通过 `ensure_collection()` 创建完整结构。已有包含 project_id 的本地版本库先执行
`scripts/skill/migrate_central_skills.sql` 添加当前指针外键需要的唯一索引，再调用 `ensure_collection()`
创建 skills 表；索引迁移可重复执行，不改写历史行。更早缺少 project_id 的库须先完成项目归属回填，
此脚本不猜测项目。已有 family 的当前版本必须由调用方明确选择，不按时间自动回填。
这些变更仅针对 mindmemos_skill 的数据库，不修改商用云服务的数据库。

没有创建旧设计中的轨迹拉取游标：现有 Skill API 没有通用 trajectory 增量拉取协议。`last_synced_at` 只是成功检查时间，不能作为无遗漏版本增量消费水位；既有 versions?since 的时间边界与分页限制应在后续同步实现中单独处理。

## 操作队列的边界

`remote_operations` 支持现有 register_skill、sync_skill、add_trace、evolve_skill 四种业务能力。request_payload 保存可重放的业务请求，调用方负责按对应云端 API 校验、提供 request_hash 和移除秘密；认证由远端配置提供。

`enqueue` 按 remote/project/idempotency_key 原子去重，相同 key 但不同请求内容或目标明确报错；业务代码负责在状态更新时保留原始 request_payload/request_hash，通用更新接口不限制这些字段。

这些是本地幂等规则，不表示现有云端支持相同幂等键。调用发出后遇到超时、进程崩溃或结果不明，应使用 `uncertain`，先核对云端结果，再决定是否重试，特别是 evolve 和 memory/add。状态字段和租约只是后续 worker 的数据基础，本次没有实现网络同步、租约认领或自动重试状态机。

## 云端平稳切换算法

后续 cloud evolve pipeline 可以保留当前外层流程：读取 head/blob 与未消费摘要 → 转为 `mindmemos_skill` 算法输入 → 调用算法 → 经原 SkillVersionStore 保存结果 → 经原 repository 标记 consumed_version_id。

算法输出应提供新内容，由旧云端代码继续生成云端 version_id/content_hash、保存单父 lineage 和处理发布状态。这样 import 算法不要求 remote 同步，不要求云端迁表，也不需要把云端摘要伪装成本地完整 rollout。本次只实现数据服务基础，没有提前切换线上算法。

## Skill 实现字段与已有数据库迁移

`skill_versions` 使用 `implementation`、`config_schema_version` 和 `config`
保存具体 Skill 实现及固定配置。可优化内容只保存在 `bundle` 中，`content_hash`
是源 bundle 的哈希，不是某次 task 组装结果的哈希。

`skills.add_skill(skill, record)` 用完整 Skill 的快照填充这些字段以及 bundle、resources 和哈希，保留 record 的
身份、父版本和其他元数据；`skills.get_skill(version_id, **dependencies)`
验证源哈希及配置版本后按注册名恢复。底层 `add/get` 继续支持原始记录，不解释 bundle。

新建数据库直接使用新字段。已有旧字段数据库需一次性执行仓库根目录下的
`scripts/skill/migrate_skill_columns.sql`，仅重命名三列，保留数据；适用于 SQLite
和 PostgreSQL。`ensure_collection` 不会自动迁移或重建已有表。迁移不会转换旧的
bundle 编码或实现注册值，只有符合新 Skill 内容协议的记录才能由 `get_skill` 恢复。

`resources` 是相对路径到 base64 字节串的 JSON 映射，完整保留辅助文件，不参与
组装或优化。`local_snapshot_hash` 覆盖 bundle 和 resources，`content_hash` 只覆盖 bundle。
