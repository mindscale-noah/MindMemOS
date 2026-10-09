# Skill 持久化

`create_persistence(database)` 在借用的数据库上注册 Skill 业务表；`ensure_collection(persistence)` 创建当前结构中缺失的表。`create_service()` 初始化服务时也会建表。数据库连接由调用方关闭。

`db_services/skills/` 管理 Skill 身份与版本，`db_services/skill_management/` 管理图、快照、节点和边，`db_services/trajectories/` 管理执行轨迹，`db_services/tasks/` 管理任务状态。表名由 `constants.py` 定义。

此服务尚未上线，持久化代码只支持当前数据库格式。如果本地 Skill 数据库结构与代码冲突，停止 Skill 服务，清理该服务的本地数据库并重启，使当前表结构重新建立。清理前确认数据库路径只供 Skill 服务使用；不得清理 `src/mindmemos/` 已部署云服务的数据。

## 版本内容

版本表直接保存 `content`（完整 SKILL.md 文本）、`resources`（相对路径到标准 Base64 的 JSON 映射）、`standard_format`、`name` 和 `description`。`content_hash` 校验原文 UTF-8 字节，`skill_hash` 校验正文与完整资源；写入及读取完整版本时均校验快照，哈希不一致会拒绝操作。版本表不再保存 `bundle` 或 `local_snapshot_hash`。

Python 服务的结构化入口是 `register_skill(RegisterRequest)` 和 `commit_skill(CommitRequest)`；本地文件入口是 `register_from_local(RegisterFromLocalRequest)` 和 `commit_from_local(CommitFromLocalRequest)`。本地入口先授权，再读取标准 Skill，复用结构化写入。注册返回家族、初始版本 ID 和两个哈希；提交返回家族信息与本次新版本，包括未激活的版本。

提交默认使用传入的完整资源集合；`inherit_resources=True` 明确继承父版本资源，此时不能同时提供非空资源。版本保存自己的名称与描述，提交不改写家族展示名；提交元数据写入版本并替换家族元数据，`commit_source` 存入版本的 `origin`。新版本、家族元数据及默认版本指针在同一事务中提交或回滚。

## 激活与版本查询

Skill 家族使用唯一的 `active_version_id` 指针，默认读取和导出选择该版本；未激活提交保留原指针。Python `active_skill(ActiveRequest)` 接收要激活的 `version_id`，从该版本解析所属家族，不创建新版本。可提供 `expected_active_version_id` 拒绝基于过期状态的操作；持久层也会比较操作开始时的指针。激活在同一事务内锁定预期指针、验证目标归属并切换，失败整体回滚。图谱自身的当前快照字段不属于 Skill 版本激活语义。

`resolve_skill_version(VersionResolveRequest)` 支持 `content_hash`、`skill_hash`、精确 `skill_version_id`，以及可选的 `skill_id` 约束。所有已提供条件同时生效；至少提供一种哈希或版本 ID。正文相同但资源不同会匹配相同 `content_hash`；多个版本即使完整 `skill_hash` 相同也有独立身份，必须通过版本 ID 等条件消除歧义。解析排除已删除家族，返回两个哈希、`version` 标签、别名、提交说明及 `is_active`。

`skill_history(HistoryRequest)` 必填 `skill_id`，默认 `limit=None` 返回全部版本。历史按 `(created_at, version_id)` 升序，支持 offset，或使用 `SkillHistoryCursor(snapshot_at, after_created_at, after_version_id)` 续页。稳定游标必须保留第一页的时间截止点，并以最后一条记录的时间和 ID 续页；不能混用 offset。新创建的版本不进入已有时间截止点。自定义持久层 `fields` 原样投影，不强制补齐完整摘要。

家族普通列表按名称、ID 排序；带 `snapshot_at` 的游标列表按创建时间、ID 降序。`limit=None` 不限制返回数量。`diff_skill(DiffRequest)` 接收目标版本和可选起始版本；省略起始版本时使用目标家族的 active 版本，显式版本允许比较同项目不同家族。元信息修改使用 `SetMetaRequest.skill_id`。
