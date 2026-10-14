# Skill 服务架构与 API

本文说明独立 `mindmemos_skill` 服务的当前调用边界。中心仓字段与完整请求示例见
[中心仓 API](central_repository_api.md)，训练见 [训练接口](training.md)，
本地使用见 [用户指南](../skill/user-guide.md)。HTTP 的最终字段约束以服务生成的 OpenAPI 为准；离线浏览见 [接口参考](fastapi_api.html)。
使用 `uv run python scripts/dev/export_skill_api.py` 重新生成该参考页。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `api/typing.py`、`api/base.py` | 统一请求、返回结构与异步服务契约 |
| `api/service.py` | 项目授权、配置选择、任务提交及 pipeline 调度 |
| `api/entries/fastapi.py` | HTTP 身份认证、请求转换与响应包装 |
| `config/` | 服务配置与 YAML 解析 |
| `pipelines/` | 仓库、训练、演化、图管理的业务编排 |
| `algos/` | 算法逻辑；训练和演化算法不读写数据库 |
| `skill/` | 正文、资源、快照、组装、导出及内容哈希 |
| `envs/` | 任务执行、私有评分与 rollout 恢复 |
| `persistence/` | 项目内身份、不可变版本、轨迹与图快照 |
| `infra/` | 数据库、异步任务、进程执行等通用基础设施 |

`create_service()` 校验配置、构造唯一共享存储并初始化当前表结构。调用方关闭服务，
pipeline 负责其自有组件的清理。导入模块不启动数据库连接或模型请求。
独立 Skill 服务尚未上线，不提供旧数据库兼容或迁移桥接；已部署云服务的数据契约不受此约定影响。

## 内容与运行时

`SkillContent` 保存 `content: str`、`resources: dict[str, bytes]`、`name`、`description`
及 `standard_format`。标准正文要求 YAML frontmatter 的名称和描述；非标准正文要求显式提供
名称和描述。正文保留输入换行，不自动补写 frontmatter。根目录 `SKILL.md` 是正文，其余文件为资源。

`content_hash` 是原文 UTF-8 字节的 SHA-256。`skill_hash` 使用共享规范编码，覆盖正文、资源路径及
字节；名称、描述和运行配置不参与两种哈希。HTTP、SDK 和快照的资源使用标准 Base64 字典。
资源路径必须是安全、规范的相对路径，不得覆盖根目录正文或与其他文件路径重叠。

`Skill.snapshot()` 保存完整内容及实现配置，恢复时验证两个哈希。`Skill.from_snapshot()`
恢复数据；需要任务时动态行为的调用方使用 `load_runtime_skill()`。TreeSkill 恢复保留路由配置，
在实际组装时才创建默认模型客户端。`with_content()` 保留资源和具体类型，资源修改使用显式操作。

`SkillSet` 选择成员并有界并行组装。环境接收组装后的 Skill，不能重复路由。
rollout 保存源快照、选择结果和组装快照；中断重试复用已保存的组装。输入变化或快照损坏要求新目录。
任务执行只能使用公开任务材料，私有答案保留在环境评分边界内。

## 中心仓与版本

家族通过 `active_version_id` 指向一个生效版本。版本不可变，保存自身名称、描述、正文、资源、
实现配置、两个哈希、提交信息和父版本身份。提交版本不暗中改写家族展示名。

Python 的四个写入口为 `register_skill`、`register_from_local`、`commit_skill`、`commit_from_local`。
前两类结构化写请求使用 `StructuredSkillInfo`；本地文件入口使用 `FileSkillInfo`，只读取标准格式。
HTTP 仅暴露结构化注册和提交，不能让远程调用者指定服务端源文件路径。

`base_version_id` 指定提交的父版本，允许历史分支。`active=False` 保存版本而不移动 active 指针。
激活或自动发布通过原子比较保护指针，竞争失败不留下孤立候选版本。
软删除保留历史数据与证据引用，同时使家族退出正常查询。

| 操作 | Python 方法 | HTTP POST 路径 |
| --- | --- | --- |
| 注册 | `register_skill` | `/v1/skills/register` |
| 列表 | `list_skills` | `/v1/skills/list` |
| 详情 | `show_skill` | `/v1/skills/show` |
| 提交 | `commit_skill` | `/v1/skills/commit` |
| 历史 | `skill_history` | `/v1/skills/history` |
| 差异 | `diff_skill` | `/v1/skills/diff` |
| 激活 | `active_skill` | `/v1/skills/activate` |
| 元信息 | `set_skill_meta` | `/v1/skills/update-info` |
| 删除 | `delete_skill` | `/v1/skills/delete` |
| 版本解析 | `resolve_skill_version` | `/v1/skills/version-resolve` |

Python 请求显式包含 `project_id`。HTTP 项目来自可信身份，不由请求正文决定。
列表的 `limit=None` 表示不限制数量；显式限制与游标遵循对应接口约束。
本地导出调用 `export_skill`，传入父目录后统一追加 `<skill-name>/`，并返回最终目录。非空的同名子目录要求 `overwrite=True`；仅替换该子目录，保留父目录中的其他内容。

## 训练与演化

训练请求必填 `algo`、`env`、`init_skill_version_id`、`work_dir`、`tasks` 和 `split`。
`algo` 选择部署配置中的训练流水线；`env` 提供本次训练的环境 `class_name/config`。训练数据交给 trainer；val 用于选优，
test 只评估最终候选。环境在每次调用中独立提供，训练 pipeline 不拥有共享环境实例。

演化算法接收调用方准备好的历史轨迹和 Skill，不执行新任务、不查询数据库。
`simple_evolution` 从存储中收集轨迹，按配置采样，读取轨迹绑定的历史版本，调用算法后保存候选。
基线可以是历史版本；TreeSkill 的配置及资源随候选保留。

HTTP 入口：

- `/v1/skills/algos/train`：提交训练任务。
- `/v1/skills/algos/evolve`：提交演化任务，正文为 `skill_version_id` 和可选 `auto_accept`。
- `/v1/skills/tasks/show`：按 `task_id` 查询状态。
- `/v1/skills/tasks/progress`：按 `after` 游标读取任务日志，包括结构化进度、物理 attempt 和失败 traceback。

训练、演化受理返回 `202` 和任务快照。演化的 `kind` 为 `skill.evolve`。
状态包括 `queued`、`running`、`succeeded`、`failed` 和 `cancelled`。
成功后 `result` 包含 `skill_id`、`base_version_id`、`version_id`、`run_id`、`content`、
`resources`、`content_hash`、`skill_hash`。`auto_accept` 默认 `True`，设为 `False` 时不激活结果版本。

Python 直接执行入口为 `run_training(TrainingRequest(...))` 和
`run_evolution(SkillPipelineRequest(...))`。演化请求使用 `skill_ref`、可选 `base_version_id`
和 `pipeline` 选择基线及配置，直接执行返回同一成功结果结构。
证据不足、无有效修改或提交冲突均不能返回成功；发布失败时保留候选产物供检查。

## 图管理接线

图算法策略说明见 [Skill 管理](skill_manage.md)。图成员绑定精确版本，不能随家族 active 指针变化。

- 组织：`OrganizeRequest(skill_refs=[...], graph_id=None, alias=None, commit_message=None, project_id=...)`。
  `skill_refs` 可以包含版本或家族引用；家族在组织开始时解析到 active 版本。
  省略 `graph_id` 创建新图，指定已有图时发布新快照，不允许同时指定新 alias。
- 组织结果包含 `graph_id`、`graph_version_id`、按输入顺序解析的 `skill_version_ids`。
- 推荐：`RecommendRequest(graph_id=..., graph_version_id=..., task="...", top_k=5, project_id=...)`。
  返回 `skills`，其中每项为被推荐的完整版本记录。
- HTTP 路径为 `/v1/skills/algos/organize` 和 `/v1/skills/algos/recommend`。
  推荐的 HTTP 请求使用 `graph_ref`、可选 `graph_version_id`、`query` 和 `top_k`。
- 图解析与查询分别使用 `/v1/skills/graph-resolve`、`/v1/skills/graphs/list`、
  `/v1/skills/graphs/snapshots` 和 `/v1/skills/graphs/snapshot`；软删除使用 `/v1/skills/graphs/delete`。

组织、推荐当前在请求内完成。图快照仍使用其自身的 current 指针术语，与 Skill 家族的 active 指针分开。
低层图算法可以接收调用方显式 Scene 记录；当前统一组织请求没有 Scene 输入字段。

## HTTP 与 SDK

同步成功包装为 `{"code":"ok","request_id":"...","data":...}`，异步受理使用 `code: "accepted"`。
错误包含稳定的 `error.code`、`error.message` 和请求 ID。项目外的资源按不可见处理。

SDK 的同步与异步 Skill 客户端使用同一请求模型，返回响应中的 `data`。
本地目录由客户端读取为结构化内容后发送。SDK 不自动重试 Skill 写请求，避免重复发布。
端到端示例见 [SDK README](../../src/mindmemos_sdk/README.md)。
