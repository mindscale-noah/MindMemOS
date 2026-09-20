# 中心仓 API

`mindmemos_skill` 的 Python API 与 HTTP 接入共用 `SkillService`，不依赖 SDK，也不调用商用云服务。
中心仓保存完整 Skill 快照；来源目录与导出目录均不跟踪。`skills.current_version_id` 是当前版本的唯一依据。

## 初始化与调用

默认配置 `config/mindmemos_skill/default.yaml` 使用本地 SQLite，且不启用 remote 表。
服务构造不连接数据库。Python 调用方显式初始化并最终关闭服务：

```python
from mindmemos_skill import create_service
from mindmemos_skill.pipelines.typing import RegisterRequest, CommitRequest, ExportRequest

service = create_service("config/mindmemos_skill/default.yaml")
try:
    service.initialize_storage()
    first = service.register_skill(RegisterRequest(
        project_id="demo", name="Data analysis", alias="analysis",
        source="./my-skill", message="Initial registration",
    ))
    second = service.commit_skill(CommitRequest(
        project_id="demo", skill_ref=first.skill.skill_id,
        source="./edited-skill", message="Improve instructions",
    ))
    destination = service.export_skill(ExportRequest(
        project_id="demo", skill_ref="analysis", directory="./application/analysis",
    ))
finally:
    service.close()
```

方法接收 `pipelines/typing.py` 中的请求对象，也接受同结构字典并校验。
`source` 可以是目录、SKILL.md 路径，或 `skill.snapshot()` 返回的完整 `SkillSnapshot`。
路径首次注册为 static Skill；目录提交保留当前版本的实现类型和固定配置。
显式提交完整 snapshot 可以同时更改实现与固定配置。导出只写 bundle 与 resources，
不把实现配置或仓库元信息偷偷写入应用目录；完整实现配置仍保存在仓内版本中。

`initialize_storage()` 只创建缺失表，不迁移已有表。已有版本库的索引迁移与项目回填要求见
[持久化说明](../../src/mindmemos_skill/mindmemos_skill/persistence/README.md)。
有多个 SkillPersistence 后端时，用 ServiceConfig 的 `persistence_backend` 指定实例名称。

## 方法与 HTTP 路由

所有下列 HTTP 路由使用 POST，接收 JSON body；请求类型与 Python 方法一致。

| Python 方法 | 请求类型 | HTTP 路由 | 返回值 |
| --- | --- | --- | --- |
| register_skill | RegisterRequest | /skills/register | 主记录及初始版本 |
| list_skills | ListRequest | /skills/list | 主记录列表 |
| show_skill | ShowRequest | /skills/show | 主记录及指定/当前版本 |
| export_skill | ExportRequest | /skills/export | 导出目录绝对路径 |
| commit_skill | CommitRequest | /skills/commit | 主记录及新版本 |
| skill_history | HistoryRequest | /skills/history | 完整版本记录列表 |
| diff_skill | DiffRequest | /skills/diff | 文件差异及实现配置是否变化 |
| switch_skill | SwitchRequest | /skills/switch | 切换后的主记录 |
| set_skill_meta | SetMetaRequest | /skills/set-meta | 修改后的主记录 |
| delete_skill | SkillRequest | /skills/delete | 无返回内容（HTTP JSON null） |

`project_id` 必填。`skill_ref` 支持完整 ID、精确 alias、项目内唯一 ID 前缀；精确匹配优先，
前缀歧义时报错并列出候选。展示名称不作为引用。alias 不得占用其他 Skill 的完整 ID。
列表与历史支持 limit/offset，默认 100，单页最多 1000。

HTTP 启动入口自动调用 initialize_storage；手工 create_app(service) 不负责初始化。
HTTP 入口用于可信本地调用，project_id 是调用参数，不是认证身份；目前没有实现多租户认证授权。
路径参数指向服务进程所在机器的文件系统。远程客户端可以直接传 snapshot，不需要共享导入目录。
错误语义为：对象不存在 404，歧义/别名/版本冲突或仍被引用 409，请求无效 422。
自动训练和演化入口仍返回 501，本轮不实现算法调用。

## 导出职责

完整 Skill 导出统一由 `Skill.export(directory, overwrite=False)` 提供，
目录暂存、替换和异常恢复由 `skill/io.py` 实现。默认拒绝非空目标；显式覆盖时替换整个目录并清除旧文件。
API 只选择版本、检查目标不包含仓库数据库，通过 `Skill.from_snapshot()` 恢复数据对象并调用其导出方法。

`SkillSnapshot` 仅保存序列化数据及执行完整性校验，不提供导出行为。通用恢复保留实现元信息，
不加载动态实现或创建 Agent；如需动态组装，由调用方显式使用算法自身的恢复入口。
TreeSkill 的默认 router 延迟到实际路由时创建，读取、恢复和导出不需要执行依赖。

## 写入与恢复边界

- register 总是创建新身份，内容或路径相同不自动覆盖已有 Skill。初始版本与主记录同事务提交。
- commit 省略 base_version_id 时，在读取输入前捕获当前版本；显式 base 必须等于当前版本。
  新版本和指针推进同事务完成，指针变化则报冲突并回滚新版本。它不推断外部目录的导出祖先。
- switch 只切换当前指针，不产生版本、不更新应用目录。历史版本仍保留。
- set-meta 只修改 name/alias，clear_alias 显式移除 alias，不创建版本或改写注册来源。
- export 默认拒绝非空目标，overwrite=true 才替换完整目录并清除残留文件。
  先在同父目录暂存，替换失败时恢复旧目录；进程被强制终止期间的备份可能留在目标父目录，
  当前不提供跨进程崩溃的自动恢复。禁止导出覆盖仓库数据库所在目录及工作目录祖先。
- delete 在事务内删除主记录与全部版本，拒绝仍被其他版本父链、轨迹、图节点或远端记录引用的 Skill。
  不删除任何导入或导出目录。轨迹引用当前保存在 JSON，采用保守扫描；它不提供并发写入轨迹时的数据库外键保证。

当前使用完整快照与明确操作，不实现分支合并、外部目录监控或端云同步。


## 调用与组装边界

调用链统一为 `SkillService → RepositoryPipeline → SkillPersistence.skills`。
Service 只处理请求校验、项目绑定、pipeline 委托和资源关闭；配置解析、依赖排序及实例构建在 `api/factory.py`。
原有仓库注册、提交、导出等业务实现位于 `pipelines/repository.py`；ID/别名/前缀解析以及版本事务位于 `persistence/db_services/skills/service.py`。
Skill 数据入口统一为 `persistence.skills`，不再单独构造 CentralSkillStore。

默认配置显式选择 `repository_pipeline: repository_ops`，该 pipeline 注入 `repository` 持久化实例。
自定义训练、演进后端通过 `training_pipeline`、`evolution_pipeline` 选择；存在多个同类实现时必须明确指定。
仅配置一个 SkillPersistence 的旧配置仍由工厂装配默认 RepositoryPipeline；`persistence_backend` 保留为这种配置的显式选择项。
它不能和已选定的 repository pipeline 同时指定，避免实际数据源歧义。

训练请求使用 `TrainingRequest(project_id=..., base_version_id=..., tasks=..., split=...)`，不接受 `skill_ref`。
精确版本查询通过 `persistence.skills.version_by_id(project_id, version_id)` 校验项目归属。
未配置具体训练/演进 pipeline 时对应操作仍报告未实现；此重构没有提供训练算法。
UI 图查询使用 `service.graph_queries(project_id)`，不再通过 Service 读取数据库连接。

此次重构不改变表、记录格式、版本关系、已有事务或云服务的数据契约，不需要数据迁移。
已有数据库直接重新打开即可；原有初始化仍只创建缺失表。回退代码无需回退数据库。
Python 内部导入从 `api.typing`/`api.content` 改为 `pipelines.typing`/`pipelines.content`，
直接构造 Service 的调用方应改用 `create_service`；HTTP 仓库路径和请求字段保持不变。
