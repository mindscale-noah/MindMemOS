# 中心仓 API

`mindmemos_skill` 的 Python API 与 HTTP 接入共用 `SkillService`，不依赖 SDK，也不调用商用云服务。
中心仓保存完整 Skill 快照；来源目录与导出目录均不跟踪。`skills.current_version_id` 是当前版本的唯一依据。

## 初始化与调用

默认配置 `config/mindmemos_skill/default.yaml` 使用本地 SQLite，且不启用 remote 表。
显式配置组装不连接数据库；省略配置参数时，本地工厂会初始化存储。工厂、pipeline 业务方法（包括仓库和图查询）及关闭接口均为异步，构造函数与配置校验仍同步。HTTP 请求格式不变。Python 调用示例：

```python
import asyncio

from mindmemos_skill import create_service
from mindmemos_skill.pipelines.typing import RegisterRequest, CommitRequest, ExportRequest

async def main():
    async with await create_service("config/mindmemos_skill/default.yaml") as service:
        await service.initialize_storage()
        first = await service.register_skill(RegisterRequest(
            project_id="demo", name="Data analysis", alias="analysis",
            source="./my-skill", message="Initial registration",
        ))
        second = await service.commit_skill(CommitRequest(
            project_id="demo", skill_ref=first.skill.skill_id,
            source="./edited-skill", message="Improve instructions",
        ))
        destination = await service.export_skill(ExportRequest(
            project_id="demo", skill_ref="analysis", directory="./application/analysis",
        ))


if __name__ == "__main__":
    asyncio.run(main())
```

方法接收 `pipelines/typing.py` 中的请求对象，也接受同结构字典并校验。
`source` 可以是目录、SKILL.md 路径，或 `skill.snapshot()` 返回的完整 `SkillSnapshot`。
路径首次注册为 static Skill；目录提交保留当前版本的实现类型和固定配置。
显式提交完整 snapshot 可以同时更改实现与固定配置。导出只写 bundle 与 resources，
不把实现配置或仓库元信息偷偷写入应用目录；完整实现配置仍保存在仓内版本中。

`await service.initialize_storage()` 创建缺失表并执行已有版本标签迁移，不重建已有数据库；本次异步接口调整不改变表、快照或版本语义。已有版本库的索引迁移与项目回填要求见
[持久化说明](../../src/mindmemos_skill/mindmemos_skill/persistence/README.md)。
每个服务只有一个全局 SkillPersistence，数据库配置直接放在 `persistence.database` 下。

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
Service 只处理请求校验、项目绑定、pipeline 委托和资源关闭；配置解析、单一存储及 pipeline 实例构建在 `api/factory.py`。
原有仓库注册、提交、导出等业务实现位于 `pipelines/repository.py`；ID/别名/前缀解析以及版本事务位于 `persistence/db_services/skills/service.py`。
Skill 数据入口统一为 `persistence.skills`，不再单独构造 CentralSkillStore。

默认配置直接定义存储与业务入口：

```yaml
persistence:
  config:
    include_remote: true
  database:
    class_name: sqlite
    config:
      url: sqlite+pysqlite:///./data/mindmemos_skill/central.db
repository_pipeline:
  class_name: repository
  config: {}
training_pipelines: {}
evolution_pipelines: {}
```

`training_pipelines` 与 `evolution_pipelines` 使用名称到完整定义的字典，例如：

```yaml
training_pipelines:
  train_a:
    class_name: custom_train_a
    config:
      algorithm:
        class_name: algorithm_a
        config: {}
  train_b:
    class_name: custom_train_b
    config:
      algorithm:
        class_name: algorithm_a
        config: {}
```

上述注册名与内部字段仅作示意，须匹配具体 pipeline 的 `config_type`。每个流程独立定义并构造自己的组件，允许重复；嵌套结构由实现决定，Agent 可以属于 Environment，也可以直接属于 Pipeline。配置没有全局资源表和名称引用。

服务工厂只构造一个 SkillPersistence，以 `persistence` 参数传给所有 pipeline。每个 pipeline 构造函数接收 `config` 和 `persistence`，前者包含其完整配置。没有配置全局存储时注入 None；需要存储的自定义流程应明确拒绝，仓库流程在工厂预检时拒绝。所有流程配置由对应模型在构造前校验，内部组件实现名称及参数也应由该模型验证。

训练和演进请求通过 `pipeline` 字段选中配置中的名称，HTTP 与 Python 一致。只有一个定义时可省略，多个时必须指定；空字典或省略字段禁用对应操作，不设默认流程，也不提供并发调度。训练与演进可使用相同的请求名称，按操作分别解析。

旧 `backends`、`persistence_backend`、字符串形式的 `repository_pipeline`、训练/演进名称列表和单数默认选择字段不再接受。迁移配置时把原数据库参数原样放进 `persistence.database`，存储选项放进 `persistence.config`，各 pipeline 参数直接放进对应定义；保留数据库 URL 和工作目录即可沿用已有数据。RepositoryPipeline 必须显式定义，不会自动补装。

关闭服务会先逆序关闭 pipeline，再关闭数据库。具体 pipeline 覆盖 `close()` 释放其内部资源，并自行处理构造失败时的部分资源；不能关闭借用的全局存储。

训练请求使用 `TrainingRequest(project_id=..., base_version_id=..., tasks=..., split=...)`，不接受 `skill_ref`。
精确版本查询通过 `persistence.skills.version_by_id(project_id, version_id)` 校验项目归属。
未配置具体训练/演进 pipeline 时对应操作仍报告未实现；此重构没有提供训练算法。
UI 图查询使用 `service.graph_queries(project_id)`，不再通过 Service 读取数据库连接。

此次重构不改变表、记录格式、版本关系、已有事务或云服务的数据契约，不需要数据迁移。
已有数据库直接重新打开即可；原有初始化仍只创建缺失表。回退代码无需回退数据库。
Python 内部导入从 `api.typing`/`api.content` 改为 `pipelines.typing`/`pipelines.content`，
直接构造 Service 的调用方应改用 `create_service`；HTTP 仓库路径和请求字段保持不变。

## 自定义异步流程

`TrainingPipeline.train()`、`EvolutionPipeline.evolve()`、`SkillTrainer.train()` 与 `SkillEvolver.evolve()` 使用 `async def` 实现，调用方直接 `await`。`RepositoryPipeline` 的存储操作及 `GraphQueryPipeline` 的查询操作同样异步；现有同步数据库事务通过工作线程执行。`Pipeline.close()` 应等待其拥有的异步资源关闭，不关闭借用的全局存储。服务按构造的逆序等待所有 pipeline 清理，即使某项清理失败，也继续关闭其他 pipeline 和数据库。

不要在这些业务方法中用 `asyncio.run(...)` 桥接；同步命令行入口统一创建一次事件循环。UI 的 `create_app(...)` 需要 `await` 完成服务组装；仅包装已有服务的 Skill HTTP `create_app(service)` 仍为同步函数。
