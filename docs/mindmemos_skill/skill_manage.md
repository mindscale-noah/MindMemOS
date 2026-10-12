# Skill Manage 图推荐与动态 SkillSet

## 服务 Pipeline

服务通过单个 `manage_pipeline` 配置管理能力，内置注册名为 `graph_manage`。它同时负责组织、推荐和 Skill Set 查询，借用服务的 `SkillPersistence`。`algorithm` 使用 `SkillManageConfig` 的字段；推荐时配置必须与目标图快照的非密钥算法设置一致。不配置 `algorithm` 时仍可查询 Skill Set，组织与推荐调用会报未实现。Service 提供 `list_skill_sets()`、`list_skill_set_snapshots()` 和 `get_skill_set_snapshot()`；当前存储表示的 `mode` 为 `graph`。

例如在 `service` 配置内定义一个入口（实际部署时填写可用的 embedding 模型地址和凭据）：

```yaml
manage_pipeline:
  class_name: graph_manage
  config:
    algorithm:
      embedding:
        model: your-embedding-model
        base_url: http://127.0.0.1:8000/v1
        dimensions: 1536
      build_scene: false
      llm_linking: false
      query_rewrite: false
```

`service.organize_skills(OrganizeRequest(...))` 接收 `graph_id`、`actor_id`、完整的 `skill_version_ids` 列表、可选的 `message` 和按 Skill 家族 ID 映射的可选 `scene_records`。服务先确认项目授权，再按精确版本读取已保存 Skill；同一图内每个家族只允许一个版本。`message` 保存于新建图版本。同一输入、说明和配置的重试复用已保存快照；说明变化会形成新快照并将其设为当前生效版本。

`service.recommend_skills(RecommendRequest(...))` 接收 `graph_id`、可选的不可变 `graph_version_id`、公开 `Task` 和正整数 `limit`。省略快照 ID 时，在请求开始读取并固定当前图头。它从该快照恢复 `skill_manage` 的成员范围，校验版本所属项目、Skill 内容与算法配置，再调用推荐算法，按排名返回精确 `skill_version_ids`。显式指定旧快照的请求不随图头变化。Pipeline、存储、Embedding 和模型操作均采用原生异步接口，调用者逐层使用 `await`。

HTTP 入口也已接通这两条异步 Pipeline：

| 接口 | 请求字段 | 成功结果 | 权限 |
| --- | --- | --- | --- |
| `POST /v1/skills/algos/organize` | `alias` 或 `graph_id`、完整 `skill_version_ids`，可选 `message`、`scene_records` | `200`，含 `graph_version_id` 和输入版本 ID | `skill:write` |
| `POST /v1/skills/algos/recommend` | `graph_ref`（当前实现按精确 `graph_id` 解释）、`query`，可选 `graph_version_id`、`top_k` | `200`，含有序的精确 `skill_version_ids` | `skill:read` |

HTTP 身份提供项目边界与构图记录的 actor ID；请求体不接受项目 ID。当前 `alias` 直接用作精确 `graph_id`，尚无独立别名身份或前缀解析。组织完成才返回成功，不提交后台任务；`graph-resolve` 仍是占位能力。现有目标架构文档中的异步组织任务、可省略版本集合、独立 alias 身份及更完整的图引用解析尚未实现，不能把这版同步接口视为那些目标能力已经上线。

`algos/manage/skill_manage/` 提供注册名为 `skill_manage` 的 `SkillManageAlgorithm` 和算法专属 `SkillManageSkillSet`。集合仅管理调用方传入的成员及组装并发，不解析版本 ID，不自动注册 Skill。外围调度负责选择版本并构造 `SkillGraphStore`；持久化层核验传入 Skill 的完整快照与已保存版本一致。

## 使用方式

```python
import asyncio

from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.algos.manage.skill_manage import (
    SceneRecord,
    SkillManageAlgorithm,
    SkillManageConfig,
    SkillManageSkillSet,
)
from mindmemos_skill.infra.embedding import OpenAIEmbeddingConfig
from mindmemos_skill.persistence.db_services.skill_management.writer import SkillGraphStore

# These objects and immutable version IDs are selected by the outer scheduler.
skills = {"csv": csv_skill, "report": report_skill}
store = SkillGraphStore(
    persistence.database,
    project_id=project_id,
    graph_id="task-skills",
    actor_id=actor_id,
    version_ids={"csv": csv_version_id, "report": report_version_id},
)
config = SkillManageConfig(
    embedding=OpenAIEmbeddingConfig(
        model=embedding_model_name,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        dimensions=embedding_dimensions,
    ),
    chat=OpenAIChatConfig(
        model=chat_model_name,
        base_url=chat_base_url,
        api_key=chat_api_key,
    ),
    build_scene=True,
    llm_linking=True,
    query_rewrite=True,
    mode="unified_ppr",
)
scenes = {
    "csv": SceneRecord.model_validate({
        "scenes": [{
            "scene_id": "parse",
            "scene_name": "Parse tabular data",
            "scene_goal": "Parse CSV rows into structured records",
            "next_scene_rules": [],
        }],
    }),
}
algorithm = SkillManageAlgorithm(config, store=store, scene_records=scenes)
skill_set = SkillManageSkillSet(
    skills, algorithm=algorithm, limit=5, max_concurrency=4,
)
result = asyncio.run(env.rollout(
    task, work_dir, reward_data=reward_data, skill_set=skill_set,
))
```

构造算法、集合和客户端不发起模型调用，也不读写数据库。首次 `select(task)` 校验外围版本绑定，然后构建或复用图；`assemble(task)` 在选择后调用选中成员自己的组装方法。每个集合使用独立的管理算法实例。省略 `skill_set` 仍为 baseline。

数据库由外围创建、初始化和关闭。算法不关闭借用的数据库或注入客户端。`embedding_model=` 接受带有匹配 `config` 和异步 `embed(texts)` 的实现；`chat_model=` 接受带有匹配配置及异步 `run(...)` 的聊天实现，可复用现有 OpenAIChatAgent。SDK 传输测试可通过 `OpenAIEmbeddingClient(config, client=...)` 注入客户端。

## 构图与检索

- **严格解析**：保留来源的 frontmatter/name 要求、重复 YAML 键检查和内容大小限制。纯 Markdown 不自动补字段；描述遵循来源规则从描述字段或正文提取。
- **成员身份**：图索引使用集合 Skill ID 的无歧义编码；Scene ID 在其成员范围内编码。名称只参与文本匹配与展示，同名成员不会覆盖。
- **规则依赖边**：按输出与输入字段重叠度生成有向依赖。保留来源分词、阈值及双向评分规则。
- **LLM 连边**：以有界候选对批次推断 semantic、workflow、alternative 边，保存解释；输出必须引用当前批次中的节点，禁止自环、越界关系及非法权重。规则依赖边不被覆盖。请求串行执行，每批最多 `llm_batch_size` 对；全量候选对数量为 N(N-1)/2。
- **Scene**：只使用调用方显式提供的 SSL 记录，映射键为集合 Skill ID。保留扩展元数据，支持字符串形式或含 `next_scene_id` 的流转规则；拒绝重复 ID、自环和悬空目标。Skill→Scene 为 containment 边，Scene→Scene 为 flow 边。缺失记录不会触发生成；启用功能时无场景需显式传 `{}`。
- **Embedding**：Skill 和 Scene 均生成带标签的文本，并持久化文本哈希与向量。客户端按配置分批，验证响应顺序、数量、维度、有限性与非零向量。模型和地址必须显式配置；不默认发起付费调用。
- **Query rewrite**：从完整公开 Task 提取 goal、task_name、domain、operations、artifacts、constraints、keywords，改写结果同时用于查询向量和字段重排。关闭时使用来源的确定性查询解析。
- **推荐**：数据库在指定项目、图和不可变快照内执行余弦搜索，以向量排名播种 PPR，然后按来源字段权重重排。`skill_only` 只使用 Skill 节点；`unified_ppr` 使用联合图，将 Scene 的 PPR 分数归并到所属 Skill，返回唯一的集合成员 ID。Scene 不作为独立 Skill 注入。

目前对外提供这两个有实际执行语义的检索模式，不暴露来源中尚未实现不同执行逻辑的 `separate` / `fuse` 模式名称。Scene 联合传播、模型连边和改写属于此次补齐实现；来源基准对照覆盖原有解析和 PPR 数值，不代表新增模型步骤与其他 GOS 实现等价。

模型、解析或存储错误明确抛出，不静默切换为其他策略。失败构建会使该管理器不可推荐，直至后续成功组织；已经发布的历史快照仍保留。

## 持久化与恢复

当前使用 `skill_graphs_v1`、`skill_graph_versions_v1`、`skill_graph_nodes_v1`、`skill_graph_edges_v1` 四张核心表。每张图由 `current_version_id` 指向当前生效快照，不保存新的发布 revision。独立 Skill 服务不提供旧图表迁移或滚动部署桥接；当前结构详见 [图存储结构](graph_storage_schema.md)。

外围必须提供同项目的精确版本绑定，且每个成员对应不同版本，以符合已有图节点唯一约束。算法只能处理显式成员，不通过数据库扩大候选范围。现有 Skill 数据、版本谱系、展示标签和中心仓当前版本指针不改变。

先在事务外完成解析、模型推理和向量校验，再在单次事务内写入完整图快照、节点、边并更新 `current_version_id`。并发发布锁定同一图的写入行；相同输入和配置按请求哈希幂等复用，跨项目和跨图隔离。失败或中断回滚本次图发布，不产生部分可见快照；重试可以重建。不同输入生成新快照，保留旧快照与节点。模型调用发生于事务前，未完成发布的重试可能重复模型费用。

重新打开数据库后可复用已保存图与向量。读取时检查节点/边数量、向量数及完整内容哈希；损坏明确报错。单次推荐固定使用已加载的 graph_version_id，不随当前 head 切换。

Environment 的恢复机制不变：相同输入直接复用已保存的选择和组装结果，不重新推荐或调用模型。集合配置快照保存选择参数和非敏感模型配置，不保存 API key。

部署时使用已有正常初始化的 Skill 数据库，再部署此包。已有只有读图服务的调用方仍可读取新发布的 Skill 节点；UI 的图查询接口仍只展示 Skill 级视图。撤回本次代码不会删除新快照或旧数据。自动云服务迁移、后台任务、图 GC 和 UI Scene 展示不在本次接入范围。

## 验证边界

测试使用真实本地 SQLite、数据库余弦搜索和模型替身，覆盖同名节点、Scene 召回、规则/模型关系、改写、重载、并发幂等、事务中断、历史保留、项目隔离和 Environment 恢复。Embedding 客户端通过本地模拟 HTTP 传输验证真实 SDK 协议，没有付费请求。PostgreSQL 使用现有通用数据库接口，但需实际 PostgreSQL 服务才能验证其运行结果。
