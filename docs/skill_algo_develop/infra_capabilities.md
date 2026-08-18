# Skill 算法基础设施能力

## 向量数据库

代码位于 `src/mindmemos_skill/mindmemos_skill/infra/vector_store/`，当前内置 PostgreSQL + pgvector 后端。算法和外部调用方只能依赖 `mindmemos_skill.infra.vector_store` 的公开入口，并通过配置中的 `provider` 切换后端；禁止直接导入或调用 `vector_store_impl` 下的具体后端。

```python
import asyncio

from mindmemos_skill.infra.vector_store import (
    BackendConfig,
    DatabaseScope,
    Record,
    TableRegistry,
    TableSpec,
    VectorFieldSpec,
    VectorQuery,
    VectorValue,
    bootstrap_vector_store,
)


async def main() -> None:
    config = {
        "provider": "pgvector",
        "options": {
            "dsn": "postgresql://postgres:postgres@127.0.0.1:5432/mindmemos",
            "schema": "mindmemos_skill",
        },
    }
    tables = TableRegistry(
        (
            TableSpec(
                name="skill_vectors",
                primary_key="skill_id",
                vectors=(VectorFieldSpec(name="embedding", dimensions=3),),
            ),
        )
    )
    tables.freeze()

    store = await bootstrap_vector_store(BackendConfig(**config), tables)
    scope = DatabaseScope(project_id="demo-project")
    try:
        await store.upsert_records(
            "skill_vectors",
            (
                Record(
                    table="skill_vectors",
                    record_id="skill-1",
                    scope=scope,
                    payload={},
                    vectors=VectorValue(dense={"embedding": (0.1, 0.2, 0.3)}),
                ),
            ),
        )
        hits = await store.search_vectors(
            VectorQuery(
                table="skill_vectors",
                scope=scope,
                vector_name="embedding",
                dense_vector=(0.1, 0.2, 0.3),
                top_k=5,
            )
        )
        print([(hit.record.record_id, hit.score) for hit in hits])
    finally:
        await store.close()


asyncio.run(main())
```

切换数据库时只修改 `config["provider"]` 和对应的 `options`；新后端必须先注册到向量数据库的后端注册表中。

## 在 persistence 中定义新表

`persistence` 拥有业务表结构，`infra` 只提供通用表协议和后端实现。普通数据库表与向量表使用两套独立的 `TableRegistry`，不能混合注册。

### 普通数据库表

在 `persistence/tables.py` 中定义表名，并把 `infra.database.TableSpec` 加入 `build_persistence_tables()`：

```python
from ..infra.database import FieldSpec, FieldType, IndexSpec, TableSpec

EXPERIENCE_TABLE = "experiences"

TableSpec(
    name=EXPERIENCE_TABLE,
    primary_key="experience_id",
    fields=(
        FieldSpec(name="experience_id", field_type=FieldType.TEXT, nullable=False),
        FieldSpec(name="content", field_type=FieldType.TEXT, nullable=False),
        FieldSpec(name="metadata", field_type=FieldType.JSON, nullable=False, default={}),
    ),
    indexes=(IndexSpec(name="experiences_content_idx", fields=("content",)),),
    scope_scoped=False,
)
```

同时需要：

1. 在 `persistence/models.py` 定义对应的 Pydantic Record 模型。
2. 在 `persistence/records.py` 的 `PersistenceRecord` 和 `_MODEL_TABLES` 中注册模型与主键映射。
3. 新增一个有序 migration 并更新当前 schema version；已经发布的 migration 不可修改。
4. 添加表结构、Record 转换和迁移测试。

### 向量表

向量表建议集中定义在独立的 `persistence/vector_tables.py`，只使用 `infra.vector_store` 的类型：

```python
from ..infra.vector_store import (
    FieldSpec,
    FieldType,
    TableRegistry,
    TableSpec,
    VectorFieldSpec,
)

EXPERIENCE_VECTOR_TABLE = "experience_vectors"


def build_persistence_vector_tables() -> TableRegistry:
    registry = TableRegistry(
        (
            TableSpec(
                name=EXPERIENCE_VECTOR_TABLE,
                primary_key="experience_id",
                fields=(
                    FieldSpec(name="experience_id", field_type=FieldType.TEXT, nullable=False),
                    FieldSpec(name="content", field_type=FieldType.TEXT, nullable=False),
                ),
                vectors=(VectorFieldSpec(name="embedding", dimensions=1536),),
                scope_scoped=True,
            ),
        )
    )
    registry.freeze()
    return registry
```

向量表必须通过 `bootstrap_vector_store(BackendConfig(...), build_persistence_vector_tables())` 单独初始化。业务层使用通用 `Record`、`VectorValue` 和 `VectorQuery` 读写，不得把向量字段加入 `infra.database.TableSpec`，也不得直接依赖 pgvector 实现。若修改已存在的向量表结构，需要显式数据库迁移，不能依赖自动建表覆盖旧结构。

## 将 algos 算法接入 SkillApplication

先按行为选择算法族：消费已有轨迹并生成一次候选放在 `algos/trace2skill/<name>/`；算法内部主动采样、验证并多轮更新 Skill 放在 `algos/evolve/<name>/`。

### 1. 实现并注册算法

配置使用严格、不可变的 Pydantic 模型；算法构造函数接收 Application 注入的 `config` 和 `context`：

```python
from pydantic import BaseModel, ConfigDict

from ....registry import ComponentRequirements, ComponentType, register
from ....typing import EvolveInput, EvolveOutput


class DemoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    epochs: int = 1


@register(
    type=ComponentType.ALGO,  # 组件类别；算法固定使用 ALGO
    name="demo_evolve",  # 全局唯一且稳定的组件类型名，对应配置中的 type
    config_model=DemoConfig,  # 用于校验 runtime.algorithms.<实例>.config 的模型
    capabilities={"evolve"},  # 此组件实现的 Application 算法能力及分发入口
    requirements=ComponentRequirements(  # Application 构造组件前必须满足的依赖
        required_model_roles=frozenset({"chat"}),  # 配置必须为 chat 角色绑定一个模型
    ),
)
class DemoEvolve:
    algorithm_name = "demo_evolve"

    def __init__(self, *, config: DemoConfig, context) -> None:
        self._config = config
        self._chat_model = context.models["chat"]
        self._agents = context.agents

    async def evolve(self, request: EvolveInput) -> EvolveOutput:
        ...
```

`trace2skill` 算法将 capability 改为 `{"optimize"}`，并实现 `async optimize(Trace2SkillInput) -> Trace2SkillOutput`。内置算法还要把包加入 `registry/_registry.py` 的 `_BUILTIN_MODULES`；外部算法包通过环境变量 `MINDMEMOS_SKILL_COMPONENT_MODULES='["my_package.demo_algo"]'` 加载。

### 2. 配置 SkillApplication

`runtime.algorithms` 的 key 是调用时使用的算法实例名，`type` 是上面注册的组件名：

```python
from mindmemos_skill import SkillApplication

application = await SkillApplication.from_config(
    {
        "runtime": {
            "models": {
                "optimizer-model": {"model": "openai/gpt-5.4-mini"},
            },
            "algorithms": {
                "demo": {
                    "type": "demo_evolve",
                    "model_roles": {"chat": "optimizer-model"},
                    "config": {"epochs": 1},
                },
            },
        },
    }
)
```

调用时使用实例名 `demo`：

```python
from mindmemos_skill.application.algorithms import AlgorithmCommitPolicy, EvolveRunRequest

result = await application.run_evolve(
    EvolveRunRequest(
        run_id="demo-run-001",
        algorithm_name="demo",
        skill_ref="my-skill",
        train_tasks=train_tasks,
        commit_policy=AlgorithmCommitPolicy.PERSIST,
    )
)
```

### 3. algos 内部代码规范

1. 算法只负责计算并返回候选、指标和本次产生的全部物理 attempt 轨迹；不得直接创建正式 Skill 版本、写 persistence、push 远端或实现另一套 Application 编排。
2. 输入输出使用 `mindmemos_skill.typing` 的公共 `Trace2SkillInput/Output` 或 `EvolveInput/Output`；算法专属配置、状态和报告使用严格 Pydantic 模型，禁止用无约束字典代替稳定契约。
3. 模型和 Agent 只能从 `context.models`、`context.agents` 获取，禁止在算法内部读取密钥或自行构造 Application 级客户端；嵌套模型调用应绑定 `llm_run_context(request.run_id)`。
4. 环境和数据集通过统一 registry/config 选择，禁止硬编码环境白名单；并发、重试和超时必须有界，失败 attempt 也要保留在输出轨迹中。
5. `algorithm.py` 只保留主流程；配置放 `config.py`，输入输出及过程结构放 `contracts.py`/`models.py`，提示词放 `prompts.py`，独立步骤拆成小模块。
6. `algorithm_name`、注册名和配置中的 `type` 保持稳定；prompt/config 发生行为变化时显式更新版本。新增算法至少覆盖注册、配置校验、协议实现、失败路径和 SkillApplication 调用测试。

### 4. Skill 读取与算法结果落库

这部分属于 Application 编排层，放在 `application/algorithms/orchestrator.py`，不能放在 `algos` 内部。职责顺序是：

```text
SkillApplication
  -> Orchestrator 通过 LocalSkillManager 读取基准 Skill
  -> 构造完整的算法 Input
  -> SkillAlgorithms 分发到 algos
  -> algos 返回未持久化 Output
  -> Orchestrator 持久化轨迹、候选和 AlgorithmLog
  -> 按 commit_policy 决定是否 push
```

现有 `run_trace2skill()` 和 `run_evolve()` 已遵守该流程：`_resolve_base_skill()` 调用 `LocalSkillManager.get_skill()/get_version()`，`_commit()` 调用 `persist_algorithm_candidate()` 和轨迹、日志 recorder。算法只接收已经加载完成的 `base_skill`，不感知数据库。

新增 `skill_graph` 时也应新增对应的 Application orchestrator 方法，而不是在 `DefaultSkillGraph` 中读写数据库：

```python
async def run_skill_graph(self, request: SkillGraphRunRequest) -> SkillGraphRunResult:
    base_skill = await self._resolve_base_skill(request.skill_ref, request.base_version_id)
    output = await self._algorithms.build_skill_graph(
        SkillGraphInput(run_id=request.run_id, base_skill=base_skill),
        algorithm_name=request.algorithm_name,
    )
    if request.commit_policy is not AlgorithmCommitPolicy.DRY_RUN:
        await self._persist_skill_graph(output)  # 调用 persistence service/repository
        await self._record_algorithm_log(...)
    return SkillGraphRunResult.from_output(output)
```

如果结果是新 Skill，复用 `LocalSkillManager.persist_algorithm_candidate()`；如果结果是 `skill_graph` 专属表记录，则在 `persistence` 中定义独立的模型、Record 转换和 repository/service，再把持久化函数注入 orchestrator。底层 repository 负责数据库读写，orchestrator 负责事务时机和 commit policy，算法负责纯计算。

### 5. 新增一种 capability

`capabilities` 不是任意标签，而是贯穿配置校验、协议检查、实例分组、分发和 Application API 的操作契约。因此新增 `skill_graph` 不能只写 `capabilities={"skill_graph"}`，需要同时完成以下边界：

1. 在 `typing/` 定义稳定的 `SkillGraphInput` 和 `SkillGraphOutput`。
2. 在 `service/protocols.py` 定义带 `@runtime_checkable` 的 `SkillGraphBuilder`，约定 `async build_skill_graph(request) -> SkillGraphOutput`。
3. 在 `application/enums.py` 的 `SkillApplicationCapability` 中增加 `SKILL_GRAPH = "skill_graph"`。
4. 在 `config/compiler.py` 的算法 capability 白名单中加入 `skill_graph`。
5. 在 `application/components.py` 增加协议校验、按实例名收集 graph builder，并注入 `SkillAlgorithms`。
6. 在 `service/skill.py` 增加 `_skill_graph_builders`、`capabilities`、`resolve_name()` 映射和 `build_skill_graph()` 分发方法。
7. 在 `SkillApplication` 增加公开方法，并在这里统一处理日志、持久化等 Application 级副作用。
8. 添加配置编译、错误协议、同 capability 多实例分发、`application.capabilities` 和公开调用测试。

算法注册形式如下：

```python
@register(
    type=ComponentType.ALGO,  # 注册为算法组件
    name="default_skill_graph",  # 配置中 runtime.algorithms.<实例>.type 的值
    config_model=SkillGraphConfig,  # skill_graph 算法自己的严格配置模型
    capabilities={"skill_graph"},  # 声明实现新增的 skill_graph 操作契约
    requirements=ComponentRequirements(  # 声明构造该算法所需的跨组件依赖
        required_model_roles=frozenset({"embedding"}),  # 要求配置绑定 embedding 模型角色
    ),
)
class DefaultSkillGraph:
    algorithm_name = "default_skill_graph"

    def __init__(self, *, config: SkillGraphConfig, context) -> None:
        self._config = config
        self._embedding_model = context.models["embedding"]

    async def build_skill_graph(self, request: SkillGraphInput) -> SkillGraphOutput:
        ...
```

配置时仍然使用实例名和组件类型名分离的方式：

```python
"algorithms": {
    "graph-builder": {
        "type": "default_skill_graph",
        "model_roles": {"embedding": "embedding-model"},
        "config": {},
    }
}
```

若 `skill_graph` 只是某个 evolve 算法内部生成的中间产物，不需要被其他调用方独立调用，则不应新增 capability；把图放入该算法的 output/artifacts，并继续声明 `capabilities={"evolve"}` 即可。

## SkillRuntime

`SkillRuntime` 负责把持久化的 Skill 转换为当前 task 可注入 Agent 的内容。公共流程是：Coordinator 根据 `skill.runtime_type` 解析 Runtime，校验 `runtime_schema_version/runtime_metadata`，调用 `on_task()` 创建 attempt 隔离的 session，再由 Agent 的 `system_prompt`、`tool` 或 `filesystem` 适配器完成注入；attempt 结束后 session 被关闭。

### 静态 SkillRuntime

静态 Runtime 适用于完整内容已经写在 `SKILL.md` 中、执行时不需要选择或加载额外内容的 Skill。它是默认机制，不需要手动注册：

```text
AgentExecutionRequest.skills
  -> StaticSkillRuntime.on_task()
  -> session.initial_content = skill.content
  -> Agent adapter 注入完整 SKILL.md
```

持久化 Skill 时使用固定 Runtime 信息：

```python
candidate = SkillCandidate(
    blob={"SKILL.md": "# Spreadsheet Skill\n\n先检查工作表结构，再执行修改。"},
    runtime_type="static",
    runtime_schema_version=1,
    runtime_metadata={},
)
```

候选由 Application 持久化后，执行时把已经加载的 `Skill` 放入请求：

```python
trajectory = await application.execute(
    "react-agent",
    AgentExecutionRequest(
        task=task,
        rollout=rollout,
        skills=[loaded_skill],
    ),
)
```

静态 Runtime 不做推荐，也不提供动态 resource；Skill 的完整内容会在每次 task 中按 Agent 配置的注入模式直接使用。

### 动态 SkillRuntime

动态 Runtime 适用于同一个 Skill 需要按 task 选择子能力、延迟加载资源或动态推荐 Skill 的场景。以 `skill_graph` 为例，离线算法负责构建并持久化推荐索引，在线 Runtime 只查询已经存在的索引：

```text
离线：SkillApplication.run_skill_graph() -> 构建图/向量 -> persistence 落库
在线：Agent task -> SkillGraphRuntime.on_task() -> 查询推荐服务 -> 创建 task session -> Agent 注入
```

Runtime 只依赖抽象推荐器和 Skill loader，具体数据库实现由 Application 注入：

```python
from collections.abc import Awaitable, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from mindmemos_skill import (
    Skill,
    SkillResourceDescriptor,
    SkillResourcePayload,
    SkillRuntime,
    SkillRuntimeRequest,
    SkillRuntimeSession,
)


class SkillGraphRuntimeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    graph_id: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


Recommend = Callable[[str, str, int], Awaitable[Sequence[str]]]
LoadSkill = Callable[[str], Awaitable[Skill]]


class SkillGraphSession(SkillRuntimeSession):
    def __init__(self, *, router_skill: Skill, recommended: Sequence[Skill]) -> None:
        self._recommended = {skill.version_id: skill for skill in recommended}
        resources = [
            SkillResourceDescriptor(
                resource_id=f"recommended-skill:{skill.version_id}",
                name=skill.name,
                description=skill.description or "",
                metadata={"skill_id": skill.skill_id, "version_id": skill.version_id},
            )
            for skill in recommended
        ]
        super().__init__(
            skill=router_skill,  # 保持路由 Skill 的不可变版本身份
            initial_content="按当前任务选择并加载下列推荐 Skill。",
            resources=resources,
        )

    async def _load(self, resource_id: str) -> SkillResourcePayload:
        version_id = resource_id.removeprefix("recommended-skill:")
        skill = self._recommended[version_id]
        return SkillResourcePayload(resource_id=resource_id, content=skill.content)


class SkillGraphRuntime(SkillRuntime):
    runtime_type = "skill_graph"
    metadata_models: Mapping[int, type[BaseModel]] = {1: SkillGraphRuntimeMetadata}

    def __init__(self, *, recommend: Recommend, load_skill: LoadSkill) -> None:
        self._recommend = recommend
        self._load_skill = load_skill

    async def on_task(self, request: SkillRuntimeRequest) -> SkillRuntimeSession:
        metadata = SkillGraphRuntimeMetadata.model_validate(
            self.parse_metadata(request.skill.runtime_schema_version, request.skill.runtime_metadata)
        )
        refs = await self._recommend(metadata.graph_id, request.task.instruction, metadata.top_k)
        recommended = [await self._load_skill(ref) for ref in refs]
        return SkillGraphSession(router_skill=request.skill, recommended=recommended)
```

在 Application 初始化后、注册或加载 `runtime_type="skill_graph"` 的 Skill 前注册 Runtime：

```python
runtime = SkillGraphRuntime(
    recommend=skill_graph_service.recommend,
    load_skill=skill_loader.load,
)
application.register_skill_runtime(runtime)
```

`SkillApplication.register_skill_runtime()` 会同时注册到 `LocalSkillManager` 和所有 Agent 的公共 `SkillRuntimeCoordinator`。持久化的路由 Skill 必须携带：

```python
runtime_type="skill_graph"
runtime_schema_version=1
runtime_metadata={"graph_id": "default", "top_k": 3}
```

执行时将这个路由 Skill 放入 `AgentExecutionRequest.skills`，Coordinator 会自动选择 `SkillGraphRuntime`：

```python
trajectory = await application.execute(
    "react-agent",
    AgentExecutionRequest(
        task=task,
        rollout=rollout,
        skills=[skill_graph_router_skill],
    ),
)
```

生产实现还需要限制候选 scope、去重并锁定 Skill 版本、固定排序规则、设置查询超时，并把推荐 ID、分数和实际加载资源写入 runtime trace。`SkillGraphRuntime` 不应直接访问 repository，也不应重新构建图。

上述 Runtime Demo 把推荐 Skill 作为路由 Skill 的动态 resource，因此轨迹首先绑定路由 Skill。如果需要每个推荐 Skill 都拥有独立 `SkillBinding`，应在 Application 层实现 `execute_with_skill_recommendation()`：先查询推荐服务，通过 `LocalSkillManager` 加载真实 Skill，再把这些 Skill 放进 `AgentExecutionRequest.skills` 并复用 `execute()`。这种方式更适合权限过滤、版本锁定和独立使用证据。
