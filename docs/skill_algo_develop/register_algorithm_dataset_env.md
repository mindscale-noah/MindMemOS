# 算法、Dataset 与 Env 注册指南

本文说明如何把新的算法、Dataset 和 Env 接入 `mindmemos_skill`，并进一步接入统一实验入口。算法设计、输入输出合同和持久化边界见 [`README.md`](README.md)，实验 YAML 与运行方式见 [`experiment_runner.md`](experiment_runner.md)。

## 1. 先区分两层注册

仓库中有两个用途不同的注册表：

| 注册表 | 位置 | 负责内容 |
| --- | --- | --- |
| 组件注册表 | `src/mindmemos_skill/mindmemos_skill/registry/_registry.py` | 让 Application 和运行时按名称构造 `ALGO`、`DATASET`、`ENV`、`AGENT` |
| 实验注册表 | `scripts/mindmemos_skill/experiments/registry.py` | 让统一实验入口按 `method` 选择算法 adapter、家族 runner、支持的环境和可选依赖 |

`@register(...)` 只完成包内组件注册，不会自动生成 CLI、实验参数、Dataset 构造逻辑或 YAML。一个算法如果只供 `SkillApplication` 使用，只注册组件即可；如果还要通过 `scripts/run_mindmemos_skill_experiment.sh` 运行，则必须同时注册实验 adapter。

内置组件依靠模块导入触发装饰器。Dataset 和 Env 的父包已经列在 `_BUILTIN_MODULES` 中，因此新增实现后必须从对应 `registered_*` 包的 `__init__.py` 导入；算法包则需要显式加入 `_BUILTIN_MODULES`。所有 `register/create/get_component` 调用都必须传 `ComponentType` 枚举成员，不能传普通字符串。

## 2. 注册新算法

### 2.1 选择已有算法家族

优先接入现有 capability：

- `trace2skill`：实现 `async optimize(request: Trace2SkillInput) -> Trace2SkillOutput`，注册 `capabilities={"optimize"}`。
- `evolve`：实现 `async evolve(request: EvolveInput) -> EvolveOutput`，注册 `capabilities={"evolve"}`。
- `analyze`：实现 `SkillAnalyzer` 协议，注册 `capabilities={"analyze"}`。

只有新增完全不同的 capability 时，才需要扩展 Application、Service 协议、编排请求和 SDK。不要为新的 `optimize` 或 `evolve` 算法增加第三个 family runner。

建议目录：

```text
src/mindmemos_skill/mindmemos_skill/algos/
└── evolve/
    └── my_algorithm/
        ├── __init__.py
        ├── algorithm.py
        ├── config.py
        ├── contracts.py
        └── prompts.py
```

最小 evolve 注册示例：

```python
from pydantic import BaseModel, ConfigDict, Field

from mindmemos_skill.application.components import AlgorithmBuildContext
from mindmemos_skill.registry import ComponentRequirements, ComponentType, register
from mindmemos_skill.typing import EvolveInput, EvolveOutput


class MyAlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(default=1, ge=1)


@register(
    type=ComponentType.ALGO,
    name="my_algorithm",
    config_model=MyAlgorithmConfig,
    capabilities={"evolve"},
    requirements=ComponentRequirements(
        required_model_roles=frozenset({"chat"}),
    ),
)
class MyAlgorithm:
    algorithm_name = "my_algorithm"

    def __init__(self, *, config: MyAlgorithmConfig, context: AlgorithmBuildContext) -> None:
        self._config = config
        self._chat_model = context.models["chat"]
        self._agents = context.agents

    async def evolve(self, request: EvolveInput) -> EvolveOutput:
        return EvolveOutput(
            run_id=request.run_id,
            final_skill=request.base_skill,
            changed=False,
        )
```

注册元数据要求：

- `name` 是稳定、唯一的组件类型名，同时应与 `algorithm_name` 保持一致。
- `config_model` 必须是严格 Pydantic 模型；配置编译器会在构造算法前完成校验。
- `capabilities` 必须至少包含 `analyze`、`optimize`、`evolve` 之一，并与真实异步方法一致。
- `required_model_roles` 声明算法需要的逻辑模型角色；实际模型由 `runtime.algorithms.<实例名>.model_roles` 注入。
- 构造函数使用 `__init__(*, config, context)`；不要在算法中读取 API Key、重新解析部署配置或直接写 Skill repository。

完成实现后：

1. 从算法目录的 `__init__.py` 导出公开类和配置。
2. 如需公共导入，再更新 `algos/<family>/__init__.py` 和 `algos/__init__.py`。
3. 将 `..algos.evolve.my_algorithm` 或对应 trace2skill 模块加入 `registry/_registry.py` 的 `_BUILTIN_MODULES`，确保装饰器一定执行。
4. 增加 registry、配置编译、runtime compose 和算法协议测试。

Application 配置中的 `type` 使用组件注册名，外层键是运行实例名：

```yaml
runtime:
  models:
    optimizer_model:
      model: openai/gpt-5.4-mini
  algorithms:
    production_optimizer:
      type: my_algorithm
      model_roles:
        chat: optimizer_model
      config:
        max_iterations: 2
```

这里 `my_algorithm` 选择组件工厂，`production_optimizer` 用于运行时具名分发。

### 2.2 接入统一实验入口

如果算法需要通过实验 YAML 运行，还要完成以下步骤：

1. 新增 `scripts/mindmemos_skill/experiments/my_algorithm.py`，暴露 `main(argv)`，负责参数解析、Dataset 构造、模型与 Agent 构造、算法调用和 artifact 落盘。
2. 在 `scripts/mindmemos_skill/experiments/registry.py` 的 `EXPERIMENTS` 中增加 `_evolve(...)` 或 `_trace2skill(...)`。
3. 在 `config/mindmemos_skill/my_algorithm/<environment>/<name>.yaml` 增加至少一份配置。
4. 扩展 `tests/scripts/test_run_mindmemos_skill_experiment.py`，并为 adapter 增加聚焦测试。

注册示例：

```python
_evolve(
    "my_algorithm",
    environments=frozenset({"mybench"}),
    environment_extras={"mybench": ("mybench",)},
    inject_environment_as_benchmark=True,
)
```

- `environments` 是该方法允许的顶层 `environment`。
- `environment_extras` 中的值必须对应 `src/mindmemos_skill/pyproject.toml` 已定义的 optional dependency extra。
- adapter 接受 `--benchmark`，且其值应等于顶层 `environment` 时，设置 `inject_environment_as_benchmark=True`。
- `common_extras` 默认包含 `llm`；不需要 LLM 或需要额外通用依赖时可显式覆盖。

不要新增算法专属 Shell 脚本，也不要在 `scripts/mindmemos_skill/runners/` 增加第三个文件；该目录固定只有 `evolve.py` 和 `trace2skill.py`。

## 3. 注册新 Dataset

Dataset 只负责把原始数据和固定划分转换成公共 `Task`，不负责执行 Agent、计算 reward 或编排并发。

建议目录：

```text
src/mindmemos_skill/mindmemos_skill/datasets/registered_datasets/mybench/
├── __init__.py
└── dataset.py
```

最小示例：

```python
from collections.abc import Mapping

from mindmemos_skill.datasets.base import TaskDataset
from mindmemos_skill.registry import ComponentType, register
from mindmemos_skill.typing import Task


@register(type=ComponentType.DATASET, name="mybench_split")
class MyBenchDataset(TaskDataset):
    def __init__(self, *, splits: Mapping[str, list[dict[str, str]]]) -> None:
        self._splits = splits

    def split(self, name: str) -> list[Task]:
        if name not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {name!r}")
        return [
            Task(
                task_id=item["id"],
                instruction=item["instruction"],
                tags=[name],
                metadata={"source_split": name},
            )
            for item in self._splits.get(name, [])
        ]
```

注册完成后：

1. 在 `registered_datasets/mybench/__init__.py` 导出 `MyBenchDataset`。
2. 在 `datasets/registered_datasets/__init__.py` 导入并导出它；`load_builtin_components()` 导入该父包时才会触发注册。
3. 如果 adapter 使用 `from mindmemos_skill.datasets import MyBenchDataset`，还要更新 `datasets/__init__.py`。
4. 增加测试，至少验证注册名、实现模块路径、三个 split、稳定且唯一的 `task_id`、非法 split 和缺失数据的错误信息。

可直接检查组件注册：

```python
from mindmemos_skill.registry import ComponentType, create, get_component

spec = get_component(type=ComponentType.DATASET, name="mybench_split")
dataset = create(
    type=ComponentType.DATASET,
    name="mybench_split",
    splits={"train": [], "validation": [], "test": []},
)
```

当前实验 adapters 仍显式实现各 benchmark 的 `argparse` 参数和 `build_dataset(...)` 分支，因此仅注册 Dataset 不会让统一实验入口自动识别它。需要在所有支持该 Dataset 的 adapter 中补充 CLI choice、构造参数和路径默认值。顶层 `environment: mybench` 是实验环境名，不是 Dataset 的组件注册名 `mybench_split`。

大型原始数据放在被 Git 忽略的 `data/mindmemos_skill/`，固定划分和初始 Skill 放在 `resources/mindmemos_skill/`；不要把数据下载、网络访问或模型调用放进 Dataset 构造函数。

## 4. 注册新 Env

Env 负责一次物理 rollout attempt 的准备、执行适配、reward 计算和资源释放；batch、重试、sample fan-out 与全局并发由 Trainer/Scheduler 管理。

建议目录：

```text
src/mindmemos_skill/mindmemos_skill/envs/registered_envs/mybench/
├── __init__.py
└── env.py
```

最小示例：

```python
from pydantic import Field

from mindmemos_skill.envs.base import BaseEnv, PreparedRollout
from mindmemos_skill.registry import ComponentType, register
from mindmemos_skill.typing import EnvConfig, Reward, Trajectory


class MyBenchEnvConfig(EnvConfig):
    success_marker: str = Field(default="PASS", min_length=1)


@register(type=ComponentType.ENV, name="mybench")
class MyBenchEnv(BaseEnv[MyBenchEnvConfig]):
    config_type = MyBenchEnvConfig

    async def _evaluate(
        self,
        *,
        trajectory: Trajectory,
        prepared: PreparedRollout,
    ) -> Reward:
        del prepared
        passed = self.config.success_marker in str(trajectory.messages)
        return Reward(score=float(passed), metadata={"passed": passed})
```

`BaseEnv.rollout()` 已统一调用 `_prepare -> _execute -> _evaluate -> _teardown`。普通文本或工具型环境通常只需实现 `_evaluate`；模拟器、sidecar 或特殊协议环境再覆盖 `_prepare`、`_execute`、`_teardown`。`_teardown` 必须可安全处理失败路径，不要删除需要保留的 rollout artifact。

注册完成后：

1. 在 `registered_envs/mybench/__init__.py` 导出 Env 与配置。
2. 在 `envs/registered_envs/__init__.py` 导入并导出它，确保内置加载时执行装饰器。
3. 如需 `from mindmemos_skill.envs import MyBenchEnv`，再更新 `envs/__init__.py`。
4. 增加 registry、配置校验、reward、异常清理、workspace 隔离和并发安全测试。

运行时按注册名构造：

```python
from mindmemos_skill.envs import get_env

env = get_env(name="mybench", config={"success_marker": "PASS"})
```

算法的 run config 必须把 `env_ref` 指向 Env 注册名，并把 Env 专用参数放入 `env_options`：

```python
{
    "dataset": {
        "env_ref": "mybench",
        "agent_ref": "react",
        "env_options": {"success_marker": "PASS"},
        "agent_options": {},
    }
}
```

如果 Env 依赖额外第三方包，在 `src/mindmemos_skill/pyproject.toml` 增加最小 optional dependency extra，并在实验注册表的 `environment_extras` 中按环境加载；不要把重型 benchmark 依赖加入核心 `dependencies`。

## 5. 接入一个全新 benchmark 的顺序

一个新 benchmark 通常同时需要 Dataset 和 Env，建议按以下顺序接入：

1. 先定义 Dataset，把 train/validation/test 稳定转换为 `Task`。
2. 再定义 Env，验证 prompt/action/feedback/reward 和 trajectory metadata。
3. 更新两个 `registered_*` 父包的导入和公开导出。
4. 先用 `create(ComponentType.DATASET, ...)` 与 `get_env(...)` 做包内 smoke test。
5. 在目标算法 adapter 中增加 benchmark 参数、Dataset 构造和 `env_options`。
6. 在实验注册表中把新环境加入目标算法的 `environments`，并声明 optional extra。
7. 增加 `config/mindmemos_skill/<method>/<environment>/<name>.yaml`。
8. 先执行 `--dry-run`，确认 method、environment、extra、最终参数、runner 和输出路径，再运行最小 smoke experiment。

Dataset 注册名、Env 注册名和实验环境名可以不同，但对应关系必须在 adapter 中显式写清。推荐约定为：Dataset 使用 `<benchmark>_split` 或更具体的划分名，Env 与实验顶层 environment 使用同一个 `<benchmark>` 名称。

## 6. 验证命令

先运行不发起模型请求的聚焦检查：

```bash
UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache \
  uv run pytest \
  tests/mindmemos_skill/test_dataset_registry.py \
  tests/mindmemos_skill/test_env_registry.py \
  tests/mindmemos_skill/test_config_compiler.py \
  tests/scripts/test_run_mindmemos_skill_experiment.py -q

UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache \
  scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/<method>/<environment>/<name>.yaml \
  --dry-run

UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache \
  uv run ruff check \
  src/mindmemos_skill/mindmemos_skill/algos/<family>/<algorithm> \
  src/mindmemos_skill/mindmemos_skill/datasets/registered_datasets/<dataset> \
  src/mindmemos_skill/mindmemos_skill/envs/registered_envs/<environment>

git diff --check
```

最终合入前确认：组件名无冲突、内置导入链完整、配置模型拒绝未知字段、算法 capability 与方法一致、Dataset task ID 稳定、Env reward 可复现、统一 runner 仍只有两个家族入口、所有新增 YAML 都能成功构建 invocation，且 dry-run 未输出任何凭证。
