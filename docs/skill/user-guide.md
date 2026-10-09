# Skill 本地使用指南

本文使用 `mindmemos_skill` 的 Python Service，所有示例都把数据保存在本地 SQLite 中，不需要云端账号或 HTTP 服务。代码在仓库根目录执行，使用当前仓库的 `.venv/bin/python`；首次使用请先安装项目依赖（例如 `uv sync --all-packages`）。文中的 `default_project_id` 是本地仓库的数据隔离标识。

## 1. 初始化 Service

启动时必须提供完整 YAML，包含 `service.home` 和 `fastapi.auth`。HOME 只决定文件位置；无鉴权项目 ID 由 `fastapi.auth.default_project_id` 指定。

```yaml
# output/my-service.yaml
service:
  home: ./my-skill-home
  persistence:
    database:
      class_name: sqlite
  repository_pipeline:
    class_name: repository
fastapi:
  auth:
    mode: none
    default_project_id: personal
```

```python
import asyncio
from mindmemos_skill import create_service

async def main() -> None:
    async with await create_service("output/my-service.yaml") as service:
        print(service.default_project_id)  # personal

asyncio.run(main())
```

相对 HOME 以 YAML 所在目录为基准，此例使用 `output/my-skill-home`。SQLite URL 未配置时，使用 HOME 下的 `data/central.db`。显式数据库 URL 保持自身语义；SQLite URL 中的相对路径仍以进程工作目录为基准。服务自动创建需要的目录和当前表结构。`async with` 关闭资源。Python 请求模型要求显式填写 `project_id`，本例填写 `service.default_project_id`。

JWT 模式使用验证通过的 JWT 中的项目 ID；本地鉴权模式使用 API key 文件中的项目 ID。这两种模式禁止配置默认项目。直接 Python 调用鉴权服务时，宿主程序必须先验证凭据，再通过 `request_principal()` 传入可信身份；仅构造请求模型不构成鉴权。

## 2. 注册和管理 Skill

Skill 源目录至少包含 `SKILL.md`。注册会读取当时的内容，生成中心仓身份和初始不可变版本；此后修改源文件不会自动修改已注册版本。下面的代码放在上一节的 `async with await create_service("output/my-service.yaml") as service:` 代码块内；先在 `output/my-skill-home/source/SKILL.md` 写入 Skill 内容。

```python
from mindmemos_skill.api.typing import (
    ActiveRequest, CommitFromLocalRequest, DiffRequest, ExportRequest,
    FileSkillInfo, HistoryRequest, ListRequest, RegisterFromLocalRequest,
    SetMetaRequest, ShowRequest,
)

project = service.default_project_id
source = "output/my-skill-home/source"
registered = await service.register_from_local(RegisterFromLocalRequest(
    project_id=project, alias="my-skill",
    skill=FileSkillInfo(path=source, commit_message="Initial version"),
))
skill_id = registered.skill_id
first_version_id = registered.skill_version_id
families = await service.list_skills(ListRequest(project_id=project))
active = await service.show_skill(ShowRequest(project_id=project, skill_id=skill_id))

# Edit SKILL.md before committing the next immutable version.
updated = await service.commit_from_local(CommitFromLocalRequest(
    project_id=project, skill_id=skill_id,
    skill=FileSkillInfo(path=source, commit_message="Improve instructions"),
    base_version_id=first_version_id,
))
versions = await service.skill_history(HistoryRequest(project_id=project, skill_id=skill_id))
change = await service.diff_skill(DiffRequest(
    project_id=project, from_version_id=first_version_id, to_version_id=updated.version.version_id,
))
await service.active_skill(ActiveRequest(project_id=project, version_id=first_version_id))
await service.set_skill_meta(SetMetaRequest(project_id=project, skill_id=skill_id, name="Renamed Skill"))
await service.export_skill(ExportRequest(
    project_id=project, skill_id=skill_id, directory="output/my-skill-home/exported",
))
```

`show_skill` 使用 `skill_id` 或 `version_id`；省略版本时读取 active 版本。别名和唯一前缀先通过解析接口转换为精确身份。`active_skill(ActiveRequest(project_id=..., version_id=...))` 激活已有版本。`commit_skill` 的 `base_version_id` 指定父版本，可以从历史版本创建分支；发布时通过 active 指针的原子比较防止覆盖并发更新。再次导出到非空目录需显式传 `overwrite=True`。`delete_skill(SkillRequest(project_id=..., skill_ref=...))` 执行软删除，保留历史记录及证据引用，已删除家族不再出现在正常读取中。

## 3. 调用内置训练算法

训练由两层选择组成：`algo` 指定已配置的 Trainer（如 `treeskill` 或 `skill_forest`），`env` 提供任务执行与评分环境的 `class_name/config` 配置。请求必须给出精确输入版本的 `init_skill_version_id`、完整 `DataSample` 列表、每个任务恰好一次的 train/val/test 分配，以及专属的 `work_dir`。`run_training()` 等待训练、评估和新版本发布完成；结果包含 `skill_id`、`version_id` 和 `run_id`。

下面是一个可用于本地 Skill-Forest 的小规模配置。先创建目录 `mkdir -p output/skill-forest`，将 YAML 保存为 `output/skill-forest/service.yaml`。在 `optimizer` 和环境 `agent` 配置中分别填写 `api_key` 和 `base_url`，不从环境变量读取。把配置中的 `model` 改成该服务实际支持的模型。配置和请求均在本地运行，但内置优化器及此例的 SearchQA Agent 会调用所配置的模型服务。

```yaml
service:
  home: .
  persistence:
    database:
      class_name: sqlite
      config:
        url: sqlite+pysqlite:///./output/skill-forest/skills.db
  repository_pipeline:
    class_name: repository
  training_pipelines:
    forest:
      class_name: training
      config:
        trainer:
          class_name: skill_forest
          config:
            optimizer:
              model: YOUR_MODEL
              api_key: YOUR_API_KEY
              base_url: https://gateway.example.com/v1
            num_trees: 1
            candidates_per_node: 1
            max_depth: 1
            expansion_budget: 1
            expansion_concurrency: 1
            rollout_concurrency: 2
            model_concurrency: 1
        evaluation_concurrency: 2
fastapi:
  auth:
    mode: none
    default_project_id: personal
```

```python
import asyncio
from pathlib import Path
from uuid import uuid4

from mindmemos_skill import create_service
from mindmemos_skill.envs.data import DataSample
from mindmemos_skill.envs.builtin_envs.searchqa.typing import SearchQARewardData, SearchQATask
from mindmemos_skill.api.typing import RegisterRequest, StructuredSkillInfo, TrainingRequest, TrainingSplit


async def main() -> None:
    run_dir = Path("output/skill-forest") / uuid4().hex
    questions = [
        ("t1", "What color is the apple?", "The apple is red.", "red"),
        ("t2", "What color is the sky?", "The sky is blue.", "blue"),
        ("t3", "What color is the leaf?", "The leaf is green.", "green"),
        ("t4", "What color is the sun?", "The sun is yellow.", "yellow"),
        ("v1", "What color is the cup?", "The cup is white.", "white"),
        ("e1", "What color is the car?", "The car is black.", "black"),
    ]
    samples = [
        DataSample(
            task=SearchQATask(id=identifier, instruction=question, context=context),
            reward_data=SearchQARewardData(answers=[answer], source_sha256="guide-v1"),
        )
        for identifier, question, context, answer in questions
    ]
    async with await create_service("output/skill-forest/service.yaml") as service:
        registered = await service.register_skill(
            RegisterRequest(
                project_id=service.default_project_id,
                alias=f"passage-qa-{uuid4().hex[:8]}",
                skill=StructuredSkillInfo(
                    content="# Passage QA\nUse only the supplied context. Return <answer>...</answer>.\n",
                    name="Passage QA", description="Answer from supplied passages", standard_format=False,
                ),
            )
        )
        result = await service.run_training(
            TrainingRequest(
                algo="forest",
                project_id=service.default_project_id,
                env={"class_name": "searchqa", "config": {"agent": {"model": "YOUR_MODEL", "api_key": "YOUR_API_KEY", "base_url": "https://gateway.example.com/v1"}, "max_turns": 1}},
                init_skill_version_id=registered.skill_version_id,
                work_dir=str(run_dir),
                tasks=samples,
                split=TrainingSplit(train=["t1", "t2", "t3", "t4"], val=["v1"], test=["e1"]),
            )
        )
        print(result.version_id, result.run_id)


asyncio.run(main())
```

这个调用会真实运行 Skill-Forest，模型输出与服务连通性会影响结果。`train` 用于优化；最终候选冻结后，pipeline 才在独立目录评估 `val` 和 `test`。运行产物保存在 `work_dir`，同一输入和目录可以用于兼容的恢复；更换任务、配置或初始版本时使用新目录。要改用 TreeSkill，把 pipeline 的 `trainer.class_name` 改为 `treeskill`，按其配置填写 `optimizer` 和 `router`；adapter 会按本次环境的注册名自动选择；详见[训练流程](../mindmemos_skill/training.md)。

## 4. 用自定义 Env 训练：完整可运行示例

[custom_env_demo.py](custom_env_demo.py) 是完整的单文件示例。先准备模型 YAML（填写 `model`、`api_key`、`base_url`），然后从仓库根目录运行：

```yaml
model: YOUR_MODEL
api_key: YOUR_API_KEY
base_url: https://gateway.example.com/v1
```

```bash
.venv/bin/python docs/skill/custom_env_demo.py --llm-config /path/to/model.yaml
```

Skill 数据保存在本地 SQLite；环境中的 Agent 会调用指定的模型服务。脚本依次完成以下操作：

1. 用 `@Environment.register("guide_lookup")` 注册自定义环境；`task_type`、`reward_data_type` 明确声明公开任务和私有答案的类型。入口文件导入此定义后，`create_service()` 才能通过注册名构造环境。
2. 环境的 `_run()` 把已组装 Skill 和公开任务交给 Agent，提取回答中的 `<answer>...</answer>`，再与私有的 `RewardData.data` 做精确比较。基类 `rollout()` 负责隔离 `audit/` 与 `workspace/`、记录输入和结果、处理恢复。
3. 用 `@Algorithm.register("guide_lookup_trainer")` 注册最小训练器。它调用自定义环境的 `rollout()` 收集训练反馈，再返回新的 Skill；数据库写入由通用 `training` pipeline 完成。
4. `SkillConfig` 配置鉴权、HOME、SQLite、仓库 pipeline、训练 pipeline；`TrainingRequest(algo="lookup", env={"class_name": "guide_lookup", "config": {...}}, init_skill_version_id=..., ...)` 调用训练，随后读取新版本和 `val`/`test` 汇总。

每次运行使用新的 `output/skill-guide/<随机 ID>/`，完成后会打印发布版本、版本数量，以及验证和测试汇总。训练器只是演示完整接口：它把训练样本中答错的词条加到 Skill。实际回答由模型生成，验证和测试得分取决于模型是否按 Skill 中的规则作答。

同一脚本还能用**自定义 Env + 内置 Skill-Forest** 运行；在同一份模型 YAML 中显式填写连接配置，再执行：

```bash
.venv/bin/python docs/skill/custom_env_demo.py --llm-config /path/to/model.yaml --forest-model YOUR_MODEL
```

这个入口把 `trainer.class_name` 切为 `skill_forest`，按环境注册名自动选择 adapter，自定义环境使用 `generic`，仍使用同一组 `DataSample` 和 `TrainingRequest`。脚本会调用实际模型服务；模型生成的候选和评分可能不同，因此验证/测试得分不保证为 1。接入自己的 Env 时可保留相同的接口边界；内置算法的训练结果应以实际运行日志和评估产物为准。
