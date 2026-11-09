# MindMemOS Skill

**把 Skill 当作可管理、可运行、可改进的版本化资产。**

[English](../../src/mindmemos_skill/README.md)

MindMemOS Skill 是 MindMemOS 的 Python Skill 服务。它把 Skill 文件及其历史版本保存在仓库中，让 Agent 可以按任务使用指定版本，并将执行轨迹、评估结果和后续改进关联起来。你可以先在本地用 SQLite 管理 Skill，再按需配置训练、演化、图组织和 HTTP 接口。

默认安装包含 FastAPI 和所有内置 env 的 Python 依赖。仅保留 `postgres`、`kafka` 两个可选分组；需要两者时安装 `mindmemos-skill[postgres,kafka]`。SpreadsheetBench 仍通过 uv 自动准备独立执行环境；LibreOffice、数据集、外部评分器源码及 Agent 可执行程序需要另行准备。

ALFWorld 依赖 TextWorld，部分平台需要从源码构建。在 macOS ARM64 上，构建需要允许挂载和卸载 Inform7 安装镜像。本次受限构建在挂载镜像时失败；允许该操作后，临时 Python 3.12 环境中的 ALFWorld 安装及一次文本游戏 reset/step 均通过。

已在 macOS ARM64、Python 3.13.13 下使用全新 uv、空虚拟环境和独立缓存完成默认 wheel 安装验证。CLI、依赖导入、文档生成、SQLite/向量扩展、两种 SpreadsheetBench 管理环境和 ALFWorld worker 的 reset/step 均通过。验证使用本机已有系统构建工具和基准数据；未验证其他平台或完整模型训练。

Python 3.13 下安装也通过。ALFWorld simulator worker 仅在 Python 3.13 及以上、TextWorld 1.7.0 时应用兼容补丁：为表达式求值显式传入变量字典，替代对 `locals().update()` 的依赖。相同文本游戏在 Python 3.12 和打补丁的 Python 3.13 下均通过 reset/step，观察、可用动作、得分及结束标志完全一致。

## 快速开始

需要 Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)。以下命令均从**仓库根目录**运行。先安装 workspace，并创建一个最小 Skill：

```bash
uv sync --all-packages
cp config/mindmemos_skill/dev.example.yaml config/mindmemos_skill/dev.yaml
mkdir -p output/demo-skill
cat > output/demo-skill/SKILL.md <<'SKILL'
---
name: demo-skill
description: Answer questions clearly.
---

# Demo Skill

Answer clearly and check the final response.
SKILL
```

**1. 注册。** `register` 从目录读取当前内容，创建 Skill 身份和第一个不可变版本。输出会显示 Skill ID 和首次版本 ID。

```bash
.venv/bin/mindmemos-skill --config config/mindmemos_skill/dev.yaml register output/demo-skill --alias demo
```

**2. 列表。** 同一 YAML 中的 HOME 和数据库配置指向同一个本地仓库；列表会显示 Skill ID、别名、名称和当前版本 ID。

```bash
.venv/bin/mindmemos-skill --config config/mindmemos_skill/dev.yaml list
```

**3. 发布新版本。** 修改源文件后，用 `publish` 保存新版本并将它设为当前版本。输出会显示新版本 ID。

```bash
printf '\nKeep the final answer short.\n' >> output/demo-skill/SKILL.md
.venv/bin/mindmemos-skill --config config/mindmemos_skill/dev.yaml publish demo output/demo-skill --message "Shorter answers"
```

**4. 版本对比。** 将注册与发布时输出的两个版本 ID 分别填入 `--from` 和 `--to`：

```bash
.venv/bin/mindmemos-skill --config config/mindmemos_skill/dev.yaml diff demo --from v_FIRST --to v_SECOND
```

这里的“发布”是写入**本地版本仓库**，不会自动推送到云端，也不会改写源目录。CLI 必须指定完整 YAML；直接本地 CLI 只接受无鉴权模式。YAML 必填 `service.home` 和 `fastapi.auth.default_project_id`。服务创建缺失的 SQLite 存储，不自动生成项目 ID。更多 Python 调用和导出示例见[本地使用指南](user-guide.md)。

安装包名是 `mindmemos-skill`，Python 导入名是 `mindmemos_skill`。仓库已将该包列入 uv workspace；如果只需单独安装此包，也可以在仓库根目录执行 `uv pip install -e ./src/mindmemos_skill`。

## 你可以做什么

| 能力 | 实际用途 |
| --- | --- |
| 管理 Skill 版本 | 从目录或快照注册 Skill，提交、查看、比较、切换和导出版本；已注册版本不会随源目录的修改自动变化。 |
| 运行任务并记录证据 | 环境用公开任务和 Skill 执行 Agent，在独立目录保存轨迹、评分与用量；私有评分数据留在审计侧。 |
| 训练与演化 | 配置 TreeSkill、Skill-Forest 等训练流程，或基于已保存轨迹生成 Skill 改进版本；模型相关流程需要可用的模型服务。 |
| 图管理 | 已发布图的列表、完整拓扑、历史快照与软删除可用；建图、推荐和成员增删接口已改造，等待完整算法接入。 |
| 接入服务 | Python Service 和 FastAPI 入口共用业务实现；本地 SQLite 可直接使用，也提供可选的 PostgreSQL 后端。 |

一次典型流程是：**注册源 Skill → 用固定版本执行任务 → 保存轨迹和评分 → 训练或演化出新版本 → 检查并选择当前版本**。仓库版本、Agent 实际加载的版本和运行证据可以分别追溯；训练和演化不会由默认本地配置自动触发。

## 从 Skill 到运行结果

Skill 的主要可优化内容是 `SKILL.md`。`Skill` 还保留 `references/`、`scripts/`、`assets/` 等辅助文件，导出时保持目录内容。静态 Skill 直接返回相同指令；动态 Skill 可根据公开任务组装本次使用的指令。`SkillSet` 负责选择成员并组装，环境只接收组装后的 Skill。

环境以 `Task` 提供公开任务，以 `RewardData` 保存答案、测试用例等私有评分输入。一次 `rollout()` 返回输出、执行轨迹、`Reward` 和 `Usage`，并在工作目录中分别保存 `workspace/` 与 `audit/`。相同输入的已完成运行可按环境能力恢复；更换任务、Skill 或影响运行的配置时应使用新目录。Agent 后端仍须自行限制文件访问范围，不能仅靠工作目录划分保护私有数据。

内置环境包括 [SearchQA](../../src/mindmemos_skill/mindmemos_skill/envs/builtin_envs/searchqa/README.md)、[LiveMath](../../src/mindmemos_skill/mindmemos_skill/envs/builtin_envs/livemath/README.md)、[OfficeQA](../../src/mindmemos_skill/mindmemos_skill/envs/builtin_envs/officeqa/README.md) 和 [SpreadsheetBench](../../src/mindmemos_skill/mindmemos_skill/envs/builtin_envs/spreadsheetbench/README.md)。如何接入自己的任务、Agent 和评分器，可直接看[自定义环境示例](custom_env_demo.py)。

## 配置训练与演化

服务配置示例位于 `config/mindmemos_skill/dev.example.yaml`。算法参数由算法目录内的配置类定义，服务加载配置时会校验所选实现的字段和取值。仓库不维护生成的服务配置 Schema；可选的编辑器提示及配置和运行示例见[训练流程](../mindmemos_skill/training.md)。

算法根据已校验的 Agent 配置，通过统一的 `Agent.from_config()` 创建实例。命名引用和内联配置都遵守字段白名单；TreeSkill 和 Skill-Forest 不再接收 `agent_factory` 构造参数。

训练请求通过 `algo` 选择服务中的训练算法，通过 `env: {class_name, config}` 提供本次任务的执行及评分环境；环境不属于 Service 启动配置。请求还需提供基础版本、任务样本、`train/val/test` 划分和独立运行目录。`run_training()` 等待训练及评估完成；配置任务管理器后可以通过后台任务入口提交。训练结果是新版本，评估数据与运行产物保存在请求指定的目录中。

演化流程从已保存的执行轨迹中选取证据，对一个确定的 Skill 版本生成改进。它依赖已配置的演化 pipeline 和满足条件的轨迹，不会仅因注册了 Skill 就自动发生。示例配置 [`dev.example.yaml`](../../config/mindmemos_skill/dev.example.yaml) 展示了训练、演化、图管理、任务管理与可观测性组件；其中的模型名称、凭据和并发参数需要按实际服务调整。

从最小配置到一次完整调用，见[本地使用指南](user-guide.md)。算法和数据划分细节分别见[训练流程](../mindmemos_skill/training.md)、[TreeSkill](../mindmemos_skill/treeskill.md)、[Skill-Forest](../mindmemos_skill/skill_forest.md) 与 [Skill Manage](../mindmemos_skill/skill_manage.md)。

## 可选：启动 HTTP 服务

FastAPI 已包含在默认安装中。Python 集成无需启动 HTTP 服务；需要通过接口访问时，在仓库根目录执行：

```bash
uv pip install -e './src/mindmemos_skill'
make dev SERVICE=skill
```

默认使用本地示例配置，在 `127.0.0.1:8001` 监听。启动后可打开 <http://127.0.0.1:8001/docs>，或检查：

```bash
curl http://127.0.0.1:8001/health
curl -X POST http://127.0.0.1:8001/v1/skills/list \
  -H 'Content-Type: application/json' -d '{}'
```

`/health` 仅表示应用可访问，不代表模型服务或所有业务流程已经就绪。业务路由使用 `/v1/skills/*`；本地示例配置的 `none` 鉴权只允许监听回环地址。部署时还可选择本地 API key 或网关 JWT，由已验证的身份确定项目范围；请求体中的 `project_id` 不能替代鉴权。完整请求、路由和鉴权约定见[接口文档](../mindmemos_skill/target_architecture_and_api.md)。

## 文档入口

| 需要了解 | 阅读 |
| --- | --- |
| 本地创建服务、注册版本、运行训练、自定义环境 | [Skill 本地使用指南](user-guide.md) |
| Python 仓库方法与版本操作 | [中心仓 API](../mindmemos_skill/central_repository_api.md) |
| HTTP 接口与部署边界 | [目标架构与接口](../mindmemos_skill/target_architecture_and_api.md) |
| 轨迹、运行记录和可观测性 | [可观测性说明](../mindmemos_skill/observability.md) |
| 数据库与存储后端 | [基础设施说明](../mindmemos_skill/database_infrastructure_ZH.md) |

## 开发与验证

在仓库根目录运行 Skill 包测试：

```bash
.venv/bin/python -m pytest src/mindmemos_skill/tests -q
```

示例配置包含真实模型调用所需的训练和演化组件；只做本地仓库操作时无需模型凭据。HTTP 入口、数据库与任务管理器的行为由所选配置决定。源码与包依赖位于 [`src/mindmemos_skill`](../../src/mindmemos_skill)，配置示例位于 [`config/mindmemos_skill`](../../config/mindmemos_skill/dev.example.yaml)。
