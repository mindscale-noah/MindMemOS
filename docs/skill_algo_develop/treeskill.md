# TreeSkill 核心接入

TreeSkill 由两个彼此独立但共享 Markdown 树合同的组件组成：

- **TreeSkill Evolution**：`trace2skill` 家族算法，从轨迹中提取证据，将证据定位到现有标题节点，并按节点自底向上融合。
- **TreeSkill Routing**：任务级 Skill Runtime，在每次 Agent 执行前调用 router，选择需要的 Markdown 子树，再复用 Agent 原有注入方式。

本接入只增加算法和 Runtime，不新增第三个 family runner，也不修改既有 Agent 注入模式。

## 1. Evolution 数据流

```text
base Skill + offline/collected trajectories
  -> normalize and select trajectory evidence
  -> analyze each trajectory
  -> parse SKILL.md headings into a tree
  -> locate atomic evidence to existing nodes
  -> group evidence by target node
  -> fuse target nodes bottom-up
  -> return SkillCandidate
```

实现位于：

```text
src/mindmemos_skill/mindmemos_skill/algos/trace2skill/treeskill/
```

算法注册名为 `treeskill`，实现现有 `optimize(Trace2SkillInput)` capability。它不直接写数据库、版本库或远端服务；正式版本仍由 `SkillAlgorithmOrchestrator` 和 commit policy 创建。

## 2. 树元数据的存储边界

TreeSkill 产生的候选使用：

```text
runtime_type: treeskill
runtime_schema_version: 1
runtime_metadata:
  enabled: true
  schema_version: 1
  router: llm_subtree_v1
  skill_content_hash: ...
  root_ids: [...]
  nodes:
    - id: ...
      level: ...
      heading: ...
      parent_id: ...
      child_ids: [...]
      ordinal: ...
      local_content_hash: ...
```

`runtime_metadata` 是执行所需的版本化树合同。它只保存结构、稳定节点 ID 和内容 hash；节点正文的唯一来源仍是该版本的 `SKILL.md`。Runtime 在每次任务开始时同时校验 metadata 与 `SKILL.md`，避免路由一个已经失配的树。

算法运行 ID、Prompt 版本和输入轨迹 ID 等审计信息单独保存在 `metadata.treeskill_evolution`，不混入执行合同。

## 3. Runtime 回调和正常注入

当前基础设施已经把“动态 Skill 组装”和“Agent 注入方式”分为两层。TreeSkill 使用新的 `runtime_type`，不增加新的 `SkillInjectionMode`：

```text
query/task
  -> SkillRuntimeCoordinator
  -> TreeSkillRuntime.on_task()
  -> TreeSkillRouter.route()
  -> validate selected node IDs
  -> render selected subtrees in original Markdown order
  -> projected_skills()
  -> existing Agent SYSTEM_PROMPT / TOOL / FILESYSTEM injection
```

这样，TreeSkill 不需要改 ReAct、Claude 或 OpenClaw 的既有执行循环。Runtime trace 会记录选择节点、内容节点、祖先节点、完整/路由字符数和 context saving ratio。

## 4. Application 接入

TreeSkill Runtime 需要一个实现 `TreeSkillRouteResolver` 的 router。使用内置 LLM router 时，在运行 TreeSkill 算法或加载 TreeSkill 版本前，将同一个 runtime 注册到 Application：

```python
from mindmemos_skill.algos.trace2skill.treeskill import TreeSkillRouter
from mindmemos_skill.skill_runtime import TreeSkillRuntime

router = TreeSkillRouter(chat_model=chat_model)
application.register_skill_runtime(TreeSkillRuntime(router=router))
```

`register_skill_runtime()` 会同时把 Runtime 注册到持久化校验和所有 Agent。之后仍通过标准入口调用：

```python
result = await application.run_trace2skill(request)
```

如果只直接调用 `TreeSkill.optimize()` 做纯算法测试，不会自动持久化候选，也不要求注册 Runtime。若使用 `persist` 或 `persist_and_push`，必须先注册 Runtime，否则版本校验会明确拒绝未知的 `treeskill` runtime type。

## 5. 配置片段

TreeSkill 复用 `runtime.algorithms`，不需要算法专属 runner：

```yaml
runtime:
  algorithms:
    treeskill_core:
      type: treeskill
      model_roles:
        chat: optimizer_model
      config:
        algorithm_version: "1"
        prompt_version: treeskill-v1
        annotation_mode: required
        min_trajectories: 1
        max_trajectories: 1000
```

调用编排层时，`algorithm_name` 使用配置实例名 `treeskill_core`，不是组件类型名 `treeskill`。

## 6. 当前范围

本次仅接入通用核心：Markdown 树、轨迹分析接口、证据定位、自底向上节点融合、元数据合同和任务级路由 Runtime。它没有复制旧实验分支中的 SpreadsheetBench 数据、评测脚本、Slurm 配置、模型部署逻辑或环境专用 recalculation 行为。若需要统一实验入口，应按 `experiment_runner.md` 单独增加薄 adapter 和 YAML，而不是把 benchmark 编排写进算法包。
