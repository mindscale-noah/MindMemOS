# 图管理与 Skill 推荐

`graph_manage` 提供完整的 Skill/Scene 图构建、批量成员增删、图排序和纯向量推荐。Service → Pipeline → `skill_manage` 算法 → 异步模型与图存储组成调用链。Pipeline 固定授权项目和精确 Skill 版本，读取或提取 Scene；算法保留原实现的文档语义解析、语义补全、联合候选连边、图排序及提示词。

## 配置与启动条件

在 `config/mindmemos_skill/dev.yaml` 的 `service.manage_pipeline.config.algorithm` 中填写 Embedding 配置，并通过 `chat` 引用 `service.agents` 中的 `skill_manage_*` 命名配置；也可直接填写内联 Chat 配置。`dev.example.yaml` 提供相同字段结构；其中公开的模型名称和端点是配置示例，复现实验时须使用原实验的模型、端点、维度及生成参数。凭据不能写入共享示例文件。

算法构造只接收 `config` 和 Pipeline 提供的 `store`、`provided`，不再接收 `chat_model` 或 `embedding_model`。
Chat 由 `Agent.from_config(config.chat)` 创建，命名引用和内联配置都遵守 `openai_chat` 白名单。
Embedding 由 `EmbeddingModel.from_config(config.embedding)` 创建；当前配置的 `class_name` 为
`openai_embedding`，省略时使用该默认值。两类创建入口复用统一注册机制，每次创建独立实例，构造时不发起模型请求。
离线测试替换对应的统一创建入口或底层调用。

`algorithm: null` 允许查询、预览和删除已存储图；构建和推荐需要明确的算法模型配置。缓存缺失或强制刷新时需要 Chat。纯向量推荐只调用 Embedding，不进行查询改写、词法融合、PPR 或字段重排。图管理调用是原生异步业务调用，不提供训练任务的后台恢复协议。

`build_scene=false` 保留原算法只构建 Skill 节点的选项；此时普通构建跳过 Scene 提取，显式 force 仍可刷新版本缓存，但不会把 Scene 加入图中。算法默认 `mode=separate`，保留 `skill_only`、`unified_ppr`、`separate`、`fuse`，以及独立的候选预算、种子预算、Scene PPR、依赖边过滤和联合连边选项。图排序的默认值、提示词和数值计算由原算法夹具及冻结基准验证。Chat 的结构修复次数、反馈方式和 SDK/外层重试参数均可配置。

## 接口与使用

以下 HTTP 路径均为 POST。项目身份由服务的认证上下文确定，不由 HTTP 请求选择。

| Service 接口 | HTTP 路径 | 功能 |
| --- | --- | --- |
| organize_skills | /v1/skills/algos/organize | 提交完整最终成员集合，构建或复用快照 |
| recommend_skills | /v1/skills/algos/recommend | 查询固定快照的 graph/vector 推荐及核心评分 |
| add_graph_members | /v1/skills/graphs/members/add | 批量添加未在图内的 Skill 家族 |
| remove_graph_members | /v1/skills/graphs/members/remove | 按家族批量移除成员，保留原 Skill 和历史快照 |
| preview_graph | /v1/skills/graphs/preview | 只读预览完整拓扑 |
| list_graph | /v1/skills/graphs/list | 列出项目内未删除图 |
| list_graph_snapshots | /v1/skills/graphs/snapshots | 列出图的不可变快照 |
| get_graph_snapshot | /v1/skills/graphs/snapshot | 读取指定快照的 Skill/Scene 节点和边 |
| resolve_graph | /v1/skills/graph-resolve | 解析 ID、唯一前缀、alias 或快照引用 |
| delete_graph | /v1/skills/graphs/delete | 软删除整图 |

同步和异步 SDK 提供 `add_graph_members`、`remove_graph_members`、`preview_graph`，保留原有 `organize`、`recommend`、`list_graphs`、`graph_snapshots`、`graph_snapshot` 和 `delete_graph` 方法名。

```sh
mindmemos skill graph publish --alias example --skill skill-ref --force-extract-scene
mindmemos skill graph add example --skill another-skill
mindmemos skill graph remove example --skill another-skill
mindmemos skill graph preview example
mindmemos skill graph recommend "Read CSV rows" --by example --strategy vector --include-scenes
```

建图接受 `skill_refs`、可选 `graph_id`、`alias`、`commit_message` 及 `force_extract_scene=false`。修改已有图不能同时改 alias。成员操作使用 `graph_ref` 和非空 `skill_refs`；add 可刷新新增成员的 Scene。重复引用、不同引用指向同一家族、add 已有成员或 remove 不存在成员，均拒绝整批操作。成员版本替换通过 organize 提交完整最终集合。可移除最后一个成员并发布空图。历史快照引用不能作为成员修改目标。

## Scene 缓存与快照

完整 Scene、逻辑步骤、条件转移及归属关系只在精确版本的 `metadata.scene_data` 中缓存。`schema_version` 必须是当前支持的严格整数，字符串和布尔值不合法。结构或语义校验失败、未知版本、缺失缓存都会重新提取；`force_extract_scene=true` 即使缓存有效也先提取。

新提取结果经相同模型校验后，仅合并更新 `scene_data`，保留其他 metadata。提取失败不覆盖旧缓存；权限和数据库失败直接报错。合法缓存可独立提交，后续建图失败不改变图头。普通读取、推荐、preview 和历史查询不提取 Scene。

成员操作捕获已有成员的快照节点和归属/条件流程边，add 只获取新增成员的 Scene。刷新版本缓存不会改变历史拓扑。图快照只持久化用于构建和查询的语义节点、关系与必要的构建身份，不另存完整 Scene 缓存或逻辑步骤副本。启用语义补全时，快照保留补全前的 Skill 节点字段；成员变更恢复这些字段，再按原算法补全，避免把上次的补全结果当成原始字段。

构建输入包含固定的精确版本、实际 Scene 内容身份、构建配置和模型/模板身份。同一图的相同有效输入复用 ready 快照；成员顺序、查询选项、提交说明不改变构建身份。强制提取后结果相同，也允许复用。成员版本、Scene 内容或有效构建配置改变时重新构建。

模型工作在事务外执行，完整结果在短事务中原子发布。数据库检查构建前捕获的图头；冲突报错，不覆盖其他成员操作。快照 `request_hash` 的同图唯一约束防止重复发布相同构建，但不承诺所有并发请求都只调用一次模型。不新增旧发布 revision、request_id、lease 或业务请求去重协议。

## 推荐评分与拓扑

Service 的 `task` 保持字符串，HTTP/SDK 继续使用 `query`。`strategy` 默认 `graph`，`include_scenes` 默认 `false`。`graph + include_scenes=true` 被拒绝；图排序是否使用 Scene 由算法 mode 等配置决定。空白查询被拒绝，不接受 diagnostics。

`vector + include_scenes=false` 只检索 Skill 节点；`true` 同时检索 Skill/Scene，按所属 Skill 归并，选择最相似节点。查询向量仅来自去除首尾空白的 instruction。结果包含实际快照 `graph_version_id`、策略、评分类型、有序 `skills` 与一一对应的 `scores`。返回接口仅映射最多 `top_k` 条结果；原算法内部的向量分页流程保留。

| 策略 | 分数字段 | 含义 |
| --- | --- | --- |
| graph | ppr_score | Skill 节点的 PPR 质量 |
| graph | scene_boost | 原算法计算的子 Scene 贡献 |
| graph | graph_score | ppr_score 与 scene_boost 的总和 |
| graph | rerank_score | 图候选截断后的字段重排分数，用于最终排序 |
| vector | similarity | 命中节点的余弦相似度 |
| vector | node_id / node_type | 所属 Skill 的最佳命中节点及 Skill/Scene 类型 |

这些图分数不能解释成统一概率。空图也返回实际快照及所选策略，Skill 和分数列表为空。

preview 与指定快照详情返回相同拓扑：Skill/Scene 类型、所属家族和精确版本、owner_node_id、核心语义，以及边类型、权重、条件和证据。结果不包含 embedding 数组或完整逻辑步骤。UI 区分归属、条件流程和依赖边；Scene 详情可以打开所属 Skill 的精确版本。UI 默认按图当前指针选择快照，不以创建时间判断当前发布内容；复用较早的历史快照时也能正确显示。

## 验证边界

`src/mindmemos_skill/tests/fixtures/scene_graph_parity.json` 冻结原实现的提示词、参数默认值和四种排序模式分数；相关测试同时覆盖原数值验收夹具、候选预算、缓存、成员操作、构建复用、历史拓扑和发布冲突。SDK/CLI/UI 契约通过各模块测试验证。

离线模型夹具可验证迁移后的调用载荷、校验、评分与流程。使用真实模型复现实验还需要原模型配置、相同数据和生成设置；离线通过不等于线上效果已复现。
