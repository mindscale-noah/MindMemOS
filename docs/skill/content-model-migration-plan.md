# Skill 内容模型迁移计划

## 目标与范围

移除 `SkillBundle` 及对外的 `bundle` 概念。一个 Skill 版本的核心内容由 `name`、`description`、`content` 和 `resources` 表示：`content` 是 `SKILL.md` 的**全文**，可能包含 YAML frontmatter；`resources` 是其他相对路径文件的原始字节。系统不得为了补齐元数据改写 `content`，也不得在创建 snapshot 时拼接 `name` 或 `description`。

本计划针对尚未上线的独立 `mindmemos_skill` 服务、它的 SDK、训练与演化调用方。它不是已部署 `src/mindmemos/` 云服务的数据迁移方案；如果后续修改云服务的数据或 API 契约，需要按该服务的生产兼容要求另行设计无感迁移。

## 迁移前实现与迁移原因（历史基线）

- `SkillBundle.files` 是文本文件映射，要求有 `SKILL.md`，但仍接受额外文本文件；磁盘读取与 SDK 白名单实际上只读取 `SKILL.md`，其余目录文件由 `SkillResources` 保存。
- `SkillSnapshot.bundle` 存储序列化的文件记录；当前 `content_hash` 是这些记录的 SHA-256，`local_snapshot_hash` 是 bundle 与 resources 的组合哈希。因此旧 `content_hash` **不是**目标定义下的正文哈希。
- 版本表、注册与提交接口、SDK 请求、版本解析、导入导出、diff、训练与演化代码均引用 `bundle` 或旧哈希。现有测试还覆盖 bundle 内的第二个文本文件，迁移必须显式处理这一历史模型。
- 当前工作区的注册请求模型已经出现 `content`、`local_path` 和 `standard_format` 字段，但服务及 HTTP 路径仍有旧 bundle 流程；实施时以实际代码为准，逐层收敛，不能将请求模型的变化视为功能已完成。

## 目标契约

| 字段 | 定义 |
| --- | --- |
| `name` | 必填、非空的 Skill 展示名称；由标准格式正文解析，或由非标准格式调用方提供。 |
| `description` | 必填、非空的 Skill 描述；来源规则与 `name` 相同。 |
| `content` | `SKILL.md` 全文，保持原文；不把单独输入的元数据写回正文。 |
| `standard_format` | `true` 时要求正文开头有有效 frontmatter，且其中 `name`、`description` 均非空；`false` 时不要求正文包含这两个字段，但调用方必须提供非空 `name`、`description`。 |
| `resources` | 除根目录 `SKILL.md` 外的相对路径与原始字节；其文件名和内容均参与 `skill_hash`。 |
| `content_hash` | 只反映 `content`；相同正文得到相同哈希，不受名称、描述、resources 或运行配置影响。 |
| `skill_hash` | 反映 `content` 与完整 resources；其中任一文件路径或字节变化都会改变哈希，不受名称、描述或运行配置影响。 |

标准格式下若调用方也传入 `name` 或 `description`，应校验与 frontmatter 一致；不一致则拒绝。非标准格式下可保留原有 frontmatter 文本，但不从中隐式补齐必填元数据。注册和提交应使用同一套规则；提交时需明确 `name` 是版本快照字段还是同时更新 Skill 家族展示名，避免两者悄悄分叉。

## 注册与提交入口拆分

提交 Skill 内容的操作拆为结构化输入和本地文件输入两条明确的路径。`StructuredSkillInfo` 只承载 `content`、`resources`、`name`、`description`、`standard_format` 及所需提交元数据；`FileSkillInfo` 承载本地文件或目录路径及所需提交元数据。两者不使用 `source: Snapshot | path` 一类混合字段，也不靠路径是否存在来猜测输入类型。

| 操作 | 输入 | 可用入口 | 处理原则 |
| --- | --- | --- | --- |
| `register` | `StructuredSkillInfo` | Python、FastAPI | 直接验证传入正文和资源，注册新 Skill。 |
| `register_from_local` | `FileSkillInfo` | 仅 Python | 读取本地 `SKILL.md` 与其他资源，再复用 `register` 的校验和注册流程。 |
| `commit` | `StructuredSkillInfo`，另带目标 Skill、父版本与激活选项 | Python、FastAPI | 直接验证新正文和资源，提交新版本。 |
| `commit_from_local` | `FileSkillInfo`，另带目标 Skill、父版本与激活选项 | 仅 Python | 读取本地文件后复用 `commit` 的校验和提交流程。 |

FastAPI 只暴露结构化 `register` 和 `commit`，请求体不含本地路径字段；不增加 `register_from_local` 或 `commit_from_local` 路由。远程 SDK 同样只调用这两个结构化 HTTP 接口；若 SDK 提供本地文件便捷方法，应在调用端读文件并构造 `StructuredSkillInfo`，不得让服务器按客户端路径访问文件。Python 模式可直接调用两个 `*_from_local` 方法，路径解析和读取发生在该 Python 进程内。

本地文件入口只接受标准 Skill 格式：从 frontmatter 解析并验证 `name`、`description`，同时读取旁边的 resources。结构化入口按 `standard_format` 分支执行本计划的目标契约。两个入口的版本编号、冲突检测、资源继承、返回值及错误语义应一致；本地入口只负责文件读取与构造结构化输入，不另写一套持久化逻辑。文件读取失败须在创建版本前报错，避免注册了不完整的 Skill。

当前工作区已有 `FileSkillInfo`、`StructuredSkillInfo` 的草稿定义，但注册和提交路径尚未统一使用它们。实施时应把共同的 Skill 内容字段放在这两个类型中，操作专属字段留给对应请求模型；核对 `commit_source`、版本标签和资源继承等现有字段，避免两条入口的语义不同。

## 查询数量参数

`api/typing.py` 中所有名为 `limit` 的入参统一改为可空，默认 `None`，并移除固定最大值约束。`None` 表示不限制返回数量；显式提供整数时仍校验该操作所需的非负或正数下界，不设置上界。当前需覆盖 `ListRequest`、`HistoryRequest`、`RecommendRequest` 和 `SkillPageRequest`，实施时再次搜索该文件，避免遗漏新增字段。

同步检查服务、仓储、推荐管线和分页逻辑，确保 `None` 不被转回旧默认值，也不会传给要求整数的切片、SQL `LIMIT` 或算法参数。保留各操作已有的排序、游标和偏移语义；无上限请求按该语义返回全部匹配项。FastAPI 与 SDK 若镜像这些请求模型，应同步接受省略或 `null`，不额外施加旧上限。

## 生效版本命名与唯一性

将 Skill 簇中“当前生效版本”的术语统一为 **active**：一个 Skill 簇可包含多个版本，但任一时刻只能有一个 active 版本。注册时首个版本成为 active；提交时仅在选择激活时替换 active 指向，未激活的提交不影响原 active 版本；显式切换时原版本失去 active 身份、新版本成为 active。切换须在同一事务中完成，并保证目标版本属于该 Skill 簇；默认读取、导出及省略版本 ID 的查询均选择 active 版本。

把 `switch_skill` 方法改名为 `active_skill`，其请求、返回模型和对外字段也统一采用 active 术语。例如 `SwitchRequest`、`current_version_id`、`current_skill_version_id`、`is_current` 分别改为表达激活操作、`active_version_id`、`active_skill_version_id`、`is_active` 的名称；具体请求类名称以最终接口风格统一确定。`active_skill` 表达“激活指定版本”这一操作，不能被解释成创建新版本。Python 服务、仓储、FastAPI 路由及响应、SDK、UI 和文档需要同步改名，清理将 `"current"` 当作版本别名的 Skill 路径。

持久化层优先用 Skill 簇上的单个 `active_version_id` 指针表达唯一性，指针须指向该簇实际存在的版本；不要在每个版本上维护可能相互矛盾的独立布尔值。并发提交与激活操作应使用现有的版本冲突检测，避免并发请求使 active 指针回退或指向错误版本。只迁移 **Skill 版本状态** 的 `current` 术语；其他业务中表示普通“当前值”的变量或图谱当前快照不在此范围。

## 哈希规范与测试向量

本次重构选择**保留输入正文的原始换行**；本地文件入口按 UTF-8 解码原始字节，不使用自动换行转换。哈希计算对象与保存、读取、导出的正文完全一致。`content_hash` 为原文 UTF-8 字节的 SHA-256。

`skill_hash` 的规范输入是 UTF-8 编码的紧凑 JSON：顶层字段为 `content`、`resources`、`schema`，键按字典序排列，`ensure_ascii=False`，分隔符为 `,` 和 `:`。`schema` 固定为 `mindmemos-skill-v1`；`resources` 是按规范化路径排序的数组，每项含 `base64`（原始字节的标准 Base64）和 `path`。空资源的规范文本为 `{"content":"","resources":[],"schema":"mindmemos-skill-v1"}`。名称、描述及运行配置均不参与两个哈希。

首批跨语言测试向量如下；所有十六进制摘要均为小写：

| `content` | resources | `content_hash` | `skill_hash` |
| --- | --- | --- | --- |
| 空字符串 | 空 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `d309b0aec2471247d5c7fae40a6c8e8c762223603200600b6c34a56376dde16f` |
| `---\r\nname: Café\r\n---\r\n` | 空 | `24488d2b8fbd36705f4702c8048ed8f82d29b335e67440ef42033e297abb2dd3` | `153eeabb772ce4d15cebefe91451ad36f524d8fefec8d352af318c0f037cf770` |
| `你好\n` | `assets/logo.bin` 为 `00 ff`，`references/note.txt` 为 `Café` 的 UTF-8 字节 | `4e0826721642ed8e3a27e7147538ac7b7013a08fe5ae343a8ef09749b7e5790f` | `fb588d7943833f61a1388f6df978b744cdf877a89c339e5fd8df8a6654eb172b` |

Skill 版本快照保存自己的 `name`、`description`；家族展示名通过元信息操作显式维护。提交版本不会暗中改写家族展示名，因此详情必须同时呈现家族与版本的名称。对版本而言，单个家族使用 `active_version_id` 指针表达唯一生效版本。

哈希只识别源文件身份，不代替正文和资源字节的完整性校验。服务端与 SDK 必须使用相同测试向量；资源路径必须为规范化相对路径，不允许覆盖根目录 `SKILL.md`，也不允许文件与目录路径重叠。

## 实施顺序

1. **冻结契约。** 确定上面的哈希编码、换行规则、`name` 在家族与版本中的归属，以及提交时是否允许改变名称。把这些决定写进 API/SDK 文档和测试向量。
2. **核心模型。** 将 Skill、snapshot、持久化版本的 `bundle` 改为 `content`；删除 `SkillBundle` 文件映射及相关序列化/反序列化。保留 resources 的路径和字节校验。snapshot 保存 `content_hash` 与 `skill_hash`，恢复时验证两者。
3. **输入与入口。** 完成 `StructuredSkillInfo` 和 `FileSkillInfo`，拆分 `register` / `register_from_local`、`commit` / `commit_from_local`。对结构化输入实现 `standard_format` 两种校验；本地输入读取 `SKILL.md` 原文及其他资源并强制标准格式。snapshot 不修改正文。
4. **版本与查询。** 更新数据库字段、索引、版本去重/解析逻辑、历史与 diff；将 Skill 簇的 `current_version_id` 改为唯一的 `active_version_id`，同步默认版本解析和原子激活。`content_hash` 查询只定位相同正文；需要唯一确定完整 Skill 时使用 `skill_hash` 或版本 ID，并明确同一正文搭配不同 resources 的歧义处理。
5. **接口与 SDK。** FastAPI 只接结构化 `register`、`commit`；Python 模式另有两个 `*_from_local` 方法。注册、提交、详情、导出及 SDK 模型改用 `content`；移除上传/下载 bundle JSON 和 SDK 的 bundle 哈希工具。把 `switch_skill` 改为 `active_skill`，同步所有 Skill 版本的 current 字段与路由语义。`api/typing.py` 的所有 `limit` 默认 `None` 且不设上界，并同步下游与 SDK。对外响应同时返回两个新哈希，更新使用文档与示例。
6. **训练与演化。** 把各处 `bundle.files["SKILL.md"]` 改为 `content`；候选 Skill 仅替换正文时保留 resources，确需改资源时使用显式操作。更新静态 Skill、TreeSkill、Skill-Forest、环境适配器和示例。
7. **清理与验证。** 搜索并删除旧 bundle 字段、旧哈希名称和多文件正文假设；完成相关模块测试及端到端注册、提交、读取、导出、训练验证。

## 已有本地数据与兼容边界

独立 Skill 服务尚未上线，按项目约定不为其本地数据库编写迁移、旧表兼容或滚动部署桥接逻辑。实施时先识别开发环境中需保留的实验数据并另行备份；如果旧数据库结构或旧哈希与新代码冲突，只清理**该 Skill 服务的本地数据库**并重启，绝不触碰已部署云服务数据。

旧 bundle 中除 `SKILL.md` 外的文本文件不应无声丢弃：导出或备份时将它们作为资源保留；新模型中再次导入时要求资源路径不与其他资源冲突。旧 `content_hash` 和 `local_snapshot_hash` 与新两种哈希的语义不同，不能原值沿用。旧 SDK/HTTP 请求形状也不能在没有明确版本约定时被误认为新请求。

## 验收条件

- 标准格式缺少或写错 frontmatter、名称或描述时拒绝；非标准格式缺少独立名称或描述时拒绝。
- 非标准正文原样经过注册、snapshot、持久化、读取和导出；不出现自动拼接的 frontmatter。
- 只改名称或描述，两种哈希均不变；只改正文，两种哈希均变化；只改资源路径或字节，仅 `skill_hash` 变化。
- 相同输入在服务端和 SDK 产生相同哈希；snapshot 被篡改时恢复失败。
- 额外文件按 resources 保留；旧的多文件正文不会被静默丢弃。
- 注册、提交、版本解析、历史、diff、导入导出和至少一条训练/演化路径使用新契约；项目内不再有面向业务的 bundle 字段。
- Python 模式分别验证四个入口；FastAPI/OpenAPI 仅包含结构化注册和提交，HTTP 请求中的本地路径字段被拒绝。
- 本地文件入口读取与结构化入口等价的内容时得到相同正文、resources 和两个哈希；文件缺失或格式无效时不创建版本。
- `api/typing.py` 中所有 `limit` 省略或为 `None` 时返回全部匹配结果；显式整数不受固定上限限制，原有排序、游标和偏移仍生效。
- 每个已注册 Skill 簇始终恰有一个 active 版本；未激活提交不改变它，激活已有版本后默认读取指向新版本，并发激活不会产生两个 active 版本。
- Python、FastAPI、SDK 和 UI 的 Skill 版本接口与字段使用 active 术语；不再暴露 `switch_skill`、`current_version_id` 或 `is_current` 等旧名称。
