# mindmemos_ui 设计提案

状态：设计记录，2026-09-20。首版模块已实现，实际启动方式与功能边界见 [使用说明](usage.md)。本文保留设计目标；早期交互稿为演示数据，正式模块只读取 SDK 与已连接的图谱存储。

文案约定：不添加宣传式副标题，例如“查看、比较，并将演进版本带回你的工作流”。页面保留功能名称、字段标签和必要的操作状态说明。

## 1. 产品定位

一个本地启动、浏览器使用的轻量 Skill 工作台，负责 SDK 配置、Skill 生命周期管理和 skill_manage 组织图谱查看。首页直接进入 Skills，不增加统计仪表盘。一级导航为 Skills、管理图谱、配置；内容、线性版本历史、对比是 Skill 详情的视图。

默认管理 UI 服务所在机器的目录。浏览器与服务分离时，目录指向服务端机器，不承诺直接写入浏览器用户电脑；首版以本机使用为基线。

## 2. 视觉规范

参考 https://mindmemos.cn/#/api-docs 的品牌色。当前页面 logo 实测背景 rgb(237,41,57)，即 #ED2939。保留红色作为品牌及主要操作色，背景改为浅色。

- 页面 #F7F8FA，内容 #FFFFFF，主文字 #20232B，次文字 #68707E，分隔线 #E8EBEF。
- 品牌 #ED2939；按钮用略深 #D92133 提升白字对比；选中背景 #FFF0F2。
- 绿色仅用于成功或 published 语义；删除采用文字动作和明确确认，不与品牌主按钮混用。
- 细边框、低阴影、8–10px 圆角、14px 正文、紧凑工具条、宽松内容区。
- 桌面窄侧栏 + 主区，详情按需出现；小屏导航改为顶部，图谱退为纵向节点列表。
- 不采用大面积渐变、暗色导航、指标卡堆叠。状态同时给文字，避免仅用颜色表达。

## 3. 页面和交互

### Skills 列表

搜索名称/别名；区分“本地管理”和“云端”，按 cloud_skill_id 关联而不是同名合并。表格展示名称、本地版本、发布版本、同步/本地修改状态。云端列表的 latest_version 单独显示，不把它等同 published_head。首次没有配置时，引导完成配置；没有 Skill 时提供注册入口。

选择 Skill 后进入详情。标题区保留名称、来源、本地目录和导出按钮；标签页为内容、版本历史、对比。可从指定版本查看其在管理图谱中的位置。较长 ID 默认缩写并可复制。刷新失败保留已有数据并显示上次同步时间。

### 配置

准确映射 SDKConfig，不新增另一套 SDK 配置格式：

| 分组 | 字段 |
| --- | --- |
| 连接 | base_url、auth.api_key、defaults.user_id |
| 存储 | storage.skill_cache_dir、storage.skill_backup_dir |
| 网络（折叠） | network.timeout_seconds、network.max_retries |

顶部只读显示解析后的 SDK home、来源及 settings.json 路径。遵循显式 config_dir > MINDMEMOS_HOME > MINDMEMOS_CONFIG_DIR > ~/.mindmemos。首版在启动参数中选择 home，不设计 SDK 尚未支持的多 profile 管理。

保留未编辑字段（包括 skills 和 metadata），通过 ConfigManager 保存。API Key 只返回“已设置”和脱敏后缀；修改以新值替换，空白默认不变，清除为独立动作。表单分“保存”和“测试连接”，后者发起只读请求，不能以网络连通推断所有写权限。未保存修改离开页面时提醒。

保存后重建本 UI 所持客户端；其他已运行程序需要重新加载配置。显示实际地址，不把 SDK 的 example.com 默认值当成可用服务。

### Skill 自身的版本历史

每个 Skill 自身的版本按线性历史展示，例如 v1、v2、v3、v4；不提供 Git 式父子、分支或合并视图。版本表显示 version_id、version_label、来源、状态、创建时间。支持选择任意两个版本进入 diff，默认“本地基准 → 已发布版本”。读取内容时校验哈希。版本顺序应由服务明确提供，不能用可重复的展示标签做身份或排序依据。

### skill_manage 管理图谱

管理图谱是独立的组织视图。每个展示节点引用“一个 Skill 的一个确定 version”，例如 literature-search@v3、evidence-check@v2。节点通过 skill_version_id 解析 Skill 身份与版本内容；同名 Skill、相同版本标签都不能用于合并节点。边来自 skill_manage 的组织关系，不从版本序列或 parent 字段推断。

页面布局：顶部选择图谱和图谱快照/修订，中央展示节点与关系，点击节点后出现详情区（Skill 名称、版本、内容入口、对比、导出）。支持搜索定位节点、按关系类型筛选；小屏显示节点列表和明确的关系列表，不能丢失边语义。具体关系名称按管理算法提供的数据呈现，现有 schema 包含 dependency、containment、flow、semantic、workflow、alternative。

Skill version、graph_version_id（图内容快照）与 revision（图发布修订）分别展示，不混用。图谱节点固定引用一个确定版本；某 Skill 出现新版本不应让历史图谱自动漂移。是否在新图谱中采用新版本，由 skill_manage 的组织结果决定。单 Skill 版本页可定位到引用它的管理图谱，图谱节点可打开该 Skill 的线性历史。

首版以查看已有组织结果为主，不默认提供手工改边、建分支或自动替换节点版本操作。图谱是本次设计的核心功能，需要补公开查询接口；不能用 SDK 的父版本字段代替，也不能以版本树替代交付。目前数据库已有 skill_graphs、skill_graph_versions、skill_graph_nodes、skill_graph_edges、skill_graph_revisions 结构，但不将这些表结构描述为已可用的 UI 查询服务。

当前源码仍有 SDK parent_version_id 和本地 parent_version_ids 字段；本次以用户明确的线性版本语义为设计依据，不使用这些字段构建 UI 图谱，也不在设计阶段修改底层 schema。底层 schema 另有 scene 类型，本次界面只按要求展示 Skill-version 节点，不擅自加入其他节点类型。

图谱不默认展示不存在的评估分数。未来有真实指标时，必须附带指标名称、数据范围与评估来源。

### 对比

同一个 Skill 的任意两版本：并排/统一 diff，标记新增和删除，支持交换方向。窄屏采用统一 diff。对比对象展示版本 ID 与 hash，而非仅标签。

不同 Skill 的内容对比：允许在两侧分别选择 Skill 和版本，明确“跨 Skill 对比”，不推断共同祖先。需要在 SDK 补公开内容读取/通用 diff 能力；不能直接复用限定单个 skill_id 的 diff 签名。

### 注册

选择本机 Skill 目录或 SKILL.md → 预览名称、路径、文件范围、hash 与云端上传目的 → 注册。复用 plan_register/register。同名已存在时明确区分新 Skill 与已有 Skill 的新版本，不能静默创建不清楚的关联。重复点击防止重复提交；失败或待上传显示真实状态，提供重试。

### 导出与生效

选择明确版本 → 输入目标 Skill 目录 → 预览将写入文件及现有差异 → 执行导出。默认建议 published head，允许显式选择其他状态，但清晰显示“未发布”。

必须区分三个动作：

1. 导出副本：写入目标目录，不修改原 Skill 注册路径或本地基准版本。
2. 更新本地：覆盖已注册目录，并在成功后推进 SDK registry。
3. 本地回滚：应用指定历史版本到已注册目录，不修改云端发布状态。

首版用户核心路径是“导出副本”。目标目录就是该 Skill 的最终目录，不再偷偷追加同名子目录；UI 显示完整预览路径。目标有文件时先展示冲突，用户选择后备份受管文件再替换，不清空目录或删除无关内容。执行时重查目标指纹，计划过期需重新预览；失败保留/恢复原文件。拒绝路径逃逸及越出授权范围的符号链接。

当前 SDK bundle 白名单仅 SKILL.md。UI 明确标注“当前仅导出 SKILL.md”；scripts/references/assets 完整 Skill 导出属于后续契约扩展，不能承诺已有能力。成功状态写“已导出”，展示目录和版本；是否已被 Agent 读取取决于 Agent 的加载/重载机制，不能直接写“已生效”。

可保存轻量导出记录（源 Skill、版本、hash、目标路径、时间、结果），与 SDK registry 区分；目标是已注册目录时转入更新/回滚流程，避免目录内容与 registry 分离。

### 删除

操作文案必须反映真实行为：现有 SkillManager.unregister 会解除云端管理关系、移除本地 registry 及对应本地历史；不是纯本地移除，也不删除本地源文件。

首版支持“解除管理”，确认区展示 Skill、云端影响与本地文件保留说明。单独“仅移除本地管理”需要 SDK 新公共方法；“删除单版本”及“彻底删除磁盘文件”不在首版实现范围，不提供无后端支持的按钮。部分失败需刷新真实状态，不能展示假成功。

## 4. 模块边界

建议新建独立 src/mindmemos_ui Python 包，浏览器前端通过同源轻量 FastAPI 接入 SDK 配置与 Skill 操作，并通过 skill_manage 的公开查询服务获取组织图谱。UI 不直接读写 SDK 的 JSON 文件，不直连算法数据库。实施时需补齐图谱查询契约，明确项目、图谱、快照及节点引用。

建议目录（本阶段仅设计，未创建实现）：

```text
src/mindmemos_ui/
  pyproject.toml
  mindmemos_ui/
    __init__.py
    app.py             # 应用生命周期、静态资源、SDK 客户端
    cli.py             # 启动服务、选择 SDK home 和端口
    routes/
      config.py
      skills.py
      graphs.py        # skill_manage 图谱查询
    schemas.py         # 仅 UI 请求与响应
    static/            # 构建后的前端
  frontend/
    src/
      app/
      features/config/
      features/skills/
      features/graphs/
      components/
```

建议 React + TypeScript + Vite；标准 CSS variables 维护样式，图谱再按需求引入成熟图库。Python 分发包含编译后的静态资源，最终使用者不需要运行 Node。单个本地服务启动，默认绑定 loopback；认证信息保留在本地服务，浏览器不直接带云端 Key 请求。写接口校验同源/会话令牌，避免其他网站触发本机写操作。

SDK 复用已有配置、注册和文件操作能力；skill_manage 及其持久化服务负责组织图谱及快照。UI 只负责校验交互输入、调用和呈现。需要明确本地 skill_version_id 与云端版本的映射，不能假定两者相等。SDK 缺少的导出/内容读取方法补在 SDK，避免前端或 UI 后端复制业务规则。不提前创建适配器注册框架。

## 5. 能力核对与实现顺序

核对来源为本工作区当前源码，不假定未提交的 mindmemos_skill 骨架已经完成业务实现。

| 能力 | 当前依据 | 工作 |
| --- | --- | --- |
| 配置 | config/models.py、ConfigManager | 表单、脱敏响应、保存后重建客户端 |
| 本地/云端 Skill 列表 | SkillManager.list、SkillCloudClient.list_skills | 合并展示、未关联状态 |
| 版本/内容 | pull/history、SkillCloudClient.get_content | 公共缓存读取接口、内容预览 |
| 线性版本历史 | Skill 身份及版本记录 | 版本列表/时间线，不展示父子分支 |
| 同 Skill diff | SkillManager.diff | diff 视图 |
| 跨 Skill diff | 已有内容接口，无通用 diff 入口 | 补最小 SDK 公共接口 |
| 注册 | plan_register/register | 预览、确认、真实失败状态 |
| 任意目录导出 | 无直接公共入口 | 补 plan_export/export 及导出记录 |
| 更新/回滚 | plan_update/apply_checkout/rollback | 计划时与执行时的一致性保护 |
| 解除管理 | unregister | 准确影响提示；不冒充本地删除 |
| skill_manage 管理图谱 | 已有图谱/快照/节点/边/修订表结构 | 补公开查询服务，节点引用确定 Skill version |

值得在实施前修正：SkillManager._published_head_for_record 当前使用 published_head or latest_version，导致无发布版本时 update 可能取 latest；应先明确并统一“更新到已发布版本”契约。sync 当前还会更新 history.last_pulled_at，完整版本历史查询应避免由于同步水位跳跃遗漏早期版本，可从 versions_since(since=None) 获取完整版本元数据。该接口只提供 Skill 版本历史，管理图谱另走 skill_manage 查询服务。

实施顺序：应用壳/配置 → 列表/内容/版本 → 图谱/diff → 注册/导出/解除管理。每一阶段都覆盖空状态、加载、离线/401/403、内容缺失和写冲突。验证重点是复用 SDK 行为、秘密不回显、导出不更改原 registry、冲突时不覆盖、恢复文件一致性；不为纯样式写重复测试。
