# mindmemos_ui 使用说明

## 启动与连接

在仓库根目录安装工作区依赖并启动 UI：

```bash
uv sync
make ui
```

打开 http://127.0.0.1:8765 。可用 `make ui UI_PORT=8765 UI_ARGS='--home /absolute/path/to/.mindmemos'` 指定 SDK 配置目录和端口。HOME 优先级为显式 `--home`、`MINDMEMOS_HOME`、`MINDMEMOS_CONFIG_DIR`、本机保存的默认目录、`~/.mindmemos`。

启动后在“配置”页的 SDK `settings.json` 中设置 Memory 和 Skill 服务。`service_type=official` 时两项服务使用官方地址，Skill 复用 Memory 的 API Key；`service_type=custom` 时填写 Memory 地址，Skill 地址留空则复用 Memory 连接，填写独立 Skill 地址时还需填写其 API Key。保存后下一次业务请求会使用新配置，无需重启 UI。服务端 `service.yaml` 用于启动服务，不由 UI 配置页修改。

左下角分别显示 Memory 与 Skill 的连接状态。状态来自独立的 SDK 只读请求，每 30 秒更新，也可点击“刷新”。“已连接”表示对应服务通过认证并响应；“未配置”“认证失败”“无法连接”分别表示缺少凭据、凭据被拒绝或探测未成功。没有可用 Skill 连接时，Skill 页面不会自动创建或读取本地仓库。项目范围由 Skill 服务验证 API Key 后确定，UI 不提供项目 ID 覆盖参数。

前端已构建并随 Python 包提供。修改前端后运行 `make ui-build`，然后刷新浏览器。

## Skill 工作台

Skill 列表、详情、历史、内容、差异、注册、提交、切换、删除和图谱读取均通过 `mindmemos_sdk` 访问当前配置的 Skill 服务。浏览器只调用本机同源 UI，API Key 不传给浏览器。服务端的身份和项目权限决定可见内容。

- **注册**：选择本机 Skill 目录或 `SKILL.md`，名称须与 frontmatter 一致，填写服务内唯一的别名。预览会列出 `SKILL.md` 与全部辅助文件，确认后将完整快照上传到 Skill 服务。
- **版本**：查看历史、文本文件和二进制资源，比较同一或不同 Skill 的两个版本。默认初始版本号为 `v1.0.0`；提交可指定新版本号，或按修订、次版本、主版本递增。提交时核对基准版本，避免覆盖并发更新。
- **切换与删除**：切换修改服务中的当前版本指针；删除受服务端已有引用和权限约束。这两项操作不会删除本机源目录或已导出的目录。
- **导出**：从服务获取完整版本快照，再写入本机指定目录。预览列出文件，确认前重新核对目标目录；覆盖会整体替换目录并移除旧文件。预览十分钟有效、令牌仅可使用一次。导出副本是否被 Agent 读取取决于其后续加载。
- **管理图谱**：SkillSets 仅显示当前服务中的真实图谱，默认选择第一张图；没有图谱时显示“暂无 SkillSet”，未连接服务时同时显示连接提示。点击节点可在节点旁查看 Skill 详情，点击连线可查看关系类型及两端 Skill；点击空白处或浮层关闭按钮可收起详情。页面顶部的“导出 JSON”会下载当前图谱快照的全部节点和关系，关系类型筛选只影响屏幕显示，不影响导出内容。从图谱下拉框选择服务图谱后，页面通过同一 Skill 服务读取当前项目的快照、节点与关系；UI 不构建、发布或手工编辑服务图谱。

动态 Skill 版本可以输入任务，调用该实现的 `Skill.assemble(task)` 在本机组装后预览。原始版本通过 SDK 读取，组装结果只在当前页面显示，不写回服务。内置加载器支持静态 Skill 和 TreeSkill；其他实现未接入时会明确报错。TreeSkill 的非敏感模型配置取自版本快照，调用方需通过 config 显式注入带凭据的 router。当前默认加载器不注入模型凭据，调用模型时会报配置缺失；自定义加载器可调用 `TreeSkill.from_snapshot(snapshot, router=...)`。组装动态 Skill 可能发起模型调用。

## 配置与安全边界

配置页编辑 SDK v2 `settings.json`，支持表单、JSON 源码、校验、保存和单独测试服务连接。旧 v1 文件由 SDK 升级为 v2。页面只读显示本次启动选定的 HOME；可执行 `mindmemos config home set "/absolute/path/to/.mindmemos"` 保存本机默认目录，也可在启动环境设置 `MINDMEMOS_HOME` 或传 `--home`。切换后需重启 UI；原目录的 `settings.json` 不会自动移动。字段定义见 [SDK 配置模型](../../src/mindmemos_sdk/mindmemos_sdk/config/models.py)。

UI 仅监听 `127.0.0.1`。本机路径均指运行 UI 的机器；导出禁止写入 SDK HOME、其子目录和上层目录，也拒绝符号链接路径。写请求检查同源和会话令牌，浏览器不会直接携带远端 API Key。

本项目仓库中的 UI 使用同一 uv workspace 内的新版 `mindmemos_sdk`。单独安装已发布的旧 SDK 版本不能提供本页面新增的 Skill 图谱和完整资源接口；独立分发需在新版 SDK 发布后更新对应依赖。

## 验证

### 一键启动 UI 检查环境

安装工作区依赖、准备服务配置并启动 Docker 引擎后，在仓库根目录运行：

```bash
make dev FEATURE=core
```

统一入口启动选中服务所需的 Docker 容器、Memory API、Skill API 和 UI，并等待健康检查。默认 UI 地址为 <http://127.0.0.1:8765>；实际地址、异步任务和监控开关会显示在启动摘要中。

```bash
# 预览，不启动进程或容器
make dev FEATURE=core PLAN=1

# 自定义端口
make dev FEATURE=core API_PORT=18010 SKILL_PORT=18011 UI_PORT=18766

# 只启动 Skill 及 UI
make dev SERVICE=skill FEATURE=core

# 查看当前状态或停止本工作区受管理的服务
make status
make stop
```

省略 `FEATURE` 启用完整功能；`FEATURE=async` 启用核心功能和异步任务。`make api` 仅启动 API，`make ui` 仅启动 UI。启动规则和配置优先级见 [Docker 与本地开发说明](../../dockers/README.md)。

`make dev` 不自动填充演示数据。页面检查需要已有数据，或通过页面注册 Skill、添加 Memory。端口占用时启动器会报错；按 `Ctrl+C` 关闭本次 API/UI 进程，保留容器和数据。`make stop` 同时停止受管理的应用及容器并保留数据卷。

### 独立 SQLite 图谱示例

需要在空白本地 Skill 服务中试用列表、详情、内容、版本历史、差异和管理图谱时，可运行演示数据脚本：

```bash
PYTHONPATH=src/mindmemos_skill:src/mindmemos_sdk .venv/bin/python scripts/ui/seed_skill_graph_demo.py
```

脚本接受绑定了本地项目的 SQLite 或回环地址 PostgreSQL Skill 服务配置；默认使用从 `dev.example.yaml` 复制的 `config/mindmemos_skill/dev.yaml`。运行中的 Skill 服务若绑定了另一个本地项目，可用 `--project-id <该服务的项目 ID>` 指定同一项目；`--config` 可选择其他本地配置。数据库类型由 Skill YAML 决定；脚本必须连接到运行中服务实际使用的数据库，并使用相同项目身份。脚本会创建 4 个虚构 Skill、其中 2 个的第二版本，以及 2 个含多种关系的服务图谱（共 3 个快照）；重复运行不会增加版本或快照。源目录位于 `resources/ui/skill-graph-demo/`，也可用于测试页面上的“注册 Skill”预览。刷新页面后，从 SkillSets 的图谱下拉框选择这些服务图谱即可查看。

这组演示数据不调用模型。动态 Skill 的任务组装需要单独配置模型，并会产生实际模型请求。

```bash
.venv/bin/pytest -q src/mindmemos_ui/tests src/mindmemos_sdk/tests src/mindmemos_skill/tests/test_fastapi.py tests/integration
make ui-build
```

前端源码位于 `src/mindmemos_ui/frontend/src/`，构建产物位于 `src/mindmemos_ui/mindmemos_ui/static/`。
