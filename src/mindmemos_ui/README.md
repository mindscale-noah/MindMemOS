# mindmemos_ui

MindMemOS 的本机 Web 工作台。你可以在浏览器中查看 Memory、管理 Skill 版本、查看 Skill 图谱，并编辑 SDK 连接配置。

UI 使用 FastAPI 提供本机接口，使用 React 和 TypeScript 显示页面。Memory 和 Skill 的业务请求通过 `mindmemos_sdk` 发送到已配置的服务。启动 UI 不会自动启动这些服务。

## 功能

| 页面 | 支持的操作 |
| --- | --- |
| Memory | 分页查看记忆、展开长文本、搜索记忆 |
| Skills | 分页查看 Skill、注册 Skill、查看内容和辅助文件、查看版本历史、比较版本、提交版本、切换当前版本、删除和导出 |
| SkillSets | 查看服务中的图谱、查看节点和关系、筛选关系类型、导出图谱 JSON |
| Algorithm Visualization | 选择本机输出目录，查看 Skill-Forest 树结构、节点 gain、验证进度及接纳/拒绝状态 |
| 配置 | 编辑 SDK `settings.json`、校验配置、保存配置、测试服务连接 |

动态 Skill 支持输入任务后预览组装结果。内置加载器支持静态 Skill 和 TreeSkill。组装结果只在当前页面显示，不写回 Skill 服务。

## 算法可视化

左侧打开 **Algorithm Visualization**，在算法下拉菜单选择 **skill-forest**，输入输出目录的绝对路径，再点击 **Visualize**。当前支持 **Skill-Forest**，无需连接 Memory、Skill 或模型服务。目录是运行 UI 的机器上的目录。

可选择包含 `trees/` 的训练目录，也可选择实验输出父目录；发现多个训练记录时，页面提供独立选择，不合并不同运行。支持切换树、点击节点查看详情、按节点 ID 和状态突出显示、缩放及适应画布。再次点击 **Visualize** 读取新的落盘进度；目录不变时保留当前树和节点选择。

验证分母从输出中的 split 或根节点读取，不依赖特定数据集和算法配置。提前拒绝的部分验证节点显示已验证数量及观察分数，gain 显示为未知，避免把占位的零当作完整结果。页面只显示已保存 checkpoint 的节点；快照不代表进程仍在运行。读取范围限于所选目录，损坏记录和扫描上限会有提示，不会执行训练或回写实验文件。

## 快速开始

以下命令均在仓库根目录执行。需要 Python 3.11–3.13 和 uv。

```bash
uv sync
make ui
```

打开 <http://127.0.0.1:8765>。

通过 `make dev` 启动时，UI 自动使用 `config/presets/local.project.yaml` 的 `default_project` 凭据连接本地 Memory 和 Skill，客户端配置保存到 `.mindmemos/settings.json`。该项目文件包含多个 project，服务根据每次请求的 API key 选择项目。`PROJECT=schema` 只切换客户端默认凭据；`UI_HOME` 可覆盖开发工作台的 SDK 目录。独立运行 `make ui` 使用下述 SDK 配置目录规则。

1. 打开“配置”页。
2. 设置 Memory 和 Skill 的服务地址及 API Key。
3. 测试连接并保存配置。
4. 打开对应的业务页面。

Python 包内包含已构建的前端。只启动 UI 时，无需安装 Node.js 或重新构建前端。

### 指定端口和配置目录

```bash
make ui UI_PORT=8876 UI_ARGS='--home /absolute/path/to/.mindmemos'
```

也可以使用命令行入口：

```bash
uv run mindmemos-ui --port 8876 --home /absolute/path/to/.mindmemos
```

| 参数 | 用途 | 默认值 |
| --- | --- | --- |
| `--port` | 设置本机监听端口 | `8765` |
| `--home` | 指定 SDK 配置目录 | 按下述优先级选择 |
| `--env-file` | 加载指定的环境文件，例如动态 Skill 使用的模型凭据 | 不指定 |

SDK 配置目录按以下顺序选择：

1. `--home`。
2. `MINDMEMOS_HOME`。
3. `MINDMEMOS_CONFIG_DIR`。
4. 通过 `mindmemos config home set` 保存的本机默认目录。
5. `~/.mindmemos`。

切换配置目录后，需要重启 UI。原目录中的配置文件不会自动移动。

## 连接配置

配置页编辑所选 SDK 目录中的 `settings.json`。页面支持表单和 JSON 源码两种编辑方式。

- `service_type=official`：使用官方服务地址。Skill 复用 Memory 的 API Key。
- `service_type=custom`：填写 Memory 服务地址。Skill 地址留空时，Skill 复用 Memory 连接。填写独立 Skill 地址时，还需填写对应的 API Key。

保存配置后，下一次业务请求使用新配置，无需重启 UI。服务端的 `service.yaml` 由服务部署者维护，不通过 SDK 配置页修改。

左下角分别显示 Memory 和 Skill 的连接状态。UI 每 30 秒发送独立的只读请求，也支持手动刷新。

Memory 和 Skill 列表每页显示 20 条，刷新只重新读取当前页。Skill 列表仅补充当前页 Skill 的版本信息，名称和别名筛选作用于当前页。业务数据没有定时刷新。

手动刷新在同一本机 UI 进程内共享每秒一次的限制；Memory 列表、Skill 列表和连接检查的读取入口也分别限制为每秒一次，避免整页刷新或多标签页绕过。超限返回 HTTP 429，页面提示“操作频繁，请稍后重试”。不设置业务响应缓存。

| 状态 | 含义 |
| --- | --- |
| 已连接 | 服务通过认证并响应探测请求 |
| 未配置 | 缺少连接凭据 |
| 认证失败 | 服务拒绝凭据 |
| 无法连接 | 探测请求未成功 |

Skill 的可见项目由服务验证 API Key 后确定。UI 不提供项目 ID 覆盖参数。没有可用 Skill 连接时，UI 不会自动使用本地 Skill 仓库。

配置字段和命令见 [SDK README](../mindmemos_sdk/README.md)。

## Skill 操作

### 注册与提交

注册时，选择运行 UI 的机器上的 Skill 目录或 `SKILL.md`。Skill 名称须与 frontmatter 一致，别名须在服务内唯一。

UI 先预览 `SKILL.md` 和辅助文件。确认后，UI 将完整快照上传到 Skill 服务。

提交版本时，可以填写版本号，也可以选择修订、次版本或主版本递增。UI 会核对基准版本，避免覆盖并发更新。

### 版本与导出

你可以查看历史版本、预览文本和二进制资源，并比较同一 Skill 或不同 Skill 的两个版本。

切换版本会修改服务中的当前版本指针。删除操作受服务端权限和已有引用约束。这些操作不会删除本机源目录或已导出的目录。

导出时，UI 从服务获取完整版本快照，并写入本机指定目录。导出预览有效期为十分钟，确认令牌只能使用一次。**选择覆盖时，UI 会整体替换目标目录，并移除目标目录中的旧文件。**

### 图谱

SkillSets 显示当前服务中的真实图谱。点击节点可查看 Skill 详情，点击连线可查看关系及两端 Skill。

关系筛选只影响页面显示。“导出 JSON”包含当前图谱快照的全部节点和关系。UI 支持读取图谱，不提供图谱构建、发布或手工编辑操作。

### 动态 Skill

动态版本的任务预览在本机调用 `Skill.assemble(task)`。TreeSkill 的非敏感模型配置来自版本快照；运行时必须通过 config 显式注入带凭据的 router，不读取环境变量。

需要加载指定的模型环境文件时：

```bash
make ui UI_ARGS='--env-file .env'
```

动态 Skill 组装可能发起模型请求。模型凭据与 Memory、Skill 服务的 API Key 用途不同。

## 本机访问与文件边界

UI 仅监听 `127.0.0.1`。浏览器调用本机同源 UI 接口，远端服务的 API Key 由 UI 后端用于 SDK 请求。业务请求不由浏览器直接发送到远端服务。

页面中的本机路径均指运行 UI 的机器。导出拒绝符号链接路径，也禁止写入 SDK 配置目录、其子目录及其上层目录。写请求检查同源和会话令牌。

## 开发与验证

### 构建前端

修改前端后，在仓库根目录执行：

```bash
make ui-build
```

此命令安装锁定的前端依赖，检查 TypeScript 类型，并构建静态文件。构建完成后刷新浏览器。

| 目录 | 内容 |
| --- | --- |
| `frontend/src/` | React 和 TypeScript 源码 |
| `mindmemos_ui/` | FastAPI 应用、SDK 接入和本机文件操作 |
| `mindmemos_ui/static/` | 随 Python 包提供的前端构建产物 |
| `tests/` | UI 后端测试 |

### 后端测试

```bash
uv run pytest -q src/mindmemos_ui/tests
```

### 准备人工检查环境

安装工作区依赖、准备服务配置并启动 Docker 引擎后，执行：

```bash
make dev FEATURE=core
```

此命令启动所需数据库、Memory API、Skill API 和 UI，关闭异步任务和 OTel。默认 UI 地址为 <http://127.0.0.1:8765>，实际地址和功能状态以启动完成后的摘要为准。省略 `FEATURE=core` 可启动完整功能。

命令不会自动填充演示数据。可通过页面注册 Skill 或添加 Memory；模型调用需要配置有效凭据。连接配置、端口和停止方式见 [UI 使用说明](../../docs/mindmemos_ui/usage.md)。

## 相关文档

- [UI 使用说明](../../docs/mindmemos_ui/usage.md)：完整操作步骤与本地检查环境。
- [SDK README](../mindmemos_sdk/README.md)：客户端配置与服务调用。
- [Skill 服务 README](../mindmemos_skill/README_ZH.md)：服务配置、启动和 Skill 管理。

本仓库中的 UI 使用同一 uv workspace 内的 SDK。独立分发时，需要确认 SDK 版本包含 UI 使用的 Skill 图谱和完整资源接口。旧版已发布 SDK 不能据此视为兼容。
