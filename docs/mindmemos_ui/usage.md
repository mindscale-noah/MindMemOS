# mindmemos_ui 使用说明

## 启动

在仓库根目录执行：

```bash
make ui
```

打开 http://127.0.0.1:8765 。自动打开本地工作区，初始化缺失的数据库表并连接图谱；无需指定数据库或项目 ID。使用现有 `.venv`。如尚未安装工作区依赖，先执行 `uv sync`。模块已经加入根 uv workspace；同步后也可运行 `uv run mindmemos-ui`。

指定独立配置目录和端口：

```bash
make ui UI_PORT=8765 UI_ARGS='--home /absolute/path/to/.mindmemos'
```

前端已构建并随 Python 包提供。使用时不需要运行 Node 服务。修改前端后执行 `make ui-build`，再刷新浏览器。

## 功能

- **Skills 中心仓**：通过 `MindMemOSSkill` 查询中心仓，按名称/别名搜索，预览后注册完整快照；不依赖云端地址或 API Key。
- **版本**：读取历史版本、查看 SKILL.md 与辅助文件列表、比较同一 Skill 的两个版本。
- **文件预览**：`.md` / `.markdown` 渲染 Markdown（含表格、任务列表）；`.py` / `.pyw` / `.pyi` 显示 Python 语法高亮，Markdown 中标记为 `python` / `py` 的代码块同样高亮。其他文本文件直接显示原文；二进制文件显示提示。版本对比保持逐行源码差异。
- **提交新版本**：从目录或 SKILL.md 读取内容和辅助资源，提交为子版本并设为当前版本；当前版本发生并发变化时拒绝提交。
- **切换当前版本**：选择中心仓默认读取和导出的版本，不修改导入来源或已导出目录。
- **导出**：选择版本和最终目录，预览文件清单后确认。完整导出 SKILL.md 与辅助资源；覆盖时整体替换目标目录并移除旧文件。确认前会核对整个目标目录是否变化；预览十分钟有效，令牌单次使用。
- **删除 Skill**：删除无引用的中心仓记录和历史，保留原导入/导出目录；有轨迹或图谱引用时拒绝删除。
- **管理图谱**：读取同一中心仓、同一项目范围的图谱及快照，查看节点绑定版本。
- **SDK 配置**：保留原 SDK 设置页面，仅用于旧 SDK 云端连接，与中心仓服务组装分开。本次不提供 Skill 服务配置编辑或新的配置文件持久化。

旧 SDK 本地注册表与云端记录不会自动导入中心仓，原 SDK 配置和记录保留。Skill 管理界面不再调用旧 `/v1/skills` 协议。

## 内置体验示例

仓库提供代码审查、实验结果核对、会议纪要整理三份示例；代码审查包含两个历史版本。它们的模板和练习输入一并保存为辅助资源。

在仓库根目录执行一次：

```bash
PYTHONPATH=src/mindmemos_skill:src/mindmemos_sdk .venv/bin/python scripts/skill/seed_examples.py
```

与 UI 一样默认解析 `MINDMEMOS_HOME`，也可通过 `--home /path/to/home` 选择目录。重复执行保留已有示例及其当前版本；中断的初始导入仅在来源、已有内容及当前版本仍匹配时继续。脚本不调用模型或云端服务，也不在 UI 启动时自动写入示例。

刷新 Skills 页面后打开“示例 · 代码审查”，可以比较两个版本，再将完整版本导出到新目录。导出的 `references/` 提供练习输入，`assets/` 提供结果模板；这些是体验材料，并非真实实验结果。其他示例位于 `resources/mindmemos_skill/examples/`。

## 本地工作区与图谱

HOME 优先级：显式 `--home` → `MINDMEMOS_HOME` → `MINDMEMOS_CONFIG_DIR` → `~/.mindmemos`。

首次打开会生成 `workspace.json`，记录稳定的内部本地范围和数据库路径；默认数据库是 `data/central.db`，相对于 HOME 解析。身份不依赖机器名或当前目录。重复启动复用同一份数据，不会生成新范围；不同 HOME 独立隔离。云端项目仍由远端身份决定，不使用本地工作区 ID 代替鉴权。

UI 的 Skill 和图谱共享同一个数据库与本地范围。新工作区没有图谱时显示空状态，不要求用户输入启动参数。管理算法写入图谱时使用服务的 `database` 和 `local_project_id`。

```python
from contextlib import closing
from mindmemos_skill.api import create_service
from mindmemos_skill.api.typing import ListRequest

with closing(create_service()) as service:
    skills = service.list_skills(ListRequest())
```

统一通过 `create_service` 组装：不传配置时使用本地默认设置，传入配置对象或 YAML 时按显式配置组装，返回原有 `SkillService`。没有 Workspace 类或第二个公开工厂。显式配置不能与 `home`、`database_path`、`project_id` 覆盖参数混用。本地请求省略 `project_id` 时使用绑定范围，显式传入其他范围则拒绝。

`make skill-local` 使用同一工作区。自定义路径可以设置 `MINDMEMOS_HOME`，或使用 `SKILL_LOCAL_ARGS='--home /path/to/home'`。需要原有 YAML 服务组装时使用 `SKILL_LOCAL_ARGS='--config /path/to/config.yaml'`；默认 YAML 未指定 SQLite URL 时，也解析工作区数据库。通用服务/HTTP 请求仍要求显式项目范围，不套用本地默认身份。

高级用户可用 `--graph-db` 覆盖整个本地仓库数据库、用 `--project-id` 覆盖本次本地范围，默认绑定不会被改写。这两个参数不再是正常启动的前提。

现阶段不迁移未发布的旧本地库，也不在错误时自动删除数据。无权限、损坏的工作区元数据或不兼容的数据库会报错，需要针对具体问题处理。UI 包依赖 Skill 包，工作区安装会一并安装。

这次实现包含图谱查询和呈现，不包含管理算法的运行、图谱构建/发布、手工编辑关系或云端数据库同步。

## 运行边界

- 服务固定监听 127.0.0.1。文件路径指向启动 UI 的机器。
- 工作区启动失败时报告错误，不以新空库掩盖错误，也不生成演示数据。
- 云端访问使用 SDK 保存的配置；没有可用地址或 Key 时需先配置。
- 导出禁止写入工作区 HOME 及其内部、包含 HOME 的上层目录、符号链接路径；服务还会阻止覆盖包含中心仓 SQLite 数据库的目录。导出副本不等于 Agent 已重载。
- 写请求需要同源会话令牌；浏览器不直接携带云端 Key。跨站请求及未知 Host 被拒绝。
- 本模块依赖同工作区的 Skill 包；当前按仓库 workspace 安装与运行。

## 验证

```bash
.venv/bin/pytest -q tests/mindmemos_ui tests/mindmemos_sdk
make ui-build
```

前端源代码位于 `src/mindmemos_ui/frontend/src/`；打包文件位于 `src/mindmemos_ui/mindmemos_ui/static/`。构建依赖由 package-lock.json 固定。

## 版本别名

UI 展示 `version_label`，例如 `v1.2.0`；数据库关联仍使用 `version_id`，完整 ID 可在历史版本的“审计信息”中查看和复制。

- 新注册默认 `v1.0.0`，也可输入合法初始版本。每段是无前导零的非负整数。
- 提交新版本可选择修订号、次版本、主版本递增，或手动指定未使用的版本别名。增加高位时低位归零；自动分配考虑同一命名空间的已有历史，切回旧版本也不会复用编号。
- 算法调用方传固定 `algorithm` 标识，例如 `treeskill`，持久层从输入版本分配 `v1.2.1-treeskill`。用户表单不提供算法身份，也不允许手填带尾缀的标签。
- 别名只用于展示，不能据其大小推断算法效果或谱系。当前指针、父版本 ID 和时间仍独立保存。
- 相同 Skill 内别名唯一，不同 Skill 可以各自从 `v1.0.0` 开始。别名创建后不改名；中心仓没有单版本删除接口，历史编号不复用。

启动新版服务会自动补齐旧版本别名；旧标签的审计映射见 `docs/mindmemos_skill/version_labels.md`。
