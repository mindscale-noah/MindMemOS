# Demo Skill 全周期体验

从仓库根目录运行。使用当前源码的 `.venv/bin/mindmemos skill`，连接独立 Skill HTTP 服务。三个示例位于 `resources/demo_skills/`：会议纪要（含参考资源）、缺陷分诊、版本说明。注册只保存指令和资源，不会执行 Skill 或调用模型。

## 1. 使用 dev.yaml 服务

服务配置固定为 `config/mindmemos_skill/dev.yaml`。若服务已经启动，直接进入下一节，不重复启动。

需要启动时，在终端 A 执行（数据库、Kafka 等依赖须按 dev.yaml 就绪）：

```bash
MINDMEMOS_SKILL_CONFIG_PATH=config/mindmemos_skill/dev.yaml \
  .venv/bin/python -m uvicorn mindmemos_skill.api.entries.fastapi:app \
  --host 127.0.0.1 --port 8001
```

此命令直接加载 dev.yaml，不替换鉴权或数据库配置。`make dev` 的默认开发流程可能将鉴权替换为 local；如需保留 dev.yaml 的服务鉴权，可使用 `make dev SERVICE=skill UI=0 SERVICE_AUTH=1 SKILL_CONFIG=config/mindmemos_skill/dev.yaml`。该开发入口仍会根据依赖规划生成运行时配置；需要原样加载时使用上面的直接启动命令。

```bash
curl http://127.0.0.1:8001/health
```

健康检查成功只证明服务运行，不能证明注册权限或客户端凭据有效。直接启动的服务在终端 A 按 Ctrl+C 停止。

## 2. 使用正式服务的客户端配置

若当前终端仍保留上一版体验的配置变量，先清除它，再检查已有配置：

```bash
unset MINDMEMOS_HOME
.venv/bin/mindmemos config show
.venv/bin/mindmemos skill list
```

若你的正式客户端使用自定义 MINDMEMOS_HOME，应将其设置回原来的目录，而不是清除。确认 Skill 地址和凭据对应目标服务；已有配置正确时，无需再次 auth。

当前 dev.yaml 的 `fastapi.auth.mode` 为 `gateway_jwt`。直接访问 `http://127.0.0.1:8001` 时，服务验证 Gateway 签发的 JWT，不能使用 `demo-local` 占位 key。若通过正式 Gateway 访问，则配置实际转发 Skill 路由的 Gateway 地址和该 Gateway 接受的 API key。

需要重新配置时，使用交互命令：

```bash
.venv/bin/mindmemos auth --service-type custom
```

按提示填写真实 Memory 地址与凭据、Skill 地址与凭据及用户 ID。直接连接 gateway_jwt 服务时，在 Skill API key 输入有效 JWT，客户端将其作为 Bearer 凭据发送；JWT 过期后需要更新。项目身份和权限由服务验证的凭据决定，用户 ID 不能替代项目身份。

本指南保留 demo Skill 的别名和本地导出目录，但所有注册、提交和删除请求均访问上述正式服务配置对应的目标项目。删除命令只指定本指南创建的三个 demo 家族。

## 3. 注册三个 Skill

```bash
.venv/bin/mindmemos skill register resources/demo_skills/demo-meeting-notes --alias demo-meeting --version v0.0.1
.venv/bin/mindmemos skill register resources/demo_skills/demo-bug-triage --alias demo-bug --version v0.0.1
.venv/bin/mindmemos skill register resources/demo_skills/demo-release-notes --alias demo-release --version v0.0.1
```

保存第一条响应的 `skill_version_id`，后面用它比较版本。重复注册同一别名会冲突；可换别名或先删除原 demo。注册会递归上传源目录内辅助文件，会议示例的 `references/output-template.md` 会一起保存。

## 4. 查看和导出

```bash
.venv/bin/mindmemos skill list
.venv/bin/mindmemos skill list --name demo-meeting-notes
.venv/bin/mindmemos skill resolve demo-meeting
.venv/bin/mindmemos skill show demo-meeting
.venv/bin/mindmemos skill history demo-meeting
.venv/bin/mindmemos skill export demo-meeting --to output/demo-skill-cli/meeting-edit
```

`resolve` 返回家族及选定版本 ID，`show` 返回完整内容及资源，`history` 返回版本标签和说明。别名选择当前激活版本；精确版本 ID 选择指定历史版本。导出目录已存在且非空时命令拒绝覆盖，确认要覆盖后才添加 `--overwrite`。

## 5. 修改并提交新版本

在导出副本上修改，保留仓库 demo 原件：

```bash
printf '\n7. 在摘要中注明记录里明确给出的会议日期；缺失时写“待确认”。\n' \
  >> output/demo-skill-cli/meeting-edit/SKILL.md

.venv/bin/mindmemos skill commit demo-meeting \
  --from output/demo-skill-cli/meeting-edit \
  --version v0.0.2 --message '增加会议日期检查'

.venv/bin/mindmemos skill history demo-meeting
.venv/bin/mindmemos skill resolve demo-meeting
```

提交响应中 `active` 为 false：新版本已经保存，当前仍是 v0.0.1。默认继承父版本资源；修改导出目录中的辅助文件不会自动上传。需要同时更新辅助文件时，在 commit 添加 `--resources-dir output/demo-skill-cli/meeting-edit`；需要清空资源时用 `--clear-resources`。

## 6. 激活、对比和回退

```bash
.venv/bin/mindmemos skill activate demo-meeting --version v0.0.2
.venv/bin/mindmemos skill show demo-meeting
```

将下方值替换为首次注册响应中的完整 `skill_version_id`：

```bash
FIRST_VERSION='替换为首次注册的skill_version_id'
.venv/bin/mindmemos skill diff "$FIRST_VERSION" demo-meeting

.venv/bin/mindmemos skill activate demo-meeting --version v0.0.1
.venv/bin/mindmemos skill resolve demo-meeting
```

diff 从旧版本比较到当前激活的 v0.0.2，应该只有 SKILL.md 的日期检查变化。回退只切换激活指针，不删除 v0.0.2。也可以提交时添加 `--activate`，直接保存并激活新版本。

## 7. 修改管理信息

```bash
.venv/bin/mindmemos skill update-info demo-meeting \
  --alias demo-meeting-managed --metadata-json '{"purpose":"cli-demo","owner":"demo-user"}'
.venv/bin/mindmemos skill resolve demo-meeting-managed
.venv/bin/mindmemos skill list --alias demo-meeting-managed
```

修改别名后，后续使用新别名。此操作修改家族管理信息，不创建正文新版本；metadata 是提供的管理对象，不要将其当成局部字段补丁。

## 8. 删除和验证

```bash
.venv/bin/mindmemos skill delete demo-meeting-managed --yes
.venv/bin/mindmemos skill delete demo-bug --yes
.venv/bin/mindmemos skill delete demo-release --yes
.venv/bin/mindmemos skill list
```

删除作用于整个 Skill 家族，属于软删除，保留历史和引用记录；不删除本地源目录或导出文件。去掉 `--yes` 可体验交互确认。若仓库中只有这三个 demo，最后列表为空。删除后的别名不能继续正常 show/resolve。

## 示例任务

- 会议纪要：“今天讨论登录异常。小李明天检查日志；是否上线修复还未决定。”
- 缺陷分诊：“Chrome 中点击保存后无响应，刷新后内容丢失，暂时没有日志。”
- 版本说明：“v0.2.0 新增导出；修复空列表错误；重构内部缓存，尚未做性能测试。”

将导出的 SKILL.md 及其引用资源提供给你的 Agent，才能进一步体验指令执行；管理命令不会替你运行任务。
