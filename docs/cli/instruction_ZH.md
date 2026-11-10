# MindMemOS CLI 使用指南

[English](instruction.md) · 简体中文

`mindmemos` 随 Python SDK 安装，用于配置服务连接、操作 Memory 和管理远端 Skill。每个命令都可以追加 `--help` 查看完整参数。

## SDK 配置与诊断

```bash
pip install mindmemos-sdk
mindmemos --help
```

要使用当前仓库源码，从根目录运行 `uv sync --all-packages`，再用 `uv run mindmemos` 执行下文命令。自部署服务的启动方式见[部署指南](../deploy/instruction_ZH.md)。

CLI 依次从 `MINDMEMOS_HOME`、兼容变量 `MINDMEMOS_CONFIG_DIR`、已保存的默认目录和 `~/.mindmemos` 中选择配置目录。新版连接写入 `settings.v2.json`。旧 `settings.json` 可供旧 CLI 继续读取。服务端凭据决定远端项目归属。

### `mindmemos auth`

选择官方或自部署服务，保存 API Key 与默认用户 ID。官方模式省略两个 base URL 参数。Memory 继续使用 `https://mindmemos.cn/v1/memory/*` 和原 Key。新版 Skill 的 API 根地址为 `https://mindmemos.cn/v2/skills`，需要具有 Skill 权限的项目 Key。

自部署时，`--memory-base-url` 填服务地址，由 SDK 拼接 `/v1/memory/...`。`--skill-base-url` 填完整的 Skill API 根地址，直连时包含 `/v1/skills`，并为该连接提供 Skill 服务的 Key。已有只填服务地址的 Skill 配置仍可使用。

```bash
mindmemos auth
mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --skill-base-url http://127.0.0.1:8001/v1/skills \
  --user-id alice
```

自部署时可留空跳过 Skill，已有独立连接时回车会保留。用 `--no-skill` 清除自部署的 Skill 连接。命令默认在提示时读取 Key。自动化可用 `--memory-api-key` 和 `--skill-api-key`，按连接模式提供。

### `mindmemos config show`

查看生效的服务地址、默认用户和配置文件路径。API Key 默认打码，`--show-secret` 会显示原值。

```bash
mindmemos config show
```

### `mindmemos config home set`

为后续启动的进程设置默认配置目录。当前进程需要重启，原目录的 `settings.json` 留在原处。

```bash
mindmemos config home set /absolute/path/to/.mindmemos
```

### `mindmemos config reset`

清除新版连接设置，保留旧 CLI 使用的 `settings.json`。默认要求确认，`--yes` 或 `-y` 可跳过确认。

```bash
mindmemos config reset
```

### `mindmemos doctor`

检查本地配置、凭据是否存在，以及能否初始化 SDK 客户端。`transport: ready` 表示客户端初始化成功。验证远端连通性和权限时，分别运行 `mindmemos memory get --top-k 1` 和 `mindmemos skill list --limit 1`。

```bash
mindmemos doctor
```

如果提示缺少配置或 Key，先用 `config show` 核对目录，再运行 `auth`。GUI 插件找不到 `mindmemos` 时，可为插件指定 [CLI 的绝对路径](../../skills/mindmemos-cli/references/openclaw-plugin.md)。

## Memory 命令

`add`、`search`、`feedback` 和 `dreaming` 都支持 `--user-id`、`--app-id`、`--agent-id` 和 `--session-id`。示例中的 `alice` 和 `MEMORY_ID` 请替换为实际值。

### `mindmemos memory add`

写入单条消息或一段对话。`--role` 默认是 `user`。多轮消息通过 `--messages-json` 或 `--messages-json-file` 传入非空 JSON 数组，业务元数据用 `--metadata-json`。

```bash
mindmemos memory add --content "我喜欢喝冰美式。"
mindmemos memory add --messages-json-file ./messages.json --async --json
```

默认使用配置中的用户并同步处理。`--async` 排队成功后返回 `request_id`，`--json` 输出完整结果。

### `mindmemos memory search`

按查询文本检索记忆，默认最多返回 10 条。使用 `--user-id` 限定用户，省略时检索当前项目。

```bash
mindmemos memory search "我喜欢喝什么咖啡？" --user-id alice --top-k 5
```

用 `--search-strategy agentic` 切换默认的 `fast` 策略。还可用 `--filter` 传入 JSON 过滤对象、用 `--token-budget` 限制结果的 token 数，或用 `--json` 输出完整结果。`--score-threshold` 只在启用 `--rerank` 时生效。

### 图文写入、检索和下载

项目启用多模态后，`--image` 和 `--query-image` 接受本地文件或 HTTP(S) URL，可重复传入。图片命中的 `metadata.blob_id` 和 `metadata.user_id` 可用于下载原图。

```bash
mindmemos memory add --content "Desk setup" --image ./desk.png
mindmemos memory search "Desk" --query-image ./desk.png --json
mindmemos memory search "Desk" --multimodal --json
mindmemos memory get-image BLOB_ID --user-id alice --output desk.png
```

### `mindmemos memory get`

列出当前项目中的记忆。用 `--top-k` 限制条数，或用 `--filter` 传入 JSON 过滤对象。

```bash
mindmemos memory get --top-k 20
```

### `mindmemos memory update`

用 `--content` 替换指定 ID 的记忆内容。

```bash
mindmemos memory update MEMORY_ID --content "用户喜欢无糖冰美式。"
```

### `mindmemos memory delete`

按 ID 删除一条记忆。默认要求确认，`--yes` 或 `-y` 可跳过确认。

```bash
mindmemos memory delete MEMORY_ID
```

### `mindmemos memory feedback`

省略 `--text` 时触发隐式反馈。显式反馈用 `--text` 和 `--messages-json` 或 `--messages-json-file` 提交原始对话，可用 `--recalled-memories-json` 或 `--recalled-memories-json-file` 附上该轮召回的记忆。

```bash
mindmemos memory feedback
mindmemos memory feedback --text "用户喜欢无糖咖啡" \
  --messages-json '[{"role":"user","content":"我喜欢无糖咖啡。"}]'
```

### `mindmemos memory dreaming`

触发记忆演进。默认异步排队，也可显式传 `--async`。用 `--sync` 等待完成。

```bash
mindmemos memory dreaming --sync
```

## Skill 命令

Skill 引用可以是别名、家族 ID 或版本 ID。别名和家族 ID 选中当前活动版本。用完整版本 ID 可固定评分、导出和对比的输入。图引用可以是别名、图家族 ID 或快照 ID。

示例中的 `SKILL_VERSION_ID`、`TRAJECTORY_ID` 和 `TASK_ID` 请替换为实际 ID。Skill 命令成功时输出 JSON，`task progress` 输出文本日志。失败时在 stderr 输出错误并返回非零退出码。

### 基础管理

管理 Skill 家族与版本、执行轨迹和图快照。

#### `mindmemos skill register`

从本地 Skill 目录或 `SKILL.md` 注册新家族。标准 `SKILL.md` 的 frontmatter 需包含 `name` 和 `description`。本地注册默认使用 `name` 作为别名，首个版本标签为 `v0.0.1`。

```bash
mindmemos skill register ./my-skill --alias my-skill
```

用 `--version`、`--message` 和 `--metadata-json` 设置首个版本信息。`--include` 和 `--exclude` 接受逗号分隔的资源匹配规则。也可用 `--json-skill` 传入完整的 `StructuredSkillInfo` JSON 对象，此时省略本地路径及本地来源选项。

#### `mindmemos skill list`

列出项目中的 Skill 家族，默认最多 20 条。可按 `--name`、`--alias`、`--user-id` 或 `--agent-id` 筛选，也可用 `--filter` 一次传入 JSON 对象。

```bash
mindmemos skill list --limit 20
```

#### `mindmemos skill show`

读取 Skill 家族信息和选定版本的完整内容，包括资源。

```bash
mindmemos skill show my-skill
```

#### `mindmemos skill resolve`

把 Skill 引用解析为家族 ID 和版本 ID。可从返回结果的 `skill_version_id` 取得当前解析到的具体版本。

```bash
mindmemos skill resolve my-skill
```

#### `mindmemos skill export`

将选定版本的内容和资源导出到 `--to` 指定的父目录下的 Skill 子目录。`--overwrite` 会替换同名子目录及其内容。

```bash
mindmemos skill export my-skill --to ./exports
```

#### `mindmemos skill commit`

以引用选中的版本为父版本创建新版本，返回新版本的 `skill_version_id`。本地来源用 `--from` 指定目录或 `SKILL.md`，并提供 `--message`。加 `--activate` 可在提交时激活。

```bash
mindmemos skill commit my-skill --from ./my-skill --message "补全输入校验步骤"
```

默认按 `patch` 递增并继承父版本的资源。可用 `--version` 指定标签，或用 `--bump minor`、`--bump major` 改变增量。资源可用 `--resources-dir` 替换或用 `--clear-resources` 清空。使用 `--json-skill` 时，提供包含 `commit_message` 的完整 JSON 对象，并省略本地来源选项。

#### `mindmemos skill history`

列出引用所属家族的版本历史。`--limit` 默认 20，`--offset` 默认 0，可组合使用。

```bash
mindmemos skill history my-skill --limit 20 --offset 0
```

#### `mindmemos skill diff`

按从左到右的顺序对比两个引用选中的版本。下面把 `my-skill` 当前活动版本与另一个指定版本对比。

```bash
mindmemos skill diff my-skill SKILL_VERSION_ID
```

#### `mindmemos skill activate`

切换家族的活动版本。用 `--version` 指定标签，或用 `--version-id` 指定同一家族的完整版本 ID。要恢复较早版本为活动版本，也使用这个命令。

```bash
mindmemos skill activate my-skill --version-id SKILL_VERSION_ID
```

#### `mindmemos skill update-info`

更新家族信息。用 `--name` 修改名称、用 `--alias` 设置别名、用 `--clear-alias` 清除别名，或用 `--metadata-json` 设置元数据。

```bash
mindmemos skill update-info my-skill --alias review-skill
```

#### `mindmemos skill delete`

软删除整个 Skill 家族。传入版本 ID 也会删除其所属家族。默认要求确认，`--yes` 或 `-y` 可跳过确认。

```bash
mindmemos skill delete my-skill
```

#### `mindmemos skill trajectory report`

上报一条执行轨迹。请求需要 `user_id`、`agent_id`、`messages` 和 `skill_version_ids`。列表包含执行时实际加载的精确版本。一个版本也用列表，无法确定版本时传 `[]`。服务返回一个轨迹 ID。用 `--file` 或 `--json-data` 提交，`--file -` 从标准输入读取。

```bash
mindmemos skill trajectory report --file ./trajectory.json
```

#### `mindmemos skill trajectory list`

列出引用选中版本的轨迹。可用 `--status`、`--received-from` 和 `--received-to` 过滤，默认最多返回 20 条。

```bash
mindmemos skill trajectory list my-skill --limit 20
```

#### `mindmemos skill trajectory show`

按轨迹 ID 读取一次完整的执行记录。

```bash
mindmemos skill trajectory show TRAJECTORY_ID
```

#### `mindmemos skill graph list`

列出当前项目中的图家族。

```bash
mindmemos skill graph list
```

#### `mindmemos skill graph resolve`

把图引用解析为图家族 ID 和所选快照 ID。

```bash
mindmemos skill graph resolve my-collection
```

#### `mindmemos skill graph history`

列出引用所属图家族的快照。

```bash
mindmemos skill graph history my-collection
```

#### `mindmemos skill graph show`

读取引用选中的已发布图快照。

```bash
mindmemos skill graph show my-collection
```

#### `mindmemos skill graph preview`

读取已发布快照的拓扑供预览。此命令只读。

```bash
mindmemos skill graph preview my-collection
```

#### `mindmemos skill graph publish`

发布图快照。用 `--alias` 或 `--graph-id` 指定目标图。成员来源三选一，可重复传入 `--skill`，也可用 `--all-skills` 或 `--empty`。快照固定成员的具体版本，每个家族最多一个版本。

```bash
mindmemos skill graph publish --alias my-collection --skill my-skill
```

可用 `--message` 记录提交说明，或用 `--force-extract-scene` 强制提取 Scene。

#### `mindmemos skill graph add`

将选定的 Skill 加入图并发布新快照。重复 `--skill` 可一次加入多个成员。可附上 `--message`，或用 `--force-extract-scene` 强制提取 Scene。

```bash
mindmemos skill graph add my-collection --skill another-skill
```

#### `mindmemos skill graph remove`

从图中移除指定成员并发布新快照，Skill 家族和版本仍可在仓库中查询。`--skill` 可重复，`--message` 可记录移除原因。

```bash
mindmemos skill graph remove my-collection --skill another-skill
```

### 算法相关

评分和评分查询依赖服务端的 Analyze 流水线。评分命令还需要启用对应的 Judge。返回 `pipeline_unavailable` 时，检查服务端是否启用了该命令所需的流水线。基准评测另见 [Skill 评测指南](../evaluation/skill_evaluation_ZH.md)。

#### `mindmemos skill analyze skill`

对选定版本的 Skill 内容和资源进行静态评分。相同输入和评分配置会复用已完成结果，`--force-refresh` 可重新运行并更新该结果。用 `--request-id` 关联外部追踪。

```bash
mindmemos skill analyze skill my-skill
```

#### `mindmemos skill analyze trajectory`

分析已保存的轨迹。省略归因开关时沿用服务端设置，可用 `--attribute` 或 `--no-attribute` 覆盖。`--verifier-score` 可指定验证分数。支持 `--request-id` 与 `--force-refresh`。

```bash
mindmemos skill analyze trajectory TRAJECTORY_ID --no-attribute --verifier-score 0
```

#### `mindmemos skill evaluations query`

读取一个或多个指定版本的已完成评分。可重复指定 `--judge-type static_skill` 或 `--judge-type trajectory`，省略时返回两类。`--limit` 默认 200，最大 1,000。查询只读取已有结果。

```bash
mindmemos skill evaluations query SKILL_VERSION_ID --judge-type static_skill --limit 20
```

返回的 `items` 中，静态评分记录包含 `skill_version_id`、`score_mean`、`scores`、`summary` 和 `findings`。使用完整版本 ID 可以在切换活动版本后继续查询同一版本。

#### `mindmemos skill evolve`

提交异步演进任务，返回 `task_id`。一次可提交多个 Skill 引用，每个家族最多选一个版本。省略 `--pipeline` 时使用服务端默认流水线，仓库示例中的流水线名为 `offline`。

```bash
mindmemos skill evolve my-skill --pipeline offline
mindmemos skill evolve my-skill another-skill --pipeline offline
```

重复 `--editable-skill` 可指定允许修改的输入，也可用 `--read-only` 保持输入家族不变。用 `--allow-create-skills` 或 `--no-allow-create-skills` 控制新增家族。加 `--auto-accept` 可激活已有家族的候选版本。

#### `mindmemos skill train`

提交异步训练任务，返回 `task_id`。提供服务端配置的 `--pipeline` 和 `--work-dir`，环境用 `--environment` 从本地 YAML 或 JSON 文件读取，也可用 `--environment-json` 内联传入。

```bash
mindmemos skill train my-skill --pipeline TRAINING_PROFILE \
  --work-dir ./runs/train-001 --environment ./environment.yaml \
  --sample-json "$(cat ./train-sample.json)" --train-id SAMPLE_ID
```

将 `TRAINING_PROFILE` 换成服务端的训练配置名，样本文件需包含 `task` 和 `reward_data`，`SAMPLE_ID` 对应样本的 `task.id`。可以重复传入 `--sample-json`、`--train-id`、`--val-id` 和 `--test-id`，各分组的 ID 必须互不重叠。加 `--auto-accept` 可激活训练生成的新版本。

#### `mindmemos skill task show`

按任务 ID 查看状态和结果快照。任务 ID 由 `evolve` 或 `train` 返回。

```bash
mindmemos skill task show TASK_ID
```

#### `mindmemos skill task progress`

输出任务文本日志。用 `--after N` 只读取序号大于 N 的事件，用 `--follow` 或 `-f` 持续读取到任务结束。

```bash
mindmemos skill task progress TASK_ID --follow
```

#### `mindmemos skill graph recommend`

从 `--by` 选定的图快照中推荐与任务相关的 Skill。用完整快照 ID 可固定推荐范围。`--strategy` 支持 `graph` 和 `vector`，默认 `graph`。

```bash
mindmemos skill graph recommend "整理 CSV 数据" --by my-collection --include-evaluation
```

用 `--top-k` 控制结果数量，默认 5。`--include-scenes` 附带 Scene，`--include-evaluation` 附带已有评分。
