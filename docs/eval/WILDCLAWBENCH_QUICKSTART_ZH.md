# WildClawBench + MindMemOS 快速上手

前提：`wildclawbench-mindmemos:v1.3` 镜像已经构建过一次，MindMemOS 基础设施（Qdrant/Kafka，通过 `docker compose -f dockers/docker-compose.memory.yml up -d` 启动）已经能起。本文档回答一个问题：**服务和镜像已经就绪时，怎么验证、怎么打通评测流程**。

**本文档的目标是打通"MindMemOS × WildClawBench"这条评测链路（容器能跑、记忆能写、能出分数），不是把每个评测参数都调到最优。** 涉及模型选择、搜索能力、judge 打分质量等跟官方默认不一致的地方，第 1.5 节会列出来，但具体要不要对齐官方配置、用什么 key，由使用者自己判断和修改，不属于本文档负责的范围。

### 路径变量约定

本文档所有命令都用两个变量代替具体路径，不同人机器上的实际路径不同：

| 变量 | 含义 | 怎么拿到 |
|---|---|---|
| `$MINDMEMOS_REPO` | 本仓库（MindMemOS）的本地克隆路径 | 你 `git clone` 这个仓库时选择的目录；不确定的话在仓库任意位置执行 `git rev-parse --show-toplevel` 可以打印出来 |
| `$WILDCLAWBENCH_REPO` | WildClawBench 上游仓库的本地克隆路径（独立仓库，与本仓库无关） | 见下方"$WILDCLAWBENCH_REPO 具体说明" |

### `$WILDCLAWBENCH_REPO` 具体说明

- **仓库地址**：`https://github.com/InternLM/WildClawBench.git`（上游官方仓库，不是本仓库的一部分）。
- **如果还没克隆过**，任选一个本地目录克隆：

  ```bash
  git clone https://github.com/InternLM/WildClawBench.git /path/to/your/WildClawBench
  ```

  克隆完成后这个目录本身就是 `$WILDCLAWBENCH_REPO`。

- **如果已经克隆过、不确定路径在哪**：先找到之前存放它的目录，`cd` 进去后执行：

  ```bash
  git remote get-url origin
  # 期望输出：https://github.com/InternLM/WildClawBench.git（或对应的 SSH 形式）
  git rev-parse --show-toplevel
  # 打印出的路径就是 $WILDCLAWBENCH_REPO
  ```

  两条都对得上（remote 指向 InternLM/WildClawBench，且目录是仓库根目录）才能确认拿对了目录，避免和别的仓库搞混。

- **`git clone` 只拿到任务定义和评测脚本（`tasks/*.md`、`script/run.sh` 等），不包含评测数据**，评测数据（图片/视频/代码仓库等原始素材，供任务运行时挂载进容器）单独放在 HuggingFace 上，必须额外下载，且**必须放进 `$WILDCLAWBENCH_REPO/workspace/` 这个固定位置**——不是随便放哪都行，`script/run.sh` 是按 `workspace/<category>/<task_id>/...` 这个相对路径去挂载容器目录的，位置不对任务会直接跑不起来：

  ```bash
  cd "$WILDCLAWBENCH_REPO"
  hf download internlm/WildClawBench workspace --repo-type dataset --local-dir .
  ```

  `--local-dir .` 是关键，保证下载出来的 `workspace/` 目录正好落在 `$WILDCLAWBENCH_REPO/workspace/`，跟 `tasks/`、`script/` 平级。数据量较大（11GB+），下载慢/网络不稳时 `hf download` 支持断点续传，重跑同一条命令即可，不用担心重复下载浪费流量。

  **顺序很重要**：先 `git clone` 仓库、`cd` 进去，再下载 `workspace/`——不要下载到别的目录再手动挪，容易挪错层级导致挂载路径对不上。**镜像（`wildclawbench-mindmemos:v1.3`）里不包含 `tasks/`/`workspace/` 任何一样**，两者是完全独立的东西：镜像负责"怎么跑"，`git clone` + `workspace/` 数据负责"跑什么"，只传镜像给别人是不够的，对方必须自己 `git clone` + 下载 `workspace/`。

- **额外确认这是"数据已就绪"的那个目录**（而不是空克隆）。下面几条是可选自检，后续评测不依赖它们，只是帮你提前发现"拿错目录 / 数据没下全"：

  ```bash
  find "$WILDCLAWBENCH_REPO"/tasks -name "*task_*.md" | wc -l   # 期望 60（60 个任务全在）
  test -f "$WILDCLAWBENCH_REPO"/script/run.sh && echo "run.sh 存在"   # 确认是对的仓库
  ls "$WILDCLAWBENCH_REPO"/workspace | wc -l   # 期望 6，和 tasks/ 下类别数一致，确认评测数据也下全了
  ```

实际使用时，把这两个变量 export 出来（或者在每条命令前用 `MINDMEMOS_REPO=... WILDCLAWBENCH_REPO=...` 显式赋值），下面所有命令直接照抄即可：

```bash
export MINDMEMOS_REPO=/path/to/your/MindMemOS       # 换成你自己的路径
export WILDCLAWBENCH_REPO=/path/to/your/WildClawBench   # 换成你自己的路径
```

---

## 1. 服务 + 镜像已就绪：验证与执行

流程顺序是：**验证服务/镜像（1.1-1.2）→ 生成 key（1.3）→ 同步镜像代码 + 用新 key 认证（1.4）→ 检查配置差异（1.5）→ 冒烟测试（1.6）→ 正式跑评测（1.7）→ 看结果（1.8）**。1.4/1.5 放在冒烟测试之前，是因为冒烟测试跑的就是这份镜像和这份 `.env` 配置，没提前对齐会导致冒烟测试本身跑出误导性的结果（比如误以为是记忆链路的问题，其实是 key 没配）。

### 1.1 验证 MindMemOS API 在跑

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
curl -s http://localhost:8001/health
```

看到进程存在、接口有响应即可。如果没起，用下面的方式启动（宿主机进程，不是容器；这条命令同时会启动 Kafka 异步 worker，不需要额外起 worker 进程）：

```bash
cd "$MINDMEMOS_REPO"
make API_HOST=0.0.0.0 API_PORT=8001 api
```

### 1.2 验证评测镜像可用

```bash
docker image inspect wildclawbench-mindmemos:v1.3 >/dev/null && echo "image ok"
```

如果这条报错说镜像不存在，**不代表你必须从零构建**——先问问团队里有没有人已经构建过这个镜像，让对方把镜像传给你，比自己从零编译快得多：

- **对方导出、你导入**（最简单，不需要额外基础设施）：

  ```bash
  # 对方那边执行
  docker save wildclawbench-mindmemos:v1.3 -o wildclawbench-mindmemos-v1.3.tar
  # 把这个 tar 传给你（scp / 网盘 / 内部文件服务器，文件较大，14GB+）

  # 你这边执行
  docker load -i wildclawbench-mindmemos-v1.3.tar
  ```

- **如果团队有私有镜像仓库**，对方 `docker push` 后你 `docker pull` 再 `docker tag` 回 `wildclawbench-mindmemos:v1.3` 即可，多人长期用更省事。

  不要把这个镜像传到公开仓库/HuggingFace——里面固化了 SDK/插件源码，也可能带有别人的认证信息。

**镜像导入后，第一件事是换成你自己的 key，不能直接拿别人认证过的镜像跑评测**（否则你写的记忆数据会进到对方的 project 里，数据不隔离）。做法：先起好你自己机器上的 MindMemOS 基础设施 + API（本节下面的检查/启动步骤），走一遍第 1.3/1.4 步生成并认证你自己的 key，`mindmemos auth` 会自动覆盖镜像里已有的认证配置。之后就跟镜像是自己从零构建的完全一样，正常往下走即可。

只有当团队里真的没人构建过这个镜像时，才需要走"从零搭建"那一套完整流程（安装 MindMemOS SDK、安装 OpenClaw 插件、认证、`docker commit` 固化），这部分工作量较大，不属于本文档范围。

### 1.3 生成本次评测专用的 key（每次评测都做，用于数据隔离）

```bash
cd "$MINDMEMOS_REPO"

python3 scripts/wildclawbench/new_key.py \
  --benchmark wildclawbench \
  --memory-algorithm schema \
  --from-memory-eval-profile schema \
  --disable-previous
```

终端会打印三行，记下其中的 `api_key` 和 `project_id`：

```
key_id:     key_wildclawbench_schema_20260706_xxxxxx_xxxxxxxx
api_key:    dev-api-key-wildclawbench-schema-20260706-xxxxxx-xxxxxxxx
project_id: proj_wildclawbench_schema_20260706_xxxxxx_xxxxxxxx
```

**`--memory-algorithm` / `--from-memory-eval-profile` 想评测哪种记忆算法就填哪个值**，两处要填成一样的：

- `--memory-algorithm` 是自由字符串，只用来拼 `key_id`/`api_key`/`project_id` 的命名（比如上面示例里 project 名带 `schema` 字样），本身不影响实际跑的算法逻辑。
- `--from-memory-eval-profile` 决定真正注入哪套算法参数（读取 `config/mindmemos_eval/memory_evaluation.yaml` 里 `algorithm_profiles` 下对应名字的 `project_override_config`），这个才是实际生效的开关。

当前 `algorithm_profiles` 里有两个可用值：`schema`（结构化 schema 抽取/合并算法）和 `vanilla`（不做 schema 抽取的基线算法，用于对比）。想跑基线对比，把两处都换成 `vanilla`：

```bash
python3 scripts/wildclawbench/new_key.py \
  --benchmark wildclawbench \
  --memory-algorithm vanilla \
  --from-memory-eval-profile vanilla \
  --disable-previous
```

两处不一致（比如一个填 `vanilla` 一个填 `schema`）不会报错，但 project 名字和实际跑的算法会对不上、容易自己搞混，务必保持一致。

### 1.4 让镜像和当前代码对齐、并用新 key 认证（推荐：一条命令搞定）

镜像里装了两样自研代码：`mindmemos_sdk`（纯 HTTP 客户端）和 OpenClaw 插件（`plugins/openclaw-plugin`）；真正跑记忆算法（schema 抽取、合并）的代码在 `src/mindmemos`，由 `make api` **直接从源码启动**，不经过镜像。所以每次评测前有两件事要做：①让服务端算法用最新代码，②让镜像里的 SDK/插件用最新代码、并用本次的新 key 重新认证。不用判断"代码到底改没改"，下面两步一律照做即可，**即使没改，重来一遍也只是白做，不会出错**。

**第 1 步：重启 API，让服务端算法用最新代码（覆盖 `src/mindmemos` 的任何改动）**

```bash
# 停掉旧 API（PID 用 lsof 查到的那个）
lsof -nP -iTCP:8001 -sTCP:LISTEN
kill <PID>

# 用当前源码重新启动（这条命令直接读源码，天然就是最新的）
cd "$MINDMEMOS_REPO"
make API_HOST=0.0.0.0 API_PORT=8001 api
```

**第 2 步：同步镜像 + 认证（一条命令，覆盖 SDK/插件改动 + 用新 key 认证 + 固化）**

用第 1.3 步生成的 `api_key`，跑一条命令：

```bash
cd "$MINDMEMOS_REPO"
bash scripts/wildclawbench/sync_image.sh --api-key <第 1.3 步的 api_key>
```

这个脚本自动做完：本地重新编译插件 → 起临时容器 → 用当前源码重装 SDK（离线，不联网）→ 覆盖插件构建产物 → **用你的 key 重新 `auth` + `doctor` 验证** → 备份当前镜像 tag → `docker commit` 固化回 `wildclawbench-mindmemos:v1.3-brave-yibu` → 成功后自动删除备份、清理临时容器。看到最后打印 `OK: ... is now synced ...` 就说明镜像已经和当前代码一致、且认证好了。

> 这条脚本已经把"认证并固化回镜像"这件事包含在内了（第 5 步的 `mindmemos auth` + `doctor`），所以用了它就**不需要**再手动做下面 1.4-备选那套认证流程。它是幂等的：代码没改时重复跑一遍也安全，只多花一两分钟编译。commit 前会自动打备份 tag，只有整个脚本成功跑完才删除备份；中途失败会留着备份供手动回滚。

> **搜索使用 Yibu Brave-compatible provider。** 首次从基础 `wildclawbench-mindmemos:v1.3` 构建搜索镜像时，安装独立插件 `resources/memory/wildclawbench/openclaw-brave-yibu-plugin`（id=`brave-yibu`，装在容器里的 `~/.openclaw/extensions/brave-yibu/`）：
>
> ```bash
> cd "$MINDMEMOS_REPO"
> bash scripts/wildclawbench/install_brave_yibu_plugin.sh
> ```
>
> 脚本默认调用 `https://yibuapi.com/brave/v1/web/search`，把 provider 固定为 `brave-yibu`，并生成 `wildclawbench-mindmemos:v1.3-brave-yibu`。之后每次评测前直接运行上面的 `sync_image.sh`，它默认同步并认证这个 Brave 镜像。运行时 `BRAVE_API_KEY` 必须是 Yibu Brave-compatible key。实现与配置说明见 [resources/memory/wildclawbench/openclaw-brave-yibu-plugin/README.md](../../resources/memory/wildclawbench/openclaw-brave-yibu-plugin/README.md)。

#### 1.4-备选：只手动认证、不同步代码（确定 SDK/插件没变时用）

如果你**很确定**镜像里的 SDK 和插件代码跟当前 clone 完全一致（没改过、也没人改过），只是想换个 key 认证，可以跳过上面的 `sync_image.sh`，用下面这套更快的手动认证（省掉编译）。**两者二选一，不要都做**——上面用了 `sync_image.sh` 就直接跳过这一段。

```bash
docker run -dit --name wildclaw-auth wildclawbench-mindmemos:v1.3-brave-yibu sleep infinity

docker exec -it wildclaw-auth mindmemos auth \
  --base-url http://host.docker.internal:8001 \
  --api-key <上一步的 api_key> \
  --user-id wildclawbench

docker exec -it wildclaw-auth mindmemos doctor
# 期望：base_url: http://host.docker.internal:8001 / api_key: configured / transport: ready

docker commit wildclaw-auth wildclawbench-mindmemos:v1.3-brave-yibu
docker rm -f wildclaw-auth
```

### 1.5 检查配置差异（跑评测前先看一眼，不属于本文档负责范围，自行决定要不要改）

跑之前建议对照一遍 `$WILDCLAWBENCH_REPO/.env`（跟仓库自带的 `.env.example` 对比）。**这里只是列出当前配置和官方默认不一致的地方，供你判断哪些是有意为之的适配、哪些需要自己补上——具体要不要对齐官方、用谁的 key，由你自己决定和修改，本文档不负责这部分配置。**

**以下是当前实际状态**（`OPENROUTER_API_KEY`/`BRAVE_API_KEY` 两个坑之前踩过、现在已经补上，这里刷新成最新情况）：

| 配置项 | 官方默认 | 目前的配置 | 影响 |
|---|---|---|---|
| **Agent 模型** | `OPENROUTER_API_KEY` + OpenRouter 上的模型（如 `openrouter/openai/gpt-5.5`） | 自定义 `my_api.json` 里的 `custom` provider（`baseUrl` 指向私有兼容端点，走 `MY_PROXY_API_KEY`） | 正常支持的用法（README 里的"Custom Model Endpoint"路径），agent 执行任务时调用的模型没问题 |
| **`OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` / `JUDGE_MODEL`**（judge 打分用） | `OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'`、`JUDGE_MODEL=openai/gpt-5.4`、真实 OpenRouter key | **已改成指向私有兼容端点**：`JUDGE_MODEL=gpt-4.1-mini`、key 复用 `MY_PROXY_API_KEY` 对应的私有 endpoint key | 已用真实请求验证通过（`curl` 直接测过 `/v1/chat/completions` 返回正常），judge 现在能正常打分，`score.json` 里出现真实 `judge_reason` 而不是 `judge_error` |
| **`BRAVE_API_KEY`**（Search 类任务用） | 真实 Brave Search API key | Yibu Brave-compatible key | `04_Search_Retrieval` 已完成全量验证，MiniMax M2.7 得分 31.8% |
| **代理变量** `HTTP_PROXY_INNER`/`HTTPS_PROXY_INNER`/`NO_PROXY_INNER` | 默认空 | 按需配成 `http://host.docker.internal:<代理端口>` | 环境特定配置（本机需要代理才能访问外部模型 API 时才要配），不是坑，是必要适配 |
| **`DOCKER_IMAGE`**（`.env` 里的默认值） | `wildclawbench-ubuntu:v1.3`（官方未认证的基础镜像） | `run_serial.sh` 默认使用 `wildclawbench-mindmemos:v1.3-brave-yibu` | 全量评测统一使用已认证且启用 Yibu Brave 搜索的镜像 |

**`my_api.json` 的 provider 名必须和 `--model` 前缀一致。**本文档所有命令统一使用 `--model custom/gpt-4.1-mini`，所以 `$WILDCLAWBENCH_REPO/my_api.json` 里也必须有 `providers.custom.models[]`。真实 endpoint 和 key 留在本地配置里，不提交到仓库：

```json
{
  "providers": {
    "custom": {
      "baseUrl": "https://<private-compatible-endpoint>/v1",
      "apiKey": "${MY_PROXY_API_KEY}",
      "api": "openai-completions",
      "models": [
        {
          "id": "gpt-4.1-mini",
          "name": "GPT-4.1 Mini"
        }
      ]
    }
  }
}
```

如果这里还叫别的名字，但命令里传 `custom/gpt-4.1-mini`，OpenClaw 会报 `Unknown model: custom/gpt-4.1-mini`，agent 会很快退出、产物为空、分数基本都是 0。

**当前 `.env` 配置**（key/URL 均已脱敏，仅供对照结构，实际值以 `$WILDCLAWBENCH_REPO/.env` 为准）：

```
DOCKER_IMAGE=wildclawbench-ubuntu:v1.3
GATEWAY_PORT=18789
TMP_WORKSPACE=/tmp_workspace

TASKS_SUBDIR=tasks
OUTPUT_SUBDIR=output

DEFAULT_MODEL=openrouter/xxx

DEFAULT_PARALLEL=1
HTTP_PROXY_INNER=http://***MASKED***
HTTPS_PROXY_INNER=http://***MASKED***
NO_PROXY_INNER=host.docker.internal,127.0.0.1,localhost

OPENROUTER_BASE_URL='https://***MASKED***'
JUDGE_MODEL=gpt-4.1-mini
OPENROUTER_API_KEY=sk-hpcR***MASKED***

BRAVE_API_KEY=sk-hpcR***MASKED***
MY_PROXY_API_KEY=sk-hpcR***MASKED***
# Using a Custom Model Endpoint
# MY_PROXY_API_KEY=

# Lobster profile env keys (add values here for skills that need them)
# GEMINI_API_KEY=
# FIRECRAWL_API_KEY=
# EXA_API_KEY=

# Task env
# 01_Productivity_Flow
# 02_Code_Intelligence
# 03_Social_Interaction
# 04_Search_Retrieval
# 05_Creative_Synthesis
# 06_Safety_Alignment
```

`BRAVE_API_KEY` 必须配置为 Yibu Brave-compatible key；模型和 judge 使用的 key 按各自 endpoint 单独配置。

### 1.6 冒烟测试（先跑一个任务，别直接上全量）

```bash
cd "$WILDCLAWBENCH_REPO"

DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu \
bash script/run.sh openclaw \
  --models-config my_api.json \
  --task tasks/06_Safety_Alignment/06_Safety_Alignment_task_1_file_overwrite.md \
  --model custom/gpt-4.1-mini

python3 "$MINDMEMOS_REPO/scripts/wildclawbench/wait_drain.py" \
  --project-id <上一步的 project_id>
```

看到 `no pending add_record, safe to continue` 说明链路通、记忆写完了，可以往下跑。如果一直卡到超时（默认 180 秒），先检查 MindMemOS API 进程和 Qdrant/Kafka 是否还活着，不要跳过这一步直接跑全量。

**冒烟测试跑完之后不要直接原地开始正式评测**，中间要处理一件事：

- 冒烟测试用的 `project_id` 现在已经不是"干净"的了——`06_Safety_Alignment_task_1_file_overwrite` 这个任务的记忆已经写进去了。MindMemOS 里同一个 project 的记忆是跨任务累积的（这是设计如此，任务 2 本来就该看到任务 1 写的记忆），但这意味着**如果接下来正式评测的范围会再次覆盖到 `06_Safety_Alignment` 这个类别（跑全量必然会），这个类别对应的任务会带着"冒烟测试期间已经写过一次"的记忆重新开始，不是真正意义上的干净起点，跟其他从未跑过的类别不可比**。
- **处理方式**：冒烟测试只是为了验证链路通不通，不代表这条数据能拿去当正式评测结果用。正式评测（不管是 1.7 的单类别验证还是全量）开始前，重新走一遍 **1.3（生成新 key）+ 1.4（用新 key 同步并认证镜像）**，拿到一个全新的 `project_id`，下面 1.7 的命令里都用这个新的。

### 1.7 跑评测

WildClawBench 官方给的全量命令是 `bash script/run.sh openclaw --category all --parallel 4 ...`，**这里不能直接用**：60 个任务共用同一个 MindMemOS project，且记忆写入是异步的（`add` 调用立刻返回 `queued`，真正落盘由后台 Kafka worker 完成）；并行跑或不等排空，会导致任务之间的记忆边界串扰。本仓库提供了串行 + 排空的 wrapper 脚本，请用它代替官方命令。

**先明确一件事：单类别验证和全量之间的关系，决定要不要再换一次 `project_id`。**

- 如果单类别验证跑完、看分数正常之后，你打算接着跑全量——**单类别验证跟冒烟测试性质一样，只是"确认这个类别没问题"，不是正式数据**。因为全量必然会把这个类别再跑一遍，用同一个 `project_id` 会导致这个类别的任务带着验证阶段已经写过的记忆重新开始，跟其他类别不可比。**跑全量之前，重新走一遍 1.3 + 1.4，换一个全新的 `project_id`。**
- 如果单类别验证本身就是你这次想要的最终结果（比如你只关心 `03_Social_Interaction` 这一个类别，不打算跑全量）——那这次跑就是"正式评测"，不需要再额外换 project；但注意它和前面冒烟测试用的是不是同一个 `project_id`：如果同一个，冒烟测试跑的是 `06_Safety_Alignment` 类别，跟 `03_Social_Interaction` 不冲突，可以放心复用。

也就是说：**每一次你真正想拿分数当结果看的评测（不管范围大小），开始前先确认这个 `project_id` 此前有没有跑过这次要跑的任务分类；有重叠就重新走 1.3 + 1.4 换新的。**

`run_serial.sh` 默认使用 `wildclawbench-mindmemos:v1.3-brave-yibu`；需要临时验证其他镜像时仍可通过 `DOCKER_IMAGE` 覆盖：

```bash
# 单类别（先验证）
WILDCLAWBENCH_DIR="$WILDCLAWBENCH_REPO" \
MINDMEMOS_PROJECT_ID=<project_id> \
DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu \
bash "$MINDMEMOS_REPO/scripts/wildclawbench/run_serial.sh" \
  --category 03_Social_Interaction \
  --models-config my_api.json \
  --model custom/gpt-4.1-mini

# 全量 60 个任务（换新 project_id 之后再跑）
WILDCLAWBENCH_DIR="$WILDCLAWBENCH_REPO" \
MINDMEMOS_PROJECT_ID=<新的 project_id> \
DRAIN_TIMEOUT=300 \
DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu \
bash "$MINDMEMOS_REPO/scripts/wildclawbench/run_serial.sh" \
  --category all \
  --models-config my_api.json \
  --model custom/gpt-4.1-mini
```

这个脚本做的事：遍历 `tasks/` 下的分类（或指定分类），对每个任务依次执行 `script/run.sh openclaw --task <file>`，每跑完一个就调用排空脚本确认这个 project 没有 `queued`/`processing` 状态的记忆写入了，再开始下一个。全程严格串行，比官方并行写法慢很多（60 个任务总耗时以小时计），这是保证记忆边界干净的必要代价，不要为了赶时间调大并行度。

**单个任务失败不会中断整轮**：基准里有些任务模型本来就会失败（`run.sh` 返回非零、`score.json` 里带 `error` 字段），脚本会把它记下来、继续跑下一个，全部跑完后在结尾打印一个失败任务汇总（`==================== run complete ====================` 那段）。所以看到某个任务报错、分数 0 是正常的，不代表整轮挂了——只有真的全部 60 个跑完才会停。

**跑全量之前，建议再顺手确认这几件事**（避免跑几个小时之后才发现问题，白跑）：

1. **单类别验证的分数是不是正常的**：看几个 `score.json`，不是清一色 0 分、也没有"Model setup failed"这种链路层面的报错（这类问题不会因为多跑几个任务就自己好，先查清楚原因再继续，参考 1.4/1.5 排查代码同步和配置差异）。
2. **清理上一轮验证残留的容器**，避免和新一轮混在一起：
   ```bash
   docker ps -a --filter "ancestor=wildclawbench-mindmemos:v1.3-brave-yibu" -q | xargs -r docker rm -f
   ```
3. **1.5 节的配置差异想清楚了没**：`OPENROUTER_API_KEY`/`BRAVE_API_KEY` 要不要补，全量耗时以小时计，带着已知的判分缺陷跑完再发现就是白跑。
4. **磁盘空间是否够**：全量 60 个任务的 `output/` 会包含完整对话轨迹、日志、产物文件，建议预留至少 20-30GB。
5. **防止电脑中途休眠 + 防终端关闭**：全量耗时长，`caffeinate` 防止系统睡眠，`nohup ... &` 让它后台跑、终端断开也不受影响。用下面这条即可（`caffeinate` 建议保留；`nohup` 后台跑基本够用）：
   ```bash
   caffeinate -i -s nohup env \
     WILDCLAWBENCH_DIR="$WILDCLAWBENCH_REPO" \
     MINDMEMOS_PROJECT_ID=<新的 project_id> \
     DRAIN_TIMEOUT=300 \
     DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu \
     bash "$MINDMEMOS_REPO/scripts/wildclawbench/run_serial.sh" \
       --category all \
       --models-config my_api.json \
       --model custom/gpt-4.1-mini \
     > ~/wildclawbench_run.log 2>&1 &
   ```

   > 说明：之前出现过"跑到一半莫名中断"，真实原因是脚本的 `set -e` bug（某个任务失败就静默杀掉整轮），**已修复**，跟休眠/终端无关。所以 `nohup` + `caffeinate` 现在就够了。如果你不放心，或者要跑很久且担心手滑关掉终端窗口，可以再套一层 `tmux`（`brew install tmux` 后 `tmux new -s wildclawbench`，在里面跑，`Ctrl-B` 再按 `D` 安全退出，`tmux attach -t wildclawbench` 回来看）——但这只是额外保险，不是必须。

### 1.8 看结果、清理

- 结果（相对于 `$WILDCLAWBENCH_REPO`）：`output/openclaw/<category>/<task_id>/<model_timestamp_runid>/score.json`（各项指标）、`usage.json`（token/耗时/花费）、`chat.jsonl`（完整对话轨迹）。
- **没有全局汇总文件**：WildClawBench 官方在 `--category` 批量模式下跑完会生成 `output/summary_all_<model_name>.json`，但 [wildclawbench/run_serial.sh](../../scripts/wildclawbench/run_serial.sh) 为了在任务之间插入排空等待，内部是逐个任务用 `--task <file>` 单任务模式调用的，不会触发这个汇总生成逻辑——不管跑单个类别还是 `--category all` 全量，都不会有这个文件，这是 wrapper 脚本的固有行为，不是运行出错。想看汇总自己写脚本聚合 `score.json`：

  ```bash
  python3 - <<'EOF'
  import json, glob

  files = sorted(glob.glob(
      "$WILDCLAWBENCH_REPO/output/openclaw/<category>/*/*/score.json"
  ))

  rows = []
  for f in files:
      d = json.load(open(f))
      rows.append((f.split("/")[-3], d.get("overall_score")))

  for task_id, score in rows:
      print(f"{task_id}: {score}")

  scores = [s for _, s in rows if s is not None]
  if scores:
      print(f"\n平均分: {sum(scores)/len(scores):.4f}  (共 {len(scores)} 个任务)")
  EOF
  ```

  把 `$WILDCLAWBENCH_REPO` 和 `<category>` 换成实际值（`glob` 不认 shell 变量，脚本里要写死或用 Python 的 `os.environ` 读）。

- **清理残留容器**：

  ```bash
  docker ps -a --filter "ancestor=wildclawbench-mindmemos:v1.3-brave-yibu" -q | xargs -r docker rm -f
  ```

  这条命令找出所有用 `wildclawbench-mindmemos:v1.3-brave-yibu` 启动过的容器（不管是不是已经停了）强制删掉。**正常情况不需要每次跑完都手动执行**——`script/run.sh` 每跑完一个任务本身就会自动清理那个任务的容器（日志里能看到 `Container cleaned up`）。这条命令是给"异常中断"场景用的：评测中途 `Ctrl-C`、电脑意外重启/休眠、脚本被 `kill`、任务在链路层面报错退出（比如插件加载失败）导致自动清理没走完——这些情况可能留下没清干净的容器。合理用法是：**评测被打断过、或不确定容器有没有清干净时跑一下**，跑之前先用 `docker ps -a --filter "ancestor=wildclawbench-mindmemos:v1.3-brave-yibu"` 看一眼有没有残留，没有就不用管。

- **`output/` 目录会持续变大，要主动清理，不然空间迟早不够**：

  `output/` 下大部分任务只有几百 KB（`chat.jsonl`/`score.json`/`usage.json` 都是文本），但少数任务要处理 PDF、图片这类二进制产物，agent 执行过程中产生的文件会被整个拷贝进 `task_output/` 存档，单个任务可能达到几百 MB。60 个任务里只要有几个这种"重"任务，累积起来就是空间大头，这是文档前面建议预留 20-30GB 的原因。

  **更关键的是空间会一直涨**：目录结构是 `output/openclaw/<category>/<task_id>/<model>_<时间戳>_<runid>/`，时间戳保证了每一次跑（冒烟测试、单类别验证、换 project 全量、重跑某个任务）都会新建一个目录，跟之前的并存，不会自动覆盖或清理旧的。按前面 1.6/1.7 建议的流程走下来，同一个任务很可能已经积累了好几份历史记录——真正让空间不够的往往不是单次全量本身，而是**历史跑批不断叠加**。

  **删除时机建议**：

  1. **失效的跑批随时可以删，不用等**：比如链路层面报错（"Model setup failed" 之类）、或者 project 被污染（同一个 project 里混了不该混在一起的任务）产生的数据，本来就不能当结果用，找到后直接删掉对应的 `<model>_<时间戳>_<runid>` 目录。
  2. **冒烟测试 / 单类别验证的数据，等"正式"那次跑完确认没问题后就可以删**——按 1.6/1.7 的说明，它们本来就只是用来验证链路通不通，不是要保留的最终结果。
  3. **正式评测的数据（要拿去分析、写结论的那份）**：`score.json`/`usage.json`/`chat.jsonl` 体积小（加起来通常几百 KB 到几 MB），建议优先保留；`task_output/` 是占空间的大头，只是留着方便排查问题用，确认分数没问题之后可以先清它，其他小文件继续留着：

     ```bash
     # 找出最占空间的 task_output/，心里有数再决定删哪些
     find "$WILDCLAWBENCH_REPO/output/openclaw" -type d -name task_output -exec du -sh {} \; | sort -rh | head -20

     # 确认没问题后，针对某次具体跑批只删 task_output/，保留 score.json/usage.json/chat.jsonl
     rm -rf "$WILDCLAWBENCH_REPO/output/openclaw/<category>/<task_id>/<model_时间戳_runid>/task_output"
     ```

  4. **整批删除某次失效/过期跑批**：

     ```bash
     rm -rf "$WILDCLAWBENCH_REPO/output/openclaw/<category>/<task_id>/<model_时间戳_runid>"
     ```
