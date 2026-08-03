# WildClawBench + MindMemOS 从零搭建文档

本文档假设你拿到的是一台全新机器，本地既没有 WildClawBench 的镜像和数据，也没有跑起来的 MindMemOS 服务。目标是走完之后，得到：

- 本地已启动的 MindMemOS 基础设施（Qdrant / Neo4j / Kafka / ClickHouse / Grafana）；
- 本地已启动、监听 `0.0.0.0:8001` 的 MindMemOS API；
- 一个独立、干净的 WildClawBench 专用 `project_id`；
- 一个自建的、验证通过的 `wildclawbench-mindmemos:v1.3` 评测镜像；
- WildClawBench 的全部任务数据（图片/视频/代码仓库等）已就绪。

搭建完成后，具体怎么"跑评测"见配套文档 [WILDCLAWBENCH_USER_GUIDE_ZH.md](WILDCLAWBENCH_USER_GUIDE_ZH.md)。

本文档与仓库里已有的 [WILDCLAWBENCH_MINDMEMOS_ZH.md](WILDCLAWBENCH_MINDMEMOS_ZH.md) 是互补关系：那份文档记录的是"当前这台机器"已经装好之后的排障笔记和最短路径命令；这份文档补的是"从零开始"缺的两块——**下载镜像/数据**、**搭建全量评测所需的额外配置**。

## 0. 前置依赖

宿主机（本文档以 macOS + Docker Desktop 为例）需要：

```bash
# Docker
brew install --cask docker
open -a Docker

# HuggingFace CLI（下载镜像和数据用）
pip install -U "huggingface_hub[cli]"

# WildClawBench 数据准备脚本依赖
brew install yt-dlp ffmpeg
pip install gdown

# 构建 OpenClaw 插件用
brew install node   # 需要 npm
```

MindMemOS 本身（`uv`、Python 3.10+）按仓库 [README.md](/Users/chenliang/code/new/0701/MindMemOS/README.md) 里的常规开发环境搭建即可，这里不重复。

## 1. 克隆两个仓库

```bash
# MindMemOS（本仓库，假设已在 /Users/chenliang/code/new/0701/MindMemOS）
git clone <mindmemos-repo-url> /Users/chenliang/code/new/0701/MindMemOS

# WildClawBench（上游仓库，独立 git remote，不要把它的文件当成本仓库的一部分提交）
git clone https://github.com/InternLM/WildClawBench.git /Users/chenliang/WildClawBench
```

## 2. 下载 WildClawBench 镜像和任务数据

WildClawBench 提供四个 harness 各自的基础镜像，全部托管在 HuggingFace。本方案只用 OpenClaw harness，所以只需要下载 `wildclawbench-ubuntu_v1.3.tar`：

```bash
cd /Users/chenliang/WildClawBench

hf download internlm/WildClawBench Images/wildclawbench-ubuntu_v1.3.tar \
  --repo-type dataset --local-dir .

docker load -i Images/wildclawbench-ubuntu_v1.3.tar
# 确认加载成功
docker images | grep wildclawbench-ubuntu
```

下载任务数据（视频、图片、代码仓库等原始素材）：

```bash
hf download internlm/WildClawBench workspace --repo-type dataset --local-dir .
```

跑数据准备脚本（下载 YouTube 视频、解压 Safety Alignment 任务用的 git 历史、下载 SAM3 权重等）：

```bash
bash script/prepare.sh
```

> 注意：YouTube 下载可能触发"Sign in to confirm you're not a bot"。参考 README 里的三种规避方式（导出 cookies.txt / `--cookies-from-browser` / 安装 Deno）。这一步经常是"数据没准备好"报错的根因，不要跳过。

准备完成后确认：

```bash
find /Users/chenliang/WildClawBench/tasks -mindepth 2 -maxdepth 2 -name '*.md' | wc -l
# 期望 60
```

## 3. 启动 MindMemOS 基础设施

```bash
cd /Users/chenliang/code/new/0701/MindMemOS
docker compose -f dockers/docker-compose.memory.yml up -d
```

这会拉起 Qdrant（6333/6334）、Neo4j（7474/7687）、Kafka（9092）、ClickHouse+OTel+Grafana（可观测性，非必需但建议一起起来方便排障）。

确认健康：

```bash
docker compose -f dockers/docker-compose.memory.yml ps
curl -s http://localhost:6333/collections | head
```

## 4. 配置独立的 WildClawBench API Key

MindMemOS 按 `api_key -> project_id` 隔离记忆。如果 WildClawBench 和其他评测（比如 LoCoMo）共用一个 key，历史记忆会被串进来，必须用独立、空白的 project。

推荐不要再手写 `api_keys.yaml`。正式评测前，直接用脚本生成一条全新的 WildClawBench key，让它像 LoCoMo 一样把 `api_key` / `project_id` / `project_override_config` 一次性写入 [api_keys.yaml](/Users/chenliang/code/new/0701/MindMemOS/config/mindmemos/api_keys.yaml)。

如果你想沿用 `config/mindmemos_eval/memory_evaluation.yaml` 里和 LoCoMo 一样的 `schema` 算法参数，直接运行：

```bash
cd /Users/chenliang/code/new/0701/MindMemOS

python3 scripts/wildclawbench/new_key.py \
  --benchmark wildclawbench \
  --memory-algorithm schema \
  --from-memory-eval-profile schema \
  --disable-previous
```

这条命令会自动：

- 生成新的 `key_id`
- 生成新的 `api_key`
- 生成新的 `project_id`
- 把 `memory_evaluation.yaml -> algorithm_profiles.schema.project_override_config` 写进新 key
- 将旧的 `key_wildclawbench_*` 标记为 `enabled: false`

注意：这表示会**完整复用当前 `memory_evaluation.yaml` 里的 `schema` profile**，包括其中的 `entity_modeling_path: config/presets/entity_modeling_locomo_schema.json`。如果你的目标就是“先按 LoCoMo 同款 schema 参数跑一版 WildClawBench”，这正是你想要的行为；如果你后面想做 WildClawBench 专用调参，再改成单独的 override 文件。

如果你明确想测试“默认参数，不带 override”的表现，就去掉 `--from-memory-eval-profile schema`：

```bash
python3 scripts/wildclawbench/new_key.py \
  --benchmark wildclawbench \
  --memory-algorithm schema \
  --disable-previous
```

如果你想用一份单独维护的 WildClawBench override 文件，而不是直接复用 `memory_evaluation.yaml`，也可以：

```bash
python3 scripts/wildclawbench/new_key.py \
  --benchmark wildclawbench \
  --memory-algorithm schema \
  --project-override-config config/presets/project_override_wildclawbench_schema.example.yaml \
  --disable-previous
```

做全量（60 任务）正式评测时，建议在开跑前就定好这件事，不要中途改 `project_override_config`，否则同一批结果前后算法参数不一致，后面很难解释。

生成完新 key 后，重启 API（见第 5 步）即可生效，不需要重建 Qdrant/Neo4j 数据。

## 5. 启动 MindMemOS API

```bash
cd /Users/chenliang/code/new/0701/MindMemOS
make API_HOST=0.0.0.0 API_PORT=8001 api
```

关键点：

- 必须 `0.0.0.0`，否则容器内 `host.docker.internal` 访问不到；
- 用 `8001` 而不是默认 `8000`，避免和本机已绑定 `127.0.0.1:8000` 的其他进程冲突；
- 这个进程同时也是 Kafka 异步 worker 的宿主——`mindmemos.api.app` 的 `lifespan` 里会调用 `register_workers()` 并启动 Kafka 消费者（`memory-add-worker` 等），**不需要额外起一个独立 worker 进程**。这也意味着：只要这个 API 进程在跑，OpenClaw 插件 `addMode: async` 提交的记忆最终都会被处理完，只是有延迟（这个延迟正是第 6 步之后要处理的"串行保证"问题的来源，见 [WILDCLAWBENCH_USER_GUIDE_ZH.md](WILDCLAWBENCH_USER_GUIDE_ZH.md)）。

## 6. 从零构建评测镜像 `wildclawbench-mindmemos:v1.3`

**不要**直接在 Dockerfile 里 `pip install mindmemos-sdk` 硬编（仓库里 `WildClawBench/docker/mindmemos/Dockerfile` 就是这种写法，装的是不存在的 `mindmemos` PyPI 包和未发布的 `@mindmemos/openclaw-plugin` npm 包，会直接失败，不要用它）。已验证可行的做法是"临时容器里装好、验证、再 commit"。

### 6.1 确认两处源码修复已存在

- SDK 支持 Python 3.10：[pyproject.toml](/Users/chenliang/code/new/0701/MindMemOS/src/mindmemos_sdk/pyproject.toml) 中 `requires-python = ">=3.10,<3.14"`（基础镜像是 Python 3.10.12，SDK 原先要求 3.11 会直接装不上）。
- OpenClaw 插件运行时入口指向构建产物：[package.json](/Users/chenliang/code/new/0701/MindMemOS/plugins/openclaw-plugin/package.json) 中 `runtimeExtensions` 指向 `./dist/index.js`（`extensions` 指向源码 `./src/index.ts`，是 OpenClaw 官方约定的双入口写法：运行时优先加载 `runtimeExtensions` 的构建产物，`extensions` 仅用于 workspace/git checkout 本地开发）。

### 6.2 构建本地插件

```bash
cd /Users/chenliang/code/new/0701/MindMemOS/plugins/openclaw-plugin
npm install
npm run build
```

### 6.3 启动临时开发容器

```bash
docker run -it --name wildclaw-dev \
  --platform=linux/amd64 \
  -v /Users/chenliang/code/new/0701/MindMemOS:/workspace/MindMemOS \
  wildclawbench-ubuntu:v1.3 bash
```

先在另一个终端里保持这个容器运行，后续用 `docker exec` 操作它。

### 6.4 容器内安装 MindMemOS SDK

```bash
docker exec -it \
  -e HTTP_PROXY=http://host.docker.internal:7890 \
  -e HTTPS_PROXY=http://host.docker.internal:7890 \
  -e http_proxy=http://host.docker.internal:7890 \
  -e https_proxy=http://host.docker.internal:7890 \
  -e PIP_CONFIG_FILE=/dev/null \
  -e PIP_INDEX_URL=https://pypi.org/simple \
  wildclaw-dev bash -lc '
python3 -m pip install -U "uv-build>=0.11.7,<0.12.0" &&
python3 -m pip install \
  --index-url https://pypi.org/simple \
  --no-build-isolation \
  /workspace/MindMemOS/src/mindmemos_sdk
'
```

`PIP_CONFIG_FILE=/dev/null` 和显式 `PIP_INDEX_URL` 是关键：容器/宿主机可能残留了错误的旧代理配置（例如 `100.104.*`），不清空会一直连不上。

### 6.5 安装本地 OpenClaw 插件并配置

如果没有可用的 `BRAVE_API_KEY`，先禁用 Brave provider，否则 gateway 可能直接起不来（依赖联网搜索的任务会因此降分，见第 7 步的取舍说明）。

```bash
docker exec -it wildclaw-dev openclaw plugins install /workspace/MindMemOS/plugins/openclaw-plugin

docker exec -it wildclaw-dev openclaw config set plugins.entries.mindmemos-memory.hooks.allowConversationAccess true
docker exec -it wildclaw-dev openclaw config set plugins.entries.mindmemos-memory.config.addMode async
docker exec -it wildclaw-dev openclaw config set plugins.entries.mindmemos-memory.config.appId wildclawbench
docker exec -it wildclaw-dev openclaw config set plugins.entries.mindmemos-memory.config.cli /usr/local/bin/mindmemos
```

### 6.6 用新生成的独立 Key 认证

```bash
docker exec -it wildclaw-dev mindmemos auth \
  --base-url http://host.docker.internal:8001 \
  --api-key <上一步脚本输出的新 api_key> \
  --user-id wildclawbench
```

### 6.7 验证

```bash
docker exec -it wildclaw-dev mindmemos doctor
# 期望: config: ok / base_url: http://host.docker.internal:8001 / api_key: configured / transport: ready

docker exec -it wildclaw-dev mindmemos memory search "connectivity test" --top-k 1 --json
# 首次独立 project 期望: {"memories":[]}
```

`{"memories":[]}` 同时证明了三件事：API 通了、认证通了、project 是干净的（没有混入 LoCoMo 等历史记忆）。

### 6.8 提交镜像

```bash
docker commit wildclaw-dev wildclawbench-mindmemos:v1.3
docker rm -f wildclaw-dev   # 临时容器用完即可清理
```

如果容器里残留了错误代理环境变量，提交前先清掉，或 `docker commit --change 'ENV http_proxy='` 之类显式覆盖成空。

### 6.9 安装 Yibu Brave 搜索 Provider

基础镜像构建完成后，生成正式用于评测的搜索镜像：

```bash
cd "$MINDMEMOS_REPO"
bash scripts/wildclawbench/install_brave_yibu_plugin.sh
```

脚本默认从 `wildclawbench-mindmemos:v1.3` 构建
`wildclawbench-mindmemos:v1.3-brave-yibu`，安装并启用 `brave-yibu`，并把
`tools.web.search.provider` 固定为 `brave-yibu`。运行评测时，WildClawBench
的 `.env` 必须提供有效的 Yibu Brave-compatible `BRAVE_API_KEY`。

## 7. 全量评测前需要额外确认的配置项

这些是"单任务能跑通"和"能放心跑满 60 个任务"之间的差距，逐条对照：

1. **`BRAVE_API_KEY`**：04_Search_Retrieval 类别（11 个任务）依赖联网搜索。必须提供有效的 Yibu Brave-compatible key；正式评测统一使用第 6.9 步生成的 `wildclawbench-mindmemos:v1.3-brave-yibu` 镜像。
2. **`OPENROUTER_API_KEY` / 自定义模型 endpoint**：如果不走 OpenRouter，需要按 `my_api.json` 格式配置 provider，并在 `.env` 里设置对应的 key。文档命令使用 `--model custom/gpt-4.1-mini`，所以 `my_api.json` 里的 provider key 必须叫 `custom`，真实 endpoint 只放在本地 `baseUrl` 里，不提交到仓库。
3. **`.env` 里的代理变量**：`HTTP_PROXY_INNER` / `HTTPS_PROXY_INNER` / `NO_PROXY_INNER`，必须包含 `host.docker.internal,127.0.0.1,localhost`，否则容器访问宿主机 API 会被代理绕开或超时。
4. **`DEFAULT_PARALLEL`**：`.env` 里默认是 `1`。全量评测**不要**为了求快调大这个值——原因和串行保证有关，见 [WILDCLAWBENCH_USER_GUIDE_ZH.md](WILDCLAWBENCH_USER_GUIDE_ZH.md) 的详细说明。
5. **磁盘空间**：60 个任务的 `output/` 会包含完整对话轨迹、agent 日志、`chat.jsonl`、生成的文件产物，加上第 2 步下载的视频/权重素材，建议预留至少 20-30GB。
6. **第 6 步末尾提到的 `project_override_config` 缺口**：如果决定要补，现在（全量评测开始前）改比中途改更安全，避免同一批评测里前后算法参数不一致。
7. **容器残留清理**：如果之前有中断的运行，先清理残留容器再开始全量评测：

   ```bash
   docker ps -a --filter "ancestor=wildclawbench-mindmemos:v1.3-brave-yibu" -q | xargs -r docker rm -f
   ```

完成以上七点之后，环境即可视为"全量评测就绪"。具体怎么发起评测、如何保证任务间的记忆写入不串扰、怎么看结果，见 [WILDCLAWBENCH_USER_GUIDE_ZH.md](WILDCLAWBENCH_USER_GUIDE_ZH.md)。
