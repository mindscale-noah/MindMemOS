# MindMemOS 评测指南

本文集中说明 LoCoMo、LongMemEval、PersonaMem、MemoryAgentBench 和 SpreadsheetBench-Verified 的复现入口，以及如何用 `metrics.py` 统计耗时与 token 用量。

---

## 前置条件

### 1. 启动依赖服务

```bash
make db          # 启动 Qdrant / Neo4j / Kafka / ClickHouse / Grafana / MinIO 等（MINIO=0 跳过 MinIO，纯文本布局专用）
```

> **注意**：若之前执行过 `make db-clean`（会删除所有 volume），务必同时重启 MindMemOS server，否则 server 的旧连接池会导致 search 返回 500。

### 2. 启动 MindMemOS server

```bash
uv run uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### 3. 配置 LLM API Key

评测时，`add` 阶段的 LLM 调用由 **server** 自己完成，使用 `config/mindmemos/dev.yaml` 里的 key；`answer` 和 `judge` 阶段由 **eval 进程**直接调用，使用 `config/mindmemos_eval/memory_evaluation_locomo.example.yaml`（或 `memory_evaluation_personamem.example.yaml`） 里的 key。两套 key 相互独立，需要分别配置。

```yaml
# config/mindmemos_eval/memory_evaluation_locomo.yaml
runner:
  llm:
    model: gpt-4.1-mini
    api_key: <YOUR_KEY>
    base_url: https://your-llm-provider/v1
  answer_llm:
    api_key: <YOUR_KEY>          # 可以和 llm 相同
    base_url: https://your-llm-provider/v1
  judge_llm:
    api_key: <YOUR_KEY>
    base_url: https://your-llm-provider/v1
```

验证 server 和 LLM 是否都可用：

```bash
# server 正常返回 JSON（非 connection refused / 500）
curl -s -X POST http://127.0.0.1:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <任意已有 key>" \
  -d '{"query":"test","user_id":"test"}' | head -c 100

# LLM API 正常返回（非 Invalid token）
curl -s -X POST https://your-llm-provider/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

---

## 各数据集复现

除非特别说明，下面的命令都从仓库根目录执行。LoCoMo、LongMemEval、PersonaMem 和 MemoryAgentBench 的数据集不随仓库分发，需要先从对应 benchmark 的官方渠道获取。

> **API key 文件**：`--api-key-output` 会完整写入一个新的 `api_keys.yaml`，仅包含本次评测生成的身份。请使用独立路径，并让服务端的 `auth.api_key_file` 指向该文件；不要覆盖日常开发环境正在使用的 key 文件。

### LoCoMo

1. 将 LoCoMo 数据集放到示例配置默认读取的位置：

   ```text
   datasets/locomo/locomo10.json
   ```

2. 复制配置并填入 eval 进程使用的 LLM API key：

   ```bash
   cp config/mindmemos_eval/memory_evaluation_locomo.example.yaml \
     config/mindmemos_eval/memory_evaluation_locomo.yaml
   ```

3. 启动服务后运行：

   ```bash
   uv run python -m mindmemos_eval.cli memory \
     --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
     --benchmark-list locomo \
     --algorithm schema \
     --manifest-output reports/locomo_schema.jsonl \
     --api-key-output config/mindmemos/eval_api_keys.yaml
   ```

### LongMemEval

LongMemEval 与 LoCoMo 共用 `memory_evaluation_locomo.example.yaml`。将数据集路径写入复制后的配置，例如：

```yaml
benchmarks:
  longmemeval:
    dataset: resources/memory/dataset/longmemeval_smoke.json
```

运行评测：

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list longmemeval \
  --algorithm vanilla \
  --manifest-output reports/longmemeval_vanilla.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

先用 `--limit 2 --session-limit 2` 做全链路冒烟测试，再移除限制运行完整数据集。

### PersonaMem

1. 准备问题和上下文数据：

   ```text
   resources/memory/dataset/questions_32k.csv
   resources/memory/dataset/shared_contexts_32k.jsonl
   ```

2. 复制配置并填入 eval 进程使用的 LLM API key：

   ```bash
   cp config/mindmemos_eval/memory_evaluation_personamem.example.yaml \
     config/mindmemos_eval/memory_evaluation_personamem.yaml
   ```

3. 启动服务后运行：

   ```bash
   uv run python -m mindmemos_eval.cli memory \
     --benchmark-config config/mindmemos_eval/memory_evaluation_personamem.yaml \
     --benchmark-list personamem \
     --algorithm schema \
     --manifest-output reports/personamem_schema.jsonl \
     --api-key-output config/mindmemos/eval_api_keys.yaml
   ```

### MemoryAgentBench

1. 复制 MemoryAgentBench 配置：

   ```bash
   cp config/mindmemos_eval/dreaming_evaluation_mab.example.yaml \
     config/mindmemos_eval/dreaming_evaluation_mab.yaml
   ```

2. 按配置中的 `benchmarks.memoryagentbench.dataset` 准备数据文件，并确认 `sub_dataset`、`chunk_size` 与目标实验一致。

3. 运行当前分支支持的 Vanilla 基线：

   ```bash
   uv run python -m mindmemos_eval.cli memory \
     --benchmark-config config/mindmemos_eval/dreaming_evaluation_mab.yaml \
     --benchmark-list memoryagentbench \
     --algorithm vanilla \
     --manifest-output reports/memoryagentbench_vanilla.jsonl \
     --api-key-output config/mindmemos/eval_api_keys.yaml
   ```

> **当前边界**：上述命令复现 MemoryAgentBench 的 Vanilla 基线。当前 `develop` 分支的 `memory` CLI 尚未暴露 Add → Dreaming → Answer/Search → Score 的完整开关，因此不能仅凭配置文件名复现主 README 表格中的 Vanilla + Dreaming 行。

### SpreadsheetBench-Verified

SpreadsheetBench 使用独立的 Skill evaluator，首次运行时会自动下载并解压 Verified-400 数据集。先执行两条任务的冒烟测试：

```bash
uv run python -m mindmemos_eval.cli skill spreadsheetbench \
  --data-root data/SpreadsheetBench \
  --download \
  --model gpt-5.4-mini \
  --api-key <YOUR_KEY> \
  --base-url https://your-llm-provider/v1 \
  --limit 2 \
  --concurrency 1 \
  --run-dir results/spreadsheetbench_smoke
```

复现 Init-skill 路径时，加入仓库内置初始 Skill：

```bash
uv run python -m mindmemos_eval.cli skill spreadsheetbench \
  --data-root data/SpreadsheetBench \
  --skill resources/skill_evolve/spreadsheetbench_init_skill/xlsx \
  --model gpt-5.4-mini \
  --api-key <YOUR_KEY> \
  --base-url https://your-llm-provider/v1 \
  --concurrency 1 \
  --run-dir results/spreadsheetbench_init
```

连接 MindMemOS Skill Evolution 服务时，再加入：

```text
--evolve
--evolve-every 1
--evolution-base-url http://127.0.0.1:8000
--evolution-api-key <MINDMEMOS_API_KEY>
```

No-skill 基线不传 `--skill`；Init-skill 传入初始 Skill，但不传 `--evolve`。

> **当前边界**：当前 CLI 可以运行 No-skill、Init-skill 和在线 evolution 路径，但没有单独暴露主 README 表格中 Unsup./Sup. 两种实验协议的完整选择参数。对外报告结果时还需记录实际模型、并发、随机种子和重复运行次数。

---

## 多模态：MemEye / Mem-Gallery（`vanilla`，多模态模式 off / basic / enhanced）

两个 benchmark 都是"多轮对话中穿插图片"的长期记忆测试，评测 `vanilla` 的多模态能力。流程：

1. 每个 session 调用一次 `memory.add`。
2. 每道题调用一次 `memory.search`；如果题目自带图片，作为 `query_images` 一并传入。
3. 检索返回的文本记忆和图片，加上题目图片，交给同一个支持看图的作答模型（`answer_llm`）作答；数据集自带的图片描述不会发送给记忆系统。
4. 由 `judge_llm` 按 benchmark 原版的评分标准打分。

| | MemEye | Mem-Gallery |
|---|---|---|
| 规模 | 8 个任务，MCQ + OPEN，共 742 题 | 20 个任务，1711 道 OPEN 题 |
| 数据 | Hugging Face `MemEyeBench/MemEye`（约 260 MB） | Hugging Face `Ethan-Bei/Mem-Gallery`（约 530 MB） |
| 指标 | EM（MCQ 选项轮换去偏）、B-1、Judge（MemEye 评分标准），按任务宏平均 | F1、B-1、Judge（原版评分标准），全部题目微平均 |

每个 benchmark 跑四组配置。**模式通过评测项目的 `project_override_config` 设置，服务端全局的 `mode` 保持默认的 `"off"` 即可**（YAML 中要写带引号的 `"off"`，否则会被解析成布尔值）：

| 配置 | `algo_config.multimodal.mode` | 检索策略 | 说明 |
|---|---|---|---|
| `b0` | `off` | fast | 纯文本 vanilla，不发送图片 |
| `b1` | `basic` | fast | 存原图，按图片相似度检索 |
| `enhanced` | `enhanced` | fast | VLM 抽取 + 视觉事实 + 证据图 |
| `enhanced_agentic` | `enhanced` | agentic | 多模态 agent 检索 |

多模态模式的配置见 [docs/config/README.md](../config/README.md) 的 `algo_config.multimodal` 一节。

### 1. 前置条件

- **机器：** 16 GB 以上内存，磁盘预留 10 GB（数据集、原图存储、模型）。GPU 可选；本地的 SigLIP2 embedding 服务在 CUDA、Apple Silicon（MPS）或 CPU 上都能跑，CPU 会慢一些。
- **软件：** Docker（用于 Qdrant、Neo4j、Kafka 等依赖）、[uv](https://docs.astral.sh/uv/)、Python 3.13，安装方式见[主 README](../../README_ZH.md)。
- **一个 OpenAI 兼容的 LLM 服务，需要提供：**
  - 能看图的 chat 模型，例如 `gpt-4.1-mini`。服务端的 enhanced 抽取、agent 和评测的作答、judge 都用它；
  - 文本 embedding 模型（vanilla 本来就需要）；
  - reranker（推荐配置；不配时视觉事实不做重排）。
- **多模态 embedding 服务：** 把文本和图片编码到同一个向量空间。可以用仓库自带的本地 SigLIP2 服务（见第 3 步），也可以换成自己的服务，只要兼容 OpenAI 协议。

### 2. 下载数据集

```bash
uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download as get
get('MemEyeBench/MemEye', repo_type='dataset', local_dir='datasets/memeye')
get('Ethan-Bei/Mem-Gallery', repo_type='dataset', local_dir='datasets/memgallery')
"
```

下载后的目录结构如下，评测配置里的 `dataset` 字段指向各自的 `data/` 目录：

```
datasets/
├── memeye/data/        dialog/*.json   image/<任务>/...
└── memgallery/data/    dialog/*.json   image/<任务>/...
```

- `datasets/` 已加入 gitignore。
- 如果数据放在别处，可以从 `datasets/memeye`、`datasets/memgallery` 建软链接到实际位置。
- 网络受限时，可以设置 `HF_ENDPOINT` 使用 Hugging Face 镜像。

### 3. 启动本地多模态 embedding 服务（SigLIP2）

```bash
uv run --no-project --with torch --with transformers --with pillow \
  python scripts/mm_embed_server.py --port 8090
```

- 首次启动会从 Hugging Face 下载 `google/siglip2-base-patch16-384`（约 1.5 GB）。
- 设备自动选择：CUDA → MPS → CPU；也可以用 `--device cpu` 指定。
- 检查服务是否正常：

  ```bash
  curl localhost:8090/health
  # {"status": "ok", "model": "siglip2-base-patch16-384", "dimensions": 768}
  ```

这个脚本只用于开发和评测，torch 不是项目依赖。生产环境请提供自己的多模态 embedding API。

### 4. 服务端配置并启动

```bash
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml
```

在 `config/mindmemos/dev.yaml` 中做以下修改：

1. 按[主 README 的"准备配置文件"一节](../../README_ZH.md)填好 `chat_model_router`、`embed_model_router`、`rerank_model_router`。chat 模型必须支持图片输入。
2. 配置多模态 embedding：

   ```yaml
   mm_embed_model_router:
     endpoints:
       - model: openai/siglip2-base-patch16-384
         api_base: http://127.0.0.1:8090/v1
         dimensions: 768
   ```

3. 允许服务端按路径读取数据集里的图片。评测以文件路径方式发送图片，这里必须写**绝对路径**，并且是软链接解析之后的真实路径：

   ```yaml
   blob_store:
     allowed_path_roots:
       - /绝对路径/MindMemOS/datasets      # 或数据集的真实所在目录；原图统一存 S3/MinIO（s3_endpoint 等）
   ```

4. 让服务端使用评测生成的 key 文件。评测时会自动写入这个文件，服务端会自动重新加载：

   ```yaml
   auth:
     api_key_file: eval_api_keys.yaml
   ```

5. `algo_config.multimodal.mode` 保持 `"off"`（评测用 `project_override_config` 按项目开启）。

启动服务：

```bash
make dev   # 启动 Docker 依赖和 API，地址 http://127.0.0.1:8000
```

服务端不会热加载配置和代码，修改 `dev.yaml` 后需要重启。

### 5. 评测配置

```bash
cp config/mindmemos_eval/memory_evaluation_multimodal.example.yaml config/mindmemos_eval/memory_evaluation_multimodal.yaml
cp config/mindmemos_eval/memory_evaluation_memgallery.example.yaml config/mindmemos_eval/memory_evaluation_memgallery.yaml
```

在两个文件中给 `runner.llm`、`runner.answer_llm`、`runner.judge_llm` 填上 `model`、`base_url`、`api_key`。作答模型必须支持看图；三者可以是同一个模型。其余字段保持默认即可——模式、检索策略和是否发送图片由下一步的脚本自动生成，不用手改。

可选项（`execution_params.<benchmark>`）：`max_answer_images`（附给作答器的图片上限，默认 8）、`image_max_side`（附图前缩放到的最长边，默认 1024）。

### 6. 一键运行（推荐）

先做冒烟测试：每个 benchmark 只跑 1 个任务、每阶段 2 题，四组配置都跑一遍，确认链路通畅。每组配置仍要先写入一个完整任务的对话，所以 8 次冒烟合计约 45 分钟：

```bash
uv run python scripts/run_multimodal_bench.py --smoke
```

然后正式运行：

```bash
uv run python scripts/run_multimodal_bench.py                 # 两个 benchmark × 四组配置，各 1 次
uv run python scripts/run_multimodal_bench.py --runs 2        # 每组 2 次（推荐，用于比较）
uv run python scripts/run_multimodal_bench.py --benchmarks memgallery --configs enhanced enhanced_agentic
```

脚本的行为：

- **启动前检查：** 服务是否可访问、数据集目录是否存在、`allowed_path_roots` 是否覆盖数据集、多模态 embedding 是否已配置、`auth.api_key_file` 与评测 key 文件是否一致。有问题会直接提示并退出。
- **严格串行。** 任一次运行有作答或 judge 调用失败（结果无效）时立即停止，不再消耗后面的预算。
- **断点续跑：** 重新执行同一条命令时，已有有效结果的运行会自动跳过。
- **输出位置：** `reports/multimodal/<tag>/` 下每次运行的 manifest（`<bench>_<config>_r<k>.jsonl`）、日志、生成的配置，以及汇总表 `summary.md`（`<tag>` 默认为日期，冒烟测试为 `smoke`）；逐题明细在 `runs/mindmemos_eval/<run_id>/<benchmark>/`。

**耗时：**两个 benchmark 的四组配置各跑一次，合计约 7 小时（单次运行，gpt-4.1-mini；耗时受 LLM 服务延迟影响很大，`enhanced_agentic` 最慢）。

### 7. 手动跑单个配置

MemEye（8 个任务，MCQ + OPEN，对话中含图片）使用同一个作答器：`answer_llm`（需支持看图）只看到检索返回的文本记忆和图片，以及题目自带的图片。算法固定为 `vanilla`；由 profile 的 `project_override_config` 设置 `algo_config.multimodal.mode`，`search_params.multimodal: true` 让每次检索都使用已存的图片。`execution_params.memeye.send_images: true` 时，对话图片和题目图片以路径方式发送，因此服务端 `blob_store.allowed_path_roots` 需要覆盖数据集目录；`off` 模式会拒绝图片，所以其基线设 `send_images: false`。数据集自带的图片描述不会发送给记忆系统。评测配置在装载时会拒绝未知的 `search_params` / `execution_params` 键（各 benchmark 的合法键见 `src/mindmemos_eval/mindmemos_eval/memory/config.py`）；`send_images: true` 配上 off 模式项目同样在装载期即报一致性错误——冲突在发出任何请求前就会暴露，而不是跑到一半才出现 `add.images_unsupported`。

1. 从 Hugging Face 下载 `MemEyeBench/MemEye`，链接到 `datasets/memeye/data`（包含 `dialog/` 和 `image/`）。
2. 跑 `basic` / `enhanced` 模式时，在服务端配置 `mm_embed_model_router`（见 `docs/config/README.md`）。
3. 复制 `config/mindmemos_eval/memory_evaluation_multimodal.example.yaml` 为 `memory_evaluation_multimodal.yaml`，填好 LLM key 并设置模式（示例为 `enhanced`；B0 用 `"off"` 并设 `send_images: false`，B1 用 `basic`；加 `search_strategy: agentic` 使用多模态 agent），然后运行：

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_multimodal.yaml \
  --benchmark-list memeye --algorithm vanilla \
  --manifest-output reports/memeye_enhanced.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml \
  --server-config config/mindmemos/dev.yaml
```

`--server-config` 让项目重置找到服务端的集合名与 S3/MinIO blob 配置，重置时把该项目的已存图片对象连同其他记录一起删除。

`eval_result.metrics` 给出百分制指标：`EM`（MCQ 四种选项轮换的平均精确匹配，按任务宏平均）、`B-1`（OPEN 的 BLEU-1）、`Judge`（OPEN，MemEye 的 0–1 评分标准），以及逐任务数值。`eval_result.cost` 记录客户端的写入/检索调用次数与延迟、作答器和 judge 的 token 用量；只要有作答或 judge 调用失败，`eval_result.valid` 即为 false。逐题明细写在 `output_root/<run_id>/memeye/`。冒烟测试可设 `execution_params.memeye.max_questions: 2`、`benchmarks.memeye.tasks: [Brand_Memory_Test]`。

多模态评测遵循与其他 benchmark 相同的 runner 协议：检索默认参数（`top_k`、`rerank`、`search_strategy`）来自 runner 配置 / CLI（`--top-k`、`--rerank`、`--search-strategy`），benchmark/profile 的 `search_params` 仍可覆盖；`--max-search-concurrency` 限制并发检索。`--judge-runs N` 对每道 OPEN 题独立跑 N 次 judge 并取平均。`metrics.overall_accuracy` 沿用 locomo 的 0–1 约定（MemEye 取 `EM`、Mem-Gallery 取 `F1` 作为 headline），作答/judge 的 token 字段也沿用 locomo 命名，因此下面的指标汇总流程同样适用于多模态 manifest。

### 8. 结果解读

manifest 中的 `eval_result` 包含：

- `metrics`：百分制总分、按任务（`per_task`）的分数，Mem-Gallery 另有按题型（`per_point`）的分数；
- `cost`：客户端的写入和检索调用次数与延迟、作答器和 judge 的 token；
- `valid`：只要有作答或 judge 调用失败就为 false，这样的结果不可用。

### 9. 参考结果

评测配置：`vanilla` 算法；所有行均以 `gpt-4.1-mini` 作答（作答器需支持看图）；图片 embedding 为本地 SigLIP2（`siglip2-base-patch16-384`）。MindMemOS 各行为本仓库实现的评测（作答与 judge 均为 `gpt-4.1-mini`，2 次均值，2026-10）；基线行为论文在同一实验条件下、同作答器的复现（3 次均值；出处为双盲评审中的在投论文，暂不附链接）。两组 Judge 模型不同（本节 `gpt-4.1-mini`、基线 `GPT-5.4-mini`），Judge 分数请各组内比较、勿跨组对比。

| 方法 | MemEye EM | MemEye B-1 | MemEye Judge | Mem-Gallery F1 | Mem-Gallery B-1 | Mem-Gallery Judge |
|---|---|---|---|---|---|---|
| MemoryOS | 45.75 | 13.12 | 34.91 | 43.09 | 37.76 | 64.64 |
| A-MEM | 49.73 | 21.11 | 38.27 | 44.08 | 38.37 | 64.35 |
| SimpleMem | 50.34 | 9.42 | 42.59 | 34.73 | 28.90 | 58.33 |
| HyperMem | 47.82 | 19.64 | 36.43 | 44.57 | 33.42 | 66.76 |
| M2A | 52.31 | 6.31 | 44.50 | 15.04 | 10.28 | 51.81 |
| MMA | 50.96 | 16.32 | 43.81 | 26.99 | 20.60 | 60.87 |
| SRAG（多模态语义检索） | 58.36 | 31.48 | 50.54 | 41.07 | 35.03 | 67.06 |
| UniversalRAG | 60.49 | 11.23 | 52.61 | 48.36 | 41.90 | 70.63 |
| MindMemOS b0（mode off，纯文本） | 43.03 | 20.32 | 26.26 | 38.72 | 32.59 | 70.68 |
| MindMemOS b1（basic） | 59.77 | 27.02 | 47.80 | 54.73 | 47.75 | 85.06 |
| MindMemOS enhanced | 63.45 | 25.49 | 50.80 | 55.66 | 48.98 | 87.06 |
| MindMemOS enhanced_agentic | **65.89** | 27.09 | **53.30** | **57.06** | **50.45** | **89.14** |

**比较结果时注意：**

- **MemEye 波动大：** OPEN 题少，单次 Judge 波动约 2–3 分，单个任务可达 10–20 分；并且会随 LLM 服务随时间漂移——同一份代码隔两天重跑，enhanced 的 Judge 从 52.8 降到 50.2、agentic 的 EM 从 67.8 降到 65.5。比较两个版本时，每个配置至少跑 2 次，并在同一时期重跑基线。
- **Mem-Gallery 较稳：** 单次 Judge 波动小于 0.6。
- **不要同时跑两条评测：** 共用同一个 LLM 服务时，并发负载会把 MemEye 的 Judge 拉低约 5 分。上面的数字都是串行跑出来的。
- **VR 计数题：** Mem-Gallery 作答器会给图片结果、事实和视觉事实都标上 image_id（VS 题需要用 image_id 作答）。快速检索模式下，作答器有时会按重复出现的 id 计数，导致 VR 计数题多数。

### 10. 试过但无效、未合入的方案

| 方案 | 结果 | 原因 |
|---|---|---|
| 视觉事实前加图片标识，并让 agent 只保留 5 条未选中的事实 | 快速检索 Judge −2.8，agentic EM −1.3 | reranker 会匹配标识，把视觉事实集中到少数几张图上 |
| 先用 VLM 描述查询图片，再拼进查询 | 两个数据集上带图的题只 +0.2～+0.4 | 作答器直接看题图，以图搜图也已经能找到对应存图，空间很小 |
| 视觉事实前加图片名（对话中对这张图的叫法） | MemEye agentic EM −2.1；Mem-Gallery 快速检索略降、agentic +0.4 | 名字常是"从另一侧看的同一个房间"这类相对说法，事实也被集中到少数几张图上 |

### 11. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `add.images_unsupported` / `search.images_unsupported` | 该项目的 `mode` 是 `off`。b0 本来就不发图片；其他配置出现说明 `project_override_config` 没生效，确认用的是脚本生成的配置 |
| `image.path_denied` / `image.path_disabled` | `blob_store.allowed_path_roots` 没覆盖数据集的真实路径（注意软链接）或为空 |
| 冒烟测试卡在 add 或很慢 | enhanced 写入要调用 VLM，单次 add 需要十几到几十秒，属正常；检查 embedding 服务和 LLM 服务的延迟 |
| 结果 `valid: false` | 作答或 judge 调用失败，多半是 LLM 服务限流或不稳定。服务恢复后重新执行同一条命令，失败的那次会重跑 |
| 更换多模态 embedding 模型后报维度不一致 | 图片集合按 `dimensions` 建立。换模型时需要删除 Qdrant 中的 `image_item_v1` 集合（或改用新的 `database.qdrant.image_collection` 名称），并重启服务 |
| 存储增长 | 每次评测使用新的项目 id，原图按项目前缀存进 MinIO bucket，记录留在 Qdrant 和 Neo4j 中。同一项目下次重跑会先清掉；`make db-clean` 会清空所有数据库卷（含 MinIO），请谨慎使用 |
| `multimodal_agentic_failed` 警告 | agent 的某次 LLM 调用失败，本次搜索退回快速检索结果，不影响运行有效性；偶发属正常 |

---

## 快速冒烟测试（推荐先跑）

用 `--limit 2` 把数据量砍到最小，验证全链路是否跑通：

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/smoke_vanilla.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --limit 2 \
  --session-limit 2 \
  --judge-runs 1
```

> **警告**：``--api-key-output`` 会**完整写入**一个新的 ``api_keys.yaml`` 文件，
> 仅包含本次生成的 benchmark 身份。请勿直接指向服务端正在使用的
> ``config/mindmemos/api_keys.yaml``（除非是隔离环境）。
> 建议使用独立路径（如 ``config/mindmemos/eval_api_keys.yaml``），
> 然后将服务端配置指向该文件，或手动将生成的 key 合并到服务端文件中。

---

## 正式评测

### 跑全量 benchmark

三个 benchmark 放在一条命令里串行执行（保证 `api_keys.yaml` 不冲突）：

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/vanilla_run.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --judge-runs 1
```

> **add 前清库**：add 阶段开始前，eval 进程会按本次 run 的 `project_id` 直接清空 Qdrant、
> Neo4j 以及 S3/MinIO blob 存储中该项目的已存图片对象，确保每次 run 从干净状态开始（每次
> run 都有独立的 `project_id`）。这要求 eval 进程能访问 Qdrant、Neo4j 和 S3/MinIO 端点；
> 用与服务端相同的环境变量配置（`MINDMEMOS_QDRANT_URL`、`MINDMEMOS_NEO4J_URI`、
> `MINDMEMOS_NEO4J_USERNAME`、`MINDMEMOS_NEO4J_PASSWORD`、`MINDMEMOS_NEO4J_DATABASE`、
> `MINDMEMOS_S3_*`），或用 `--qdrant-url` / `--neo4j-*` 参数（S3 配置取自 `--server-config`
> 的 `blob_store.s3_*`；blob 存储不可达会直接中止评测）。`--no-add` 跳过清库、
> 复用已有数据；`--skip-clean` 跳过清库但仍执行 add。

### 只跑部分 benchmark

```bash
# 只跑 LoCoMo 和 LongMemEval
--benchmark-list locomo,longmemeval
```

### 复用已入库的数据（跳过 add 阶段）

重新评测**同一 project** 上一次已入库的数据时，组合使用 ``--reuse-api-key``（复用上次的
api_key/project_id）和 ``--no-add``（跳过 add 阶段**以及** add 前的清库，保留下来的数据可直接复用）。
``--reuse-api-key`` 一次只复用一个 project，因此只传一个 benchmark：

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo \
  --manifest-output reports/vanilla_run_retry.jsonl \
  --reuse-api-key config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --no-add \
  --judge-runs 1
```

> 搭配 ``--reuse-api-key`` 时**不需要** ``--api-key-output``（不会生成新身份）。要复用另一个
> benchmark 的数据，把 ``--reuse-api-key`` 指向含该 benchmark key 的文件，并单独传该 benchmark。

### 跳过 judge 阶段（只要答案不打分）

```bash
--no-score
```

### 调整并发

```bash
--max-conv-concurrency 5    # 同时处理的 conversation/sample 数
--max-qa-concurrency 10     # 每个 conversation 内同时处理的问题数
--max-search-concurrency 10
--max-score-concurrency 10
```

---

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `--algorithm` | 算法 profile，当前支持 `vanilla` / `schema` |
| `--limit N` | 每个 benchmark 最多处理 N 条（locomo=N 个对话，longmemeval=N 个 sample，personamem=N 道题） |
| `--session-limit N` | LongMemEval 每个 sample 最多 add 前 N 个 session，用于加速冒烟测试 |
| `--judge-runs N` | 每道题独立跑 N 次 judge，取多数票；默认 1。这 N 次是**串行**跑的，judge 阶段耗时基本随 N 线性增长。PersonaMem 不适用（确定性判分）；MemEye / Mem-Gallery 的 judge 分数是连续值，N 次结果取**平均**而非多数票 |
| `--no-add` | 跳过 add 阶段**以及** add 前的清库；复用该 project 已在 Qdrant/Neo4j 中的数据 |
| `--no-score` | 跳过 judge 阶段 |
| `--manifest-output` | 评测结果写入的 JSONL 文件，供后续 metrics 统计使用 |
| `--api-key-output` | **全新写入** API key 文件，仅含本次生成的 key。勿直接覆盖服务端正在使用的文件。fresh run 必填；搭配 `--reuse-api-key` 时可选（不使用） |
| `--reuse-api-key` | 复用已有 API key 文件（即上次的 api_key/project_id）；配合 `--no-add` 重新评测其数据。一次只能复用一个 benchmark |
| `--skip-clean` | 跳过 add 前清空该 run 的 `project_id`（Qdrant/Neo4j）。`--no-add` 从不清库；本参数用于跳过清库但仍执行 add |

---

## 统计耗时与 Token

评测跑完后，执行以下命令生成统计报告：

```bash
uv run python -m mindmemos_eval.memory.metrics \
  --manifest reports/vanilla_run.jsonl \
  --output reports/vanilla_run_metrics.jsonl \
  --xlsx-output reports/vanilla_run_metrics.xlsx \
  --json-output reports/vanilla_run_metrics_sheets.json
```

`--output` 写的是每个 run 一行的原始数据（manifest 内容 + Qdrant/ClickHouse 查询结果），不是下面这四个整理好的 sheet。想要程序化读取下面四个 sheet 的内容，用 `--json-output`——它和 xlsx 的四个 sheet 内容完全一致，只是换成 `{"summary": [...], "eval_metrics": [...], "llm_by_task": [...], "percentiles": [...]}` 这样的 JSON 结构（每个 sheet 一个数组，数组元素是以列名为 key 的字典）。

生成的 xlsx（或 `--json-output`）包含四个 sheet：

### summary sheet

每个 benchmark 一行，包含：

| 列 | 来源 | 说明 |
|----|------|------|
| `add_count` / `add_total_ms` / `add_avg_ms` | Qdrant | add 请求数和耗时 |
| `search_count` / `search_total_ms` / `search_avg_ms` | Qdrant | search 请求数和耗时 |
| `memory_count` | Qdrant | 最终存储的 memory 条数 |
| `llm_calls` / `llm_total_tokens` / `llm_prompt_tokens` | ClickHouse | server 侧 LLM 调用（add 阶段的 chunk/extract） |
| `search_total_tokens` / `search_avg_tokens/query` / `search_avg_tokens/call` | ClickHouse | search 阶段 token 总量，按 `TraceId` 聚合到"每次 search 请求"级别（不是每次 LLM call）——`SearchResult` 不回传 per-call token，这里的数字都是跑完之后从 ClickHouse OTel trace 重新算出来的 |
| `overall_accuracy` | manifest | 评测得分 |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_total_tokens` | manifest | eval 侧 answer 阶段 token |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_total_tokens` | manifest | eval 侧 judge 阶段 token |
| `add_token_avg` / `answer_token_avg` / `judge_token_avg` | ClickHouse / manifest | 平均每次 add 调用 / 每道题 answer / 每道题 judge 的 token 数 |

### eval_metrics sheet

每个 benchmark 一行，包含：

| 列 | 来源 | 说明 |
|----|------|------|
| `overall_accuracy` / `correct` / `total` | manifest | 整体准确率和题数 |
| `search_llm_calls` / `search_prompt_tokens` / `search_completion_tokens` / `search_total_tokens` | manifest，缺失时回退 ClickHouse 逐 query 聚合 | search 阶段 token（`vanilla`/`fast` 两边都是 0） |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_completion_tokens` / `answer_total_tokens` | manifest | answer 阶段 token |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_completion_tokens` / `judge_total_tokens` | manifest | judge 阶段 token |
| `build_elapsed_seconds` / `search_elapsed_seconds` / `answer_elapsed_seconds` / `total_elapsed_seconds` | manifest | 各阶段累计耗时（秒） |
| `by_question_type` / `by_topic` | manifest | 按题型 / 主题拆分的准确率 |

### llm_by_task sheet

按 LLM 任务类型分行，包含：

| 列 | 来源 | 说明 |
|----|------|------|
| `memory.add.extract` | ClickHouse | server 侧 add 时的 LLM 抽取调用 |
| `eval.answer` | manifest | eval 侧 answer 生成调用 |
| `eval.judge` | manifest | eval 侧 judge 打分调用 |

### percentiles sheet

每个操作的 token 和耗时分位数（min / max / p50 / p95，没有均值列——均值只在 summary sheet）：

| 列 | 来源 | 说明 |
|----|------|------|
| `add` token 分位数 | ClickHouse（逐次 add 调用） | |
| `add` time 分位数 | Qdrant（逐请求） | |
| `search` token 分位数 | ClickHouse（逐次 search **请求**，把一次 search 内部所有 `search.*` LLM call 按 `TraceId` 加总后再算分位数） | 反映真实的单次 search 成本，不是单次 LLM call 的成本 |
| `search` time 分位数 | Qdrant（逐请求） | |
| `eval.answer` token 分位数 | manifest（逐题记录） | 真实分位数 |
| `eval.answer` time 分位数 | manifest（仅有总耗时） | `time_p50_ms` = 平均值，非真实分位数 |
| `eval.judge` token 分位数 | manifest（逐题记录） | 真实分位数 |
| `eval.judge` time 分位数 | 无逐题时间 | `time_p50_ms` 为空 |

> PersonaMem 使用确定性评分，无 answer/judge LLM 调用，percentiles sheet 中不产生 `eval.*` 行。
> `vanilla`/`fast` search 完全不调用 LLM，所以这类 run 的 `search` token 分位数是空的；只有 `agentic`/`schema`（一次 query 会触发多次 LLM call）才会填上数据。
