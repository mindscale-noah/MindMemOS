# 多模态 Benchmark 运行指南（MemEye / Mem-Gallery）

本文说明如何在 MemEye 和 Mem-Gallery 两个多模态记忆 benchmark 上评测 `vanilla` 的多模态能力，覆盖从下载数据到一键跑完四组配置的全过程。多模态功能本身的实现和接口见 [docs/multimodal/README_ZH.md](../multimodal/README_ZH.md)。

---

## 0. 评测内容

两个 benchmark 都是"多轮对话中穿插图片"的长期记忆测试。流程如下：

1. 每个 session 调用一次 `memory.add`。
2. 每道题调用一次 `memory.search`；如果题目自带图片，作为 `query_images` 一并传入。
3. 检索返回的文本记忆和图片，加上题目图片，交给同一个支持看图的作答模型（`answer_llm`）作答。
4. 由 `judge_llm` 按 benchmark 原版的评分标准打分。

| | MemEye | Mem-Gallery |
|---|---|---|
| 规模 | 8 个任务，MCQ + OPEN | 20 个任务，1711 道 OPEN 题 |
| 数据 | Hugging Face `MemEyeBench/MemEye`（约 260 MB） | Hugging Face `Ethan-Bei/Mem-Gallery`（约 530 MB） |
| 指标 | EM（MCQ 选项轮换去偏）、B-1、Judge，按任务宏平均 | F1、B-1、Judge（原版评分标准），全部题目微平均 |

每个 benchmark 跑四组配置：

| 配置 | `algo_config.multimodal.mode` | 检索策略 | 说明 |
|---|---|---|---|
| `b0` | `off` | fast | 纯文本 vanilla，不发送图片 |
| `b1` | `basic` | fast | 存原图，按图片相似度检索 |
| `enhanced` | `enhanced` | fast | VLM 抽取 + 视觉事实 + 证据图 |
| `enhanced_agentic` | `enhanced` | agentic | 多模态 agent 检索 |

**模式通过评测项目的 `project_override_config` 设置，服务端全局的 `mode` 保持默认的 `"off"` 即可。**

**耗时和用量参考**（单次运行，gpt-4.1-mini，耗时受 LLM 服务延迟影响很大）：

| | b0 | b1 | enhanced | enhanced_agentic |
|---|---|---|---|---|
| MemEye 耗时 | ~15 min | ~30 min | ~40 min | ~70–85 min |
| Mem-Gallery 耗时 | ~30 min | ~40 min | ~55–70 min | ~95–125 min |
| 作答器 token（MemEye / Mem-Gallery） | 2 M / 2.5 M | 12 M / 8 M | 15 M / 9 M | 15 M / 9 M |

两个 benchmark 的四组配置各跑一次，合计约 7 小时。

---

## 1. 前置条件

- **机器：** 16 GB 以上内存，磁盘预留 10 GB（数据集、原图存储、模型）。GPU 可选；本地的 SigLIP2 embedding 服务在 CUDA、Apple Silicon（MPS）或 CPU 上都能跑，CPU 会慢一些。
- **软件：** Docker（用于 Qdrant、Neo4j、Kafka 等依赖）、[uv](https://docs.astral.sh/uv/)、Python 3.13。
- **一个 OpenAI 兼容的 LLM 服务，需要提供：**
  - 能看图的 chat 模型，例如 `gpt-4.1-mini`。服务端的 enhanced 抽取、agent 和评测的作答、judge 都用它；
  - 文本 embedding 模型（vanilla 本来就需要）；
  - reranker（推荐配置；不配时视觉事实不做重排）。
- **多模态 embedding 服务：** 把文本和图片编码到同一个向量空间。可以用仓库自带的本地 SigLIP2 服务（见第 4 步），也可以换成自己的服务，只要兼容 OpenAI 协议。

---

## 2. 安装

```bash
git clone <MindMemOS 仓库地址> MindMemOS
cd MindMemOS
git checkout feat_multimodal
make dev-setup        # uv sync + NLP 资源
cp .env.example .env  # 数据库连接等环境变量
```

---

## 3. 下载数据集

```bash
uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download as get
get('MemEyeBench/MemEye', repo_type='dataset', local_dir='datasets/memeye')
get('Ethan-Bei/Mem-Gallery', repo_type='dataset', local_dir='datasets/memgallery')
"
```

下载后的目录结构如下。评测配置里的 `dataset` 字段指向各自的 `data/` 目录：

```
datasets/
├── memeye/data/        dialog/*.json   image/<任务>/...
└── memgallery/data/    dialog/*.json   image/<任务>/...
```

- `datasets/` 已加入 gitignore。
- 如果数据放在别处，可以从 `datasets/memeye`、`datasets/memgallery` 建软链接到实际位置。
- 网络受限时，可以设置 `HF_ENDPOINT` 使用 Hugging Face 镜像。

---

## 4. 启动多模态 embedding 服务（本地 SigLIP2）

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

---

## 5. 服务端配置并启动

```bash
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml
```

在 `config/mindmemos/dev.yaml` 中做以下修改：

1. 按 [README 的"准备配置文件"一节](../../README_ZH.md) 填好 `chat_model_router`、`embed_model_router`、`rerank_model_router`。chat 模型必须支持图片输入。
2. 配置多模态 embedding：

   ```yaml
   mm_embed_model_router:
     endpoints:
       - model: openai/siglip2-base-patch16-384
         api_base: http://127.0.0.1:8090/v1
         dimensions: 768
   ```

   > `model` 是 LiteLLM 标识符，必须带 provider 前缀（`openai/...`，前缀不会发给服务）；不再有 `protocol` 字段——mm embedding 路由统一走 OpenAI 风格的 embeddings 协议。

3. 允许服务端按路径读取数据集里的图片。评测以文件路径方式发送图片，这里必须写**绝对路径**，并且是软链接解析之后的真实路径：

   ```yaml
   blob_store:
     local_root: data/blobs                 # 原图存放位置
     allowed_path_roots:
       - /绝对路径/MindMemOS/datasets      # 或数据集的真实所在目录
   ```

4. 让服务端使用评测生成的 key 文件。评测时会自动写入这个文件，服务端会自动重新加载：

   ```yaml
   auth:
     api_key_file: eval_api_keys.yaml
   ```

5. `algo_config.multimodal.mode` 保持 `"off"`；YAML 中必须写带引号的 `"off"`，否则会被解析成布尔值。

启动服务：

```bash
make dev   # 启动 Docker 依赖和 API，地址 http://127.0.0.1:8000
```

服务端不会热加载配置和代码。修改 `dev.yaml` 后需要重启。

---

## 6. 评测配置

```bash
cp config/mindmemos_eval/memory_evaluation_multimodal.example.yaml config/mindmemos_eval/memory_evaluation_multimodal.yaml
cp config/mindmemos_eval/memory_evaluation_memgallery.example.yaml config/mindmemos_eval/memory_evaluation_memgallery.yaml
```

在两个文件中：
- 给 `runner.llm`、`runner.answer_llm`、`runner.judge_llm` 填上 `model`、`base_url`、`api_key`。作答模型必须支持看图；三者可以是同一个模型。
- 其余字段保持默认即可。模式、检索策略和是否发送图片由下一步的脚本自动生成，不用手改。

可选项（`execution_params.<benchmark>`）：
- `max_answer_images`：附给作答器的图片上限，默认 8；
- `image_max_side`：附图前缩放到的最长边，默认 1024。

---

## 7. 一键运行

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
- **输出位置：**
  - `reports/multimodal/<tag>/`：每次运行的 manifest（`<bench>_<config>_r<k>.jsonl`）、日志、生成的配置，以及汇总表 `summary.md`。`<tag>` 默认为日期，冒烟测试为 `smoke`。
  - 逐题明细在 `runs/mindmemos_eval/<run_id>/<benchmark>/`。

不用脚本、手动跑单个配置的方式见 [docs/eval/README_ZH.md](README_ZH.md) 的多模态一节。

---

## 8. 结果解读

manifest 中的 `eval_result` 包含：
- `metrics`：百分制总分、按任务（`per_task`）的分数，Mem-Gallery 另有按题型（`per_point`）的分数；
- `cost`：客户端的写入和检索调用次数与延迟、作答器和 judge 的 token；
- `valid`：只要有作答或 judge 调用失败就为 false，这样的结果不可用。

**参考结果**（gpt-4.1-mini 作答和 judge，本地 SigLIP2，各 2 次均值；2026-10 测于 `feat_multimodal`）：

| MemEye | EM | B-1 | Judge |
|---|---|---|---|
| b0 | 43.03 | 20.32 | 26.26 |
| b1 | 59.77 | 27.02 | 47.80 |
| enhanced | 63.45 | 25.49 | 50.80 |
| enhanced_agentic | 65.89 | 27.09 | 53.30 |

| Mem-Gallery | F1 | B-1 | Judge |
|---|---|---|---|
| b0 | 38.72 | 32.59 | 70.68 |
| b1 | 54.73 | 47.75 | 85.06 |
| enhanced | 55.66 | 48.98 | 87.06 |
| enhanced_agentic | 57.06 | 50.45 | 89.14 |

**比较结果时注意：**
- **MemEye 波动大：** 单次 Judge 波动约 2–3 分，单个任务可达 10–20 分；并且会随 LLM 服务随时间漂移，同一份代码隔两天重跑，Judge 差了约 2.6 分。比较两个版本时，每个配置至少跑 2 次，并在同一时期重跑基线。
- **Mem-Gallery 较稳：** 单次 Judge 波动小于 0.6。
- **不要同时跑两条评测：** 共用同一个 LLM 服务时，并发负载会把 MemEye 的 Judge 拉低约 5 分。
- **VR 计数题：** Mem-Gallery 作答器会给图片结果、事实和视觉事实都标上 image_id（VS 题需要用 image_id 作答）。快速检索模式下，作答器有时会按重复出现的 id 计数，导致 VR 计数题多数。

---

## 9. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `add.images_unsupported` / `search.images_unsupported` | 该项目的 `mode` 是 `off`。b0 本来就不发图片；其他配置出现说明 `project_override_config` 没生效，确认用的是脚本生成的配置 |
| `image.path_denied` / `image.path_disabled` | `blob_store.allowed_path_roots` 没覆盖数据集的真实路径（注意软链接）或为空 |
| 冒烟测试卡在 add 或很慢 | enhanced 写入要调用 VLM，单次 add 需要十几到几十秒，属正常；检查 embedding 服务和 LLM 服务的延迟 |
| 结果 `valid: false` | 作答或 judge 调用失败，多半是 LLM 服务限流或不稳定。服务恢复后重新执行同一条命令，失败的那次会重跑 |
| 更换多模态 embedding 模型后报维度不一致 | 图片集合按 `dimensions` 建立。换模型时需要删除 Qdrant 中的 `image_item_v1` 集合（或改用新的 `database.qdrant.image_collection` 名称），并重启服务 |
| 磁盘占用增长 | 每次评测使用新的项目 id，原图存放在 `data/blobs/<项目>/`，记录留在 Qdrant 和 Neo4j 中。开发环境用完可以删除旧项目对应的原图目录；`make db-clean` 会清空所有数据库卷，请谨慎使用 |
| `multimodal_agentic_failed` 警告 | agent 的某次 LLM 调用失败，本次搜索退回快速检索结果，不影响运行有效性；偶发属正常 |
