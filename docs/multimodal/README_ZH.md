# vanilla 多模态：合入说明

> 分支 `feat_multimodal`，基于 `main`。本文说明多模态能力的实现方式、接口、测试和评测结果，供代码评审和交接使用。
> 运行多模态 benchmark 的步骤见 [docs/eval/multimodal_benchmark_ZH.md](../eval/multimodal_benchmark_ZH.md)。

---

## 1. 结论摘要

- **多模态是 vanilla 的一个开关。** 由 `algo_config.multimodal.mode` 切换：`off`、`basic`、`enhanced`，默认 `off`，可按项目覆盖。
  - `off` 下行为与原 vanilla 完全一致，带图请求返回 400。
  - 不带图、也不设 `multimodal=true` 的检索，在任何模式下都与原 vanilla **逐字节一致**，由一致性测试守护。
- **实现方式是包装与继承，不复制 vanilla 代码。**
  - 写入侧：`VanillaMultimodalAddPipeline` 继承 `VanillaAddPipeline`。
  - 检索侧：`VanillaMultimodalSearchEngine` 继承 `VanillaSearchEngine`。
  - 对 vanilla 和通用检索管线只加了几个可选钩子。
- **效果（各 2 次均值）：**

  | 数据集 | 配置 | 指标 | 分数 |
  |---|---|---|---|
  | MemEye | enhanced 快速检索 | EM / B-1 / Judge | 63.45 / 25.49 / 50.80 |
  | MemEye | enhanced agentic | EM / B-1 / Judge | 65.89 / 27.09 / 53.30 |
  | Mem-Gallery | enhanced 快速检索 | F1 / B-1 / Judge | 55.66 / 48.98 / 87.06 |
  | Mem-Gallery | enhanced agentic | F1 / B-1 / Judge | 57.06 / 50.45 / 89.14 |

  - 相对纯文本基线：MemEye EM 约 +20、Judge 约 +25；Mem-Gallery F1 约 +18、Judge 约 +18。
  - 相对 basic：两个数据集的 Judge 再高 2–6 分（Mem-Gallery 快速检索 +2.0 至 MemEye agentic +5.5）。
- **代价：**
  - enhanced 写入时，含图的对话段用 VLM 抽取。单次 add 的耗时是纯文本时的 2–3 倍，chat 调用数基本不变；图片多的数据（Mem-Gallery）写入 chat 调用多约 23%。
  - 快速检索单次约 7–9 s，纯文本约 3.5 s。
  - agentic 检索单次约 20–30 s，每次搜索多 2–3 次 chat 调用。
- **合入 main 前需要补的缺口：**
  1. 没有产品级的图片删除 API，见 §3.6；
  2. 原图目前只能存本地磁盘；
  3. 生产环境需要自备多模态 embedding API。

  其余已知限制见 §7.2。

---

## 2. 代码合入情况

### 2.1 提交历史（`main..feat_multimodal`）

| 阶段 | 提交 | 内容 |
|---|---|---|
| 基础版 | `4559a78` | 独立算法 vanilla_mm 的基础版：存原图，按图片相似度检索 |
| 增强版 | `d60b7d3`、`ae86d56` | enhanced：VLM 抽取、锚定、证据图、agentic；会话展开 |
| 并入 vanilla | `0dad832` | 并入 vanilla，用 `algo_config.multimodal.mode` 切换，删除 vanilla_mm |
| 优化 | `3a3a619`、`6980fb6`、`3969812` | 视觉事实作为独立结果返回；agent 的搜索工具也返回视觉事实；OmegaConf 兼容修复 |
| 试验并回退 | `61abd93`、`4872ea0` → `d24d574`、`d010c0e` | 给视觉事实加图片标识：无效，已回退 |
| 修复 | `ede98de` | VLM 返回非法 mem_type 时回退到 vanilla 抽取，不再返回 500 |
| 评测 | `1d91ce9` | Mem-Gallery 评测 adapter |
| 修复 | `8b65531`、`6df5b48` | 抽取缺口：图片编号解析 bug，单次超过 6 张图的部分另发调用 |
| 清理 | `9b5867f` | 删除已失效的 vanilla_mm 开发 key |
| 文档与工具 | 最后一个提交 | 本文、benchmark 运行指南、`scripts/mm_embed_server.py`、`scripts/run_multimodal_bench.py` |

"并入 vanilla"之前的提交使用独立算法名 `vanilla_mm`，之后统一为 vanilla 的模式开关。

### 2.2 改动规模与分布

与 `main` 相比约 100 个文件，新增约 8000 行，几乎全是新文件。

| 部分 | 主要文件 |
|---|---|
| 检索 | `src/mindmemos/mindmemos/pipelines/search/vanilla_multimodal/engine.py`、`agentic.py` |
| 写入 | `src/mindmemos/mindmemos/pipelines/add/vanilla_multimodal/vanilla_multimodal_add.py` |
| 组件 | `src/mindmemos/mindmemos/components/multimodal/`：`extraction.py`（VLM 抽取）、`image_loader.py`（图片加载）、`messages.py` |
| 基础设施 | `llm/mm_embedding.py`（多模态 embedding 客户端）、`infra/blob/`（原图存储）、`infra/db/collections/image.py`（Qdrant 图片集合）、`pipelines/memory_db/image_store.py` |
| 配置 | `config/algo/multimodal.py`（`algo_config.multimodal`）、`config/multimodal.py`（embedding 路由、原图存储）、`config/validation.py` |
| API、类型、SDK | `api/*`、`typing/*`、`src/mindmemos_sdk/*` |
| 评测 | `src/mindmemos_eval/mindmemos_eval/memory/envs/memeye/`、`envs/memgallery/` |
| 测试 | `tests/pipelines/test_vanilla_multimodal*.py` 等 15 个文件 |
| 文档、脚本 | `docs/config/README.md`（多模态配置）、`docs/eval/`、`docs/multimodal/`、`scripts/mm_embed_server.py`、`scripts/run_multimodal_bench.py` |

对 vanilla 和通用代码的改动只有以下几处：
- `api/algorithm.py`：vanilla 绑定多模态写入管线，并标记 `supports_images`；
- `api/routes.py`、`api/services/memory_service.py`：图片接口和 400 校验；
- `pipelines/search/pipeline.py`：新增 `augment_results`、`owns_agentic`、`agentic_refine` 三个可选钩子；
- `pipelines/add/vanilla/vanilla_add.py`：抽出 `_memory_extractor_for` 钩子；
- `typing/*`：`ImageRef`、`query_images`、`multimodal` 字段。

---

## 3. 实现方式

### 3.1 模式与开关

| 模式 | 写入 | 检索（只在多模态检索时生效） |
|---|---|---|
| `off` | 原 vanilla；带图请求返回 400 | 原 vanilla；带 `query_images` 返回 400 |
| `basic` | 存原图，生成图片向量，记录出现上下文；文本照常走 vanilla | 文本结果之后追加按相似度检索到的图片（`image_top_k`=5） |
| `enhanced` | 含图的对话段由 VLM 抽取；对话事实进文本记忆，视觉事实和描述存在图片记录上 | 混合打分、经图召回事实、视觉事实结果、证据图；可选 agentic |

**什么情况下是"多模态检索"：** 模式不为 `off`，并且请求带 `query_images` 或设 `multimodal=true`。其余检索完全走原 vanilla，不做多模态 embedding、不读图片元数据、不查图片集合。

### 3.2 写入链路（`VanillaMultimodalAddPipeline`）

1. **分流。** 不带图的请求直接交给 `VanillaAddPipeline`。
2. **加载图片（`ImageLoader`）。** `ImageRef` 必须恰好指定一种来源：`path`、`url`、`data`（base64）或 `blob_id`。安全限制：
   - `path` 只能落在 `blob_store.allowed_path_roots` 下（注意：roots 不区分项目——多项目/共享部署下，roots 内的图片对每个项目都可见，建议只在单项目部署启用）；
   - `url` 默认拒绝私网地址（防 SSRF）；
   - 单张图上限 20 MB；会嗅探真实 MIME 类型。
3. **存原图（blob store）。** 按内容 sha256 去重，每个项目一个命名空间。支持本地磁盘与 S3/MinIO 两种后端（`blob_store.backend`）。
4. **图片索引。** 通过 `mm_embed_model_router` 生成图片向量，写入 Qdrant `image_item_v1`：
   - 同一项目、同一用户的同一张图只有一条记录，point id = uuid5(project, user, blob_id)；
   - 每次出现追加一条 `occurrence`：session、时间、所在轮次与下一轮的对话上下文、调用方传入的 `image_id` 和 caption；
   - 未配置多模态 embedding 时，只存原图、不建索引。
5. **文本部分。** 去掉图片后交给 vanilla。只有图、没有文字的轮次不进文本记忆，它的图交给下一条保留的消息。
6. **enhanced 额外步骤（`MultimodalMemoryExtractor`，包装 vanilla 抽取器）：**
   - 含图的对话段仍然只做一次抽取调用：沿用 vanilla 的 prompt，前面加上看图规则，附上图片。输出分两部分：
     - `memories`：对话事实，照常写入文本记忆，`metadata.image_blob_ids` 记录来源轮次的图（锚定）；
     - `image_memories`：每张图的描述和视觉事实，只存在图片记录上，**不进文本记忆**。图片记录同时生成一个文本向量 `text`，内容是描述、视觉事实和对话上下文。
   - 一个对话段超过 `max_extraction_images`（6）张图时，超出部分另发"只看图"的调用，每次最多 6 张，与主抽取并行。
   - 兼容 VLM 写成 `"IMAGE 1"` 之类的图片编号。仍有图片没拿到视觉事实时，记录警告 `multimodal_visuals_missing`。
   - VLM 抽取失败或输出非法时，回退到 vanilla 文本抽取，请求不失败。
7. **异步 add。** 入队前先把图片存好，队列消息里只带 blob_id。

### 3.3 检索链路（`VanillaMultimodalSearchEngine`）

文本候选完全由 `VanillaSearchEngine` 计算，照常经过最终过滤（rerank、阈值、top_k）。图片相关的结果在 `augment_results` 中追加：

- **basic：** 用多模态模型对查询文本编码，按图片相似度取前 `image_top_k` 张。
- **enhanced：**
  1. **图片打分。** 每张图的分数由两部分混合，各自按分布做归一化（DBSF），文本权重 `image_text_weight`=0.5：
     - 查询文本与图片 `text` 向量的相似度；
     - 查询文本与图片向量的相似度（SigLIP）。
  2. **经图召回事实。** 只有图片相似度明显高于中位数（`image_hit_margin`）的命中，才召回锚定在这些图上的对话事实，最多 `anchored_fact_limit`=5 条。这条规则用来避免同模板截图带来的噪声。
  3. **视觉事实。** 候选来自证据图和混合得分前 `visual_fact_images`=12 张图的视觉事实，经 reranker 重排后取前 `visual_fact_limit`=10 条，作为 `memory_type=visual_fact` 的独立结果返回。
  4. **证据图，最多 `evidence_images`=6 张，依次取：**
     - 与查询图明显相似的图；
     - 相似度最高那张图所在 session 的相邻图片（`session_expansion_images`=4）；
     - 已返回事实所锚定的图，与混合得分最高的图交替选入。
- **agentic**（`search_strategy=agentic`，仅 enhanced）：
  - 以快速检索的结果为起点，运行一个简化的 ReAct agent，最多 `agentic_max_steps`=4 步。工具有四个：`search_memories`（返回事实和视觉事实）、`search_images`、`view_images`、`submit`。
  - agent 选中的证据排在前面，快速检索的结果跟在后面，所以召回不会低于快速检索。
  - agent 任一步出错，整次搜索退回快速检索的结果，并记录 `multimodal_agentic_failed`。

### 3.4 与 vanilla 的边界

| 场景 | 行为 |
|---|---|
| mode=`off` | 等同原 vanilla |
| 纯文本 add（任何模式） | 等同原 vanilla |
| 非多模态检索（任何模式） | 等同原 vanilla；检索记录的序列化也一致 |
| 视觉事实 | 永远不进文本记忆，因此不影响 vanilla 检索、相关记忆召回、dreaming 和 feedback |

以上由 `tests/pipelines/test_vanilla_multimodal_parity.py` 守护：逐项比较 LLM prompt、embedding 输入、写入计划和检索结果。

### 3.5 依赖的外部服务

- **多模态 embedding**（`mm_embed_model_router`）：文本和图片映射到同一个向量空间。目前只实现了 OpenAI 兼容协议（`input: [{"image": dataURI}]`），支持多个端点按顺序故障转移和批量拆分。端点配置没有 `protocol` 字段（早期分支版本有过，已删除；升级时请删掉该行，模型名需带 `openai/` 等前缀）。
  - 开发和评测可以用 `scripts/mm_embed_server.py`，它把本地 SigLIP2 包装成 HTTP 服务（768 维）。
  - **生产环境需要自备这个 API。**
- **VLM：** 复用 vanilla 的 chat 模型，要求能接收图片。评测使用 gpt-4.1-mini。
- **Reranker：** 用于视觉事实重排；没有 reranker 时按图片排序。

### 3.6 删除与数据生命周期

- 删除单条文本记忆后它变为 archived；经图召回时会跳过 archived 的记忆。
- 图片和原图的清理有 `ImageMemoryStore.purge_images`：可按项目或用户删除；用户级删除时，原图只有在项目内没有其他记录引用时才删。**目前只有测试在用它，没有对外的产品 API**；评测的项目重置是直接按 blob 目录与 Qdrant 集合清库（因此 S3 后端下评测重置不会清理 S3 上的原图，需要手动处理）。纯文本侧本来也没有批量删除 API。
- **缺口：** 用户或项目被删除时，产品侧无法删除他们的图片记录和原图。

---

## 4. 接口

### 4.1 HTTP API

| 接口 | 变化 |
|---|---|
| `POST /v1/memory/add` | `messages[].images: ImageRef[]`；允许只有图、没有文字的轮次 |
| `POST /v1/memory/search` | 新增 `query_images: ImageRef[]`（隐含多模态）和 `multimodal: bool`（默认 false） |
| `GET /v1/memory/images/{blob_id}` | 新增：返回本项目已存图片的原始字节，需要 `memory:read` 权限与 `?user_id=`（图片属主，服务端会对照图片记录校验，非属主一律 404 `image.not_found`），私有缓存 1 天 |

**`ImageRef` 字段：** `path` / `url` / `data` / `blob_id`（四者恰好一个），以及可选的 `mime_type`、`image_id`（调用方的 id，检索结果会原样带回）、`caption`。持久化请求记录时，内联的 base64 会替换为 `<redacted>`。

请求示例：

```json
POST /v1/memory/add
{"user_id": "u1", "messages": [
  {"role": "user", "content": "这是我新买的沙发", "images": [{"url": "https://example.com/sofa.jpg", "image_id": "img-1"}]},
  {"role": "assistant", "content": "绿色丝绒三人沙发，很好看。"}
]}

POST /v1/memory/search
{"user_id": "u1", "query": "我的沙发是什么颜色？", "multimodal": true}
```

**检索结果中新增的条目类型：**
- `memory_type=image`：图片结果。`memory` 是 `[image] 描述或上下文`；`metadata` 包括 `blob_id`、`user_id`（属主，取原图时的必传参数）、`score`、`session_ids`、`caller_image_ids`、`occurrences`、`description`、`visual_facts`、`anchored_memory_ids`、`evidence`（query_image / session / anchored / similar），以及会话展开时的 `session_position`。
- `memory_type=visual_fact`：单条视觉事实。`metadata` 包括 `blob_id`、`image_id`、`user_id` 和 rerank `score`。
- 对话事实的 `metadata.image_blob_ids` 指向它锚定的图。

**`metadata.image_url`（图片懒加载直链）：** 带 `blob_id` 的图片命中还会附带 `image_url`——
`s3` 后端是短时效预签名 URL（本地 HMAC 签名，默认 1 小时，见 `blob_store.s3_presign_expiry_seconds`），
客户端可直接 GET 原图，字节不过本 API;`local` 等无法签名的后端回落为上面的代理路由的**相对路径**
（`/v1/memory/images/{blob_id}?user_id=<属主>`，客户端按自己的 API 地址拼全，仅当命中带属主时——
不取请求的 Host 头，避免把凭据指向不可信源）。URL 过期或缺失时，
仍用 `blob_id` + 属主 `user_id` 走 `get_image`。该字段在响应组装时注入、不落遥测。

**错误码（400 或 404）：**
- 写入与检索：`add.images_unsupported`、`search.images_unsupported`、`add.too_many_images`（单次最多 32 张）；
- 图片加载：`image.path_denied`、`image.path_disabled`、`image.url_denied`、`image.url_disabled`、`image.fetch_failed`、`image.too_large`、`image.unsupported`；
- 读取图片：`image.not_found`。

### 4.2 SDK（`mindmemos_sdk`，同步和异步客户端都支持）

- `add(messages)`：`DialogueMessage.images` 可以传 `ImageRef` 或 dict；
- `search(query, query_images=..., multimodal=...)`：命中 `metadata.image_url` 随 `metadata` dict 原样透传；
- `get_image(blob_id, user_id) -> (bytes, mime_type)`（`user_id` 为图片属主，即检索结果 `metadata.user_id`）；
- 导出了 `ImageRef`。

### 4.3 配置

`algo_config.multimodal`（可按项目覆盖），主要字段：

| 字段 | 默认值 | 作用 |
|---|---|---|
| `mode` | `"off"` | off / basic / enhanced（YAML 中要写成带引号的 `"off"`） |
| `image_top_k` | 5 | basic：追加的图片数 |
| `evidence_images` | 6 | enhanced：证据图上限 |
| `session_expansion_images` | 4 | 会话展开的图片数 |
| `anchored_fact_limit` | 5 | 经图召回的事实上限 |
| `image_hit_margin` | 0.03 | 判定"明显命中"的边距 |
| `image_text_weight` | 0.5 | 图片打分中文本相似度的权重 |
| `visual_fact_limit` / `visual_fact_images` | 10 / 12 | 返回的视觉事实条数 / 候选图片数 |
| `max_extraction_images` | 6 | 单次抽取调用的图片数，超出部分另发调用 |
| `extraction_image_detail` | auto | 传给 VLM 的图片精度 |
| `agentic_max_steps` | 4 | agent 的工具调用步数 |
| `max_images_per_request` | 32 | 单次 add 的图片上限 |

基础设施配置（完整说明见 `docs/config/README.md`）：
- `mm_embed_model_router.endpoints[]`：model、api_base、api_key、dimensions、超时、重试、批大小；
- `blob_store`：backend（local / s3）、local_root、max_image_bytes、allowed_path_roots、URL 下载开关和私网限制；s3 后端还有 `s3_endpoint`/`s3_bucket`/`s3_region`/`s3_access_key`/`s3_secret_key`/`s3_public_base_url`，以及检索直链相关的 `s3_presign_endpoint`/`s3_presign_expiry_seconds`。从旧版本升级后需重跑 `make dev-setup`（uv sync）才能拿到 s3 extra 依赖；本地启用 s3 后端还需 `make dev-blobs` 启动 MinIO；
- `database.qdrant.image_collection`：默认 `image_item_v1`。

---

## 5. 测试

| 测试 | 覆盖内容 |
|---|---|
| `tests/pipelines/test_vanilla_multimodal_parity.py` | off 模式和非多模态检索与 vanilla 逐项一致 |
| `tests/pipelines/test_vanilla_multimodal.py` | basic：存储、去重、索引、400 校验、异步 add |
| `tests/pipelines/test_vanilla_multimodal_enhanced.py` | VLM 抽取、锚定、编号兼容、溢出调用、混合打分、证据图、视觉事实、agentic |
| `tests/pipelines/memory_db/test_image_purge.py` | 图片与原图的清理 |
| `tests/api/test_multimodal_requests.py` | API 层校验与错误码 |
| `tests/components/test_image_loader.py` | 加载安全限制 |
| `tests/infra/test_blob_store.py`、`test_blob_store_s3.py`（live 见文件头）、`tests/config/test_multimodal_config.py`、`tests/typing/test_multimodal_typing.py`、`tests/mindmemos_sdk/test_multimodal_sdk.py` | 基础设施（local/S3/presign）、配置校验、类型、SDK |
| `tests/api/test_memory_image_route.py`、`test_memory_search_image_url.py` | 取图路由、search 的 `image_url` 注入 |
| `tests/mindmemos_eval/test_memeye_adapter.py`、`test_memgallery_adapter.py`、`test_db_reset_blobs.py` | 评测 adapter（离线） |

- **全量测试（Windows 开发机）：** 1321 passed、92 skipped。另有 11 个 failed 和 8 个 collection errors，均为本批之前就存在的环境/包问题，与本批改动无关：
  - architecture guard 测试 2 个（违规在 `main` 既有代码，本批没有新增）；
  - eval 侧 memoryagentbench/memoryarena（数据集 zip 缺失）、skill_runners（包结构）共 5 个；
  - SDK 侧 package boundary、skill 检测共 4 个（Windows 路径分隔符 / CRLF / 测试顺序）；
  - prompts（GBK 编码）、scripts（缺 `scripts/locomo`）、spreadsheetbench（ImportError）的收集错误。
- **S3/MinIO live 用例：** `MINDMEMOS_TEST_S3_ENDPOINT=http://localhost:9100 uv run pytest tests/infra/test_blob_store_s3_live.py`（`make dev-blobs` 后），含预签名 URL 匿名 GET 回读。

---

## 6. 评测结果

### 6.1 评测方法与注意事项

| | MemEye | Mem-Gallery |
|---|---|---|
| 规模 | 8 个任务，MCQ + OPEN | 20 个任务，1711 道 OPEN 题 |
| 指标 | EM（MCQ 选项轮换去偏）、BLEU-1、Judge（MemEye 评分标准），按任务宏平均 | F1、BLEU-1、Judge（原版评分标准），全部题目微平均 |

- **评测流程：** 每个 session 调一次 add，每道题调一次 search，作答器统一使用 gpt-4.1-mini（VLM），judge 也是 gpt-4.1-mini。b0 不发图片。复现方法见运行指南。
- **噪声与漂移：**
  - MemEye 的 OPEN 题少，单次 Judge 波动约 2–3 分，单个任务可以波动 10–20 分。
  - **同一份代码隔两天重跑，MemEye 快速检索的 Judge 从 52.8 降到 50.2，agentic 的 EM 从 67.8 降到 65.5。** 比较不同版本时，必须用同一时期的基线。
  - Mem-Gallery 单次 Judge 波动小于 0.6，更稳定。
- **负载影响：** 两条评测链同时跑、共用同一个 LLM 服务时，MemEye 的 Judge 会低约 5 分。下面的数字都是串行跑出来的。
- **作答器的 image_id 标注：** Mem-Gallery 作答器在事实和视觉事实上也标注 image_id。图片覆盖率提高后，快速检索模式下的 VR 计数题容易多数。

### 6.2 MemEye（百分制，各 2 次均值，括号内为区间）

| 配置 | EM | B-1 | Judge |
|---|---|---|---|
| b0（mode off，纯文本） | 43.03 | 20.32 | 26.26 |
| b1（basic） | 59.77 | 27.02 | 47.80 |
| **enhanced 快速检索** | **63.45** (63.08–63.81) | **25.49** | **50.80** (49.36–52.24) |
| **enhanced agentic** | **65.89** (65.35–66.43) | **27.09** | **53.30** (53.18–53.41) |
| 同期对照：抽取修复前的 enhanced 快速检索（单次） | 63.11 | 27.59 | 50.19 |
| 同期对照：抽取修复前的 enhanced agentic（单次） | 65.45 | 27.02 | 52.78 |

各任务的 EM / Judge：

| 任务 | b0 | b1 | enhanced 快速检索 | enhanced agentic |
|---|---|---|---|---|
| Brand_Memory | 39.2 / 34.1 | 47.4 / 50.9 | 58.6 / 49.6 | 56.0 / 60.3 |
| Card_Playlog | 41.7 / 33.3 | 39.6 / 29.2 | 41.7 / 35.4 | 39.8 / 31.2 |
| Cartoon_Entertainment | 53.6 / 32.2 | 73.0 / 57.7 | 76.3 / 57.6 | 75.3 / 64.8 |
| Home_Renovation | 40.1 / 26.9 | 72.4 / 63.0 | 72.4 / 57.0 | 76.4 / 57.9 |
| Multi-Scene_Case_Archive | 36.2 / 33.5 | 65.2 / 62.8 | 73.0 / 66.0 | 78.5 / 68.2 |
| Outdoor_Navigation | 62.5 / 7.6 | 85.3 / 35.7 | 82.6 / 41.5 | 83.0 / 39.3 |
| Personal_Health | 25.5 / 18.1 | 35.8 / 30.9 | 37.5 / 47.3 | 50.7 / 49.8 |
| Social_Chat | 45.3 / 24.3 | 59.5 / 52.4 | 65.5 / 52.0 | 67.2 / 54.7 |

### 6.3 Mem-Gallery（各 2 次均值，括号内为区间）

| 配置 | F1 | B-1 | Judge |
|---|---|---|---|
| b0（off） | 38.72 | 32.59 | 70.68 |
| b1（basic） | 54.73 | 47.75 | 85.06 |
| **enhanced 快速检索** | **55.66** | **48.98** | **87.06** (86.70–87.43) |
| **enhanced agentic** | **57.06** | **50.45** | **89.14** (88.98–89.30) |
| 对照：抽取修复前的 enhanced 快速检索 | 56.72 | 49.89 | 87.61 |
| 对照：抽取修复前的 enhanced agentic | 56.95 | 50.14 | 87.84 |
| 参考：MHG 论文方法的原版实现（作答协议不同，不可直接比较） | 52.48 | 45.41 | 84.80 |

各题型的 F1 / Judge：

| 题型 | b0 | b1 | enhanced 快速检索 | enhanced agentic |
|---|---|---|---|---|
| VS（以图找图，答 image_id） | 9.2 / 15.8 | 79.2 / 75.7 | 89.2 / 89.5 | 91.3 / 93.9 |
| VR（视觉推理） | 23.5 / 45.0 | 50.0 / 69.1 | 39.6 / 60.5 | 50.3 / 71.6 |
| KR（知识更新） | 21.5 / 77.8 | 30.8 / 85.5 | 32.3 / 88.9 | 31.4 / 89.8 |
| TTL（看图学新知识） | 33.1 / 88.5 | 32.5 / 91.2 | 35.4 / 93.9 | 35.1 / 93.3 |
| TR（时间推理） | 43.5 / 81.1 | 45.6 / 84.3 | 46.5 / 81.5 | 47.3 / 86.0 |
| CD（冲突检测） | 69.8 / 69.8 | 64.8 / 64.8 | 69.1 / 69.1 | 66.0 / 66.0 |
| AR（拒答） | 89.5 / 94.8 | 90.6 / 93.3 | 85.9 / 90.1 | 88.6 / 92.9 |
| FR / MR（事实 / 多实体） | 45 / 92–93 | 46 / 94 | 45–47 / 94 | 44–46 / 93 |

### 6.4 成本（每次评测的均值）

| 配置 | 单次 add 平均 | 单次 search 平均 / p95 | 写入 chat 调用 | 每次检索的 chat 调用 | 作答器 token |
|---|---|---|---|---|---|
| MemEye b0 | 7.6 s | 3.7 s / 6.2 s | 221 | 0 | 2.0 M |
| MemEye b1 | 8.5 s | 3.6 s / 6.0 s | 221 | 0 | 12.0 M |
| MemEye enhanced 快速检索 | 18.5 s | 8.9 s / 13.2 s | 236 | 0 | 15.1 M |
| MemEye enhanced agentic | 14.1 s | 31.1 s / 69.2 s | 236 | 约 2.8 | 15.1 M |
| Mem-Gallery b0 | 16.0 s | 3.5 s / 6.0 s | 240 | 0 | 2.5 M |
| Mem-Gallery b1 | 17.2 s | 3.4 s / 5.5 s | 240 | 0 | 8.2 M |
| Mem-Gallery enhanced 快速检索 | 28.6 s | 8.3 s / 11.7 s | 302 | 0 | 9.2 M |
| Mem-Gallery enhanced agentic | 31.9 s | 20.9 s / 41.4 s | 302 | 约 1.9 | 9.2 M |

说明：
- 延迟受 LLM 服务影响很大。同一配置在不同时段，add 的平均耗时可以从 14 s 变到 27 s，所以延迟只能看量级。
- 作答器 token 增加，是因为检索返回的图片会附给作答器。这部分成本发生在调用方，不在服务端。
- basic 写入只多一次多模态 embedding（本地 SigLIP2 约 0.2 s/张），检索多一次多模态 embedding（约 0.04 s）。

### 6.5 试过但无效、未合入的方案

| 方案 | 结果 | 原因 |
|---|---|---|
| 视觉事实前加图片标识，并让 agent 只保留 5 条未选中的事实 | 快速检索 Judge −2.8，agentic EM −1.3 | reranker 会匹配标识，把视觉事实集中到少数几张图上 |
| 先用 VLM 描述查询图片，再拼进查询 | 两个数据集上带图的题只 +0.2～+0.4 | 作答器直接看题图，以图搜图也已经能找到对应存图，空间很小 |
| 视觉事实前加图片名（对话中对这张图的叫法） | MemEye agentic EM −2.1；Mem-Gallery 快速检索略降、agentic +0.4 | 名字常是"从另一侧看的同一个房间"这类相对说法，事实也被集中到少数几张图上 |

---

## 7. 评估

### 7.1 优点

- 对现有产品零侵入：默认 `off`，纯文本行为与 vanilla 逐字节一致，有测试守护，可以按项目逐步开启。
- 分层清楚：basic 只需要多模态 embedding API；enhanced 再要求 chat 模型能看图；agentic 是可选的检索策略。
- 视觉事实不进文本记忆，避免影响 vanilla 的检索、dreaming 和 feedback。
- 图片按内容去重；输入有安全限制（路径白名单、SSRF 防护、大小上限）；请求记录中不保存内联图片字节。
- 失败都会退回到可用结果：VLM 抽取失败时用文本抽取，agentic 失败时用快速检索结果，embedding 未配置时只存原图。
- 在两个数据集上，相对纯文本和 basic 都有明显提升。

### 7.2 风险与限制

1. **删除：** 没有产品级 API 删除用户或项目的图片和原图，合规上要补。
2. **存储：** 原图目前只能存本地磁盘，多实例部署或生产环境需要实现对象存储后端。接口已预留。
3. **外部依赖：** 需要多模态 embedding API；仓库自带的 SigLIP2 服务只用于开发和评测。
4. **同模板截图：** 视觉事实分不清出自哪张图，例如仪表盘、游戏截图类任务（MemEye 的 Personal_Health、Card_Playlog），这类任务仍然偏弱。
5. **延迟：** enhanced 的写入和检索都明显变慢，agentic 检索在 20–30 s 量级，只适合对时延不敏感的场景。
6. **稳健性：** agent 的单次 LLM 错误会让整次 agentic 检索退回快速检索结果，评测中约 0.3%。加一次重试就能缓解。
7. **评测可信度：** MemEye 波动和漂移都大，做小幅优化时，结论必须靠同期基线和多次运行。

### 7.3 后续建议

- 合入 `main` 前：补用户和项目级的图片删除，评估对象存储后端，补上线配置样例（多模态 embedding 端点）。
- 如需继续优化：用同期基线评估，优先考虑同模板截图的事实归属问题，以及 agentic 的重试。
