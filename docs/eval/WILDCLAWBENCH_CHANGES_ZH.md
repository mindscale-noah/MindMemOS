# WildClawBench 实验改动全记录

基线：`origin/develop` @ `d94340c`，分支 `feat/task_memory`。
范围：MindMemOS 仓库（本仓库）13 个提交 + 未提交工作区改动；WildClawBench 仓库（`C:\working_projects\Memory\WildClawBench`）未提交改动；若干不入库的一次性脚本。
截至：2026-08-28。

---

## 一、已提交改动（按时间序）

### 阶段 1：基础设施搭建（07-10）

| 提交 | 内容 |
|---|---|
| `8c7f6ce` | 实验基础设施首次落地：`scripts/wildclawbench/` 下新增 `run_serial.sh`（串行跑全量任务 + 每任务后等记忆写入完成）、`sync_image.sh`（把本地 SDK 与插件烘焙进容器镜像）、`new_key.py`（签发绑定 project override 的 API key）、`wait_drain.py`（轮询 add_record 状态直到抽取完成）；新增 openclaw-serpapi-plugin（网页搜索提供商，SerpApi 后端） |
| `4bad92c` | 补交 serpapi 插件 dist/（.gitignore 误排除）；修 `sync_image.sh` 的 `MINDMEMOS_REPO` 路径 |
| `f3ae816` | openclaw-plugin 的 package.json 恢复 src 入口并把 src 列入 npm files |
| `ff37a86` | serpapi 插件从手写 dist JS 重构为 TS 源码 + 构建（dist 不再入库，package-lock 入库） |
| `a7b8e56` | serpapi 插件资源目录迁移，安装脚本适配 |

### 阶段 2：搜索提供商切换（08-03 ~ 08-04）

| 提交 | 内容 |
|---|---|
| `96a31e2` | 默认搜索提供商从 SerpApi 换成 Yibu Brave（走 yibuapi 中转的 Brave Search）：新增 `plugins/openclaw-brave-yibu-plugin/`，删除 serpapi 插件与安装脚本，新增 `install_brave_yibu_plugin.sh`；首次加入四份中文文档（后精简） |
| `d6c5657` | 文档精简为一份对外评测用的 `WILDCLAWBENCH_QUICKSTART_ZH.md`，删三份内部过程文档 |

### 阶段 3：schema 记忆模式与检索定制（08-25）

| 提交 | 内容 |
|---|---|
| `354c7ba` | .gitignore 放开 `resources/memory` 下插件源码的可见性 |
| `14a32a3` | **schema 模式核心配置**：新增 `config/presets/entity_modeling_wildclawbench.json`（任务经验实体模型：task / environment / method / behavioral experience + episodes 兜底）与首个 project override 预设（后拆分） |
| `1b05885` | override 预设拆成 **v1 / v2 两个变体**：v1 = LLM 合并决策的原始流程；v2 = 规则图融合、单任务轨迹更少 LLM 调用（不跑高阶综合、忽略 v1 专属键）。正式实验用 v2 |
| `93cb78a` | **max_rounds 端到端打通**：SearchRequest → SDK（cli/async_client/client/core/models）→ 服务端 agentic loop；插件默认单轮 agentic 搜索（检索一次，不做充分性评估/查询改写），可用 `MINDMEMOS_SEARCH_MAX_ROUNDS` 覆盖 |
| `c5ec942` | agentic 第 1 轮的图跳数改由 `agentic.num_hops` 配置驱动（原先硬编码），loop.py 与测试同步更新 |
| `c727a34` | **episode 实体置零**：entity model 中 episodes 的 search_weight 置 0；override 注释说明"沉底"策略。此提交的配置方案后来被发现有缺陷，由未提交的服务端修复接管（见下） |

---

## 二、未提交改动 — MindMemOS 服务端（src/）

### `components/searcher/schema/schema_search_expander.py`

balanced 路径的 episode 权重置零真正生效（c727a34 只改了配置，配置是死配置）：

1. **零权重整侧跳过**：原实现 `max(1, int(top_k * weight))` 在 weight=0 时仍保留 1 个 episode 名额、照付两次召回（dense+BM25）。现改为 `run_episodes = ep_weight > 0`，episode 侧不召回、不占合并槽位；两侧全零直接返回空。
2. 召回任务从固定四个改为按需构造的 dict + `asyncio.gather`。
3. **property 路径堵漏**：新增 `_episode_entity_exclusion_filter()`，在 `episode_weight == 0` 时给 property 检索加 `must_not entity_type=episodes`。原先只排除 `input_messages`，episode 自有的 `detail` 等属性会从属性通道漏回来。

### `components/searcher/schema/_entity_weights.py`

`schema_search_apply_weights_to_ranked` 对权重为 0 的实体**直接剔除**（原为沉底）。原因：请求级最终精排只按 memory 文本重打分，episode 携带原始对话文本、交叉编码器分数反而高，沉底的 episode 会被最终精排"复活"。

---

## 三、未提交改动 — openclaw 插件（plugins/openclaw-plugin/）

### `src/index.ts`

1. **记忆读写全量落盘**：每次 search / add 往返 dump 成 `/tmp/openclaw/mindmemos-logs/<时间戳>-{search,add}.json`，随 runner 收集的 task_output 存活，供事后按任务归因记忆读写（add dump 实际会随容器早死丢失，写侧记录改由 qdrant 重建，见 collect_memory_records.py）。
2. **topK 默认 5 → 50**。
3. **请求侧搜索参数可配**：`MINDMEMOS_SEARCH_TOP_K`（默认取 config topK=50）、`MINDMEMOS_SEARCH_RERANK`（默认开）、`MINDMEMOS_SEARCH_SCORE_THRESHOLD`（默认 0.15）、`MINDMEMOS_SEARCH_STRATEGY`（默认 agentic）、`MINDMEMOS_SEARCH_MAX_ROUNDS`（默认 1）。rerank 开时向 CLI 传 `--rerank --score-threshold`。
4. add 也改走 `spawnFileJson`（原先 `spawnFileOk` 丢弃输出），把 add 结果一并 dump。

### `src/mindmemos-cli.ts`

`spawnFileJson` 增加可选 stdin 参数；删除不再使用的 `spawnFileOk`。

---

## 四、未提交改动 — scripts/wildclawbench/

### `sync_image.sh`（镜像烘焙流程，6 步 → 7 步）

1. **第 3 步 SDK 拷贝防嵌套**：GitBash/MSYS 会把 `docker cp <dir> cname:<已存在目录>` 拷成嵌套目录，先 `rm -rf` 目的地再拷。
2. **第 4 步插件覆盖改 tar 流**：`docker cp dist/.` 在 MSYS 下静默嵌套成 `dist/dist/`，插件加载器一直读旧 dist（smoke 期间实际跑的是 8 月 3 日旧插件）。改为 `(cd src && tar -cf - dist) | docker exec -i ... tar -xf -`，容器绝对路径包在 `bash -c` 字符串里防 MSYS 转换；拷后 `grep -q mindmemos-logs` marker 校验，拷错立即失败。
3. **新增第 6 步（sync add 模式，实验成败关键）**：
   - 根因：async add 只写 buffer，flush 需 rule chunker 切出完整 episode（≥50 条消息、不按说话人/时间切），单任务轨迹 12-30 条消息永远达不到 → **记忆永不生成**。
   - 修复：`openclaw config set plugins.entries.mindmemos-memory.config.addMode sync`；`add_sync` 内部 force drain，一次 add 当场抽取（一个任务轨迹 = 一个 episode），与 override 设计意图一致。
   - 配套：SDK CLI 的 `~/.mindmemos/settings.json` 设 `network.timeout_seconds=600`、`max_retries=0`（默认 30s 盖不住内联抽取的 25-60s LLM 调用）。
   - 同步固化清华 PyPI 镜像到 `/etc/pip.conf` 与 `/root/.config/pip/pip.conf`（部分任务 warmup 要 pip install 精确版本，pypi.org 本网络不可达）。

### `run_serial.sh`

新增 `RESUME_AFTER=YYYYMMDD_HHMM` 断点续跑：跳过该时刻之后已有 score.json 的任务，失败/无分任务自动重跑。

### `run_serial5.sh`（新文件，未入库）

gpt-5.5 sweep 专用变体：resume 跳过的 glob 从 `gpt-4.1-mini_*` 收紧到 `gpt-5.5_*`。起因：切 5.5 时旧 4.1-mini runner 没杀干净整夜并行跑，原 resume 逻辑扫"任意模型的已评分目录"，把 15 个已被 4.1-mini 评分的任务静默跳过。

### `collect_memory_records.py`（新文件）

按任务重建记忆读写记录：
- **recall**：各任务 task_output/mindmemos-logs/*-search.json（search 在容器存活期内完成，dump 安全）。
- **写入**：add dump 随容器早死丢失（agent_end 后 ~4s 容器被清理，sync 抽取 ~60s 未完返回），改从 qdrant entity_item_v1 按 created_at 落入各任务串行时间窗归因（run_serial 的 wait_drain 保证窗口不相交）。
- **重放归因**：`--replay-log/--replay-since` 把重放实体按 request_id 分批、按创建序 zip 回原始 add 的 event_time（qdrant created_at 是 UTC、run 目录名是本地时间，差 8h）。
- 产物：`memory_records_gpt5.5.json`（123 个任务 run、1620 实体）与 `memory_records_gpt5.5_final.json`（每任务取最新有效 gpt-5.5 run，task_7 用 0952 补救轮覆盖）。

### `replay_adds.py`（新文件）

抽取 LLM key 配额耗尽事故的恢复工具：sync add 仍返回 status=ok 但 memories=[]（"episode memory generation failed permanently" 只在 API 日志）。消息原文都在 qdrant，本脚本把零记忆 add_record 按原序重放，无需重跑任务。

---

## 五、未提交改动 — 运行时配置（config/）

### `config/mindmemos/api_keys.yaml`

新增 wildclawbench schema key 条目（`new_key.py` 生成），其 `project_override_config` 即 v2 预设内容。当前生效值相对预设的演进：

1. `search.rerank.enabled: true` + `request_timeout: 600`：开启请求级最终精排（cross-encoder 评分 + score_threshold 0.15 过滤）。`request_timeout` 必须显式设置——RerankClient 默认 5s，qwen3-reranker-4b@yibuapi 对宽候选池常超 5s。
2. `entity_weights.force_balanced_split: true` + `episode_weight: 0.0` + `non_episode_weight: 1.0`：配合服务端修复，episode 整侧出局。
3. `agentic.use_rerank: true`（2026-08-28）：开启 agentic 每轮的**引擎内重排**（RRF 融合后、top_n 截断前，池宽上限 `max_rerank_candidates: 100`）。
4. `agentic.max_rounds: 1`、`num_hops: 1`、`top_k_per_round: 30`、`top_n_per_round: 5`。

### `config/presets/project_override_wildclawbench_schema_v1/v2.example.yaml`

与 api_keys.yaml 同步：rerank 块 `enabled: true + request_timeout: 600`；entity_weights 换成 balanced + episode 置零方案（v1 原是 unified 路径 + search_weight 沉底，v2 原是注释掉的双路）；`agentic.use_rerank: true`。

### 两层 rerank 的准确认知（2026-08-28 澄清）

| 层 | 开关 | 热加载 | 失败行为 |
|---|---|---|---|
| 最终精排（评分过滤所在） | 请求体 `rerank: true`（插件恒发）+ schema 策略恒允许 + `search.rerank.enabled` | 是（每请求现读 scoped config） | 优雅回退顺序截断 |
| 轮内引擎重排 | `agentic.use_rerank` | **否**（AgenticSearchWrapper 是进程级单例，`__init__` 一次性捕获配置，改 yaml 须重启 API） | 异常直接 500 |

gpt-5.5 全量 60 任务在"最终精排开、轮内关"下产生（sweep 期 dump 的 params 字段直接证明 `rerank: true`）。2026-08-28 重启 API 后两层全开：同 query 单次 search 从 1 次 rerank 调用变 3 次，延迟 5-7s → ~20s，召回结果组成改变。

---

## 六、未提交改动 — WildClawBench 仓库

### `eval/run_batch.py`

判分 except 分支保护：`run_grading` 可能已把真实分数写进 score.json，后续崩溃（如控制台编码）不得用 error 分数覆盖——score.json 已存在则读回。起因：GBK 控制台打印 █░ 分数条触发 UnicodeEncodeError，进 except 后真实分数被覆盖（task_3/7/8 事故）。

### `src/utils/grading.py`

判分 docker exec 超时从硬编码 120s 改为 `GRADE_TIMEOUT` 环境变量（默认 600s）。05 类两个任务死于 120s 超时。

### `my_api.json`

providers 从本地代理改为 `custom` @ `https://yibuapi.com/v1`，模型 gpt-4.1-mini + gpt-5.5，**`timeoutSeconds: 600`**（单次 LLM 请求空闲超时；原先无此键，白天中转延迟下 agent 单请求等不到回复就放弃，task_7 两轮死亡）。

### `tasks/05_.../task_10_social_poster_multi_crop.md`

`timeout_seconds: 300 → 1800`（全 05 类唯一 300 低值，207k tokens 跑不完被杀）。

---

## 七、不入库的一次性脚本（%TEMP%）

- `rerun_05.sh`：05 类 4 任务补救重跑（带 `PYTHONUTF8=1` 防 GBK 崩溃）。
- `regrade_task7_0952.py`：task_7 0952 run 的判分重建——临时容器 + exec 输入 + 该 run 的 poster 产物 + 复跑任务 warmup（镜像无 PIL，grade 里 Image.open 抛 ImportError 走全 0 早退，必须 `pip install requests Pillow pymupdf`）+ `run_grading` 原参数重放。恢复分数 0.4667。
- `probe_rerank2*.py`：rerank 验证探测（鉴权用 `Authorization: Bearer`，key 从 api_keys.yaml 内部读取）。

---

## 八、镜像内 vs 宿主侧的边界

`sync_image.sh` 只烘焙 SDK（src/mindmemos_sdk）与插件 dist；**服务端（src/mindmemos）跑在宿主**（`make api`），第二节的服务端修复无需进镜像。镜像当前与本地源一致（dist hash 校验过）。若在别机复现实验：跑 sync_image.sh 后，服务端修复需随仓库工作区带走（本文档第二、三、四、五节全部未提交，迁机前须提交或打包）。
