# SWE-bench Verified 双层 Agent 记忆实验

本框架位于 `src/mindmemos_eval/mindmemos_eval/swebench/`。已有训练轨迹和部分无记忆基线轨迹落盘；完整的 50 题基线、记忆提取、带记忆测试和官方评分仍需分别确认终态。

## 实验协议

- 从用户提供的本地 Verified JSONL 中，按 instance_id 排序后使用固定 seed 打乱，取前 50 题构建记忆，随后 50 题测试；保存不相交的题目清单。这里的 train/test 是实验自定义划分，并非上游数据集的官方训练/测试划分。
- 每题只有一次物理 rollout。父 agent 只能使用 `todo` 和 `delegate`；前者维护计划，后者下发任务并同步等待子 agent。父 agent 不持有执行 shell 的工具。
- 每次 delegate 创建新子对话，所有子 agent 顺序使用该题同一个 Docker 容器，因此代码修改保留。子 agent 只有容器内 `shell`，可查看、编辑、执行检查；没有宿主机 shell、宿主目录挂载或 Docker socket。
- 训练 rollout 不搜索记忆。提取阶段父轨迹使用 `metadata.extract_type=plan`，子轨迹使用 `metadata.extract_type=experience`。
- 同一道原始题目的多个子轨迹合并为一次 experience 提取请求，task 使用原始题目而不是子任务标题。因此默认检索的 5 个任务对应原始 SWE-bench 题目。
- 父记忆提取使用 `parent-memory-input.json`：根据委派工具返回的 trajectory_id，将对应子轨迹插入父轨迹的返回位置之前，保持同步执行顺序。父消息全部保留；子 agent 最后一个 assistant 轮次及随后工具返回全部保留，更早的每条子消息按 600 字符上限保留首尾、中间截断（含截断标记）。原始轨迹文件不改写。
- 这份视图通过 `metadata.trajectory_view=chronological_parent_v1` 提交；Lite 的 plan 分支保留其中 system 消息作为轨迹证据，并跳过通用工具消息二次截断。提取器能看到压缩子过程，不意味着执行时父 agent 能看到它们。未匹配到父委派返回的子轨迹会报错，不猜测拼接位置。
- 测试中父、子 **每次模型调用前**各调用一次 search，并把结果追加到本次请求的初始 user prompt。基础对话不保存反复累积的检索块。查询为该 agent 的原始指令，子 agent 的查询包含原题及委派任务。
- search 使用 `task_top_k=5`，从返回的 `tasks` 逐组保留任务和附属经验；`top_k=100` 是独立的每任务经验数量上限，不是任务数量。返回不足 5 组时使用实际结果，不补造经验。
- 测试只读记忆，不将测试轨迹回写。模型调用、记忆请求和整题 rollout 均不自动重试。
- 当前只输出 patch，不调用官方评分器。`finished` 仅表示 agent 返回最终答复，`completed` 仅表示执行与补丁导出完成，均不代表题目 resolved。训练提取也不使用官方通过/失败标签。

## 准备条件

1. 已可用的 Python 环境中安装本仓库的 eval 与 SDK 包及声明依赖。框架复用已有 OpenAI 客户端和记忆 SDK，没有增加依赖安装动作。
2. 本地 Verified JSONL，每行包含 `instance_id`、`repo`、`base_commit`、`problem_statement`。至少 100 个不重复实例。其他字段（包括参考 patch 和 test_patch）不会进入 split 或 agent 输入。数据来源由使用者保证；框架记录源文件 SHA256，但不会联网验证数据身份。
3. Docker CLI 和 daemon，以及每题已经准备好的本地镜像。镜像应含干净的对应 base_commit、代码及完整依赖。缺镜像直接报错，框架不会 build/pull；容器禁网。镜像内须有 bash、git、timeout，运行用户须能写仓库。
4. 本地镜像映射 JSON，键为 instance_id，值为本地镜像名称，例如 `{"django__django-12345": "local/swe-django-12345:ready"}`。这是格式示例，不是可下载镜像。
5. 如镜像依赖 Conda 激活，在 `docker.command_prefix` 中配置镜像实际支持的初始化命令；默认不假定依赖环境路径。
6. 已启动、支持当前任务经验接口的记忆服务。Lite 服务需使用 `trajectory_add` 提取管线和 `task_experience_search` 搜索管线，默认搜索模式为 `experience`。这里复用服务现有提取提示词，不另造提示词。

父规划记忆和子执行记忆使用两个**全新且独立的项目**。当前任务经验搜索按项目隔离，不能依靠 agent_id 过滤角色。因此要在服务端分别创建并配置两个项目的 API key。客户端的 `plan_project_id` / `experience_project_id` 仅记录期望项目，HTTP 请求实际项目由 API key 决定；框架无法仅凭这两个字符串验证服务端绑定。整个实验期间不得向这两个项目加入其他数据。

凭据通过以下环境变量提供，不写入 YAML 或输出配置：

| 环境变量 | 用途 |
| --- | --- |
| `SWEBENCH_PARENT_API_KEY` | 父模型密钥 |
| `SWEBENCH_PARENT_BASE_URL` | 父模型 OpenAI-compatible API 地址 |
| `SWEBENCH_CHILD_API_KEY` | 子模型密钥 |
| `SWEBENCH_CHILD_BASE_URL` | 子模型 OpenAI-compatible API 地址 |
| `SWEBENCH_MEMORY_BASE_URL` | 记忆服务地址 |
| `SWEBENCH_PLAN_MEMORY_KEY` | 规划记忆项目密钥 |
| `SWEBENCH_EXPERIENCE_MEMORY_KEY` | 执行记忆项目密钥 |

父子模型可以相同，但两个记忆项目和对应 key 必须不同。

## 四阶段运行

先复制 `config/eval/swebench_verified.example.yaml` 并修改路径、模型和项目名。相对路径以运行时当前目录为基准。输出目录必须是全新目录；split 后配置被冻结，修改参数应使用新实验目录。

在已配置好 eval 包依赖的环境中，从仓库根目录运行。`scripts/swebench/run_four_phases.sh` 接受一个阶段、同一份冻结配置和本地凭据文件；`prepare` 只在全新输出目录执行一次，它冻结划分而不覆盖自备镜像映射。先根据生成的 split 准备覆盖 100 题的本地镜像映射，再运行 `train`。用官方题目镜像时，配置的 `image_map_path` 必须位于新输出目录内，并显式启用 `docker.pull_missing` 和 `docker.restore_base_commit`；此时以 `prepare-official` 代替 `prepare`，自动生成镜像映射和评分数据。两种准备方式只选其一。

```bash
cp config/eval/swebench_verified.example.yaml config/eval/swebench_verified.yaml
# Edit model IDs, dataset path, output path, memory project IDs, and Docker settings.
# Keep credentials.env outside the repository. It needs both model endpoints/keys;
# extraction and memory testing also need SWEBENCH_MEMORY_BASE_URL plus separate
# SWEBENCH_PLAN_MEMORY_KEY and SWEBENCH_EXPERIENCE_MEMORY_KEY values.

bash scripts/swebench/run_four_phases.sh prepare  config/eval/swebench_verified.yaml /path/to/credentials.env
bash scripts/swebench/run_four_phases.sh train    config/eval/swebench_verified.yaml /path/to/credentials.env
bash scripts/swebench/run_four_phases.sh extract  config/eval/swebench_verified.yaml /path/to/credentials.env
bash scripts/swebench/run_four_phases.sh baseline config/eval/swebench_verified.yaml /path/to/credentials.env
bash scripts/swebench/run_four_phases.sh test     config/eval/swebench_verified.yaml /path/to/credentials.env
```

如果使用官方题目镜像并按上段要求设置了输出路径和 Docker 选项，把第一条 `prepare` 替换为 `prepare-official`。已有 `recovery06` 等输出目录已冻结，不再执行任何准备命令。

`train` 采集 50 题轨迹并导出 `train-trajectories.json`；`extract` 从这些轨迹向两个独立项目写入记忆；`baseline` 对固定的 50 道测试题执行无记忆 rollout；`test` 对同一批题执行每次模型调用前检索记忆的 rollout。后两步各自产生 `predictions.jsonl` 和 `summary.json`，仍需官方 harness 判分才能报告 resolved 准确率。`train` 不自动触发 `extract`；提取本身也可能调用记忆服务配置的模型。运行前应确认对应模型、服务端提取模型及费用。

如果使用源码而非已安装的 eval/SDK 包，需要将 `src/mindmemos_eval`、`src/mindmemos_sdk` 加入现有 `PYTHONPATH`，并保证其依赖已安装。不要误用独立 Lite 环境作为完整 eval 环境。

## 输出与恢复

每题目录 `train/<instance_id>/` 或 `test/<instance_id>/` 包含：

- `parent.json`、`child-001.json` 等：原生消息、工具参数/返回、角色、抽取类型、父子关联、终止状态。
- `calls.jsonl`：逐次模型响应和 token 用量；`searches.jsonl`：测试时逐次查询、任务记忆组和实际注入的 user prompt。
- `attempt.json`、`environment.json`、`result.json`：一次执行标记、实际镜像 ID/提交、执行状态。
- `model.patch`：相对题目 base_commit 的补丁，包含新文件；训练题另有两份 `extraction-<role>.json` 提取回执。

每个阶段最终生成 `predictions.jsonl`，字段为 `instance_id`、`model_name_or_path`、`model_patch`，供后续官方 SWE-bench harness 判分使用。`summary.json` 记录完成/失败数和未进行官方评分的事实。

重新运行 train/test 会跳过已有完整 result 的尝试，包括已记录的失败，绝不偷偷重跑。若发现有题目目录但没有 result，说明进程中断，框架会停止并要求人工核对。新题仍只执行一次。不应同时启动同一实验的多个写入进程。

extract 会跳过已完成的角色回执；发起写入前落盘 pending 标记。若网络中断后服务端是否完成未知，后续不会盲目重发，需要人工核对服务端与本地回执。正常提取可以返回零条经验，回执如实记录。test 要求所有 50 题的两类提取回执均完成，而且每类至少有非零的总返回条数；这不等价于保证每题都有可检索经验。

目前没有实现主 agent 历史压缩或记忆 token 总预算限制；5 是任务数上限，并非 token 上限。若父模型上下文很小，后续需另加明确的预算分配策略，以免不透明截断影响实验解释。

## 仅采集训练轨迹并汇总 JSON

批量入口为 `scripts/swebench/collect_train.py`，可直接使用仓库中已安装依赖的 Python 环境运行。它自动补齐 eval/SDK 的源码导入路径，不读取其他项目的凭据。先在配置中填好模型、数据路径、镜像映射和全新输出目录，并通过上述 `SWEBENCH_*` 环境变量提供模型连接信息。纯训练采集不需要启动记忆服务。

```bash
.venv/bin/python scripts/swebench/collect_train.py --config config/eval/swebench_verified.yaml --mode prepare
.venv/bin/python scripts/swebench/collect_train.py --config config/eval/swebench_verified.yaml --mode check
.venv/bin/python scripts/swebench/collect_train.py --config config/eval/swebench_verified.yaml --mode run
```

- `prepare`：冻结 50/50 划分与配置；输出目录已存在时拒绝覆盖。
- `check`：检查模型配置、凭据变量和全部 50 个本地镜像，不调用模型。
- `run`：额外逐个启动临时容器，检查全部训练环境的 base_commit 与干净状态，通过后才开始模型调用。任务并发为 1，每题的子 Agent 也顺序执行。不会下载镜像、提取记忆或执行测试集。
- `export`：只重新读取已保存的轨迹，生成汇总；适合中断后审计，不调用模型。

输出根目录中的 `train-trajectories.json` 含 50 条任务记录、每题原始父子轨迹、工具参数与返回、模型响应、token 用量、补丁和完整性错误；同时记录实际响应模型名、源文件与划分 SHA256，并生成汇总文件自身的 `.sha256`。`collection_complete=true` 要求全部 50 题执行完成且通过基础证据检查，仍不表示官方评分通过，也不要求 Agent 在轮数上限前自行结束。

缺失、失败、中断会分别记录；遇到执行异常停止批次并在退出前汇总已有结果，不自动重试。再次 `run` 会沿用原 runner 的恢复规则，跳过已有 result 的题目（包括失败），遇到未完成的 attempt 要求人工检查。使用进程锁防止本脚本并发写入同一输出目录；不要同时从旧 CLI 启动同一实验。

现有单题 review 的临时脚本曾在容器内恢复 base_commit；正式 runner 要求镜像启动时已处于该提交。不能直接把未经准备的官方镜像映射给正式采集，否则运行前容器检查会失败。

## 50 条训练采集 → 50 条无记忆基线（关闭 thinking）

本次队列配置为 `config/eval/swebench_train50_baseline50.yaml`，完整入口为 `scripts/swebench/run_experiment.py`。父子都请求 `deepseek-v4-flash`；服务返回模型名另行记录。显式发送 `thinking.type=disabled`，正式调用不发送 `max_tokens`。父最多 12 轮，每题最多 4 次委派，每个子最多 20 轮；任务与模型调用并发均为 1。每题一次 rollout，不读写记忆，不自动重试。纯基线阶段名为 `baseline`，原 `test` 阶段仍指记忆增强测试。

```bash
uv venv outputs/swebench-harness/.venv
uv pip install --python outputs/swebench-harness/.venv/bin/python 'swebench==4.1.0'

# Use a fresh output directory for a new run. Do not repeat prepare on existing output.
.venv/bin/python scripts/swebench/run_experiment.py \
  --config config/eval/swebench_train50_baseline50.yaml \
  --env-file /path/to/credentials.env --mode prepare

.venv/bin/python scripts/swebench/run_experiment.py \
  --config config/eval/swebench_train50_baseline50.yaml \
  --env-file /path/to/credentials.env \
  --harness-python outputs/swebench-harness/.venv/bin/python --mode run
```

凭据文件接受已有 `SWEBENCH_*` 模型变量，或使用同一模型服务的 `MINDMEMOS_AGENT_API_KEY`、`MINDMEMOS_AGENT_BASE_URL` 作为两个角色的后备值。凭据不会写入配置或输出文件。启动时用最小真实请求检查模型服务；探测 token 用量单独存入 `probe.json`，不计入训练或测试轨迹统计。

此配置显式开启 `docker.pull_missing`、`docker.restore_base_commit` 和 `docker.remove_downloaded`：按 SWE-bench 官方命名拉取 x86_64 题目镜像，在一次性容器内恢复任务提交并检查工作区，每题结束后移除该生命周期下载的镜像。原有镜像保留，不执行全局清理。镜像实际 ID 保存在每题 `environment.json`。其他配置默认保持本地镜像模式，不隐式下载。

输出根包含：

- `train-trajectories.json`：50 条训练题的完整父子轨迹；每题结束刷新。
- `baseline-trajectories.json`：50 条测试题的无记忆轨迹。
- `progress.json`：当前阶段与完成状态；`failure.json`：停止时的脱敏错误。
- `baseline/predictions.jsonl`：全部测试补丁。
- `grading/`：官方评分日志、逐题报告与聚合报告。
- `baseline-accuracy.json`：准确率为官方 resolved 数 / 50。评分基础设施错误或缺失题目存在时，`grading_complete=false`、`accuracy=null`，不能当成最终准确率。空补丁按未解决计入分母。

官方评分单 worker、每题测试超时 1800 秒，使用独立安装的 `swebench==4.1.0`。参考补丁与测试补丁仅存在于独立的评分数据文件中，不传给 Agent。若需单独重新执行评分入口，使用 `scripts/swebench/grade_baseline.py --config ... --harness-python ...`；不要改动同一 run ID 对应的预测补丁，官方 harness 会复用已有报告。

### 后续改为服务默认 thinking 的运行

用户后续明确要求不传 thinking 控制参数。此轮使用 `config/eval/swebench_train50_baseline50_default.yaml` 和新输出根 `outputs/swebench-verified/train50-baseline50-defaultthinking-20260923`，完整命令如下：

```bash
.venv/bin/python scripts/swebench/run_experiment.py \
  --config config/eval/swebench_train50_baseline50_default.yaml \
  --env-file /path/to/credentials.env \
  --harness-python outputs/swebench-harness/.venv/bin/python --mode run
```

已创建的输出目录不要再次 prepare。探测与正式请求均不发送 `thinking`、`enable_thinking`、`reasoning_effort` 或 `max_tokens`；保留服务返回的 reasoning_content，不因存在推理内容而终止。其余轮数、并发、无记忆与官方评分设置不变。`lineage.json` 记录与旧轮的关系，50/50 划分保持一致，旧失败产物不纳入新一轮统计。此轮结果应标注“服务默认 thinking”，不能标注“关闭 thinking”。

### 10 题并发

配置 `config/eval/swebench_train50_baseline50_c10.yaml` 将 `task_concurrency` 设为 10。训练和无记忆测试各自使用同一个容量为 10 的题目 worker 池，两个阶段顺序执行；每题内部的子 Agent 仍串行，因此 API 请求并发上限为 10。官方评分仍为 1 个 worker。

任一题出现基础设施或模型异常后，停止领取新题，等待已在执行的题目落盘后退出；不会自动重试。已有完整结果的题目跳过。此次通过新目录继承并校验已完成结果，`lineage.json` 记录继承文件哈希及经用户授权重新采样的中断题，原始目录保持不变。
