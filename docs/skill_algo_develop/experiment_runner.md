# MindMemOS Skill 实验运行与配置

新 Agent 处理实验问题时，按以下顺序定位：

1. 先读本文件和目标 YAML，不要从历史 Shell 脚本猜参数。
2. 用 `--dry-run` 查看最终 runner、adapter、参数、run ID 和输出目录。
3. 按 `run_mindmemos_skill_experiment.sh → run_experiment.py → runners/{evolve|trace2skill}.py → experiments/registry.py → adapter` 追踪调用。
4. 算法实现查 `src/mindmemos_skill/mindmemos_skill/algos/`；dataset、Env 分别查包内 `datasets/`、`envs/`；CLI、输出和评测编排只查 `scripts/mindmemos_skill/experiments/`。

## 唯一入口

目录固定为 `config/mindmemos_skill/<实验方法>/<环境>/<配置名>.yaml`。统一入口只需要一个配置文件：

```bash
scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/skill_grpo_without_replay_buffer/alfworld/default.yaml
```

默认使用 `.skill.env` 时，本仓库常用的完整形式为：

```bash
UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/<method>/<environment>/<name>.yaml
```

不要直接调用 `runners/evolve.py`、`runners/trace2skill.py` 或具体 adapter；它们是统一入口的内部实现。

## 数据集与资源

配置只依赖本仓库中的路径：大型数据集放在被 Git 忽略的 `data/mindmemos_skill/`，固定的数据划分和
初始 Skill 放在 `resources/mindmemos_skill/`。首次运行前使用跨平台 Python 下载器：

```bash
uv run --package mindmemos-skill --extra dataset-download python \
  scripts/benchmark_download/download_mindmemos_skill_datasets.py
```

只下载一个或多个数据集：

```bash
uv run --package mindmemos-skill --extra dataset-download python \
  scripts/benchmark_download/download_mindmemos_skill_datasets.py alfworld

uv run --package mindmemos-skill --extra dataset-download python \
  scripts/benchmark_download/download_mindmemos_skill_datasets.py livemath spreadsheetbench

uv run --package mindmemos-skill --extra dataset-download python \
  scripts/benchmark_download/download_mindmemos_skill_datasets.py officeqa
```

OfficeQA 是 gated Hugging Face 数据集；下载前需要接受 `databricks/officeqa` 条款，并通过 `HF_TOKEN`
提供有权限的账号令牌。下载器会落盘 `officeqa_full.csv`、本地 Treasury bulletin 文本语料和按页解析 JSON。
若用 `--data-root` 改变 OfficeQA 下载位置，实验还需同时覆盖 `environment_options.docs_root`。

OfficeQA 默认使用本地文档工具。要按 controller-managed 协议启用 Serper Web Search，在 `.skill.env`
中配置 base URL 和 key；官方 Serper 的配置如下，也可以把 base URL 换成遵循同一协议的其他地址：

```dotenv
OFFICEQA_SERPAPI_PROTOCOL=serper
OFFICEQA_SERPAPI_BASE_URL=https://google.serper.dev
OFFICEQA_SERPAPI_API_KEY=<your-key>
```

使用 Yibu 时改为：

```dotenv
OFFICEQA_SERPAPI_PROTOCOL=yibu
OFFICEQA_SERPAPI_BASE_URL=https://yibuapi.com/serpapi
OFFICEQA_SERPAPI_API_KEY=<your-key>
```

然后对任一 OfficeQA 配置追加：

```bash
--set environment_options.env_options.search_mode=custom_search
```

`serper` 使用 `POST {"q": ..., "num": ...}` 与 `X-API-KEY`；`yibu` 使用 GET 查询参数与
`Authorization: Bearer`。两者都不支持 Chat Completions。

脚本可在 Windows、Linux 和 macOS 的 Python 3.11–3.13 环境运行，默认目录正好对应仓库内 YAML；
可用 `--data-root <目录>` 改变下载根目录，此时运行实验时也要通过 `--set dataset.data_root=<目录>` 覆盖。
下载是幂等的，完整数据已存在时会跳过；只有明确需要刷新缓存时才使用 `--force`。

资源布局、上游 revision 和划分数量见 `resources/mindmemos_skill/README.md` 及各数据集的
`split_manifest.json`。

## YAML 结构与参数解析

配置的顶层结构如下：

```yaml
version: 1
method: skill_grpo_without_replay_buffer
environment: alfworld
launcher:
  env_file: .skill.env
parameters:
  dataset:
    data_root: path/to/data
  models:
    target_model: openai/gpt-5.4-mini
  training:
    epochs: 4
```

`parameters` 下的分组只用于提高可读性，叶子键会转换成同名 CLI 参数，例如 `training.batch_size` 转为 `--batch-size`。不同分组不能出现同名叶子键。`run_id` 和 `output_dir` 默认自动生成为 `<environment>_<method>_<timestamp>` 和 `outputs/<environment>/<method>/<run_id>`；需要固定时可在任意参数分组中显式设置 `run_id`、`output_dir`，字符串可使用 `{timestamp}`、`{method}`、`{environment}`、`{run_id}` 模板变量。字符串中的 `$NAME` 或 `${NAME}` 从进程环境或 `launcher.env_file` 展开。

顶层 `method` 选择 adapter，`environment` 只作为内置组件默认值和 artifact 标签；真正执行的组件由
`dataset_ref` 和 `env_ref` 决定。`parameters` 分组名不参与 CLI，只展开叶子。例如：

```text
evaluation.test_rollouts: 2  -> --test-rollouts 2
rollout.max_concurrent_rollouts: 16 -> --max-concurrent-rollouts 16
```

`test_rollouts=2` 表示测试集的每个 task 各跑两次；只想测试两个 task 时使用 `test_limit=2`。

实验方法不声明环境白名单，也不会因为顶层 `environment`、Dataset 或 Env 名称是自定义值而拒绝运行。
内置环境名会映射到默认的已注册 Dataset 和 Env；自定义环境必须通过 `dataset_ref`、`env_ref` 选择已注册
组件，并通过 `dataset_options`、`env_options` 原样传入组件工厂。自定义组件不会收到 `data_root`、
`split_dir`、`max_turns` 等内置 benchmark 兼容参数，所需参数必须全部写入对应 options。

外部 Python 包提供组件时，把触发 `@register(...)` 的模块写入 `launcher.component_modules`；该包必须已在
运行环境中，或通过 `src/mindmemos_skill/pyproject.toml` 中已声明的 optional-dependency extra 安装，并把
extra 名写入 `launcher.extra_dependencies`：

```yaml
version: 1
method: skill_grpo_without_replay_buffer
environment: customer-benchmark
launcher:
  env_file: .skill.env
  extra_dependencies: [customer-benchmark]
  component_modules: [customer_benchmark.mindmemos_components]
parameters:
  dataset:
    dataset_ref: customer_dataset
    dataset_options: {manifest: data/customer/splits.json}
    initial_skill: resources/customer/SKILL.md
  environment_options:
    env_ref: customer_env
    env_options: {endpoint: "http://127.0.0.1:9000"}
    max_turns: 20
```

`component_modules` 在实验子进程启动后导入，因此模块及其依赖必须能由最终 `uv run` 环境加载。对于直接
放入 `registered_datasets/`、`registered_envs/` 并由父包导入的内置扩展，不需要填写该字段。

每个 rollout 的交互预算统一使用 `environment_options.max_turns`：一轮表示环境驱动的一次模型决策。
例如 ALFWorld 的 `max_turns: 50` 最多执行 50 次动作，SpreadsheetBench 的 `max_turns: 30` 最多生成
30 轮回复/工具调用，LiveMath 的 `max_turns: 1` 只生成一次答案。所有 built-in EnvConfig 和实验 YAML
都只使用 `max_turns`，实验配置、adapter CLI 和 EnvConfig 不接受其他同义字段。

## 临时覆盖与 dry-run

临时覆盖参数无需复制 YAML：

```bash
scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/skill_grpo_without_replay_buffer/alfworld/default.yaml \
  --set training.epochs=1 \
  --set limits.train_limit=2
```

只校验并查看最终命令：

```bash
scripts/run_mindmemos_skill_experiment.sh --config <config.yaml> --dry-run
```

`skill_grpo_with_experience_validation` 的“上一 batch 无效经验反馈”默认关闭。启用时，只汇总当前 batch
中实际执行过注入验证且被门控拒绝的经验（包括性能持平的拒绝项），记录注入前后正确数和劣化数，并只把
该摘要注入紧邻的下一 batch 经验提取 prompt；不会累计更早 batch：

```bash
scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/skill_grpo_with_experience_validation/alfworld/default.yaml \
  --set algorithm.previous_batch_invalid_experience_feedback=true
```

该算法还可通过 `algorithm.max_tool_observation_chars` 限制经验提取 prompt 中单个工具返回的长度。超过限制时
只保留开头和结尾，并在中间写入截断标记；rollout 实际看到的内容和持久化的原始轨迹不受影响。
SpreadsheetBench 默认配置为 `24000`，设为 `null` 可完全关闭该限制。

排错时必须先看 dry-run 中的 `method`、`environment`、`run_id`、`output_dir` 和最终 `command`。它能发现参数名错误、环境不匹配、错误 Skill 路径以及加载了哪个 family runner，但不会发起模型请求。

## 独立测试 Skill

独立测试使用 `method: skill_evaluation`。`dataset.skill` 可以指向 `SKILL.md` 或包含它的目录；真正的 no-skill 测试通过删除该值并启用 `no_skill`：

```bash
scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/skill_evaluation/alfworld/default.yaml \
  --set dataset.skill=null \
  --set dataset.no_skill=true
```

测试 ALFWorld 默认 initial Skill，并让每个测试任务执行两次：

```bash
UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/skill_evaluation/alfworld/default.yaml \
  --set evaluation.test_rollouts=2
```

测试统一从对应 `TaskDataset.test_tasks()` 取任务，通过注册 Env 执行；运行时在 stderr 显示 rollout 进度条、正确数、异常数和平均 reward。结果写入 `<output_dir>/test/summary.json`、`results.jsonl` 和 `skill.json`。`skill.json` 在 no-skill 模式下为 `null`，每条 rollout 的 `skill_content_hashes` 为空。

所有正式 evolve 配置必须把最终 Skill 跑一次 test；`trajectory_evidence_patch` 的 trace2skill runner 也会在候选生成后立即测试候选，未产生候选时测试原 Skill。`summary.json.test` 是统一测试摘要，不能只凭 `final_skill.md` 判断实验完成。

## 环境变量、凭证与输出

`OPENAI_API_KEY`、`OPENAI_BASE_URL`/`OPENAI_ENDPOINT` 等密钥和端点放在 `.skill.env` 或进程环境中，不写入实验 YAML。launcher 先复制当前进程环境，再加载 `launcher.env_file`，因此 `.skill.env` 中同名变量会覆盖终端里 export 的值。诊断时不要输出 key 明文，只确认来源、长度和短 hash。实验成功后，展开且不含密钥的配置会保存到输出目录的 `experiment_config.yaml`。

默认输出示例：

```text
outputs/alfworld/skill_evaluation/alfworld_skill_evaluation_20260811-155224/
├── experiment_config.yaml
├── summary.json
└── test/
    ├── summary.json
    ├── results.jsonl
    ├── skill.json
    └── workspace/
```

## 代码所有权与新增方法

`scripts/mindmemos_skill/runners/` 固定只保留 `evolve.py` 和 `trace2skill.py` 两个家族入口。实验 CLI、评测编排和方法注册表位于 `scripts/mindmemos_skill/experiments/`；`mindmemos_skill` 包只提供算法、数据集、Env、Agent 和公共类型。新增算法时更新脚本侧注册表、添加 YAML 和配置构建测试，不再增加算法专属 runner 或 Shell 脚本。
