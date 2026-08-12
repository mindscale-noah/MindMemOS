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
```

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

顶层 `method` 选择 adapter，`environment` 选择 dataset/Env 组合；`parameters` 分组名不参与 CLI，只展开叶子。例如：

```text
evaluation.test_rollouts: 2  -> --test-rollouts 2
rollout.max_concurrent_rollouts: 16 -> --max-concurrent-rollouts 16
```

`test_rollouts=2` 表示测试集的每个 task 各跑两次；只想测试两个 task 时使用 `test_limit=2`。

演进算法本身不声明环境白名单。以 `skill_grpo_with_replay_buffer` 为例，内置环境名会映射到默认的已注册
Dataset 和同名 Env；也可以通过 `dataset_ref`、`env_ref` 分别选择其他已注册组件，并通过
`dataset_options`、`env_options` 传入组件配置。非内置环境所需的可选依赖放入
`launcher.extra_dependencies`，不写入算法支持列表。

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
