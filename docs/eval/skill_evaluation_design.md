# mindmemos_eval Skill 测评重构设计

本文规定 Skill 测评的开发范围、运行协议和验收条件。现有入口为 `mindmemos-eval skill run/report`，具体使用方式见 README_ZH.md。

## 1. 设计原则与范围

第一期现有 `mindmemos_eval/skills/` 整体替换，不保留旧 runner、agent、SpreadsheetBench environment 或兼容转发壳。第二期第一批实现 Skill-Forest 在 ALFWorld、LiveMath、SpreadsheetBench 原生 tool-calling、SearchQA、OfficeQA 上的测评。保留现有 memory 测评和顶层 memory 命令。

只实现当前需要的流程：初始化配置 → 下载数据 → 运行测评 → 查看报告。

- 初始化由 `make init` 提供入口；下载直接用 `uv` 调用现有脚本。具体文件操作和下载继续放在 `scripts/`。
- eval 只保留 `skill run` 和 `skill report`。删除 init、data、list、check、status、resume、compare 子命令；运行前检查直接由 run 执行。
- 第一批直接调用进程内 Skill service，不建 backend 层、protocol 层、注册框架或远端 SDK 执行模式。
- 共用现有 Data、Environment、Skill-Forest training pipeline、JSON 工具和评分结果；不复制它们。
- 每次 run 创建新输出目录；指定的目录已非空就报错。eval 不实现跨进程恢复状态机、输入摘要或恢复命令。算法内部已有 checkpoint 保留，但不扩展成 eval 恢复功能。
- manage/evolve 后续有真实实现时再增加函数和文件，不预建空目录或抽象基类。

## 2. 初始化与数据下载

### 2.1 `make init`

`make init` 调用 `scripts/dev/init.py`，复制缺失的本地配置，保留所有已有文件。复制清单如下：

| 模板 | 目标 |
| --- | --- |
| `.env.example` | `.env` |
| `config/mindmemos/dev.example.yaml` | `config/mindmemos/dev.yaml` |
| `config/mindmemos_skill/dev.example.yaml` | `config/mindmemos_skill/dev.yaml` |
| `config/mindmemos_eval/dev.example.yaml` | `config/mindmemos_eval/dev.yaml` |
| `config/presets/local.project.example.yaml` | `config/presets/local.project.yaml` |
| `config/mindmemos_eval/memory_evaluation/*.example.yaml` | 同目录对应 `*.yaml` |
| `config/mindmemos_eval/dreaming_evaluation/*.example.yaml` | 同目录对应 `*.yaml` |

前五项使用明确清单，后两项只扫描指定目录；不递归复制所有名为 example 的文件，避免把说明性 preset 当成本地配置。缺失模板报错，不能显示初始化成功。脚本复用现有 workspace 的 Python，不安装依赖、不启动服务、不写 SDK 状态、不生成密钥。

控制台先逐行打印每个文件的处理结果，显示源文件和目标文件。复制成功使用 `file copied: <source> -> <destination>`；目标已存在使用 `file skipped: <source> -> <destination> (destination already exists)`。不输出文件内容。例如：

```text
file copied: .env.example -> .env
file skipped: config/mindmemos/dev.example.yaml -> config/mindmemos/dev.yaml (destination already exists)
file copied: config/mindmemos_skill/dev.example.yaml -> config/mindmemos_skill/dev.yaml
file copied: config/mindmemos_eval/dev.example.yaml -> config/mindmemos_eval/dev.yaml
file copied: config/presets/local.project.example.yaml -> config/presets/local.project.yaml

Next, configure the key information in these files:
- config/mindmemos_eval/dev.yaml: target and optimizer model names, API keys, and base URLs.
- config/mindmemos/dev.yaml and config/mindmemos_skill/dev.yaml: service model connections and required storage connections.
- config/presets/local.project.yaml: local project identity and credentials.
- .env: required service settings and HF_TOKEN when downloading gated Hugging Face datasets.
```

上面是输出格式示例；实际必须逐行打印复制清单中的全部文件，包括 memory/dreaming 评测配置，并按本次结果显示 copied 或 skipped。全部文件处理完成后，再输出配置提示；复制失败应显示失败原因并返回非零，不能显示文件已复制。

这里的初始化只复制文件；必填连接和凭据仍由用户填写。重复执行不会覆盖用户已经填写的 model、URL、key 或 HF_TOKEN。配置模型及默认值的变更同步更新对应 dev.yaml/dev.example.yaml，保留本地敏感值。

### 2.2 用户下载命令

不增加 Makefile 下载目标。用户直接用 `uv` 调用下载脚本，分组和参数直接交给脚本解析：

```bash
uv run --no-sync python scripts/download_datasets.py download memory --all
uv run --no-sync python scripts/download_datasets.py download skill --all
uv run --no-sync python scripts/download_datasets.py download skill --name alfworld
uv run --no-sync python scripts/download_datasets.py download skill --name searchqa
```

安装完成后使用 `--no-sync` 保留已安装的可选环境依赖。Makefile 仅增加初始化入口，保留现有 dev/setup 等命令，不增加分组辅助目标或参数转发逻辑。

### 2.3 下载脚本接口

修改现有 `scripts/download_datasets.py`，采用同一 parser 支持：

```bash
python scripts/download_datasets.py list skill
python scripts/download_datasets.py download memory --all
python scripts/download_datasets.py download skill --name alfworld
python scripts/download_datasets.py download skill --all
python scripts/download_datasets.py download skill --name searchqa
```

`--all` 与 `--name` 必须且只能选一个；`--name` 初期只接收一个数据集，不额外设计筛选表达式。保留 `--data-root`、`--verify-only`、`--offline`。取消 `--source-archive` 和手动归档导入分支；旧 `--benchmark/--dataset` 参数及调用文档、测试一起更新，不保留第二套参数兼容逻辑。

`--all` 覆盖来源清单中该分组的全部支持数据集，不等同于当前 Skill-Forest 的五个评测基准。实施时从来源清单删除 DocVQA；skill 除五基准外保留 HiTab、WikiTQ、SpreadsheetBench full，memory 保留四个记忆数据集。DocVQA 不在下载或评测支持范围。

批量下载串行执行。已存在目录按当前 receipt 逻辑验证，不覆盖、不重新发布不同版本。某一数据集失败后继续处理其他数据集，结尾打印 `Downloaded / Verified / Failed` 表格。需要认证、遇到配额限制或下载错误时最终返回非零，不能把不完整批量下载显示成成功。

SearchQA 统一通过 Hugging Face 下载 `lucadiliello/searchqa`，固定 revision 为 `c1a979068ba118d85467179b704031d113d689cc`。来源清单指定 `data/*.parquet`，复用现有 snapshot 下载、暂存发布和 receipt 校验。`--offline` 可使用本地缓存，正常评测仅读取已下载的本地文件。下载目录不混入官方原始 JSON 归档或实验 split 文件。

### 2.4 HF_TOKEN

在 `.env.example` 和本地 `.env` 中同步增加下载凭据变量；已有值保留，示例为空：

```dotenv
# Hugging Face token for gated or private dataset downloads.
HF_TOKEN=
```

下载脚本用现有 `python-dotenv` 显式加载仓库 `.env`，遵循现有进程环境优先的方式，并向 `snapshot_download` 显式传 token。公开数据集不强制要求 token。环境变量名称 `HF_TOKEN` 在职责明确的 `scripts/constants.py` 定义一次，由脚本引用；不在各读取点硬编码。

认证错误区分以下情况，错误文案使用英文且不输出 token 或完整认证 header：

| 错误 | 必须给用户的提示 |
| --- | --- |
| 401 或明确 token 错误 | `Set or update HF_TOKEN in <repo>/.env, then retry the same uv download command.` |
| gated repo / 403 | `Set HF_TOKEN in <repo>/.env and request or accept access on <dataset page>. A token alone may not grant access.` |
| 网络超时、限流、缓存缺失 | 输出具体网络或 offline 原因，不误报为 HF_TOKEN 缺失 |

复用 huggingface_hub 已有异常类型，不自己解析 HTTP 文本。单数据集和批量下载都使用相同错误转换；批量摘要显示需要用户处理的数据集。

## 3. 评测命令与默认流程

新增正式入口 `mindmemos-eval = mindmemos_eval.cli:main`，支持 `python -m mindmemos_eval`。skill 子命令只有两个：

```text
mindmemos-eval skill run [--algorithm skill_forest] [--benchmark livemath|alfworld|spreadsheetbench|searchqa|officeqa|all]
mindmemos-eval skill report RUN_DIR
```

`run` 默认 skill_forest + livemath + 完整固定 split + 三组对照 + 一次训练/一次测试。`--benchmark all` 顺序执行本期五个基准。运行前校验全部选中基准；输入不完整时汇总错误并退出，不提交一半训练。某一基准运行时失败则记录原因并继续其他基准，最终退出码非零。

运行命令：

```bash
# Python 3.12/3.13；安装后保留环境可选依赖。
uv sync --all-packages --extra alfworld --extra spreadsheetbench
make init

# 填写 config/mindmemos_eval/dev.yaml 的 target/optimizer 模型连接。
# 如果数据需要 Hugging Face 认证，再填写 .env 的 HF_TOKEN。
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
uv run --no-sync mindmemos-eval skill run
```

完整套件：

```bash
uv run --no-sync python scripts/download_datasets.py download skill --all
uv run --no-sync mindmemos-eval skill run --benchmark all
```

SearchQA 纳入同一批量自动下载流程。eval 只检查本次需要的数据；下载整个 skill 分组与五基准测评的范围分别明确。

必要运行参数：

| 参数 | 含义 |
| --- | --- |
| `--config PATH` | 默认 `config/mindmemos_eval/dev.yaml`，共享配置只填一次 |
| `--benchmark NAME\|all` | 默认 livemath，all 固定表示本期五基准 |
| `--algorithm skill_forest` | 第一期只支持此算法，未来沿用参数扩展 |
| `--smoke` | 固定 split 前缀 4 train / 2 val / 2 test；1 tree、1 candidate、depth/budget 1，报告标注 smoke |
| `--test-repeats N` | 默认 1；一次训练后固定版本测评 N 次；三组使用相同重复数 |
| `--arms LIST` | 默认 no_skill,initial_skill,trained_skill；只选前两组时不训练 |
| `--data-root PATH` | 默认 data/mindmemos_skill，算法和基准不再复制原始数据路径 |
| `--output-dir PATH` | 默认自动创建带时间/UUID 的新 run 目录；非空目录报错 |

四类并发及详细环境参数放 YAML，不扩展大量 CLI 覆盖开关。第一期不支持用 `--algorithm` 随意选择未实现算法、不提供独立 kind/protocol 参数、不提供单独检查命令。`report` 只读取已有结果；run 正常结束自动生成报告，因此 report 仅用于再次导出。

## 4. 最小文件架构

目标结构共七个主要 Skill 源码文件，另有包入口。不按五个基准或未来三种算法建立多层目录。

```text
Makefile
scripts/
  
  dev/download.py                # 单个/全部下载、校验、认证及错误汇总 + 环境变量名称CONSTANT
  dev/init.py                        # 复制明确的配置模板并打印操作提示

src/mindmemos_eval/
  pyproject.toml                     # 注册 console entry point
  mindmemos_eval/
    __init__.py
    __main__.py                      # 调用顶层 CLI
    cli.py                           # 现有 memory + 新 skill 分发
    llm.py                           # 保留 memory 的既有用途
    memory/                          # 保留现有实现
    skills/
      __init__.py
      cli.py                         # run/report 参数和结果展示
      config.py                      # EvalConfig、RunOptions 及校验
      typing.py                      # DatasetInputs、CaseResult、报告类型
      data.py                        # 五基准路径解析、必要转换和固定 split 核验
      runner.py                      # 检查、创建服务、对照/训练顺序及版本导出
      evaluation.py                  # 固定 SkillSet 的 rollout 和 pipeline 结果采用
      report.py                      # 指标聚合及 JSON/CSV/Markdown 报告
  tests/                             # 按实际行为组织测试，不要求逐文件镜像

config/mindmemos_eval/
  memory_evaluation/                 # 保留现有模板
  dreaming_evaluation/               # 保留现有模板
  dev.example.yaml                  # 同步维护的共享配置模板
  dev.yaml                          # 本地模型连接与环境参数

resources/mindmemos_skill_eval/
  skill_forest-<benchmark>/
    initial_skill/
    data_splits/

docs/eval/
  skill_evaluation_design.md
  README_ZH.md
```

具体文件职责：

- `config.py` 使用现有模型及 SkillForestTrainerConfig，不重新声明 optimizer 参数。保存 target/optimizer、森林参数、evaluation concurrency、默认根目录及五基准环境覆盖。定义运行参数模型并集中校验。
- `typing.py` 只定义调用间真实需要的数据：已加载任务/划分、单任务结果、基准报告、套件报告。复用 DataSample、RolloutResult、SkillVersionRecord，不加 RunPlan/RunManifest/ProtocolOutcome 等同义包装。
- `data.py` 用五个直接函数完成路径和环境参数解析，统一调用 Data.build/load/split。除 SearchQA 必需的转换外，不复制数据。转换复杂到影响可读性时才单独提取 `searchqa.py`，不先为每个基准建文件。
- `runner.py` 中的 `run()` 直接负责流程，通过 create_service、register_from_local、run_training、get_skill_version 完成训练。少量校验和路径函数就地定义；不分 preflight/backend/artifacts 层。
- `evaluation.py` 用一个 `evaluate()` 完成空 SkillSet 或冻结 Skill 的测试，复用 Environment.rollout 和已有受限并发能力；用 `read_pipeline_test()` 核验训练 pipeline 已完成的测试。
- `report.py` 只做读取和统计；不调模型、不补跑任务、不提供实验比较或恢复机制。

旧 `skills/` 的所有文件、引用和旧测试随替换清理，不保留 wrapper。删除只限于旧 Skill eval，不删除 memory，也不删除 mindmemos_skill 的正式环境和算法。现有 `config/mindmemos_eval/skill_evaluation/` 及 `skill_local_service/` 被统一配置替代时删除，更新旧文档和引用。

## 5. 配置与内部接口

### 5.1 一个共享配置文件

用户只维护 `config/mindmemos_eval/dev.yaml`；模板为同步维护的 dev.example.yaml。目标形状如下，model 字段使用既有模型配置验证：

```yaml
models:
  target:
    model: gpt-5.4-mini
    api_key: null
    base_url: null
    reasoning_effort: medium
    max_completion_tokens: 16384
  optimizer:
    model: gpt-5.4-mini
    api_key: null
    base_url: null
    max_completion_tokens: 16384

skill_forest:
  num_trees: 7
  candidates_per_node: 3
  max_depth: 3
  expansion_budget: 30
  seed: 42
  expansion_concurrency: 4
  rollout_concurrency: 4
  model_concurrency: 4

evaluation_concurrency: 4

environments:
  livemath: {}
  searchqa:
    max_turns: 1
    max_context_chars: 6000
  officeqa:
    max_tool_turns: 24
  spreadsheetbench:
    max_turns: 30
    skill_loading_mode: tool
    transactional_recalculation: false
  alfworld:
    max_steps: 50
    history_length: 2
    seed: 42
```

target 注入五个环境的 agent；optimizer 注入 SkillForestTrainerConfig.optimizer。两者独立配置，缺少任一需要的连接时在 run 开始前说明配置位置。环境覆盖按现有环境配置模型验证，拒绝未知字段。HF_TOKEN 只用于下载，模型连接仍显式取自 YAML。

本地 service 的 SQLite/home/project 由 runner 按 run/benchmark 设置，使用最小的 repository + training pipeline 配置，不要求用户启动 HTTP 服务或复制第二份评测 service YAML。非敏感默认值和字段同步维护两份 dev 文件；本地连接值保留，不写入模板。

### 5.2 函数接口

接口签名草案，非已实现 API：

```python
# config.py
load_config(path: Path) -> EvalConfig

# data.py
load_dataset(benchmark: BenchmarkName, data_root: Path, run_dir: Path) -> DatasetInputs

# runner.py
async run(config: EvalConfig, options: RunOptions) -> SuiteReport

# evaluation.py
async evaluate(
    env: Environment,
    samples: Sequence[DataSample],
    skill_set: SkillSet | None,
    output_dir: Path,
    concurrency: int,
) -> list[CaseResult]
read_pipeline_test(directory: Path, expected_task_ids: list[str]) -> list[CaseResult]

# report.py
write_report(report: SuiteReport, output_dir: Path) -> None
read_report(run_dir: Path) -> SuiteReport
```

DatasetInputs 含 DataSample 列表、ordered split IDs、环境准备字段、初始 Skill 路径和数据来源引用；RunOptions 含 benchmark 选择、algorithm、smoke、arms、test repeats、data root 和 output dir。CaseResult 含 task ID、score、success、环境专属指标、执行/评分错误、耗时、可得 usage 和原始结果路径。

不加共享注册器、抽象 evaluator、通用 operation request 或动态字符串反射。第一期 runner 直接调用 `_run_skill_forest()`；未来 manage/evolve 增加各自执行函数，真正出现重复流程时再提取共用逻辑。

## 6. 默认数据路径与准备

默认位置严格沿用下载脚本。资源路径以项目根目录定位；用户 CLI 路径相对当前工作目录，YAML 用户路径相对配置文件。原始下载目录不写派生文件，防止下载校验失效。必须转换的文件写本次 run 的 `prepared/`，不建跨 run 缓存或准备状态库。

| benchmark | 默认解析及复用接口 | train / val / test |
| --- | --- | --- |
| ALFWorld | data/mindmemos_skill/alfworld；资源 games.json 直接作为 Data 输入；读取下载的 configs/base_config.yaml，替换数据路径、关闭 CUDA | 39 / 18 / 134 |
| LiveMath | data/mindmemos_skill/livemath/data；直接复用四个月 JSON 和 month:no ID | 35 / 18 / 124 |
| SpreadsheetBench | data/mindmemos_skill/spreadsheetbench/spreadsheetbench_verified_400；读取 dataset.json 与同层 workbook root，固定 tool-calling 环境 | 80 / 40 / 280 |
| SearchQA | data/mindmemos_skill/searchqa；必要时转换为本次 run 的 prepared/searchqa/items.json | 400 / 200 / 1400 |
| OfficeQA | data/mindmemos_skill/officeqa/officeqa_full.csv；docs_root 指向 benchmark 根目录，复用 loader 的嵌套 parsed 文档支持 | 50 / 24 / 172 |

全部 split 使用 `resources/mindmemos_skill_eval/skill_forest-<benchmark>/data_splits/splits.json`，保持 ID 和顺序；初始 Skill 来自同组 initial_skill。不生成新划分、不复制数据到 skill_forest 历史目录。

路径解析只支持固定来源的明确布局，多个候选或缺少文件时报错，不遍历后随意选第一个 JSON。校验 download receipt、全部 split ID、互斥性及外部文件存在性。

SearchQA 的固定 split 使用 `lucadiliello/searchqa` 原生 `key`，保持原项目 400/200/1,400 的题目及顺序。`data_splits/` 只保留 `splits.json`。评测从本地 Parquet 筛选记录，原样保留 `question`、`context` 和 `answers`，仅将 `key` 用作 task ID；必要的 JSON 格式适配写入本次 run 的 prepared 目录。无需官方数字 ID 映射、逐题文档索引或截断清单。

## 7. 执行与结果协议

run 直接顺序执行：读取配置 → 创建新 run 目录供必要转换使用 → 加载数据并检查全部输入/模型连接/真实运行依赖 → 每基准执行对照和训练 → 自动写报告。检查失败时只保留诊断及必要的转换文件，不创建服务或提交训练。

1. no_skill、initial_skill 在相同 test IDs 上执行；使用空 SkillSet 或固定初始 Skill，结果不传给 optimizer。
2. 训练通过现有 Skill-Forest pipeline 执行。仅 train pool 用于森林优化；按当前 trainer 合同，外部 val 不参与优化，test 只用于冻结后的测试。
3. pipeline 已执行 trained Skill 第一轮 test。核验任务顺序、完成结果、frozen snapshot 和版本关系后采用，避免再跑一次。
4. test_repeats 大于 1 时，加载同一个精确发布版本，新增测试目录执行后续次数；不同 repeat 不复用完成结果。记录各次实际环境 seed，不声称 provider 的重复响应可完全复现。
5. 本地 state/project 按 run/benchmark 隔离。完整导出 initial/trained version 和算法配置，保证报告能定位实际输入和产物。

每个 repeat 使用全新工作目录。模型、工具或评分基础设施错误单独记录，不伪装成正常 0 分；正常完成但答案错误仍是 0 分。runtime 初始化失败直接导致该基准失败。保留评分参考数据与公开 task/workspace 的既有分离，不复制 golden 到 agent workspace。

默认 SpreadsheetBench 不评分前重算，不要求 LibreOffice。显式打开重算时检查真实依赖并记录协议变化。五基准 model 参数和环境配置来自同一最终配置，不声称默认值自动保证与论文所有实验细节一致。

## 8. 输出与报告

```text
output/mindmemos_eval/skill/<time>-<uuid>/
  config.json                         # 脱敏的实际配置、参数、来源和 split 引用
  summary.json
  summary.csv
  report.md
  <benchmark>/
    prepared/                         # 仅在需要转换时存在
    state/                            # 本地 SQLite 和 service home
    initial_skill.json
    trained_skill.json
    training/                         # 原 pipeline 产物，trained 第一轮 test 在此
    no_skill/repeat-001/
    initial_skill/repeat-001/
    trained_skill/repeat-002/          # 后续 repeat 才额外执行
    results.json                      # 指向采用的原始结果，含明确错误和覆盖率
```

不再创建 manifest/status/events 恢复体系；保留正常日志和现有算法内部输出即可。测试结果按基准完成后保存，发生中断时用户可读取已存在结果，但不提供 eval 恢复承诺。

报告按 benchmark/arm/repeat 展示：planned/completed/error 数量、覆盖率、mean score、success rate、可得环境指标、墙钟耗时、逐任务耗时总和和 token usage。usage 缺失为 null，不能写成 0。训练与测试成本分开；未提供 optimizer token 时标记缺失，不以 rollout token 冒充总训练 token。

SearchQA 主 score 为现有 EM，并保留 F1/sub-EM；OfficeQA 是官方二元评分，不把其 f1 误解为 token F1。重复测试报告每次得分及均值/标准差；三组增益只在同任务和相同协议下计算。不同 benchmark 保持独立行，不输出混淆指标的统一总分。

基础设施错误或中断使报告 incomplete，不能只用成功执行子集分母输出完整准确率。report 命令只读原始结果并重新聚合，不补跑测试，不比较其他 run。

## 9. 后续扩展及实施顺序

manage：未来增加固定 Skill collection → organize → frozen graph snapshot → recommend → 下游任务测评。已有 relevance 标注时才计算检索指标，不能为无标注任务虚构 Recall@k。

evolve：未来增加独立 adaptation trajectories → evolve → frozen version → held-out test。test 反馈不参与演化。沿用 run/report 命令，只扩展 algorithm 选择和对应配置；确有不同协议需求再增加必要字段。

实施顺序：

1. 增加 `make init` 及初始化脚本，修改下载 parser 并提供直接调用的 uv 命令；移除 DocVQA 和手动归档分支，接入固定版本的 HF SearchQA 下载；补 `.env` 和 `.env.example` 的 HF_TOKEN，完成批量下载摘要及认证提示。
2. 下载 SearchQA HF Parquet 并核验原生 key；将五个下载目录直接接入 Data/Environment。
3. 在独立临时实现目录完成新的最小 skills 实现，不导入旧 skills；迁移顶层入口、配置、文档和调用测试，验证后删除旧实现并恢复正式路径。用户已授权旧设计整体替换，无需维护旧 CLI 兼容。
4. 验证默认 LiveMath 和五基准流程、三组对照、pipeline 测试采用、重复测试与报告。

必要检查集中在真实行为：init 不覆盖已存在配置；uv 下载命令和分组 all/name 校验；SearchQA 固定版本下载、缓存读取和完整性核验；DocVQA 不出现在支持清单；批量下载失败时继续且非零退出；HF_TOKEN 的 .env 加载和 401/403 提示且不泄露凭据；prepare 不改变下载目录；固定 split 和私有参考隔离；重复测评不会复用结果；run 不重复执行 pipeline 第一轮测试；memory 原有入口继续工作。

新 Skill eval 单元测试放 src/mindmemos_eval/tests；init/download 脚本测试放 tests/scripts；跨模块联调放 tests/integration。更新原下载测试的错误目录归属和引用。真实模型 smoke 与完整实验单独明确执行范围，本设计优化不启动模型、不清库、不安排监控。


## 通用环境测试与报告约定

先提供 `mindmemos-eval skill test`，通过原生注册名选择 Environment 与 Data，或传入原生 task/reward_data 样本。一次只运行 no-skill 或用户给定的一组静态 Skill，两个模式互斥；不增加自动对比。实现与配置入口见评测 README。

测评报告固定包含：①实际执行数据量（同时记录选中数和正常完成数）；②墙钟耗时；③输入、输出、总 token；④执行或评分异常数（例如 API 超时，正常 reward=0 不属于异常）；⑤成功率；⑥逐任务 JSONL 轨迹。

直接环境测试的成功率按环境 reward.success 为真的数量除以正常完成数量，全部异常时为 null。token 未返回时标记缺失并保留已知部分，不把缺失当作零。JSONL 在每个任务结束时追加，保留任务、原生交互轨迹、评分、耗时、用量或错误；失败任务没有返回的轨迹及用量保持缺失。新测评入口均遵循以上报告约定。
