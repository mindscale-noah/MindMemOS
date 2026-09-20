# Benchmark environments

新增环境沿用 LiveMath 的 `Data → Task + RewardData → Environment.rollout → RolloutResult` 接口。数据划分仍使用公共 `Data.split`；每次 rollout 的持久化、计时和恢复仍由公共 `Environment` 负责。

恢复是可选能力：现有环境默认 `supports_resume = True`。新环境可声明 `supports_resume = False`，此时调用方必须传 `resume=False`；请求 `resume=True`（含默认值）会在组装和文件操作前抛出 `NotImplementedError`。禁用恢复仍保留输入校验、审计记录及执行计时。

| 注册名称 | 协议 | `reward.detail` |
| --- | --- | --- |
| `searchqa` | context QA；EM、token F1、substring EM | skillopt 原始 `[EVALUATION RESULT]` 文本 |
| `officeqa` | 本地文档工具；官方固定版本的二元评分 | skillopt 原始失败原因；成功为空字符串 |
| `spreadsheetbench` | `workspace_tools` 或 `bash_only`；分别使用独立交互函数 | 统一 `[POST-EXECUTION VERIFICATION]` 逐格报告 |
| `alfworld_bounded_history` | 单个独立 game，保留有限数量的历史 observation/action | skillopt 原始 episode 失败原因；成功为空字符串 |

SpreadsheetBench 数据入口加载 Verified 400：每题必须恰好匹配一个 `*init*.xlsx` 和一个 `*golden*.xlsx`，不要求文件名前缀一致；缺失或匹配到多个文件时明确报错。当前数据加载器不支持原始 912 题多用例版本。

SpreadsheetBench 两种交互模式的 `score` 是通过 case 的比例，`success`/`hard` 要求全部 case 通过；`soft` 也保留原始比例。OfficeQA 的 `f1` 按参考代码等于官方二元分数，不是 token F1。

两种交互模式共用同一 reward 计算：采用统一范围解析与行优先扫描，支持带逗号的 sheet 名、整行/整列范围及省略结束列的范围；不重算公式，只比较缓存值。空 answer position 明确报错。评分与逐格报告使用相同的范围解析。加载时会结合 `answer_sheet` 补全无 sheet 的范围。公开 task 包含数据集给出的 answer-position 提示，但不含 golden workbook 路径或答案值。

## 使用

```python
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

data = Data.build("searchqa", {"data_path": "/path/to/items.json"})
samples = data.load()
splits = data.split(samples, split_file="/path/to/splits.json")
sample = splits["test"][0]

env = Environment.build("searchqa", {"agent": {"model": "your-model"}})
result = env.rollout(
    sample.task,
    "/path/to/output/one-rollout",
    reward_data=sample.reward_data,
)
print(result.reward.score)
print(result.reward.detail)
```

不同并发 rollout 必须使用不同 `work_dir`。恢复时需要相同的 task、私有 reward data、skills 和配置。数据加载器将输入文件和评分文件的内容 hash 纳入 task 或 reward data；文件变化后应重新加载数据。

环境的 `rollout` 和 `_run` 均接收 `list[Skill]`；公共流程按 task 调用一次 `Skill.assemble`，直接把返回的完整 Skill 交给环境；Skill 自己替换任务副本的 bundle，保留 references 等资源和导出能力。环境不区分静态或动态 Skill。TreeSkill 环境负责其 workbook policy 协议，不在 env 内复制 tree routing、skill synthesis 或训练调度逻辑；这些仍属于算法层。

## SpreadsheetBench 执行依赖

安装 `mindmemos-skill[spreadsheetbench]`。本地运行能力统一放在 `mindmemos_skill.infra.execution`，两个表格环境直接复用，其他 env 也可使用，不依赖 Docker。

```python
# Existing virtual environment (macOS/Linux).
execution = {"python_command": ["/path/to/.venv/bin/python"], "timeout_s": 120}

# Existing uv project; use an absolute project path.
execution = {"python_command": ["uv", "run", "--project", "/path/to/project", "python"]}

# Windows interpreter with optional PowerShell.
execution = {
    "python_command": ["C:/project/.venv/Scripts/python.exe"],
    "shell_command": ["powershell.exe", "-NoProfile", "-Command"],
}
```

`python_command` 是 argv 列表，默认当前服务的 Python 解释器。模型在 shell 中调用 `python` 或 `python3` 时，也通过临时启动脚本使用该配置，不需要改写生成的 Python 源码。uv 项目应提前安装所需依赖（openpyxl、pandas）；路径参数使用绝对路径。

`shell_command` 默认在 macOS/Linux 使用 `/bin/sh -c`，Windows 使用 `cmd.exe /d /s /c`。也可显式指定 PowerShell 或已安装的其他 shell。命令需要符合所选 shell 的语法；不会将 Bash 脚本自动翻译为 PowerShell。每条命令启动新进程，工作目录文件跨轮保留，内存、cd 和 export 不跨命令保留。

`execution.timeout_s` 是命令默认超时；workspace shell 的显式 `timeout_ms` 可覆盖它；超时后 POSIX 清理进程组，Windows 使用 taskkill 清理进程树。read/write/edit 工具保留工作目录边界检查；shell 是本地进程，不提供容器级文件系统或网络隔离。golden 不会写入提示词或复制到公开工作目录，但本地进程的操作系统权限不因此受到限制。

`max_turns` 控制 react 模型轮数。评分仍读取公式缓存值；如需公式重算工具，由指定运行环境提供。

## OfficeQA

仅支持本地文档模式，无需配置模式字段。提供 `glob`、`read`、`grep` 三个只读工具，只能读取 task 明确列出的公开文本文件。数据目录可包含 `jsons/*.json`，或 `treasury_bulletins_parsed/jsons/*.json`；根据 `source_docs` 中的页码选择公开 parsed pages。HTML 表格内容转成保留单元格分隔的文本。这里不保证工具输出和原始 prompt 字符一致；严格一致性范围是评分和 reward 详细文本。

## ALFWorld

安装 `mindmemos-skill[alfworld]`，准备官方数据和配置。将官方 YAML 解析后的字典传入 `simulator`，完整设置进入 rollout identity。数据 manifest 指向具体 `gamefile`；每次 rollout 独立创建、reset、step 并在 `finally` 中关闭 simulator。`history_length` 控制 prompt 中最近的 observation/action 数量，完整 trajectory 仍会保留。

模型或基础设施异常直接向上传播，不沿用旧代码把 provider error 转成 `look` 动作继续计分的做法。有效模型响应缺少 action tag 时，仍保留参考实现的 `look` fallback。

## 验证

```sh
.venv/bin/python -m pytest tests/mindmemos_skill/test_benchmark_envs.py -q
.venv/bin/python scripts/skill/verify_env_parity.py \
  --skillopt-envs /path/to/skillopt-self-validate/skillopt/envs \
  --treeskill-env /path/to/legacy/mindmemos_skill/envs/registered_envs/spreadsheetbench
```

差分脚本直接读取指定 checkout 的参考 scorer 和反馈生成函数，比较分数以及 UTF-8 文本，并输出参考文件 SHA-256。它不导入旧项目的模型客户端，也不请求模型。测试包括本地 workbook 的真实读取与评分；rollout 测试用确定性 agent 和执行替身覆盖交互、恢复和异常边界，另有本地 Python/uv 启动与超时的真实进程测试；Windows 分支使用模拟测试，尚未在 Windows 实机验证，ALFWorld 集成也需单独验证。


## SpreadsheetBench 模式配置

唯一环境和数据注册名均为 `spreadsheetbench`；旧的独立 bash 环境入口已移除。分支由行为命名，交互函数不读取私有评分数据；统一入口负责工作簿准备、分派与评分。

```yaml
# Workspace file/shell tools (defaults)
interaction_mode: workspace_tools
tool_agent: openai_tool
skill_loading_mode: tool
max_turns: 30
```

```yaml
# Single-workbook bash action loop
interaction_mode: bash_only
tool_agent: text_action
skill_loading_mode: system_prompt
max_turns: 100
stagnation_limit: null
```

以上为 env options 片段，仍需提供 `agent.model`。统一配置的 `max_turns` 默认是 30；bash 示例按原始实验 pipeline 显式设为 100。`stagnation_limit` 只在 `bash_only` 中使用，默认 null 禁用，与原始循环一致；设置整数属于额外的提前停止策略。新配置包含交互模式，已有 rollout 请使用新的 `work_dir`。

## SpreadsheetBench react 对齐

`spreadsheetbench` 对齐 Skill-Forest T5/T7 实验中的 workspace react，模型通过工具修改 workbook；不是 `spreadsheetbench_python` 的 multi 代码生成模式。`tool_agent` 可选 `openai_tool`（默认）或 `text_action`，二者统一使用 `OpenAIChatAgent`，通过解析、格式错误提示和 observation 回调切换协议，环境持有工具执行与评分。`interaction_mode="bash_only"` 使用独立的 `run_bash_only` 函数，保留单 workbook、前 5 行预览、仅 bash action、输出缺失时提醒一次及可选的重复动作停止行为（默认关闭）；必须显式设置 `tool_agent="text_action"` 和 `skill_loading_mode="system_prompt"`。默认 `interaction_mode="workspace_tools"` 使用独立的 `run_workspace_tools` 函数，支持两种 tool agent 和两种 Skill 加载方式。

```python
env = Environment.build("spreadsheetbench", {
    "agent": {"model": "your-model", "max_completion_tokens": 16384},
    "tool_agent": "openai_tool",
    "max_turns": 30,
    "skill_loading_mode": "tool",
    "execution": {"python_command": ["/path/to/.venv/bin/python"], "timeout_s": 120},
})
```

| 配置/行为 | Skill-Forest 参考 react | 当前实现 |
| --- | --- | --- |
| 模型轮数 | 默认 30 | 默认 30 |
| 工具集合 | read/write/edit/shell/skill | 相同 |
| Skill 加载 | 默认 skill 工具，可 system_prompt | 相同；tool 结果先确认，再以 user 消息传入 Skill |
| 完成条件 | 无 tool calls 时结束 | openai_tool 相同；text_action 使用文本完成标记 |
| completion tokens | 实验配置 16384 | 共用模型配置默认 4096，以上示例显式设为 16384 |
| reasoning effort | 实验配置 medium | workspace_tools.yaml 显式设为 medium |
| 模型重试 | actor 最多 6 次，2–30 秒带抖动，含 HTTP 425；底层另有重试 | workspace 预设 6 次外层尝试、equal jitter、SDK 2 次重试；完整错误包装行为仍有差异，见下文 |
| 命令超时 | shell 默认 120 秒，可由 timeout_ms 覆盖 | 相同；显式 timeout_ms 不再受默认值上限截断 |
| 整题超时/并发 | 实验调度层配置 | 不放入单题环境，由调用层负责 |
| 执行位置 | 宿主机 case 工作目录 | 本地 case 工作目录；Python/uv 和 shell 可配置，文件持续保留 |
| 最终详细反馈 | react 原分支不追加完整 verification report | 两种模式共用逐格报告格式，结束后写入 reward.detail 和 trajectory，不送入本轮 actor |

数据仅支持 Verified 400 单用例，含官方编号不一致的兼容；公开模型消息不含 golden。未要求加载的 Skill 内容不会提前注入 tool 模式的初始提示词。Skill 文本按公共 skill_text 规则组装，暂不支持随 Skill 携带额外脚本文件。


## 在其他环境中复用执行能力

```python
from pathlib import Path
from mindmemos_skill.infra.execution import ExecutionConfig, execute

config = ExecutionConfig(python_command=["/path/to/.venv/bin/python"])
workspace = Path("/path/to/task-workspace")
result = execute(["python", "solution.py"], workspace, config)
# For shell commands, use the configured platform-specific prefix.
result = execute([*config.shell_command, "python solution.py"], workspace, config)
```

公共层只负责 Python/uv 启动、shell 进程执行、输出捕获和超时清理，不依赖任何 env、Agent 或 workbook 类型。环境负责准备工作目录、定义工具、解释执行结果及计算 reward。已有 `execution` 配置结构保持不变，原 SpreadsheetBench 内的 executor/constants 文件已移除。

## 固定实验数据与划分

实验划分位于 `resources/mindmemos_skill/<algo>/<benchmark>/splits.json`，作为本地资源随代码
提交到 GitHub，不通过下载脚本获取。官方数据统一下载到 `data/mindmemos_skill/<benchmark>/`，
不按算法复制。下表描述已有实验资源及其历史输入格式，不代表官方下载结果已经转换为这些格式。
划分保留原实验 ID 和顺序，不重新随机抽样；`val` 是验证集，空数组表示原协议没有独立验证集。

| 算法 | benchmark | train / val / test | 数据入口 |
| --- | --- | --- | --- |
| skill_forest | searchqa | 400 / 200 / 1400 | `items.json` |
| skill_forest | livemath | 35 / 18 / 124 | `data/`，四个月的官方 JSON |
| skill_forest | officeqa | 50 / 24 / 172 | `officeqa_full.csv`；`docs_root=corpus`，含正文与 parsed JSON |
| skill_forest | spreadsheetbench | 80 / 40 / 280 | `dataset.json`；`data_root` 为该 benchmark 目录 |
| skill_forest | alfworld | 39 / 18 / 134 | `items.json`；`game_root` 为该 benchmark 目录 |
| skill_forest | docvqa | 107 / 53 / 374 | `items.json` 与 `images/` |
| treeskill | spreadsheetbench | 200 / 0 / 200 | `dataset.json`，Verified 400 原始顺序 |
| treeskill | spreadsheetbench_soft_hard | 0 / 0 / 912 | `dataset.json`，912 题、2,529 对评测用例 |
| treeskill | hitab | 0 / 0 / 1584 | `dataset.json` 与转换后的 spreadsheet |
| treeskill | wikitq | 0 / 0 / 2810 | `dataset.json` 与转换后的 spreadsheet |

Skill-Forest 使用原实验仓库发布的 ID/path manifests。TreeSkill 主实验按 Verified 400
前 200 条演化、后 200 条测试；迁移评测单独存放，不能与主实验视为同一个互斥任务集合。
Soft/Hard 从完整 2,729 个用例中排除演化所用的 200 个 case-1 输入与答案，排除规则保存在
`excluded_cases.json`，并按原实验规则规范化三个含多余空格的输入文件名。
HiTab 和 WikiTQ 的历史清单对应转换后的 XLSX；官方下载保留原始格式，不能直接套用历史
转换结果的 checksums。这些实验资源与下载脚本独立，下载时不读取 split、排除规则或算法配置。

### 官方数据下载

来源集中在 `config/datasets/sources.json`。下载入口按 skill / memory 分组及 dataset 选择，不接受算法参数。
使用 Python 3.12+；Hugging Face 来源需要安装 `huggingface_hub`：

```bash
python -m pip install huggingface_hub
python scripts/download_datasets.py list --benchmark skill
python scripts/download_datasets.py download --benchmark skill --dataset spreadsheetbench
python scripts/download_datasets.py download --benchmark skill --dataset spreadsheetbench --verify-only
python scripts/download_datasets.py download --benchmark skill --dataset hitab --data-root /path/to/data
```

| benchmark | 官方来源 | 下载后格式 |
| --- | --- | --- |
| livemath | LiveMathematicianBench/LiveMathematicianBench（Hugging Face） | 固定四个月的 `data/*/qa_*_final.json` |
| officeqa | databricks/officeqa（Hugging Face） | 官方 CSV 和 `treasury_bulletins_parsed/` |
| spreadsheetbench | KAKA22/SpreadsheetBench（官方项目链接的 Hugging Face） | 解压后的 `spreadsheetbench_verified_400/` |
| spreadsheetbench_full | RUCKBReasoning/SpreadsheetBench | 官方 912 题完整归档，保留全部用例 |
| alfworld | alfworld/alfworld releases | 全部文本游戏与逻辑文件，保留发布目录结构 |
| hitab | microsoft/HiTab | 固定版本的官方仓库内容，保留 `data/` 等目录 |
| wikitq | ppasupat/WikiTableQuestions | 固定版本的官方仓库内容，保留 `csv/`、`data/` 等目录 |
| searchqa | nyu-dl/dl4ir-searchQA README 链接的 Google Drive | 从官方入口取得压缩包后导入 |
| docvqa | DocVQA 官方网站链接的 RRC portal | 登录官网下载 Single Document VQA 压缩包后导入 |

SearchQA 的官方入口是 Google Drive，DocVQA 需要 RRC 登录；脚本不回退到第三方 Hugging Face
整理版。取得官方压缩包后可执行（多个包重复传参）：

```bash
python scripts/download_datasets.py download --benchmark skill --dataset searchqa --source-archive /path/to/SearchQA.zip
python scripts/download_datasets.py download --benchmark skill --dataset docvqa --source-archive /path/to/train.tar.gz --source-archive /path/to/val.tar.gz
```

OfficeQA 如需访问许可，通过 `HF_TOKEN` 或 Hugging Face 已有登录状态认证。
Hugging Face 和 GitHub 来源锁定 revision；ALFWorld 锁定发布版本。
官方手动下载包没有统一不可变 revision，导入时记录压缩包 SHA-256。
每个目录的 `download_receipt.json` 记录来源、归档哈希和本地文件校验和；它用于检查后续
本地文件是否变化，不冒充发布方提供的校验和。仅在下载、解压完成后发布目标目录。
已有目录仅校验，不覆盖或静默更新版本；更换来源版本时指定新的 `--data-root`。
`--offline` 只允许 Hugging Face 缓存或本地官方压缩包，直接 URL 的首次下载需要联网。

下载结果不生成或下载 `splits.json`，不抽样、不排除演化用例，也不转换为算法专属格式。
运行环境时按上表中的官方路径配置 data loader；需要格式转换的数据集应另行准备转换步骤。
数据本体仍由 `data/` 忽略规则排除；下载脚本、来源配置和已有 split 资源随代码管理。

SpreadsheetBench 统一评分版本为 `spreadsheetbench-cached-values-v2`；旧结果目录的请求配置不同，应使用新 `work_dir`。`bash_only` 的 `finished` 和 `stop_reason` 单独记录交互终止状态，`fail_reason` 统一表示评分失败原因。差分脚本仅对 Skill-Forest 的标准范围比较通过标记，对 TreeSkill 的非空范围比较分数与首个差异；不再要求旧版逐格反馈文本一致。


## 原始仓库 actor 协议核对（2026-09-20）

参考源是用户指定的 [TreeSkill-colleague-share](https://github.com/ChengzhePiao/TreeSkill-colleague-share)，
本地 checkout 提交 `bc3b0a9d55870d8ad77c38839c4fa5baf111e638`，相关 tracked 文件无修改。
Skill-Forest 参考本地 `skillopt-self-validate`，HEAD 为
`5d041ffe6e63409d7f05135e7c705cf85357900e`，但 `react_agent.py`、`azure_openai.py` 有未提交修改，
因此此次对齐的是当前文件快照，不能将其表述为纯 HEAD 版本或历史实验运行时的确定版本。

配置文件均为 Environment options，不含数据路径或算法调度：

| 文件（config/mindmemos_skill/spreadsheetbench/） | 实际适用范围 |
| --- | --- |
| workspace_tools.yaml | Skill-Forest workspace actor，gpt-5.4-mini，30 轮，16384 completion tokens，medium reasoning |
| bash_only_instruct.yaml | TreeSkill CLI/preloaded actor 的 instruct 请求设置，Qwen3.5-35B-A3B，seed 41，100 轮 |
| bash_only_thinking.yaml | 同一 actor 的 thinking 请求设置，seed 41，100 轮 |

TreeSkill pipeline 的 MAX_TURNS 是 100；类构造默认的 20 不是实验生效值。
原始 generation JSON 写 temperature=1.0，但每次调用的 ModelSettings 默认值覆盖成 0.7；这里记录生效值。
原始 max_tokens=None，因此 bash 预设不发送 completion token 上限；stop 为 `Observation:`。
原始循环没有 stagnation cutoff，缺失输出时仅在剩余预算内提醒一次，格式错误也占用总轮数。
预设保留 instruct/thinking 对应的 top_p、presence_penalty、top_k、enable_thinking 和 HTTP timeout。
endpoint、API key、实际部署别名和 Python 依赖环境仍须由运行方提供，未启动真实模型调用。

验证入口（不请求模型）：

```bash
.venv/bin/python scripts/skill/verify_spreadsheet_protocols.py \
  --workspace-source /path/to/skillopt-self-validate \
  --bash-source /path/to/TreeSkill-colleague-share
```

脚本直接执行参考 actor，逐轮比较同一模型回复下的 messages，而非只比较复制后的常量。
覆盖 4 项提示词/工具 schema、7 组 workspace 对话、7 组 bash 对话、6 项 workspace 工具观察。
workspace 对话包含多个 Skill 导出后的 tool/system_prompt 两种加载方式：正文仅以空行连接，
不添加标题或路径说明；reference 保留原相对路径，同路径不同内容在写入前报错。
bash 场景包括 baseline/preloaded、缺失输出后的恢复、格式错误、预算耗尽和连续重复动作。
测试输入使用共同绝对工作目录和 input.xlsx/output.xlsx；bash 命令在对话对照中使用确定性替身。
SDK 请求测试另外验证三个 YAML 的实际请求体、thinking 去除、token 上限省略和 retry waits。
这些是有范围的确定性验证，不是对全部输入/故障路径的等价证明。

参考文件 SHA-256：

- Skill-Forest `skillopt/envs/spreadsheetbench/react_agent.py`：`34ee1f148ef22afe1ecf53ef5367c8cd53eb62460444201e3ec66896ad15919f`
- TreeSkill `src/react_agent/agent.py`：`18600c6b1b453d16719603bfb45c93ddc1d45fd9314ddedc9b35afb3447fa8cb`
- TreeSkill `src/react_agent/models.py`：`392bbdb46f252901089171c032e7af1ff6a3def355f4add1448d749a752773cb`

### 不能宣称完整实验完全一致的边界

1. 按用户决定，两种模式共享 cached-value reward 和 TreeSkill 范围解析，均不进行公式重算。
   TreeSkill 的其他 evaluator/recalc 变体、原始成功状态与评分关联、Skill-Forest 旧解析不是复刻目标。
2. TreeSkill 完整实验包含 tree routing、retrieval、fusion 和具体 Skill 选择。本 env 只接收已经组装的 SkillBundle，
   bash 分支对应 CLI/preloaded 执行协议，不实现 `cli_skill_tree_only_routed` 的算法行为或额外资源文件。
3. 原始 TreeSkill 保留 workbook 文件名；统一 env 使用 input.xlsx/output.xlsx，并使用受控 Python launcher。
   绝对路径、可用依赖、shell 环境和资源布局可能影响模型输出，消息对照只在共同路径下成立。
4. Provider 异常链尚未逐项复刻：原始 TreeSkill 有请求缓存与 context-length 专门终止；
   Skill-Forest 有底层异常包装、额外失败等待及进程级重试覆盖。当前共享模型客户端保留显式重试配置，
   不保证这些异常路径或请求耗时完全相同。HTTP timeout 600 是 workspace 参考 SDK 默认值，
   不是实验调度的 task timeout。
5. 数据 split、outer task timeout、workers、训练/融合预算属于调用方配置。
   TreeSkill 默认 seed 41（另有 42/43），Verified 400 按顺序 200 train/200 test；
   Skill-Forest 参考 T7 seed42 日志为 80 train/40 val/280 test、workers40、exec_timeout600、实际 task_timeout660s。
   Skill-Forest YAML 默认 workers24 与该日志不同。上述完整实验设置未由这三个 env options 自动建立。

因此当前结论是：指定场景中的 actor 消息协议通过直接差分，评分按用户要求统一；
完整实验复现仍需固定算法配置、数据 manifest、Skill 资源、模型服务和参考源码快照，不能仅切换 env mode 就保证分数相同。
