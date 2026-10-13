# AgentSkillOS 端到端测评

`agentskillos` 环境执行完整任务链路：SkillSet 选择与组装 → 导出 Skill 及附属资源 →
Codex Agent 读取输入并生成实际产物 → 官方评分器检查产物 → 返回任务成功 reward。

## Reward 口径

- `reward.score`：官方 `TaskEvaluationResult.passed` 转为 0 或 1。
- `reward.success`：官方 `passed` 原值。
- 汇总 `task_success_rate`：已完成评分任务的成功比例，即逐任务 reward 均值。
- `reward.detail`：JSON 文本，`task_evaluation` 保留官方总分、每项检查和通过结果；
  `retrieval` 保存实际注入的 Skill 顺序、标签来源，以及 Recall、nDCG、MRR 等诊断指标。

```json
{
  "score": 1.0,
  "success": true,
  "detail": "{\"task_evaluation\": {\"passed\": true, \"total_score\": 80.0, ...}, \"retrieval\": {\"ranked_skill_ids\": [...], \"label_source\": \"official_task_skills\", \"metrics\": {\"Recall@10\": 0.5, ...}}}"
}
```

检索命中不决定任务是否成功：召回完全正确但未生成合格产物，主 reward 仍为 0；
召回未命中但 Agent 完成任务并通过官方检查，主 reward 为 1。

官方客观评分将结构检查加权为 0–100 分，达到 **60 分**即通过。检查包括文件存在、格式、关键词、
文档页数和 JSON 字段等。该通过率不等同于对内容质量或语义正确性的全面判断。
官方另有成对 LLM 比较和 Bradley–Terry 质量排名；本环境的任务通过率使用官方客观评分，未自动执行质量排名。
依据：[官方说明](https://github.com/ynulihao/AgentSkillOS/blob/c3cfae10ea34a13973c461273ad9c5cc4aa907fe/benchmark/AgentSkillOS_bench/README.md)。

## 获取可复现数据和评分器

```bash
.venv/bin/python scripts/agentskillos/download.py --output data/agentskillos/source
uv pip install --python .venv/bin/python -e 'src/mindmemos_skill'
```

下载脚本固定官方 Git revision `c3cfae10ea34a13973c461273ad9c5cc4aa907fe`，稀疏拉取 `benchmark/`，
包含 30 个任务、输入资产和原始评分代码。重复执行校验 revision 和工作树；不自动切换版本。
原始评分函数没有复制或重写，由独立 Python 子进程从指定 checkout 导入。

该版本将任务 4 所需的 `ml_project/` 放在 `task_data/data_computation_task5/`，适配器显式将它提供给任务 4；
任务 5 是纯提示词任务，不注入该项目。企鹅数据被放到任务提示词要求的 `artifacts/penguins.csv`。
其他有效输入按原目录结构提供。完整加载后应有 30 个唯一任务、5 个带本地输入的任务。

执行依赖是现有 `CodexAgent` 与已安装的 Codex CLI。生成 PDF、视频、网页等所需的软件、网络或模型能力，
由执行机器及选定 Skill 提供；安装评分依赖并不等于所有任务的产物生成依赖均已安装。
运行涉及模型调用，应在明确端点、模型、范围和预算后启动。

## 图算法接入状态

旧 SkillManageAlgorithm 与专属 SkillSet 已清理。AgentSkillOS 环境继续支持通用 SkillSet；新图算法接入前不提供旧算法示例。

## 批量运行入口

将环境配置保存为 JSON，例如 `/path/to/agentskillos.json`：

```json
{
  "benchmark_root": "data/agentskillos/source",
  "agent": {"model": "YOUR_MODEL", "timeout_s": 1800}
}
```

凭据和地址在 Agent config 的 `api_key`、`base_url` 字段中显式提供，不读取环境变量。
不要把真实凭据写入提交的配置文件。CLI 本身不读取 `.env`，需要在调用前导出环境变量。

```bash
# Baseline without Skills
PYTHONPATH=src/mindmemos_skill .venv/bin/python scripts/agentskillos/run.py \
  --config /path/to/agentskillos.json \
  --selection none --output outputs/agentskillos/no-skills-001

# Execute the first 5 entries of a previously generated skill_manage ranking
PYTHONPATH=src/mindmemos_skill .venv/bin/python scripts/agentskillos/run.py \
  --config /path/to/agentskillos.json \
  --selection rankings --skills data/agentskillos_eval/skill_pool_200 \
  --rankings outputs/retrieval/rankings.json --limit 5 \
  --output outputs/agentskillos/skill-manage-001
```

CLI 支持 `none`、`all`、`rankings` 三种显式选择方式，串行执行任务；在线检索使用上面的 Python SkillSet 接口。
`rankings.json` 格式为 `{"task-id": ["skill-directory-id", ...]}`，任务集合必须与所选测评任务完全一致。
只读取池的直接子目录，嵌套 SKILL.md 保留为资源。`--split-file` 使用共享格式 `{"test": ["task-id", ...]}`，
`--split` 选择分组；同一个 split 文件不同分组不得重叠。

结果目录记录：

- `run-inputs.json`：环境配置、任务和私有评分数据、Skill 快照及固定排名；应作为私有审计文件。
- 每条任务 `audit/request.json`、`audit/result.json`、`audit/official-score-*.json`：恢复依据及官方评分证据。
- `workspace/attempt-*/`：独立执行根目录；公开输入及 `.agents/skills/` 在此，最终产物位于其 `workspace/` 子目录。
- `summary.json`：任务成功率、评分覆盖率、成功数、已评分 ID、独立错误列表和检索诊断均值。

执行异常、超时和评分器异常会抛出并记入错误列表，不伪装成任务零分。
正常产物不满足检查才得到零分。存在未评分任务时 summary 为 `partial`，必须同时报告 `scoring_coverage`，
不能将已评分子集的成功率当成完整数据集结果。CLI 有错误时返回非零状态。

使用同一命令加 `--resume` 恢复；任务、配置、候选 Skill 或排名变化会被拒绝。
每次实际执行创建新 attempt 目录，保留旧尝试但不把旧文件混入评分。
官方评分只扫描产物目录，Skill 资源和输入文件不会因为同名被误当成输出。
代码指纹纳入环境配置，输入文件内容有哈希且在复制时复核；更新评分源码后须重新构造环境并用新目录。

评分规则、参考标签仅供私有评分，不写入执行提示词或公开文件。现有 Codex CLI 的 workspace-write
并不是严格的文件读取隔离；如需对抗主动读取私有文件的执行器，应在外部容器/沙箱中限制可读挂载。
本适配不会把 `.env` 复制给 Agent。

## 检索诊断与历史对比

默认使用官方任务 `skills` 作诊断标签，标记 `official_task_skills`。它与旧报告 `core_gt_ids` 不保证相同。
需要旧口径时为 Data 配置 `relevance_path`（CLI 为 `--relevance`），此时标记 `legacy_core_gt_ids`，
校验标签与任务完整对应。标签不参与执行任务或官方通过判定。

独立工具 `scripts/agentskillos/evaluate.py` 保留原有 12 项检索指标及新旧排名比较：

```bash
PYTHONPATH=src/mindmemos_skill .venv/bin/python scripts/agentskillos/evaluate.py \
  --tasks data/agentskillos_eval/tasks.jsonl \
  --relevance data/agentskillos_eval/relevance.json \
  --rankings outputs/retrieval/rankings.json --baseline data/agentskillos_eval/gos.json \
  --protocol retrieval --output outputs/retrieval/comparison.json
```

该工具不运行任务、不产生环境 reward。`legacy_replay` 仅是旧种子驱动的检索诊断，不能解释为独立端到端结果。
之前以 nDCG 为主 reward 的 `agentskillos` 配置已移除，不允许旧配置静默继续使用；旧结果保留，新环境需新目录。

## 验证范围

测试包含真实官方评分器与实际文件，覆盖 reward 与召回解耦、60 分阈值、执行失败、评分错误、
完整 Skill 资源注入、输入校验、恢复与新 attempt、以及原生 Codex CLI 适配路径。
执行模型在自动化测试中使用替身，不产生外部模型费用；这些测试通过不代表 30 个真实任务已完成模型测评。
