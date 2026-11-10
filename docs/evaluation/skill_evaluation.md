# MindMemOS Skill Evaluation

[简体中文](skill_evaluation_ZH.md) · [Memory evaluation](memory_evaluation.md)

Install the dependencies and initialize the configuration from the repository root. Skill evaluations run in the local process.

```bash
uv sync --all-packages
make init
```

## CLI

```bash
uv run --no-sync mindmemos-eval skill show envs
uv run --no-sync mindmemos-eval skill show pipelines
uv run --no-sync mindmemos-eval skill evaluate train --help
```

Choose `train` for `skill-forest` or `tree-skill`, `manage` for `skill-graph`, and `evolve` for `offline-evolv`. Use `--config FILE` to select a YAML file. Use `--env ENV` to select a supported environment. `--output-dir DIR` overrides `output_dir` in the YAML.

### Evaluation outputs

Each pipeline evaluator resolves its arguments and lets the selected algorithm write its own results and report. Find these files in the run's output directory.

| Evaluation | Where to read the results | Report contents |
| --- | --- | --- |
| Skill-Forest | `report.md`, `summary.json`, `summary.csv` | Control-arm results, repeated-run statistics, matched gains, training and failures |
| TreeSkill | `report.md`, `summary.json`, `trajectories.jsonl` | Held-out pass rate, execution coverage, errors, and usage |
| Offline Evolution | `report.md`, `summary.json` | Collection and retest scores and usage, with per-task evolution outcomes |
| Skill Graph · `retrieval` | `summary.json`, `rankings.json` | Macro-average and per-task retrieval metrics, plus the rankings |
| Skill Graph · `accuracy` | `summary.json`, `selected_skills.json`, `tasks/` | Accuracy, scoring coverage, per-task errors, and execution artifacts |
| Single-environment `direct` | `report.md`, `summary.json`, `tasks/trajectories.jsonl` | Pass rate, execution coverage, and usage for one fixed-Skill or no-Skill condition |

## Skill-Forest

### 1. Prepare the datasets and runtime

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
uv run --no-sync python scripts/download_datasets.py download skill --name alfworld
uv run --no-sync python scripts/download_datasets.py download skill --name spreadsheetbench
uv run --no-sync python scripts/download_datasets.py download skill --name searchqa
uv run --no-sync python scripts/download_datasets.py download skill --name officeqa
```

The script saves datasets under `data/mindmemos_skill/<dataset>/`. For gated Hugging Face data, obtain access and set `HF_TOKEN` before downloading.

For a single benchmark, download only its dataset. Before running ALFWorld, install the system dependencies described in the [deployment guide](../deploy/instruction.md#11-install-dependencies). If you downloaded the data elsewhere, pass the parent directory containing the benchmark directories as `--data-root`.

#### Dataset splits

Counts appear in train, validation, and test order. Each benchmark directory also contains its initial Skill.

| Dataset | Task environment | Split file | Split counts | Dataset description |
| --- | --- | --- | ---: | --- |
| LiveMath | `livemath` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-livemath/data_splits/splits.json) | 35 / 18 / 124 | Multiple-choice math reasoning questions. |
| ALFWorld | `alfworld_bounded_history` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-alfworld/data_splits/splits.json) | 39 / 18 / 134 | Household tasks in a text-based simulator. |
| SpreadsheetBench Verified 400 | `spreadsheetbench_tool_calling` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-spreadsheetbench/data_splits/splits.json) | 80 / 40 / 280 | Excel workbook edits specified in natural language. |
| SearchQA | `searchqa` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-searchqa/data_splits/splits.json) | 400 / 200 / 1,400 | Open-domain questions with search snippets, split by native task key. |
| OfficeQA | `officeqa` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-officeqa/data_splits/splits.json) | 50 / 24 / 172 | Questions requiring retrieval and reasoning over U.S. Treasury Bulletins. |

### 2. Configure the evaluation

For a first run, create a local copy of the repository's [Skill-Forest evaluation template](../../config/mindmemos_eval/skill_train_skill_forest/paper_repro.example.yaml). Edit the existing `paper_repro.yaml` if you already have one.

```bash
cp -n config/mindmemos_eval/skill_train_skill_forest/paper_repro.example.yaml config/mindmemos_eval/skill_train_skill_forest/paper_repro.yaml
```

#### Models and runtime settings

These settings come from `paper_repro.yaml`. The benchmark determines the task environment as shown in the dataset table.

| Role | Model | Thinking | Configuration |
| --- | --- | --- | --- |
| Execute tasks and collect trajectories | `gpt-5.4-mini` | `reasoning_effort: medium` | `models.target` |
| Optimize, select, and merge Skills | `gpt-5.4-mini` | `reasoning_effort: medium` | `models.optimizer` |

Both model groups set `max_completion_tokens: 16384`. Fill in `model`, `api_key`, and the OpenAI-compatible `base_url` in each group. If an endpoint does not support `reasoning_effort`, use `null` for that model to defer to the endpoint's default.

`skill_forest.ensemble_strategy` defaults to `select`. Switch it to `hierarchical_merge` and rerun to evaluate merging. `evaluation_concurrency` controls concurrency on held-out tasks. If you hit API rate limits, also review `expansion_concurrency`, `rollout_concurrency`, and `model_concurrency` under `skill_forest`.

Both strategies score each tree winner on the full training split. `select` picks the highest scorer and breaks ties by TF-IDF novelty relative to the initial Skill. `hierarchical_merge` keeps winners within `merge_threshold` of the best score and merges them in pairs across levels.

### 3. Run a smoke test and the full evaluation

Run a small LiveMath evaluation first. `--smoke` calls the models and reduces the search size.

```bash
uv run --no-sync mindmemos-eval skill evaluate train --algo skill-forest \
  --config config/mindmemos_eval/skill_train_skill_forest/paper_repro.yaml \
  --env livemath --smoke --output-dir output/skill-forest-smoke
```

Each run evaluates one dataset. Select it with `--benchmark` and use a separate output directory for each run.

```bash
uv run --no-sync mindmemos-eval skill evaluate train --algo skill-forest \
  --config config/mindmemos_eval/skill_train_skill_forest/paper_repro.yaml \
  --benchmark livemath --test-repeats 3 --output-dir output/skill-forest-livemath
```

### 4. Read the results

By default, the evaluation writes to a new directory under `output/mindmemos_eval/skill/`. The command prints the path to `report.md`. Read `summary.json` for structured results. If the input or runtime checks fail, inspect `diagnostics.json` in the same directory.

Rebuild a report from saved results without calling models.

```bash
uv run --no-sync mindmemos-eval skill evaluate rebuild-report output/skill-forest-livemath
```

#### Scoring

Compare no Skill, the initial Skill, and the frozen trained Skill on the same held-out test split. Run each condition three times and average its task success rate.

- **LiveMath** matches the final choice against the gold option.
- **ALFWorld** uses the simulator's task success flag.
- **SpreadsheetBench** requires matching cell values in the scored ranges of every workbook case.
- **SearchQA** requires a normalized exact match with any reference answer.
- **OfficeQA** applies the benchmark's numeric, unit, ordered-list, and text rules.

### Validation results

Select and Merge each use seven trees. The table reports mean Hard scores (%) over three held-out test runs per benchmark. The overall score weights all five benchmarks equally.

| Dataset | No Skill | Initial Skill | Trace2Skill | GEPA | SkillOpt | Select | Merge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiveMath | 27.42 | 28.23 | 27.69 | 31.45 | 31.72 | 36.02 | 34.68 |
| ALFWorld | 56.97 | 70.65 | 76.62 | 79.85 | 75.37 | 82.34 | 82.84 |
| SpreadsheetBench Verified 400 | 80.24 | 79.40 | 81.79 | 82.26 | 80.71 | 81.67 | 81.07 |
| SearchQA | 76.90 | 76.67 | 77.50 | 79.62 | 80.67 | 80.21 | 80.52 |
| OfficeQA | 28.49 | 43.99 | 54.84 | 53.68 | 52.71 | 53.88 | 55.23 |
| **Overall** | 54.00 | 59.79 | 63.69 | 65.37 | 64.24 | 66.82 | 66.87 |

## TreeSkill

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name spreadsheetbench
```

The [TreeSkill template](../../config/mindmemos_eval/skill_train_treeskill/dev.example.yaml) uses the repository's [200 train / 200 test split](../../resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json) and bundled initial `xlsx` Skill. `make init` creates its local `dev.yaml` and `config/mindmemos_skill/dev.yaml`.

Install LibreOffice and check that `soffice --version` works in your shell. Copy the [TreeSkill training profile](skill_evaluation_ZH.md#3-配置-treeskill-训练和模型) into `service.training_pipelines.treeskill` in `config/mindmemos_skill/dev.yaml`. Set the profile's `api_key` and `base_url` for an OpenAI-compatible endpoint serving `qwen3.5-35b-a3b`. If you downloaded the dataset to a different location, set `data_root` in the evaluation YAML to the directory containing `dataset.json` and its workbooks.

### Models and runtime settings

This configuration covers the 35B setup. The environment is `spreadsheetbench_text_action`, with at most 100 turns per task and formula recalculation enabled. Model settings are under `trainer.config` in the training profile.

| Role | Model | Thinking | Configuration |
| --- | --- | --- | --- |
| Analyze trajectories and optimize Skills | `qwen3.5-35b-a3b` | `enable_thinking: true` | `optimizer` |
| Analyze failures | `qwen3.5-35b-a3b` | `enable_thinking: false` | `failure_optimizer` |
| Execute training and test tasks | `qwen3.5-35b-a3b` | Inherited from `failure_optimizer`, off | The evaluator derives this model from `failure_optimizer` and sets `temperature: 0.7` |
| Route to a Skill subtree | `qwen3.5-35b-a3b` | `enable_thinking: false` | `router` |

`enable_thinking` is under each model's `extra_body.chat_template_kwargs`.

```bash
uv run --no-sync mindmemos-eval skill evaluate train --algo tree-skill \
  --config config/mindmemos_eval/skill_train_treeskill/dev.yaml --check
uv run --no-sync mindmemos-eval skill evaluate train --algo tree-skill \
  --config config/mindmemos_eval/skill_train_treeskill/dev.yaml --output-dir output/treeskill
```

The preflight prints `200 train / 200 test`. The full run trains a dynamic Skill from the bundled initial Skill and scores workbook cell values on the held-out test split. By default, the run writes `report.md`, `summary.json`, and `trajectories.jsonl` to a new directory under `output/mindmemos_eval/treeskill/`. The CLI reports the pass rate among completed tasks.

### In-domain SpreadsheetBench results

Scores are percentages. Parentheses show gains in percentage points over Human-Written Skill for the same policy model.

| Policy model | Method | Vrf (%) | Soft (%) | Hard (%) |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | No Skill | 19.00 | 18.00 | 4.60 |
| Qwen3.5-35B-A3B | Human-Written Skill | 9.67 | 13.03 | 3.37 |
| Qwen3.5-35B-A3B | Trace2Skill +Combined | 29.67 (+20.00) | 18.80 (+5.77) | 5.73 (+2.36) |
| Qwen3.5-35B-A3B | SkillCAT | 55.50 (+45.83) | 39.02 (+25.99) | 20.61 (+17.24) |
| Qwen3.5-35B-A3B | **TreeSkill (routed)** | **66.00 (+56.33)** | **41.72 (+28.69)** | **25.22 (+21.85)** |
| Qwen3.5-9B | No Skill | 25.50 | 21.00 | 9.21 |
| Qwen3.5-9B | Human-Written Skill | 14.00 | 16.76 | 7.24 |
| Qwen3.5-9B | Trace2Skill +Combined | 16.50 (+2.50) | 18.97 (+2.21) | 8.66 (+1.42) |
| Qwen3.5-9B | **TreeSkill (routed)** | **45.50 (+31.50)** | **30.28 (+13.52)** | **14.58 (+7.34)** |

The 35B baselines are reported scores. Both TreeSkill rows and the 9B baselines shown here come from our runs.

## Offline Evolution

### 1. Prepare the dataset and runtime

Download SkillsBench v1.0. The evaluation uses the initial Skill bundled with each task. Start Docker and install Claude Code locally. Harbor 0.24.0 requires Python 3.12 or 3.13 and is installed when you sync the project dependencies.

```bash
uv sync --all-packages --python 3.13
uv run --no-sync python scripts/download_datasets.py download skill --name skillsbench
uv run --no-sync harbor --version
docker info
claude --version
```

### 2. Configure the execution and evolution models

For a first run, copy the [evolution template](../../config/mindmemos_eval/skill_evolve_offline/dev.example.yaml).

```bash
cp -n config/mindmemos_eval/skill_evolve_offline/dev.example.yaml config/mindmemos_eval/skill_evolve_offline/dev.yaml
```

#### Models and runtime settings

Harbor runs the `skillsbench` task environment (`environment.class_name`) in Docker. The host runs Claude Code for Skill evolution. The table shows template values.

| Role | Agent and model | Thinking | Configuration |
| --- | --- | --- | --- |
| Collect initial trajectories and retest | `claude-code` · `claude-opus-4-6` | Template leaves it unset for the agent launched by Harbor | `environment.config.agent` and `environment.config.model` in the evaluation YAML |
| Evolve the Skill | `claude` · `deepseek-v4-flash` | `reasoning_effort` is unset, so the client uses its default | `service.agents.offline_evolution_evolver_claude.config` in the service YAML |

Keep the variable names listed in `environment.config.agent_env`. Export `ANTHROPIC_API_KEY` in the run shell, or export `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL` for a compatible gateway. Set `model`, `api_key`, and `base_url` for the local evolution agent in the service YAML. `service.evolution_pipelines.offline.config.evolver.config.agent` selects that agent.

For a custom download path, set `data.config.benchmark_root` in the evaluation YAML to the directory containing `tasks/`. The default is `data/mindmemos_skill/skillsbench`.

### 3. Check inputs and run

`--check` prepares the selected tasks and their initial Skills. Then run the second command. To try one case with models, set `cases` to a list containing one directory name from `tasks/` and set `repeats: 1`. Restore `cases: null` and `repeats: 3` for the full evaluation.

```bash
uv run --no-sync mindmemos-eval skill evaluate evolve --algo offline-evolv \
  --config config/mindmemos_eval/skill_evolve_offline/dev.yaml --check
uv run --no-sync mindmemos-eval skill evaluate evolve --algo offline-evolv \
  --config config/mindmemos_eval/skill_evolve_offline/dev.yaml --output-dir output/offline-evolv
```

### 4. Read the results

By default, the run writes `report.md` and `summary.json` under `output/mindmemos_eval/skill_evolve/<timestamp>/`. The evaluator collects trajectories with the initial Skill and retests the same tasks with the evolved Skill. It averages rewards across repeats per task, then across scored tasks. Tasks without a usable evolved Skill retain their initial scores. Full comparison results are pending.

Check the per-task `Status` to verify evolution. `evolved` means the pipeline produced a new Skill and scheduled a retest, `unchanged` means it kept the initial Skill, and `evolution_failed` means evolution failed and the task kept its initial score. The command can exit with status zero after writing the report, so check each task's status and retest results when validating evolution.

| Dataset | Metric | Initial Skill | Offline Evolution |
| --- | --- | ---: | ---: |
| SkillsBench | Mean of per-task average rewards | — | — |

## Skill Graph evaluation

Both modes load packages from `--skills-dir` and build a local graph. Place each Skill at `<source Skill ID>/SKILL.md` with its accompanying resources. Run `make init` and configure the chat and embedding models in `config/mindmemos_skill/dev.yaml`. Use `--service-config` to select a different service configuration.

Use a fresh `--output` directory for each run. The command queries the graph with each task's public instruction, writes the ordered results to `rankings.json` in that directory, and uses those rankings in the selected evaluation mode. The reference labels stay on the evaluation side.

Install Git and download the pinned AgentSkillOS release. Both evaluation modes use the same download.

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name agentskillos
```

If `data/mindmemos_skill/agentskillos` contains the earlier Hugging Face download, move that directory aside before running the command.

### Retrieval scoring

The evaluator reads reference labels from the `skills` field in the official task JSON files.

```bash
uv run --no-sync mindmemos-eval skill evaluate manage --algo skill-graph --env agentskillos --eval-mode retrieval \
  --tasks-dir data/mindmemos_skill/agentskillos/benchmark/AgentSkillOS_bench/tasks \
  --skills-dir path/to/skills \
  --output-dir output/agentskillos_retrieval
```

In `summary.json`, `per_task` contains per-task scores and `metrics` contains the macro-average. For each task, Recall@K divides the number of reference Skills found in the top K by the total number in the original `skills` list. The report also includes nDCG, MRR, Hit, Precision, and FullCoverage.

### End-to-end accuracy

`accuracy` uses Setting 1's execution model, `MiniMax-M2.7`. Claude Code receives the full Skill packages and resources for the top K retrieval results and produces task deliverables. The official AgentSkillOS evaluator determines whether the deliverables pass.

Run `make init` and set `agent.base_url` and `agent.api_key` in `config/mindmemos_eval/skill_graph_setting_1/dev.yaml`. Use an Anthropic Messages-compatible endpoint. Install Claude Code. Set `execution_top_k` to adjust how many Skills each task receives. [MiniMax setup for Claude Code](https://platform.minimax.io/docs/m-plan/claude-code)

```bash
uv run --no-sync mindmemos-eval skill evaluate manage --algo skill-graph --env agentskillos \
  --config config/mindmemos_eval/skill_graph_setting_1/dev.yaml --eval-mode accuracy \
  --skills-dir path/to/skills \
  --output-dir output/agentskillos_accuracy
```

Use a fresh output directory. In `summary.json`, `accuracy` is the fraction of successfully scored tasks that pass the official evaluation. `scoring_coverage` records scoring coverage separately. Execution or scoring failures appear in `errors` and cause a nonzero exit. The `tasks/` directory retains per-task results, deliverables, and trajectories.

### Benchmark results

The tables list the reported results for three benchmarks. `Ours` denotes MindMemOS Skill Graph. The commands above cover AgentSkillOS, and the end-to-end command uses Setting 1.

#### Table 1. Skill retrieval

Emb. means embedding-based retrieval and RR means reranking. Bold and underlined values mark first and second place in each column, including ties. Asterisks mark scores from the corresponding original papers.

| Method | ALFWorld Ret@1 | ALFWorld Ret@K | ALFWorld MRR | AgentSkillOS Ret@1 | AgentSkillOS Ret@K | AgentSkillOS MRR | SkillsBench Ret@1 | SkillsBench Ret@K | SkillsBench MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vector-based methods** |  |  |  |  |  |  |  |  |  |
| Vector Skills | 0.7 | 4.5 | 0.4 | 56.7 | 83.3 | 66.2 | 3.6\* | 10.8\* | 5.8\* |
| SkillRouter (Emb.) | 13.6 | 48.6 | 26.2 | 50.0 | <ins>86.7</ins> | 64.3 | <ins>77.6</ins> | **93.4** | <ins>83.1</ins> |
| SkillRouter (Emb.+RR) | 12.1 | **74.3** | <ins>36.2</ins> | 56.7 | 70.0 | 61.9 | **81.6** | <ins>88.2</ins> | **84.5** |
| **Tree-structured methods** |  |  |  |  |  |  |  |  |  |
| AgentSkillOS | 0.7 | 50.7 | 18.5 | 70.0 | 83.3 | 76.7 | 38.2 | 44.7 | 41.0 |
| SkillRAE | 9.3 | 35.6 | 17.5 | 40.0 | 80.0 | 55.2 | 69.7 | 86.8 | 76.8 |
| **Graph-structured methods** |  |  |  |  |  |  |  |  |  |
| GoS | 8.6 | 9.3 | 8.9 | <ins>76.7</ins> | <ins>86.7</ins> | <ins>80.3</ins> | 50.6\* | 65.5\* | 57.3\* |
| SkillDAG | <ins>20.0</ins> | 44.8 | 29.2 | 43.3 | 56.1 | 60.8 | 66.7\* | 78.2\* | 71.3\* |
| Ours | **49.3** | <ins>71.4</ins> | **57.9** | **80.0** | **93.3** | **83.3** | 69.2 | 86.8 | 76.8 |

#### Table 2. End-to-end R scores

Setting 1 uses MiniMax-M2.7. Rankings are per benchmark within each setting and use the same markings as Table 1.

| Setting | Method | ALFWorld | AgentSkillOS | SkillsBench |
| --- | --- | ---: | ---: | ---: |
| Setting 1 | Vanilla Skills | 47.1\* | 70.8 | 17.2\* |
|  | Vector Skills | 50.7\* | <ins>72.4</ins> | 10.4\* |
|  | GoS | 54.3\* | 68.0 | 18.7\* |
|  | SkillDAG | <ins>67.9</ins> | 54.9 | <ins>27.3</ins>\* |
|  | Ours | **76.4** | **80.9** | **28.0** |
| Setting 2 | Vector Skills | 60.0 | 74.9\* | 27.3 |
|  | GoS | 63.6 | <ins>96.2</ins> | <ins>35.4</ins> |
|  | SkillDAG | <ins>82.0</ins> | 94.2 | 31.4 |
|  | AgentSkillOS | 13.6 | 83.5\* | 17.2\* |
|  | SkillRouter | 29.3 | 82.3\* | 22.0\* |
|  | SkillRAE | 27.1 | 84.6\* | 29.3\* |
|  | Ours | **86.0** | **97.4** | **43.2** |
| Setting 3 | GoS | 79.3 | <ins>96.2</ins> | 40.1 |
|  | SkillDAG | <ins>85.0</ins> | 92.3 | <ins>49.0</ins> |
|  | Ours | **85.7** | **96.5** | **51.0** |

## Single-environment Skill evaluation

### 1. Prepare the dataset

This example compares LiveMath with no Skill and with a fixed initial Skill. Reuse the dataset if you downloaded it for Skill-Forest.

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
```

### 2. Configure the test

`make init` creates `config/mindmemos_eval/skill_test_direct/dev.yaml` from the [direct test template](../../config/mindmemos_eval/skill_test_direct/dev.example.yaml). The template loads LiveMath from the download directory and uses the repository's fixed `test` split.

#### Models and runtime settings

| Role | Environment | Model | Thinking | Configuration |
| --- | --- | --- | --- | --- |
| Answer with no Skill or with the fixed initial Skill | `livemath` | `gpt-5.4-mini` | `reasoning_effort: medium` | `environment.class_name`, `environment.config.agent` |

The template sets `max_completion_tokens: 16384`. Fill in `model`, `api_key`, and `base_url` under `environment.config.agent`. Use the same model settings in both conditions.

### 3. Run a smoke test and the full evaluation

Start with `--limit 2`. Remove it to run the full test split. Each command writes a separate run directory.

```bash
uv run --no-sync mindmemos-eval skill evaluate direct \
  --config config/mindmemos_eval/skill_test_direct/dev.yaml --no-skill --limit 2
uv run --no-sync mindmemos-eval skill evaluate direct \
  --config config/mindmemos_eval/skill_test_direct/dev.yaml --limit 2 \
  --skills resources/mindmemos_skill_eval/skill_forest-livemath/initial_skill
```

### 4. Read the results

Under `output/mindmemos_eval/test/<run>/`, open `report.md` and `summary.json`. Per-task traces are in `tasks/trajectories.jsonl`. LiveMath matches the final choice against the reference option. Use this fixed-Skill entry point to check the environment and compare the two pass rates.
