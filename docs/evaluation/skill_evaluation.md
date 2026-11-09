# MindMemOS Skill Evaluation

[简体中文](skill_evaluation_ZH.md) · [Memory evaluation](memory_evaluation.md)

Run commands from the repository root. Each section specifies the data, evaluation entry point, and score to compare. Blank score cells are reserved for results.

```bash
uv sync --all-packages
make init
uv run --no-sync mindmemos-eval skill envs
```

## Skill-Forest

### Datasets and scoring

Download one or more of the five supported datasets. The downloader uses [pinned sources](../../config/presets/dataset_download_sources.json) and saves them under `data/mindmemos_skill/`. For gated Hugging Face data, accept the dataset terms and set `HF_TOKEN` in the environment or local `.env`.

```bash
for benchmark in livemath alfworld spreadsheetbench searchqa officeqa; do
  uv run --no-sync python scripts/download_datasets.py download skill --name "$benchmark" || break
done
```

| Dataset | Evaluation logic |
| --- | --- |
| LiveMath | Extract the final choice and match it to the gold option. |
| ALFWorld | Run the simulator with bounded interaction history. An episode passes when the simulator reports `won`. Install the `alfworld` extra. |
| SpreadsheetBench Verified 400 | Use native tool calling. A task passes when all workbook cases match the reference values in the scored cell ranges. Automatic recalculation is off by default. Install the `spreadsheetbench` extra. |
| SearchQA | Use the fixed 400/200/1,400 train/val/test split by native task key. Success requires normalized exact match against any gold answer. Token F1 and substring match are also recorded. |
| OfficeQA | Use the Treasury document corpus. The pinned OfficeQA scorer checks the final answer with its numeric, unit, ordered-list, and text rules and returns a binary score. The turn limit is `max_tool_turns`. |

The committed splits and initial Skills are under [resources/mindmemos_skill_eval](../../resources/mindmemos_skill_eval/). The runner checks receipts, splits, and runtime dependencies before training. For all five benchmarks, install both extras with `uv sync --all-packages --extra alfworld --extra spreadsheetbench`.

### Skill-Forest Select

Copy [the Select example](../../config/mindmemos_eval/forest_select/dev.example.yaml) to `config/mindmemos_eval/forest_select/dev.yaml`. Fill in `models.target` and `models.optimizer` with their model names and API connections. The evaluator creates its own local Skill service in the process.

```bash
uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_select/dev.yaml \
  --benchmark all --test-repeats 3
```

Use `--benchmark livemath` or another dataset name to run one benchmark. Select scores the per-tree winners on the full training split. With `selection_tiebreak: novelty`, it breaks a top-score tie by pairwise TF-IDF novelty relative to the initial Skill. It freezes the selected Skill before held-out testing.

### Skill-Forest Merge

Copy [the Merge example](../../config/mindmemos_eval/forest_merge/dev.example.yaml) to `config/mindmemos_eval/forest_merge/dev.yaml` and configure the same models.

```bash
uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_merge/dev.yaml \
  --benchmark all --test-repeats 3
```

Merge scores the per-tree winners on the full training split, keeps candidates within `merge_threshold` of the best score, and merges them in pairs across levels. It freezes the merged Skill before held-out testing.

### Score comparison

Each configuration tests `no_skill`, `initial_skill`, and `trained_skill` on the same held-out tasks. `--test-repeats 3` tests the same frozen version three times. Use the mean task success rate and record execution coverage. Keep the model, split, initial Skill, and evaluation settings fixed when comparing Select and Merge. The configured candidate-validation batch size and patch behavior also matter for reproducing training.

| Dataset | No Skill | Initial Skill | Select | Merge | Reference method and score |
| --- | ---: | ---: | ---: | ---: | --- |
| LiveMath | | | | | |
| ALFWorld | | | | | |
| SpreadsheetBench Verified 400 | | | | | |
| SearchQA | | | | | |
| OfficeQA | | | | | |

`--smoke` uses the first 4 train, 2 val, and 2 test tasks and reduces the search budget. `--arms no_skill,initial_skill` runs the two controls without training. Each run writes `summary.json`, `summary.csv`, and `report.md` under `output/mindmemos_eval/skill/` by default. Incomplete coverage leaves full accuracy unavailable. To regenerate reports from saved results, pass the run directory to `mindmemos-eval skill report`.

## TreeSkill

Download SpreadsheetBench Verified 400 and use the committed [TreeSkill split and initial Skill](../../resources/mindmemos_skill_eval/treeskill-spreadsheetbench/). Install the `spreadsheetbench` extra and make LibreOffice available for workbook recalculation.

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name spreadsheetbench
```

For the recorded protocol, obtain the reference run directory `output/treeskill_full_20261012_092954` separately. Set `reference_run` in [the local evaluation config](../../config/mindmemos_eval/treeskill/dev.example.yaml) and match the server training profile selected by `profile` (the current example uses `treeskill`). The reference uses 200 train and 200 test tasks, `qwen3.5-35b-a3b`, seed 41, 100 turns, and `transactional_recalculation: true`. The preflight also checks the recorded inputs and trainer settings.

Configure the SDK and Skill HTTP service as described in the [deployment guide](../deploy/instruction.md). The client and service need access to the same absolute data and output paths.

```bash
uv run --no-sync mindmemos-eval skill treeskill --env spreadsheetbench_text_action --check
uv run --no-sync mindmemos-eval skill treeskill --env spreadsheetbench_text_action
```

The runner registers the initial Skill over the SDK, trains the dynamic TreeSkill, then tests the final version on the held-out split. The text-action environment scores workbook cell values. Read success, completion counts, and execution errors together in `summary.json`. Other registered environments can supply their own config, data, split, and initial Skill.

| Dataset and protocol | TreeSkill | Reference method and score |
| --- | ---: | --- |
| SpreadsheetBench Verified 400, text action, held-out test | 122/200 (61.0%, historical reference) | |

The 61.0% comes from the previously documented reference run. Compare it with runs using the same text-action and recalculation protocol.

## Offline Evolution

The default uses SkillsBench v1.0 and each task's bundled Skill. Download the dataset and copy [the evolution example](../../config/mindmemos_eval/skill_evolve/dev.example.yaml) to `config/mindmemos_eval/skill_evolve/dev.yaml`. Configure the execution agent and credentials, start Docker for the Harbor environment, and configure the evolution models in `config/mindmemos_skill/dev.yaml`. For another registered environment, provide matching data and the initial Skill via `skills` or `case_skills`.

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name skillsbench
uv run --no-sync mindmemos-eval skill evolve --check
uv run --no-sync mindmemos-eval skill evolve
```

Collect initial-Skill rollouts, submit completed trajectories to the `offline` pipeline, export the evolved Skills, and retest the **same cases** with the same execution agent and verifier. This measures same-case improvement. The report averages rewards across repeats per case, then across cases. By default it uses three repeats for each arm. Perfect cases and cases with no usable change can carry forward the initial score. Execution errors and excluded cases are reported separately. `--check` validates data and Skill sources without calling models.

| Dataset | Metric | Initial Skill | Offline Evolution | Reference method and score |
| --- | --- | ---: | ---: | --- |
| SkillsBench | Mean per-case Avg@N reward | | | |

## Skill Management

### AgentSkillOS retrieval

The [public AgentSkillOS release](https://huggingface.co/datasets/NPULH/agentskillos-benchmark) provides the upstream task data. This retrieval evaluator also needs the exact candidate Skill pool and its relevance annotations. Obtain the prepared retrieval bundle with `skill_pool_manifest.json`, `splits.json`, and `queries/{tasks.jsonl,relevance.json}`, or the original bundle with `skill_pool_200/`, `ssl_pool_join.json`, `tasks.jsonl`, and `relevance.json`. The repository downloader has no recipe for this bundle. Its download location is pending.

Start the Skill HTTP service and configure the SDK. Use separate output directories for each arm.

```bash
for strategy in graph vector; do
  uv run --no-sync python -m scripts.skill_eval.agentskillos.evaluate \
    --dataset-root data/mindmemos_skill/agentskillos_eval \
    --strategy "$strategy" --top-k 50 \
    --output "output/agentskillos_$strategy" || break
done
```

The script registers the full pool, organizes a graph, pins the graph version, and ranks Skills per query. It skips generic-only queries and scores binary core labels. Recall uses the complete relevant set as its denominator, and metrics are macro-averaged over scored queries. The report keeps failed requests and scoring coverage separate. Keep the pool, annotations, graph settings, and retrieval budget fixed across comparisons. To reuse exact Skill versions, supply the first run's `registrations.json` via `--reuse-registrations`.

| Strategy | nDCG@10 | MRR@10 | Recall@50 | FullCoverage@10 |
| --- | ---: | ---: | ---: | ---: |
| Vector | | | | |
| Graph | | | | |

## Skill Judge

The judge analyzes Skill instructions and readable resources. For long inputs, it collects evidence in overlapping windows and combines the findings. Use the repository's Skill fixtures. The regression checks the analysis output, overlapping windows for long documents, coverage limits, and filtering of sensitive or unreadable resources. A labeled benchmark and accuracy protocol are pending.

```bash
uv run --no-sync pytest -q src/mindmemos_skill/tests/test_analyze_algos.py -k skill_judge
```

| Labeled dataset | Metric | Skill Judge | Reference method and score |
| --- | --- | ---: | --- |
| | | | |

## Trajectory Judge

The judge uses the execution trace and any available verifier score to produce a verdict and extracts Skill usage from the trace. Use the repository's trajectory fixtures. The regression checks verdicts, Skill usage metrics, and optional attribution to a detected Skill when a task fails. A labeled benchmark and accuracy protocol are pending.

```bash
uv run --no-sync pytest -q src/mindmemos_skill/tests/test_analyze_algos.py -k trajectory_judge
```

| Labeled dataset | Metric | Trajectory Judge | Reference method and score |
| --- | --- | ---: | --- |
| | | | |

## Environment-only controls

`skill test` runs one registered environment with either no Skills or the same static Skill set for every task. For the default LiveMath config, use the downloaded LiveMath data and configure the agent in `config/mindmemos_eval/env_test/dev.yaml`.

```bash
uv run --no-sync mindmemos-eval skill test --no-skill --split test
uv run --no-sync mindmemos-eval skill test --split test \
  --skills resources/mindmemos_skill_eval/skill_forest-livemath/initial_skill
```

The report shows native successes divided by normally completed tasks, alongside planned, attempted, and completed counts. Review errors and coverage before comparing runs. The output includes `summary.json`, `report.md`, and `tasks/trajectories.jsonl`.
