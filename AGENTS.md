# Repository Guidelines

## Project Structure & Module Organization

MindMemOS is a Python `uv` monorepo. The main FastAPI service and runtime live in `src/mindmemos/`; benchmark runners and evaluation environments are in `src/mindmemos_eval/`; the Python SDK and CLI are in `src/mindmemos_sdk/`; and the standalone Skill runtime is in `src/mindmemos_skill/`. Tests are grouped by boundary (`api`, `architecture`, `infra`, `llm`, `config`, `workers`, and so on). Use `config/` for examples and experiment definitions, `dockers/` for local dependencies, `docs/` for operational guidance, `plugins/openclaw-plugin/` for the TypeScript integration, and `skills/`, `resources/`, and `assets/` for shipped agent content. Keep `scripts/` limited to stable entrypoints and operational utilities. MindMemOS Skill has exactly two family entrypoints in `scripts/mindmemos_skill/runners/` (`evolve.py` and `trace2skill.py`); experiment adapters and their registry live in `scripts/mindmemos_skill/experiments/`, outside the pure algorithm-support package.

The root `uv` workspace includes `mindmemos`, `mindmemos-eval`, `mindmemos-sdk`, and `mindmemos-skill`.

## Build, Test, and Development Commands

- `make dev-setup` syncs Python dependencies, installs hooks, and prepares NLP assets.
- `make format` applies Ruff fixes and formatting; `make lint` runs Ruff checks only.
- `uv run pytest tests -q` runs the Python suite; narrow runs such as `uv run pytest tests/api -q` are preferred during iteration.
- `scripts/run_mindmemos_skill_experiment.sh --config config/mindmemos_skill/<method>/<environment>/<name>.yaml` is the only supported MindMemOS Skill experiment entrypoint. Add `--dry-run` to validate without model calls or `--set training.epochs=1` for a temporary override.
- `make dev-core` starts core Docker services; `make dev` starts the full dependency stack and FastAPI; `make api` starts only FastAPI; `make dev-down` stops services.
- For the OpenClaw plugin: `cd plugins/openclaw-plugin && npm ci && npm run typecheck && npm run build`.

## Coding Style & Naming Conventions

Target Python 3.11–3.13, four-space indentation, double quotes, and a 120-character Ruff line length. Use `snake_case` for Python names, `PascalCase` for classes, and `test_*.py` / `test_*` for tests.

## Testing Guidelines

Tests use pytest and pytest-asyncio; no project-wide coverage threshold is configured. Add focused regression tests beside the affected subsystem and run the narrow suite before the full suite.

## Skill Experiment Configuration

Read `docs/skill_algo_develop/experiment_runner.md` before changing or running experiments. The only supported public entrypoint is:

```bash
UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/<method>/<environment>/<name>.yaml
```

Use `--dry-run` first to inspect the resolved family runner, adapter, CLI arguments, run ID, and output path without model calls. Temporary overrides use repeatable `--set group.leaf=value`; do not copy a YAML merely to change one value. For example, ALFWorld initial-Skill evaluation with two rollouts per test task is:

```bash
UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/skill_evaluation/alfworld/default.yaml \
  --set evaluation.test_rollouts=2
```

Store every Skill experiment under `config/mindmemos_skill/<method>/<environment>/<name>.yaml` with `version`, `method`, `environment`, optional `launcher`, and grouped `parameters`. Groups such as `dataset`, `models`, `training`, `rollout`, `evaluation`, and `algorithm` are readability-only; every leaf name must be unique because it becomes one CLI flag. The dispatch path is `run_mindmemos_skill_experiment.sh -> scripts/mindmemos_skill/run_experiment.py -> runners/{evolve|trace2skill}.py -> experiments/registry.py -> adapter`. Select algorithms and environments through YAML, never through new method-specific or environment-specific shell scripts.

`launcher.env_file` defaults by convention to `.skill.env`. The launcher begins with the process environment and then applies the env file, so values in `.skill.env` override same-named exported variables. Keep `OPENAI_API_KEY`, `OPENAI_BASE_URL`/`OPENAI_ENDPOINT`, and other credentials there or in the process environment, never in YAML. Do not print secret values while diagnosing; report the source and a short hash/length only.

Default runs use `run_id=<environment>_<method>_<timestamp>` and `output_dir=outputs/<environment>/<method>/<run_id>`. Successful runs persist the resolved, secret-free `experiment_config.yaml`; test evaluation additionally writes `test/summary.json`, `test/results.jsonl`, and `test/skill.json` and shows rollout progress on stderr. `evaluation.test_rollouts=N` means N rollouts for every test task, not N tasks; use `evaluation.test_limit=N` to limit task count.

When adding a method, add one adapter under `scripts/mindmemos_skill/experiments/`, register it in `scripts/mindmemos_skill/experiments/registry.py`, add at least one YAML config, and extend `tests/scripts/test_run_mindmemos_skill_experiment.py`. Do not add a third family runner and do not put experiment CLI/orchestration code under `src/mindmemos_skill/mindmemos_skill/`.

## Commit & Pull Request Guidelines

Use concise conventional-style subjects such as `feat:`, `fix:`, `refactor:`, `docs:`, or `style:`. Keep commits focused. PRs should explain the behavior change, affected package, configuration or migration impact, and tests run; link an issue when applicable and attach UI or plugin screenshots/logs when they clarify the change.

## Security & Configuration

Copy `.env.example` and `config/**/dev.example.yaml` for local setup. Never commit API keys, passwords, generated datasets, or `.env` files; pre-commit includes a Gitleaks staged-secret scan.
Do NOT push `.design/` to github remote.
