# Repository Guidelines

## Project Structure & Module Organization

MindMemOS is a Python `uv` monorepo. The main FastAPI service and runtime live in `src/mindmemos/`; benchmark runners and evaluation environments are in `src/mindmemos_eval/`; the Python SDK and CLI are in `src/mindmemos_sdk/`; and the standalone Skill runtime is in `src/mindmemos_skill/`. Tests are grouped by boundary (`api`, `architecture`, `infra`, `llm`, `config`, `workers`, and so on). Use `config/` for examples, `dockers/` for local dependencies, `docs/` for operational guidance, `plugins/openclaw-plugin/` for the TypeScript integration, and `skills/`, `resources/`, and `assets/` for shipped agent content.

The root `uv` workspace includes `mindmemos`, `mindmemos-eval`, `mindmemos-sdk`, and `mindmemos-skill`.

## Build, Test, and Development Commands

- `make dev-setup` syncs Python dependencies, installs hooks, and prepares NLP assets.
- `make format` applies Ruff fixes and formatting; `make lint` runs Ruff checks only.
- `uv run pytest tests -q` runs the Python suite; narrow runs such as `uv run pytest tests/api -q` are preferred during iteration.
- `make dev-core` starts core Docker services; `make dev` starts the full dependency stack and FastAPI; `make api` starts only FastAPI; `make dev-down` stops services.
- For the OpenClaw plugin: `cd plugins/openclaw-plugin && npm ci && npm run typecheck && npm run build`.

## Coding Style & Naming Conventions

Target Python 3.11–3.13, four-space indentation, double quotes, and a 120-character Ruff line length. Use `snake_case` for Python names, `PascalCase` for classes, and `test_*.py` / `test_*` for tests.

## Testing Guidelines

Tests use pytest and pytest-asyncio; no project-wide coverage threshold is configured. Add focused regression tests beside the affected subsystem and run the narrow suite before the full suite.

## Commit & Pull Request Guidelines

Use concise conventional-style subjects such as `feat:`, `fix:`, `refactor:`, `docs:`, or `style:`. Keep commits focused. PRs should explain the behavior change, affected package, configuration or migration impact, and tests run; link an issue when applicable and attach UI or plugin screenshots/logs when they clarify the change.

## Security & Configuration

Copy `.env.example` and `config/**/dev.example.yaml` for local setup. Never commit API keys, passwords, generated datasets, or `.env` files; pre-commit includes a Gitleaks staged-secret scan.
Do NOT push `.design/` to github remote.
