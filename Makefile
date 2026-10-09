.DEFAULT_GOAL := help
SHELL := /bin/bash

# One command selects the action; these options select scope and capabilities.
SERVICE ?= all
FEATURE ?= all
UI ?= auto
PLAN ?= 0
UV ?= uv
export

RUN = $(UV) run python -m scripts.dev.run
OPTION = --service "$(SERVICE)" --feature "$(FEATURE)" --ui "$(UI)" $(if $(filter 1,$(PLAN)),--plan)

.PHONY: help dev docker api ui status stop clean setup hook format lint ui-build profile

help: ## Show commands and common options (default)
	@printf '%s\n' 'Usage: make <command> [SERVICE=all|memory|skill] [FEATURE=core|async]' 'Default: all services and all features; dev includes UI.' ''
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+( [a-z-]+)*:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\n%s\n' 'Options: PLAN=1 previews; UI=0|1 overrides UI; API_PORT, SKILL_PORT, UI_PORT override ports.' 'Config: MEMORY_CONFIG, SKILL_CONFIG, UI_HOME; reload: API_RELOAD=1, SKILL_RELOAD=1.'

dev: ## Start Docker dependencies, API services, and UI
	@$(RUN) dev $(OPTION)

docker: ## Start only the selected Docker dependencies
	@$(RUN) docker $(OPTION)

api: ## Start APIs against existing dependencies; UI is off by default
	@$(RUN) api $(OPTION)

ui: ## Start only the UI using its existing SDK configuration
	@$(RUN) ui $(OPTION)

status: ## Show managed processes, addresses, features, and containers
	@$(RUN) status

stop: ## Stop managed applications and Docker; preserve data
	@$(RUN) stop

clean: ## Stop services and delete Docker volumes after confirmation
	@$(RUN) clean

# Development tools do not start applications.
setup: ## Sync dependencies, install hooks, and prepare NLP assets
	$(UV) sync
	$(UV) run pre-commit install --install-hooks
	$(UV) run python scripts/install_nlp_assets.py
	$(UV) run python scripts/check_nlp_assets.py

hook: ## Install Git pre-commit hooks
	$(UV) run pre-commit install --install-hooks

format: ## Fix lint issues and format Python code
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

lint: ## Check Python code without modifying files
	$(UV) run ruff check .

ui-build: ## Install frontend dependencies and build UI assets
	cd src/mindmemos_ui/frontend && npm ci && npm run build

profile: ## Record a CPU flame graph; supports PID and PROFILE_SECONDS
	@$(UV) run python -m scripts.dev.profile
