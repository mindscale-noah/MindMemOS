#!/usr/bin/env python3
"""Run a MindMemOS Skill experiment from one YAML configuration file."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from experiments import EXPERIMENTS, ExperimentSpec  # noqa: E402

_TOP_LEVEL_KEYS = {"version", "method", "environment", "launcher", "parameters", "resolved"}
_ENV_PATTERN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


RUNNERS: dict[str, ExperimentSpec] = EXPERIMENTS


@dataclass(frozen=True)
class RunnerOption:
    name: str
    required: bool
    action: str | None
    accepts_many: bool


@dataclass(frozen=True)
class ExperimentInvocation:
    config_path: Path
    method: str
    environment: str
    run_id: str | None
    output_dir: Path | None
    command: list[str]
    environment_values: dict[str, str]
    resolved_config: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML path")
    parser.add_argument("--env-file", type=Path, help="override launcher.env_file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override a YAML value, for example training.epochs=1",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print the command without running it")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER, help="arguments after -- are passed to the runner")
    return parser.parse_args(argv)


def load_experiment_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment config does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("experiment config root must be a mapping")
    unknown = sorted(set(loaded) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"unknown top-level config keys: {', '.join(unknown)}")
    if loaded.get("version") != 1:
        raise ValueError("experiment config version must be 1")
    if not isinstance(loaded.get("parameters"), dict):
        raise ValueError("experiment config parameters must be a mapping")
    launcher = loaded.get("launcher", {})
    if not isinstance(launcher, dict):
        raise ValueError("experiment config launcher must be a mapping")
    unknown_launcher = sorted(set(launcher) - {"env_file", "extra_dependencies"})
    if unknown_launcher:
        raise ValueError(f"unknown launcher config keys: {', '.join(unknown_launcher)}")
    return loaded


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must use PATH=VALUE: {item}")
        raw_path, raw_value = item.split("=", 1)
        keys = [part for part in raw_path.split(".") if part]
        if not keys:
            raise ValueError(f"override path is empty: {item}")
        if keys[0] not in _TOP_LEVEL_KEYS - {"version"}:
            keys.insert(0, "parameters")
        target: dict[str, Any] = config
        for key in keys[:-1]:
            child = target.get(key)
            if child is None:
                child = {}
                target[key] = child
            if not isinstance(child, dict):
                raise ValueError(f"override path crosses a non-mapping value: {raw_path}")
            target = child
        target[keys[-1]] = yaml.safe_load(raw_value)


def _expand_environment(value: str, environment: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in environment:
            raise ValueError(f"environment variable {name} referenced by config is not set")
        return environment[name]

    return _ENV_PATTERN.sub(replace, value)


def _render_value(value: Any, *, context: dict[str, str], environment: dict[str, str]) -> Any:
    if isinstance(value, str):
        expanded = _expand_environment(value, environment)
        try:
            return expanded.format_map(context)
        except KeyError as exc:
            raise ValueError(f"unknown config template variable: {exc.args[0]}") from exc
    if isinstance(value, list):
        return [_render_value(item, context=context, environment=environment) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, context=context, environment=environment) for key, item in value.items()}
    return value


def _flatten_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(mapping: dict[str, Any], path: tuple[str, ...]) -> None:
        for key, value in mapping.items():
            if not isinstance(key, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", key) is None:
                raise ValueError(f"invalid parameter key: {'.'.join((*path, str(key)))}")
            if isinstance(value, dict):
                visit(value, (*path, key))
                continue
            flag_name = key.replace("_", "-")
            if flag_name in flattened:
                previous = ".".join((*path, key))
                raise ValueError(f"duplicate runner parameter --{flag_name}; second occurrence: {previous}")
            flattened[flag_name] = value

    visit(parameters, ())
    return flattened


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def inspect_runner_options(script_path: Path) -> dict[str, RunnerOption]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    options: dict[str, RunnerOption] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        option_name = next(
            (value for argument in node.args if (value := _literal_string(argument)) and value.startswith("--")),
            None,
        )
        if option_name is None:
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}
        action_node = keywords.get("action")
        action: str | None = None
        if isinstance(action_node, ast.Attribute):
            action = action_node.attr
        elif action_node is not None:
            action = _literal_string(action_node)
        required_node = keywords.get("required")
        required = isinstance(required_node, ast.Constant) and required_node.value is True
        nargs_node = keywords.get("nargs")
        accepts_many = nargs_node is not None
        name = option_name.removeprefix("--")
        options[name] = RunnerOption(name=name, required=required, action=action, accepts_many=accepts_many)
    return options


def _append_option(command: list[str], option: RunnerOption, value: Any) -> None:
    flag = f"--{option.name}"
    if option.action == "BooleanOptionalAction":
        if not isinstance(value, bool):
            raise ValueError(f"{flag} must be true or false")
        command.append(flag if value else f"--no-{option.name}")
        return
    if option.action in {"store_true", "store_false"}:
        if not isinstance(value, bool):
            raise ValueError(f"{flag} must be true or false")
        enabled = value if option.action == "store_true" else not value
        if enabled:
            command.append(flag)
        return
    if isinstance(value, bool):
        raise ValueError(f"{flag} does not accept a boolean value")
    if isinstance(value, list):
        if not option.accepts_many:
            raise ValueError(f"{flag} does not accept a list")
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    if value is not None:
        command.extend((flag, str(value)))


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"environment file does not exist: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid environment assignment at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment name at {path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def build_invocation(
    config_path: Path,
    *,
    overrides: list[str] | None = None,
    env_file_override: Path | None = None,
    runner_args: list[str] | None = None,
    timestamp: str | None = None,
    base_environment: dict[str, str] | None = None,
) -> ExperimentInvocation:
    config_path = config_path.expanduser().resolve()
    config = load_experiment_config(config_path)
    apply_overrides(config, overrides or [])
    method = config.get("method")
    environment_name = config.get("environment")
    if not isinstance(method, str) or method not in RUNNERS:
        available = ", ".join(sorted(RUNNERS))
        raise ValueError(f"unknown experiment method {method!r}; available: {available}")
    spec = RUNNERS[method]
    if not isinstance(environment_name, str) or environment_name not in spec.environments:
        available = ", ".join(sorted(spec.environments))
        raise ValueError(f"method {method!r} does not support environment {environment_name!r}; available: {available}")

    environment_values = dict(base_environment or os.environ)
    launcher = config.get("launcher", {})
    configured_env_file = env_file_override or launcher.get("env_file")
    if configured_env_file is not None:
        env_file = Path(configured_env_file).expanduser()
        if not env_file.is_absolute():
            env_file = REPO_ROOT / env_file
        environment_values.update(_load_env_file(env_file.resolve()))

    run_stamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    context = {
        "timestamp": run_stamp,
        "method": method,
        "environment": environment_name,
        "run_id": "{run_id}",
    }
    rendered_parameters = _render_value(config["parameters"], context=context, environment=environment_values)
    flattened = _flatten_parameters(rendered_parameters)
    script_path = REPO_ROOT / "scripts" / "mindmemos_skill" / "runners" / f"{spec.family.value}.py"
    options = inspect_runner_options(spec.implementation_path)

    if spec.inject_environment_as_benchmark:
        configured_benchmark = flattened.setdefault("benchmark", environment_name)
        if configured_benchmark != environment_name:
            raise ValueError("parameters.dataset.benchmark must match top-level environment")

    if "run-id" in options:
        run_id_template = flattened.get("run-id", "{environment}_{method}_{timestamp}")
        if not isinstance(run_id_template, str):
            raise ValueError("run_id must be a string")
        run_id = _render_value(run_id_template, context=context, environment=environment_values)
        context["run_id"] = run_id
        flattened["run-id"] = run_id
    else:
        run_id = None
        context["run_id"] = ""

    if "output-dir" in options and ("output-dir" in flattened or run_id is not None):
        output_template = flattened.get("output-dir", "outputs/{environment}/{method}/{run_id}")
        if not isinstance(output_template, str):
            raise ValueError("output_dir must be a string")
        rendered_output = _render_value(output_template, context=context, environment=environment_values)
        output_dir = Path(rendered_output).expanduser()
        flattened["output-dir"] = str(output_dir)
    else:
        output_dir = None

    unknown_options = sorted(set(flattened) - set(options))
    if unknown_options:
        raise ValueError(
            f"algorithm {method!r} does not accept: {', '.join('--' + key for key in unknown_options)}"
        )
    missing_options = sorted(
        name for name, option in options.items() if option.required and flattened.get(name) is None
    )
    if missing_options:
        raise ValueError(f"missing required runner parameters: {', '.join('--' + key for key in missing_options)}")

    extra_dependencies = launcher.get("extra_dependencies", [])
    if not isinstance(extra_dependencies, list) or not all(isinstance(item, str) for item in extra_dependencies):
        raise ValueError("launcher.extra_dependencies must be a list of strings")
    extras = tuple(dict.fromkeys((*spec.extras_for(environment_name), *extra_dependencies)))
    command = ["uv", "run", "--package", "mindmemos-skill"]
    for extra in extras:
        command.extend(("--extra", extra))
    command.extend(("python", str(script_path), "--algorithm", method))
    for name, value in flattened.items():
        _append_option(command, options[name], value)
    command.extend(runner_args or [])

    if environment_name == "alfworld" and "data-root" in flattened:
        data_root = Path(str(flattened["data-root"])).expanduser()
        if not data_root.is_absolute():
            data_root = REPO_ROOT / data_root
        environment_values["ALFWORLD_DATA"] = str(data_root.resolve())

    resolved_config = {
        **config,
        "method": method,
        "environment": environment_name,
        "parameters": rendered_parameters,
        "resolved": {
            "run_id": run_id,
            "output_dir": str(output_dir) if output_dir is not None else None,
            "runner": str(script_path.relative_to(REPO_ROOT)),
            "implementation": spec.module,
            "extras": list(extras),
        },
    }
    return ExperimentInvocation(
        config_path=config_path,
        method=method,
        environment=environment_name,
        run_id=run_id,
        output_dir=output_dir,
        command=command,
        environment_values=environment_values,
        resolved_config=resolved_config,
    )


def persist_resolved_config(invocation: ExperimentInvocation) -> None:
    if invocation.output_dir is None:
        return
    output_dir = invocation.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if not output_dir.is_dir():
        return
    destination = output_dir / "experiment_config.yaml"
    destination.write_text(
        yaml.safe_dump(invocation.resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    try:
        invocation = build_invocation(
            args.config,
            overrides=args.overrides,
            env_file_override=args.env_file,
            runner_args=runner_args,
        )
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": invocation.method,
                    "environment": invocation.environment,
                    "run_id": invocation.run_id,
                    "output_dir": str(invocation.output_dir) if invocation.output_dir is not None else None,
                    "command": shlex.join(invocation.command),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    completed = subprocess.run(invocation.command, cwd=REPO_ROOT, env=invocation.environment_values, check=False)
    if completed.returncode == 0:
        persist_resolved_config(invocation)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
