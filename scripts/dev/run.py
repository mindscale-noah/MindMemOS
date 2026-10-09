"""Start configured local dependencies and API processes with one environment."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import yaml
from dotenv import load_dotenv

from . import constants as env
from .config import finish_plan, memory_document, prepare_environment, skill_document
from .control import ControlSession
from .docker import compose, manage, prepare_collector
from .project import configure_local_sdk, read_local_projects, select_local_project, write_project_keys
from .typing import DependencyPlan

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_CONFIG = ROOT / "config" / "presets" / "local.project.yaml"
DEFAULT_UI_HOME = ROOT / ".mindmemos"


def _parser() -> argparse.ArgumentParser:
    """Build the development command without duplicating application settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="dev", choices=("dev", "docker", "api", "ui", "status", "stop", "clean")
    )
    parser.add_argument("--service", choices=("all", "memory", "skill"), default="all")
    parser.add_argument("--feature", choices=("all", "core", "async"), default="all")
    parser.add_argument("--ui", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--ui-port", type=int, default=os.environ.get(env.INPUT_UI_PORT) or 8765)
    parser.add_argument("--ui-home", type=Path, default=os.environ.get(env.INPUT_UI_HOME) or None)
    parser.add_argument("--memory-config", type=Path, default=os.environ.get(env.INPUT_MEMORY_CONFIG) or None)
    parser.add_argument("--skill-config", type=Path, default=os.environ.get(env.INPUT_SKILL_CONFIG) or None)
    parser.add_argument("--memory-host", default=os.environ.get(env.INPUT_MEMORY_HOST) or None)
    parser.add_argument("--memory-port", type=int, default=os.environ.get(env.INPUT_MEMORY_PORT) or None)
    parser.add_argument("--skill-host", default=os.environ.get(env.INPUT_SKILL_HOST) or None)
    parser.add_argument("--skill-port", type=int, default=os.environ.get(env.INPUT_SKILL_PORT) or None)
    parser.add_argument(
        "--memory-reload", action="store_true", default=os.environ.get(env.INPUT_MEMORY_RELOAD) in {"1", "true", "yes"}
    )
    parser.add_argument(
        "--skill-reload", action="store_true", default=os.environ.get(env.INPUT_SKILL_RELOAD) in {"1", "true", "yes"}
    )
    parser.add_argument(
        "--project",
        default=os.environ.get(env.INPUT_PROJECT) or None,
        help="Client default project; servers load all configured projects",
    )
    parser.add_argument(
        "--project-config", type=Path, default=os.environ.get(env.INPUT_PROJECT_CONFIG) or DEFAULT_PROJECT_CONFIG
    )
    parser.add_argument("--service-auth", action="store_true", default=os.environ.get(env.INPUT_SERVICE_AUTH) == "1")
    parser.add_argument("--plan", action="store_true", help="Validate and preview without starting services")
    return parser


def _write_config(directory: Path, name: str, data: dict) -> Path:
    """Write a private runtime snapshot without modifying the source YAML."""
    path = directory / f"{name}.yaml"
    with path.open("x", encoding="utf-8") as file:
        path.chmod(0o600)
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    return path


def _wait_connections(plan: DependencyPlan, timeout: float = 60) -> None:
    """Wait a bounded time for the actual endpoints used by both applications."""
    deadline = time.monotonic() + timeout
    for host, port in sorted(plan.connections):
        while True:
            try:
                with socket.create_connection((host, port), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Dependency is unreachable at {host}:{port}. Run make docker with the same SERVICE and FEATURE."
                    ) from None
                time.sleep(0.2)


def _run_apis(commands: list[list[str]], *, ready_urls: dict[str, str] | None = None, summary: str = "") -> int:
    """Supervise API process groups and stop siblings when a child exits."""
    _check_ports(commands)
    children = []
    control = ControlSession(ROOT, summary)
    pending = dict(ready_urls or {})
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 180
    announced = False
    try:
        for command in commands:
            children.append(subprocess.Popen(command, cwd=ROOT, start_new_session=True))
        while True:
            if control.poll(announced):
                print("Stopping managed applications.", flush=True)
                return 0
            for child in children:
                status = child.poll()
                if status is not None:
                    name = list(ready_urls)[children.index(child)] if ready_urls else "Application"
                    print(f"{name} exited with status {status}; stopping the other processes.", flush=True)
                    return status if status > 0 else 1
            for name, url in list(pending.items()):
                try:
                    with opener.open(url, timeout=0.5) as response:
                        if response.status == 200:
                            del pending[name]
                except OSError:
                    pass
            if not pending and not announced:
                if summary:
                    print("\nDevelopment services ready\n" + summary, flush=True)
                announced = True
            if pending and time.monotonic() >= deadline:
                raise RuntimeError("Startup timed out waiting for: " + ", ".join(pending))
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 130
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        for child in children:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        control.close()


def _check_ports(commands: list[list[str]]) -> None:
    """Reject occupied listeners instead of reporting an older process as ready."""
    for command in commands:
        if "--port" not in command:
            continue
        port = int(command[command.index("--port") + 1])
        host = command[command.index("--host") + 1] if "--host" in command else "127.0.0.1"
        for family, kind, protocol, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
            with socket.socket(family, kind, protocol) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    listener.bind(address)
                except OSError:
                    raise RuntimeError(
                        f"Cannot bind {host}:{port}; stop its current owner or choose another port"
                    ) from None


def _api_command(module: str, host: str, port: int, reload: bool) -> list[str]:
    """Build an API command using the launcher's Python environment."""
    if not 1 <= port <= 65535:
        raise ValueError("API ports must be between 1 and 65535")
    command = [sys.executable, "-m", "uvicorn", module, "--host", host, "--port", str(port)]
    if reload:
        command.append("--reload")
    return command


def _service_url(command: list[str]) -> str:
    """Return a browser-accessible address for a launched application."""
    host = command[command.index("--host") + 1] if "--host" in command else "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1" if host == "0.0.0.0" else "::1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{command[command.index('--port') + 1]}"


def _summary(args: argparse.Namespace, urls: dict[str, str], plan: DependencyPlan) -> str:
    """Describe selected applications and features without exposing credentials."""
    lines = [f"  Action: {args.action} | Service: {args.service} | Feature: {args.feature}"]
    lines.extend(f"  {name:<16} {urls.get(name, 'NOT STARTED')}" for name in ("Memory API", "Skill API", "UI"))
    for name in ("Memory API", "Skill API"):
        if name in urls:
            lines.append(f"  {name} docs: {urls[name]}/docs")
    lines.extend(
        [
            f"  Kafka tasks:     {'ENABLED' if args.async_tasks else 'DISABLED'}",
            f"  Skill consumer:  {'ENABLED' if args.skill and args.async_tasks else 'DISABLED'}",
            f"  OTel monitoring: {'ENABLED' if args.otel else 'DISABLED'}",
            (
                "  Containers to start: "
                if args.action in {"dev", "docker"}
                else "  Required local containers (not started): "
            )
            + (", ".join(sorted(plan.services)) or "none"),
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Validate application configuration, start dependencies, and supervise APIs.

    Args:
        argv: Optional command-line arguments; None reads the process arguments.

    Returns:
        Exit status from the supervised applications or zero for a valid dry run.
    """
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env", override=False, interpolate=False)
    args = _parser().parse_args(argv)
    if args.action in {"status", "stop", "clean"}:
        return manage(ROOT, args.action)
    if args.action == "docker" and args.ui == "1":
        raise RuntimeError("Docker-only startup does not start UI; use make dev or make ui")
    args.ui = args.action == "ui" or (
        args.action != "docker" and (args.ui == "1" or (args.ui == "auto" and args.action == "dev"))
    )
    args.memory = args.action != "ui" and args.service in {"all", "memory"}
    args.skill = args.action != "ui" and args.service in {"all", "skill"}
    args.async_tasks = args.action != "ui" and args.feature != "core"
    args.otel = args.action != "ui" and args.feature == "all"
    prepare_environment(ROOT / ".env")
    plan = DependencyPlan()
    commands = []
    urls = {}
    ready_urls = {}
    if args.action in {"dev", "api"} and not args.service_auth and not args.project_config.is_file():
        raise RuntimeError(
            f"Local project configuration not found: {args.project_config}. "
            "Set PROJECT_CONFIG to your local project YAML, or use SERVICE_AUTH=1 to keep the service YAML authentication."
        )
    projects = (
        None
        if args.service_auth or args.action in {"docker", "ui"}
        else read_local_projects(args.project_config.expanduser().resolve())
    )
    project = select_local_project(projects, args.project) if projects is not None else None
    with tempfile.TemporaryDirectory(prefix="mindmemos-dev-") as temporary:
        directory = Path(temporary)
        if project is not None:
            memory_keys, skill_key = write_project_keys(projects, directory)
        if args.memory:
            from mindmemos.config.app import build_config, default_config_path

            path = args.memory_config or Path(
                os.environ.get(env.MEMORY_CONFIG)
                or default_config_path(os.environ.get(env.MEMORY_CONFIG_NAME) or "dev")
            )
            document = memory_document(path, plan, async_tasks=args.async_tasks, otel=args.otel)
            if project is not None:
                document["auth"] = {"mode": "api_key", "api_key_file": str(memory_keys)}
            runtime = _write_config(directory, "memory", document)
            build_config(config_path=runtime)
            os.environ[env.MEMORY_CONFIG] = str(runtime)
            commands.append(
                _api_command(
                    "mindmemos.api.app:app",
                    args.memory_host or os.environ.get(env.MEMORY_HOST) or "127.0.0.1",
                    args.memory_port if args.memory_port is not None else int(os.environ.get(env.MEMORY_PORT) or 8000),
                    args.memory_reload,
                )
            )
            urls["Memory API"] = _service_url(commands[-1])
            ready_urls["Memory API"] = urls["Memory API"] + "/healthz"
        if args.skill:
            from mindmemos_skill.api.constants import SKILL_API_HOST_ENV, SKILL_CONFIG_PATH_ENV
            from mindmemos_skill.config.loader import read_skill_config

            path = args.skill_config or Path(os.environ.get(SKILL_CONFIG_PATH_ENV) or "config/mindmemos_skill/dev.yaml")
            host = args.skill_host or os.environ.get(SKILL_API_HOST_ENV) or "127.0.0.1"
            document = skill_document(path, plan, host, async_tasks=args.async_tasks, otel=args.otel)
            if project is not None:
                document["fastapi"]["auth"] = {"mode": "local", "api_key_file": str(skill_key)}
            runtime = _write_config(directory, "skill", document)
            read_skill_config(runtime)
            os.environ[SKILL_CONFIG_PATH_ENV] = str(runtime)
            os.environ[SKILL_API_HOST_ENV] = host
            commands.append(
                _api_command(
                    "mindmemos_skill.api.entries.fastapi:app",
                    host,
                    args.skill_port if args.skill_port is not None else int(os.environ.get(env.SKILL_PORT) or 8001),
                    args.skill_reload,
                )
            )
            urls["Skill API"] = _service_url(commands[-1])
            ready_urls["Skill API"] = urls["Skill API"] + "/health"
        if args.ui:
            if not 1 <= args.ui_port <= 65535:
                raise ValueError("UI port must be between 1 and 65535")
            command = [sys.executable, "-m", "mindmemos_ui.cli", "--port", str(args.ui_port)]
            ui_home = args.ui_home or (DEFAULT_UI_HOME if project is not None else None)
            if ui_home:
                command.extend(["--home", str(ui_home.expanduser().resolve())])
            commands.append(command)
            urls["UI"] = _service_url(command)
            ready_urls["UI"] = urls["UI"] + "/"
        finish_plan(plan)
        if args.action == "docker":
            urls.clear()
            commands.clear()
        ports = [command[command.index("--port") + 1] for command in commands]
        if len(ports) != len(set(ports)):
            raise ValueError("Memory, Skill and UI must use different development ports")
        summary = _summary(args, urls, plan)
        if projects is not None:
            summary += (
                f"\n  Project config:  {args.project_config}"
                f"\n  Server projects: {', '.join(projects.projects)}"
                f"\n  Client default:  {args.project or projects.default_project} ({project.project_id})"
            )
        print("Development startup plan\n" + summary, flush=True)
        if args.plan:
            return 0
        if args.ui and project is not None:
            configure_local_sdk(project, ui_home.expanduser().resolve(), urls)
        if args.action in {"dev", "docker"} and plan.services:
            prepare_collector(ROOT, plan)
            compose(ROOT, "up", "-d", "--wait", "--wait-timeout", "180", *sorted(plan.services))
        if args.action == "docker":
            print("Docker dependencies ready; no API or UI processes were started.", flush=True)
            return 0
        _wait_connections(plan)
        return _run_apis(commands, ready_urls=ready_urls, summary=summary)


def _interrupt(signum, frame) -> None:
    """Route termination through the same child cleanup as Ctrl-C."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError as error:
        print(f"Development startup failed: missing file or executable {error.filename!r}.", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as error:
        print(f"Development startup failed: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        # Config-validation exceptions may contain private YAML values.
        print(
            f"Development startup failed ({type(error).__name__}). Check configuration and dependency availability.",
            file=sys.stderr,
        )
        sys.exit(1)
