"""Bind development APIs and the workbench to one private local project."""

from pathlib import Path

import yaml
from mindmemos_sdk.config import ConfigManager, SDKConfig

from .config import LocalProjectConfig, LocalProjectsConfig


def read_local_projects(path: Path) -> LocalProjectsConfig:
    """Read all identities supported by the local services.

    Args:
        path: Private YAML containing named projects and their permissions.

    Returns:
        Validated projects and the client's default project name.
    """
    if not path.is_file():
        raise ValueError(
            f"Local project configuration not found: {path}. "
            "Copy config/presets/local.project.example.yaml to this path and configure it."
        )
    return LocalProjectsConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def select_local_project(config: LocalProjectsConfig, name: str | None = None) -> LocalProjectConfig:
    """Select the client's default identity without limiting server projects.

    Args:
        config: Complete project collection loaded by both services.
        name: Optional client project name overriding the file's default.

    Returns:
        Project whose credential the workbench sends by default.
    """
    selected = name or config.default_project
    if selected not in config.projects:
        raise ValueError(f"Unknown local project: {selected}")
    return config.projects[selected]


def read_local_project(path: Path, name: str | None = None) -> LocalProjectConfig:
    """Read the client's selected identity from a shared project file.

    Args:
        path: Shared project YAML file.
        name: Optional client project name overriding the default.

    Returns:
        Selected client identity.
    """
    return select_local_project(read_local_projects(path), name)


def write_project_keys(projects: LocalProjectsConfig, directory: Path) -> tuple[Path, Path]:
    """Write every project's credentials in each service's local key format.

    Args:
        projects: All configured identities, algorithms, and permissions.
        directory: Private runtime directory owned by the development launcher.

    Returns:
        Memory and Skill key-table paths supporting all configured projects.
    """
    memory_entries = []
    skill_entries = []
    for project in projects.projects.values():
        key = project.api_key.get_secret_value()
        memory = project.model_dump(exclude={"api_key"}, exclude_none=True)
        memory.update(api_key=key, scopes=[scope for scope in project.scopes if scope.startswith("memory:")])
        memory_entries.append(memory)
        skill_entries.append(
            {
                "project_id": project.project_id,
                "api_key": key,
                "enabled": project.enabled,
                "scopes": [scope for scope in project.scopes if scope.startswith("skill:")],
            }
        )
    paths = []
    for name, data in (
        ("memory-keys", {"api_keys": memory_entries}),
        ("skill-keys", {"api_keys": skill_entries}),
    ):
        path = directory / f"{name}.yaml"
        with path.open("x", encoding="utf-8") as file:
            path.chmod(0o600)
            yaml.safe_dump(data, file, sort_keys=False)
        paths.append(path)
    return paths[0], paths[1]


def configure_local_sdk(project: LocalProjectConfig, home: Path, urls: dict[str, str]) -> None:
    """Save workbench connections using the shared project's credential.

    Args:
        project: Shared local credential accepted by both APIs.
        home: Dedicated development SDK home or an explicit UI home.
        urls: Addresses of the enabled APIs; disabled services remain unconfigured.
    """
    manager = ConfigManager(home)
    values = manager.load_or_default().model_dump(mode="json")
    key = project.api_key.get_secret_value()
    values.update(
        service_type="custom",
        memory_service={
            "base_url": urls.get("Memory API", "http://127.0.0.1:8000"),
            "auth": {"api_key": key if "Memory API" in urls else None},
        },
        skill_service=({"base_url": urls["Skill API"], "auth": {"api_key": key}} if "Skill API" in urls else {}),
    )
    manager.save(SDKConfig.model_validate(values))
