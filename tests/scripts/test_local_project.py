"""Validate shared development identity without changing deployed authentication."""

import pytest
import yaml
from mindmemos.api.auth import ApiKeyAuthProvider
from mindmemos.errors import AuthenticationError
from mindmemos_sdk.config import ConfigManager
from mindmemos_skill.api.auth import LocalKeyResolver, SkillHttpError
from pydantic import SecretStr

from scripts.dev.config import LocalProjectConfig, LocalProjectsConfig
from scripts.dev.project import configure_local_sdk, read_local_project, read_local_projects, write_project_keys


@pytest.mark.parametrize("algorithm", ["vanilla", "schema"])
def test_both_authenticators_resolve_shared_project_and_reject_other_keys(tmp_path, algorithm):
    project = LocalProjectConfig(
        project_id="existing-project",
        api_key="private-test-key",
        key_id="existing-key",
        memory_algorithm=algorithm,
        scopes=["memory:read", "memory:write", "skill:write"],
        project_override_config={"algo_config": {"add": {"schema": {"version": "v1"}}}},
    )
    memory_path, skill_path = write_project_keys(
        LocalProjectsConfig(default_project="test", projects={"test": project}), tmp_path
    )
    memory = ApiKeyAuthProvider(api_key_file=memory_path)
    skill = LocalKeyResolver(skill_path)
    memory_identity = memory.resolve_api_key("private-test-key")
    skill_identity = skill("private-test-key")
    assert memory_identity.project_id == skill_identity.project_id == "existing-project"
    assert memory_identity.key_id == "existing-key"
    assert memory_identity.memory_algorithm == algorithm
    assert memory_identity.project_override_config == project.project_override_config
    assert memory_identity.scopes == ["memory:read", "memory:write"]
    assert skill_identity.scopes == {"skill:write"}
    for path in (memory_path, skill_path):
        assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(AuthenticationError):
        memory.resolve_api_key("wrong-key")
    with pytest.raises(SkillHttpError):
        skill("wrong-key")


def test_sdk_connections_follow_project_and_effective_ports_preserving_options(tmp_path):
    home = tmp_path / "sdk"
    manager = ConfigManager(home)
    config = manager.load_or_default()
    config.network.timeout_seconds = 99
    config.defaults.user_id = "existing-user"
    manager.save(config)
    project = LocalProjectConfig(
        project_id="existing-project",
        api_key="shared-key",
        scopes=["memory:read", "memory:write", "skill:read", "skill:write"],
    )
    configure_local_sdk(project, home, {"Memory API": "http://127.0.0.1:18000", "Skill API": "http://127.0.0.1:18001"})
    saved = manager.load()
    assert saved.resolve_memory_connection() == ("http://127.0.0.1:18000", "shared-key")
    assert saved.resolve_skill_connection() == ("http://127.0.0.1:18001", "shared-key")
    assert saved.network.timeout_seconds == 99
    assert saved.defaults.user_id == "existing-user"
    project = project.model_copy(update={"api_key": SecretStr("rotated-key")})
    configure_local_sdk(project, home, {"Memory API": "http://127.0.0.1:18000", "Skill API": "http://127.0.0.1:18001"})
    assert manager.load().resolve_skill_connection()[1] == "rotated-key"


def test_skill_only_sdk_does_not_send_project_key_to_memory(tmp_path):
    project = LocalProjectConfig(
        project_id="local-project",
        api_key="shared-key",
        scopes=["memory:read", "memory:write", "skill:read", "skill:write"],
    )
    configure_local_sdk(project, tmp_path, {"Skill API": "http://127.0.0.1:18001"})
    saved = ConfigManager(tmp_path).load()
    assert saved.resolve_memory_connection()[1] is None
    assert saved.resolve_skill_connection() == ("http://127.0.0.1:18001", "shared-key")


def test_missing_project_fails_with_setup_instruction(tmp_path):
    with pytest.raises(ValueError, match="local.project.example.yaml"):
        read_local_project(tmp_path / "missing.yaml")


def test_invalid_project_does_not_expose_key(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "default_project": "local",
                "projects": {
                    "local": {
                        "project_id": "local-project",
                        "api_key": "secret-value",
                        "memory_algorithm": "invalid",
                        "scopes": [],
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError) as error:
        read_local_project(path)
    assert "secret-value" not in str(error.value)


def test_shared_file_selects_default_joint_project_and_explicit_memory_projects():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "config" / "presets" / "local.project.example.yaml"
    joint = read_local_project(path)
    vanilla = read_local_project(path, "vanilla")
    schema = read_local_project(path, "schema")
    assert joint.project_id == "proj-dev-memory-skill"
    assert set(joint.scopes) == {"memory:read", "memory:write", "skill:read", "skill:write"}
    assert vanilla.memory_algorithm == "vanilla"
    assert schema.memory_algorithm == "schema"
    assert all(scope.startswith("memory:") for project in (vanilla, schema) for scope in project.scopes)
    with pytest.raises(ValueError, match="Unknown local project"):
        read_local_project(path, "missing")


def test_disabled_project_is_rejected_by_both_services(tmp_path):
    project = LocalProjectConfig(project_id="disabled", api_key="shared-key", enabled=False, scopes=[])
    memory_path, skill_path = write_project_keys(
        LocalProjectsConfig(default_project="test", projects={"test": project}), tmp_path
    )
    with pytest.raises(AuthenticationError) as memory_error:
        ApiKeyAuthProvider(api_key_file=memory_path).resolve_api_key("shared-key")
    assert memory_error.value.code == "auth.api_key_disabled"
    with pytest.raises(SkillHttpError) as skill_error:
        LocalKeyResolver(skill_path)("shared-key")
    assert skill_error.value.code == "auth.api_key_disabled"


def test_memory_only_project_does_not_gain_skill_write_permissions(tmp_path):
    project = LocalProjectConfig(project_id="memory-only", api_key="shared-key", scopes=["memory:read"])
    memory_path, skill_path = write_project_keys(
        LocalProjectsConfig(default_project="test", projects={"test": project}), tmp_path
    )
    assert ApiKeyAuthProvider(api_key_file=memory_path).resolve_api_key("shared-key").scopes == ["memory:read"]
    assert LocalKeyResolver(skill_path)("shared-key").scopes == set()


def test_one_server_accepts_all_project_keys_and_preserves_project_isolation(tmp_path):
    from pathlib import Path

    from mindmemos_skill.api.auth import authorize_project

    path = Path(__file__).resolve().parents[2] / "config" / "presets" / "local.project.example.yaml"
    projects = read_local_projects(path)
    memory_path, skill_path = write_project_keys(projects, tmp_path)
    memory = ApiKeyAuthProvider(api_key_file=memory_path)
    skill = LocalKeyResolver(skill_path)
    for name, project in projects.projects.items():
        key = project.api_key.get_secret_value()
        memory_identity = memory.resolve_api_key(key)
        skill_identity = skill(key)
        assert memory_identity.project_id == skill_identity.project_id == project.project_id
        assert memory_identity.memory_algorithm == project.memory_algorithm
        assert skill_identity.scopes == ({"skill:read", "skill:write"} if name == "local" else set())
        with pytest.raises(PermissionError):
            authorize_project(skill_identity, "another-project")
    projects.default_project = "schema"
    assert memory.resolve_api_key(projects.projects["vanilla"].api_key.get_secret_value()).memory_algorithm == "vanilla"
    assert skill(projects.projects["local"].api_key.get_secret_value()).project_id == "proj-dev-memory-skill"
