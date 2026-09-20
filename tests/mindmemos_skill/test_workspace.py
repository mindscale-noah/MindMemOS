"""Local service assembly reuses identity and preserves explicit scope boundaries."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
import yaml
from mindmemos_skill.api import ServiceConfig, create_service
from mindmemos_skill.api.factory import resolve_local_settings
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.persistence.constants import SKILL_GRAPHS_TABLE
from mindmemos_skill.persistence.db_services.skill_management.typing import SkillGraphRecord
from mindmemos_skill.pipelines.typing import ListRequest, RegisterRequest, ShowRequest


def test_concurrent_initialization_and_stable_identity(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("MINDMEMOS_HOME", str(home))
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda _: resolve_local_settings(), range(12)))
    assert len({config.project_id for _, config in results}) == 1
    with closing(create_service()) as first:
        identity = first.local_project_id
        path = first.backends["database"].engine.url.database
    monkeypatch.chdir(tmp_path)
    with closing(create_service()) as second:
        assert second.local_project_id == identity
        assert second.backends["database"].engine.url.database == path == str(home / "data/central.db")


def test_reopen_and_scope_isolation(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "SKILL.md").write_text("# Example")
    home = tmp_path / "home"
    with closing(create_service(home=home)) as service:
        item = service.register_skill(RegisterRequest(source=str(root), name="Example"))
        service.backends["database"].insert(
            SKILL_GRAPHS_TABLE,
            [
                SkillGraphRecord(
                    project_id=service.local_project_id, graph_id="graph", name="Example graph"
                ).model_dump()
            ],
        )
        with pytest.raises(ValueError, match="bound local scope"):
            service.list_skills(ListRequest(project_id="other-local-scope"))
    with closing(create_service(home=home)) as reopened:
        assert reopened.list_skills(ListRequest())[0].skill_id == item.skill.skill_id
        graphs = reopened.graph_queries()
        assert graphs.list_graphs()[0]["name"] == "Example graph"
    with closing(create_service(home=home, project_id="other-local-scope")) as other:
        assert other.list_skills(ListRequest()) == []
        graphs = other.graph_queries()
        assert graphs.list_graphs() == []
        with pytest.raises(SkillNotFoundError):
            other.show_skill(ShowRequest(skill_ref=item.skill.skill_id))
        with pytest.raises(ValueError):
            graphs.version(item.version.version_id)
    with closing(create_service(home=tmp_path / "different-home")) as different:
        assert different.local_project_id != item.skill.project_id
        assert different.list_skills(ListRequest()) == []
    assert resolve_local_settings(home)[1].project_id == item.skill.project_id


def test_home_precedence_and_invalid_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("MINDMEMOS_HOME", str(tmp_path / "env"))
    monkeypatch.setenv("MINDMEMOS_CONFIG_DIR", str(tmp_path / "legacy"))
    assert resolve_local_settings()[0] == tmp_path / "env"
    assert resolve_local_settings(tmp_path / "explicit")[0] == tmp_path / "explicit"
    manifest = tmp_path / "env/workspace.json"
    manifest.write_text("broken")
    with pytest.raises(ValueError):
        create_service()
    assert manifest.read_text() == "broken"


def test_omitted_database_url_uses_home_without_default_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("MINDMEMOS_HOME", str(tmp_path / "home"))
    config = yaml.safe_load(Path("config/mindmemos_skill/default.yaml").read_text())
    config["backends"]["database"]["config"].pop("url", None)
    with closing(create_service(ServiceConfig.model_validate(config))) as service:
        service.initialize_storage()
        with closing(create_service()) as local:
            assert service.backends["database"].engine.url.database == local.backends["database"].engine.url.database
        with pytest.raises(ValueError, match="project_id is required"):
            service.list_skills(ListRequest())
        assert service.list_skills(ListRequest(project_id="explicit")) == []


def test_concurrent_first_open(tmp_path):
    def open_one(_):
        with closing(create_service(home=tmp_path / "concurrent")) as service:
            assert service.list_skills(ListRequest()) == []
            return service.local_project_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = list(executor.map(open_one, range(8)))
    assert len(set(ids)) == 1


@pytest.mark.parametrize("override", [{"home": "unused"}, {"database_path": "unused.db"}, {"project_id": "local"}])
def test_explicit_config_rejects_local_overrides(override):
    from mindmemos_skill.api import ServiceConfig

    with pytest.raises(ValueError, match="cannot be combined"):
        create_service(ServiceConfig(), **override)
