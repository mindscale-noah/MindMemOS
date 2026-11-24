"""End-to-end checks for the local Skill command line interface."""

import subprocess
import sys

import yaml

from .service_config import repository_config


def test_register_list_publish_and_diff_across_processes(tmp_path):
    """Persist versions and compare them through separate CLI invocations."""
    source = tmp_path / "demo-skill"
    source.mkdir()
    skill_file = source / "SKILL.md"
    skill_file.write_text("---\nname: demo-skill\ndescription: Demo\n---\n\n# Demo\nAnswer clearly.\n")
    config_path = tmp_path / "service.yaml"
    config_path.write_text(yaml.safe_dump(repository_config(tmp_path / "home").model_dump(mode="json")))

    def run(*args):
        """Invoke the CLI in a fresh process using the same local workspace."""
        return subprocess.run(
            [sys.executable, "-m", "mindmemos_skill.cli", "--config", str(config_path), *args],
            text=True,
            capture_output=True,
            check=True,
        ).stdout

    registered = run("register", str(source), "--alias", "demo")
    first_version = registered.strip().split()[-1]
    listed = run("list")
    assert "demo-skill" in listed
    assert first_version in listed

    skill_file.write_text("---\nname: demo-skill\ndescription: Demo\n---\n\n# Demo\nAnswer briefly.\n")
    published = run("publish", "demo", str(source), "--base-version", first_version)
    second_version = published.strip().split()[-1]
    assert second_version != first_version

    compared = run("diff", "demo", "--from", first_version, "--to", second_version)
    assert "SKILL.md" in compared
    assert "Answer briefly." in compared
    assert "Answer clearly." in compared


def test_direct_cli_rejects_authenticated_deployment_before_opening_storage(tmp_path):
    """Authenticated deployments require the SDK HTTP client rather than a local bypass."""
    config = repository_config(tmp_path / "home").model_dump(mode="json")
    config["fastapi"]["auth"] = {"mode": "gateway_jwt", "gateway_jwt_secret": "test-secret"}
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, "-m", "mindmemos_skill.cli", "--config", str(path), "list"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires none mode" in result.stderr
    assert not (tmp_path / "home").exists()
