"""Exercise official and custom connection setup through the public CLI."""

from __future__ import annotations

from mindmemos_sdk.config import ConfigManager, paths
from mindmemos_sdk.config.constants import MINDMEMOS_HOME_ENV

from mindmemos_sdk import cli


def test_config_home_set_selects_default_for_new_clients(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "_home_preference_path", lambda: tmp_path / "preferences" / "home.json")
    monkeypatch.delenv(MINDMEMOS_HOME_ENV, raising=False)
    directory = tmp_path / "workspace" / ".mindmemos"

    assert cli.main(["config", "home", "set", str(directory)]) == 0
    assert str(directory) in capsys.readouterr().out
    assert ConfigManager().config_dir == directory
    assert not directory.exists()


def test_official_auth_shows_both_services_with_masked_key(tmp_path, monkeypatch, capsys):
    """Official setup stores one credential and shows both effective connections."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))

    assert (
        cli.main(["auth", "--service-type", "official", "--memory-api-key", "official-secret", "--user-id", "alice"])
        == 0
    )

    config = ConfigManager().load()
    assert config.service_type == "official"
    assert config.skill_service.base_url is None
    assert config.skill_service.auth is None
    assert config.resolve_skill_connection() == ("https://mindmemos.cn/v2/skills", "official-secret")

    capsys.readouterr()
    assert cli.main(["config", "show"]) == 0
    output = capsys.readouterr().out
    assert "deployment:  official" in output
    assert "skill URL:   https://mindmemos.cn/v2/skills" in output
    assert "official-secret" not in output


def test_custom_auth_configures_independent_services(tmp_path, monkeypatch, capsys):
    """Custom setup persists separate destinations and credentials."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))

    assert (
        cli.main(
            [
                "auth",
                "--service-type",
                "custom",
                "--memory-base-url",
                "https://memory.example",
                "--memory-api-key",
                "memory-secret",
                "--skill-base-url",
                "https://skill.example",
                "--skill-api-key",
                "skill-secret",
                "--user-id",
                "alice",
            ]
        )
        == 0
    )

    config = ConfigManager().load()
    assert config.service_type == "custom"
    assert config.resolve_memory_connection() == ("https://memory.example", "memory-secret")
    assert config.resolve_skill_connection() == ("https://skill.example/v1/skills", "skill-secret")

    capsys.readouterr()
    assert cli.main(["config", "show"]) == 0
    output = capsys.readouterr().out
    assert "deployment:  custom" in output
    assert "memory URL:  https://memory.example" in output
    assert "skill URL:   https://skill.example/v1/skills" in output
    assert "memory-secret" not in output
    assert "skill-secret" not in output


def test_official_auth_accepts_an_independent_skill_only_key(tmp_path, monkeypatch):
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))
    assert cli.main(["auth", "--service-type", "official", "--skill-api-key", "skill-key", "--user-id", "alice"]) == 0
    config = ConfigManager().load()
    assert config.resolve_memory_connection() == ("https://mindmemos.cn", None)
    assert config.resolve_skill_connection() == ("https://mindmemos.cn/v2/skills", "skill-key")


def test_custom_auth_can_configure_only_memory(tmp_path, monkeypatch):
    """A single custom gateway remains a supported explicit choice."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))

    assert (
        cli.main(
            [
                "auth",
                "--service-type",
                "custom",
                "--memory-base-url",
                "http://127.0.0.1:8000",
                "--memory-api-key",
                "local-key",
                "--no-skill",
                "--user-id",
                "alice",
            ]
        )
        == 0
    )

    config = ConfigManager().load()
    assert config.skill_service.base_url is None
    assert config.resolve_skill_connection() == (None, None)


def test_interactive_custom_auth_prompts_for_both_services(tmp_path, monkeypatch):
    """The interactive flow asks for each service URL directly."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))
    answers = iter(["custom", "https://memory.example", "https://skill.example", "alice"])
    secrets = iter(["memory-key", "skill-key"])
    prompts: list[str] = []

    def prompt(message: str) -> str:
        prompts.append(message)
        return next(answers)

    monkeypatch.setattr(cli, "_prompt", prompt)
    monkeypatch.setattr(cli, "_prompt_secret", lambda _message: next(secrets))

    assert cli.main(["auth"]) == 0

    config = ConfigManager().load()
    assert config.service_type == "custom"
    assert config.resolve_memory_connection() == ("https://memory.example", "memory-key")
    assert config.resolve_skill_connection() == ("https://skill.example/v1/skills", "skill-key")
    assert prompts[2] == "Skill service URL [skip]: "


def test_invalid_memory_url_is_retried_before_other_fields(tmp_path, monkeypatch, capsys):
    """An invalid Memory URL is rejected at its prompt without losing progress."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))
    answers = iter(["custom", "x2", "https://memory.example", "", "alice"])
    events: list[str] = []

    def prompt(message: str) -> str:
        events.append(message)
        return next(answers)

    def prompt_secret(message: str) -> str:
        events.append(message)
        return "memory-key"

    monkeypatch.setattr(cli, "_prompt", prompt)
    monkeypatch.setattr(cli, "_prompt_secret", prompt_secret)

    assert cli.main(["auth"]) == 0
    assert events[:4] == [
        "Deployment [official/custom] [official]: ",
        "Memory service URL: ",
        "Memory service URL: ",
        "Skill service URL [skip]: ",
    ]
    assert "Invalid Memory service URL" in capsys.readouterr().out
    assert ConfigManager().load().resolve_memory_connection() == ("https://memory.example", "memory-key")


def test_invalid_skill_url_is_retried_at_skill_prompt(tmp_path, monkeypatch, capsys):
    """The Skill URL is checked before requesting credentials."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))
    answers = iter(["custom", "https://memory.example", "not-a-url", "https://skill.example", "alice"])
    secrets = iter(["memory-key", "skill-key"])
    events: list[str] = []

    def prompt(message: str) -> str:
        events.append(message)
        return next(answers)

    def prompt_secret(message: str) -> str:
        events.append(message)
        return next(secrets)

    monkeypatch.setattr(cli, "_prompt", prompt)
    monkeypatch.setattr(cli, "_prompt_secret", prompt_secret)

    assert cli.main(["auth"]) == 0
    assert events[2:5] == [
        "Skill service URL [skip]: ",
        "Skill service URL [skip]: ",
        "Memory API key: ",
    ]
    assert "Invalid Skill service URL" in capsys.readouterr().out
    assert ConfigManager().load().resolve_skill_connection() == ("https://skill.example/v1/skills", "skill-key")


def test_invalid_explicit_url_fails_before_prompting(tmp_path, monkeypatch, capsys):
    """Noninteractive invalid URL arguments fail without writing settings."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))
    monkeypatch.setattr(cli, "_prompt", lambda _message: (_ for _ in ()).throw(AssertionError("unexpected prompt")))

    assert cli.main(["auth", "--service-type", "custom", "--memory-base-url", "x2"]) == 2
    assert "Invalid Memory service URL" in capsys.readouterr().out
    assert not ConfigManager().exists()


def test_memory_reauth_preserves_previous_skill_connection(tmp_path, monkeypatch):
    """Updating Memory leaves an existing independent Skill connection intact."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))
    assert (
        cli.main(
            [
                "auth",
                "--service-type",
                "custom",
                "--memory-base-url",
                "https://memory.example",
                "--memory-api-key",
                "memory-key",
                "--skill-base-url",
                "https://skill.example",
                "--skill-api-key",
                "skill-key",
                "--user-id",
                "alice",
            ]
        )
        == 0
    )
    prompts: list[str] = []

    def prompt(message: str) -> str:
        prompts.append(message)
        return ""

    monkeypatch.setattr(cli, "_prompt", prompt)

    assert (
        cli.main(
            [
                "auth",
                "--service-type",
                "custom",
                "--memory-base-url",
                "https://memory.example",
                "--memory-api-key",
                "memory-key",
                "--user-id",
                "alice",
            ]
        )
        == 0
    )

    config = ConfigManager().load()
    assert config.resolve_skill_connection() == ("https://skill.example/v1/skills", "skill-key")
    assert prompts == []


def test_switch_to_official_clears_independent_skill_connection(tmp_path, monkeypatch):
    """Changing deployment does not retain an old custom Skill destination."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))
    assert (
        cli.main(
            [
                "auth",
                "--service-type",
                "custom",
                "--memory-base-url",
                "https://memory.example",
                "--memory-api-key",
                "memory-key",
                "--skill-base-url",
                "https://skill.example",
                "--skill-api-key",
                "skill-key",
                "--user-id",
                "alice",
            ]
        )
        == 0
    )

    assert (
        cli.main(["auth", "--service-type", "official", "--memory-api-key", "official-key", "--user-id", "alice"]) == 0
    )

    config = ConfigManager().load()
    assert config.service_type == "official"
    assert config.skill_service.base_url is None
    assert config.skill_service.auth is None
    assert config.resolve_skill_connection() == ("https://mindmemos.cn/v2/skills", "official-key")


def test_changed_skill_destination_requires_a_new_key(tmp_path, monkeypatch, capsys):
    """An old Skill secret cannot silently be sent to a new custom host."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))
    assert (
        cli.main(
            [
                "auth",
                "--service-type",
                "custom",
                "--memory-base-url",
                "https://memory.example",
                "--memory-api-key",
                "memory-key",
                "--skill-base-url",
                "https://old-skill.example",
                "--skill-api-key",
                "old-skill-key",
                "--user-id",
                "alice",
            ]
        )
        == 0
    )
    attempts = iter(["", ""])

    def prompt_secret(_message: str) -> str:
        try:
            return next(attempts)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(cli, "_prompt_secret", prompt_secret)

    assert (
        cli.main(
            [
                "auth",
                "--service-type",
                "custom",
                "--memory-base-url",
                "https://memory.example",
                "--skill-base-url",
                "https://new-skill.example",
                "--user-id",
                "alice",
            ]
        )
        == 2
    )

    output = capsys.readouterr().out
    assert "Skill API key is required" in output
    assert "input cancelled" in output
    assert ConfigManager().load().resolve_skill_connection() == ("https://old-skill.example/v1/skills", "old-skill-key")


def test_official_auth_rejects_custom_url(tmp_path, monkeypatch, capsys):
    """The official choice cannot silently accept an unrelated service URL."""
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path))

    assert cli.main(["auth", "--service-type", "official", "--memory-base-url", "https://memory.example"]) == 2

    assert "does not accept custom service URLs" in capsys.readouterr().out
    assert not ConfigManager().exists()
