"""Verify configuration initialization and the Makefile entry point."""

import subprocess
from pathlib import Path

from scripts.dev import init

ROOT = Path(__file__).resolve().parents[2]


def test_init_copies_all_templates_and_keeps_user_configuration(tmp_path, monkeypatch, capsys):
    """Copy missing files, retain credentials, and show both source and destination."""
    monkeypatch.setattr(init, "ROOT", tmp_path)
    for name in init.TEMPLATES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example configuration")
    for name in init.EVAL_DIRECTORIES:
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / "benchmark.example.yaml").write_text("benchmark configuration")
    (tmp_path / ".env").write_text("existing private settings")
    assert init.main() == 0
    output = capsys.readouterr().out
    assert "file skipped: .env.example -> .env (destination already exists)" in output
    assert "file copied: config/mindmemos_eval/dev.example.yaml -> config/mindmemos_eval/dev.yaml" in output
    assert (tmp_path / "config/mindmemos_eval/env_test/dev.yaml").exists()
    assert output.index("Next, configure") > output.rindex("file copied:")
    assert (tmp_path / ".env").read_text() == "existing private settings"
    for name in init.EVAL_DIRECTORIES:
        assert (tmp_path / name / "benchmark.yaml").read_text() == "benchmark configuration"
    assert init.main() == 0
    assert "file copied:" not in capsys.readouterr().out


def test_init_missing_template_fails_before_copying(tmp_path, monkeypatch, capsys):
    """Do not claim initialization success when required templates are missing."""
    monkeypatch.setattr(init, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("example")
    assert init.main() == 1
    assert not (tmp_path / ".env").exists()
    assert "Missing configuration template" in capsys.readouterr().err


def test_make_init_uses_the_initializer():
    """The Make entry point runs initialization without starting applications."""
    result = subprocess.run(["make", "-n", "init"], cwd=ROOT, capture_output=True, text=True, check=True)
    assert "run --no-sync python -m scripts.dev.init" in result.stdout


def test_failed_copy_does_not_leave_a_partial_configuration(tmp_path, monkeypatch, capsys):
    """A failed copy must not make the next initialization skip an incomplete file."""
    monkeypatch.setattr(init, "ROOT", tmp_path)
    monkeypatch.setattr(init, "TEMPLATES", (".env.example",))
    (tmp_path / ".env.example").write_text("complete template")

    def fail(original, local):
        local.write(b"partial")
        raise OSError("disk write failed")

    monkeypatch.setattr(init.shutil, "copyfileobj", fail)
    assert init.main() == 1
    assert not (tmp_path / ".env").exists()
    assert "file copy failed:" in capsys.readouterr().err


def test_init_applies_env_to_all_real_configuration_locations(tmp_path, monkeypatch, capsys):
    """Apply defaults to existing and copied files without rewriting templates or unrelated settings."""
    import yaml

    monkeypatch.setattr(init, "ROOT", tmp_path)
    for name in init.TEMPLATES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((ROOT / name).read_text())
    for source in (ROOT / "config").rglob("*.yaml"):
        # Public templates and credential-free benchmark files exercise the actual layouts.
        if source.name.endswith(".example.yaml"):
            destination = tmp_path / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source.read_text())
    memory = tmp_path / "config/mindmemos/dev.yaml"
    memory.write_text((ROOT / "config/mindmemos/dev.example.yaml").read_text())
    env_lines = []
    settings = {}
    for kind, fields in init.MODEL_ENV_FIELDS.items():
        settings[kind] = {}
        for field, variable in fields.items():
            value = 512 if field == "dimensions" else f"{kind}-{field}-secret:#${{LITERAL}}"
            settings[kind][field] = value
            env_lines.append(f"{variable}='{value}'")
    (tmp_path / ".env").write_text("\n".join(env_lines))
    template = tmp_path / "config/mindmemos/dev.example.yaml"
    original_template = template.read_bytes()
    monkeypatch.setenv(init.CHAT_MODEL_ENV, "process-value-must-not-override-dotenv")
    assert init.main() == 0
    output = capsys.readouterr().out
    assert "secret" not in output
    assert template.read_bytes() == original_template
    data = yaml.safe_load(memory.read_text())
    for kind, router in (
        ("chat", "chat_model_router"),
        ("embedding", "embed_model_router"),
        ("rerank", "rerank_model_router"),
    ):
        for endpoint in data[router]["endpoints"]:
            assert endpoint["model"] == settings[kind]["model"]
            assert endpoint["api_key"] == settings[kind]["api_key"]
            assert endpoint["api_base"] == settings[kind]["base_url"]
    assert data["database"]["qdrant"]["vector_size"] == 512
    skill = yaml.safe_load((tmp_path / "config/mindmemos_skill/dev.yaml").read_text())
    for preset in skill["service"]["agents"].values():
        if any(field in preset["config"] for field in ("api_key", "base_url")):
            assert preset["config"]["model"] == settings["chat"]["model"]
    benchmark = yaml.safe_load((tmp_path / "config/mindmemos_eval/dev.yaml").read_text())
    assert benchmark["models"]["target"]["model"] == settings["chat"]["model"]
    assert benchmark["models"]["optimizer"]["api_key"] == settings["chat"]["api_key"]
    direct = yaml.safe_load((tmp_path / "config/mindmemos_eval/env_test/dev.yaml").read_text())
    assert direct["environment"]["config"]["agent"]["model"] == settings["chat"]["model"]
    assert direct["environment"]["config"]["agent"]["api_key"] == settings["chat"]["api_key"]
    original = memory.read_bytes()
    assert init.main() == 0
    assert memory.read_bytes() == original
    assert "model settings updated:" not in capsys.readouterr().out


def test_partial_chat_settings_preserve_empty_fields_comments_and_aliases(tmp_path, monkeypatch):
    """Only the supplied key changes, including flow mappings and YAML aliases."""
    monkeypatch.setattr(init, "ROOT", tmp_path)
    monkeypatch.setattr(init, "TEMPLATES", (".env.example",))
    (tmp_path / ".env.example").write_text("")
    (tmp_path / ".env").write_text(f"{init.CHAT_API_KEY_ENV}=new-key\n{init.CHAT_MODEL_ENV}=\n")
    path = tmp_path / "config/custom.yaml"
    path.parent.mkdir()
    original = """# Keep this comment
optimizer: &model {model: old, api_key: &key old-key, base_url: null} # Keep inline
router: *model
other_key: *key
server: {base_url: http://localhost:8000, api_key: local-key}
"""
    path.write_text(original)
    assert init.main() == 0
    assert path.read_text() == original.replace("&key old-key", '&key "new-key"')


def test_invalid_embedding_defaults_fail_before_copying(tmp_path, monkeypatch, capsys):
    """Missing, fractional and nonpositive dimensions fail without changing configurations."""
    monkeypatch.setattr(init, "ROOT", tmp_path)
    monkeypatch.setattr(init, "TEMPLATES", (".env.example",))
    (tmp_path / ".env.example").write_text("")
    for dimensions in ("", "0", "-1", "1.5", "invalid"):
        (tmp_path / ".env").write_text(
            f"{init.EMBEDDING_MODEL_ENV}=embedding\n{init.EMBEDDING_DIMENSIONS_ENV}={dimensions}\n"
        )
        assert init.main() == 1
        assert init.EMBEDDING_DIMENSIONS_ENV in capsys.readouterr().err


def test_invalid_yaml_does_not_partially_update_model_settings(tmp_path, monkeypatch, capsys):
    """Validate every file before overwriting any existing configuration and hide input secrets."""
    monkeypatch.setattr(init, "ROOT", tmp_path)
    monkeypatch.setattr(init, "TEMPLATES", (".env.example",))
    (tmp_path / ".env.example").write_text("")
    (tmp_path / ".env").write_text(f"{init.CHAT_MODEL_ENV}=new\n")
    directory = tmp_path / "config"
    directory.mkdir()
    original = "model: old\napi_key: old-key\nbase_url: null\n"
    (directory / "a.yaml").write_text(original)
    (directory / "z.yaml").write_text("model: [secret-invalid\n")
    assert init.main() == 1
    assert (directory / "a.yaml").read_text() == original
    assert "secret-invalid" not in capsys.readouterr().err
