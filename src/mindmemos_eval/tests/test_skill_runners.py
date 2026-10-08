"""Tests for skill benchmark CLI routing."""

from mindmemos_eval.cli import build_arg_parser


def test_skill_local_command_accepts_preflight_options() -> None:
    """Expose local training evaluation without changing the legacy benchmark name."""
    parser = build_arg_parser()
    local = parser.parse_args(["skill", "local", "--config", "one.yaml", "--output-dir", "run", "--preflight-only"])
    legacy = parser.parse_args(["skill", "spreadsheetbench"])
    assert (local.benchmark, local.config, local.output_dir, local.preflight_only) == (
        "local",
        "one.yaml",
        "run",
        True,
    )
    assert legacy.benchmark == "spreadsheetbench"


AMBIENT_LLM_ENV = {
    "OPENAI_API_KEY": "ambient-secret",
    "OPENAI_ENDPOINT": "https://ambient.invalid/v1",
    "OPENAI_BASE_URL": "https://ambient.invalid/v1",
    "OPENAI_MODEL": "ambient-model",
}
"""Ambient model settings ignored by benchmark configuration."""


def test_skill_model_arguments_ignore_environment(monkeypatch):
    """Use explicit CLI configuration instead of ambient model connection values."""
    for variable, value in AMBIENT_LLM_ENV.items():
        monkeypatch.setenv(variable, value)
    parser = build_arg_parser()
    args = parser.parse_args(["skill", "spreadsheetbench"])
    assert args.api_key is None and args.base_url is None
    assert args.model == "gpt-5.4-mini"
    explicit = parser.parse_args(
        [
            "skill",
            "spreadsheetbench",
            "--api-key",
            "configured-secret",
            "--base-url",
            "https://configured.invalid/v1",
        ]
    )
    assert explicit.api_key == "configured-secret"
    assert explicit.base_url == "https://configured.invalid/v1"
