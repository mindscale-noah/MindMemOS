"""Initialize local configurations and apply optional model settings from .env."""

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from dotenv import dotenv_values
from pydantic import PositiveInt, TypeAdapter, ValidationError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

ROOT = Path(__file__).resolve().parents[2]
"""Repository root containing configuration templates."""

TEMPLATES = (
    ".env.example",
    "config/mindmemos/dev.example.yaml",
    "config/mindmemos_skill/dev.example.yaml",
    "config/mindmemos_eval/dev.example.yaml",
    "config/mindmemos_eval/env_test/dev.example.yaml",
    "config/mindmemos_eval/treeskill/dev.example.yaml",
    "config/presets/local.project.example.yaml",
)
"""Required templates copied to corresponding local filenames."""

EVAL_DIRECTORIES = ("config/mindmemos_eval/memory_evaluation", "config/mindmemos_eval/dreaming_evaluation")
"""Specific evaluation directories whose example configurations are initialized."""

CHAT_API_KEY_ENV = "MINDMEMOS_CHAT_API_KEY"
CHAT_BASE_URL_ENV = "MINDMEMOS_CHAT_BASE_URL"
CHAT_MODEL_ENV = "MINDMEMOS_CHAT_MODEL"
EMBEDDING_API_KEY_ENV = "MINDMEMOS_EMBEDDING_API_KEY"
EMBEDDING_BASE_URL_ENV = "MINDMEMOS_EMBEDDING_BASE_URL"
EMBEDDING_MODEL_ENV = "MINDMEMOS_EMBEDDING_MODEL"
EMBEDDING_DIMENSIONS_ENV = "MINDMEMOS_EMBEDDING_DIMENSIONS"
RERANK_API_KEY_ENV = "MINDMEMOS_RERANK_API_KEY"
RERANK_BASE_URL_ENV = "MINDMEMOS_RERANK_BASE_URL"
RERANK_MODEL_ENV = "MINDMEMOS_RERANK_MODEL"

MODEL_ENV_FIELDS = {
    "chat": {"api_key": CHAT_API_KEY_ENV, "base_url": CHAT_BASE_URL_ENV, "model": CHAT_MODEL_ENV},
    "embedding": {
        "api_key": EMBEDDING_API_KEY_ENV,
        "base_url": EMBEDDING_BASE_URL_ENV,
        "model": EMBEDDING_MODEL_ENV,
        "dimensions": EMBEDDING_DIMENSIONS_ENV,
    },
    "rerank": {"api_key": RERANK_API_KEY_ENV, "base_url": RERANK_BASE_URL_ENV, "model": RERANK_MODEL_ENV},
}
"""Model kinds mapped to YAML fields and optional .env variable names."""


def _read_model_settings() -> dict[str, dict[str, str | int]]:
    """Read nonempty .env fields without process overrides or variable interpolation."""
    env_path = ROOT / ".env"
    values = dotenv_values(env_path if env_path.exists() else ROOT / ".env.example", interpolate=False)
    settings = {}
    for kind, fields in MODEL_ENV_FIELDS.items():
        settings[kind] = {
            field: values[variable].strip()
            for field, variable in fields.items()
            if values.get(variable) and values[variable].strip()
        }
    dimensions = settings["embedding"].get("dimensions")
    if dimensions is not None:
        try:
            settings["embedding"]["dimensions"] = TypeAdapter(PositiveInt).validate_python(dimensions)
        except ValidationError:
            raise ValueError(f"{EMBEDDING_DIMENSIONS_ENV} must be a positive integer") from None
    elif settings["embedding"]:
        raise ValueError(f"Set {EMBEDDING_DIMENSIONS_ENV} when configuring embedding in .env")
    return settings


def _replace_model_settings(text: str, settings: dict[str, dict[str, str | int]]) -> str:
    """Replace model scalars through YAML positions, retaining comments and layout."""
    replacements = {}
    visited = set()

    def replace(node: Node, value: str | int) -> None:
        if not isinstance(node, ScalarNode):
            raise ValueError("Model connection fields must be YAML scalars")
        # JSON strings are valid YAML strings, including keys with quotes or '#'.
        original = text[node.start_mark.index : node.end_mark.index]
        anchor = re.match(r"&[^\s]+\s+", original)
        prefix = anchor.group() if anchor else ""
        replacements[node.start_mark.index, node.end_mark.index] = prefix + json.dumps(value, ensure_ascii=False)

    def visit(node: Node, kind: str = "chat", parent: str = "") -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, SequenceNode):
            for child in node.value:
                visit(child, kind, parent)
        elif isinstance(node, MappingNode):
            fields = {key.value: value for key, value in node.value if isinstance(key, ScalarNode)}
            if "model" in fields and any(key in fields for key in ("api_key", "api_base", "base_url", "dimensions")):
                if "dimensions" in fields:
                    kind = "embedding"
                for field, value in settings[kind].items():
                    for yaml_field in ("base_url", "api_base") if field == "base_url" else (field,):
                        if yaml_field in fields:
                            replace(fields[yaml_field], value)
            dimensions = settings["embedding"].get("dimensions")
            if parent == "qdrant" and "vector_size" in fields and dimensions is not None:
                replace(fields["vector_size"], dimensions)
            for key, child in fields.items():
                child_kind = kind
                if key in ("embedding", "embed_model_router", "embedding_model_router"):
                    child_kind = "embedding"
                elif key in ("rerank", "rerank_model_router"):
                    child_kind = "rerank"
                elif key in ("chat", "chat_model_router"):
                    child_kind = "chat"
                visit(child, child_kind, key)

    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if root is not None:
        visit(root)
    for (start, end), value in sorted(replacements.items(), reverse=True):
        text = text[:start] + value + text[end:]
    return text


def _sync_model_settings(settings: dict[str, dict[str, str | int]]) -> None:
    """Validate all local YAML files before atomically replacing changed files."""
    updates = []
    for path in sorted((ROOT / "config").rglob("*.yaml")):
        if path.name.endswith(".example.yaml"):
            continue
        original = path.read_text(encoding="utf-8")
        try:
            updated = _replace_model_settings(original, settings)
            yaml.safe_load(updated)
        except (ValueError, yaml.YAMLError):
            # YAML exception messages can include credentials from the input.
            raise ValueError(f"Invalid model configuration in {path.relative_to(ROOT)}") from None
        if updated != original:
            updates.append((path, updated))
    for path, updated in updates:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                file.write(updated)
            temporary.chmod(0o600)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"model settings updated: {path.relative_to(ROOT)}")


def main() -> int:
    """Copy missing configurations and display required user settings.

    Returns:
        Zero on success, or one when copying, model validation, or updates fail.
    """
    sources = [ROOT / name for name in TEMPLATES]
    for directory in EVAL_DIRECTORIES:
        sources.extend(sorted((ROOT / directory).glob("*.example.yaml")))
    missing = [source for source in sources if not source.is_file()]
    if missing:
        for source in missing:
            print(f"Missing configuration template: {source}", file=sys.stderr)
        return 1
    try:
        settings = _read_model_settings()
    except (OSError, ValueError) as error:
        print(f"Invalid .env model settings: {error}", file=sys.stderr)
        return 1
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = source.with_name(source.name.replace(".example", ""))
        label = f"{relative} -> {destination.relative_to(ROOT)}"
        if destination.exists():
            print(f"file skipped: {label} (destination already exists)")
            continue
        created = False
        try:
            with source.open("rb") as original, destination.open("xb") as local:
                created = True
                shutil.copyfileobj(original, local)
            if destination.name == ".env" or destination.suffix == ".yaml":
                destination.chmod(0o600)
        except OSError as error:
            if created:
                destination.unlink(missing_ok=True)
            print(f"file copy failed: {label}: {error}", file=sys.stderr)
            return 1
        print(f"file copied: {label}")
    if any(settings.values()):
        try:
            _sync_model_settings(settings)
        except (OSError, ValueError) as error:
            print(f"Model configuration update failed: {error}", file=sys.stderr)
            return 1
    print(
        "\nNext, configure the key information in these files:\n"
        "Optional .env model defaults override corresponding fields in all local config YAML files.\n"
        "Fill unset model fields manually in config/.\n"
        "- config/mindmemos_eval/dev.yaml: target and optimizer model names, API keys, and base URLs.\n"
        "- config/mindmemos/dev.yaml and config/mindmemos_skill/dev.yaml: service model and storage connections.\n"
        "- config/presets/local.project.yaml: local project identity and credentials.\n"
        "- .env: required service settings and HF_TOKEN for gated Hugging Face datasets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
