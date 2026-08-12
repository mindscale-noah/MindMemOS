"""Regression tests: verify schema_general.json guides LLM to classify
technical artifacts (software, tools, frameworks) as "item".
The schema uses context-based guidance (not absolute "never" rules):
a term like Docker may be "item" when discussed as a tool, or
"organization" when discussed as a company.

Structural tests (no model needed) always run.
Live model tests require LLM_API_KEY env var or config/mindmemos/dev.yaml.
Skip live tests with: pytest -m "not llm"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Skip guard for live model tests (same pattern as test_real_llm_integration)
# ---------------------------------------------------------------------------

_DEV_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "mindmemos" / "dev.yaml"


def _dev_chat_endpoint() -> dict:
    if not _DEV_CONFIG_PATH.exists():
        return {}
    with _DEV_CONFIG_PATH.open() as fh:
        data = yaml.safe_load(fh) or {}
    endpoints = (data.get("chat_model_router") or {}).get("endpoints") or []
    if not endpoints:
        return {}
    return endpoints[0] or {}


_DEV_CHAT_ENDPOINT = _dev_chat_endpoint()
_LLM_API_KEY = os.environ.get("LLM_API_KEY") or _DEV_CHAT_ENDPOINT.get("api_key")
_LLM_API_BASE = os.environ.get("LLM_API_BASE") or _DEV_CHAT_ENDPOINT.get("api_base") or "https://api.openai.com/v1"
_LLM_MODEL = os.environ.get("LLM_MODEL") or _DEV_CHAT_ENDPOINT.get("model") or "gpt-4o-mini"
_LLM_EXTRA_BODY = (
    _DEV_CHAT_ENDPOINT.get("extra_body") if isinstance(_DEV_CHAT_ENDPOINT.get("extra_body"), dict) else None
)

skip_no_llm_key = pytest.mark.skipif(
    not _LLM_API_KEY,
    reason="No LLM key in LLM_API_KEY or config/mindmemos/dev.yaml chat_model_router.endpoints[0]",
)


# ---------------------------------------------------------------------------
# Test dialogues + expected minimum entity classification
# ---------------------------------------------------------------------------
# Each dialogue is (timestamp, text, expected_entities) where
# expected_entities maps entity_name_substring → expected entity_type.
# The model may produce longer names ("Apache Kafka") — we match by
# case-insensitive substring so "Kafka" matches "Apache Kafka".

DIALOGUE_1 = (
    "2024-08-10 10:00:00",
    (
        "Alice: We're evaluating message queues for the new platform. "
        "I think Kafka is the right choice for our event streaming needs.\n"
        "Bob: Yeah, Kafka is solid. But we also need to containerize everything "
        "with Docker. And for package management, we're standardizing on npm.\n"
        "Alice: What about the API gateway? I was looking at Kong or maybe just "
        "using Nginx with some Lua scripting.\n"
        "Bob: For orchestration, Kubernetes is the obvious pick. And we should "
        "use Prometheus for monitoring and Grafana for dashboards.\n"
        "Alice: Makes sense. I'll set up a proof of concept with Kafka + Docker "
        "this week."
    ),
    # All discussed as tools/products → expected type "item"
    {
        "kafka": "item",
        "docker": "item",
        "npm": "item",
        "nginx": "item",
        "kubernetes": "item",
        "prometheus": "item",
        "grafana": "item",
    },
)

DIALOGUE_2 = (
    "2024-08-10 11:00:00",
    (
        "Carol: For the backend, I'm leaning towards PostgreSQL with the "
        "pgvector extension for the vector search part.\n"
        "Dave: PostgreSQL is great. But have you considered using Qdrant "
        "as a dedicated vector database? It's purpose-built for that.\n"
        "Carol: We could. The Python service will use FastAPI with Pydantic "
        "for validation. And for the frontend, we're going with Next.js.\n"
        "Dave: Make sure to use Redis for caching and set up proper "
        "CI/CD with GitHub Actions.\n"
        "Carol: Good call. I'll add Redis to the architecture doc."
    ),
    # All discussed as tools/products → expected type "item"
    {
        "postgresql": "item",
        "qdrant": "item",
        "fastapi": "item",
        "pydantic": "item",
        "next.js": "item",
        "redis": "item",
        "github actions": "item",
    },
)

ALL_DIALOGUES: list[tuple[str, str, dict[str, str]]] = [DIALOGUE_1, DIALOGUE_2]


def _normalize(name: str) -> str:
    """Normalize entity name for matching: lowercase, strip whitespace."""
    return name.strip().lower()


def _entity_matches(normalized_entity: str, expected_key: str) -> bool:
    """Check whether a model-returned entity name matches the expected key.

    Handles variants like "Apache Kafka" → "kafka", "Docker Engine" → "docker".
    """
    return expected_key in normalized_entity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_schema() -> list[dict[str, Any]]:
    """Load schema_general.json from the presets directory."""
    schema_path = Path(__file__).resolve().parents[2] / "config" / "presets" / "schema_general.json"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def find_item_description(schema: list[dict[str, Any]]) -> str:
    for entry in schema:
        if entry.get("entity_type") == "item":
            return entry.get("entity_description", "")
    return ""


def find_item_instruction(schema: list[dict[str, Any]]) -> str:
    for entry in schema:
        if entry.get("entity_type") == "item":
            return entry.get("entity_instruction", "")
    return ""


# ---------------------------------------------------------------------------
# Structural tests (no model call)
# ---------------------------------------------------------------------------

class TestSchemaGeneralStructure:
    """Verify schema_general.json contains correct technical artifact guidance."""

    def test_schema_file_exists(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2] / "config" / "presets" / "schema_general.json"
        )
        assert schema_path.exists(), f"schema_general.json not found at {schema_path}"

    def test_item_description_covers_technical_artifacts(self) -> None:
        desc = find_item_description(load_schema())
        assert "technical artifacts" in desc.lower(), (
            f"item entity_description must mention technical artifacts, got: {desc}"
        )
        assert "software" in desc.lower(), (
            f"item entity_description must mention software, got: {desc}"
        )

    def test_item_instruction_guides_context_based_choice(self) -> None:
        """Instruction should guide context-based choice, not absolute 'never'."""
        instr = find_item_instruction(load_schema())
        assert "context" in instr.lower(), (
            f"item entity_instruction must mention context-based judgment, got: {instr}"
        )
        assert "organization" in instr.lower(), (
            f"item entity_instruction must mention organization for disambiguation, got: {instr}"
        )

    def test_schema_is_valid_json_array(self) -> None:
        schema = load_schema()
        assert isinstance(schema, list), "schema_general.json must be a JSON array"
        assert len(schema) > 0, "schema_general.json must not be empty"

    def test_all_entity_types_have_description(self) -> None:
        schema = load_schema()
        for entry in schema:
            assert entry.get("entity_type"), f"Missing entity_type in entry: {entry}"
            assert entry.get("entity_description"), (
                f"Missing entity_description for {entry['entity_type']}"
            )


# ---------------------------------------------------------------------------
# Prompt injection test (no model call)
# ---------------------------------------------------------------------------

class TestSchemaPromptIntegration:
    """Verify the schema is correctly injected into entity generation prompts."""

    def test_schema_injects_into_prompt(self) -> None:
        """Simulate what extract_memory does: inject schema into the prompt."""
        from mindmemos.components.extractor.schema._schema_utils import (
            format_schema_summary,
            strip_for_generation,
        )
        from mindmemos.prompts import get_add_prompts

        schema = load_schema()
        stripped = strip_for_generation(schema)
        summary = format_schema_summary(stripped)

        assert "technical artifacts" in summary.lower() or "software" in summary.lower(), (
            f"formatted schema must contain tech artifact guidance\nGot:\n{summary[:2000]}"
        )

        prompts = get_add_prompts("EN")
        prompt = (
            prompts.entity_generation.replace("{entity_schema}", str(stripped))
            .replace("{dialogue_timestamp}", "2024-08-10")
            .replace("{chat_chunk}", "test dialogue")
        )
        assert "technical artifacts" in prompt.lower() or "software" in prompt.lower(), (
            "Final prompt must include technical artifact guidance from schema"
        )

    def test_zh_prompt_receives_same_schema(self) -> None:
        """Verify ZH prompt also receives the schema with tech artifact guidance."""
        from mindmemos.components.extractor.schema._schema_utils import strip_for_generation
        from mindmemos.prompts import get_add_prompts

        schema = load_schema()
        stripped = strip_for_generation(schema)
        prompts = get_add_prompts("ZH")
        prompt = (
            prompts.entity_generation.replace("{entity_schema}", str(stripped))
            .replace("{dialogue_timestamp}", "2024-08-10")
            .replace("{chat_chunk}", "test dialogue")
        )
        assert "technical artifacts" in prompt.lower() or "software" in prompt.lower(), (
            "ZH prompt must include technical artifact guidance from schema"
        )


# ---------------------------------------------------------------------------
# Live model regression tests (require LLM_API_KEY or dev.yaml)
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestLiveModelEntityClassification:
    """Call the live model via litellm to verify technical artifacts → 'item'.

    Requires LLM_API_KEY env var or config/mindmemos/dev.yaml with endpoints.
    Skip with: pytest -m "not llm"
    """

    @classmethod
    def _build_prompt(cls, dialogue_timestamp: str, chat_chunk: str) -> str:
        from mindmemos.components.extractor.schema._schema_utils import strip_for_generation
        from mindmemos.prompts import get_add_prompts

        schema = strip_for_generation(load_schema())
        prompts = get_add_prompts("EN")
        return (
            prompts.entity_generation.replace("{entity_schema}", str(schema))
            .replace("{dialogue_timestamp}", dialogue_timestamp)
            .replace("{chat_chunk}", chat_chunk)
        )

    @staticmethod
    def _parse_entities(raw: dict) -> dict[str, list[str]]:
        """Group entity names by entity_type."""
        by_type: dict[str, list[str]] = {}
        for entity in raw.get("entities", []):
            entity_type = entity.get("entity_type", "unknown")
            name = entity.get("name", "?")
            by_type.setdefault(entity_type, []).append(name)
        return by_type

    def _verify_expected(
        self,
        by_type: dict[str, list[str]],
        expected: dict[str, str],
        label: str,
    ) -> None:
        """Verify every expected entity is found under its expected type.

        Uses substring matching: "Apache Kafka" matches expected key "kafka".
        Reports ALL mismatches at once rather than failing on the first.
        """
        mismatches: list[str] = []
        not_found: list[str] = []

        for expected_key, expected_type in expected.items():
            # Search across all types for this entity
            found_type: str | None = None
            for entity_type, names in by_type.items():
                for name in names:
                    if _entity_matches(_normalize(name), expected_key):
                        found_type = entity_type
                        break
                if found_type is not None:
                    break

            if found_type is None:
                not_found.append(expected_key)
            elif found_type != expected_type:
                mismatches.append(
                    f"'{expected_key}' expected '{expected_type}', got '{found_type}'"
                )

        errors: list[str] = []
        if not_found:
            errors.append(
                f"[{label}] Entities not extracted at all: {not_found}"
            )
        if mismatches:
            errors.append(
                f"[{label}] Classification mismatches: {mismatches}"
            )

        assert not errors, (
            "\n".join(errors) + f"\nAll entities by type: {by_type}"
        )

    @skip_no_llm_key
    @pytest.mark.asyncio
    async def test_infra_terms_classified_correctly(self) -> None:
        """Kafka, Docker, npm, etc. → 'item' (dialogue context = using tools)."""
        import litellm
        from mindmemos.components.extractor.schema._schema_utils import parse_json_object

        ts, dialogue, expected = DIALOGUE_1
        prompt = self._build_prompt(ts, dialogue)

        resp = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            api_key=_LLM_API_KEY,
            api_base=_LLM_API_BASE,
            extra_body=_LLM_EXTRA_BODY,
            drop_params=True,
        )
        content = resp.choices[0].message.content or ""
        raw = parse_json_object(content)

        assert isinstance(raw, dict), f"Expected dict, got {type(raw)}"
        by_type = self._parse_entities(raw)
        print(f"\n[Infra dialogue] entities by type: {by_type}")

        self._verify_expected(by_type, expected, "infra")

    @skip_no_llm_key
    @pytest.mark.asyncio
    async def test_database_terms_classified_correctly(self) -> None:
        """PostgreSQL, Qdrant, FastAPI, etc. → 'item' (dialogue context = using tools)."""
        import litellm
        from mindmemos.components.extractor.schema._schema_utils import parse_json_object

        ts, dialogue, expected = DIALOGUE_2
        prompt = self._build_prompt(ts, dialogue)

        resp = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            api_key=_LLM_API_KEY,
            api_base=_LLM_API_BASE,
            extra_body=_LLM_EXTRA_BODY,
            drop_params=True,
        )
        content = resp.choices[0].message.content or ""
        raw = parse_json_object(content)

        assert isinstance(raw, dict), f"Expected dict, got {type(raw)}"
        by_type = self._parse_entities(raw)
        print(f"\n[Database dialogue] entities by type: {by_type}")

        self._verify_expected(by_type, expected, "database")
