"""Normalize complete public tasks and render versioned retrieval queries."""

import json
import re
from collections.abc import Mapping

from ....envs.typing import Task
from .linking import signature_tokens
from .typing import QuerySchema


def public_task_text(task: Task) -> str:
    """Render the instruction and all public extension fields, excluding dataset identity.

    Args:
        task: Public task, including subclass fields; never private reward data.

    Returns:
        Deterministic retrieval text with any public extension payload.
    """
    extra = task.model_dump(mode="json", exclude={"id", "instruction"})
    return task.instruction + ("\n" + json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else "")


def normalize_query(query: str, inferred: QuerySchema | None = None) -> QuerySchema:
    """Merge validated rewrite fields with deterministic evidence from the full task.

    Args:
        query: Complete public task text.
        inferred: Validated model fields, or None when rewriting is disabled.

    Returns:
        Whitespace-normalized fields with source artifacts and keywords retained.
    """
    fallback = QuerySchema.model_validate(fallback_query_schema(query))
    if inferred is None:
        return fallback
    goal = " ".join(inferred.goal.split()) or fallback.goal
    task_name = " ".join(inferred.task_name.split()) or fallback.task_name
    domain = _dedupe_text(inferred.domain)
    operations = _dedupe_text(inferred.operations)
    artifacts = _dedupe_text(inferred.artifacts + fallback.artifacts)
    constraints = _dedupe_text(inferred.constraints)
    keywords = sorted(
        signature_tokens(
            [
                goal,
                task_name,
                *domain,
                *operations,
                *artifacts,
                *constraints,
                *inferred.keywords,
                *fallback.keywords,
            ]
        )
    )
    return QuerySchema(
        goal=goal,
        task_name=task_name,
        domain=domain,
        operations=operations,
        artifacts=artifacts,
        constraints=constraints,
        keywords=keywords,
    )


def query_fields(schema: QuerySchema) -> list[tuple[str, str]]:
    """Return ordered query fields for scoring and embedding.

    Args:
        schema: Normalized public query.

    Returns:
        Nonempty field names and values in template order.
    """
    fields = [
        ("GOAL", schema.goal),
        ("TASK_NAME", schema.task_name),
        ("DOMAIN", "\n".join(schema.domain)),
        ("OPERATIONS", "\n".join(schema.operations)),
        ("ARTIFACTS", "\n".join(schema.artifacts)),
        ("CONSTRAINTS", "\n".join(schema.constraints)),
        ("KEYWORDS", "\n".join(schema.keywords)),
    ]
    return [(name, value) for name, value in fields if value]


def query_text(schema: QuerySchema, template: str) -> str:
    """Render the declared query template without changing stored node vectors.

    Args:
        schema: Normalized public query.
        template: Supported embedding-profile query template identifier.

    Returns:
        Untagged legacy or tagged joint query text.

    Raises:
        ValueError: The requested template is unsupported.
    """
    fields = query_fields(schema)
    if template == "legacy_query_v1":
        return "\n".join(value for _, value in fields)
    if template == "scene_graph_query_v2":
        return "\n".join(f"[{name}] {value}" for name, value in fields)
    raise ValueError("Unsupported query embedding template")


def extract_task_name(query: str) -> str:
    """Derive the source-compatible short task slug.

    Args:
        query: Public task retrieval text.

    Returns:
        At most eight normalized tokens and 80 characters, or an empty slug.
    """
    tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", query.strip()) if token]
    if not tokens:
        return ""
    slug = "-".join(token.lower() for token in tokens[:8])
    return slug[:80]


def extract_artifacts(query: str) -> list[str]:
    """Find explicit artifact filenames in the public task.

    Args:
        query: Public task retrieval text.

    Returns:
        Explicit artifact paths in first-occurrence order.
    """
    artifacts = re.findall(r"[A-Za-z0-9_./-]+\.(?:py|md|json|csv|stl|dot|txt|yaml|yml|bib|pptx|xlsx|docx)", query)
    return _dedupe_text([artifact.strip() for artifact in artifacts if artifact.strip()])


def fallback_query_schema(query: str) -> dict[str, list[str] | str]:
    """Derive deterministic query fields when rewriting is disabled.

    Args:
        query: Public task retrieval text.

    Returns:
        Rule-derived query fields with no inferred domain or constraints.
    """
    normalized_query = re.sub(r"\s+", " ", query.strip())
    artifacts = extract_artifacts(normalized_query)
    keywords = sorted(signature_tokens([normalized_query, *artifacts]))
    return {
        "goal": normalized_query,
        "task_name": extract_task_name(normalized_query),
        "domain": [],
        "operations": [],
        "artifacts": artifacts,
        "constraints": [],
        "keywords": list(keywords),
    }


def query_schema_values(query_schema: Mapping[str, object]) -> list[str]:
    """Flatten public query fields in source order.

    Args:
        query_schema: Public goal, operations, constraints, and search terms.

    Returns:
        Nonempty serialized legacy field values in canonical order.
    """
    values: list[str] = []
    for key in ("goal", "task_name", "domain", "operations", "artifacts", "constraints", "keywords"):
        raw = query_schema.get(key, [])
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, list):
            values.extend(str(item) for item in raw)
    return [value for value in values if value]


def _dedupe_text(values: list[str]) -> list[str]:
    """Normalize whitespace and retain the first nonblank case-insensitive value."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value or "").strip())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def query_values(schema: QuerySchema) -> list[str]:
    """Flatten typed query fields for reference-compatible token scoring.

    Args:
        schema: Normalized public query.

    Returns:
        Individual field values in canonical order, without artificial joined-list tokens.
    """
    return [
        value
        for value in [
            schema.goal,
            schema.task_name,
            *schema.domain,
            *schema.operations,
            *schema.artifacts,
            *schema.constraints,
            *schema.keywords,
        ]
        if value
    ]
