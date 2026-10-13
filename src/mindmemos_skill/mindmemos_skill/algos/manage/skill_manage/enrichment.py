"""Optional source-grounded completion before final embedding text is constructed."""

import json
from time import perf_counter

from ....common.json import parse_model
from ....persistence.db_services.graphs.writer import graph_digest
from .constant import (
    METADATA_EXAMPLE_TASKS,
    METADATA_INTENT,
    METADATA_KEY,
    METADATA_TOOLING,
)
from .errors import SceneGraphError, SceneGraphFailure
from .extraction import SceneExtractor
from .typing import DerivedNodeMetadata, ExtractionSummary, SkillEnrichment

ENRICHMENT_SYSTEM = (
    "Fill missing Skill retrieval fields using only the supplied SKILL.md. "
    "Keep unsupported fields empty; do not invent tools, inputs, outputs, examples, or operations. "
    "Return only JSON matching the supplied schema. Never return source text, logic steps, "
    "Scene records, reasoning, or identity fields. Existing fields take precedence."
)
"""Packaged completion instructions, included by hash in enabled build identities."""


def enrichment_identity():
    """Identify the packaged completion contract without retaining model payloads."""
    return {
        "prompt_hash": graph_digest(ENRICHMENT_SYSTEM),
        "schema_hash": graph_digest(SkillEnrichment.model_json_schema()),
    }


class SkillEnricher(SceneExtractor):
    """Reuse the existing bounded parser retries and payload-free transport accounting."""

    def __init__(self, client):
        """Borrow the shared transport with a completion-specific repair instruction.

        Args:
            client: Configured chat transport used only for in-memory completion.
        """
        super().__init__(client)
        self.parse_feedback = lambda error: (
            "The response violates the supplied completion schema. Return corrected JSON "
            "with only the allowed fields, grounded in the original document."
        )

    def _parse(self, text):
        """Validate only completion fields and sanitize parser errors before retry handling."""
        try:
            return parse_model(text, SkillEnrichment)
        except (ValueError, TypeError):
            self.summary = self.summary.model_copy(update={"error_type": "invalid_output"})
        raise SceneGraphError(SceneGraphFailure.MODEL_FAILED)

    async def enrich(self, node, document):
        """Complete missing fields without changing the source object or node identity.

        Args:
            node: Rule/SSL-derived Skill node before embedding.
            document: Exact source SKILL.md already checked against the input limit.

        Returns:
            A new Skill node with allowlisted missing fields filled.

        Raises:
            SceneGraphError: The transport or bounded structural repair fails.
        """
        self.summary = ExtractionSummary()
        start = perf_counter()
        try:
            result = await self.run(
                system=ENRICHMENT_SYSTEM,
                user=json.dumps(
                    {"skill_md": document, "schema": SkillEnrichment.model_json_schema()}, ensure_ascii=False
                ),
                response_format={"type": "json_object"},
            )
            fields = result.output
            if not isinstance(fields, SkillEnrichment):
                raise SceneGraphError(SceneGraphFailure.MODEL_FAILED)
            metadata = DerivedNodeMetadata.model_validate(node.metadata)
            return node.model_copy(
                update={
                    "description": node.description or fields.description,
                    "one_line_capability": node.one_line_capability or fields.one_line_capability,
                    "inputs": node.inputs or _items(fields.inputs),
                    "outputs": node.outputs or _items(fields.outputs),
                    METADATA_TOOLING: node.tooling or _items(fields.tooling),
                    "domain_tags": node.domain_tags or _items(fields.domain_tags),
                    METADATA_EXAMPLE_TASKS: node.example_tasks or _items(fields.example_tasks),
                    METADATA_KEY: metadata.model_copy(
                        update={METADATA_INTENT: metadata.intent or fields.intent}
                    ).model_dump(mode="json"),
                }
            )
        finally:
            self.summary = self.summary.model_copy(update={"elapsed_seconds": perf_counter() - start})


def _items(values):
    """Retain nonempty normalized completion values in source order."""
    return list(dict.fromkeys(value for value in values if value))
