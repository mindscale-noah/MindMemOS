"""Embedding template, content hashing and model port of ``skill_manage``.

Covers ``algos/skill_manage/embedding.py`` directly and the way
``DefaultSkillManageAlgorithm`` threads it through ``plan_build``,
``plan_upsert``, ``plan_remove`` and ``embed``.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from mindmemos_skill.algos.skill_manage import embedding as embedding_module
from mindmemos_skill.algos.skill_manage.algorithm import DefaultSkillManageAlgorithm
from mindmemos_skill.algos.skill_manage.config import SkillManageConfig
from mindmemos_skill.algos.skill_manage.embedding import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_TEMPLATE_VERSION,
    content_hash,
    embed_texts,
    embedding_text,
    require_template,
    skill_embedding_texts,
    with_content_hash,
)
from mindmemos_skill.algos.skill_manage.errors import SkillManageError
from mindmemos_skill.errors import EmbeddingDimensionError
from mindmemos_skill.typing import (
    EdgeType,
    EmbeddingProfile,
    SkillDocumentInput,
    SkillGraphEdge,
    SkillGraphEmbeddingText,
    SkillGraphMembership,
    SkillGraphNode,
    SkillGraphSnapshot,
    SkillManageBuildInput,
    SkillManageEmbedInput,
    SkillManageRemoveInput,
    SkillManageRemoveSelector,
    SkillManageScope,
    SkillManageUpsertInput,
    SkillVersionRef,
)

_TESTS_MINDMEMOS = Path(__file__).parent
if str(_TESTS_MINDMEMOS) not in sys.path:
    sys.path.insert(0, str(_TESTS_MINDMEMOS))

from skill_manage_embedding_stub import stub_algorithm_context  # noqa: E402

FIXTURE_SKILLS = Path(__file__).parent / "fixtures" / "skill_manage" / "gos_minimal" / "skills"

FULL_NODE = SkillGraphNode(
    graph_node_id="skill:csv-parser",
    node_type="skill",
    skill_id="csv-parser",
    version_id="v1",
    name="CSV Parser",
    description="Parses raw CSV text into records.",
    one_line_capability="Parse CSV into records",
    inputs=("raw_csv_text", "delimiter"),
    outputs=("parsed_records",),
    domain_tags=("data-processing", "text"),
    tooling=("pandas", "csv"),
    example_tasks=("parse a csv export", "split a csv into columns"),
    rendered_snippet="# CSV Parser\n\nParses raw CSV text into records.",
)

FULL_TEXT = (
    "[Skill] CSV Parser\n"
    "[DESCRIPTION] Parses raw CSV text into records.\n"
    "[CAPABILITY] Parse CSV into records\n"
    "[INPUTS] raw_csv_text\ndelimiter\n"
    "[OUTPUTS] parsed_records\n"
    "[DOMAIN_TAGS] data-processing\ntext\n"
    "[TOOLING] pandas\ncsv\n"
    "[EXAMPLE_TASKS] parse a csv export\nsplit a csv into columns\n"
    "[SNIPPET] # CSV Parser\n\nParses raw CSV text into records."
)

MINIMAL_NODE = SkillGraphNode(
    graph_node_id="skill:minimal",
    node_type="skill",
    skill_id="minimal",
    version_id="v1",
    name="Minimal",
    description="Does one thing.",
)

SCENE_NODE = SkillGraphNode(
    graph_node_id="scene:pipeline",
    node_type="scene",
    skill_id="csv-parser",
    parent_skill_id="csv-parser",
    name="Data pipeline",
    description="Parse then filter.",
)


def _profile(
    dimension: int = 8, template_version: str = EMBEDDING_TEMPLATE_VERSION, normalization: str = "nfc"
) -> EmbeddingProfile:
    return EmbeddingProfile(
        model="m", dimension=dimension, template_version=template_version, normalization=normalization
    )


def _scope() -> SkillManageScope:
    return SkillManageScope(project_id="p", graph_id="g")


def _algorithm(context: object = None) -> DefaultSkillManageAlgorithm:
    return DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=object() if context is None else context)


def _fixture_document(skill_id: str, file_name: str, version_id: str = "v1") -> SkillDocumentInput:
    return SkillDocumentInput(
        skill_id=skill_id,
        version_id=version_id,
        content=(FIXTURE_SKILLS / file_name).read_text(encoding="utf-8"),
        source_path=f"skills/{file_name}",
    )


def _build_input(*documents: SkillDocumentInput, profile: EmbeddingProfile | None = None) -> SkillManageBuildInput:
    return SkillManageBuildInput(
        scope=_scope(),
        version_refs=tuple(SkillVersionRef(skill_id=d.skill_id, version_id=d.version_id) for d in documents),
        documents=documents,
        embedding_profile=profile or _profile(),
    )


class _FakeModel:
    """Minimal ``EmbedClient`` stand-in; ``respond`` maps one batch to raw vectors."""

    def __init__(self, respond: Callable[[list[str]], list[list[float]]]) -> None:
        self._respond = respond
        self.calls: list[dict[str, Any]] = []

    async def embed(
        self,
        task: str,
        text: str | list[str],
        *,
        model: str | None = None,
        expected_dim: int | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        texts = [text] if isinstance(text, str) else list(text)
        self.calls.append({"task": task, "texts": texts, "model": model, "expected_dim": expected_dim})
        return SimpleNamespace(embeddings=self._respond(texts))


def _index_vectors(texts: list[str]) -> list[list[float]]:
    """Two-dimensional vectors encoding the text's global index (``t<n>``) for order checks."""

    return [[float(t[1:]), 1.0] for t in texts]


# --- embedding_text -----------------------------------------------------------


def test_embedding_text_full_node_matches_gos_template() -> None:
    assert embedding_text(FULL_NODE) == FULL_TEXT


def test_embedding_text_tag_order_follows_gos_to_str() -> None:
    tags = re.findall(r"^\[([A-Za-z_]+)\]", embedding_text(FULL_NODE), flags=re.MULTILINE)
    assert tags == [
        "Skill",
        "DESCRIPTION",
        "CAPABILITY",
        "INPUTS",
        "OUTPUTS",
        "DOMAIN_TAGS",
        "TOOLING",
        "EXAMPLE_TASKS",
        "SNIPPET",
    ]


def test_embedding_text_minimal_node_skips_empty_fields() -> None:
    assert embedding_text(MINIMAL_NODE) == "[Skill] Minimal\n[DESCRIPTION] Does one thing."


def test_embedding_text_name_only_node_is_a_single_line() -> None:
    node = SkillGraphNode(graph_node_id="skill:bare", node_type="skill", name="Bare")
    assert embedding_text(node) == "[Skill] Bare"


def test_embedding_text_lists_are_newline_joined_and_blank_items_dropped() -> None:
    node = MINIMAL_NODE.model_copy(update={"inputs": ("  a ", "", "   ", "b")})
    assert embedding_text(node) == "[Skill] Minimal\n[DESCRIPTION] Does one thing.\n[INPUTS] a\nb"


def test_embedding_text_scene_node_uses_scene_header() -> None:
    assert embedding_text(SCENE_NODE) == "[Scene] Data pipeline\n[DESCRIPTION] Parse then filter."


# --- content_hash / with_content_hash / skill_embedding_texts ------------------


@pytest.mark.parametrize("text", ["hello", "", "héllo wörld\n[SNIPPET] 日本語"])
def test_content_hash_is_sha256_hex_of_utf8(text: str) -> None:
    digest = content_hash(text)
    assert digest == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert len(digest) == 64
    assert digest == digest.lower()


def test_with_content_hash_stamps_skill_nodes_only() -> None:
    stamped = with_content_hash((FULL_NODE, SCENE_NODE, MINIMAL_NODE))

    assert len(stamped) == 3
    assert stamped[0].content_hash == content_hash(FULL_TEXT)
    assert stamped[2].content_hash == content_hash(embedding_text(MINIMAL_NODE))
    assert stamped[1] is SCENE_NODE
    assert stamped[1].content_hash is None
    for original, node in zip((FULL_NODE, SCENE_NODE, MINIMAL_NODE), stamped):
        assert node.model_dump(exclude={"content_hash"}) == original.model_dump(exclude={"content_hash"})


def test_with_content_hash_replaces_a_stale_hash() -> None:
    stale = FULL_NODE.model_copy(update={"content_hash": "stale"})
    (fresh,) = with_content_hash((stale,))
    assert fresh.content_hash == content_hash(FULL_TEXT)


def test_skill_embedding_texts_one_entry_per_skill_node() -> None:
    texts = skill_embedding_texts((FULL_NODE, SCENE_NODE, MINIMAL_NODE))

    assert [entry.graph_node_id for entry in texts] == ["skill:csv-parser", "skill:minimal"]
    for entry, node in zip(texts, (FULL_NODE, MINIMAL_NODE)):
        assert isinstance(entry, SkillGraphEmbeddingText)
        assert entry.text == embedding_text(node)
        assert entry.content_hash == content_hash(embedding_text(node))


def test_skill_embedding_texts_reuses_stamped_hash() -> None:
    stamped = FULL_NODE.model_copy(update={"content_hash": "pre-stamped"})
    (entry,) = skill_embedding_texts((stamped,))
    assert entry.content_hash == "pre-stamped"
    assert entry.text == FULL_TEXT


def test_skill_embedding_texts_is_empty_without_skill_nodes() -> None:
    assert skill_embedding_texts((SCENE_NODE,)) == ()
    assert skill_embedding_texts(()) == ()


# --- require_template ----------------------------------------------------------


def test_require_template_accepts_current_version() -> None:
    assert EMBEDDING_TEMPLATE_VERSION == "1"
    require_template(_profile(template_version="1"))
    require_template(EmbeddingProfile(model="m", dimension=3))  # default template_version


def test_require_template_rejects_other_version() -> None:
    with pytest.raises(SkillManageError) as exc_info:
        require_template(_profile(template_version="2"))
    error = exc_info.value
    assert error.error_code == "SKILL_MANAGE_EMBEDDING_PROFILE_MISMATCH"
    assert error.retryable is False
    assert error.details == {
        "expected": {"template_version": "1", "normalization": "nfc"},
        "actual": {"template_version": "2", "normalization": "nfc"},
    }


def test_require_template_rejects_other_normalization() -> None:
    with pytest.raises(SkillManageError) as exc_info:
        require_template(_profile(normalization="nfkc"))
    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_PROFILE_MISMATCH"
    assert exc_info.value.details["actual"]["normalization"] == "nfkc"


# --- embed_texts ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_texts_returns_float_tuples_in_input_order() -> None:
    model = _FakeModel(lambda texts: [[index, len(text)] for index, text in enumerate(texts)])

    vectors = await embed_texts(model, _profile(dimension=2), ["ab", "cde", "f"], task="skill_manage.build")

    assert vectors == ((0.0, 2.0), (1.0, 3.0), (2.0, 1.0))
    assert isinstance(vectors, tuple)
    assert all(isinstance(vector, tuple) for vector in vectors)
    assert all(isinstance(value, float) for vector in vectors for value in vector)
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_embed_texts_batches_at_most_64_texts_per_call() -> None:
    model = _FakeModel(_index_vectors)
    texts = [f"t{index}" for index in range(EMBEDDING_BATCH_SIZE * 2 + 2)]

    vectors = await embed_texts(model, _profile(dimension=2), texts, task="skill_manage.build")

    assert EMBEDDING_BATCH_SIZE == 64
    assert [len(call["texts"]) for call in model.calls] == [64, 64, 2]
    assert [call["texts"][0] for call in model.calls] == ["t0", "t64", "t128"]
    assert len(vectors) == len(texts)
    assert [vector[0] for vector in vectors] == [float(index) for index in range(len(texts))]


@pytest.mark.asyncio
async def test_embed_texts_empty_input_skips_the_model() -> None:
    model = _FakeModel(_index_vectors)
    assert await embed_texts(model, _profile(dimension=2), [], task="skill_manage.build") == ()
    assert model.calls == []


@pytest.mark.asyncio
async def test_embed_texts_count_mismatch() -> None:
    model = _FakeModel(lambda texts: _index_vectors(texts)[:-1])

    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(model, _profile(dimension=2), ["t0", "t1", "t2"], task="skill_manage.build")

    error = exc_info.value
    assert error.error_code == "SKILL_MANAGE_EMBEDDING_COUNT_MISMATCH"
    assert error.retryable is True
    assert error.details == {"expected": 3, "actual": 2, "model": "m"}


@pytest.mark.asyncio
async def test_embed_texts_missing_embeddings_attribute_is_count_mismatch() -> None:
    class _NoEmbeddings:
        async def embed(self, task: str, text: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace()

    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(_NoEmbeddings(), _profile(dimension=2), ["t0"], task="skill_manage.build")
    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_COUNT_MISMATCH"
    assert exc_info.value.details == {"expected": 1, "actual": 0, "model": "m"}


@pytest.mark.asyncio
async def test_embed_texts_wrong_vector_length_is_dimension_mismatch() -> None:
    model = _FakeModel(lambda texts: [[1.0, 2.0], [1.0, 2.0, 3.0]])

    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(model, _profile(dimension=2), ["t0", "t1"], task="skill_manage.build")

    error = exc_info.value
    assert error.error_code == "SKILL_MANAGE_EMBEDDING_DIMENSION_MISMATCH"
    assert error.retryable is False
    assert error.details == {"expected": 2, "actual": 3, "model": "m"}


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.asyncio
async def test_embed_texts_non_finite_value(bad: float) -> None:
    model = _FakeModel(lambda texts: [[0.0, 1.0], [bad, 1.0]])

    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(model, _profile(dimension=2), ["t0", "t1"], task="skill_manage.build")

    error = exc_info.value
    assert error.error_code == "SKILL_MANAGE_EMBEDDING_NON_FINITE"
    assert error.retryable is True
    assert error.details == {"index": 1, "model": "m"}


@pytest.mark.asyncio
async def test_embed_texts_non_finite_index_is_global_across_batches() -> None:
    def respond(texts: list[str]) -> list[list[float]]:
        return [[math.nan, 1.0] if text == "t64" else [0.0, 1.0] for text in texts]

    texts = [f"t{index}" for index in range(EMBEDDING_BATCH_SIZE + 1)]
    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(_FakeModel(respond), _profile(dimension=2), texts, task="skill_manage.build")
    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_NON_FINITE"
    assert exc_info.value.details["index"] == 64


@pytest.mark.asyncio
async def test_embed_texts_wraps_provider_exception_as_retryable() -> None:
    def respond(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("boom")

    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(_FakeModel(respond), _profile(dimension=2), ["t0"], task="skill_manage.query")

    error = exc_info.value
    assert error.error_code == "SKILL_MANAGE_EMBEDDING_PROVIDER_ERROR"
    assert error.retryable is True
    # Only the exception type is recorded: provider messages can carry endpoint
    # URLs, keys or response bodies and the envelope is persisted and returned.
    assert error.details == {"model": "m", "task": "skill_manage.query", "reason": "RuntimeError"}
    assert "boom" not in error.message
    assert isinstance(error.__cause__, RuntimeError)
    assert str(error.__cause__) == "boom"


@pytest.mark.asyncio
async def test_embed_texts_maps_client_dimension_error() -> None:
    def respond(texts: list[str]) -> list[list[float]]:
        raise EmbeddingDimensionError(expected=2, actual=5, model="m", task="skill_manage.build")

    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(_FakeModel(respond), _profile(dimension=2), ["t0"], task="skill_manage.build")

    error = exc_info.value
    assert error.error_code == "SKILL_MANAGE_EMBEDDING_DIMENSION_MISMATCH"
    assert error.retryable is False
    assert error.details == {"expected": 2, "actual": 5, "model": "m"}
    assert isinstance(error.__cause__, EmbeddingDimensionError)


@pytest.mark.asyncio
async def test_embed_texts_passes_domain_errors_through_unchanged() -> None:
    domain_error = SkillManageError("SKILL_MANAGE_CUSTOM", "already typed")

    def respond(texts: list[str]) -> list[list[float]]:
        raise domain_error

    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(_FakeModel(respond), _profile(dimension=2), ["t0"], task="skill_manage.build")
    assert exc_info.value is domain_error


@pytest.mark.asyncio
async def test_embed_texts_forwards_profile_model_and_dimension_to_the_client() -> None:
    model = _FakeModel(lambda texts: [[0.0] * 5 for _ in texts])
    profile = EmbeddingProfile(model="text-embedding-3-large", dimension=5)

    await embed_texts(model, profile, ["t0", "t1"], task="skill_manage.query")

    assert model.calls == [
        {
            "task": "skill_manage.query",
            "texts": ["t0", "t1"],
            "model": "text-embedding-3-large",
            "expected_dim": 5,
        }
    ]


@pytest.mark.asyncio
async def test_embed_texts_rejects_foreign_template_before_calling_the_model() -> None:
    model = _FakeModel(_index_vectors)
    with pytest.raises(SkillManageError) as exc_info:
        await embed_texts(model, _profile(dimension=2, template_version="2"), ["t0"], task="skill_manage.build")
    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_PROFILE_MISMATCH"
    assert model.calls == []


# --- DefaultSkillManageAlgorithm -------------------------------------------------


@pytest.mark.asyncio
async def test_algorithm_embed_without_bound_model_is_configuration_error() -> None:
    algorithm = _algorithm(object())
    with pytest.raises(SkillManageError) as exc_info:
        await algorithm.embed(SkillManageEmbedInput(embedding_profile=_profile(), texts=("alpha",)))
    error = exc_info.value
    assert error.error_code == "SKILL_MANAGE_CONFIGURATION_ERROR"
    assert error.retryable is False
    assert "embedding" in error.details["reason"]


@pytest.mark.asyncio
async def test_algorithm_embed_uses_the_bound_embedding_model() -> None:
    context = stub_algorithm_context(8)
    algorithm = _algorithm(context)

    output = await algorithm.embed(
        SkillManageEmbedInput(embedding_profile=_profile(dimension=8), texts=("alpha", "beta"))
    )

    assert output.embedding_profile == _profile(dimension=8)
    assert len(output.vectors) == 2
    assert all(len(vector) == 8 for vector in output.vectors)
    assert all(isinstance(value, float) for vector in output.vectors for value in vector)
    assert context.embedding_models["embedding"].calls == [
        {"task": "skill_manage.build", "count": 2, "model": "m", "expected_dim": 8}
    ]


@pytest.mark.asyncio
async def test_algorithm_embed_surfaces_model_dimension_mismatch() -> None:
    algorithm = _algorithm(stub_algorithm_context(8))
    with pytest.raises(SkillManageError) as exc_info:
        await algorithm.embed(
            SkillManageEmbedInput(embedding_profile=_profile(dimension=16), texts=("alpha",), task="skill_manage.query")
        )
    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_DIMENSION_MISMATCH"
    assert exc_info.value.details == {"expected": 16, "actual": 8, "model": "m"}


@pytest.mark.asyncio
async def test_plan_build_stamps_hashes_and_aligns_embedding_texts() -> None:
    algorithm = _algorithm()
    plan = await algorithm.plan_build(
        _build_input(
            _fixture_document("csv-parser", "csv_parser.md"),
            _fixture_document("record-filter", "record_filter.md"),
        )
    )

    assert [node.graph_node_id for node in plan.nodes] == ["skill:csv-parser", "skill:record-filter"]
    assert all(node.content_hash for node in plan.nodes)
    assert [entry.graph_node_id for entry in plan.embedding_texts] == [node.graph_node_id for node in plan.nodes]
    assert [entry.content_hash for entry in plan.embedding_texts] == [node.content_hash for node in plan.nodes]
    for entry, node in zip(plan.embedding_texts, plan.nodes):
        assert entry.text == embedding_text(node)
        assert entry.content_hash == content_hash(entry.text)
    assert plan.embedding_texts[0].text.startswith("[Skill] CSV Parser Skill\n[DESCRIPTION] Parses raw CSV text")
    assert "[INPUTS] raw_csv_text\n[OUTPUTS] parsed_records\n" in plan.embedding_texts[0].text
    # The fixture dependency (parser -> filter) still links after hashing.
    assert [(edge.source_id, edge.target_id) for edge in plan.edges] == [("skill:csv-parser", "skill:record-filter")]


@pytest.mark.asyncio
async def test_plan_build_rejects_foreign_template_before_parsing() -> None:
    algorithm = _algorithm()
    unparseable = SkillDocumentInput(skill_id="broken", version_id="v1", content="no frontmatter here")

    with pytest.raises(SkillManageError) as exc_info:
        await algorithm.plan_build(_build_input(unparseable, profile=_profile(template_version="2")))
    assert exc_info.value.error_code == "SKILL_MANAGE_EMBEDDING_PROFILE_MISMATCH"

    with pytest.raises(SkillManageError) as exc_info:
        await algorithm.plan_build(_build_input(unparseable))
    assert exc_info.value.error_code == "SKILL_MANAGE_INVALID_SKILL_DOCUMENT"


@pytest.mark.asyncio
async def test_plan_build_is_deterministic() -> None:
    request = _build_input(
        _fixture_document("csv-parser", "csv_parser.md"),
        _fixture_document("record-filter", "record_filter.md"),
    )

    first = await _algorithm().plan_build(request)
    second = await _algorithm().plan_build(request)

    assert first.nodes == second.nodes
    assert first.embedding_texts == second.embedding_texts
    assert [node.content_hash for node in first.nodes] == [node.content_hash for node in second.nodes]


@pytest.mark.asyncio
async def test_plan_upsert_without_document_keeps_nodes_and_fills_texts() -> None:
    nodes = (
        SkillGraphNode(graph_node_id="skill:csv", node_type="skill", skill_id="csv", version_id="v1", name="CSV"),
        SkillGraphNode(
            graph_node_id="skill:filter", node_type="skill", skill_id="filter", version_id="v1", name="Filter"
        ),
        SkillGraphNode(graph_node_id="n-scene", node_type="scene", skill_id="csv", parent_skill_id="csv", name="Scene"),
    )
    snapshot = SkillGraphSnapshot(
        generation_id="g",
        revision=0,
        nodes=nodes,
        edges=(
            SkillGraphEdge(
                edge_id="e1", edge_type=EdgeType.DEPENDENCY, source_id="skill:csv", target_id="skill:filter"
            ),
        ),
        memberships=(
            SkillGraphMembership(skill_id="csv", version_id="v1", graph_node_id="skill:csv"),
            SkillGraphMembership(skill_id="filter", version_id="v1", graph_node_id="skill:filter"),
        ),
    )

    plan = await _algorithm().plan_upsert(
        SkillManageUpsertInput(
            scope=_scope(),
            version_ref=SkillVersionRef(skill_id="csv", version_id="v2"),
            snapshot=snapshot,
            embedding_profile=_profile(),
        )
    )

    assert plan.nodes == snapshot.nodes
    assert plan.edges == snapshot.edges
    assert plan.memberships == snapshot.memberships
    assert plan.embedding_texts == (
        SkillGraphEmbeddingText(
            graph_node_id="skill:csv", content_hash=content_hash("[Skill] CSV"), text="[Skill] CSV"
        ),
        SkillGraphEmbeddingText(
            graph_node_id="skill:filter", content_hash=content_hash("[Skill] Filter"), text="[Skill] Filter"
        ),
    )


@pytest.mark.asyncio
async def test_plan_upsert_with_document_rehashes_only_the_replaced_node() -> None:
    algorithm = _algorithm()
    built = await algorithm.plan_build(
        _build_input(
            _fixture_document("csv-parser", "csv_parser.md"),
            _fixture_document("record-filter", "record_filter.md"),
        )
    )
    snapshot = SkillGraphSnapshot(
        generation_id="g",
        revision=0,
        nodes=built.nodes,
        edges=built.edges,
        memberships=tuple(
            SkillGraphMembership(skill_id=node.skill_id, version_id=node.version_id, graph_node_id=node.graph_node_id)
            for node in built.nodes
        ),
    )
    original = {node.graph_node_id: node.content_hash for node in built.nodes}

    same_content = await algorithm.plan_upsert(
        SkillManageUpsertInput(
            scope=_scope(),
            version_ref=SkillVersionRef(skill_id="csv-parser", version_id="v2"),
            snapshot=snapshot,
            embedding_profile=_profile(),
            document=_fixture_document("csv-parser", "csv_parser.md", version_id="v2"),
        )
    )
    assert {node.graph_node_id: node.content_hash for node in same_content.nodes} == original

    changed = _fixture_document("csv-parser", "csv_parser.md", version_id="v3")
    changed = changed.model_copy(
        update={"content": changed.content.replace("Parses raw CSV text into structured rows", "Parses CSV exports")}
    )
    rehashed = await algorithm.plan_upsert(
        SkillManageUpsertInput(
            scope=_scope(),
            version_ref=SkillVersionRef(skill_id="csv-parser", version_id="v3"),
            snapshot=snapshot,
            embedding_profile=_profile(),
            document=changed,
        )
    )
    hashes = {node.graph_node_id: node.content_hash for node in rehashed.nodes}
    assert hashes["skill:record-filter"] == original["skill:record-filter"]
    assert hashes["skill:csv-parser"] != original["skill:csv-parser"]
    assert [entry.graph_node_id for entry in rehashed.embedding_texts] == [
        node.graph_node_id for node in rehashed.nodes
    ]
    assert [entry.content_hash for entry in rehashed.embedding_texts] == [node.content_hash for node in rehashed.nodes]
    assert "[DESCRIPTION] Parses CSV exports" in rehashed.embedding_texts[0].text


@pytest.mark.asyncio
async def test_plan_remove_keeps_remaining_nodes_identical_and_texts_for_skills_only() -> None:
    csv = FULL_NODE.model_copy(update={"content_hash": content_hash(FULL_TEXT)})
    filter_node = SkillGraphNode(
        graph_node_id="skill:filter",
        node_type="skill",
        skill_id="filter",
        version_id="v1",
        name="Filter",
        description="Filters records.",
        content_hash=content_hash("[Skill] Filter\n[DESCRIPTION] Filters records."),
    )
    filter_scene = SkillGraphNode(
        graph_node_id="scene:filter",
        node_type="scene",
        skill_id="filter",
        parent_skill_id="filter",
        name="Filter scene",
    )
    snapshot = SkillGraphSnapshot(
        generation_id="g",
        revision=0,
        nodes=(csv, SCENE_NODE, filter_node, filter_scene),
        edges=(
            SkillGraphEdge(
                edge_id="e1", edge_type=EdgeType.DEPENDENCY, source_id=csv.graph_node_id, target_id="skill:filter"
            ),
            SkillGraphEdge(
                edge_id="e2", edge_type=EdgeType.CONTAINMENT, source_id="skill:filter", target_id="scene:filter"
            ),
        ),
        memberships=(
            SkillGraphMembership(skill_id="csv-parser", version_id="v1", graph_node_id=csv.graph_node_id),
            SkillGraphMembership(skill_id="filter", version_id="v1", graph_node_id="skill:filter"),
        ),
    )

    plan = await _algorithm().plan_remove(
        SkillManageRemoveInput(
            scope=_scope(),
            selector=SkillManageRemoveSelector(kind="family", skill_id="csv-parser"),
            snapshot=snapshot,
        )
    )

    assert plan.nodes == (filter_node, filter_scene)
    assert plan.nodes[0] is filter_node
    assert plan.edges == (snapshot.edges[1],)
    assert [m.skill_id for m in plan.memberships] == ["filter"]
    assert plan.embedding_texts == (
        SkillGraphEmbeddingText(
            graph_node_id="skill:filter",
            content_hash=filter_node.content_hash,
            text="[Skill] Filter\n[DESCRIPTION] Filters records.",
        ),
    )


# --- architecture ----------------------------------------------------------------


def test_embedding_module_imports_no_driver_or_storage_internals() -> None:
    tree = ast.parse(Path(embedding_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    drivers = {"fast_graphrag", "psycopg", "psycopg_pool", "sqlite3", "litellm"}
    forbidden_roots = ("mindmemos_skill.application", "mindmemos_skill.persistence", "mindmemos_skill.infra")
    assert not {name.split(".")[0] for name in imported} & drivers
    assert not [name for name in imported if name.startswith(forbidden_roots)]
    # The package-level architecture test only sees absolute module names, so
    # also make sure no relative import reaches application/persistence/infra.
    relative = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level > 0]
    assert relative, "embedding.py is expected to import its errors/typing relatively"
    assert not [n.module for n in relative if (n.module or "").startswith(("application", "persistence", "infra"))]
