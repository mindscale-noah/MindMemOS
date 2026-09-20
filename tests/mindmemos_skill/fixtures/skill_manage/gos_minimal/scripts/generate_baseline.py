#!/usr/bin/env python3
"""GOS minimal baseline golden generator (AC-P0-003).

Runs the REAL GOS engine (gos.core.engine.SkillGraphRAG) over the fixture
skills + SSL records with deterministic content-hash embedding/LLM stubs, then
captures normalized, reproducible intermediate artifacts:

    parsed   - parse_skill_document outputs for every fixture Skill
    nodes    - Skill + Scene nodes actually present in graph storage
    edges    - edges actually present in graph storage (deduped by logical key)
    scenes   - scene structure recovered from SSL records
    seeds    - retrieval seed scores for a fixed query
    ppr      - personalized PageRank over the real stored graph
    top_k    - retrieval ranking for the fixed query
    context  - retrieval summary/counts

Every numeric/ordering artifact comes from the engine itself or from the
canonical GOS retrieval functions over the engine's own graph — nothing is
hardcoded. Run twice with the same locked environment and compare SHA-256.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # .../gos_minimal/scripts
FIXTURE_DIR = SCRIPT_DIR.parent  # .../gos_minimal
SKILLS_DIR = FIXTURE_DIR / "skills"
SSL_DIR = FIXTURE_DIR / "ssl_records"
EXPECTED_DIR = FIXTURE_DIR / "expected"
WORKSPACE_DIR = FIXTURE_DIR / ".baseline_workspace"


def reference_source() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gos-source",
        required=True,
        type=Path,
        help="External reference checkout containing gos/core/engine.py; see baseline_manifest.json for provenance.",
    )
    source = parser.parse_args().gos_source.expanduser().resolve()
    if not (source / "gos" / "core" / "engine.py").is_file():
        parser.error(f"--gos-source must contain gos/core/engine.py: {source}")
    return source


# The GOS reference engine is intentionally not shipped with this repository.
# Validate the explicit source before loading any optional baseline dependencies.
sys.path.insert(0, str(reference_source()))

import numpy as np  # noqa: E402
from gos.core.engine import SkillGraphRAG  # noqa: E402
from gos.core.parsing import parse_skill_document  # noqa: E402
from gos.core.retrieval import (  # noqa: E402
    build_personalization,
    build_transition_matrix,
    personalized_pagerank,
)

FIXED_QUERY = "parse a csv file, filter its records, and merge them into a json document"
EMBEDDING_DIM = 3072
EMBEDDING_STUB_SEED = "fixed-seed-12345"
LLM_STUB_SEED = "fixed-seed-67890"


# --------------------------------------------------------------------------
# Deterministic content-hash embedding: similar text -> similar vector.
# Feature-hashing over character trigrams and word tokens with signed buckets.
# --------------------------------------------------------------------------
class ContentHashEmbeddingService:
    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.embedding_dim = dimension
        self.dimension = dimension

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]

    @staticmethod
    def _grams(text: str) -> list[str]:
        lowered = text.lower()
        return [lowered[i : i + 3] for i in range(max(len(lowered) - 2, 0))]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.embedding_dim, dtype=np.float64)
        for gram in self._grams(text) + self._tokens(text):
            digest = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            vec[digest % self.embedding_dim] += 1.0 if (digest >> 8) % 2 == 0 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32).tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


# --------------------------------------------------------------------------
# Deterministic LLM stub: GOSGraph extraction -> empty; relation validation
# -> emit exactly one semantic edge between the data-transfer pair when both
# are present in the prompt (content-driven, not positional).
# --------------------------------------------------------------------------
class StubLLMService:
    def __init__(self, seed: str = LLM_STUB_SEED):
        self.seed = seed

    async def send_message(self, prompt, system_prompt=None, history_messages=None, response_model=None, **kwargs):
        from gos.core.schema import GOSGraph, GOSRelation, GOSRelationList

        model_name = getattr(response_model, "__name__", "") if response_model else ""
        if model_name == "GOSGraph":
            graph = GOSGraph(nodes=[], edges=[])
            return (graph, '{"nodes": [], "edges": []}')
        if model_name == "GOSRelationList":
            # Content-driven: both members of the data-transfer pair present?
            if "S3 Upload Skill" in prompt and "Data Sync Skill" in prompt:
                relations = GOSRelationList(
                    relations=[
                        GOSRelation(
                            source="S3 Upload Skill",
                            target="Data Sync Skill",
                            type="semantic",
                            description="Shares the data-transfer domain",
                            confidence=0.9,
                        )
                    ]
                )
            else:
                relations = GOSRelationList(relations=[])
            return (relations, relations.model_dump_json())
        return "{}"

    async def generate(self, prompt, **kwargs) -> str:
        result = await self.send_message(prompt, **kwargs)
        if isinstance(result, tuple):
            return result[1]
        return str(result)


# --------------------------------------------------------------------------
# Engine construction
# --------------------------------------------------------------------------
def build_engine(workspace: str, ssl_dir: str) -> SkillGraphRAG:
    config = SkillGraphRAG.Config(
        llm_service=StubLLMService(),
        embedding_service=ContentHashEmbeddingService(),
        working_dir=workspace,
        domain="gos-minimal-baseline",
        link_top_k=10,
        retrieval_top_n=5,
        seed_top_k=3,
        enable_semantic_linking=True,
        dependency_match_threshold=0.5,
        max_skill_chars=4000,
        snippet_chars=800,
        use_ssl=True,
        ssl_records_dir=ssl_dir,
        ssl_join_file="",
        scene_flow_in_embed=True,
        enable_flow_edges=True,
        enable_scene_llm_linking=False,
        dep_edge_filter="none",
        retrieval_mode="skill_only",
    )
    return SkillGraphRAG(config=config)


# --------------------------------------------------------------------------
# Capture helpers
# --------------------------------------------------------------------------
def normalize_node(node) -> dict:
    """Node -> stable JSON (no storage-internal ids/paths/timestamps)."""
    return {
        "name": node.name,
        "type": getattr(node, "type", "Skill"),
        "description": getattr(node, "description", ""),
        "one_line_capability": getattr(node, "one_line_capability", ""),
        "inputs": list(getattr(node, "input_types", []) or []),
        "outputs": list(getattr(node, "output_types", []) or []),
        "domain_tags": list(getattr(node, "domain_tags_list", []) or []),
        "tooling": list(getattr(node, "tooling_list", []) or []),
    }


def normalize_edge(edge) -> dict:
    return {
        "source": edge.source,
        "target": edge.target,
        "type": edge.type,
        "weight": float(edge.weight),
    }


def edge_key(edge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.type)


async def load_graph(engine) -> tuple[list, list]:
    """Load all nodes and edges from real graph storage."""
    nodes: list = []
    edges: list = []
    await engine.state_manager.query_start()
    try:
        node_count = await engine.state_manager.graph_storage.node_count()
        for i in range(node_count):
            node = await engine.state_manager.graph_storage.get_node_by_index(i)
            if node is not None:
                nodes.append(node)
        edge_count = await engine.state_manager.graph_storage.edge_count()
        for i in range(edge_count):
            edge = await engine.state_manager.graph_storage.get_edge_by_index(i)
            if edge is not None:
                edges.append(edge)
    finally:
        await engine.state_manager.query_done()
    return nodes, edges


# --------------------------------------------------------------------------
# Main golden generation
# --------------------------------------------------------------------------
async def generate() -> dict:
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fixture skills + metadata
    skill_files = sorted(SKILLS_DIR.glob("*.md"))
    contents = []
    metadatas = []
    for skill_file in skill_files:
        content = skill_file.read_text(encoding="utf-8")
        contents.append(content)
        metadatas.append(
            {
                "source_path": f"skills/{skill_file.name}",
                "raw_content": content,
                "snippet_chars": "800",
            }
        )

    # 2. Parsed golden (pure parser)
    parsed_entries = []
    for skill_file, content in zip(skill_files, contents):
        parsed = parse_skill_document(content, source_path=f"skills/{skill_file.name}", snippet_chars=800)
        if parsed is None:
            raise RuntimeError(f"parse_skill_document returned None for {skill_file.name}")
        parsed_entries.append(
            {
                "name": parsed.name,
                "description": parsed.description,
                "source_path": parsed.source_path,
                "inputs": list(parsed.inputs),
                "outputs": list(parsed.outputs),
                "domain_tags": list(parsed.domain_tags),
            }
        )
    parsed_entries.sort(key=lambda p: p["name"])

    # 3. Run the real engine
    engine = build_engine(str(WORKSPACE_DIR), str(SSL_DIR))
    sync_result = await engine.async_ensure_skills(contents, metadatas)
    # Full linking pass: incremental linking skips new-new pairs (so the
    # S3<->DataSync semantic pair would never be LLM-validated in one batch);
    # the full pass evaluates every pair and produces the semantic edge.
    await engine._link_all_skills()
    scene_stats = await engine.async_build_scene_graph()

    # 4. Real stored nodes/edges (insertion order from storage)
    nodes, edges = await load_graph(engine)
    name_by_index = {i: n.name for i, n in enumerate(nodes)}
    skill_nodes_sorted = sorted([n for n in nodes if getattr(n, "type", "Skill") == "Skill"], key=lambda n: n.name)
    scene_nodes_sorted = sorted([n for n in nodes if getattr(n, "type", "Skill") != "Skill"], key=lambda n: n.name)

    # Dedup edges by logical key (engine emits the semantic pair twice in full
    # linking: once from each direction).
    seen_keys: set[tuple[str, str, str]] = set()
    unique_edges: list = []
    for edge in sorted(edges, key=lambda e: (e.source, e.target, e.type)):
        key = edge_key(edge)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_edges.append(edge)

    # 5. Retrieval result from the real engine (fixed query, skill_only).
    #    Run BEFORE PPR so the canonical ppr.scores below uses the real
    #    semantic seed weights (mirrors the engine's own skill_only PPR).
    try:
        retrieval = await engine.async_retrieve(FIXED_QUERY, retrieval_mode="skill_only")
        retrieval_dict = serialize_retrieval(retrieval)
    except Exception as exc:  # noqa: BLE001
        retrieval_dict = {"error": f"{type(exc).__name__}: {exc}"}

    # 6. Canonical PPR: skill_only projection (Skill nodes, Skill-to-Skill
    #    edges) with the real seed weights, replicating the engine branch
    #    ``mode == "skill_only" and has_scenes``.
    proj_nodes = [n for n in nodes if getattr(n, "type", "Skill") == "Skill"]
    proj_names = {n.name for n in proj_nodes}
    proj_edges = [e for e in edges if e.source in proj_names and e.target in proj_names]
    proj_edges = dedup_edges(proj_edges)
    proj_index_by_name = {n.name: i for i, n in enumerate(proj_nodes)}
    transition, _ = build_transition_matrix(proj_nodes, proj_edges)
    seed_indices = []
    seed_weights = []
    for seed in getattr(retrieval, "seeds", []) or []:
        proj_index = proj_index_by_name.get(seed.name)
        if proj_index is not None:
            seed_indices.append(proj_index)
            seed_weights.append(seed.seed_weight)
    personalization = build_personalization(len(proj_nodes), seed_indices, seed_weights or None)
    ppr_scores = personalized_pagerank(
        transition,
        personalization,
        damping=0.2,
        max_iter=50,
        tol=1e-6,
    )
    ppr_serialized = {
        "damping": 0.2,
        "max_iter": 50,
        "scores": {proj_nodes[i].name: float(ppr_scores[i]) for i in range(len(proj_nodes))},
    }

    # 7. Scene summary from SSL records
    scenes_serialized = serialize_scenes(skill_files)

    golden = {
        "schema_version": 1,
        "description": "GOS minimal baseline golden (real engine, content-hash stubs)",
        "query": FIXED_QUERY,
        "embedding_stub_seed": EMBEDDING_STUB_SEED,
        "llm_stub_seed": LLM_STUB_SEED,
        "parsed": parsed_entries,
        "nodes": [normalize_node(n) for n in skill_nodes_sorted + scene_nodes_sorted],
        "edges": [normalize_edge(e) for e in unique_edges],
        "scenes": scenes_serialized,
        "seeds": retrieval_dict.get("seeds", []),
        "ppr": ppr_serialized,
        "top_k": retrieval_dict.get("top_k", []),
        "context": {
            "skill_count": len(skill_nodes_sorted),
            "scene_count": len(scene_nodes_sorted),
            "edge_count": len(unique_edges),
            "sync_result": {
                "reused": getattr(sync_result, "reused_count", None),
                "inserted": getattr(sync_result, "inserted_count", None),
                "updated": getattr(sync_result, "updated_count", None),
            },
            "scene_stats": {k: v for k, v in scene_stats.items() if isinstance(v, (int, str))},
            "retrieval": retrieval_dict.get("context", {}),
        },
    }
    return golden


def dedup_edges(edges):
    """Dedup edges by logical key, preserving deterministic sort."""
    seen: set[tuple[str, str, str]] = set()
    result = []
    for edge in sorted(edges, key=lambda e: (e.source, e.target, e.type)):
        key = edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        result.append(edge)
    return result


def serialize_scenes(skill_files) -> list[dict]:
    """Scene structure from fixture SSL records, joined to Skill names."""
    scenes_out: list[dict] = []
    for record_file in sorted(SSL_DIR.glob("*.json")):
        record = json.loads(record_file.read_text(encoding="utf-8"))
        scenes = []
        for scene in record.get("scenes") or []:
            scenes.append(
                {
                    "scene_id": scene.get("scene_id", ""),
                    "scene_name": scene.get("scene_name", ""),
                    "scene_goal": scene.get("scene_goal", ""),
                }
            )
        scenes_out.append({"skill": record_file.stem, "scenes": scenes})
    scenes_out.sort(key=lambda s: s["skill"])
    return scenes_out


def serialize_retrieval(retrieval) -> dict:
    """Flatten a SkillRetrievalResult to stable JSON."""
    seeds = []
    for seed in getattr(retrieval, "seeds", []) or []:
        seeds.append(
            {
                "name": getattr(seed, "name", None),
                "seed_weight": float(getattr(seed, "seed_weight", 0.0) or 0.0),
                "semantic_rank": getattr(seed, "semantic_rank", None),
            }
        )
    top_k = []
    for hit in getattr(retrieval, "skills", []) or []:
        top_k.append(
            {
                "name": getattr(hit, "name", None),
                "score": float(getattr(hit, "score", 0.0) or 0.0),
                "rerank_score": float(getattr(hit, "rerank_score", 0.0) or 0.0),
            }
        )
    relations = []
    for rel in getattr(retrieval, "relations", []) or []:
        relations.append(
            {
                "source": getattr(rel, "source", None),
                "target": getattr(rel, "target", None),
                "type": getattr(rel, "type", None),
                "weight": float(getattr(rel, "weight", 0.0) or 0.0),
            }
        )
    return {
        "seeds": seeds,
        "top_k": top_k,
        "relations": relations,
        "context": {
            "summary": getattr(retrieval, "summary", None),
            "rendered_context": getattr(retrieval, "rendered_context", None),
        },
    }


async def main() -> int:
    print(f"Generating GOS baseline golden... (query={FIXED_QUERY!r})")
    golden = await generate()
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = EXPECTED_DIR / "golden.json"
    payload = json.dumps(golden, indent=2, sort_keys=True, ensure_ascii=False)
    golden_path.write_text(payload + "\n", encoding="utf-8")
    digest = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    print(f"golden.json SHA-256: {digest}")
    print(f"nodes={len(golden['nodes'])} edges={len(golden['edges'])} scenes={len(golden['scenes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
