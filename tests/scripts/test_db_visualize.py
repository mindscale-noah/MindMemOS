from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "db_visualize.py"
SPEC = importlib.util.spec_from_file_location("db_visualize", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
db_visualize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(db_visualize)


def test_default_project_id_uses_internal_value():
    assert db_visualize.default_project_id() == "proj_dev_0001"


class FakeQdrantClient:
    def __init__(self) -> None:
        self.scroll_calls: list[str] = []
        self.closed = False

    def scroll(self, *, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        self.scroll_calls.append(collection_name)
        return (
            [
                SimpleNamespace(
                    payload={
                        "memory_id": "mem-1",
                        "project_id": "proj",
                        "content": "User likes tea.",
                        "mem_type": "fact",
                        "status": "active",
                    }
                )
            ],
            None,
        )

    def close(self) -> None:
        self.closed = True


def test_load_memories_uses_current_qdrant_memory_collection(monkeypatch):
    fake_client = FakeQdrantClient()
    monkeypatch.setattr(db_visualize, "qdrant_client", lambda: fake_client)
    monkeypatch.setattr(
        db_visualize,
        "get_config",
        lambda: SimpleNamespace(
            database=SimpleNamespace(
                qdrant=SimpleNamespace(
                    memory_collection="memory_item_v1",
                    entity_collection="entity_item_v1",
                    source_collection="source_item_v1",
                    add_record_collection="add_record_v1",
                    search_record_collection="search_record_v1",
                )
            )
        ),
    )

    result = db_visualize.load_memories("proj", "active", None)

    assert fake_client.scroll_calls == ["memory_item_v1"]
    assert fake_client.closed is True
    assert result["collection"] == "memory_item_v1"
    assert result["collections"]["entity"] == "entity_item_v1"
    assert result["memories"][0]["memory_id"] == "mem-1"


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class FakeNeo4jDriver:
    def __init__(self) -> None:
        self.closed = False

    def execute_query(self, query, params, **kwargs):
        return (
            [
                FakeRecord(
                    {
                        "nodes": [
                            {
                                "id": "mem-1",
                                "label": "Memory",
                                "labels": ["Memory"],
                                "properties": {"memory_id": "mem-1", "content": "User likes tea."},
                            },
                            {
                                "id": "ent-1",
                                "label": "Entity",
                                "labels": ["Entity"],
                                "properties": {"entity_id": "ent-1", "entity_name": "tea"},
                            },
                        ],
                        "edges": [
                            {
                                "id": "rel-1",
                                "type": "MENTIONS",
                                "source": "mem-1",
                                "target": "ent-1",
                                "properties": {"project_id": "proj"},
                            }
                        ],
                    }
                )
            ],
            None,
            None,
        )

    def close(self) -> None:
        self.closed = True


def test_load_graph_returns_visual_nodes_and_edges(monkeypatch):
    fake_driver = FakeNeo4jDriver()
    monkeypatch.setattr(db_visualize, "neo4j_driver", lambda: fake_driver)
    monkeypatch.setattr(
        db_visualize,
        "get_config",
        lambda: SimpleNamespace(database=SimpleNamespace(neo4j=SimpleNamespace(database="neo4j"))),
    )

    result = db_visualize.load_graph("proj", "mem-1")

    assert fake_driver.closed is True
    assert result["nodes"][0]["id"] == "mem-1"
    assert result["nodes"][0]["label"] == "Memory"
    assert result["edges"] == [
        {
            "id": "rel-1",
            "type": "MENTIONS",
            "source": "mem-1",
            "target": "ent-1",
            "properties": {"project_id": "proj"},
        }
    ]
    assert result["entities"][0]["id"] == "ent-1"
