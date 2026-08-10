"""Ensure-collection idempotency tests for :class:`QdrantEngine`."""

import pytest
from grpc import StatusCode
from grpc.aio import AioRpcError
from mindmemos.config import QdrantConfig
from mindmemos.infra.db.engine import QdrantEngine
from mindmemos.infra.db.models import QdrantCollectionSpec
from qdrant_client.http.exceptions import UnexpectedResponse


class _FakeQdrantClient:
    """Records calls; ``create_collection`` behaviour is configured per test."""

    def __init__(self, *, exists: bool = False, create_error: Exception | None = None) -> None:
        self._exists = exists
        self._create_error = create_error
        self.created: list[dict] = []
        self.exists_calls: list[str] = []

    async def collection_exists(self, collection_name: str, **kwargs) -> bool:
        self.exists_calls.append(collection_name)
        return self._exists

    async def create_collection(self, **kwargs) -> None:
        if self._create_error is not None:
            raise self._create_error
        self.created.append(kwargs)

    async def create_payload_index(self, **kwargs) -> None:
        return None

    async def update_collection(self, **kwargs) -> None:
        return None


def _spec(name: str = "test_memos") -> QdrantCollectionSpec:
    return QdrantCollectionSpec(name=name, vector_size=2, enable_dense=True)


def _engine(client: _FakeQdrantClient) -> QdrantEngine:
    cfg = QdrantConfig(
        url="http://unused",
        max_client_concurrency=1,
        max_client_concurrency_cap=1,
        max_retries=1,
        retry_base_delay=0.0,
    )
    return QdrantEngine(cfg, client=client)


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing():
    client = _FakeQdrantClient(exists=False)
    await _engine(client).ensure_collection(_spec())

    assert len(client.created) == 1
    assert client.created[0]["collection_name"] == "test_memos"
    assert client.created[0]["vectors_config"]["semantic"].size == 2


@pytest.mark.asyncio
async def test_ensure_collection_skips_create_when_exists():
    client = _FakeQdrantClient(exists=True)
    await _engine(client).ensure_collection(_spec())

    assert client.created == []
    assert client.exists_calls == ["test_memos"]


@pytest.mark.asyncio
async def test_ensure_collection_tolerates_rest_409():
    error = UnexpectedResponse(status_code=409, reason_phrase="Conflict", headers={}, content=b"already exists")
    client = _FakeQdrantClient(exists=False, create_error=error)
    await _engine(client).ensure_collection(_spec())

    assert client.created == []


@pytest.mark.asyncio
async def test_ensure_collection_tolerates_grpc_already_exists():
    error = AioRpcError(
        code=StatusCode.ALREADY_EXISTS,
        initial_metadata=None,
        trailing_metadata=None,
        details="Collection test_memos already exists!",
    )
    client = _FakeQdrantClient(exists=False, create_error=error)
    await _engine(client).ensure_collection(_spec())

    assert client.created == []


@pytest.mark.asyncio
async def test_ensure_collection_propagates_other_errors():
    error = RuntimeError("boom")
    client = _FakeQdrantClient(exists=False, create_error=error)
    with pytest.raises(RuntimeError, match="boom"):
        await _engine(client).ensure_collection(_spec())
