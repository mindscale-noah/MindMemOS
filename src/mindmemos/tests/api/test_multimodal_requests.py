"""Images reach vanilla only with algo_config.multimodal.mode on; otherwise they are rejected (400)."""

from types import SimpleNamespace

import pytest
from mindmemos.api.algorithm import binding_for_memory_algorithm
from mindmemos.api.mappers import to_search_pipeline_input
from mindmemos.api.schemas import AddRequest, AuthContext, SearchRequest
from mindmemos.api.services import memory_service
from mindmemos.api.services.memory_service import MemoryService, _images_unsupported_reason
from mindmemos.config import MultimodalConfig
from mindmemos.errors import BadRequestError


def _auth(algorithm: str) -> AuthContext:
    return AuthContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        memory_algorithm=algorithm,
    )


class _NeverCalled:
    async def add_sync(self, *args, **kwargs):
        raise AssertionError("pipeline must not run")

    async def search(self, *args, **kwargs):
        raise AssertionError("pipeline must not run")


IMAGE_TURN = {"role": "user", "content": "", "images": [{"url": "https://example.com/a.png"}]}


@pytest.fixture
def mode(monkeypatch):
    """Bind ``algo_config.multimodal.mode`` as the request config would (project override)."""

    def bind(value: str) -> None:
        config = SimpleNamespace(algo_config=SimpleNamespace(multimodal=MultimodalConfig(mode=value)))
        monkeypatch.setattr(memory_service, "get_config", lambda: config)

    return bind


def test_vanilla_binds_the_multimodal_wrapper() -> None:
    binding = binding_for_memory_algorithm("vanilla")
    assert (binding.add_pipeline, binding.search_pipeline, binding.supports_images) == (
        "vanilla_multimodal_add",
        "vanilla",
        True,
    )
    assert binding_for_memory_algorithm("schema").supports_images is False


def test_image_only_turns_pass_request_validation() -> None:
    request = AddRequest(user_id="u1", messages=[IMAGE_TURN])
    assert request.messages[0].images[0].url == "https://example.com/a.png"
    with pytest.raises(ValueError):
        AddRequest(user_id="u1", messages=[{"role": "user", "content": " "}])


@pytest.mark.asyncio
@pytest.mark.parametrize(("algorithm", "multimodal_mode"), [("vanilla", "off"), ("schema", "enhanced")])
async def test_images_are_rejected_without_multimodal_mode(mode, algorithm: str, multimodal_mode: str) -> None:
    mode(multimodal_mode)
    service = MemoryService(add_pipeline=_NeverCalled(), search_pipeline=_NeverCalled(), skill_store=None)

    with pytest.raises(BadRequestError) as add_error:
        await service.add(_auth(algorithm), AddRequest(user_id="u1", messages=[IMAGE_TURN]))
    with pytest.raises(BadRequestError) as search_error:
        await service.search(
            _auth(algorithm), SearchRequest(user_id="u1", query="q", query_images=[{"url": "https://x/a.png"}])
        )

    assert add_error.value.code == "add.images_unsupported"
    assert search_error.value.code == "search.images_unsupported"


@pytest.mark.parametrize("multimodal_mode", ["basic", "enhanced"])
def test_vanilla_accepts_images_with_multimodal_mode_on(mode, multimodal_mode: str) -> None:
    mode(multimodal_mode)
    assert _images_unsupported_reason(_auth("vanilla")) is None


def test_multimodal_flag_reaches_the_pipeline_input() -> None:
    def convert(**fields):
        # top_k=None skips the request_top_k_max check, which reads the server config.
        return to_search_pipeline_input(
            SearchRequest(user_id="u1", query="q", top_k=None, **fields), search_pipeline="vanilla"
        )

    flagged, plain = convert(multimodal=True), convert()
    assert flagged.multimodal is True and plain.multimodal is False
    assert "multimodal" not in plain.model_dump()
