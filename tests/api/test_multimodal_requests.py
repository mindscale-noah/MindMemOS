"""Images reach only multimodal algorithms; text-only algorithms reject them with a clear error."""

import pytest
from mindmemos.api.algorithm import binding_for_memory_algorithm
from mindmemos.api.schemas import AddRequest, AuthContext, SearchRequest
from mindmemos.api.services.memory_service import MemoryService
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


def test_vanilla_mm_binding_reuses_the_algorithm_registry() -> None:
    binding = binding_for_memory_algorithm("vanilla_mm")
    assert (binding.add_pipeline, binding.search_pipeline, binding.supports_images) == (
        "vanilla_mm_add",
        "vanilla_mm",
        True,
    )
    assert binding_for_memory_algorithm("vanilla").supports_images is False


def test_image_only_turns_pass_request_validation() -> None:
    request = AddRequest(user_id="u1", messages=[IMAGE_TURN])
    assert request.messages[0].images[0].url == "https://example.com/a.png"
    with pytest.raises(ValueError):
        AddRequest(user_id="u1", messages=[{"role": "user", "content": " "}])


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["vanilla", "schema"])
async def test_text_only_algorithms_reject_images(algorithm: str) -> None:
    service = MemoryService(add_pipeline=_NeverCalled(), search_pipeline=_NeverCalled(), skill_store=None)

    with pytest.raises(BadRequestError) as add_error:
        await service.add(_auth(algorithm), AddRequest(user_id="u1", messages=[IMAGE_TURN]))
    with pytest.raises(BadRequestError) as search_error:
        await service.search(
            _auth(algorithm), SearchRequest(user_id="u1", query="q", query_images=[{"url": "https://x/a.png"}])
        )

    assert add_error.value.code == "add.images_unsupported"
    assert search_error.value.code == "search.images_unsupported"
