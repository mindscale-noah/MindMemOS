from types import SimpleNamespace

import pytest
from mindmemos.api import mappers as api_mappers
from mindmemos.api.schemas import SearchRequest
from mindmemos.config import SearchConfig
from mindmemos.errors import BadRequestError
from pydantic import ValidationError


def test_search_patches_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    search_config = SearchConfig()
    monkeypatch.setattr(
        api_mappers,
        "get_config",
        lambda: SimpleNamespace(algo_config=SimpleNamespace(search=search_config)),
    )

    inp = api_mappers.to_search_pipeline_input(
        SearchRequest(user_id="u1", query="Qdrant"),
        search_pipeline="vanilla",
    )

    assert search_config.include_patches is True
    assert inp.include_patches is True


def test_search_request_rejects_removed_search_controls() -> None:
    with pytest.raises(ValidationError, match="include_scores"):
        SearchRequest.model_validate({"user_id": "u1", "query": "Qdrant", "include_scores": True})


def test_search_request_accepts_token_budget() -> None:
    request = SearchRequest.model_validate({"user_id": "u1", "query": "Qdrant", "token_budget": 850})

    assert request.token_budget == 850
    assert SearchRequest.model_validate({"user_id": "u1", "query": "Qdrant"}).token_budget is None


def test_search_pipeline_rejects_token_budget_out_of_configured_range(monkeypatch: pytest.MonkeyPatch) -> None:
    search_config = SearchConfig()
    monkeypatch.setattr(
        api_mappers,
        "get_config",
        lambda: SimpleNamespace(algo_config=SimpleNamespace(search=search_config)),
    )

    with pytest.raises(BadRequestError, match="token_budget"):
        api_mappers.to_search_pipeline_input(
            SearchRequest(user_id="u1", query="Qdrant", token_budget=999_999_999),
            search_pipeline="vanilla",
        )
