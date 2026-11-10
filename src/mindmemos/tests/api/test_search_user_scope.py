from types import SimpleNamespace

import pytest
from mindmemos.api import mappers as api_mappers
from mindmemos.api.schemas import SearchRequest
from mindmemos.config.algo.search import SearchConfig
from mindmemos.mappers import parse_schema_search_filters
from mindmemos.pipelines.search.vanilla.engine import _request_filter as vanilla_request_filter
from mindmemos.typing.memory import MemoryRequestContext, SearchFilter


@pytest.fixture(autouse=True)
def search_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SearchConfig()
    monkeypatch.setattr(
        api_mappers,
        "get_config",
        lambda: SimpleNamespace(algo_config=SimpleNamespace(search=config)),
    )


def make_context(*, user_id: str | None) -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id=user_id,
    )


def mandatory_matches(search_filter: SearchFilter | None, field: str) -> list[str]:
    if search_filter is None:
        return []
    values: list[str] = []
    for clause in search_filter.must:
        if isinstance(clause, SearchFilter):
            values.extend(mandatory_matches(clause, field))
        elif clause.field == field and clause.op == "match" and clause.value is not None:
            values.append(clause.value)
    return values


def test_search_user_id_is_optional_and_adds_no_filter_when_omitted() -> None:
    request_filters = {"mem_type": "fact"}

    inp = api_mappers.to_search_pipeline_input(
        SearchRequest(query="Qdrant", filters=request_filters),
        search_pipeline="vanilla",
    )

    assert inp.filters == request_filters


def test_search_user_id_is_a_mandatory_filter_for_schema_and_vanilla() -> None:
    inp = api_mappers.to_search_pipeline_input(
        SearchRequest(
            query="Qdrant",
            user_id="user-1",
            filters={"OR": [{"user_id": "other-user"}, {"mem_type": "fact"}]},
        ),
        search_pipeline="vanilla",
    )
    context = make_context(user_id="user-1")

    schema_filters = parse_schema_search_filters(inp.filters, context)
    vanilla_filters = vanilla_request_filter(inp, context)

    assert mandatory_matches(schema_filters.memory_filter, "user_id") == ["user-1"]
    assert mandatory_matches(vanilla_filters, "user_id") == ["user-1"]
    assert schema_filters.context.user_id == "user-1"
