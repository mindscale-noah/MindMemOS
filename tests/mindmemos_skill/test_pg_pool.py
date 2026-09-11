"""Connection-pool helpers shared by the PostgreSQL-backed stores."""

from __future__ import annotations

import pytest
from mindmemos_skill.infra.pg_pool import _SERVER_DEFAULT_SEARCH_PATH, _pin_default_search_path


class _Connection:
    def __init__(self, search_path: str) -> None:
        self.search_path = search_path
        self.statements: list[str] = []
        self.commits = 0

    async def execute(self, statement, params=None):
        text = statement if isinstance(statement, str) else statement.as_string(None)
        self.statements.append(text)
        return self

    async def fetchone(self):
        return {"search_path": self.search_path}

    async def commit(self) -> None:
        self.commits += 1


@pytest.mark.asyncio
async def test_only_the_server_default_search_path_is_pinned() -> None:
    """A search_path set on the DSN, the role or the database is a deliberate choice and stays."""

    configure = _pin_default_search_path("public")

    untouched = _Connection(_SERVER_DEFAULT_SEARCH_PATH)
    await configure(untouched)
    assert untouched.statements == ["SHOW search_path", 'SET search_path TO "public"']
    assert untouched.commits == 1

    deliberate = _Connection("app, public")
    await configure(deliberate)
    assert deliberate.statements == ["SHOW search_path"]
    assert deliberate.commits == 1
