"""psycopg connection pool construction shared by the PostgreSQL-backed stores."""

from __future__ import annotations

import select
from typing import Any

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

# What ``SHOW search_path`` reports when neither the DSN, the role nor the
# database has set one.
_SERVER_DEFAULT_SEARCH_PATH = '"$user", public'


def validating_connection_pool(*, default_search_path: str | None = None, **kwargs: Any) -> AsyncConnectionPool:
    """An ``AsyncConnectionPool`` that never hands out a connection the server has closed.

    With ``default_search_path`` every new connection whose ``search_path`` is
    still the server default (nothing set on the DSN, the role or the
    database) is pinned to that schema. The default resolves unqualified names
    through ``"$user"`` first, so the moment a schema named after the role
    exists (the vector store creates one, and by default it shares the role's
    name) a restarted process would address a different set of tables than
    the one it created. A ``search_path`` someone did set is a deliberate
    choice and is left alone.

    After a server restart or failover every idle pooled connection is dead at
    once. The pool's own ``check`` would catch that, but it costs a round trip
    per checkout and, worse, the pool swallows a ``CancelledError`` that lands
    inside the check. So the checkout first peeks at the socket without
    awaiting: an idle connection with nothing pending is alive. Only when
    something is pending (EOF, the server's FATAL, an unsolicited notice) is
    the round-trip probe run; its first failure purges the whole pool at once,
    because the pool's retry policy would otherwise validate the dead
    connections one per attempt with exponential backoff and time out.
    """

    purging = False

    async def check(connection: Any) -> None:
        nonlocal purging
        if _quiet(connection):
            return
        try:
            await AsyncConnectionPool.check_connection(connection)
        except Exception:
            if not purging:
                purging = True
                try:
                    await pool.check()
                finally:
                    purging = False
            raise

    configure = _pin_default_search_path(default_search_path) if default_search_path else None
    pool = AsyncConnectionPool(check=check, configure=configure, **kwargs)
    return pool


def _pin_default_search_path(schema: str) -> Any:
    async def configure(connection: Any) -> None:
        row = await (await connection.execute("SHOW search_path")).fetchone()
        current = row["search_path"] if isinstance(row, dict) else row[0]
        if current == _SERVER_DEFAULT_SEARCH_PATH:
            await connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        await connection.commit()

    return configure


async def serialize_schema_bootstrap(connection: Any, key: str) -> None:
    """Hold a transaction-scoped advisory lock so concurrent bootstraps run one at a time.

    ``CREATE ... IF NOT EXISTS`` and the migration ledger insert are not safe
    against a concurrent identical transaction; the loser would fail on the
    catalog's unique index and the process would not start. The lock is taken
    under READ COMMITTED whatever the session default: a waiter must see the
    winner's ledger once it is released, not a snapshot from before.
    """

    await connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
    await connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))


def _quiet(connection: Any) -> bool:
    """True when an idle connection's socket has nothing to read: no EOF, no pending message."""

    try:
        readable, _, _ = select.select([connection.fileno()], [], [], 0)
    except Exception:
        return False
    return not readable


__all__ = ["serialize_schema_bootstrap", "validating_connection_pool"]
