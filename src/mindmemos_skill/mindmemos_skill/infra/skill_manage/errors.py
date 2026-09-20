"""Narrow classification of transient failures from a storage call."""

from __future__ import annotations


def is_transient_storage_error(exc: Exception) -> bool:
    """Recognize transport failures without retrying arbitrary database errors.

    Call only at a storage boundary: a timeout raised by an algorithm has no
    implied storage-retry semantics. SQLSTATEs for authorization, invalid SQL
    and unknown ``OperationalError`` failures deliberately remain excluded.
    """

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        from psycopg import Error
        from psycopg.errors import ConnectionTimeout
        from psycopg_pool import PoolTimeout
    except ImportError:
        return False
    return isinstance(exc, (ConnectionTimeout, PoolTimeout)) or (
        isinstance(exc, Error)
        and exc.sqlstate
        in {
            "08003",  # connection does not exist
            "08006",  # connection failure
            "40001",  # serialization failure
            "40P01",  # deadlock detected
            "57P01",  # administrative shutdown
            "57P02",  # crash shutdown
            "57P03",  # server cannot accept connections yet
        }
    )
