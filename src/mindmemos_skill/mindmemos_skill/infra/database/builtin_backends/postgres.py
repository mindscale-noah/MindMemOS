"""PostgreSQL connections and pgvector-specific SQL adaptation."""

from sqlalchemy import Float, event, literal
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.types import TypeDecorator

from ...base import Infrastructure
from ..base import Database
from ..config import PostgresConfig
from ..vector import Vector


async def _check_extension(connection) -> None:
    """Check pgvector through the native driver and leave no open transaction."""
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            installed = await cursor.fetchone() is not None
    finally:
        await connection.rollback()
    if not installed:
        raise RuntimeError("PostgreSQL requires the vector extension; install it during deployment")


class PostgresVector(TypeDecorator):
    """Use pgvector's codec while preserving portable Python list results."""

    impl = Float
    cache_ok = True

    def __init__(self, dimensions):
        from pgvector.sqlalchemy import Vector as PGVector

        super().__init__()
        self.dimensions = dimensions
        self.impl = PGVector(dimensions)

    def process_bind_param(self, value, dialect):
        """Validate values before pgvector serialization.

        Args:
            value: Vector values, or None.
            dialect: SQLAlchemy dialect.
        """
        return None if value is None else Vector(self.dimensions).validate(value)

    def process_result_value(self, value, dialect):
        """Convert pgvector arrays into portable Python lists.

        Args:
            value: Decoded vector, or None.
            dialect: SQLAlchemy dialect.
        """
        return None if value is None else list(value)


@Infrastructure.register("postgres")
class PostgresDatabase(Database):
    """Remote storage using PostgreSQL with a preinstalled vector extension."""

    config_type = PostgresConfig
    """Validated remote connection and pool settings."""

    def _create_engine(self, config):
        engine = create_async_engine(
            config.url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_pre_ping=True,
        )
        engine.dialect.colspecs = {**engine.dialect.colspecs, Vector: PostgresVector}

        @event.listens_for(engine.sync_engine, "connect")
        def check_extension(connection, record):
            connection.run_async(_check_extension)

        return engine

    def _insert_statement(self, table):
        return insert(table)

    def _distance(self, column, vector):
        return column.op("<=>", return_type=Float)(literal(vector, type_=column.type))
