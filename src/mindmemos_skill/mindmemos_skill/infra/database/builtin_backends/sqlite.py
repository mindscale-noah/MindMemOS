"""SQLite engine setup and sqlite-vec vector adaptation."""

import struct
from pathlib import Path

from sqlalchemy import LargeBinary, create_engine, event, func, literal
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import make_url
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

from ...base import Infrastructure
from ..base import Database
from ..config import SQLiteConfig
from ..vector import Vector


class SQLiteVector(TypeDecorator):
    """Encode portable vectors as sqlite-vec float32 blobs."""

    impl = LargeBinary
    cache_ok = True

    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions

    def process_bind_param(self, value, dialect):
        """Encode a vector for SQLite.

        Args:
            value: Vector values, or None.
            dialect: SQLAlchemy dialect.
        """
        if value is None:
            return None
        values = Vector(self.dimensions).validate(value)
        return struct.pack(f"{len(values)}f", *values)

    def process_result_value(self, value, dialect):
        """Decode a stored vector into a Python list.

        Args:
            value: Stored blob, or None.
            dialect: SQLAlchemy dialect.
        """
        return None if value is None else list(struct.unpack(f"{len(value) // 4}f", value))


@compiles(Vector, "sqlite")
def compile_vector(type_, compiler, **kwargs):
    """Render the physical SQLite vector column type.

    Args:
        type_: Portable vector declaration.
        compiler: SQLAlchemy compiler.
        **kwargs: Compiler options.
    """
    return "BLOB"


@Infrastructure.register("sqlite")
class SQLiteDatabase(Database):
    """Local storage using SQLite and sqlite-vec."""

    config_type = SQLiteConfig
    """Validated local connection settings."""

    def __init__(self, config: SQLiteConfig | None = None, *, tables=None):
        """Create a lazy engine without creating tables or opening connections.

        Args:
            config: Local connection settings; defaults to a local file.
            tables: Business-owned table registry.
        """
        super().__init__(config or SQLiteConfig(), tables=tables)

    def _create_engine(self, config):
        url = make_url(config.url)
        options = {"connect_args": {"timeout": config.timeout, "check_same_thread": False}}
        if url.database in (None, "", ":memory:"):
            options["poolclass"] = StaticPool
        engine = create_engine(url, **options)
        engine.dialect.colspecs = {**engine.dialect.colspecs, Vector: SQLiteVector}

        @event.listens_for(engine, "do_connect")
        def prepare_directory(dialect, record, args, params):
            if url.database not in (None, "", ":memory:"):
                Path(url.database).parent.mkdir(parents=True, exist_ok=True)

        @event.listens_for(engine, "connect")
        def initialize(connection, record):
            import sqlite_vec

            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
            # SQLAlchemy owns BEGIN, including read-only and DDL transactions.
            connection.isolation_level = None
            cursor = connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
            finally:
                cursor.close()

        @event.listens_for(engine, "begin")
        def begin(connection):
            connection.exec_driver_sql("BEGIN")

        return engine

    def _insert_statement(self, table):
        return insert(table)

    def _distance(self, column, vector):
        return func.vec_distance_cosine(column, literal(vector, type_=column.type))
