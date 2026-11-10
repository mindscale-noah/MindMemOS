"""Database operations, schema types, and concrete backends."""

from .base import Database
from .builtin_backends import PostgresDatabase, SQLiteDatabase
from .config import PostgresConfig, SQLiteConfig
from .tables import TableRegistry
from .typing import Filter, Scope, SearchHit, TableSpec
from .vector import Vector

__all__ = [
    "Database",
    "SQLiteDatabase",
    "PostgresDatabase",
    "SQLiteConfig",
    "PostgresConfig",
    "TableRegistry",
    "TableSpec",
    "Filter",
    "Scope",
    "SearchHit",
    "Vector",
]
