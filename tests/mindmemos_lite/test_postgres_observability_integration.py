"""Run with a disposable PostgreSQL database; no pgvector extension is needed."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import psycopg
import pytest
from mindmemos_lite.config import PostgresObservabilityConfig
from mindmemos_lite.infra.observability import PostgresObservabilityBackend
from mindmemos_lite.infra.observability.constants import LLM_CALLS_TABLE, SPAN_EVENTS_TABLE, SPANS_TABLE, TRACES_TABLE
from psycopg import sql
from psycopg.rows import dict_row
from test_postgres_observability import make_span

TEST_DSN_ENV = "MINDMEMOS_TEST_OBSERVABILITY_POSTGRES_DSN"


@pytest.fixture
def postgres_store():
    dsn = os.getenv(TEST_DSN_ENV)
    if not dsn:
        pytest.skip(f"set {TEST_DSN_ENV} to run PostgreSQL observability integration tests")
    config = PostgresObservabilityConfig(dsn=dsn, schema=f"trace_test_{uuid4().hex}")
    backend = PostgresObservabilityBackend(config, retention_days=None)
    try:
        with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as connection:
            yield config, backend, connection
    finally:
        backend.close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(config.schema)))


def rows(connection, config, table):
    return connection.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(config.schema, table))).fetchall()


def test_roundtrip_replay_and_out_of_order_spans(postgres_store):
    config, backend, connection = postgres_store
    root = make_span()
    child = make_span(span_id="c" * 16, parent_span_id=root.span_id, end_time_ns=root.end_time_ns + 100)
    backend.write_spans([child])
    backend.write_spans([root])
    backend.write_spans([root])
    trace = rows(connection, config, TRACES_TABLE)[0]
    assert trace["root_span_id"] == root.span_id
    assert trace["end_time_ns"] == child.end_time_ns
    assert trace["attributes_json"] == {"run_id": "run-1"}
    assert len(rows(connection, config, SPANS_TABLE)) == 2
    assert len(rows(connection, config, SPAN_EVENTS_TABLE)) == 2
    assert len(rows(connection, config, LLM_CALLS_TABLE)) == 2
    assert rows(connection, config, SPAN_EVENTS_TABLE)[0]["attributes_json"] == {"value": "测试"}
    backend.write_spans([replace(root, attributes={}, events=())])
    assert len(rows(connection, config, SPAN_EVENTS_TABLE)) == 1
    assert len(rows(connection, config, LLM_CALLS_TABLE)) == 1
    backend.force_flush()


def test_failed_batch_rolls_back_and_next_batch_recovers(postgres_store):
    config, backend, connection = postgres_store
    # PostgreSQL rejects NUL in text, after earlier spans have been queued.
    invalid = make_span(span_id="z" * 16, name="invalid\x00name")
    with pytest.raises((psycopg.Error, ValueError)):
        backend.write_spans([make_span(), invalid])
    for table in (TRACES_TABLE, SPANS_TABLE, SPAN_EVENTS_TABLE, LLM_CALLS_TABLE):
        assert rows(connection, config, table) == []
    backend.write_spans([make_span()])
    assert len(rows(connection, config, SPANS_TABLE)) == 1


def test_independent_schemas_on_same_database(postgres_store):
    config, backend, connection = postgres_store
    second_config = replace(config, schema=f"trace_test_{uuid4().hex}")
    second = PostgresObservabilityBackend(second_config, retention_days=None)
    try:
        backend.write_spans([make_span()])
        assert rows(connection, second_config, SPANS_TABLE) == []
        second.write_spans([make_span(name="other")])
        assert rows(connection, config, SPANS_TABLE)[0]["name"] == "llm.chat.provider"
        assert rows(connection, second_config, SPANS_TABLE)[0]["name"] == "other"
        second.close()
        backend.write_spans([make_span(span_id="d" * 16)])
    finally:
        second.close()
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(second_config.schema)))


def test_multiple_writers_and_startup_retention(postgres_store):
    config, backend, connection = postgres_store
    second = PostgresObservabilityBackend(config, retention_days=None)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(store.write_spans, [make_span(span_id=f"{index:016x}")])
                for index, store in enumerate((backend, second))
            ]
            for future in futures:
                future.result()
        assert len(rows(connection, config, SPANS_TABLE)) == 2
    finally:
        second.close()
    cleaner = PostgresObservabilityBackend(config, retention_days=1)
    cleaner.close()
    for table in (TRACES_TABLE, SPANS_TABLE, SPAN_EVENTS_TABLE, LLM_CALLS_TABLE):
        assert rows(connection, config, table) == []


def test_concurrent_initialization_of_same_schema(postgres_store):
    config, _, connection = postgres_store
    concurrent_config = replace(config, schema=f"trace_test_{uuid4().hex}")
    stores = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(PostgresObservabilityBackend, concurrent_config, retention_days=None) for _ in range(2)
            ]
            for future in futures:
                stores.append(future.result())
        for store in stores:
            store.write_spans([make_span()])
        assert len(rows(connection, concurrent_config, SPANS_TABLE)) == 1
    finally:
        for store in stores:
            store.close()
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(concurrent_config.schema)))
