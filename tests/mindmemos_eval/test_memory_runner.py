from __future__ import annotations

from types import SimpleNamespace

import pytest
from mindmemos_eval.memory.base import RunContext
from mindmemos_eval.memory.config import load_runner_config
from mindmemos_eval.memory.identity import RunIdentity
from mindmemos_eval.memory.runner import (
    RequestIdMemoryClient,
    _auth_config_output,
    add_memory_args,
    run_benchmark_matrix,
)


class _NoopAdapter:
    name = "locomo"

    async def run(self, **_kwargs):
        return {"ok": True}


class _MemoryBackend:
    def __init__(self, memory, *, closeable=None):
        self.memory = memory
        self._closeable = closeable
        self.started = False

    async def start(self):
        self.started = True
        return self

    async def aclose(self):
        if self._closeable is not None:
            await self._closeable.aclose()


def _write_config(path):
    path.write_text(
        """
benchmarks:
  locomo:
    dataset: data/locomo.json
    memory_algorithm: vanilla
""".lstrip(),
        encoding="utf-8",
    )


def _write_api_keys(path):
    path.write_text(
        """
api_keys:
  - key_id: key-1
    api_key: dev-api-key
    project_id: proj-1
    memory_algorithm: vanilla
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )


def test_auth_config_output_cli_and_deprecated_alias_share_dest():
    import argparse

    parser = argparse.ArgumentParser()
    add_memory_args(parser)
    required = [
        "--benchmark-config",
        "matrix.yaml",
        "--benchmark-list",
        "locomo",
        "--manifest-output",
        "manifest.jsonl",
    ]

    current = parser.parse_args([*required, "--auth-config-output", "auth.yaml"])
    legacy = parser.parse_args([*required, "--api-key-output", "legacy.yaml"])

    assert current.auth_config_output == "auth.yaml"
    assert legacy.auth_config_output == "legacy.yaml"
    assert not hasattr(current, "api_key_output")


def test_auth_config_output_falls_back_to_legacy_namespace_field():
    assert _auth_config_output(SimpleNamespace(api_key_output="legacy.yaml")) == "legacy.yaml"
    assert _auth_config_output(SimpleNamespace(auth_config_output="", api_key_output="legacy.yaml")) == "legacy.yaml"
    assert (
        _auth_config_output(SimpleNamespace(auth_config_output="current.yaml", api_key_output="legacy.yaml"))
        == "current.yaml"
    )


def test_auth_config_output_help_describes_lifecycle(capsys):
    import argparse

    parser = argparse.ArgumentParser()
    add_memory_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])

    help_text = " ".join(capsys.readouterr().out.split())
    assert "--auth-config-output PATH" in help_text
    assert "--api-key-output PATH" in help_text
    assert "complete benchmark auth config YAML" in help_text
    assert "required for fresh runs" in help_text
    assert "not needed with --reuse-api-key" in help_text
    assert "completely overwritten" in help_text
    assert "deprecated compatibility alias" in help_text
    assert "--memory-connection-mode" not in help_text
    assert "--lite-" not in help_text


def test_dreaming_after_add_can_be_enabled_by_cli_or_runner_yaml(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    add_memory_args(parser)
    required = [
        "--benchmark-config",
        "matrix.yaml",
        "--benchmark-list",
        "memoryagentbench",
        "--manifest-output",
        "manifest.jsonl",
    ]
    cli_args = parser.parse_args([*required, "--dreaming-after-add"])
    assert cli_args.dreaming_after_add is True

    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        """
runner:
  dreaming_after_add: true
benchmarks: {}
""".lstrip(),
        encoding="utf-8",
    )
    assert load_runner_config(config_path).dreaming_after_add is True


@pytest.mark.asyncio
async def test_request_id_memory_client_records_dreaming_stage():
    class _Memory:
        async def dreaming(self, **_kwargs):
            return SimpleNamespace(request_id="dreaming-request")

    identity = RunIdentity(
        benchmark="memoryagentbench",
        run_id="run",
        key_id="key",
        api_key="secret",
        project_id="project",
        memory_algorithm="vanilla",
    )
    ctx = RunContext(identity=identity)

    await RequestIdMemoryClient(_Memory(), ctx).dreaming(mode="sync")

    assert ctx.request_ids["dreaming"] == ["dreaming-request"]
    assert ctx.request_metadata["dreaming-request"]["stage"] == "dreaming"


@pytest.mark.asyncio
async def test_reuse_api_key_rejects_multiple_benchmarks(tmp_path):
    config_path = tmp_path / "memory_eval.yaml"
    config_path.write_text(
        """
benchmarks:
  locomo:
    dataset: data/locomo.json
    memory_algorithm: vanilla
  longmemeval:
    dataset: data/longmemeval.json
    memory_algorithm: vanilla
""".lstrip(),
        encoding="utf-8",
    )
    api_key_path = tmp_path / "api_keys.yaml"
    api_key_path.write_text(
        """
api_keys:
  - key_id: key-1
    api_key: dev-api-key
    project_id: proj-1
    memory_algorithm: vanilla
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        benchmark_config=str(config_path),
        benchmark_list="locomo,longmemeval",
        manifest_output=str(tmp_path / "manifest.jsonl"),
        api_key_output=str(tmp_path / "generated_api_keys.yaml"),
        reuse_api_key=str(api_key_path),
    )

    with pytest.raises(ValueError, match="--reuse-api-key can only be used with exactly one benchmark"):
        await run_benchmark_matrix(
            args,
            adapters={"locomo": object(), "longmemeval": object()},
        )


@pytest.mark.asyncio
async def test_reuse_api_key_logs_with_standard_logger_kwargs(tmp_path, monkeypatch):
    config_path = tmp_path / "memory_eval.yaml"
    _write_config(config_path)
    api_key_path = tmp_path / "api_keys.yaml"
    _write_api_keys(api_key_path)
    args = SimpleNamespace(
        benchmark_config=str(config_path),
        benchmark_list="locomo",
        manifest_output=str(tmp_path / "manifest.jsonl"),
        api_key_output=str(tmp_path / "generated_api_keys.yaml"),
        reuse_api_key=str(api_key_path),
    )

    from mindmemos_eval.memory import runner

    monkeypatch.setattr(runner.logger, "level", runner.logging.INFO)

    async def _noop_reset(cfg, project_id):
        return {}

    monkeypatch.setattr(runner, "reset_project", _noop_reset)

    async def memory_client_factory(_identity):
        return _MemoryBackend(object())

    manifests = await run_benchmark_matrix(
        args,
        adapters={"locomo": _NoopAdapter()},
        memory_client_factory=memory_client_factory,
        answer_llm_factory=lambda: object(),
        judge_llm_factory=lambda: object(),
    )

    assert manifests[0].eval_result == {"ok": True}
    assert manifests[0].api_key_file == str(api_key_path)


class _RecordingMemory:
    """Fake memory client recording add/search user_ids."""

    def __init__(self):
        self.add_user_ids: list[str] = []
        self.search_user_ids: list[str] = []

    async def add(self, messages, *, user_id=None, **_kwargs):
        self.add_user_ids.append(user_id)
        return SimpleNamespace(request_id="add-1")

    async def search(self, query, *, user_id=None, **_kwargs):
        self.search_user_ids.append(user_id)
        return SimpleNamespace(request_id="search-1", results=[])


class _ScopeAdapter:
    """Adapter that adds (when add=True) then searches a LocomoEnv-derived user_id.

    Mirrors ``LocomoEnv._conv_user_id`` to prove the scope is stable across runs (no
    run_id suffix), which is what lets ``--reuse-api-key + --no-add`` read a prior run's
    memories.
    """

    name = "locomo"

    def __init__(self):
        self.added: list[str] = []
        self.searched: list[str] = []

    async def run(self, *, memory, answer_llm, judge_llm, ctx, bench_config, args):
        from mindmemos_eval.memory.envs.locomo.env import LocomoEnv

        env = LocomoEnv(memory, answer_llm=answer_llm, judge_llm=judge_llm, scorer=object())
        uid = env._conv_user_id(0)
        runner_cfg = getattr(args, "runner_config", None)
        if runner_cfg is not None and runner_cfg.add:
            await memory.add([{"role": "user", "content": "hi"}], user_id=uid, mode="sync", session_id=uid)
            self.added.append(uid)
        await memory.search("hi", user_id=uid)
        self.searched.append(uid)
        return {"user_id": uid}


@pytest.mark.asyncio
async def test_ingest_then_reuse_no_add_reads_same_scope(tmp_path, monkeypatch):
    """End-to-end: a fresh ingest run and a reuse+no-add run share the same user_id,
    no-add skips both add and the pre-add reset, and reuse needs no --auth-config-output."""
    from mindmemos_eval.memory import runner as runner_mod

    reset_calls: list[str] = []

    async def _spy_reset(cfg, project_id):
        reset_calls.append(project_id)
        return {}

    monkeypatch.setattr(runner_mod, "reset_project", _spy_reset)

    config_path = tmp_path / "memory_eval.yaml"
    _write_config(config_path)
    api_keys_path = tmp_path / "api_keys.yaml"

    # Run 1: fresh run with add -> writes api_keys.yaml, adds under conv_0, resets first.
    adapter1 = _ScopeAdapter()
    mem1 = _RecordingMemory()

    async def mem_factory1(_identity):
        return _MemoryBackend(mem1)

    args1 = SimpleNamespace(
        benchmark_config=str(config_path),
        benchmark_list="locomo",
        manifest_output=str(tmp_path / "m1.jsonl"),
        auth_config_output=str(api_keys_path),
        reuse_api_key=None,
        add=True,
        skip_clean=False,
    )
    manifests1 = await run_benchmark_matrix(
        args1,
        adapters={"locomo": adapter1},
        memory_client_factory=mem_factory1,
        answer_llm_factory=lambda: object(),
        judge_llm_factory=lambda: object(),
    )
    assert adapter1.added == ["conv_0"]
    assert adapter1.searched == ["conv_0"]
    assert len(reset_calls) == 1  # add stage cleared the project first
    assert api_keys_path.exists()
    assert manifests1[0].api_key_file == str(api_keys_path)

    # Run 2: reuse + no-add, no --auth-config-output -> same user_id, no add, no reset.
    adapter2 = _ScopeAdapter()
    mem2 = _RecordingMemory()

    async def mem_factory2(_identity):
        return _MemoryBackend(mem2)

    args2 = SimpleNamespace(
        benchmark_config=str(config_path),
        benchmark_list="locomo",
        manifest_output=str(tmp_path / "m2.jsonl"),
        auth_config_output=None,
        reuse_api_key=str(api_keys_path),
        add=False,
        skip_clean=False,
    )
    await run_benchmark_matrix(
        args2,
        adapters={"locomo": adapter2},
        memory_client_factory=mem_factory2,
        answer_llm_factory=lambda: object(),
        judge_llm_factory=lambda: object(),
    )
    assert adapter2.added == []  # no-add skipped ingestion
    assert adapter2.searched == ["conv_0"]  # same scope as run 1
    assert len(reset_calls) == 1  # no-add did not clear


@pytest.mark.asyncio
async def test_fresh_http_run_requires_auth_config_output(tmp_path, monkeypatch):
    from mindmemos_eval.memory import runner as runner_mod

    async def _noop_reset(cfg, project_id):
        return {}

    monkeypatch.setattr(runner_mod, "reset_project", _noop_reset)

    config_path = tmp_path / "memory_eval.yaml"
    _write_config(config_path)
    args = SimpleNamespace(
        benchmark_config=str(config_path),
        benchmark_list="locomo",
        manifest_output=str(tmp_path / "m.jsonl"),
        auth_config_output=None,
        reuse_api_key=None,
        add=True,
        skip_clean=False,
    )

    async def memory_client_factory(_identity):
        return _MemoryBackend(object())

    with pytest.raises(ValueError, match="--auth-config-output is required for fresh runs"):
        await run_benchmark_matrix(
            args,
            adapters={"locomo": _NoopAdapter()},
            memory_client_factory=memory_client_factory,
            answer_llm_factory=lambda: object(),
            judge_llm_factory=lambda: object(),
        )


@pytest.mark.asyncio
async def test_reset_failure_aborts_benchmark_and_closes_backend(tmp_path, monkeypatch, caplog):
    from mindmemos_eval.memory import runner as runner_mod
    from mindmemos_eval.memory.db_reset import ProjectResetError

    config_path = tmp_path / "memory_eval.yaml"
    _write_config(config_path)
    manifest_path = tmp_path / "manifest.jsonl"
    adapter_calls: list[str] = []

    class _Adapter:
        name = "locomo"

        async def run(self, **_kwargs):
            adapter_calls.append("run")
            return {"ok": True}

    class _CloseTracker:
        closed = False

        async def aclose(self):
            self.closed = True

    close_tracker = _CloseTracker()

    async def memory_client_factory(_identity):
        return _MemoryBackend(object(), closeable=close_tracker)

    async def fail_reset(_cfg, project_id):
        raise ProjectResetError(
            project_id=project_id,
            store="qdrant",
            operation="delete",
            resource="memory_item_v1",
            reason="PermissionError: unauthorized",
        )

    monkeypatch.setattr(runner_mod, "reset_project", fail_reset)
    legacy_output_path = tmp_path / "api_keys.yaml"
    args = SimpleNamespace(
        benchmark_config=str(config_path),
        benchmark_list="locomo",
        manifest_output=str(manifest_path),
        api_key_output=str(legacy_output_path),
        reuse_api_key=None,
        add=True,
        skip_clean=False,
    )

    with (
        caplog.at_level("CRITICAL", logger="mindmemos_eval.memory.runner"),
        pytest.raises(ProjectResetError, match="unauthorized"),
    ):
        await run_benchmark_matrix(
            args,
            adapters={"locomo": _Adapter()},
            memory_client_factory=memory_client_factory,
            answer_llm_factory=lambda: object(),
            judge_llm_factory=lambda: object(),
        )

    assert adapter_calls == []
    assert close_tracker.closed is True
    assert legacy_output_path.exists()
    assert not manifest_path.exists()
    assert any(
        "benchmark_aborted_database_reset_failed" in record.message
        and "qdrant" in record.message
        and "memory_item_v1" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_reset_project_clears_qdrant_and_neo4j(monkeypatch):
    from mindmemos_eval.memory import db_reset

    cfg = db_reset.ResetConfig(
        qdrant_url="http://x",
        qdrant_api_key=None,
        neo4j_uri="bolt://x",
        neo4j_username="u",
        neo4j_password="p",
        neo4j_database="neo4j",
        collections=("memory_item_v1",),
    )

    qdrant_deletes: list[tuple] = []

    class _FakeQdrant:
        def __init__(self):
            self.counts = iter((2, 0))

        async def count(self, *, collection_name, count_filter, exact):
            return SimpleNamespace(count=next(self.counts))

        async def delete(self, *, collection_name, points_selector, wait):
            qdrant_deletes.append((collection_name, points_selector))
            assert wait is True

        async def close(self):
            pass

    monkeypatch.setattr(db_reset, "AsyncQdrantClient", lambda **kw: _FakeQdrant())

    neo4j_queries: list[str] = []

    class _Rec:
        def __init__(self, total):
            self.total = total

        def __getitem__(self, key):
            return self.total

    class _Result:
        def __init__(self, total):
            self.records = [_Rec(total)]

    class _Driver:
        def __init__(self):
            self.counts = iter((4, 0))

        async def execute_query(self, query, params, *, routing_=None, database_=None):
            neo4j_queries.append(query)
            if "RETURN count" in query:
                return _Result(next(self.counts))
            return _Result(0)

        async def close(self):
            pass

    class _GraphDatabase:
        @staticmethod
        def driver(uri, *, auth):
            return _Driver()

    import sys

    fake_neo4j = type(
        "neo4j",
        (),
        {
            "AsyncGraphDatabase": _GraphDatabase,
            "RoutingControl": type("RC", (), {"READ": "READ", "WRITE": "WRITE"}),
        },
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    counts = await db_reset.reset_project(cfg, "proj-123")

    assert counts["qdrant:memory_item_v1"] == 2
    assert counts["neo4j:nodes"] == 4
    assert len(qdrant_deletes) == 1
    coll, selector = qdrant_deletes[0]
    assert coll == "memory_item_v1"
    assert selector.filter.must[0].key == "project_id"
    assert selector.filter.must[0].match.value == "proj-123"
    assert sum("RETURN count" in q for q in neo4j_queries) == 2
    assert any("DETACH DELETE" in q for q in neo4j_queries)


@pytest.mark.asyncio
async def test_reset_project_raises_when_qdrant_delete_fails(monkeypatch):
    from mindmemos_eval.memory import db_reset

    cfg = db_reset.ResetConfig(collections=("memory_item_v1",))
    closed = False

    class _FakeQdrant:
        async def count(self, *, collection_name, count_filter, exact):
            return SimpleNamespace(count=2)

        async def delete(self, *, collection_name, points_selector, wait):
            raise PermissionError("unauthorized")

        async def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(db_reset, "AsyncQdrantClient", lambda **_kwargs: _FakeQdrant())

    with pytest.raises(db_reset.ProjectResetError) as exc_info:
        await db_reset.reset_project(cfg, "proj-123")

    exc = exc_info.value
    assert exc.store == "qdrant"
    assert exc.operation == "delete"
    assert exc.resource == "memory_item_v1"
    assert "unauthorized" in exc.reason
    assert closed is True


@pytest.mark.asyncio
async def test_reset_project_raises_when_qdrant_still_contains_project_data(monkeypatch):
    from mindmemos_eval.memory import db_reset

    cfg = db_reset.ResetConfig(collections=("memory_item_v1",))

    class _FakeQdrant:
        def __init__(self):
            self.counts = iter((2, 1))

        async def count(self, *, collection_name, count_filter, exact):
            return SimpleNamespace(count=next(self.counts))

        async def delete(self, *, collection_name, points_selector, wait):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(db_reset, "AsyncQdrantClient", lambda **_kwargs: _FakeQdrant())

    with pytest.raises(db_reset.ProjectResetError) as exc_info:
        await db_reset.reset_project(cfg, "proj-123")

    exc = exc_info.value
    assert exc.operation == "verify_empty"
    assert exc.resource == "memory_item_v1"
    assert "1 project-scoped points remain" in exc.reason


def test_memory_cli_returns_one_when_database_reset_fails(monkeypatch):
    from mindmemos_eval.memory.db_reset import ProjectResetError

    from mindmemos_eval import cli

    class _Parser:
        def parse_args(self, _argv):
            return SimpleNamespace(command="memory")

    async def fail_run(_args):
        raise ProjectResetError(
            project_id="proj-123",
            store="neo4j",
            operation="verify_empty",
            resource="neo4j",
            reason="1 project-scoped node remains after delete",
        )

    monkeypatch.setattr(cli, "build_arg_parser", lambda: _Parser())
    monkeypatch.setattr(cli, "run_benchmark_matrix", fail_run)

    assert cli.main([]) == 1


# ---------- resolve_collections (dynamic Qdrant collection names) ----------


def _write_server_config(path, qdrant=None):
    import yaml

    data = {"database": {}}
    if qdrant is not None:
        data["database"]["qdrant"] = qdrant
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_resolve_collections_none_returns_defaults():
    from mindmemos_eval.memory.db_reset import DEFAULT_COLLECTIONS, resolve_collections

    assert resolve_collections(None) == DEFAULT_COLLECTIONS


def test_resolve_collections_reads_custom_names(tmp_path):
    from mindmemos_eval.memory.db_reset import resolve_collections

    cfg = tmp_path / "server.yaml"
    _write_server_config(
        cfg,
        {
            "memory_collection": "custom_memory",
            "entity_collection": "custom_entity",
            "source_collection": "custom_source",
            "add_record_collection": "custom_add",
            "schema_add_buffer_collection": "custom_buffer",
            "search_record_collection": "custom_search",
        },
    )
    assert resolve_collections(cfg) == (
        "custom_memory",
        "custom_entity",
        "custom_source",
        "custom_add",
        "custom_buffer",
        "custom_search",
    )


def test_resolve_collections_falls_back_missing_fields(tmp_path):
    from mindmemos_eval.memory.db_reset import DEFAULT_COLLECTIONS, resolve_collections

    cfg = tmp_path / "server.yaml"
    # Mirror dev.example.yaml: only memory/entity/source are configured.
    _write_server_config(
        cfg,
        {
            "memory_collection": "m",
            "entity_collection": "e",
            "source_collection": "s",
        },
    )
    assert resolve_collections(cfg) == (
        "m",
        "e",
        "s",
        DEFAULT_COLLECTIONS[3],
        DEFAULT_COLLECTIONS[4],
        DEFAULT_COLLECTIONS[5],
    )


def test_resolve_collections_missing_file_falls_back(tmp_path, caplog):
    from mindmemos_eval.memory.db_reset import DEFAULT_COLLECTIONS, resolve_collections

    missing = tmp_path / "nope.yaml"
    with caplog.at_level("WARNING", logger="mindmemos_eval.memory.db_reset"):
        result = resolve_collections(missing)
    assert result == DEFAULT_COLLECTIONS
    assert any("resolve_qdrant_collections_failed" in rec.message for rec in caplog.records)


def test_resolve_collections_non_mapping_falls_back(tmp_path, caplog):
    from mindmemos_eval.memory.db_reset import DEFAULT_COLLECTIONS, resolve_collections

    cfg = tmp_path / "server.yaml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="mindmemos_eval.memory.db_reset"):
        result = resolve_collections(cfg)
    assert result == DEFAULT_COLLECTIONS
    assert any("resolve_qdrant_collections_failed" in rec.message for rec in caplog.records)
