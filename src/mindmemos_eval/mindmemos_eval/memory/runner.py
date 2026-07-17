"""Benchmark matrix runner."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mindmemos_sdk.memory import AsyncMemoryClient
from mindmemos_sdk.transport import AsyncHttpTransport

from mindmemos_eval.memory.envs.locomo import LocomoAdapter
from mindmemos_eval.memory.envs.longmemeval.adapter import LongMemEvalAdapter
from mindmemos_eval.memory.envs.memoryagentbench.adapter import MemoryAgentBenchAdapter
from mindmemos_eval.memory.envs.personamem import PersonaMemAdapter

from ..llm import LLMClient, LLMConfig
from .base import BenchmarkAdapter, BenchmarkSpec, RunContext, RunnerConfig
from .config import _merged_runner_config, _option, load_benchmark_specs, validate_memory_algorithm
from .db_reset import ResetConfig, resolve_collections, reset_project
from .identity import RunIdentity, new_identity, write_api_keys
from .manifest import BenchmarkRunManifest, write_manifests

logger = logging.getLogger("mindmemos_eval.memory.runner")


def _load_existing_identity(path: str, *, benchmark: str) -> RunIdentity:
    """Load the first enabled api_key entry from an existing api_keys YAML.

    Reuses the api_key/project_id from the prior run so ``--no-add`` can read its
    memories. A fresh ``run_id`` is generated as a run label only - it no longer
    participates in memory scoping, which is now handled at the project_id level
    (each run gets its own project_id, and the add stage clears it first).
    """
    import secrets
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("api_keys") or []
    for entry in entries:
        if not entry.get("enabled", True):
            continue
        suffix = f"{datetime.now(UTC):%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"
        return RunIdentity(
            benchmark=benchmark,
            run_id=suffix,
            key_id=str(entry.get("key_id", "")),
            api_key=str(entry.get("api_key", "")),
            project_id=str(entry.get("project_id", "")),
            memory_algorithm=str(entry.get("memory_algorithm", "")),
            profile=None,
            project_override_config=entry.get("project_override_config"),
        )
    raise ValueError(f"No enabled api_key found in {path}")


def add_memory_args(parser: argparse.ArgumentParser) -> None:
    """Register memory benchmark matrix CLI arguments."""
    parser.add_argument(
        "--benchmark-config",
        required=True,
        metavar="PATH",
        help="Path to the matrix YAML that defines runner defaults, algorithm profiles, and benchmark specs; must exist and parse as a mapping.",
    )
    parser.add_argument(
        "--benchmark-list",
        required=True,
        metavar="NAMES",
        help="Comma-separated benchmark names to run from the config, for example locomo,longmemeval; each name must have a spec and adapter.",
    )
    parser.add_argument(
        "--manifest-output",
        required=True,
        metavar="PATH",
        help="Path to the JSONL manifest written for this run; parent directories are created if needed.",
    )
    parser.add_argument(
        "--api-key-output",
        required=False,
        metavar="PATH",
        help="Path to the generated api_keys YAML consumed by the FastAPI server; it should match the server "
        "auth.api_key_file. Required for fresh runs; optional (unused) when --reuse-api-key is set.",
    )
    parser.add_argument(
        "--algorithm",
        metavar="NAME",
        type=validate_memory_algorithm,
        help="Global algorithm profile name applied to every benchmark; currently validated to vanilla or schema.",
    )
    parser.add_argument(
        "--memory-algorithm",
        metavar="NAME",
        type=validate_memory_algorithm,
        help="Legacy override for only the memory_algorithm binding; currently validated to vanilla or schema.",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="Base URL of the running MindMemOS FastAPI service used for add/search requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        metavar="SECONDS",
        help="HTTP timeout for memory API calls in seconds; must be parseable as a float.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of dataset items to run per benchmark; must be parseable as an integer.",
    )
    parser.add_argument(
        "--session-limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of sessions added per LongMemEval sample; must be parseable as an integer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        metavar="N",
        help="Number of memories requested for each search call; must be parseable as an integer.",
    )
    parser.add_argument(
        "--search-strategy",
        metavar="MODE",
        help="Public search mode sent to the memory API, usually fast or agentic; benchmark aliases vanilla/schema map to fast.",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether benchmark searches request reranking; use --rerank or --no-rerank.",
    )
    parser.add_argument(
        "--max-conv-concurrency",
        type=int,
        metavar="N",
        help="Maximum concurrent conversation/sample build tasks; must be parseable as an integer.",
    )
    parser.add_argument(
        "--max-qa-concurrency",
        type=int,
        metavar="N",
        help="Maximum concurrent question-answer tasks; must be parseable as an integer.",
    )
    parser.add_argument(
        "--max-search-concurrency",
        type=int,
        metavar="N",
        help="Maximum concurrent memory search tasks; must be parseable as an integer.",
    )
    parser.add_argument(
        "--max-score-concurrency",
        type=int,
        metavar="N",
        help="Maximum concurrent judge/scoring tasks; must be parseable as an integer.",
    )
    parser.add_argument(
        "--judge-runs",
        type=int,
        metavar="N",
        help="Number of independent judge runs per question; majority vote decides the final judge result.",
    )
    parser.add_argument(
        "--add",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to execute the memory ingestion stage before answering; use --add or --no-add.",
    )
    parser.add_argument(
        "--score",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to execute the judge/scoring stage after answering; use --score or --no-score.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to show benchmark progress bars/log progress; use --show-progress or --no-show-progress.",
    )

    parser.add_argument(
        "--reuse-api-key",
        metavar="PATH",
        default=None,
        help="Path to an existing api_keys YAML file. When set, skips generating fresh identities and reuses the "
        "first api_key entry from this file instead. Use with --no-add to rerun evaluation against previously "
        "added memories without overwriting the server's api_keys.yaml.",
    )

    parser.add_argument(
        "--qdrant-url",
        metavar="URL",
        default=None,
        help="Qdrant HTTP endpoint for the add-stage project cleanup. Defaults to $MINDMEMOS_QDRANT_URL or "
        "http://localhost:6333. Only used when the add stage runs (not with --no-add).",
    )
    parser.add_argument(
        "--qdrant-api-key",
        metavar="KEY",
        default=None,
        help="Qdrant API key for project cleanup; usually unset for local dev. Defaults to $MINDMEMOS_QDRANT_API_KEY.",
    )
    parser.add_argument(
        "--neo4j-uri",
        metavar="URI",
        default=None,
        help="Neo4j Bolt URI for the add-stage project cleanup. Defaults to $MINDMEMOS_NEO4J_URI or "
        "bolt://localhost:7687.",
    )
    parser.add_argument(
        "--neo4j-username",
        metavar="USER",
        default=None,
        help="Neo4j username for project cleanup. Defaults to $MINDMEMOS_NEO4J_USERNAME or neo4j.",
    )
    parser.add_argument(
        "--neo4j-password",
        metavar="PASSWORD",
        default=None,
        help="Neo4j password for project cleanup. Defaults to $MINDMEMOS_NEO4J_PASSWORD.",
    )
    parser.add_argument(
        "--neo4j-database",
        metavar="NAME",
        default=None,
        help="Neo4j database for project cleanup. Defaults to $MINDMEMOS_NEO4J_DATABASE or neo4j.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        default=False,
        help="Skip clearing the run's project_id from Qdrant/Neo4j before the add stage. By default the add stage "
        "starts from a clean project; --no-add never clears.",
    )

    parser.add_argument(
        "--server-config",
        metavar="PATH",
        default=None,
        help="Path to the MindMemOS server yaml config (e.g. config/mindmemos/dev.yaml). Used to read the actual "
        "Qdrant collection names so the add-stage cleanup clears exactly what the server writes, even when "
        "collections are renamed away from the _v1 defaults. When omitted, the default collection names are used.",
    )

    parser.add_argument(
        "--llm-model",
        metavar="MODEL",
        help="Default LLM model name used when answer or judge specific models are not set.",
    )
    parser.add_argument(
        "--llm-api-key",
        metavar="KEY",
        help="Default LLM API key used when answer or judge specific keys are not set.",
    )
    parser.add_argument(
        "--llm-base-url",
        metavar="URL",
        help="Default OpenAI-compatible LLM base URL used when answer or judge specific URLs are not set.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        metavar="FLOAT",
        help="Default sampling temperature for answer and judge LLM calls; must be parseable as a float.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        metavar="N",
        help="Default maximum output tokens for answer and judge LLM calls; must be parseable as an integer.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        metavar="SECONDS",
        help="Default timeout for answer and judge LLM calls in seconds; must be parseable as a float.",
    )
    parser.add_argument(
        "--answer-llm-model",
        default=None,
        metavar="MODEL",
        help="LLM model name used only for answer generation; falls back to --llm-model when unset.",
    )
    parser.add_argument(
        "--answer-llm-api-key",
        default=None,
        metavar="KEY",
        help="LLM API key used only for answer generation; falls back to --llm-api-key when unset.",
    )
    parser.add_argument(
        "--answer-llm-base-url",
        default=None,
        metavar="URL",
        help="OpenAI-compatible base URL used only for answer generation; falls back to --llm-base-url when unset.",
    )
    parser.add_argument(
        "--judge-llm-model",
        default=None,
        metavar="MODEL",
        help="LLM model name used only for judge/scoring calls; falls back to --llm-model when unset.",
    )
    parser.add_argument(
        "--judge-llm-api-key",
        default=None,
        metavar="KEY",
        help="LLM API key used only for judge/scoring calls; falls back to --llm-api-key when unset.",
    )
    parser.add_argument(
        "--judge-llm-base-url",
        default=None,
        metavar="URL",
        help="OpenAI-compatible base URL used only for judge/scoring calls; falls back to --llm-base-url when unset.",
    )


class RequestIdMemoryClient:
    """Thin async memory wrapper that records server-generated request ids."""

    def __init__(self, inner: AsyncMemoryClient, ctx: RunContext) -> None:
        self._inner = inner
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def add(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``add`` and record the request id returned by the server."""
        result = await self._inner.add(*args, **kwargs)
        self._ctx.record_request_id("add", getattr(result, "request_id", None))
        return result

    async def search(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``search`` and record the request id returned by the server."""
        result = await self._inner.search(*args, **kwargs)
        self._ctx.record_request_id("search", getattr(result, "request_id", None))
        return result


class NotImplementedAdapter:
    """Placeholder adapter for planned benchmark datasets."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def run(
        self,
        *,
        memory: Any,
        answer_llm: LLMClient,
        judge_llm: LLMClient,
        ctx: RunContext,
        bench_config: BenchmarkSpec,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Raise a clear phase-1 placeholder error."""
        del memory, answer_llm, judge_llm, ctx, bench_config, args
        raise NotImplementedError(f"{self.name} adapter is reserved for a later phase")


def default_adapters() -> dict[str, BenchmarkAdapter]:
    """Return the phase-1 adapter registry."""
    return {
        "locomo": LocomoAdapter(),
        "longmemeval": LongMemEvalAdapter(),
        "memoryagentbench": MemoryAgentBenchAdapter(),
        "personamem": PersonaMemAdapter(),
        "persona": NotImplementedAdapter("persona"),
    }


def _parse_benchmark_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_memory_client(
    base_url: str, api_key: str, timeout_seconds: float
) -> tuple[AsyncMemoryClient, AsyncHttpTransport]:
    transport = AsyncHttpTransport(base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
    return AsyncMemoryClient(transport), transport


def _build_llm_client(config: RunnerConfig, *, prefix: str) -> LLMClient:
    model = getattr(config, f"{prefix}_llm_model") or config.llm_model
    api_key = getattr(config, f"{prefix}_llm_api_key") or config.llm_api_key
    base_url = getattr(config, f"{prefix}_llm_base_url") or config.llm_base_url
    return LLMClient(
        LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            timeout=config.llm_timeout,
        )
    )


def _build_reset_config(args: argparse.Namespace) -> ResetConfig:
    """Build a :class:`ResetConfig` from CLI overrides, falling back to env-var defaults."""
    env = ResetConfig()  # picks up MINDMEMOS_* env vars
    return ResetConfig(
        qdrant_url=_option(args, "qdrant_url") or env.qdrant_url,
        qdrant_api_key=_option(args, "qdrant_api_key") or env.qdrant_api_key,
        neo4j_uri=_option(args, "neo4j_uri") or env.neo4j_uri,
        neo4j_username=_option(args, "neo4j_username") or env.neo4j_username,
        neo4j_password=_option(args, "neo4j_password") or env.neo4j_password,
        neo4j_database=_option(args, "neo4j_database") or env.neo4j_database,
        collections=resolve_collections(_option(args, "server_config")),
    )


async def run_benchmark_matrix(
    args: argparse.Namespace,
    *,
    adapters: dict[str, BenchmarkAdapter] | None = None,
    memory_client_factory: Callable[[RunIdentity], Awaitable[tuple[Any, Any]]] | None = None,
    answer_llm_factory: Callable[[], LLMClient] | None = None,
    judge_llm_factory: Callable[[], LLMClient] | None = None,
) -> list[BenchmarkRunManifest]:
    """Run configured benchmarks and write api-key and manifest outputs."""
    specs = load_benchmark_specs(
        args.benchmark_config,
        algorithm_override=_option(args, "algorithm"),
        memory_algorithm_override=_option(args, "memory_algorithm"),
    )
    runner = _merged_runner_config(args)
    setattr(args, "runner_config", runner)
    benchmark_names = _parse_benchmark_list(args.benchmark_list)
    registry = adapters or default_adapters()

    missing_specs = [name for name in benchmark_names if name not in specs]
    if missing_specs:
        raise ValueError(f"benchmark(s) not found in config: {', '.join(missing_specs)}")
    missing_adapters = [name for name in benchmark_names if name not in registry]
    if missing_adapters:
        raise ValueError(f"benchmark adapter(s) not registered: {', '.join(missing_adapters)}")

    reuse_path = _option(args, "reuse_api_key")
    if reuse_path:
        if len(benchmark_names) != 1:
            raise ValueError("--reuse-api-key can only be used with exactly one benchmark")
        existing = _load_existing_identity(reuse_path, benchmark=benchmark_names[0])
        identities = [existing]
        setattr(args, "_reused_key_file", reuse_path)
        logger.info(
            "reusing existing api key project_id=%s api_key_prefix=%s",
            existing.project_id,
            existing.api_key[:40],
        )
    else:
        if not _option(args, "api_key_output"):
            raise ValueError("--api-key-output is required for fresh runs (omit it only with --reuse-api-key)")
        identities = [
            new_identity(
                name,
                specs[name].memory_algorithm,
                profile=specs[name].profile,
                project_override_config=specs[name].project_override_config,
            )
            for name in benchmark_names
        ]
        write_api_keys(args.api_key_output, identities)

    reset_cfg = _build_reset_config(args)
    skip_clean = bool(_option(args, "skip_clean"))

    manifests: list[BenchmarkRunManifest] = []
    for identity in identities:
        spec = specs[identity.benchmark]
        adapter = registry[identity.benchmark]
        ctx = RunContext(identity=identity)
        started_at = datetime.now(UTC)
        transport: Any | None = None

        if memory_client_factory is None:
            memory, transport = _build_memory_client(runner.base_url, identity.api_key, runner.timeout_seconds)
        else:
            memory, transport = await memory_client_factory(identity)

        wrapped_memory = RequestIdMemoryClient(memory, ctx)
        answer_llm = answer_llm_factory() if answer_llm_factory else _build_llm_client(runner, prefix="answer")
        judge_llm = judge_llm_factory() if judge_llm_factory else _build_llm_client(runner, prefix="judge")
        if runner.add and not skip_clean:
            logger.info("resetting project before add project_id=%s", identity.project_id)
            await reset_project(reset_cfg, identity.project_id)
        try:
            eval_result = await adapter.run(
                memory=wrapped_memory,
                answer_llm=answer_llm,
                judge_llm=judge_llm,
                ctx=ctx,
                bench_config=spec,
                args=args,
            )
        finally:
            if transport is not None and hasattr(transport, "aclose"):
                await transport.aclose()

        finished_at = datetime.now(UTC)
        manifest = BenchmarkRunManifest(
            benchmark=identity.benchmark,
            run_id=identity.run_id,
            key_id=identity.key_id,
            project_id=identity.project_id,
            memory_algorithm=identity.memory_algorithm,
            api_key_file=str(getattr(args, "_reused_key_file", args.api_key_output)),
            request_ids=ctx.request_ids,
            request_metadata=ctx.request_metadata,
            eval_result=eval_result,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
        )
        manifests.append(manifest)
    write_manifests(args.manifest_output, manifests)
    return manifests
