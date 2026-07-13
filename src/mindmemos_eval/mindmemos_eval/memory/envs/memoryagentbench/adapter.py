"""MemoryAgentBench benchmark matrix adapter."""

from __future__ import annotations

import argparse
from typing import Any

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.base import BenchmarkSpec, RunContext
from mindmemos_eval.memory.config import _merged_runner_config, _option, resolve_public_search_strategy

from .env import MemoryAgentBenchEnv


class MemoryAgentBenchAdapter:
    """MemoryAgentBench adapter backed by :class:`MemoryAgentBenchEnv`."""

    name = "memoryagentbench"

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
        """Run MemoryAgentBench using the dedicated eval environment."""
        del ctx, judge_llm  # MAB scores via text metrics, not a separate LLM judge
        runner = getattr(args, "runner_config", None)
        if runner is None:
            runner = _merged_runner_config(args)

        # Resolve MAB-specific params from benchmark raw config.
        sub_dataset = str(bench_config.raw.get("sub_dataset") or "")
        chunk_size = int(bench_config.execution_params.get("chunk_size", 512))
        max_queries = int(bench_config.raw.get("max_queries", 0))

        data = MemoryAgentBenchEnv.load_dataset(
            bench_config.dataset,
            sub_dataset=sub_dataset,
        )
        limit = bench_config.limit if bench_config.limit is not None else _option(args, "limit")
        if limit is not None:
            data = data[: int(limit)]

        search_params = bench_config.search_params
        public_search_strategy = resolve_public_search_strategy(
            search_params.get("public_search_strategy")
            or ("agentic" if search_params.get("agentic") is True else None)
            or runner.search_strategy
            or "fast"
        )
        top_k = search_params["top_k"] if "top_k" in search_params else runner.top_k
        rerank = search_params["rerank"] if "rerank" in search_params else runner.rerank

        env = MemoryAgentBenchEnv(
            memory,
            answer_llm=answer_llm,
            sub_dataset=sub_dataset,
            top_k=None if top_k is None else int(top_k),
            search_strategy=public_search_strategy,
            rerank=bool(rerank),
            chunk_size=chunk_size,
        )
        run = await env.run_dataset(
            data,
            max_context_concurrency=runner.max_conv_concurrency,
            max_qa_concurrency=runner.max_qa_concurrency,
            add=runner.add,
            score=runner.score,
            max_queries=max_queries,
            show_progress=runner.show_progress,
        )
        return run.model_dump()
