"""Mem-Gallery benchmark adapter: MemEye's build / retrieve / answer / judge loop, Mem-Gallery protocol.

1. Build: each dialogue session is one ``memory.add`` call (sync, in session order), images sent by
   path with their dataset ids when ``execution_params.send_images`` (default true); otherwise the
   same turns go without images. Dataset captions are never sent.
2. Retrieve: one ``memory.search`` per question, with the question image when images are sent and
   ``search_params.multimodal`` passed through.
3. Answer: the shared VLM answerer with Mem-Gallery's prompts (``answer.py``).
4. Score: F1, BLEU-1 and the Mem-Gallery judge (``scoring.py``), micro means over every question as
   the benchmark reports, plus per-task and per-category means and client-side cost.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from pathlib import Path
from typing import Any

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.base import BenchmarkSpec, RunContext

from ..memeye.adapter import _Run, _write_summary, run_tasks
from ..memeye.dataset import MemEyeTask, Question
from .answer import answer_options, question_block
from .dataset import load_task, task_names
from .scoring import f1, judge_messages, parse_judge

logger = logging.getLogger("mindmemos_eval.memory.memgallery")


class MemGalleryAdapter:
    """Run Mem-Gallery (20 tasks, OPEN) against any memory algorithm with the shared answerer."""

    name = "memgallery"

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
        data_root = Path(bench_config.dataset)
        tasks = list(bench_config.raw.get("tasks") or task_names(data_root))
        if not tasks:
            raise FileNotFoundError(f"no Mem-Gallery dialog files under {data_root / 'dialog'}")
        result = await run_tasks(
            _GalleryRun,
            tasks,
            lambda name: load_task(data_root, name),
            memory=memory,
            answer_llm=answer_llm,
            judge_llm=judge_llm,
            ctx=ctx,
            bench_config=bench_config,
            args=args,
        )
        result["metrics"] = metrics(result.pop("per_task_results"))
        _write_summary(result)
        logger.info("memgallery_metrics %s", {k: v for k, v in result["metrics"].items() if k[0].isupper()})
        return result


class _GalleryRun(_Run):
    bench = "memgallery"

    async def _question(self, task: MemEyeTask, user_id: str, mode: str, q: Question) -> dict[str, Any]:
        row = await super()._question(task, user_id, mode, q)
        row["f1"] = f1(row["pred"], q.answer)
        return row

    def _open_block(self, q: Question) -> str:
        return question_block(q.question, q.point, getattr(q, "speaker", "user"))

    def _answer_options(self, mode: str, memories: list[Any], q: Question) -> dict[str, Any]:
        return answer_options(memories)

    def _judge_messages(self, q: Question, pred: str) -> list[dict[str, Any]]:
        return judge_messages(q.question, q.answer, pred)

    def _parse_judge(self, text: str) -> float | None:
        return parse_judge(text)


def metrics(per_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """F1 / B-1 / Judge in percent: micro means over all questions, then per task and per category."""

    rows = {name: task["rows"].get("open", []) for name, task in per_task.items()}
    every = [row for task_rows in rows.values() for row in task_rows]
    points = sorted({str(row.get("point")) for row in every})
    return {
        **_block(every),
        "unjudged": sum(1 for row in every if row.get("judge") is None),
        "per_task": {name: _block(task_rows) for name, task_rows in rows.items()},
        "per_point": {point: _block([r for r in every if str(r.get("point")) == point]) for point in points},
    }


def _block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def pct(key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        return round(100 * statistics.fmean(values), 2) if values else None

    return {"n": len(rows), "F1": pct("f1"), "B-1": pct("bleu1"), "Judge": pct("judge")}
