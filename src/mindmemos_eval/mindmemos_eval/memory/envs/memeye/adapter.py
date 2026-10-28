"""MemEye benchmark adapter: the same build / retrieve / answer / judge loop for every algorithm.

1. Build: each dialogue session is one ``memory.add`` call (sync, in session order). With
   ``execution_params.send_images`` (default true) the turn images are sent by path; otherwise the
   same turns go without images (image-only turns are skipped), e.g. for a text-only baseline
   (vanilla with ``algo_config.multimodal.mode: off`` rejects images). Dataset captions are never sent.
2. Retrieve: one ``memory.search`` per question (MCQ rotations share it), with the question images
   when images are sent and ``search_params.multimodal`` passed through. Search defaults
   (``top_k`` / ``rerank`` / ``search_strategy``) come from the runner config (CLI ``--top-k`` /
   ``--rerank`` / ``--search-strategy``) like every other benchmark; benchmark/profile
   ``search_params`` still win over them.
3. Answer: the shared VLM answerer (``answer.py``) sees the retrieved text, the returned images
   and the question images.
4. Score: debiased MCQ EM over option rotations; OPEN BLEU-1 and the MemEye judge rubric, the
   judge run ``--judge-runs`` times per question with the scores averaged.

Metrics are macro means over tasks (as the MemEye paper reports), plus micro means and client-side
cost (call counts, latencies, answerer/judge tokens).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.base import BenchmarkSpec, RunContext
from mindmemos_eval.memory.config import _merged_runner_config, _option
from mindmemos_sdk import DialogueMessage, ImageRef

from .answer import ImageEncoder, build_messages, options_block
from .dataset import MEMEYE_TASKS, MODES, MemEyeTask, Question, Session, load_task
from .scoring import bleu1, extract_choice, judge_messages, parse_judge

logger = logging.getLogger("mindmemos_eval.memory.memeye")

RETRIES = 6
BACKOFF_SECONDS = 10.0


class MemEyeAdapter:
    """Run MemEye (8 tasks x MCQ/OPEN) against any memory algorithm with one shared answerer."""

    name = "memeye"

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
        raw = bench_config.raw
        tasks = list(raw.get("tasks") or MEMEYE_TASKS)
        modes = tuple(raw.get("modes") or MODES)
        data_root = Path(bench_config.dataset)
        result = await run_tasks(
            _Run,
            tasks,
            lambda name: load_task(data_root, name, modes),
            memory=memory,
            answer_llm=answer_llm,
            judge_llm=judge_llm,
            ctx=ctx,
            bench_config=bench_config,
            args=args,
        )
        result["metrics"] = _manifest_metrics(_metrics(result.pop("per_task_results")), result, headline="EM")
        _write_summary(result)
        logger.info("memeye_metrics %s", result["metrics"])
        return result


async def run_tasks(
    run_cls: type[_Run],
    tasks: list[str],
    load: Callable[[str], Any],
    *,
    memory: Any,
    answer_llm: LLMClient,
    judge_llm: LLMClient,
    ctx: RunContext,
    bench_config: BenchmarkSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build, retrieve, answer and judge every task with ``run_cls``; metrics are left to the caller.

    The result holds ``per_task_results`` (task name -> ``_Run.task`` output) for the metrics, then
    validity, cost and the run settings.
    """

    runner = getattr(args, "runner_config", None) or _merged_runner_config(args)
    raw, execution = bench_config.raw, bench_config.execution_params
    limit = bench_config.limit if bench_config.limit is not None else _option(args, "limit")
    if limit is not None:
        tasks = tasks[: int(limit)]
    # Defaults come from the runner config (YAML ``runner:`` / CLI overrides) exactly like the
    # other benchmarks; benchmark/profile search_params win over them.
    search = {"top_k": runner.top_k, "rerank": runner.rerank, **dict(bench_config.search_params or {})}
    strategy = (
        search.pop("search_strategy", None)
        or search.pop("public_search_strategy", None)
        or runner.search_strategy
        or "fast"
    )
    search["search_strategy"] = strategy if strategy in ("fast", "agentic") else "fast"
    run = run_cls(
        memory=memory,
        answer_llm=answer_llm,
        judge_llm=judge_llm if runner.score else None,
        send_images=bool(execution.get("send_images", True)),
        add=runner.add,
        search_params={k: search[k] for k in ("top_k", "rerank", "search_strategy", "multimodal") if k in search},
        max_questions=int(execution.get("max_questions") or 0),
        max_images=int(execution.get("max_answer_images") or 8),
        encoder=ImageEncoder(memory, max_side=int(execution.get("image_max_side") or 1024)),
        qa_semaphore=asyncio.Semaphore(max(1, int(runner.max_qa_concurrency))),
        judge_semaphore=asyncio.Semaphore(max(1, int(runner.max_score_concurrency or 8))),
        search_semaphore=(
            asyncio.Semaphore(max(1, int(runner.max_search_concurrency))) if runner.max_search_concurrency else None
        ),
        judge_runs=max(1, int(runner.judge_runs)),
        show_progress=runner.show_progress,
        output_dir=Path(raw.get("output_root") or "runs/mindmemos_eval") / ctx.identity.run_id / run_cls.bench,
    )
    run.output_dir.mkdir(parents=True, exist_ok=True)
    task_semaphore = asyncio.Semaphore(max(1, int(runner.max_conv_concurrency)))

    async def one(name: str) -> dict[str, Any]:
        async with task_semaphore:
            return await run.task(load(name))

    started = time.time()
    per_task = dict(zip(tasks, await asyncio.gather(*(one(name) for name in tasks)), strict=True))
    return {
        "per_task_results": per_task,
        # A run with failed answer or judge calls scores those questions as wrong: not comparable.
        "valid": not any(run.errors.values()),
        "cost": run.cost(),
        "judge_runs": run.judge_runs,
        "per_task": {name: task["summary"] for name, task in per_task.items()},
        "output_dir": str(run.output_dir),
        "send_images": run.send_images,
        "search_params": run.search_params,
        "wall_seconds": round(time.time() - started, 1),
    }


def _write_summary(result: dict[str, Any]) -> None:
    path = Path(result["output_dir"]) / "summary.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")


def _manifest_metrics(metrics: dict[str, Any], result: dict[str, Any], *, headline: str) -> dict[str, Any]:
    """Add the manifest fields every benchmark owes ``metrics.py`` and the reports consumers.

    ``overall_accuracy`` follows the locomo convention (0-1 fraction) using the benchmark's
    headline metric (MemEye ``EM``, Mem-Gallery ``F1``); the token/count fields mirror locomo's
    names so the standard metrics collector fills its sheets from multimodal manifests too.
    """

    cost = result.get("cost") or {}
    per_task = result.get("per_task") or {}
    total = sum(
        int(mode.get("n") or 0) for summary in per_task.values() for mode in summary.values() if isinstance(mode, dict)
    )
    headline_value = metrics.get(headline)
    fields: dict[str, Any] = {
        "overall_accuracy": None if headline_value is None else round(headline_value / 100, 6),
        "total_questions": total,
        "total": total,
        "judge_runs_used": int(result.get("judge_runs") or 1),
        "answer_llm_calls": cost.get("answer_calls"),
        "answer_prompt_tokens": cost.get("answer_prompt_tokens"),
        "answer_completion_tokens": cost.get("answer_completion_tokens"),
        "answer_total_tokens": cost.get("answer_tokens"),
        "judge_llm_calls": cost.get("judge_calls"),
        "judge_prompt_tokens": cost.get("judge_prompt_tokens"),
        "judge_completion_tokens": cost.get("judge_completion_tokens"),
        "judge_total_tokens": cost.get("judge_tokens"),
    }
    return {**metrics, **{key: value for key, value in fields.items() if value is not None}}


class _Run:
    bench = "memeye"
    """Prefix of the log keys (``<bench>_answer_failed`` etc.); the run scripts watch them."""

    judge_runs = 1
    search_semaphore: asyncio.Semaphore | None = None
    show_progress = True

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.timings: dict[str, list[float]] = {"add": [], "search": []}
        self.tokens: dict[str, int] = {
            "answer_calls": 0,
            "answer_tokens": 0,
            "answer_prompt_tokens": 0,
            "answer_completion_tokens": 0,
            "judge_calls": 0,
            "judge_tokens": 0,
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
        }
        self.errors: dict[str, int] = {"answer": 0, "judge": 0}

    async def task(self, task: MemEyeTask) -> dict[str, Any]:
        user_id = f"{self.bench}-{task.name.lower()}"
        if self.add:
            for session in task.sessions:
                messages = self._session_messages(session)
                if messages:
                    await self._timed(
                        "add",
                        lambda m=messages, s=session: self.memory.add(
                            m, user_id=user_id, mode="sync", session_id=s.session_id
                        ),
                    )
            if self.show_progress:
                logger.info(
                    "%s_build task=%s sessions=%d rounds=%d", self.bench, task.name, len(task.sessions), task.rounds
                )
        summary: dict[str, Any] = {"rounds": task.rounds}
        rows_by_mode: dict[str, list[dict[str, Any]]] = {}
        for mode, questions in task.questions.items():
            if self.max_questions:
                questions = questions[: self.max_questions]
            rows = await asyncio.gather(*(self._question(task, user_id, mode, q) for q in questions))
            path = self.output_dir / f"{task.name}_{mode}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
            summary[mode] = _phase_summary(mode, rows)
            rows_by_mode[mode] = rows
            if self.show_progress:
                logger.info("%s_phase task=%s mode=%s %s", self.bench, task.name, mode, summary[mode])
        return {"summary": summary, "rows": rows_by_mode}

    def _session_messages(self, session: Session) -> list[DialogueMessage]:
        messages: list[DialogueMessage] = []
        for index, rnd in enumerate(session.rounds):
            timestamp = session.timestamp_ms(index)
            images = (
                [ImageRef(path=str(path), image_id=image_id) for path, image_id in _zip_ids(rnd.images, rnd.image_ids)]
                if self.send_images
                else []
            )
            if rnd.user.strip() or images:
                messages.append(DialogueMessage(role="user", content=rnd.user, timestamp=timestamp, images=images))
            if rnd.assistant.strip():
                messages.append(DialogueMessage(role="assistant", content=rnd.assistant, timestamp=timestamp))
        return messages

    async def _question(self, task: MemEyeTask, user_id: str, mode: str, q: Question) -> dict[str, Any]:
        async with self.qa_semaphore:
            started = time.perf_counter()
            query_images = [ImageRef(path=str(path)) for path in q.images] if self.send_images else []
            result = await self._gated_search(
                lambda: self._timed(
                    "search",
                    lambda: self.memory.search(
                        q.question,
                        user_id=user_id,
                        **self.search_params,
                        **({"query_images": query_images} if query_images else {}),
                    ),
                )
            )
            search_ms = (time.perf_counter() - started) * 1000
            memories = list(result.memories)
            row: dict[str, Any] = {
                "task": task.name,
                "mode": mode,
                "question_id": q.question_id,
                "question": q.question,
                "gt": q.answer,
                "point": q.point,
                "clue": list(q.clue),
                "search_ms": round(search_ms, 1),
                "retrieved": [
                    {
                        "id": m.id,
                        "type": m.memory_type,
                        "memory": m.memory[:300],
                        "score": (m.metadata or {}).get("score"),
                    }
                    for m in memories
                ],
                "n_image_results": sum(1 for m in memories if m.memory_type == "image"),
            }
            if mode == "mcq":
                rotations = []
                for rotation in q.rotations:
                    block, valid = options_block(q.question, rotation)
                    pred, attached = await self._answer(mode, block, memories, q)
                    choice = extract_choice(pred, valid)
                    rotations.append(
                        {
                            "pred": pred,
                            "choice": choice,
                            "gold": rotation.get("answer"),
                            "em": float(choice == str(rotation.get("answer", "")).strip().upper()),
                        }
                    )
                row["rotations"] = rotations
                row["em"] = statistics.fmean(r["em"] for r in rotations) if rotations else 0.0
                row["attached_images"] = attached if rotations else 0
            else:
                pred, row["attached_images"] = await self._answer(mode, self._open_block(q), memories, q)
                row["pred"] = pred
                row["bleu1"] = bleu1(pred, q.answer)
                row["judge"] = await self._judge(q, pred) if self.judge_llm is not None else None
            return row

    def _open_block(self, q: Question) -> str:
        """What the answerer is asked for an OPEN question."""

        return q.question

    def _answer_options(self, mode: str, memories: list[Any], q: Question) -> dict[str, Any]:
        """Answer protocol: ``build_messages`` overrides plus ``max_tokens``."""

        return {"max_tokens": 8 if mode == "mcq" else 128}

    async def _answer(self, mode: str, block: str, memories: list[Any], q: Question) -> tuple[str, int]:
        options = self._answer_options(mode, memories, q)
        max_tokens = options.pop("max_tokens")
        messages, attached = await build_messages(
            mode=mode,
            question_block=block,
            memories=memories,
            question_images=list(q.images),
            encoder=self.encoder,
            max_images=self.max_images,
            **options,
        )
        try:
            completion = await self.answer_llm.complete(messages, temperature=0.0, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 - one failed answer must not abort the run
            self.errors["answer"] += 1
            logger.warning("%s_answer_failed qid=%s error=%s", self.bench, q.question_id, str(exc)[:200])
            return "", attached
        self.tokens["answer_calls"] += 1
        self.tokens["answer_tokens"] += completion.total_tokens
        self.tokens["answer_prompt_tokens"] += completion.prompt_tokens
        self.tokens["answer_completion_tokens"] += completion.completion_tokens
        return completion.content, attached

    async def _judge(self, q: Question, pred: str) -> float | None:
        """Mean of ``judge_runs`` independent judge scores (runner ``--judge-runs``).

        Judge scores are continuous 0-1, so the analog of locomo's majority vote is the mean
        over the N runs; the runs execute sequentially per question like locomo's.
        """

        async with self.judge_semaphore:
            scores = []
            for _ in range(self.judge_runs):
                score = await self._judge_once(q, pred)
                if score is not None:
                    scores.append(score)
        if not scores:
            self.errors["judge"] += 1
            return None
        return scores[0] if len(scores) == 1 else round(statistics.fmean(scores), 6)

    async def _judge_once(self, q: Question, pred: str) -> float | None:
        for _ in range(3):
            try:
                completion = await self.judge_llm.complete(
                    self._judge_messages(q, pred), temperature=0.0, max_tokens=300
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s_judge_failed qid=%s error=%s", self.bench, q.question_id, str(exc)[:200])
                continue
            self.tokens["judge_calls"] += 1
            self.tokens["judge_tokens"] += completion.total_tokens
            self.tokens["judge_prompt_tokens"] += completion.prompt_tokens
            self.tokens["judge_completion_tokens"] += completion.completion_tokens
            score = self._parse_judge(completion.content)
            if score is not None:
                return score
        return None

    def _judge_messages(self, q: Question, pred: str) -> list[dict[str, Any]]:
        return judge_messages(q.question, q.answer, pred)

    def _parse_judge(self, text: str) -> float | None:
        return parse_judge(text)

    async def _gated_search(self, call) -> Any:
        """Run one search under the runner's ``--max-search-concurrency`` when it is set."""
        if self.search_semaphore is None:
            return await call()
        async with self.search_semaphore:
            return await call()

    async def _timed(self, kind: str, call) -> Any:
        for attempt in range(RETRIES + 1):
            started = time.perf_counter()
            try:
                result = await call()
            except Exception as exc:  # noqa: BLE001 - retried only when transient
                if attempt >= RETRIES or not _transient(exc):
                    raise
                logger.warning("%s_%s_retry attempt=%d error=%s", self.bench, kind, attempt + 1, str(exc)[:200])
                await asyncio.sleep(BACKOFF_SECONDS * (attempt + 1))
                continue
            self.timings[kind].append((time.perf_counter() - started) * 1000)
            return result
        raise RuntimeError("unreachable")

    def cost(self) -> dict[str, Any]:
        cost: dict[str, Any] = {}
        for kind, values in self.timings.items():
            cost[kind] = {"calls": len(values), **_latency(values)}
        cost.update(self.tokens)
        cost["errors"] = dict(self.errors)
        return cost


def _zip_ids(paths: tuple[Path, ...], ids: tuple[str, ...]) -> list[tuple[Path, str | None]]:
    return [(path, ids[i] if i < len(ids) else None) for i, path in enumerate(paths)]


def _transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "connect",
        "timeout",
        "timed out",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "internal",
        "remoteprotocolerror",
        "readerror",
        "unavailable",
        "transporterror",
    )
    return any(marker in text for marker in markers)


def _latency(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "total_s": 0.0}
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.fmean(ordered), 1),
        "p50_ms": round(ordered[len(ordered) // 2], 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
        "total_s": round(sum(ordered) / 1000, 1),
    }


def _phase_summary(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.fmean(values) if values else None

    summary: dict[str, Any] = {"n": len(rows), "image_results_per_q": mean("n_image_results")}
    if mode == "mcq":
        summary["em"] = mean("em")
    else:
        summary.update(
            bleu1=mean("bleu1"), judge=mean("judge"), unjudged=sum(1 for r in rows if r.get("judge") is None)
        )
    return summary


def _metrics(per_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def pct(value: float | None) -> float | None:
        return None if value is None else round(100 * value, 2)

    def macro(mode: str, key: str) -> float | None:
        values = [
            t["summary"][mode][key]
            for t in per_task.values()
            if mode in t["summary"] and t["summary"][mode].get(key) is not None
        ]
        return statistics.fmean(values) if values else None

    return {
        "EM": pct(macro("mcq", "em")),
        "B-1": pct(macro("open", "bleu1")),
        "Judge": pct(macro("open", "judge")),
        "per_task": {
            name: {
                "EM": pct(t["summary"].get("mcq", {}).get("em")),
                "B-1": pct(t["summary"].get("open", {}).get("bleu1")),
                "Judge": pct(t["summary"].get("open", {}).get("judge")),
            }
            for name, t in per_task.items()
        },
    }
