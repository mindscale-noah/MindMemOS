#!/usr/bin/env python3
"""Build per-task memory write/recall records for a WildClawBench serial run.

Two record sources:
  1. Recall (used memories): each task's task_output/mindmemos-logs/*-search.json,
     dumped by the openclaw plugin at search time (these survive because the
     search completes while the task container is still alive).
  2. Writes (stored memories): the plugin's add dump dies with the container
     (sync extraction outlives it by ~60s), so writes are reconstructed from
     qdrant entity_item_v1 -- every entity created inside a task's serial
     execution window [this task's run-dir timestamp, next task's) belongs to
     that task's add (run_serial.sh's wait_drain makes the windows disjoint).
  3. Replay recovery: if the extraction LLM died mid-sweep, zero-memory
     add_records can be re-POSTed later (replay_adds.py). Replayed entities
     carry created_at = replay time, which breaks window attribution, so pass
     --replay-log/--replay-since: entities created during the replay are
     grouped by request_id (one per replay batch), ordered by creation, and
     zipped with the replay log back to the ORIGINAL add event_time, which
     then attributes via the normal task windows.

Usage:
  python collect_memory_records.py \
    --output-root C:/working_projects/Memory/WildClawBench/output/openclaw \
    --qdrant-url http://localhost:6333 \
    --project-id proj_wildclawbench_schema_20260826_031128_e67bcf5e \
    --out memory_records.json \
    [--replay-log replay_task_output.log --replay-since 2026-08-26T22:10:00+00:00]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import urllib.request

RUN_DIR_RE = re.compile(r"^gpt-(?:4\.1-mini|5\.5)_(\d{8})_(\d{4})_[0-9a-f]+$")


def _load_json(path: pathlib.Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _parse_run_stamp(match: re.Match) -> dt.datetime:
    return dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")


def collect_runs(output_root: pathlib.Path) -> list[dict]:
    """Every task run dir with its start time (from the dir name), search dumps, score."""
    runs = []
    for category_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                m = RUN_DIR_RE.match(run_dir.name)
                if not m:
                    continue
                searches = []
                log_dir = run_dir / "task_output" / "mindmemos-logs"
                if log_dir.is_dir():
                    for f in sorted(log_dir.glob("*-search.json")):
                        d = _load_json(f)
                        if d is not None:
                            searches.append(d)
                runs.append(
                    {
                        "task": f"{category_dir.name}/{task_dir.name}",
                        "run_dir": str(run_dir),
                        "start": _parse_run_stamp(m),
                        "searches": searches,
                        "score": _load_json(run_dir / "score.json"),
                    }
                )
    runs.sort(key=lambda r: r["start"])
    return runs


def scroll_add_records(qdrant_url: str, project_id: str) -> list[dict]:
    """All add_record_v1 points (used to map replayed adds back to event times)."""
    url = qdrant_url.rstrip("/") + "/collections/add_record_v1/points/scroll"
    body = {
        "limit": 256,
        "with_payload": True,
        "filter": {"must": [{"key": "project_id", "match": {"value": project_id}}]},
    }
    points: list[dict] = []
    offset = None
    while True:
        payload = dict(body)
        if offset is not None:
            payload["offset"] = offset
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.load(r)["result"]
        points.extend(result["points"])
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return points


def scroll_entities(qdrant_url: str, project_id: str) -> list[dict]:
    url = qdrant_url.rstrip("/") + "/collections/entity_item_v1/points/scroll"
    body = {
        "limit": 256,
        "with_payload": True,
        "filter": {"must": [{"key": "project_id", "match": {"value": project_id}}]},
    }
    points: list[dict] = []
    offset = None
    while True:
        payload = dict(body)
        if offset is not None:
            payload["offset"] = offset
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.load(r)["result"]
        points.extend(result["points"])
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return points


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True, type=pathlib.Path)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--replay-log", type=pathlib.Path, default=None,
                    help="task output log of replay_adds.py --yes (for zero-memory recovery)")
    ap.add_argument("--replay-since", default=None,
                    help="UTC ISO cutoff after which entities were created by the replay, e.g. 2026-08-26T22:10:00+00:00")
    args = ap.parse_args()

    runs = collect_runs(args.output_root)
    entities = scroll_entities(args.qdrant_url, args.project_id)

    def entity_time(p: dict) -> dt.datetime | None:
        raw = p.get("payload", {}).get("created_at")
        if not raw:
            return None
        try:
            # qdrant stores UTC; run-dir stamps are local time, so convert.
            return dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
        except ValueError:
            return None

    # Replay-batch attribution: map replayed entities back to the ORIGINAL add
    # event_time (task attribution then proceeds via the normal windows).
    # entity_id -> original event_time (local, naive)
    replay_origin: dict[str, dt.datetime] = {}
    if args.replay_log and args.replay_since:
        cutoff = dt.datetime.fromisoformat(args.replay_since).astimezone().replace(tzinfo=None)
        # ordered original add_record ids, oldest first, as replayed
        replayed_ids = re.findall(r"replaying ([0-9a-f-]{36})", args.replay_log.read_text(encoding="utf-8"))
        # original add_record event_time lookup from qdrant
        rec_times = {
            p["payload"]["add_record_id"]: dt.datetime.fromisoformat(p["payload"]["event_time"])
            .astimezone()
            .replace(tzinfo=None)
            for p in scroll_add_records(args.qdrant_url, args.project_id)
            if p["payload"].get("add_record_id") and p["payload"].get("event_time")
        }
        # group replayed entities by request_id (one per batch), order by first creation
        batches: dict[str, list] = {}
        for p in entities:
            t = entity_time(p)
            rid = p.get("payload", {}).get("request_id")
            if t is None or rid is None or t <= cutoff:
                continue
            batches.setdefault(rid, []).append((t, p["payload"].get("entity_id")))
        ordered = sorted(batches.items(), key=lambda kv: min(t for t, _ in kv[1]))
        if len(ordered) >= len(replayed_ids) + 1:
            # A 1-record smoke replay ran before the full sweep; it targeted
            # records[0], which is also the full log's first id. Prepending it
            # realigns the positional zip (batch i <-> the id the log shows).
            # Later non-replay batches (e.g. salvage reruns) sort after the
            # replay batches and are simply left unpaired -> time-window path.
            replayed_ids = [replayed_ids[0]] + replayed_ids
        if len(ordered) != len(replayed_ids):
            print(f"WARNING: {len(ordered)} replay batches vs {len(replayed_ids)} replayed ids; zipping by position")
        for (rid, members), orig_id in zip(ordered, replayed_ids):
            orig_t = rec_times.get(orig_id)
            if orig_t is None:
                continue
            for _, eid in members:
                replay_origin[eid] = orig_t

    records = []
    for i, run in enumerate(runs):
        window_end = runs[i + 1]["start"] if i + 1 < len(runs) else dt.datetime.max
        written = []
        for p in entities:
            pl = p.get("payload", {})
            orig_t = replay_origin.get(pl.get("entity_id"))
            if orig_t is not None:
                # replayed entity: attribute by the original add's event_time
                if not (run["start"] <= orig_t < window_end):
                    continue
            else:
                t = entity_time(p)
                if t is None or not (run["start"] <= t < window_end):
                    continue
            written.append(
                {
                    "entity_id": pl.get("entity_id"),
                    "name": pl.get("entity_name"),
                    "type": pl.get("entity_type"),
                    "description": pl.get("description"),
                    "created_at": pl.get("created_at"),
                }
            )
        recalls = []
        for s in run["searches"]:
            recalls.append(
                {
                    "query": (s.get("query") or "")[:300],
                    "hit_count": s.get("hit_count"),
                    "memories": [
                        {"memory": m.get("memory"), "score": m.get("score")}
                        for m in (s.get("result", {}).get("memories") or [])
                    ],
                }
            )
        score = run["score"] or {}
        records.append(
            {
                "task": run["task"],
                "run_dir": run["run_dir"],
                "started_at": run["start"].isoformat(timespec="seconds"),
                "overall": score.get("overall_score") if isinstance(score, dict) else None,
                "written_entity_count": len(written),
                "written_entities": written,
                "search_count": len(recalls),
                "recalls": recalls,
            }
        )

    args.out.write_text(
        json.dumps(
            {
                "project_id": args.project_id,
                "task_count": len(records),
                "total_entities_written": sum(r["written_entity_count"] for r in records),
                "total_searches": sum(r["search_count"] for r in records),
                "tasks": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"tasks={len(records)} entities={sum(r['written_entity_count'] for r in records)} -> {args.out}")


if __name__ == "__main__":
    main()
