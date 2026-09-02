#!/usr/bin/env python3
"""Replay add_records whose extraction produced zero memories.

Context: the schema extraction LLM's token quota died mid-sweep
(2026-08-26T17:21Z). Every sync add after that still recorded
(add_record_v1, status=ok, task_completed_at set) but extracted 0
memories -- the episode was marked permanently failed and never retried.
This script re-POSTs those stored message payloads to /v1/memory/add so
extraction runs again against a working LLM key.

The API key is read from config/mindmemos/api_keys.yaml (never printed).

Usage:
  python replay_adds.py --project-id proj_xxx [--since 2026-08-26T17:21:00+00:00]
                        [--api-url http://localhost:8001] [--limit N] [--yes]
Without --yes it replays exactly ONE record (smoke mode) and reports the
extracted memory count from the add response.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import urllib.request

import yaml

QDRANT = "http://localhost:6333"


def load_api_key(project_id: str) -> str:
    cfg = yaml.safe_load(pathlib.Path("config/mindmemos/api_keys.yaml").read_text(encoding="utf-8"))
    for entry in cfg.get("api_keys", []):
        if entry.get("project_id") == project_id:
            return entry["api_key"]
    raise SystemExit(f"no api key for {project_id}")


def scroll_zero_memory_adds(project_id: str, since_iso: str) -> list[dict]:
    """add_records that completed but extracted no memories, oldest first."""
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
            f"{QDRANT}/collections/add_record_v1/points/scroll",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.load(r)["result"]
        points.extend(result["points"])
        offset = result.get("next_page_offset")
        if offset is None:
            break
    cutoff = dt.datetime.fromisoformat(since_iso)
    out = []
    for p in points:
        pl = p["payload"]
        if "status" not in pl or pl.get("mode") != "sync":
            continue  # skip internal buffer rows
        et = pl.get("event_time")
        if not et:
            continue
        if dt.datetime.fromisoformat(et) < cutoff:
            continue
        if pl.get("memories"):
            continue  # already extracted something
        out.append(pl)
    out.sort(key=lambda pl: pl["event_time"])
    return out


def replay(api_url: str, api_key: str, pl: dict) -> dict:
    body = {
        "messages": pl["messages"],
        "session_id": pl.get("session_id"),
        "user_id": pl.get("user_id"),
        "app_id": pl.get("app_id"),
        "mode": "sync",
        "metadata": {"source": "openclaw-plugin", "replay_of": pl["add_record_id"]},
    }
    req = urllib.request.Request(
        f"{api_url}/v1/memory/add",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--api-url", default="http://localhost:8001")
    ap.add_argument("--since", default="2026-08-26T17:21:00+00:00")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--yes", action="store_true", help="replay up to --limit records (default: 1 smoke replay)")
    args = ap.parse_args()

    api_key = load_api_key(args.project_id)
    records = scroll_zero_memory_adds(args.project_id, args.since)
    print(f"zero-memory add_records since {args.since}: {len(records)}")
    targets = records if args.yes else records[:1]
    targets = targets[: args.limit]

    for i, pl in enumerate(targets, 1):
        event = pl["event_time"][:19]
        print(f"[{i}/{len(targets)}] replaying {pl['add_record_id']} (event {event}, {len(pl['messages'])} msgs) ...", flush=True)
        try:
            resp = replay(args.api_url, api_key, pl)
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code}: {e.read().decode()[:300]}")
            continue
        data = resp.get("data", resp)
        mems = data.get("memories") or []
        print(f"    -> status={data.get('status')} memories={len(mems)}")
    print("done")


if __name__ == "__main__":
    main()
