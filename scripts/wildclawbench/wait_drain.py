"""Block until a WildClawBench project's pending add_record writes finish.

WildClawBench tasks call MindMemOS's memory ``add`` endpoint through the
OpenClaw plugin with ``addMode: async``. The HTTP call returns immediately
with ``status: queued``; the actual write lands later, processed by the
Kafka consumer running inside the MindMemOS API process (see
``mindmemos.workers.memory_add``). run_batch.py has no idea about this
background step, so if the next task starts before it finishes, that task's
recall can miss or race against the previous task's memories.

This script polls Qdrant's ``add_record_v1`` collection directly (it's a
local dev deployment, so this is faster and simpler than adding a new
MindMemOS API surface) and exits 0 once no record for the given
``project_id`` is still ``queued`` or ``processing``. Intended to run once,
serially, between two WildClawBench task invocations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PENDING_STATUSES = ("queued", "processing")


def count_pending(qdrant_url: str, collection: str, project_id: str) -> int:
    body = {
        "filter": {
            "must": [
                {"key": "project_id", "match": {"value": project_id}},
                {"key": "status", "match": {"any": list(PENDING_STATUSES)}},
            ]
        },
        "limit": 1,
        "with_payload": False,
        "with_vector": False,
    }
    req = urllib.request.Request(
        f"{qdrant_url.rstrip('/')}/collections/{collection}/points/scroll",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read())
    return len(payload["result"]["points"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True, help="e.g. proj_wildclawbench_20260706_112221")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="add_record_v1")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="seconds between checks")
    parser.add_argument("--timeout", type=float, default=180.0, help="max seconds to wait before giving up")
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    while True:
        try:
            pending = count_pending(args.qdrant_url, args.collection, args.project_id)
        except urllib.error.URLError as exc:
            print(f"[wait_drain] cannot reach Qdrant at {args.qdrant_url}: {exc}", file=sys.stderr)
            sys.exit(2)

        if pending == 0:
            print(f"[wait_drain] project {args.project_id}: no pending add_record, safe to continue")
            return

        if time.monotonic() >= deadline:
            print(
                f"[wait_drain] timed out after {args.timeout}s with {pending} add_record(s) "
                f"still queued/processing for {args.project_id}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[wait_drain] {pending} add_record(s) still queued/processing, waiting...")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
