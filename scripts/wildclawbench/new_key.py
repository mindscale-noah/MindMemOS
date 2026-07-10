"""Mint a fresh, uniquely-named WildClawBench API key/project and append it
to config/mindmemos/api_keys.yaml.

Mirrors the naming convention used by the other eval harnesses in this repo
(``mindmemos_eval.memory.identity.new_identity``): ``key_<benchmark>_<algo>_<
timestamp>_<random hex>``. The LoCoMo key already in api_keys.yaml
(``key_locomo_schema_20260703_032425_a1a8c806``) was generated this way; the
WildClawBench key added on 2026-07-06 was not (it was hand-typed once), which
is the inconsistency this script fixes going forward -- run it once per eval
campaign instead of hand-editing the YAML.

Algorithm knobs are NOT hardcoded here. You can either:

- pass ``--project-override-config`` with a YAML file containing the same
  shape as an existing entry's ``project_override_config:`` block (see
  ``config/presets/project_override_wildclawbench_schema.example.yaml`` for a
  starting point), or
- pass ``--from-memory-eval-profile`` to copy
  ``algorithm_profiles.<name>.project_override_config`` from
  ``config/mindmemos_eval/memory_evaluation.yaml``.

Omit both to run on plain defaults.

This only edits the YAML file. It does not touch a running MindMemOS API --
restart/reload the API process after running this so it picks up the new key.
"""

from __future__ import annotations

import argparse
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCOPES = ["memory:read", "memory:write"]
DEFAULT_MEMORY_EVAL_CONFIG = "config/mindmemos_eval/memory_evaluation.yaml"


def build_entry(
    *,
    benchmark: str,
    memory_algorithm: str,
    project_override_config: dict[str, Any] | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    timestamp = now or datetime.now(timezone.utc)
    suffix = f"{timestamp:%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"
    entry: dict[str, Any] = {
        "key_id": f"key_{benchmark}_{memory_algorithm}_{suffix}",
        "api_key": f"dev-api-key-{benchmark}-{memory_algorithm}-{suffix}".replace("_", "-"),
        "project_id": f"proj_{benchmark}_{memory_algorithm}_{suffix}",
        "memory_algorithm": memory_algorithm,
        "enabled": True,
        "scopes": list(DEFAULT_SCOPES),
    }
    if project_override_config:
        entry["project_override_config"] = project_override_config
    return entry


def load_api_keys(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"api_keys": []}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("api_keys", [])
    return data


def atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_project_override_from_memory_eval(path: Path, profile_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("memory evaluation config must be a mapping")

    profiles = data.get("algorithm_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("memory evaluation config is missing algorithm_profiles")

    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(name) for name in profiles.keys()))
        raise ValueError(
            f"algorithm profile {profile_name!r} not found in {path}; available profiles: {available or '(none)'}"
        )

    override = profile.get("project_override_config")
    if override is None:
        raise ValueError(f"algorithm profile {profile_name!r} does not define project_override_config")
    if not isinstance(override, dict):
        raise ValueError(f"algorithm profile {profile_name!r}.project_override_config must be a mapping")
    return dict(override)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/mindmemos/api_keys.yaml",
        help="path to the api_keys.yaml to append into",
    )
    parser.add_argument("--benchmark", default="wildclawbench")
    parser.add_argument("--memory-algorithm", default="schema")
    parser.add_argument(
        "--project-override-config",
        default=None,
        help="optional path to a YAML file with the algo_config knobs to pin for this key",
    )
    parser.add_argument(
        "--from-memory-eval-config",
        default=DEFAULT_MEMORY_EVAL_CONFIG,
        help="path to memory_evaluation.yaml used with --from-memory-eval-profile",
    )
    parser.add_argument(
        "--from-memory-eval-profile",
        default=None,
        help="copy algorithm_profiles.<name>.project_override_config from memory_evaluation.yaml",
    )
    parser.add_argument(
        "--disable-previous",
        action="store_true",
        help="set enabled: false on any earlier key_id entries for this benchmark",
    )
    args = parser.parse_args()

    override_config = None
    if args.project_override_config and args.from_memory_eval_profile:
        raise ValueError("use either --project-override-config or --from-memory-eval-profile, not both")

    if args.project_override_config:
        override_path = Path(args.project_override_config)
        with override_path.open("r", encoding="utf-8") as fh:
            override_config = yaml.safe_load(fh)
    elif args.from_memory_eval_profile:
        override_config = load_project_override_from_memory_eval(
            Path(args.from_memory_eval_config),
            args.from_memory_eval_profile,
        )

    config_path = Path(args.config)
    data = load_api_keys(config_path)

    if args.disable_previous:
        prefix = f"key_{args.benchmark}_"
        for existing in data["api_keys"]:
            if str(existing.get("key_id", "")).startswith(prefix):
                existing["enabled"] = False

    new_entry = build_entry(
        benchmark=args.benchmark,
        memory_algorithm=args.memory_algorithm,
        project_override_config=override_config,
    )
    data["api_keys"].append(new_entry)
    atomic_write_yaml(config_path, data)

    print(f"appended to {config_path}")
    print(f"  key_id:     {new_entry['key_id']}")
    print(f"  api_key:    {new_entry['api_key']}")
    print(f"  project_id: {new_entry['project_id']}")
    if args.from_memory_eval_profile:
        print(
            "  override:   copied from "
            f"{args.from_memory_eval_config} profile {args.from_memory_eval_profile}"
        )
    elif args.project_override_config:
        print(f"  override:   copied from {args.project_override_config}")
    else:
        print("  override:   none (plain server defaults)")
    print("restart the MindMemOS API to pick this up, then authenticate the eval image with:")
    print(
        "  mindmemos auth --base-url http://host.docker.internal:8001 "
        f"--api-key {new_entry['api_key']} --user-id {args.benchmark}"
    )


if __name__ == "__main__":
    main()
