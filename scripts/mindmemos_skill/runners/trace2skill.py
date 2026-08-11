#!/usr/bin/env python3
"""Run one registered trajectory-to-Skill algorithm."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from experiments import ExperimentFamily, dispatch_experiment, list_experiments  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    algorithms = list_experiments(family=ExperimentFamily.TRACE2SKILL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=algorithms)
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args, algorithm_args = parse_args(argv)
    try:
        return dispatch_experiment(ExperimentFamily.TRACE2SKILL, args.algorithm, algorithm_args)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
