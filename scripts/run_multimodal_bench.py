"""Run the multimodal benchmarks (MemEye, Mem-Gallery) for vanilla's multimodal modes in one go.

For each benchmark it derives, from one filled-in eval config, the four configurations of the
ablation ladder and runs them strictly one after another against the running server:

    b0                mode off, no images sent (text-only vanilla)
    b1                mode basic
    enhanced          mode enhanced, fast search
    enhanced_agentic  mode enhanced, multimodal agentic search

All of them search with multimodal=true. Each run writes a manifest under reports/multimodal/<tag>/
and its log next to it; a run whose manifest already holds a valid result is skipped, so an
interrupted session resumes where it stopped. A run with failed answerer/judge calls is invalid:
the script stops there instead of spending the rest of the budget. A summary table (mean and range
over runs) is printed and written to reports/multimodal/<tag>/summary.md.

    uv run python scripts/run_multimodal_bench.py                      # both benchmarks, 4 configs, 1 run
    uv run python scripts/run_multimodal_bench.py --smoke              # 1 task, 2 questions per phase
    uv run python scripts/run_multimodal_bench.py --benchmarks memgallery --configs enhanced --runs 2

See the multimodal section of docs/eval/README_ZH.md for the setup (datasets, server config,
embedding API).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LADDER = {
    "b0": ("off", "fast"),
    "b1": ("basic", "fast"),
    "enhanced": ("enhanced", "fast"),
    "enhanced_agentic": ("enhanced", "agentic"),
}
BENCHMARKS = {
    "memeye": "config/mindmemos_eval/memory_evaluation_multimodal.yaml",
    "memgallery": "config/mindmemos_eval/memory_evaluation_memgallery.yaml",
}
SMOKE_TASKS = {"memeye": "Brand_Memory_Test", "memgallery": "AI_Robotics_Automation_Future_Tech"}
COLUMNS = {"memeye": ("EM", "B-1", "Judge"), "memgallery": ("F1", "B-1", "Judge")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    parser.add_argument("--configs", nargs="+", choices=list(LADDER), default=list(LADDER))
    parser.add_argument("--runs", type=int, default=1, help="runs per configuration (default 1)")
    parser.add_argument("--smoke", action="store_true", help="first task only, 2 questions per phase")
    parser.add_argument("--tag", default=None, help="output folder name (default: date, or 'smoke')")
    parser.add_argument("--memeye-config", default=BENCHMARKS["memeye"])
    parser.add_argument("--memgallery-config", default=BENCHMARKS["memgallery"])
    parser.add_argument("--server-config", default="config/mindmemos/dev.yaml")
    parser.add_argument("--api-key-output", default="config/mindmemos/eval_api_keys.yaml")
    args = parser.parse_args()

    os.chdir(ROOT)
    _load_dotenv(ROOT / ".env")
    tag = args.tag or ("smoke" if args.smoke else time.strftime("%Y%m%d"))
    out = Path("reports/multimodal") / tag
    (out / "configs").mkdir(parents=True, exist_ok=True)
    sources = {"memeye": args.memeye_config, "memgallery": args.memgallery_config}
    bases = {bench: _read_yaml(Path(sources[bench])) for bench in args.benchmarks}
    if not _preflight(bases, args):
        return 2

    results: dict[tuple[str, str], list[dict]] = {}
    for bench in args.benchmarks:
        for name in args.configs:
            config_path = out / "configs" / f"{bench}_{name}.yaml"
            config_path.write_text(
                yaml.safe_dump(_variant(bases[bench], bench, name, args.smoke), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            for run in range(1, args.runs + 1):
                label = f"{bench}_{name}_r{run}"
                manifest = out / f"{label}.jsonl"
                result = _valid_result(manifest, bench)
                if result is None:
                    print(f"{time.strftime('%H:%M:%S')} start {label}", flush=True)
                    code = _run(bench, config_path, manifest, out / f"{label}.log", args)
                    result = _valid_result(manifest, bench)
                    if result is None:
                        print(
                            f"STOP: {label} failed (exit {code}) or is invalid (failed answerer/judge calls); "
                            f"see {out / (label + '.log')}",
                            flush=True,
                        )
                        _summary(results, out)
                        return 1
                    print(f"{time.strftime('%H:%M:%S')} done  {label} {_fmt_metrics(bench, result)}", flush=True)
                else:
                    print(f"skip {label} (valid result exists) {_fmt_metrics(bench, result)}", flush=True)
                results.setdefault((bench, name), []).append(result)
    _summary(results, out)
    return 0


def _variant(base: dict, bench: str, name: str, smoke: bool) -> dict:
    mode, strategy = LADDER[name]
    cfg = json.loads(json.dumps(base))  # deep copy
    profile = cfg.setdefault("algorithm_profiles", {}).setdefault("vanilla", {"memory_algorithm": "vanilla"})
    profile["project_override_config"] = {"algo_config": {"multimodal": {"mode": mode}}}
    profile.setdefault("search_params", {}).update(multimodal=True, search_strategy=strategy)
    execution = cfg.setdefault("execution_params", {}).setdefault(bench, {})
    execution["send_images"] = mode != "off"  # mode off rejects images
    if smoke:
        cfg["benchmarks"][bench]["tasks"] = [SMOKE_TASKS[bench]]
        execution["max_questions"] = 2
    return cfg


def _run(bench: str, config: Path, manifest: Path, log: Path, args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "mindmemos_eval.cli",
        "memory",
        "--benchmark-config",
        str(config),
        "--benchmark-list",
        bench,
        "--algorithm",
        "vanilla",
        "--manifest-output",
        str(manifest),
        "--api-key-output",
        args.api_key_output,
        "--server-config",
        args.server_config,
    ]
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)


def _valid_result(manifest: Path, bench: str) -> dict | None:
    if not manifest.exists():
        return None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line) if line.strip() else {}
        result = row.get("eval_result") or {}
        if row.get("benchmark") == bench and result.get("valid") and result.get("metrics"):
            return result
    return None


def _preflight(bases: dict[str, dict], args: argparse.Namespace) -> bool:
    """Catch the usual setup mistakes before hours of runs."""

    ok = True
    server = _read_yaml(Path(args.server_config))
    for bench, base in bases.items():
        url = (base.get("runner") or {}).get("base_url", "http://127.0.0.1:8000")
        try:
            urllib.request.urlopen(f"{url}/docs", timeout=10)  # noqa: S310 - local server
        except Exception as exc:  # noqa: BLE001
            print(f"[{bench}] server {url} not reachable ({exc}); start it first (make dev)")
            ok = False
        dataset = Path(base["benchmarks"][bench]["dataset"])
        if not (dataset / "dialog").is_dir():
            print(f"[{bench}] dataset {dataset} has no dialog/ folder; see the download step of the guide")
            ok = False
        roots = [
            Path(r).expanduser().resolve() for r in (server.get("blob_store") or {}).get("allowed_path_roots") or []
        ]
        if dataset.exists() and not any(dataset.resolve().is_relative_to(r) for r in roots):
            print(f"[{bench}] {dataset.resolve()} is not under blob_store.allowed_path_roots of {args.server_config}")
            ok = False
    if any(name != "b0" for name in args.configs) and not (server.get("mm_embed_model_router") or {}).get("endpoints"):
        print(f"mm_embed_model_router.endpoints is empty in {args.server_config}: basic/enhanced need it")
        ok = False
    key_file = (server.get("auth") or {}).get("api_key_file")
    if key_file and Path(key_file).name != Path(args.api_key_output).name:
        print(
            f"auth.api_key_file is {key_file!r} but the eval writes its keys to {args.api_key_output}; "
            "point the server at that file (it reloads it)"
        )
        ok = False
    return ok


def _summary(results: dict[tuple[str, str], list[dict]], out: Path) -> None:
    lines = []
    for bench in BENCHMARKS:
        rows = [(name, runs) for (b, name), runs in results.items() if b == bench]
        if not rows:
            continue
        columns = COLUMNS[bench]
        lines += [
            f"## {bench}",
            "",
            "| config | runs | " + " | ".join(columns) + " |",
            "|---" * (len(columns) + 2) + "|",
        ]
        for name, runs in rows:
            cells = []
            for column in columns:
                values = [r["metrics"].get(column) for r in runs if r["metrics"].get(column) is not None]
                if not values:
                    cells.append("-")
                elif len(values) == 1:
                    cells.append(f"{values[0]:.2f}")
                else:
                    cells.append(f"{statistics.fmean(values):.2f} ({min(values):.2f}-{max(values):.2f})")
            lines.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")
        lines.append("")
    if lines:
        text = "\n".join(lines)
        print("\n" + text)
        (out / "summary.md").write_text(text + "\n", encoding="utf-8")


def _fmt_metrics(bench: str, result: dict) -> str:
    return " ".join(f"{c}={result['metrics'].get(c)}" for c in COLUMNS[bench])


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing {path}: copy the .example.yaml next to it and fill in the model keys (see the guide)")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_dotenv(path: Path) -> None:
    """Export KEY=VALUE lines of .env (as `make` does) so the project reset reaches Qdrant/Neo4j."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip().removeprefix("export ").strip(), value.strip().strip("'\""))


if __name__ == "__main__":
    raise SystemExit(main())
