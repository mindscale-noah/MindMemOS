"""Collect matched execution evidence, evolve complete Skills, and retest."""

import re
from datetime import datetime, timezone
from uuid import uuid4

from mindmemos_skill.common.asyncio import map_bounded
from mindmemos_skill.common.json import read_json, read_jsonl, write_json
from mindmemos_skill.common.secrets import redact_credentials
from mindmemos_skill.envs import Environment
from mindmemos_skill.skill import Skill, SkillSet, SkillSnapshot

from .config import ROOT, EnvTestConfig, SkillEvolveConfig
from .evaluation import evaluate
from .evolve_service import evolution_connection
from .testing import _load_samples
from .typing import CaseResult

# Explicit HTTP statuses take precedence over generic transport error text.
_HTTP_STATUS = re.compile(
    r"\b(?:http(?:/\d(?:\.\d)?)?|error code|status code|api error)\s*:?\s*([45]\d\d)\b",
    re.IGNORECASE,
)
_RETRYABLE = re.compile(
    r"connectionerror|connection (?:closed|reset|refused|aborted|error)|econn"
    r"|authentication|unauthorized|banned|rate limit"
    r"|api error|api timed out|environmentstarttimeouterror|无可用渠道",
    re.IGNORECASE,
)


def sanitize_trajectory(value):
    """Copy public messages while replacing long binary payloads."""
    if isinstance(value, str):
        return re.sub(r"[A-Za-z0-9+/=]{400,}", "[omitted binary]", value)
    if isinstance(value, list):
        return [sanitize_trajectory(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_trajectory(item) for key, item in value.items()}
    return value


def classify_error(error: str | None) -> str:
    """Separate scored timeouts from excluded setup failures and retryable faults."""
    if not error:
        return "completed"
    if "AgentSetupTimeoutError" in error:
        return "excluded"
    if any(name in error for name in ("AgentTimeoutError", "VerifierTimeoutError")):
        return "timeout"
    statuses = [int(match) for match in _HTTP_STATUS.findall(error)]
    if statuses:
        return "retryable" if all(code in {401, 403, 408, 429} or code >= 500 for code in statuses) else "error"
    return "retryable" if _RETRYABLE.search(error) else "error"


def prepare(config: SkillEvolveConfig):
    """Validate ordered tasks and complete initial Skill sources before execution.

    Args:
        config: Dataset, environment, task selection, and initial sources.

    Returns:
        Environment, ordered samples, and task-keyed runtime Skill collections.
    """
    env = Environment.build(config.environment.class_name, config.environment.config)
    base = EnvTestConfig.model_validate(
        {key: value for key, value in config.model_dump().items() if key in EnvTestConfig.model_fields}
    )
    base.limit = None
    samples = _load_samples(base, env)
    selected = config.cases
    if config.cases_file is not None:
        rows = read_json(config.cases_file)
        selected = [row if isinstance(row, str) else row["name"] for row in rows]
    index = {sample.task.id: sample for sample in samples}
    if selected is not None:
        if len(set(selected)) != len(selected) or set(selected) - index.keys():
            raise ValueError("Case selection contains duplicate or unknown task IDs")
        samples = [index[key] for key in selected]
    samples = [sample for sample in samples if sample.task.id not in config.exclude_cases][: config.limit]
    if not samples:
        raise ValueError("No cases selected")
    skills = {}
    for sample in samples:
        if sample.task.id in {".", ".."} or "/" in sample.task.id or "\\" in sample.task.id:
            raise ValueError(f"Task ID is not a safe output directory name: {sample.task.id}")
        paths = config.case_skills.get(sample.task.id, config.skills)
        if not paths and config.environment.class_name == "skillsbench":
            root = sample.reward_data.task_dir / "environment/skills"
            paths = sorted(path for path in root.iterdir() if (path / "SKILL.md").is_file()) if root.is_dir() else []
        if not paths:
            raise ValueError(f"No initial Skills configured for {sample.task.id}")
        skills[sample.task.id] = {f"skill-{i:03d}": Skill.from_path(path) for i, path in enumerate(paths)}
    return env, samples, skills


def _progress(phase, slots, total, *, reused=0, retrying=0):
    """Print observed final slot counts separately from active retries."""
    counts = {
        status: sum(slot["status"] == status for slot in slots)
        for status in ("completed", "timeout", "error", "retryable", "excluded")
    }
    print(
        f"[{phase}] {len(slots)}/{total} done | ok {counts['completed']} | "
        f"timeout->0 {counts['timeout']} | errors {counts['error'] + counts['retryable']} | "
        f"excluded {counts['excluded']} | reused {reused} | retrying {retrying}",
        flush=True,
    )


async def _slot(config, env, sample, skills, directory):
    """Run or restore one scored slot, retaining unsuccessful attempts separately."""
    saved = directory / "slot.json"
    if saved.exists():
        return read_json(saved)
    for attempt in range(config.network_retries + 1):
        target = directory / "attempts" / f"attempt-{attempt:03d}"
        result_file = target / "results.json"
        if result_file.exists():
            outcome = CaseResult.model_validate(read_json(result_file)[0])
        elif target.exists():
            # Incomplete workspaces cannot be safely adopted as completed executions.
            target.rename(target.with_name(target.name + "_aborted_" + uuid4().hex[:8]))
            outcome = (await evaluate(env, [sample], SkillSet(skills), target, 1))[0]
        else:
            outcome = (await evaluate(env, [sample], SkillSet(skills), target, 1))[0]
        status = classify_error(outcome.error)
        if status == "retryable" and attempt < config.network_retries:
            print(f"[retry] {sample.task.id} retrying=1 attempt={attempt + 1} | {outcome.error}", flush=True)
            continue
        messages = None
        if status == "completed":
            record = next(read_jsonl(target / "trajectories.jsonl"))
            messages = record["result"]["trajectory"]
        data = dict(
            outcome=outcome.model_dump(mode="json"),
            status=status,
            score=0.0 if status == "timeout" else outcome.score,
            success=False if status == "timeout" else outcome.success,
            messages=messages,
        )
        write_json(saved, data)
        return data


async def _evolve_case(service, row, skills, state, state_path, project_id):
    """Register, report, and evolve one case, persisting each completed step."""
    case = row["case"]
    for index, (key, skill) in enumerate(skills.items()):
        if len(state["registrations"]) <= index:
            state["registrations"].append(await service.register(skill, f"{project_id}/{case}/{key}"))
            write_json(state_path, state)
    for repeat, slot in enumerate(row["collect"]):
        if not slot["messages"]:
            continue
        for registration in state["registrations"]:
            identity = f"{repeat}/{registration['skill_version_id']}"
            if identity not in state["reported"]:
                await service.report(
                    registration["skill_version_id"], sanitize_trajectory(slot["messages"]), f"{case}/{repeat}"
                )
                state["reported"].append(identity)
                write_json(state_path, state)
    if not state["members"]:
        state["members"] = await service.evolve(state["registrations"])
        write_json(state_path, state)


def _stats(slots):
    """Aggregate scored slots without treating missing scores as zero."""
    valid = [slot for slot in slots if slot["status"] in {"completed", "timeout"}]
    return dict(
        mean_score=sum(slot["score"] for slot in valid) / len(valid) if valid else None,
        accuracy=sum(bool(slot["success"]) for slot in valid) / len(valid) if valid else None,
        scored=len(valid),
        errors=len(slots) - len(valid),
    )


async def run(config: SkillEvolveConfig) -> dict:
    """Execute the matched protocol and save authoritative stage outcomes.

    Args:
        config: Validated collection, evolution, and testing protocol.

    Returns:
        Saved summary containing per-case and aggregate comparison outcomes.
    """
    env, samples, initial = prepare(config)
    directory = (
        (
            config.output_dir
            or ROOT / "output/mindmemos_eval/skill_evolve" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        .expanduser()
        .resolve()
    )
    directory.mkdir(parents=True, exist_ok=True)
    public = redact_credentials(config.model_dump(mode="json"))
    config_path = directory / "config.json"
    if config_path.exists() and read_json(config_path) != public:
        raise ValueError("Output directory belongs to a different configuration")
    write_json(config_path, public)
    project_file = directory / "project.json"
    project = read_json(project_file) if project_file.exists() else {"project_id": "eval-" + uuid4().hex}
    write_json(project_file, project)
    total = len(samples) * config.repeats
    collect_slots = []
    collect_reused = 0

    async def collect_case(sample):
        nonlocal collect_reused
        case_dir = directory / sample.task.id
        collect = []
        for repeat in range(config.repeats):
            target = case_dir / "collect" / f"repeat-{repeat:03d}"
            reused = (target / "slot.json").exists()
            slot = await _slot(config, env, sample, initial[sample.task.id], target)
            collect.append(slot)
            collect_slots.append(slot)
            collect_reused += int(reused)
            _progress("collect", collect_slots, total, reused=collect_reused)
        return dict(case=sample.task.id, collect=collect, test=[], status="pending", members=[])

    rows = await map_bounded(collect_case, samples, concurrency=config.concurrency)
    eligible = [
        row
        for row in rows
        if not any(slot["status"] == "excluded" for slot in row["collect"])
        and not (config.skip_perfect and _stats(row["collect"])["mean_score"] == 1)
        and any(slot["messages"] for slot in row["collect"])
    ]
    published = {}
    for count, row in enumerate(eligible, 1):
        case_config = config
        if config.pipeline == "simple" and config.connection == "inprocess":
            available = sum(bool(slot["messages"]) for slot in row["collect"])
            overrides = dict(config.pipeline_overrides)
            collection = dict(overrides.get("traj_collect", {}))
            collection.setdefault("sample_size", available)
            collection["trajectory_limit"] = max(
                collection.get("trajectory_limit", 100), available, collection["sample_size"]
            )
            overrides["traj_collect"] = collection
            case_config = config.model_copy(update={"pipeline_overrides": overrides})
        async with evolution_connection(case_config, directory, project["project_id"]) as service:
            case = row["case"]
            state_path = directory / case / "evolution.json"
            state = (
                read_json(state_path) if state_path.exists() else {"registrations": [], "reported": [], "members": []}
            )
            if state.get("failed"):
                row["status"] = "evolution_failed"
                row["evolution_error"] = state["failed"]
                continue
            try:
                # Saved progress makes each retry resume after the last completed step.
                for attempt in range(config.evolution_retries + 1):
                    try:
                        await _evolve_case(service, row, initial[case], state, state_path, project["project_id"])
                        break
                    except Exception as exc:
                        if attempt == config.evolution_retries:
                            raise
                        print(f"[evolve] retry {case} attempt={attempt + 1} | {type(exc).__name__}: {exc}", flush=True)
                row["members"] = state["members"]
                evolved = {}
                for member in state["members"]:
                    skill = Skill.from_snapshot(SkillSnapshot.model_validate(member["snapshot"]))
                    skill.export(directory / case / "published", overwrite=True)
                    evolved[member["skill_id"]] = skill
                original_sources = [skill.source for skill in initial[case].values()]
                changed = len(evolved) != len(original_sources) or any(
                    skill.source not in original_sources for skill in evolved.values()
                )
                row["status"] = "evolved" if changed else "unchanged"
            except Exception as exc:
                row["status"] = "evolution_failed"
                row["evolution_error"] = f"{type(exc).__name__}: {exc}"
                state["failed"] = row["evolution_error"]
                write_json(state_path, state)
                print(f"[evolve] failed | {case}: {row['evolution_error']}", flush=True)
            finished = [item["status"] for item in eligible[:count]]
            print(
                f"[evolve] {count}/{len(eligible)} done | succeeded {finished.count('evolved')} | "
                f"unchanged {finished.count('unchanged')} | failed {finished.count('evolution_failed')}",
                flush=True,
            )
            if row["status"] == "evolved":
                published[case] = evolved

    test_rows = [row for row in rows if row["status"] == "evolved"]
    test_total = len(test_rows) * config.repeats
    test_slots = []
    test_reused = 0
    sample_index = {sample.task.id: sample for sample in samples}

    async def test_case(row):
        nonlocal test_reused
        case = row["case"]
        for repeat in range(config.repeats):
            target = directory / case / "test" / f"repeat-{repeat:03d}"
            reused = (target / "slot.json").exists()
            slot = await _slot(config, env, sample_index[case], published[case], target)
            row["test"].append(slot)
            test_slots.append(slot)
            test_reused += int(reused)
            _progress("test", test_slots, test_total, reused=test_reused)

    await map_bounded(test_case, test_rows, concurrency=config.concurrency)
    for row in rows:
        if any(slot["status"] == "excluded" for slot in row["collect"] + row["test"]):
            row["status"] = "excluded"
        elif row["status"] == "pending":
            row["status"] = (
                "perfect" if config.skip_perfect and _stats(row["collect"])["mean_score"] == 1 else "no_trajectories"
            )
        row["collect_stats"] = _stats(row["collect"])
        row["test_stats"] = _stats(row["test"] if row["status"] == "evolved" else row["collect"])
        write_json(directory / row["case"] / "results.json", row)
    summary = _summary(rows, config)
    summary["output_dir"] = str(directory)
    write_json(directory / "summary.json", summary)
    _write_report(summary, directory)
    return summary


def _summary(rows, config):
    """Compute paired case means and execution token totals, including resumed slots."""
    included = [
        row
        for row in rows
        if row["status"] != "excluded"
        and row["collect_stats"]["mean_score"] is not None
        and row["test_stats"]["mean_score"] is not None
    ]
    metrics = {}
    for phase in ("collect", "test"):
        metrics[phase] = {
            key: sum(row[f"{phase}_stats"][key] for row in included) / len(included) if included else None
            for key in ("mean_score", "accuracy")
        }
        slots = [slot for row in rows for slot in row[phase]]
        usages = [slot["outcome"]["usage"] for slot in slots if slot["outcome"]["usage"] is not None]
        metrics[phase]["reported_usage"] = {
            key: sum(usage[key] for usage in usages) for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        metrics[phase]["usage_missing"] = len(slots) - len(usages)
        metrics[phase]["usage_reported"] = len(usages)
        metrics[phase]["usage_field_missing"] = {}
        metrics[phase]["average_usage"] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "cache_read_input_tokens", "cost_usd"):
            values = [usage[key] for usage in usages if usage.get(key) is not None]
            metrics[phase]["reported_usage"][key] = sum(values) if values else None
            metrics[phase]["average_usage"][key] = sum(values) / len(values) if values else None
            metrics[phase]["usage_field_missing"][key] = len(slots) - len(values)
    reference = read_json(config.reference_file) if config.reference_file else {}
    matched = [row for row in included if row["case"] in reference]
    reference_means = {
        key: sum(reference[row["case"]][key] for row in matched) / len(matched) if matched else None
        for key in ("baseline", "offline-evolution")
    }
    errors = sum(slot["status"] in {"error", "retryable"} for row in rows for slot in row["collect"] + row["test"])
    return dict(
        cases=rows,
        included_cases=len(included),
        metrics=metrics,
        errors=errors,
        reference=reference,
        reference_means=reference_means,
        reference_cases=len(matched),
    )


def _write_report(summary, directory):
    """Render the comparison without counting reused scores as fresh executions."""
    lines = [
        "# Skill evolution evaluation",
        "",
        f"Included cases: {summary['included_cases']}; execution errors: {summary['errors']}",
        "",
        "| Case | Initial mean | Evolved mean | Status | Reference initial | Reference evolved |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["cases"]:
        reference = summary["reference"].get(row["case"], {})
        lines.append(
            f"| {row['case']} | {row['collect_stats']['mean_score']} | {row['test_stats']['mean_score']} | {row['status']} | {reference.get('baseline', '-')} | {reference.get('offline-evolution', '-')} |"
        )
    for phase, metrics in summary["metrics"].items():
        lines.extend(["", f"## {phase}", "", str(metrics)])
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
