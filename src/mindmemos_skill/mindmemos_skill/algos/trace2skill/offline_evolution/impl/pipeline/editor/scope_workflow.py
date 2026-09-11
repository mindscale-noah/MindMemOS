"""Evolve workflow: one read+judge+edit agent session per mutable scope.

The former investigate and edit stages merged (docs §4): the same agent
reads the family evidence inline, verifies claims against raw trajectories,
edits ``skills/`` in place, and returns a structured manifest accounting for
every change. The harness then reconciles the manifest against the actual
tree diff -- bidirectional: an edited-but-unclaimed skill and a
claimed-but-unedited skill are both validation failures.

Guardrails preserved from the two-stage design: immutable out-of-workspace
baseline (re-seed + diff base), skill_md_only deterministic post-check,
skill format validation, one validation retry with concrete errors.
"""

from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import setting_int
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import invoke as llm_invoke
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm import stage_tools
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.invocation import InvokeError
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.artifacts import (
    scope_artifact_dir,
    scope_baseline_skills,
    scope_workspace,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.diff import (
    directory_diff,
    scope_tree_diff,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.skill_validation import (
    failed_skill_errors,
    validate_skill_tree,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.utils.files import write_json_file

# === System prompt ===


def evolve_system_prompt() -> str:
    """Constant contract: mission, materials, ordered procedure with rule gates."""
    return """You are the evolve agent for one scope's skill library. The corpus you receive contains real execution trajectories; some of them failed. Your job: find what the library could have taught that would have prevented those failures -- and teach it. An edit that stops a future agent from repeating an observed failure is success. An empty result is success only when you have read the flagged trajectories and the corpus genuinely teaches nothing.

## Materials

1. Task families (inlined): trajectories grouped by task similarity -- task lines, consulted skills, behaviors as ordered narrative summaries, adoption tables, behavior clusters. ALL of it is derived index from an earlier extraction pass: a map that tells you WHERE to look, never what to conclude. No change may rest on it directly -- evidence is only what you read in THIS session from the raw files. Score labels, where present, are mechanically joined verbatim from an external verifier: exact as labels, but a label is not a mechanism -- its use is telling you which raw trajectories to open first.
2. The skill library under skills/ -- open the SKILL.md files yourself; an inlined copy would go stale the moment you edit.
3. Raw trajectories under trajectory/ -- the source behind every summary.

## Procedure

Work through these steps in order.

STEP 1 -- SURVEY. From the index, list where the signal is. Flag every family that shows: failing score labels next to succeeding ones; behavior clusters whose member narratives end differently (one side errors or redoes, the other does not); divergent concrete outputs; or a skill consulted by some members and not others. This list is your investigation agenda -- state it before reading anything.

STEP 2 -- READ. Open the raw trajectories your agenda points at, failing side first. You are looking for what the failing side shares that the succeeding side lacks -- Grep across trajectory/ is cheap, use it. Do not skip this step: every conclusion you output later, positive or negative, must trace to files you opened here.

STEP 3 -- DECIDE. Test each agenda item against the evidence gates:
- Grounds for a change are ONLY verifiable facts from files you read: an explicit unfixed error, a concrete output divergence, a delivery without verification, or a skill never consulted. Frequency, similarity, cross-trajectory agreement, or "the users look better" are NOT grounds.
- Failing trajectories CONSULTED the skill and it already states the right rule -> re-emphasize that rule where the executor will see it before acting; do not add or rewrite.
- Failing trajectories NEVER consulted the skill -> the remedy is the frontmatter `description:` line, the only text every agent sees when deciding whether to open the skill (<=1024 chars, no angle brackets, never change `name:`); body additions are NOT a remedy -- the failing agents never reached that tier.
- Trajectories disagree on a concrete value, formula, or mapping -> do not pick a side, do not read majority agreement as validation; the change is a verification step at the exact point of disagreement.
- When you cite a score relation, state it separately from the trajectory facts, labels verbatim; never present a score as an in-trajectory event.
- Cannot verify -> drop it, and record the reason.

STEP 4 -- EDIT. Apply each accepted item to the library:
- Read the target SKILL.md first; you decide final position and wording.
- A concrete mechanism becomes a teaching section: the observable mechanism, a failing-vs-verified example pair only when your evidence establishes both sides, then the operational guidance. Weaker items stay one or two sentences in an existing section.
- Several items sharing one mechanism become ONE section; do not scatter near-duplicates. A new rule that contradicts an existing rule replaces it -- two contradictory rules in one skill is a hard failure.
- Rules must generalize: strip identifiers, file names, and values specific to one trajectory; keep the KIND of thing.
- Visibility tiers: frontmatter description > SKILL.md body > scripts//references//assets/. Place each remedy at the tier the failing agents would actually have seen; keep the body concise, route optional depth to references/ and stable procedures to scripts/ with observable activation conditions and exact relative paths.
- New skills: you alone decide whether the evidence shows a reusable capability nothing covers -- no support-count threshold, but never from an isolated behavior, a casual interaction, or an environment failure.
- Write in the language the skill document already uses (a new skill follows its evidence trajectories' language). Never delete a skill directory. Never invent a mechanism the evidence does not state; never add answers or values specific to a single task.

STEP 5 -- MANIFEST. Stop and return the structured manifest: one `changes` entry per skill you touched (action `create` or `update`, reasoning, evidence, a one-line edit summary) and one `dropped` entry per agenda item you did not act on, with the reason. Each change's `evidence.observations` is the comparable record behind it: one entry per group of trajectories -- the ids you opened and the fact you observed in those files, worded so a reviewer can find it there. Juxtaposed observations form the contrast; a single observation is legal for a self-contained failure; no polarity labels -- meaning lives in `reasoning`. Every skill you edited must appear in `changes`; every `changes` entry must correspond to a real edit. An empty `changes` list is legitimate only after STEP 2 -- your `dropped` entries then say what you read and what you failed to find.

Paths are cwd-relative; never prepend run or workspace prefixes.
"""


# === Manifest schema and validation ===


def evolve_manifest_schema() -> dict[str, Any]:
    """Provider-level schema for the closing manifest."""
    observation = {
        "type": "object",
        "properties": {
            "trajectory_ids": {"type": "array", "items": {"type": "string"}},
            "observed": {"type": "string"},
        },
        "required": ["trajectory_ids", "observed"],
        "additionalProperties": False,
    }
    evidence = {
        "type": "object",
        "properties": {
            "observations": {"type": "array", "items": observation},
        },
        "required": ["observations"],
        "additionalProperties": False,
    }
    change = {
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "action": {"type": "string", "enum": ["create", "update"]},
            "reasoning": {"type": "string"},
            "evidence": evidence,
            "edit_summary": {"type": "string"},
        },
        "required": ["skill", "action", "reasoning", "evidence", "edit_summary"],
        "additionalProperties": False,
    }
    dropped = {
        "type": "object",
        "properties": {
            "observation": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["observation", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "changes": {"type": "array", "items": change},
            "dropped": {"type": "array", "items": dropped},
        },
        "required": ["changes", "dropped"],
        "additionalProperties": False,
    }


def validate_manifest(
    manifest: Any,
    *,
    tree_diff: dict[str, Any],
    baseline_skills: set[str],
    trajectory_ids: set[str],
    skill_md_only: bool,
) -> list[str]:
    """Semantic + accounting checks; returns a flat error list (empty = pass)."""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]

    changes = manifest.get("changes")
    dropped = manifest.get("dropped")
    if not isinstance(changes, list):
        errors.append("manifest.changes must be a list")
        changes = []
    if not isinstance(dropped, list):
        errors.append("manifest.dropped must be a list")
        dropped = []

    claimed: dict[str, str] = {}
    for index, change in enumerate(changes):
        if not isinstance(change, dict):
            errors.append(f"changes[{index}] must be an object")
            continue
        skill = str(change.get("skill") or "").strip()
        action = str(change.get("action") or "").strip()
        if not skill:
            errors.append(f"changes[{index}].skill must be a non-empty string")
            continue
        if skill in claimed:
            errors.append(f"changes[{index}]: skill {skill!r} appears more than once; merge into one entry")
            continue
        if action not in ("create", "update"):
            errors.append(f"changes[{index}].action must be 'create' or 'update'")
            continue
        if action == "update" and skill not in baseline_skills:
            errors.append(f"changes[{index}]: action 'update' but {skill!r} is not an existing skill")
            continue
        if action == "create":
            if skill_md_only:
                errors.append(f"changes[{index}]: creating skills is forbidden in skill_md_only mode")
                continue
            if skill in baseline_skills:
                errors.append(f"changes[{index}]: action 'create' but {skill!r} already exists")
                continue
        if not str(change.get("reasoning") or "").strip():
            errors.append(f"changes[{index}].reasoning must be non-empty")
        if not str(change.get("edit_summary") or "").strip():
            errors.append(f"changes[{index}].edit_summary must be non-empty")
        evidence = change.get("evidence")
        observations = (evidence or {}).get("observations") if isinstance(evidence, dict) else None
        if not isinstance(observations, list) or not observations:
            errors.append(f"changes[{index}].evidence.observations must be a non-empty list")
        else:
            for o_index, observation in enumerate(observations):
                if not isinstance(observation, dict):
                    errors.append(f"changes[{index}].evidence.observations[{o_index}] must be an object")
                    continue
                ids = observation.get("trajectory_ids")
                if not isinstance(ids, list) or not ids:
                    errors.append(
                        f"changes[{index}].evidence.observations[{o_index}].trajectory_ids must be a non-empty list"
                    )
                else:
                    for raw_id in ids:
                        cid = str(raw_id).strip()
                        if cid not in trajectory_ids:
                            errors.append(
                                f"changes[{index}].evidence.observations[{o_index}]: "
                                f"{cid!r} is not a trajectory id in this scope"
                            )
                if not str(observation.get("observed") or "").strip():
                    errors.append(f"changes[{index}].evidence.observations[{o_index}].observed must be non-empty")
        claimed[skill] = action

    for index, entry in enumerate(dropped):
        if not isinstance(entry, dict):
            errors.append(f"dropped[{index}] must be an object")
            continue
        if not str(entry.get("observation") or "").strip():
            errors.append(f"dropped[{index}].observation must be non-empty")
        if not str(entry.get("reason") or "").strip():
            errors.append(f"dropped[{index}].reason must be non-empty")

    # Bidirectional accounting against the actual tree diff.
    actual: dict[str, str] = {}
    for skill_entry in tree_diff.get("skills") or []:
        name = str(skill_entry.get("skill_name") or "")
        status = str(skill_entry.get("status") or "")
        if status == "removed":
            errors.append(f"skill {name!r} was removed from the tree; deleting skills is forbidden -- restore it")
        elif status == "created":
            actual[name] = "create"
        elif status == "updated":
            actual[name] = "update"
    for name, action in actual.items():
        claimed_action = claimed.get(name)
        if claimed_action is None:
            errors.append(f"skill {name!r} was edited on disk but has no manifest.changes entry")
        elif claimed_action != action:
            errors.append(f"skill {name!r}: manifest says action {claimed_action!r} but the tree diff shows {action!r}")
    for name in claimed:
        if name not in actual:
            errors.append(f"manifest claims a change to {name!r} but the tree shows no edit to it")

    return errors


# === Family evidence rendering ===


def _render_member(member: dict[str, Any]) -> list[str]:
    lines = [f"#### `{member.get('trajectory_id')}`"]
    lines.append(f"- tasks: {member.get('tasks') or '(missing)'}")
    skills_used = member.get("skills_used") or []
    lines.append(f"- skills consulted: {', '.join(skills_used) if skills_used else 'none'}")
    if member.get("score_outcome") is not None:
        lines.append(f"- external verifier score (verbatim): {member['score_outcome']}")
    if member.get("trajectory_truncated"):
        lines.append("- note: the extraction saw a truncated view of this trajectory")
    lines.append("- behaviors, in execution order:")
    for behavior in member.get("behaviors") or []:
        index = behavior.get("behavior_index")
        summary = str(behavior.get("summary") or "").strip()
        lines.append(f"  {index}. {summary}")
    return lines


def _render_family(family: dict[str, Any]) -> str:
    lines = [f"### Family {family.get('family_id')} ({len(family.get('members') or [])} trajectories)", ""]
    for member in family.get("members") or []:
        lines.extend(_render_member(member))
        lines.append("")
    adoption = family.get("adoption") or {}
    if adoption:
        lines.append("Adoption (per skill: who consulted it, who did not):")
        for skill in sorted(adoption):
            entry = adoption[skill] or {}
            used = ", ".join(entry.get("used") or []) or "-"
            not_used = ", ".join(entry.get("not_used") or []) or "-"
            lines.append(f"- {skill}: used by [{used}]; NOT used by [{not_used}]")
        lines.append("")
    clusters = family.get("behavior_clusters") or []
    multi = [c for c in clusters if len(c.get("members") or []) > 1]
    if multi:
        lines.append("Behavior clusters (comparable moves across trajectories; refs are trajectory#behavior):")
        for cluster in multi:
            refs = ", ".join(
                f"{m.get('trajectory_id')}#{m.get('behavior_index')}" for m in cluster.get("members") or []
            )
            lines.append(f"- cluster {cluster.get('cluster_id')}: {refs}")
        lines.append("")
    return "\n".join(lines)


def render_families_block(
    patterns_payload: dict[str, Any],
    *,
    char_budget: int,
) -> str:
    """Render families as markdown; overflow degrades WHOLE families to pointers.

    Family-level degradation preserves the no-selection-bias property: a
    family is either fully inline or fully behind its Read pointer.
    """
    families = patterns_payload.get("families") or []
    rendered: list[str] = []
    pointered: list[int] = []
    spent = 0
    for family in families:
        block = _render_family(family)
        if spent + len(block) > char_budget and rendered:
            pointered.append(int(family.get("family_id", -1)))
            continue
        rendered.append(block)
        spent += len(block)
    lines = ["## Task families", ""]
    lines.extend(rendered)
    if pointered:
        lines.append(
            "The following families were omitted from this view for prompt budget; "
            "read them in full at `patterns.json`: " + ", ".join(f"family {fid}" for fid in pointered)
        )
    return "\n".join(lines)


# === Prompt assembly ===


def _evolve_user_prompt(
    *,
    families_block: str,
    skill_names: list[str],
    skill_md_only: bool,
    supervised: bool,
    validation_errors: list[str] | None = None,
) -> str:
    skills_line = (
        "skills/<name>/SKILL.md -- one per skill, the always-visible core; supporting files are frozen"
        if skill_md_only
        else "skills/ -- the skill library: each skill is SKILL.md plus optional scripts/, references/, assets/"
    )
    entry_lines = [
        "Entry points:",
        f"- {skills_line}",
        "- trajectory/ -- raw trajectories, one JSON file per run",
        "- patterns.json -- the full family/adoption/cluster structure (this message inlines it, minus any noted omissions)",
    ]
    if supervised:
        entry_lines.append("- scores/ -- external verifier scores, one file per trajectory (verbatim labels)")
    edit_scope = (
        "Edit only " + ", ".join(f"skills/{name}/SKILL.md" for name in skill_names) + ". "
        "Do not create, remove, or rename skills; supporting files are frozen."
        if skill_md_only
        else "The editable tree is skills/."
    )
    parts = [
        "\n".join(entry_lines),
        "",
        families_block,
        "",
        f"Skill library of this scope: {', '.join(skill_names) if skill_names else '(empty)'}",
        edit_scope,
    ]
    if validation_errors:
        parts.extend(
            [
                "",
                "The previous candidate failed validation. Keep the current edits where they are correct and fix every error below, then return the complete manifest again:",
                *[f"- {error}" for error in validation_errors],
            ]
        )
    parts.extend(
        [
            "",
            "RECAP -- 1 survey the index into an agenda; 2 read the flagged raw trajectories, "
            "failing side first; 3 decide each item against the evidence gates; 4 edit what "
            "survives; 5 return the manifest accounting for every edit and every drop.",
        ]
    )
    return "\n".join(parts)


# === Filesystem helpers ===


def _seed_scope_skills(baseline: Path, skills: Path) -> None:
    """Reset the editable ``skills/`` tree from the out-of-workspace baseline."""
    import shutil

    if not baseline.is_dir():
        raise FileNotFoundError(f"skills baseline not found: {baseline}")
    if skills.exists():
        shutil.rmtree(skills)
    skills.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(baseline, skills)


def _skill_names(skills_dir: Path) -> list[str]:
    if not skills_dir.is_dir():
        return []
    return sorted(path.name for path in skills_dir.iterdir() if path.is_dir() and not path.name.startswith("."))


def _skill_write_roots(scope_root: Path, skill_names: list[str], *, skill_md_only: bool) -> list[str]:
    skills = scope_root / "skills"
    if not skill_md_only:
        return [str(skills)]
    # ReAct write roots must be directories. The prompt narrows this to
    # SKILL.md and _enforce_skill_md_only restores every other file change.
    return [str(skills / name) for name in skill_names]


def _enforce_skill_md_only(baseline: Path, skills: Path) -> list[str]:
    """Keep candidate bytes only for existing ``<skill>/SKILL.md`` files."""
    skill_names = _skill_names(baseline)
    allowed = {f"{name}/SKILL.md" for name in skill_names}
    candidate_diff = directory_diff(baseline, skills)
    violations = [
        f"{item['status']}:{item['path']}"
        for item in candidate_diff["changed_files"]
        if item["path"] not in allowed or item["status"] != "modified"
    ]

    source_dirs = set(skill_names)
    evolved_dirs = set(_skill_names(skills))
    violations.extend(f"created_skill:{name}" for name in sorted(evolved_dirs - source_dirs))
    violations.extend(f"removed_skill:{name}" for name in sorted(source_dirs - evolved_dirs))

    candidate_skill_md: dict[str, bytes] = {}
    for name in skill_names:
        source_md = baseline / name / "SKILL.md"
        candidate_md = skills / name / "SKILL.md"
        if source_md.is_file() and candidate_md.is_file():
            candidate_skill_md[name] = candidate_md.read_bytes()

    _seed_scope_skills(baseline, skills)
    for name, content in candidate_skill_md.items():
        (skills / name / "SKILL.md").write_bytes(content)
    return sorted(set(violations))


def _trajectory_ids(scope_root: Path) -> set[str]:
    trajectory_dir = scope_root / "trajectory"
    if not trajectory_dir.is_dir():
        return set()
    return {path.stem for path in trajectory_dir.iterdir() if path.is_file()}


def _load_patterns(scope_root: Path) -> dict[str, Any]:
    path = scope_root / "patterns.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


# === Invocation ===


def _families_char_budget() -> int:
    return max(1, setting_int("evolve.families_char_budget"))


def _invoke_evolve_agent(
    *,
    scope_id: str,
    scope_root: Path,
    skill_names: list[str],
    skill_md_only: bool,
    supervised: bool,
    families_block: str,
    validation_errors: list[str] | None = None,
) -> tuple[dict[str, Any] | None, Any, list[str]]:
    """Run one evolve session; return (manifest, invoke result, parse errors)."""
    result = llm_invoke(
        evolve_system_prompt(),
        _evolve_user_prompt(
            families_block=families_block,
            skill_names=skill_names,
            skill_md_only=skill_md_only,
            supervised=supervised,
            validation_errors=validation_errors,
        ),
        tools=stage_tools(["Read", "Glob", "Grep", "Edit", "Write"], with_bash=True),
        output_schema=evolve_manifest_schema(),
        permission_mode="dontAsk",
        cwd=str(scope_root),
        read_roots=[str(scope_root)],
        write_roots=_skill_write_roots(scope_root, skill_names, skill_md_only=skill_md_only),
        timeout_stage="evolve",
        trace_name=("evolve/" + ("validation-retry/invoke" if validation_errors else "invoke")),
        logs_root=scope_artifact_dir(scope_root.parent, scope_id, "logs"),
    )
    try:
        manifest = json.loads(result.text or "")
    except json.JSONDecodeError as exc:
        return None, result, [f"manifest is not valid JSON: {exc}"]
    if not isinstance(manifest, dict):
        return None, result, ["manifest must be a JSON object"]
    return manifest, result, []


def _attempts(result: Any = None, error: Exception | None = None) -> int:
    if result is not None:
        return int(getattr(result, "attempts", 1) or 1)
    if isinstance(error, InvokeError):
        return int(error.attempts or 1)
    return 1


def _write_review(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(str(path), payload)


def evolve_scope(
    workspace_root: str | Path,
    scope_id: str,
    *,
    baseline_root: str | Path,
    skill_md_only: bool = False,
    review_path: str | Path,
) -> dict[str, Any]:
    """Evolve one scope's skill library; write review/<scope>.json.

    A candidate that fails validation (skill format OR manifest accounting)
    receives its concrete errors and one repair attempt.
    """
    workspace = Path(workspace_root)
    scope_root = scope_workspace(workspace, scope_id)
    baseline = scope_baseline_skills(baseline_root, scope_id)
    skills = scope_artifact_dir(workspace, scope_id, "skills")
    _seed_scope_skills(baseline, skills)

    skill_names = _skill_names(baseline)
    trajectory_ids = _trajectory_ids(scope_root)
    patterns_payload = _load_patterns(scope_root)
    supervised = (scope_root / "scores").is_dir()
    families_block = render_families_block(patterns_payload, char_budget=_families_char_budget())

    attempts = 0
    validation_retries = 0
    manifest: dict[str, Any] | None = None
    constraint_violations: list[str] = []
    tree_validation: dict[str, Any] = {}
    diff: dict[str, Any] = {}

    def run_once(errors: list[str] | None) -> tuple[dict[str, Any] | None, list[str]]:
        nonlocal attempts, constraint_violations, tree_validation, diff
        current_manifest, invoke_result, parse_errors = _invoke_evolve_agent(
            scope_id=scope_id,
            scope_root=scope_root,
            skill_names=skill_names,
            skill_md_only=skill_md_only,
            supervised=supervised,
            families_block=families_block,
            validation_errors=errors,
        )
        attempts += _attempts(invoke_result)
        if skill_md_only:
            constraint_violations.extend(_enforce_skill_md_only(baseline, skills))
        tree_validation = validate_skill_tree(baseline, skills)
        diff = scope_tree_diff(baseline, skills)
        all_errors = list(parse_errors)
        for skill_name, skill_errors in sorted(failed_skill_errors(tree_validation).items()):
            all_errors.extend(f"skills/{skill_name}: {error}" for error in skill_errors)
        if current_manifest is not None:
            all_errors.extend(
                validate_manifest(
                    current_manifest,
                    tree_diff=diff,
                    baseline_skills=set(skill_names),
                    trajectory_ids=trajectory_ids,
                    skill_md_only=skill_md_only,
                )
            )
        return current_manifest, all_errors

    try:
        manifest, errors = run_once(None)
        if errors:
            validation_retries = 1
            manifest, errors = run_once(errors)
    except Exception as exc:
        attempts += _attempts(error=exc)
        _seed_scope_skills(baseline, skills)
        diff = scope_tree_diff(baseline, skills)
        payload = {
            "scope_id": scope_id,
            "status": "reverted",
            "error": f"evolve agent failed: {type(exc).__name__}: {exc}",
            "attempts": attempts,
            "validation_retries": validation_retries,
            "validation": validate_skill_tree(baseline, skills),
            "constraint_violations": sorted(set(constraint_violations)),
            "manifest": manifest,
            "diff": diff,
        }
        _write_review(Path(review_path), payload)
        return payload

    if errors:
        # Validation exhausted: revert to baseline rather than export an
        # unaccounted tree.
        _seed_scope_skills(baseline, skills)
        diff = scope_tree_diff(baseline, skills)
        payload = {
            "scope_id": scope_id,
            "status": "reverted",
            "error": "validation failed after retry: " + "; ".join(errors[:5]),
            "attempts": attempts,
            "validation_retries": validation_retries,
            "validation": validate_skill_tree(baseline, skills),
            "constraint_violations": sorted(set(constraint_violations)),
            "manifest": manifest,
            "diff": diff,
        }
        _write_review(Path(review_path), payload)
        return payload

    changed = bool(diff.get("summary", {}).get("changed_files"))
    payload = {
        "scope_id": scope_id,
        "status": "edited" if changed else "unchanged",
        "attempts": attempts,
        "validation_retries": validation_retries,
        "validation": tree_validation,
        "constraint_violations": sorted(set(constraint_violations)),
        "manifest": manifest,
        "diff": diff,
    }
    _write_review(Path(review_path), payload)
    return payload


def run_scope_evolve_agents(
    workspace_root: str | Path,
    *,
    scope_ids: list[str],
    baseline_root: str | Path,
    llm_parallelism: int = 1,
    skill_md_only: bool = False,
    review_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_root)
    reviews = Path(review_root) if review_root is not None else workspace.parent / "review"
    results: dict[str, dict[str, Any]] = {}
    # Worker threads start from an empty context; run each scope in a copy of
    # the caller's context so the bound backend is visible inside the pool.
    parent_context = contextvars.copy_context()

    def run_one(scope_id: str) -> dict[str, Any]:
        return parent_context.copy().run(
            evolve_scope,
            workspace,
            scope_id,
            baseline_root=baseline_root,
            skill_md_only=skill_md_only,
            review_path=reviews / f"{scope_id}.json",
        )

    with ThreadPoolExecutor(max_workers=max(1, llm_parallelism)) as executor:
        futures = {executor.submit(run_one, scope_id): scope_id for scope_id in scope_ids}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    ordered = [results[scope_id] for scope_id in sorted(results)]
    return {
        "scopes": ordered,
        "summary": {
            "scopes": len(ordered),
            "edited": sum(item["status"] == "edited" for item in ordered),
            "unchanged": sum(item["status"] == "unchanged" for item in ordered),
            "reverted": sum(item["status"] == "reverted" for item in ordered),
            "attempts": sum(int(item.get("attempts") or 0) for item in ordered),
            "validation_retries": sum(int(item.get("validation_retries") or 0) for item in ordered),
            "constraint_violations": sum(len(item.get("constraint_violations") or []) for item in ordered),
        },
    }


__all__ = [
    "evolve_manifest_schema",
    "evolve_scope",
    "evolve_system_prompt",
    "render_families_block",
    "run_scope_evolve_agents",
    "validate_manifest",
]
