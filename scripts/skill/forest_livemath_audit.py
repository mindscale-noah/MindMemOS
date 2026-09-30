"""Compare LiveMath reflection inputs against historical source without model calls."""

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.agents.typing import AgentResult
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter
from mindmemos_skill.algos.train.skill_forest.typing import SkillForestReflectionContext, SkillForestTrajectoryContext
from mindmemos_skill.envs.builtin_envs.livemath.config import LiveMathConfig, LiveMathDataConfig
from mindmemos_skill.envs.builtin_envs.livemath.data import LiveMathData
from mindmemos_skill.envs.builtin_envs.livemath.reward import evaluate
from mindmemos_skill.envs.builtin_envs.livemath.rollout import LiveMathEnvironment
from mindmemos_skill.envs.typing import RolloutResult
from mindmemos_skill.skill import Skill, SkillBundle


def render_trajectories(batch):
    adapter = SkillForestEnvAdapter.build("livemath", {})
    return "\n\n---\n\n".join(
        adapter.render_trajectory(SkillForestTrajectoryContext(task=sample.task, result=result, index=index))
        for index, (sample, result) in enumerate(batch, 1)
    )


def reflection_user(skill, batch, *, success, edit_budget):
    return SkillForestEnvAdapter.build("livemath", {}).build_reflection_messages(
        SkillForestReflectionContext(
            skill=skill,
            trajectories=tuple(
                SkillForestTrajectoryContext(task=sample.task, result=result, index=index)
                for index, (sample, result) in enumerate(batch, 1)
            ),
            success=success,
            edit_budget=edit_budget,
        )
    )[1]["content"]


def legacy_renderers(path):
    """Compile only historical formatting code, cutting off all optimizer calls.

    Args:
        path: Historical reflection implementation to audit.

    Returns:
        Source-derived formatting functions under the fixed patch-only protocol.
    """
    names = {
        "_clip_text",
        "fmt_trajectory",
        "fmt_minibatch_trajectories",
        "run_error_analyst_minibatch",
        "run_success_analyst_minibatch",
    }
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            if node.name.startswith("run_"):
                stop = next(index for index, statement in enumerate(node.body) if isinstance(statement, ast.Try))
                node.body = node.body[:stop] + [ast.parse("return {'system': actual_system, 'user': user}").body[0]]
            nodes.append(node)
    namespace = {
        "os": os,
        "json": json,
        "normalize_update_mode": lambda mode: mode,
        "is_full_rewrite_minibatch_mode": lambda mode: False,
        "_resolve_prompt": lambda custom, name, mode: custom,
        "payload_label": lambda mode, title=False: "Edits" if title else "edits",
        "format_meta_skill_context": lambda value: "",
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def main():
    """Write exact input comparisons and provenance without invoking providers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-project", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--choice-seed", type=int, default=10042, help="Historical shared validation uses 42 + 10000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.legacy_project / "skillopt/gradient/reflect.py"
    legacy = legacy_renderers(source)
    root = args.reference_run / "forest"
    predictions = root / "shared_root_validation_rollout/predictions"
    rows = [json.loads(line) for line in (predictions.parent / "results.jsonl").read_text().splitlines()]
    pool = {
        sample.task.id: sample
        for sample in LiveMathData(LiveMathDataConfig(data_path=args.data_path, choice_seed=args.choice_seed)).load()
    }
    skill = (root / "trees/tree_000/nodes/t000-n000000/skill.md").read_text()
    comparisons = []
    paired = []
    for index, row in enumerate(rows):
        sample = pool[row["id"]]
        historical = predictions / row["id"]
        system = (historical / "target_system_prompt.txt").read_text()
        user = (historical / "target_user_prompt.txt").read_text()
        reward = evaluate(row["response"], sample.task, sample.reward_data)
        result = RolloutResult(
            output=row["response"],
            reward=reward,
            trajectory=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": row["response"]},
            ],
        )
        batch = [(sample, result)]
        paired.append((row, batch[0]))
        success = bool(row["hard"])
        stem = "analyst_success" if success else "analyst_error"
        old_system = (args.legacy_project / f"skillopt/envs/livemathematicianbench/prompts/{stem}.md").read_text()
        old = legacy["run_success_analyst_minibatch" if success else "run_error_analyst_minibatch"](
            skill,
            [row],
            str(predictions),
            edit_budget=4,
            system_prompt=old_system,
        )
        new = {
            "system": (
                SkillForestEnvAdapter.build("livemath", {}).prompts.analyst_success
                if success
                else SkillForestEnvAdapter.build("livemath", {}).prompts.analyst_error
            ),
            "user": reflection_user(skill, batch, success=success, edit_budget=4),
        }
        folder = args.output / f"{index:03d}"
        folder.mkdir()
        agent = Mock(config=OpenAIChatConfig(model="offline-replay", api_key="offline-replay"))
        agent.run.return_value = AgentResult(output=row["response"])
        env = LiveMathEnvironment(LiveMathConfig(agent=agent.config), agent=agent)
        env._run(
            task=sample.task,
            reward_data=sample.reward_data,
            skills=[Skill(bundle=SkillBundle(files={"SKILL.md": skill}))],
            workspace=folder,
        )
        target = agent.run.call_args.kwargs
        for name, payload in [("legacy", old), ("current", new)]:
            (folder / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        difference = "".join(
            difflib.unified_diff(
                old["user"].splitlines(True), new["user"].splitlines(True), fromfile="legacy", tofile="current"
            )
        )
        (folder / "user.diff").write_text(difference)
        comparisons.append(
            {
                "task_id": row["id"],
                "system_equal": old["system"] == new["system"],
                "user_equal": old["user"] == new["user"],
                "reward_equal": reward.success == success,
                "target_system_equal": target["system"] == system,
                "target_user_equal": target["user"] == user,
                "verification_present": "[verification] [EVALUATION RESULT]" in render_trajectories(batch),
            }
        )
    batch_comparisons = []
    for success in (False, True):
        selected = [(row, pair) for row, pair in paired if bool(row["hard"]) == success]
        for start in range(0, len(selected), 8):
            chunk = selected[start : start + 8]
            old_text = legacy["fmt_minibatch_trajectories"]([row for row, pair in chunk], str(predictions))
            new_text = render_trajectories([pair for row, pair in chunk])
            batch_comparisons.append({"success": success, "count": len(chunk), "equal": old_text == new_text})
    report = {
        "mode": "offline historical-response replay; no provider calls",
        "scope": "Reflection inputs from recorded target context; not a fresh end-to-end training run",
        "legacy_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference_run": str(args.reference_run),
        "count": len(comparisons),
        "choice_seed": args.choice_seed,
        "all_equal": all(
            row["system_equal"]
            and row["user_equal"]
            and row["reward_equal"]
            and row["target_system_equal"]
            and row["target_user_equal"]
            for row in comparisons
        )
        and all(row["equal"] for row in batch_comparisons),
        "comparisons": comparisons,
        "batch_comparisons": batch_comparisons,
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
