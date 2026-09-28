"""Run one frozen seven-tree LiveMath Select pilot with explicit paper overrides."""

import argparse
import asyncio
import hashlib
import json
import traceback
from pathlib import Path

from dotenv import load_dotenv
from mindmemos_skill.agents import OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.skill_forest.config import SkillForestTrainerConfig
from mindmemos_skill.algos.train.skill_forest.generation import generate_patch
from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime, parallel
from mindmemos_skill.algos.train.skill_forest.trainer import SkillForestTrainer
from mindmemos_skill.algos.train.skill_forest.tree import mean_score
from mindmemos_skill.algos.train.skill_forest.typing import TreeSplit
from mindmemos_skill.common.json import write_json
from mindmemos_skill.envs.builtin_envs.livemath.config import LiveMathConfig, LiveMathDataConfig
from mindmemos_skill.envs.builtin_envs.livemath.data import LiveMathData
from mindmemos_skill.envs.builtin_envs.livemath.rollout import LiveMathEnvironment
from mindmemos_skill.skill import Skill, SkillBundle


class PaperLiveMathTrainer(SkillForestTrainer):
    """Keep historical split membership, choice seeds and edit selection explicit."""

    def __init__(self, config, *, reference_run: Path, data_path: Path):
        """Bind one historical seed-42 seven-tree reference and raw benchmark.

        Args:
            config: Global algorithm and model limits.
            reference_run: Completed historical seven-tree seed-42 experiment.
            data_path: Official local monthly benchmark files.
        """
        super().__init__(config)
        self.reference_run = reference_run
        self.data_path = data_path
        self.pools = {}

    def pool(self, seed):
        """Materialize the historical role-specific option permutation.

        Args:
            seed: Historical per-tree or evaluation option seed.

        Returns:
            Samples indexed by stable benchmark task IDs.
        """
        if seed not in self.pools:
            data = LiveMathData(LiveMathDataConfig(data_path=self.data_path, choice_seed=seed))
            self.pools[seed] = {sample.task.id: sample for sample in data.load()}
        return self.pools[seed]

    def create_splits(self, task_ids):
        """Load frozen paper memberships instead of reinterpreting tree seeds.

        Args:
            task_ids: Complete released training-pool IDs.

        Returns:
            Seven ordered train/validation splits verified against that pool.
        """
        splits = []
        for index in range(self.config.num_trees):
            manifest = json.loads(
                (self.reference_run / f"forest/trees/tree_{index:03d}/data_manifest.json").read_text()
            )
            split = TreeSplit(
                train_ids=tuple(manifest["train_source_ids"]), validation_ids=tuple(manifest["validation_ids"])
            )
            if set(split.train_ids) & set(split.validation_ids) or set(split.train_ids + split.validation_ids) != set(
                task_ids
            ):
                raise ValueError("Historical tree split does not match the supplied training pool")
            splits.append(split)
        return splits

    async def _collect(self, skill, ids, samples, env, runtime, directory):
        """Apply the historical option seed for roots, training and full-pool scoring."""
        if directory.name == "root_validation":
            seed = self.config.seed + 10000
        elif directory.name == "train" and directory.parent.name.startswith("tree-"):
            seed = self.config.seed + int(directory.parent.name.split("-")[1]) * 1000003
        else:
            seed = self.config.seed
        return await super()._collect(skill, ids, self.pool(seed), env, runtime, directory)

    async def _candidate(self, parent, index, split, samples, evidence, env, runtime, root):
        """Use the tree-local validation choice seed without changing stored evidence."""
        validation = self.pool(self.config.seed + parent.tree_id * 1000003 + 10000)
        return await super()._candidate(parent, index, split, validation, evidence, env, runtime, root)

    async def _generate_patch(self, parent, index, split, samples, evidence, runtime, directory):
        """Use the historical proposal seed and depth-dependent optimizer edit ranking."""
        tree_seed = self.config.seed + parent.tree_id * 1000003
        training = self.pool(tree_seed)
        return await generate_patch(
            skill=parent.skill,
            samples=[training[key] for key in split.train_ids],
            results=[evidence[key] for key in split.train_ids],
            runtime=runtime,
            directory=directory,
            seed=tree_seed + parent.depth * 100000 + (index + 1) * 10000019,
            depth=parent.depth,
        )

    async def _ensemble(self, initial, winners, samples, env, runtime, root):
        """Select by the historical full-training score and deterministic tie order."""

        async def score(node):
            results = await self._collect(
                node.skill, list(samples), samples, env, runtime, root / "ensemble" / node.id / "score"
            )
            scores = {key: float(result.reward.success) for key, result in results.items()}
            write_json(root / "ensemble" / node.id / "scores.json", {"scores": scores, "mean": mean_score(scores)})
            return mean_score(scores)

        values = await parallel(score(node) for node in winners)
        ranking = sorted(
            range(len(winners)),
            key=lambda index: (
                -values[index],
                -mean_score(winners[index].scores),
                winners[index].depth,
                len(winners[index].skill.bundle.files["SKILL.md"]),
                winners[index].id,
                winners[index].tree_id,
            ),
        )
        selected = winners[ranking[0]]
        write_json(root / "ensemble/selection.json", {"selected": selected.id})
        write_json(
            root / "ensemble/paper_ranking.json",
            [{"id": winners[index].id, "train_hard": values[index]} for index in ranking],
        )
        return selected.skill


async def run(args):
    """Validate, probe or execute the explicitly authorized single-run protocol."""
    load_dotenv(args.env_file, override=True)
    options = dict(
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        timeout_s=21600,
        max_completion_tokens=16384,
        max_retries=0,
        sdk_max_retries=0,
        parse_max_attempts=1,
    )
    config = SkillForestTrainerConfig(
        optimizer=options,
        num_trees=7,
        expansion_budget=30,
        max_depth=3,
        candidates_per_node=3,
        rollout_concurrency=40,
        model_concurrency=40,
        expansion_concurrency=40,
        env_adapter={"class_name": "livemath"},
        root_share_train_trajectory=False,
        ranking_enabled=True,
        ranking_edit_budgets=(4, 3, 2),
    )
    trainer = PaperLiveMathTrainer(config, reference_run=args.reference_run, data_path=args.data_path)
    split = json.loads(args.splits.read_text())
    pool = trainer.pool(42)
    training = [pool[key] for key in split["train"]]
    validation = [pool[key] for key in split["val"]]
    testing = [pool[key] for key in split["test"]]
    if [len(training), len(validation), len(testing)] != [35, 18, 124]:
        raise ValueError("Pilot requires the frozen 35/18/124 benchmark split")
    trainer.create_splits([sample.task.id for sample in training])
    skill_path = args.reference_run / "forest/trees/tree_000/nodes/t000-n000000/skill.md"
    skill = Skill(bundle=SkillBundle(files={"SKILL.md": skill_path.read_text()}))
    agent_config = OpenAIChatConfig(**options)
    plan = {
        "model": agent_config.model,
        "reasoning_effort": "medium",
        "seed": 42,
        "algorithm": config.model_dump(mode="json", exclude={"optimizer": {"api_key"}}),
        "counts": {"train": 35, "validation_unused": 18, "test_per_repeat": 124, "test_repeats": 3},
        "candidate_edit_budgets": list(config.ranking_edit_budgets),
        "tree_seed_stride": 1000003,
        "validation_choice_seed_offset": 10000,
        "reference_run": str(args.reference_run),
        "initial_skill_sha256": hashlib.sha256(skill_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        "reference_select_hard": 0.3602150537634409,
        "intentional_differences": [
            "global round-robin rollout scheduling",
            "no provider or parse retries",
            "hierarchical node IDs used for final stable tie order",
            "invalid ranking fails instead of silent truncation",
        ],
    }
    if args.mode == "preflight":
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "plan.json", plan)
        print(
            json.dumps({"status": "preflight_passed", "output": str(args.output), "plan": plan}, ensure_ascii=False),
            flush=True,
        )
        return
    if args.resume:
        if args.mode != "run":
            raise ValueError("Only run mode supports resume")
        if json.loads((args.output / "plan.json").read_text()) != plan:
            raise ValueError("Pilot recovery requires the original effective plan")
    elif args.output.exists():
        raise ValueError("Live execution requires a fresh output root")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    write_json(args.output / "plan.json", plan)
    if args.mode == "probe":
        agent = OpenAIChatAgent(agent_config)
        response = await asyncio.to_thread(agent.run, system="You are a concise assistant.", user="Reply with OK.")
        write_json(args.output / "probe.json", response.model_dump(mode="json"))
        if not isinstance(response.output, str) or not response.output.strip():
            raise ValueError("The model probe returned no text")
        print(json.dumps({"status": "probe_passed", "nonempty": bool(response.output)}), flush=True)
        return
    write_json(args.output / "pilot_status.json", {"status": "running", "stage": "training"})
    env = LiveMathEnvironment(LiveMathConfig(agent=agent_config))
    try:
        selected = await trainer.train(
            skill=skill,
            env=env,
            train_samples=training,
            validation_samples=validation,
            work_dir=args.output / "train",
        )
        selected.export(args.output / "selected", overwrite=args.resume)
        write_json(args.output / "pilot_status.json", {"status": "running", "stage": "test"})
        repeats = []
        runtime = ForestRuntime(config, work_dir=args.output / "evaluation", resume=args.resume)
        try:
            for index in range(1, 4):
                results = await runtime.scheduler.collect(
                    env=env, samples=testing, skill=selected, directory=args.output / f"test/run_{index:02d}"
                )
                if len(results) != 124:
                    raise RuntimeError("Held-out test is incomplete")
                scores = {key: float(result.reward.success) for key, result in results.items()}
                row = {"repeat": index, "count": len(scores), "hard": mean_score(scores), "scores": scores}
                write_json(args.output / f"test/run_{index:02d}/summary.json", row)
                repeats.append(row)
        finally:
            await runtime.close()
        summary = {
            "status": "completed",
            "n_results": sum(row["count"] for row in repeats),
            "expected_results": 372,
            "hard": sum(row["hard"] for row in repeats) / 3,
            "reference_hard": plan["reference_select_hard"],
            "per_run": repeats,
        }
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "pilot_status.json", {"status": "completed"})
        print(json.dumps({key: value for key, value in summary.items() if key != "per_run"}), flush=True)
    except BaseException as exc:
        write_json(args.output / "pilot_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        raise


def main():
    """Parse explicit paths without embedding credentials in experiment artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true", help="Resume a compatible run without repeating completed calls"
    )
    parser.add_argument("--mode", choices=["preflight", "probe", "run"], required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
