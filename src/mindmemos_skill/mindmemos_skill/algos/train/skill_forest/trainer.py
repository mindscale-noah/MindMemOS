"""Grow a bounded Skill forest using node-level recovery and compact outputs."""

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from filelock import AsyncFileLock
from pydantic import JsonValue

from ....common.checkpoint import fingerprint
from ....common.json import read_json, read_jsonl, write_json
from ....common.secrets import redact_credentials
from ....envs.base import Environment
from ....envs.data.typing import DataSample
from ....envs.typing import RolloutResult
from ....logging import get_logger, log_context
from ....skill import Skill
from ...base import Algorithm
from ..base import SkillTrainer
from ..recovery import sample_identity
from .checkpoint import load_node, load_skill, node_directory, read_node, read_node_file, write_node
from .config import ALGORITHM_VERSION, SkillForestTrainerConfig
from .env_adapter import SkillForestEnvAdapter
from .env_specific.selection import select_env_adapter
from .generation import InvalidPatchError, generate_patch
from .runtime import ForestRuntime, parallel
from .synthesis import split_merge_content
from .tree import apply_patch, cannot_improve, expansion_wave, mean_score, novelty, split_tasks, tree_winners
from .typing import (
    EnsembleCandidate,
    Node,
    NodeCheckpoint,
    NodeSummary,
    Patch,
    SkillForestSkillMergeContext,
    TreeSplit,
    WinnerScore,
)

logger = get_logger(__name__)


@Algorithm.register("skill_forest")
class SkillForestTrainer(SkillTrainer[SkillForestTrainerConfig]):
    """Train independent Skill trees and select or synthesize their winners."""

    config_type = SkillForestTrainerConfig
    """Configuration model used by the shared algorithm registry."""

    def __init__(self, config: SkillForestTrainerConfig) -> None:
        """Bind a validated run configuration without starting physical work.

        Args:
            config: Forest training controls.
        """
        super().__init__(config.model_copy(deep=True))

    def create_splits(self, task_ids: Sequence[str]) -> list[TreeSplit]:
        """Create one deterministic train and validation partition per tree.

        Args:
            task_ids: Unique IDs from the complete training pool.

        Returns:
            Splits in tree-index order.
        """
        return [
            split_tasks(task_ids, train_ratio=self.config.train_ratio, seed=self.config.seed + index * 1_000_003)
            for index in range(self.config.num_trees)
        ]

    def build_report(self, work_dir: Path) -> dict[str, JsonValue]:
        """Return compact forest metadata for the published Skill version.

        Args:
            work_dir: Completed Skill-Forest training directory.

        Returns:
            Per-tree winner gains and selection scores.
        """
        result = read_json(Path(work_dir) / "result.json")
        if result.get("status") != "completed":
            raise ValueError("Skill-Forest result is incomplete")
        return result["training"]["algorithm_report"]

    async def train(
        self,
        *,
        skill: Skill,
        env: Environment,
        train_samples: Sequence[DataSample],
        validation_samples: Sequence[DataSample],
        work_dir: Path,
    ) -> Skill:
        """Validate, resume and train under one run-directory lock.

        Args:
            skill: Original static Skill, unchanged by training.
            env: Caller-owned rollout environment.
            train_samples: Ordered training samples.
            validation_samples: External samples excluded from optimization.
            work_dir: New directory or a matching node-checkpoint directory.

        Returns:
            Selected or synthesized Skill.

        Raises:
            ValueError: Inputs differ from the existing run or are invalid.
            filelock.Timeout: Another writer holds the run lock.
        """
        root = Path(work_dir).expanduser().resolve()
        root.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncFileLock(str(root.parent / f".{root.name}.lock"), timeout=0, run_in_executor=False):
            logger.info("[skill_forest.training] started")
            env_adapter = select_env_adapter(env)
            samples, splits = self._prepare_run(skill, env, train_samples, validation_samples, root, env_adapter)
            reused = (root / "result.json").exists()
            selected = await self._run_train(skill, env, samples, splits, root, env_adapter)
            result = read_json(root / "result.json")
            decisions = result["training"]["summary"]["decisions"]
            logger.info(
                f"[skill_forest.training] completed | accepted_candidates={decisions.get('accepted', 0)}"
                f" rejected_candidates={decisions.get('rejected', 0)}"
                f" | skill_chars={len(skill.content)}->{len(selected.content)}"
                f" | elapsed={result['duration_s']:.1f}s reused={reused}"
            )
            return selected

    def _prepare_run(
        self,
        skill: Skill,
        env: Environment,
        train_samples: Sequence[DataSample],
        validation_samples: Sequence[DataSample],
        root: Path,
        env_adapter: SkillForestEnvAdapter,
    ) -> tuple[dict[str, DataSample], list[TreeSplit]]:
        """Validate all inputs and bind a resumable run to their fingerprint."""
        if skill.snapshot().implementation != "static":
            raise ValueError("Skill-Forest requires a static source Skill")
        env_adapter.validate_environment(env)
        samples = {sample.task.id: sample for sample in train_samples}
        if len(samples) != len(train_samples):
            raise ValueError("Training task IDs must be unique")
        external_ids = [sample.task.id for sample in validation_samples]
        if len(external_ids) != len(set(external_ids)) or samples.keys() & set(external_ids):
            raise ValueError("External validation IDs must be unique and disjoint from training")
        splits = self.create_splits(list(samples))
        config = redact_credentials(self.config.model_dump(mode="json"), preserve_execution_settings=True)
        environment = {
            "class": f"{type(env).__module__}.{type(env).__qualname__}",
            "config": redact_credentials(env.config.model_dump(mode="json"), preserve_execution_settings=True)
            if isinstance(env, Environment)
            else None,
        }
        adapter_identity = {
            "class": f"{type(env_adapter).__module__}.{type(env_adapter).__qualname__}",
            "config": env_adapter.config.model_dump(mode="json"),
        }
        identity = fingerprint(
            {
                "config": config,
                "adapter": adapter_identity,
                "skill": skill.snapshot().model_dump(mode="json"),
                "samples": [sample_identity(sample, env) for sample in train_samples],
                "validation": [sample_identity(sample, env) for sample in validation_samples],
                "splits": [split.model_dump(mode="json") for split in splits],
                "trainer": f"{type(self).__module__}.{type(self).__qualname__}",
                "environment": environment,
            }
        )
        path = root / "config.json"
        if root.exists() and any(root.iterdir()):
            if not path.exists():
                raise ValueError("No compatible config.json; legacy runs require a new work_dir")
            stored = read_json(path)
            if stored.get("schema_version") != ALGORITHM_VERSION or stored.get("identity") != identity:
                raise ValueError("Skill-Forest recovery identity mismatch")
        else:
            root.mkdir(parents=True, exist_ok=True)
            write_json(
                path,
                redact_credentials(
                    {
                        "schema_version": ALGORITHM_VERSION,
                        "identity": identity,
                        "trainer": config,
                        "adapter": adapter_identity,
                        "environment": environment,
                        "splits": [split.model_dump(mode="json") for split in splits],
                    }
                ),
            )
        return samples, splits

    async def _run_train(
        self,
        skill: Skill,
        env: Environment,
        samples: dict[str, DataSample],
        splits: Sequence[TreeSplit],
        root: Path,
        env_adapter: SkillForestEnvAdapter,
    ) -> Skill:
        """Grow accepted nodes, freeze tree winners and publish one final result."""
        result_path = root / "result.json"
        if result_path.exists():
            result = read_json(result_path)
            if result.get("status") != "completed":
                raise ValueError("Skill-Forest result is incomplete")
            return load_skill(result["selected_skill"])
        runtime = ForestRuntime(self.config, work_dir=root, env_adapter=env_adapter)
        started = perf_counter()
        try:
            stages = {}
            phase_start = perf_counter()
            cached_roots = [read_node(root, index, f"tree-{index:03d}-root") for index in range(len(splits))]
            logger.info("[skill_forest.root] started")
            roots, shared_evidence = await self._initialize_roots(skill, env, samples, splits, root, runtime)
            scores = [score for node in roots for score in node.scores.values()]
            logger.info(
                f"[skill_forest.root] progress | scored={len(scores)} | success_count={int(sum(scores))}"
                f" | accuracy={sum(scores) / len(scores) if scores else None}"
                f" | score_scope=tree_validation_memberships | shared_trajectories={len(shared_evidence)}"
                f" | reused={all((item is not None for item in cached_roots))}"
            )
            stages["root_s"] = perf_counter() - phase_start
            phase_start = perf_counter()
            nodes, expansions = await self._grow_forest(roots, shared_evidence, env, samples, splits, root, runtime)
            stages["expansion_s"] = perf_counter() - phase_start
            winners = tree_winners(nodes)
            logger.info(f"[skill_forest.selection] progress | winner_count={len(winners)} | node_count={len(nodes)}")
            if not winners:
                raise RuntimeError("No positive-gain non-root winner; the paper does not substitute the shared root")
            for winner in winners:
                path = root / "skills" / f"skill-tree-{winner.tree_id}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(winner.skill.content, encoding="utf-8")
            phase_start = perf_counter()
            logger.info("[skill_forest.selection] started")
            selected, selected_id, winner_scores = await self._ensemble(skill, winners, samples, env, runtime, root)
            logger.info(
                f"[skill_forest.selection] progress | selected_node_id={selected_id}"
                f" | skill_chars_before={len(skill.content)} | skill_chars_after={len(selected.content)}"
                f" | skill_chars_delta={len(selected.content) - len(skill.content)}"
            )
            stages["selection_s"] = perf_counter() - phase_start
            node_rows = [self._node_summary(node) for node in nodes]
            report = self._algorithm_report(node_rows, expansions, winner_scores, selected_id)
            write_json(
                result_path,
                {
                    "schema_version": ALGORITHM_VERSION,
                    "status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                    "duration_s": perf_counter() - started,
                    "selected_skill": selected.snapshot().model_dump(mode="json"),
                    "training": {
                        "stages": stages,
                        "metrics": runtime.metrics,
                        "forest": {"expansions": expansions, "nodes": node_rows},
                        "winners": winner_scores,
                        "summary": self._summarize(root),
                        "algorithm_report": report,
                    },
                    "validation": None,
                    "test": None,
                },
            )
            return selected
        finally:
            await runtime.close()

    async def _initialize_roots(
        self,
        skill: Skill,
        env: Environment,
        samples: dict[str, DataSample],
        splits: Sequence[TreeSplit],
        root: Path,
        runtime: ForestRuntime,
    ) -> tuple[list[Node], dict[str, RolloutResult]]:
        """Load committed roots or score their shared initial Skill once."""
        cached = [read_node(root, index, f"tree-{index:03d}-root") for index in range(len(splits))]
        if all(item is not None and item.status == "root" for item in cached):
            roots = [load_node(item) for item in cached]
            if not self.config.root_share_train_trajectory or all(item.expanded for item in cached):
                return roots, {}
            ids = list(dict.fromkeys(key for split in splits for key in (*split.validation_ids, *split.train_ids)))
            evidence = await self._collect(
                skill, ids, samples, env, runtime, node_directory(root, 0, roots[0].id), "train"
            )
            self._record_shared_root_rollouts(skill, splits, root, runtime, evidence, set(evidence))
            return roots, evidence

        share_train = self.config.root_share_train_trajectory
        share_validation = share_train or self.config.root_share_validation
        validation_ids = sorted({key for split in splits for key in split.validation_ids})
        first = node_directory(root, 0, "tree-000-root")
        shared = {}
        if share_validation:
            ids = (
                list(dict.fromkeys([*validation_ids, *(key for split in splits for key in split.train_ids)]))
                if share_train
                else validation_ids
            )
            shared = await self._collect(
                skill,
                ids,
                samples,
                env,
                runtime,
                first,
                "eval",
                seed=self.config.seed if share_train else self.config.seed + 10_000,
            )
        roots = []
        for index, split in enumerate(splits):
            node_id = f"tree-{index:03d}-root"
            directory = node_directory(root, index, node_id)
            validation = (
                shared
                if share_validation
                else await self._collect(
                    skill, split.validation_ids, samples, env, runtime, directory, "eval", tree_id=index
                )
            )
            scores = {key: float(validation[key].reward.success) for key in split.validation_ids}
            checkpoint = NodeCheckpoint(id=node_id, tree_id=index, skill=skill.snapshot(), status="root", scores=scores)
            write_node(root, checkpoint)
            roots.append(load_node(checkpoint))
            if share_validation:
                (directory / "evaluation.jsonl").unlink(missing_ok=True)
                for key in split.validation_ids:
                    runtime._append_trajectory(
                        directory / "evaluation.jsonl",
                        {
                            "task_id": key,
                            "skill_hash": skill.source.skill_hash,
                            "physical": index == 0,
                            "result": validation[key].model_dump(mode="json"),
                        },
                    )
        if share_train:
            self._record_shared_root_rollouts(skill, splits, root, runtime, shared, set(shared))
        return roots, shared if share_train else {}

    @staticmethod
    def _record_shared_root_rollouts(
        skill: Skill,
        splits: Sequence[TreeSplit],
        root: Path,
        runtime: ForestRuntime,
        evidence: dict[str, RolloutResult],
        physical_ids: set[str],
    ) -> None:
        """Place shared root evidence in each tree's training trajectory stream."""
        for index, split in enumerate(splits):
            directory = node_directory(root, index, f"tree-{index:03d}-root")
            path = directory / "rollouts.jsonl"
            path.unlink(missing_ok=True)
            for key in split.train_ids:
                runtime._append_trajectory(
                    path,
                    {
                        "task_id": key,
                        "skill_hash": skill.source.skill_hash,
                        "physical": index == 0 and key in physical_ids,
                        "result": evidence[key].model_dump(mode="json"),
                    },
                )

    async def _collect(
        self,
        skill: Skill,
        ids: Sequence[str],
        samples: dict[str, DataSample],
        env: Environment,
        runtime: ForestRuntime,
        node_dir: Path,
        purpose: Literal["train", "eval"],
        *,
        tree_id: int = 0,
        seed: int | None = None,
    ) -> dict[str, RolloutResult]:
        """Rebuild one complete rollout group without per-task recovery."""
        name = "rollouts.jsonl" if purpose == "train" else "evaluation.jsonl"
        (node_dir / name).unlink(missing_ok=True)
        phase_seed = (
            seed if seed is not None else self.config.seed + tree_id * 1_000_003 + (10_000 if purpose == "eval" else 0)
        )
        results = await runtime.scheduler.collect(
            env=env,
            samples=[
                runtime.env_adapter.prepare_sample(
                    samples[key],
                    seed=phase_seed,
                )
                for key in ids
            ],
            skill=skill,
            directory=node_dir / "workspaces" / purpose,
            stage_name="skill_forest.train.rollout" if purpose == "train" else "skill_forest.root.evaluation",
        )
        return {key: results[key] for key in ids}

    async def _grow_forest(
        self,
        roots: Sequence[Node],
        shared_evidence: dict[str, RolloutResult],
        env: Environment,
        samples: dict[str, DataSample],
        splits: Sequence[TreeSplit],
        root: Path,
        runtime: ForestRuntime,
    ) -> tuple[list[Node], int]:
        """Expand uncommitted nodes until the global parent budget is exhausted."""
        nodes = list(roots)
        for path in sorted((root / "trees").glob("tree-*/node-*/node.json")):
            saved = read_node_file(path)
            if saved.parent_id is not None and saved.status == "accepted":
                nodes.append(load_node(saved))
        expansions = sum(bool(read_node(root, node.tree_id, node.id).expanded) for node in nodes)
        logger.info(
            f"[skill_forest.expansion] started | expansion_concurrency={self.config.expansion_concurrency}"
            f" | rollout_concurrency={self.config.rollout_concurrency}"
            f" | candidates_per_node={self.config.candidates_per_node}"
        )
        while expansions < self.config.expansion_budget:
            pending_roots = [
                node for node in nodes if node.parent_id is None and not read_node(root, node.tree_id, node.id).expanded
            ]
            if pending_roots:
                wave = pending_roots[
                    : min(self.config.expansion_concurrency, self.config.expansion_budget - expansions)
                ]
            else:
                frontier = [node for node in nodes if not read_node(root, node.tree_id, node.id).expanded]
                wave = expansion_wave(
                    frontier,
                    limit=min(self.config.expansion_concurrency, self.config.expansion_budget - expansions),
                    max_depth=self.config.max_depth,
                )
            if not wave:
                break
            children = await parallel(
                self._expand(node, splits[node.tree_id], samples, env, runtime, root, shared_evidence) for node in wave
            )
            for parent, batch in zip(wave, children, strict=True):
                checkpoint = read_node(root, parent.tree_id, parent.id)
                checkpoint.expanded = True
                write_node(root, checkpoint)
                expansions += 1
                known = {item.id for item in nodes}
                nodes.extend(child for child in batch if child.id not in known)
                logger.info(
                    f"[skill_forest.expansion] progress | node_count={len(nodes)}"
                    f" | accepted_candidates={len(nodes) - len(roots)}"
                    f" | best_gain={max((item.gain for item in nodes), default=0.0)} | tree_id={parent.tree_id}"
                    f" | node_id={parent.id}"
                )
        return nodes, expansions

    async def _expand(
        self,
        parent: Node,
        split: TreeSplit,
        samples: dict[str, DataSample],
        env: Environment,
        runtime: ForestRuntime,
        root: Path,
        shared_evidence: dict[str, RolloutResult],
    ) -> list[Node]:
        """Generate and validate each child of one parent node."""
        with log_context(tree_id=parent.tree_id, node_id=parent.id):
            directory = node_directory(root, parent.tree_id, parent.id)
            if parent.parent_id is None and self.config.root_share_train_trajectory:
                evidence = {key: shared_evidence[key] for key in split.train_ids}
            else:
                evidence = await self._collect(
                    parent.skill, split.train_ids, samples, env, runtime, directory, "train", tree_id=parent.tree_id
                )
            candidates = await parallel(
                self._candidate(parent, index, split, samples, evidence, env, runtime, root)
                for index in range(self.config.candidates_per_node)
            )
            return [candidate for candidate in candidates if candidate is not None]

    async def _candidate(
        self,
        parent: Node,
        index: int,
        split: TreeSplit,
        samples: dict[str, DataSample],
        evidence: dict[str, RolloutResult],
        env: Environment,
        runtime: ForestRuntime,
        root: Path,
    ) -> Node | None:
        """Reuse a patch-applied Skill and redo validation when unfinished."""
        node_id = f"{parent.id}-{index:02d}"
        with log_context(tree_id=parent.tree_id, parent_id=parent.id, node_id=node_id):
            logger.info("[skill_forest.candidate] started")
            checkpoint = read_node(root, parent.tree_id, node_id)
            if checkpoint is not None and checkpoint.status == "accepted":
                logger.info(
                    f"[skill_forest.candidate] progress | accepted=True | gain={checkpoint.gain}"
                    " | reason=checkpoint_reused | reused=True"
                )
                return load_node(checkpoint)
            if checkpoint is not None and checkpoint.status == "rejected":
                logger.info(
                    "[skill_forest.candidate] progress | reason=checkpoint_rejected | accepted=False | reused=True"
                )
                return None
            directory = node_directory(root, parent.tree_id, node_id)
            if checkpoint is None:
                invalid = False
                try:
                    patch = await self._generate_patch(
                        parent, index, split, samples, evidence, runtime, directory / "generation"
                    )
                except* InvalidPatchError:
                    invalid = True
                if invalid:
                    runtime.metrics["invalid_patches"] += 1
                    logger.info("[skill_forest.candidate] progress | reason=invalid_patch | accepted=False")
                    return None
                try:
                    candidate = apply_patch(parent.skill, patch, reference=self.config.patch_application == "reference")
                except ValueError:
                    runtime.metrics["invalid_patches"] += 1
                    logger.info("[skill_forest.candidate] progress | reason=patch_application_failed | accepted=False")
                    return None
                checkpoint = NodeCheckpoint(
                    id=node_id,
                    tree_id=parent.tree_id,
                    parent_id=parent.id,
                    depth=parent.depth + 1,
                    skill=candidate.snapshot(),
                    status="generated",
                )
                write_node(root, checkpoint)
            candidate = load_skill(checkpoint.skill)
            logger.info(
                f"[skill_forest.candidate] progress | skill_chars_before={len(parent.skill.content)}"
                f" | skill_chars_after={len(candidate.content)}"
                f" | skill_chars_delta={len(candidate.content) - len(parent.skill.content)}"
            )
            if candidate.source.content_hash == parent.skill.source.content_hash:
                checkpoint.status = "rejected"
                write_node(root, checkpoint)
                logger.info("[skill_forest.candidate] progress | accepted=False | reason=unchanged_skill")
                return None
            ordered = sorted(split.validation_ids, key=lambda key: (parent.scores[key], key))

            def reject(results: dict[str, RolloutResult]) -> bool:
                scores = {key: float(result.reward.success) for key, result in results.items()}
                return cannot_improve(scores, parent.scores, self.config.min_gain)

            (directory / "evaluation.jsonl").unlink(missing_ok=True)
            results: dict[str, RolloutResult] = {}
            for start in range(0, len(ordered), self.config.validation_batch_size):
                if self.config.early_rejection and reject(results):
                    break
                batch = await runtime.scheduler.collect(
                    env=env,
                    samples=[
                        runtime.env_adapter.prepare_sample(
                            samples[key], seed=self.config.seed + parent.tree_id * 1_000_003 + 10_000
                        )
                        for key in ordered[start : start + self.config.validation_batch_size]
                    ],
                    skill=candidate,
                    directory=directory / "workspaces" / "eval" / f"batch-{start:06d}",
                    stage_name="skill_forest.candidate.evaluation",
                )
                results.update(batch)
            scores = {key: float(result.reward.success) for key, result in results.items()}
            complete = len(scores) == len(parent.scores)
            gain = mean_score(scores) - mean_score(parent.scores) if complete else 0.0
            accepted = complete and gain > self.config.min_gain
            checkpoint.status = "accepted" if accepted else "rejected"
            checkpoint.scores = scores
            checkpoint.gain = gain
            write_node(root, checkpoint)
            logger.info(
                f"[skill_forest.candidate] progress | scored={len(scores)}"
                f" | success_count={int(sum(scores.values()))}"
                f" | accuracy={mean_score(scores) if scores else None} | gain={gain}"
                f" | parent_accuracy={mean_score(parent.scores)} | accepted={accepted}"
                f" | reason={'gain_above_threshold' if accepted else 'insufficient_gain' if complete else 'early_rejection'}"
            )
            return load_node(checkpoint) if accepted else None

    async def _generate_patch(
        self,
        parent: Node,
        index: int,
        split: TreeSplit,
        samples: dict[str, DataSample],
        evidence: dict[str, RolloutResult],
        runtime: ForestRuntime,
        directory: Path,
    ) -> Patch:
        """Generate one candidate patch from its parent's training evidence."""
        logger.info("[skill_forest.patch_generation] started")
        patch = await generate_patch(
            skill=parent.skill,
            samples=[samples[key] for key in split.train_ids],
            results=[evidence[key] for key in split.train_ids],
            runtime=runtime,
            directory=directory,
            seed=self.config.seed + parent.tree_id * 1_000_003 + parent.depth * 100_000 + (index + 1) * 10_000_019,
            depth=parent.depth,
        )
        runtime.metrics["proposed_patches"] += 1
        runtime.metrics["proposed_edits"] += len(patch.edits)
        logger.info(f"[skill_forest.patch_generation] progress | proposed_patches=1 | patch_edits={len(patch.edits)}")
        return patch

    async def _ensemble(
        self,
        initial: Skill,
        winners: Sequence[Node],
        samples: dict[str, DataSample],
        env: Environment,
        runtime: ForestRuntime,
        root: Path,
    ) -> tuple[Skill, str | None, list[WinnerScore]]:
        """Score frozen winners and apply Select or hierarchical synthesis."""
        values = await parallel(
            self._score_ensemble_skill(node.skill, node, node.id, samples, env, runtime, root) for node in winners
        )
        distances = [novelty(initial.content, node.skill.content) for node in winners]
        winner_scores = [
            {"id": node.id, "tree_id": node.tree_id, "score": value, "novelty": distance}
            for node, value, distance in zip(winners, values, distances, strict=True)
        ]
        if self.config.ensemble_strategy == "select":
            if self.config.selection_tiebreak == "novelty":
                best = max(range(len(winners)), key=lambda index: (values[index], distances[index]))
            else:
                best = min(
                    range(len(winners)),
                    key=lambda index: (
                        -values[index],
                        -mean_score(winners[index].scores),
                        winners[index].depth,
                        len(winners[index].skill.content),
                        winners[index].id,
                        winners[index].tree_id,
                    ),
                )
            return winners[best].skill, winners[best].id, winner_scores
        threshold = max(values) - self.config.merge_threshold
        items = [
            EnsembleCandidate(name=node.id, skill=node.skill, score=value, owner=node)
            for node, value in zip(winners, values, strict=True)
            if value >= threshold
        ]
        items.sort(key=lambda item: hashlib.sha256(f"42:{item.owner.tree_id}:{item.name}".encode()).hexdigest())
        level = 0
        while len(items) > 1:
            merged = await parallel(
                self._merge_ensemble_pair(
                    items[index],
                    items[index + 1],
                    f"merge-{level:03d}-{index // 2:03d}",
                    initial,
                    samples,
                    env,
                    runtime,
                    root,
                )
                for index in range(0, len(items) - 1, 2)
            )
            if len(items) % 2:
                merged.append(items[-1])
            items = merged
            level += 1
        return items[0].skill, None, winner_scores

    async def _score_ensemble_skill(
        self,
        skill: Skill,
        owner: Node,
        name: str,
        samples: dict[str, DataSample],
        env: Environment,
        runtime: ForestRuntime,
        root: Path,
    ) -> float:
        """Score one frozen Skill over the complete training pool."""
        with log_context(tree_id=owner.tree_id, node_id=name):
            directory = node_directory(root, owner.tree_id, owner.id) / "workspaces" / "eval" / f"selection-{name}"
            results = await runtime.scheduler.collect(
                env=env,
                samples=list(samples.values()),
                skill=skill,
                directory=directory,
                stage_name="skill_forest.selection.evaluation",
            )
            scores = {key: float(result.reward.success) for key, result in results.items()}
            return mean_score(scores)

    async def _merge_ensemble_pair(
        self,
        left: EnsembleCandidate,
        right: EnsembleCandidate,
        name: str,
        initial: Skill,
        samples: dict[str, DataSample],
        env: Environment,
        runtime: ForestRuntime,
        root: Path,
    ) -> EnsembleCandidate:
        """Synthesize and score one pair of previously ranked Skills."""
        with log_context(node_id=name):
            logger.info("[skill_forest.synthesis] started")
            text = await runtime.chat(
                messages=runtime.env_adapter.build_skill_merge_messages(
                    SkillForestSkillMergeContext(
                        left=split_merge_content(left.skill.content)[1],
                        right=split_merge_content(right.skill.content)[1],
                        left_score=left.score,
                        right_score=right.score,
                    )
                ),
                directory=root / "trees" / "merge",
            )
            _, text = split_merge_content(text)
            if not text.strip():
                raise ValueError("Skill synthesis returned an empty body")
            if initial.source.standard_format:
                header, _ = split_merge_content(initial.content)
                text = header + ("" if header.endswith(("\r", "\n")) else "\n") + text
            candidate = initial.with_content(text)
            value = await self._score_ensemble_skill(candidate, left.owner, name, samples, env, runtime, root)
            logger.info(
                f"[skill_forest.synthesis] progress | scored={len(samples)}"
                f" | success_count={round(value * len(samples))} | accuracy={value}"
                f" | skill_chars_before={len(left.skill.content)} | skill_chars_after={len(candidate.content)}"
                f" | skill_chars_delta={len(candidate.content) - len(left.skill.content)}"
            )
            return EnsembleCandidate(name=name, skill=candidate, score=value, owner=left.owner)

    @staticmethod
    def _node_summary(node: Node) -> NodeSummary:
        """Keep final forest metadata independent of rollout workspaces."""
        return {
            "id": node.id,
            "tree_id": node.tree_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "gain": node.gain,
            "scores": node.scores,
            "content_hash": node.skill.source.content_hash,
        }

    def _algorithm_report(
        self, nodes: Sequence[NodeSummary], expansions: int, winners: Sequence[WinnerScore], selected_id: str | None
    ) -> dict[str, JsonValue]:
        """Build the compact version metadata already consumed by publication."""
        by_id = {node["id"]: node for node in nodes}
        selected = {winner["tree_id"]: winner for winner in winners}
        trees = []
        for tree_id in range(self.config.num_trees):
            members = [node for node in nodes if node["tree_id"] == tree_id]
            winner = selected.get(tree_id)
            node = by_id[winner["id"]] if winner is not None else None
            trees.append(
                {
                    "tree_id": tree_id,
                    "node_count": len(members),
                    "winner": {
                        "node_id": node["id"],
                        "gain": node["gain"],
                        "validation_mean_score": mean_score(node["scores"]),
                        "train_top_select_score": winner["score"],
                    }
                    if node is not None
                    else None,
                }
            )
        return {
            "schema_version": ALGORITHM_VERSION,
            "algorithm": "skill_forest",
            "tree_count": self.config.num_trees,
            "node_count": len(nodes),
            "expansion_count": expansions,
            "ensemble_strategy": self.config.ensemble_strategy,
            "selected_node_id": selected_id,
            "trees": trees,
        }

    @staticmethod
    def _summarize(root: Path) -> dict[str, dict[str, int]]:
        """Count persisted node decisions and complete trajectory records."""
        decisions = {}
        trajectories = {"evaluation": 0, "rollouts": 0}
        for path in (root / "trees").glob("tree-*/node-*/node.json"):
            state = read_node_file(path).status
            if state != "root":
                decisions[state] = decisions.get(state, 0) + 1
        for name in ("evaluation", "rollouts"):
            for path in (root / "trees").glob(f"tree-*/node-*/{name}.jsonl"):
                for _ in read_jsonl(path):
                    trajectories[name] += 1
        return {"decisions": decisions, "trajectories": trajectories}
