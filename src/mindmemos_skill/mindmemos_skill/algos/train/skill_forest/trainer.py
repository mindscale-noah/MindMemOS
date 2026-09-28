"""Async orchestration of paper-defined, globally bounded Skill-Forest growth."""

import difflib
import hashlib
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

from filelock import FileLock

from ....agents import OpenAIChatAgent
from ....common.checkpoint import boundary
from ....common.json import read_json
from ....envs.base import Environment
from ....persistence.observability import get_log_context
from ....skill import Skill
from ...base import Algorithm
from ..base import SkillTrainer
from ..recovery import implementation_identity
from .artifacts import ARTIFACT_SCHEMAS, record, summarize_run
from .checkpoint import load_skill, recovery_event
from .config import SkillForestTrainerConfig
from .env_adapter import SkillForestEnvAdapter
from .generation import InvalidPatchError, generate_patch
from .runtime import ForestRuntime, parallel
from .tree import apply_patch, cannot_improve, expansion_wave, mean_score, novelty, split_tasks, tree_winners
from .typing import Node, SkillForestSkillMergeContext, TreeSplit


@Algorithm.register("skill_forest")
class SkillForestTrainer(SkillTrainer[SkillForestTrainerConfig]):
    """Arrange stateless tree operations without database or environment ownership."""

    config_type = SkillForestTrainerConfig
    """Configuration model used by the shared algorithm registry."""

    def __init__(self, config: SkillForestTrainerConfig, *, agent_factory=OpenAIChatAgent):
        """Configure training without allocating resources or calling providers.

        Args:
            config: Validated algorithm parameters.
            agent_factory: Optional optimizer constructor for offline tests.
        """
        super().__init__(config.model_copy(deep=True))
        self.agent_factory = agent_factory
        self.env_adapter = SkillForestEnvAdapter.build(
            self.config.env_adapter.class_name, self.config.env_adapter.config
        )

    def create_splits(self, task_ids: Sequence[str]) -> list[TreeSplit]:
        """Create tree-local partitions; subclasses may override the split strategy.

        Args:
            task_ids: Unique task IDs from the complete training pool.

        Returns:
            One split per configured tree, in tree-index order. Each split must
            contain nonempty, disjoint train and validation IDs covering the pool.
        """
        return [
            split_tasks(task_ids, train_ratio=self.config.train_ratio, seed=self.config.seed + index)
            for index in range(self.config.num_trees)
        ]

    def build_report(self, work_dir: Path) -> dict:
        """Summarize validated forest artifacts for published version metadata.

        Args:
            work_dir: Completed Skill-Forest training directory.

        Returns:
            Compact, versioned JSON with per-tree winner gains and select scores.
        """
        root = Path(work_dir)
        forest = ARTIFACT_SCHEMAS["forest.json"].validate_python(read_json(root / "forest.json"))
        winners = ARTIFACT_SCHEMAS["winners.json"].validate_python(read_json(root / "ensemble" / "winners.json"))
        nodes = {node.id: node for node in forest.nodes}
        scored = {winner.tree_id: winner for winner in winners}
        trees = []
        for tree_id in range(self.config.num_trees):
            members = [node for node in forest.nodes if node.tree_id == tree_id]
            winner = scored.get(tree_id)
            selected = nodes[winner.id] if winner is not None else None
            trees.append(
                {
                    "tree_id": tree_id,
                    "node_count": len(members),
                    "winner": {
                        "node_id": selected.id,
                        "gain": selected.gain,
                        "validation_mean_score": mean_score(selected.scores),
                        "train_top_select_score": winner.score,
                    }
                    if selected is not None
                    else None,
                }
            )
        selection_path = root / "ensemble" / "selection.json"
        selection = (
            ARTIFACT_SCHEMAS["selection.json"].validate_python(read_json(selection_path)).selected
            if selection_path.exists()
            else None
        )
        return {
            "schema_version": 1,
            "algorithm": "skill_forest",
            "tree_count": self.config.num_trees,
            "node_count": len(forest.nodes),
            "expansion_count": forest.expansions,
            "ensemble_strategy": self.config.ensemble_strategy,
            "selected_node_id": selection,
            "trees": trees,
        }

    async def train(self, *, skill, env, train_samples, validation_samples, work_dir) -> Skill:
        """Run or resume a forest while excluding concurrent writers.

        Args:
            skill: Original initial Skill, unchanged during recovery.
            env: Caller-owned environment with matching configuration.
            train_samples: Original ordered training samples.
            validation_samples: Original external validation samples.
            work_dir: Fresh run directory, or the original directory for resume.

        Returns:
            Selected or synthesized Skill.

        Raises:
            ValueError: Recovery inputs differ or no compatible manifest exists.
            filelock.Timeout: Another process owns this run.
        """
        root = Path(work_dir).expanduser().resolve()
        root.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root.parent / f".{root.name}.lock"), timeout=0):
            return await self._run_train(
                skill=skill,
                env=env,
                train_samples=train_samples,
                validation_samples=validation_samples,
                work_dir=root,
                resume=root.exists() and any(root.iterdir()),
            )

    async def _run_train(self, *, skill, env, train_samples, validation_samples, work_dir, resume) -> Skill:
        """Validate run identity and execute under the caller-held directory lock."""
        if skill.snapshot().implementation != "static":
            raise ValueError("Skill-Forest requires a static source Skill")
        self.env_adapter.validate_environment(env)
        samples = {sample.task.id: sample for sample in train_samples}
        if len(samples) != len(train_samples):
            raise ValueError("Training task IDs must be unique")
        external_ids = [sample.task.id for sample in validation_samples]
        if len(external_ids) != len(set(external_ids)) or samples.keys() & set(external_ids):
            raise ValueError("External validation IDs must be unique and disjoint from training")
        splits = self.create_splits(list(samples))
        if "SKILL.md" not in skill.bundle.files:
            raise ValueError("Skill-Forest requires a SKILL.md entrypoint")
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        if not resume and any(root.iterdir()):
            raise ValueError("Skill-Forest requires a fresh work_dir if resume=false")
        identity = {
            "implementation": implementation_identity(self, env),
            "config": self.config.model_dump(mode="json", exclude={"optimizer": {"api_key"}}),
            "skill": skill.snapshot().model_dump(mode="json"),
            "samples": [sample.model_dump(mode="json") for sample in train_samples],
            "validation": [sample.model_dump(mode="json") for sample in validation_samples],
            "splits": [split.model_dump(mode="json") for split in splits],
            "trainer": f"{type(self).__module__}.{type(self).__qualname__}",
            "environment": f"{type(env).__module__}.{type(env).__qualname__}",
            "env_config": env.config.model_dump(mode="json", exclude={"agent": {"api_key"}})
            if isinstance(env, Environment)
            else None,
        }
        manifest = root / "recovery_manifest.json"
        if resume and not manifest.exists():
            raise ValueError("No compatible recovery manifest; legacy runs require explicit migration")
        boundary(manifest, identity, payload={"protocol": 1})
        recovery_event(root, "run_resumed" if resume else "run_started")
        completed = boundary(root / "completed.json", identity) if resume else None
        if completed is not None:
            return load_skill(completed["skill"])
        record(root / "config.json", self.config.model_dump(mode="json", exclude={"optimizer": {"api_key"}}))
        record(root / "splits.json", [split.model_dump(mode="json") for split in splits])
        record(root / "initial_skill.json", skill.snapshot().model_dump(mode="json"))
        runtime = ForestRuntime(
            self.config, agent_factory=self.agent_factory, work_dir=root, env_adapter=self.env_adapter, resume=resume
        )
        start = perf_counter()
        record(root / "status.json", {"status": "running"})
        try:
            result = await self._train(skill, env, samples, splits, root, runtime)
            record(root / "selected_skill.json", result.snapshot().model_dump(mode="json"))
            record(root / "summary.json", summarize_run(root))
            record(
                root / "status.json",
                {
                    "status": "completed",
                    "duration_s": perf_counter() - start,
                    "content_hash": result.bundle.content_hash,
                },
            )
            boundary(root / "completed.json", identity, payload={"skill": result.snapshot().model_dump(mode="json")})
            return result
        except BaseException as exc:
            record(root / "summary.json", summarize_run(root))
            record(
                root / "status.json",
                {
                    "status": "failed",
                    "duration_s": perf_counter() - start,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        finally:
            await runtime.close()

    async def _collect(self, skill, ids, samples, env, runtime, directory):
        """Collect each requested task exactly once using forest-wide rollout slots."""
        results = await runtime.scheduler.collect(
            env=env, samples=[samples[key] for key in ids], skill=skill, directory=directory
        )
        return {key: results[key] for key in ids}

    async def _train(self, skill, env, samples, splits, root, runtime):
        """Own the frontier, shared root evidence and expansion budget for one run."""
        validation_ids = list(dict.fromkeys(key for split in splits for key in split.validation_ids))
        root_rollout_results = await self._collect(
            skill, validation_ids, samples, env, runtime, root / "root_validation"
        )
        if self.config.root_share_train_trajectory:
            missing = [key for key in samples if key not in root_rollout_results]
            root_rollout_results.update(await self._collect(skill, missing, samples, env, runtime, root / "root_train"))
        roots = [
            Node(
                id=f"tree-{index:03d}-root",
                tree_id=index,
                skill=skill,
                scores={key: float(root_rollout_results[key].reward.success) for key in split.validation_ids},
            )
            for index, split in enumerate(splits)
        ]
        nodes = list(roots)
        frontier = []
        expansions = 0
        pending_roots = list(roots)
        wave_index = 0
        # Expand pending roots first, then frontier nodes in concurrency-limited waves.
        # Queue children below max_depth and save the forest after each wave.
        # Stop when the expansion budget is exhausted or no eligible nodes remain.
        while pending_roots or frontier:
            remaining = self.config.expansion_budget - expansions
            if remaining <= 0:
                break
            limit = min(remaining, self.config.expansion_concurrency)
            if pending_roots:
                wave, pending_roots = pending_roots[:limit], pending_roots[limit:]
            else:
                wave = expansion_wave(frontier, limit=limit, max_depth=self.config.max_depth)
                taken = {node.id for node in wave}
                frontier = [node for node in frontier if node.id not in taken]
            if not wave:
                break
            boundary(
                root / "waves" / f"{wave_index:06d}.json",
                {"wave": wave_index},
                payload={"parents": [node.id for node in wave], "expansions_before": expansions},
            )
            wave_index += 1
            expansions += len(wave)
            children = await parallel(
                self._expand(node, splits[node.tree_id], samples, env, runtime, root, root_rollout_results)
                for node in wave
            )
            for batch in children:
                nodes.extend(batch)
                frontier.extend(node for node in batch if node.depth < self.config.max_depth)
            record(
                root / "forest.json",
                {
                    "expansions": expansions,
                    "nodes": [
                        {
                            "id": node.id,
                            "tree_id": node.tree_id,
                            "parent_id": node.parent_id,
                            "depth": node.depth,
                            "gain": node.gain,
                            "scores": node.scores,
                            "content_hash": node.skill.bundle.content_hash,
                        }
                        for node in nodes
                    ],
                },
            )
        winners = tree_winners(nodes)
        if not winners:
            raise RuntimeError("No positive-gain non-root winner; the paper does not substitute the shared root")
        return await self._ensemble(skill, winners, samples, env, runtime, root)

    async def _expand(self, node, split, samples, env, runtime, root, root_rollout_results):
        """Collect one parent batch, then generate and validate its independent children."""
        session = get_log_context()
        scope = (
            session.span(
                "algorithm.tree.expand",
                {"algorithm.tree_id": node.tree_id, "algorithm.node_id": node.id, "algorithm.depth": node.depth},
            )
            if session is not None
            else nullcontext()
        )
        with scope:
            return await self._expand_untraced(node, split, samples, env, runtime, root, root_rollout_results)

    async def _expand_untraced(self, node, split, samples, env, runtime, root, root_rollout_results):
        """Execute one expansion inside its optional audit scope."""
        directory = root / "nodes" / node.id
        if node.parent_id is None and self.config.root_share_train_trajectory:
            evidence = {key: root_rollout_results[key] for key in split.train_ids}
            record(directory / "shared_train.json", {"task_ids": list(evidence)})
        else:
            evidence = await self._collect(node.skill, split.train_ids, samples, env, runtime, directory / "train")
        candidates = await parallel(
            self._candidate(node, index, split, samples, evidence, env, runtime, root)
            for index in range(self.config.candidates_per_node)
        )
        return [candidate for candidate in candidates if candidate is not None]

    async def _candidate(self, parent, index, split, samples, evidence, env, runtime, root):
        """Restore a committed decision or finish only this candidate's remaining work."""
        node_id = f"{parent.id}-{index:02d}"
        session = get_log_context()
        scope = (
            session.span(
                "algorithm.candidate",
                {
                    "algorithm.tree_id": parent.tree_id,
                    "algorithm.node_id": node_id,
                    "algorithm.parent_node_id": parent.id,
                },
            )
            if session is not None
            else nullcontext()
        )
        with scope:
            return await self._candidate_untraced(parent, index, split, samples, evidence, env, runtime, root)

    async def _candidate_untraced(self, parent, index, split, samples, evidence, env, runtime, root):
        """Complete or restore the candidate inside its optional audit scope."""
        node_id = f"{parent.id}-{index:02d}"
        directory = root / "nodes" / node_id
        identity = {
            "parent": parent.skill.snapshot().model_dump(mode="json"),
            "parent_scores": parent.scores,
            "split": split.model_dump(mode="json"),
            "id": node_id,
        }
        path = directory / "validated.json"
        saved = boundary(path, identity) if runtime.resume else None
        if saved is not None:
            recovery_event(root, "decision_reused", node_id=node_id)
            if not saved["accepted"]:
                return None
            return Node(
                id=node_id,
                tree_id=parent.tree_id,
                parent_id=parent.id,
                depth=parent.depth + 1,
                skill=load_skill(saved["skill"]),
                scores=saved["scores"],
                gain=saved["gain"],
            )
        result = await self._evaluate_candidate(parent, index, split, samples, evidence, env, runtime, root)
        payload = {"accepted": result is not None}
        if result is not None:
            payload.update(
                skill=result.skill.snapshot().model_dump(mode="json"), scores=result.scores, gain=result.gain
            )
        payload["decision"] = read_json(directory / "decision.json")
        boundary(path, identity, payload=payload)
        recovery_event(root, "decision_committed", node_id=node_id, accepted=result is not None)
        return result

    async def _evaluate_candidate(self, parent, index, split, samples, evidence, env, runtime, root):
        """Create one patch and gate it against its parent on the fixed validation set."""
        node_id = f"{parent.id}-{index:02d}"
        directory = root / "nodes" / node_id
        generation_identity = {
            "parent": parent.skill.bundle.content_hash,
            "id": node_id,
            "split": split.model_dump(mode="json"),
        }
        generation_path = directory / "generated.json"
        generated = boundary(generation_path, generation_identity) if runtime.resume else None
        if generated is not None:
            candidate = load_skill(generated["skill"])
            recovery_event(root, "generation_reused", node_id=node_id)
        else:
            invalid = None
            try:
                patch = await self._generate_patch(
                    parent, index, split, samples, evidence, runtime, directory / "generation"
                )
            except* InvalidPatchError as exc:
                invalid = str(exc)
            if invalid is not None:
                record(directory / "decision.json", {"status": "invalid_patch", "error": invalid})
                return None
            try:
                candidate = apply_patch(parent.skill, patch)
            except ValueError as exc:
                record(directory / "decision.json", {"status": "invalid_patch", "error": str(exc)})
                return None
            boundary(
                generation_path,
                generation_identity,
                payload={"skill": candidate.snapshot().model_dump(mode="json"), "patch": patch.model_dump(mode="json")},
            )
            recovery_event(root, "generation_committed", node_id=node_id)
        before = parent.skill.bundle.files["SKILL.md"]
        after = candidate.bundle.files["SKILL.md"]
        record(directory / "skill.json", candidate.snapshot().model_dump(mode="json"))
        (directory / "change.diff").write_text(
            "".join(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="parent/SKILL.md",
                    tofile="candidate/SKILL.md",
                )
            ),
            encoding="utf-8",
        )
        if candidate.bundle.content_hash == parent.skill.bundle.content_hash:
            record(directory / "decision.json", {"status": "no_change"})
            return None
        ordered = sorted(split.validation_ids, key=lambda key: parent.scores[key])

        def reject(results):
            scores = {key: float(result.reward.success) for key, result in results.items()}
            return cannot_improve(scores, parent.scores, self.config.min_gain)

        results = await runtime.scheduler.collect(
            env=env,
            samples=[samples[key] for key in ordered],
            skill=candidate,
            directory=directory / "validation",
            stop_when=reject if self.config.early_rejection else None,
        )
        scores = {key: float(result.reward.success) for key, result in results.items()}
        complete = len(scores) == len(parent.scores)
        gain = mean_score(scores) - mean_score(parent.scores) if complete else None
        accepted = complete and gain > self.config.min_gain
        record(
            directory / "decision.json",
            {
                "status": "accepted" if accepted else "rejected",
                "scores": scores,
                "gain": gain,
                "evaluated": len(scores),
                "expected": len(parent.scores),
                "early_rejected": not complete,
            },
        )
        if not accepted:
            return None
        return Node(
            id=node_id,
            tree_id=parent.tree_id,
            parent_id=parent.id,
            depth=parent.depth + 1,
            skill=candidate,
            scores=scores,
            gain=gain,
        )

    async def _generate_patch(self, parent, index, split, samples, evidence, runtime, directory):
        """Generate one proposal; subclasses may specialize a frozen experiment protocol."""
        node_id = f"{parent.id}-{index:02d}"
        return await generate_patch(
            skill=parent.skill,
            samples=[samples[key] for key in split.train_ids],
            results=[evidence[key] for key in split.train_ids],
            runtime=runtime,
            directory=directory,
            seed=int.from_bytes(hashlib.sha256(f"{self.config.seed}:{node_id}".encode()).digest()[:8]),
            depth=parent.depth,
        )

    async def _ensemble(self, initial, winners, samples, env, runtime, root):
        """Score frozen winners and apply Select or unfiltered hierarchical synthesis."""
        session = get_log_context()
        scope = (
            session.span("algorithm.ensemble", {"algorithm.winner_count": len(winners)})
            if session is not None
            else nullcontext()
        )
        with scope:
            return await self._ensemble_untraced(initial, winners, samples, env, runtime, root)

    async def _ensemble_untraced(self, initial, winners, samples, env, runtime, root):
        """Execute final selection or synthesis inside its optional audit scope."""

        async def score(skill, name):
            results = await self._collect(
                skill, list(samples), samples, env, runtime, root / "ensemble" / name / "score"
            )
            scores = {key: float(result.reward.success) for key, result in results.items()}
            record(root / "ensemble" / name / "scores.json", {"scores": scores, "mean": mean_score(scores)})
            return mean_score(scores)

        values = await parallel(score(node.skill, node.id) for node in winners)
        distances = [novelty(initial.bundle.files["SKILL.md"], node.skill.bundle.files["SKILL.md"]) for node in winners]
        record(
            root / "ensemble" / "winners.json",
            [
                {"id": node.id, "tree_id": node.tree_id, "score": value, "novelty": distance}
                for node, value, distance in zip(winners, values, distances)
            ],
        )
        if self.config.ensemble_strategy == "select":
            best = max(range(len(winners)), key=lambda index: (values[index], distances[index]))
            record(root / "ensemble" / "selection.json", {"selected": winners[best].id})
            return winners[best].skill
        threshold = max(values) - self.config.merge_threshold
        items = [(node.id, node.skill, value) for node, value in zip(winners, values) if value >= threshold]
        tree_ids = {node.id: node.tree_id for node in winners}
        items.sort(key=lambda item: hashlib.sha256(f"42:{tree_ids[item[0]]}:{item[0]}".encode()).hexdigest())
        record(root / "ensemble" / "merge_pool.json", {"threshold": threshold, "ids": [item[0] for item in items]})
        level = 0

        async def merge(left, right, name):
            left_text, right_text = left[1].bundle.files["SKILL.md"], right[1].bundle.files["SKILL.md"]
            text = await runtime.chat(
                messages=self.env_adapter.build_skill_merge_messages(
                    SkillForestSkillMergeContext(
                        left=left_text, right=right_text, left_score=left[2], right_score=right[2]
                    )
                ),
                directory=root / "ensemble" / name / "model",
            )
            if not text.strip().startswith("#"):
                raise ValueError("Skill synthesis requires complete Markdown starting with '#'")
            bundle = initial.bundle
            bundle.files["SKILL.md"] = text.strip() + "\n"
            candidate = initial.with_bundle(bundle)
            record(root / "ensemble" / name / "skill.json", candidate.snapshot().model_dump(mode="json"))
            record(root / "ensemble" / name / "parents.json", [left[0], right[0]])
            value = await score(candidate, name)
            return name, candidate, value

        while len(items) > 1:
            merged = await parallel(
                merge(items[index], items[index + 1], f"merge-{level:03d}-{index // 2:03d}")
                for index in range(0, len(items) - 1, 2)
            )
            if len(items) % 2:
                merged.append(items[-1])
            items = merged
            level += 1
        return items[0][1]
