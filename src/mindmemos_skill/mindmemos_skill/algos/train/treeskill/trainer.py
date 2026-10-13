"""One-round active TreeSkill training with immutable input and local artifacts."""

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any, Final
from urllib.parse import quote
from uuid import uuid4

from filelock import AsyncFileLock
from pydantic import JsonValue

from ....agents import Agent
from ....common.asyncio import map_bounded
from ....common.checkpoint import boundary, fingerprint, read_boundary
from ....common.json import write_json
from ....envs import Environment
from ....envs.data import DataSample
from ....envs.data.typing import SplitManifest
from ....logging import get_logger, stage
from ....skill import Skill, SkillContent, SkillSet, SkillSnapshot
from ...base import Algorithm
from ..base import SkillTrainer
from ..config import EnvAdapterSelection
from ..recovery import implementation_identity, sample_identity
from .artifacts import (
    experiment_config,
    model_usage,
    record_experiment_report,
    record_patch,
    record_training_stage,
    score_report,
)
from .config import TreeSkillConfig, TreeSkillTrainerConfig
from .construction.analysis import prepare_evidence
from .construction.fusion import fuse_node
from .construction.localization import locate_record
from .construction.markdown import (
    compile_tree_metadata,
    normalize_skill_text,
    parse_skill_markdown,
    parse_tree_with_metadata,
)
from .env_adapter import TreeSkillEnvAdapter
from .env_specific.selection import select_env_adapter
from .treeskill import TreeSkill
from .typing import (
    AnalysisInput,
    AnalysisOutcome,
    AppliedEditRecord,
    FusionFailure,
    FusionOutcome,
    LocalizationFailure,
    LocalizationOutcome,
    LocatedEvidence,
    MarkdownSkillTree,
    StageOutcome,
    TraceEvidence,
    TrainingRollout,
    TrajectoryAnalysisRecord,
    TrajectoryAnalyzer,
    TreeSkillAnalysisRuntime,
)

logger = get_logger(__name__)

ANALYSIS_TASK: Final[str] = "treeskill_trajectory_analysis"
"""Audit label for trajectory analysis outcomes."""

LOCALIZATION_TASK: Final[str] = "treeskill_evidence_localization"
"""Audit label for evidence localization outcomes."""

FUSION_TASK: Final[str] = "treeskill_node_fusion"
"""Audit label for node fusion outcomes."""


async def sample_tasks(
    *,
    env: Environment,
    skill: Skill,
    samples: Sequence[DataSample],
    directory: Path,
    repeats: int,
    concurrency: int,
    stage_name: str = "treeskill.train.rollout",
) -> tuple[list[TrainingRollout], list[dict[str, JsonValue]]]:
    """Actively sample independent rollouts, preserving every failed attempt.

    Args:
        env: Borrowed environment with caller-managed lifecycle.
        skill: Immutable Skill supplied to every rollout.
        samples: Materialized public tasks paired with private reward inputs.
        directory: Parent for unique slot and attempt directories.
        repeats: Independent rollouts per task.
        concurrency: Maximum simultaneous rollouts.
        stage_name: Explicit training or evaluation stage label.

    Returns:
        Completed physical trajectories and explicit failed-slot records.
    """
    started = perf_counter()
    slots = [(index, sample, repeat) for index, sample in enumerate(samples) for repeat in range(repeats)]
    completed = 0
    logger.info(
        f"[{stage_name}] started | tasks={len(samples)} | repeats={repeats} | concurrency={concurrency}"
        f" | skill_chars={len(skill.content)} | scored=0 | success_count=0 | accuracy=None"
        f" | completed=0/{len(slots)} | progress=0.0% | elapsed=0.0s"
    )
    reused_slots: set[str] = set()
    routed_lengths: dict[str, int] = {}

    async def execute_slot(
        slot: tuple[int, DataSample, int],
    ) -> tuple[TrainingRollout | None, dict[str, JsonValue] | None]:
        """Execute slot.

        Args:
            slot: Explicit slot for this operation.
        """
        index, sample, repeat = slot
        identifier = f"task-{index:06d}-sample-{repeat:03d}"
        slot_dir = directory / identifier
        checkpoint = slot_dir / "completed.json"
        identity = {
            "task": sample_identity(sample, env),
            "skill": skill.snapshot().model_dump(mode="json"),
            "environment": env.config.model_dump(mode="json"),
        }
        saved = boundary(checkpoint, identity)
        if saved is not None:
            reused_slots.add(identifier)
            if "routed_skill_chars" in saved:
                routed_lengths[identifier] = saved["routed_skill_chars"]
            if "failure" in saved:
                return None, saved["failure"]
            result = env.result_type.model_validate(saved["result"])
            return TrainingRollout(
                trajectory_id=identifier, sample=sample, result=result, work_dir=Path(saved["work_dir"])
            ), None
        attempts = [int(path.name.removeprefix("attempt-")) for path in slot_dir.glob("attempt-*")]
        attempt = max(attempts, default=0) + 1
        work_dir = slot_dir / f"attempt-{attempt:02d}"
        rollout_skill = skill
        if isinstance(skill, TreeSkill):
            routing_start = perf_counter()
            try:
                routing = await skill.route(sample.task)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                _save_outcome(
                    work_dir / "routing",
                    StageOutcome(error=error, failed_calls=1),
                    "treeskill_subtree_routing",
                    perf_counter() - routing_start,
                )
                failure = {"trajectory_id": identifier, "task_id": sample.task.id, "attempt": attempt, "error": error}
                write_json(work_dir / "error.json", failure)
                return None, failure
            _save_outcome(
                work_dir / "routing",
                StageOutcome(responses=[routing.response] if routing.response else []),
                "treeskill_subtree_routing",
                perf_counter() - routing_start,
            )
            write_json(work_dir / "routing/selection.json", routing.model_dump(mode="json"))
            routed_lengths[identifier] = len(routing.content)
            rollout_skill = Skill(
                SkillContent(
                    content=routing.content,
                    resources=skill.resources,
                    name=skill.source.name,
                    description=skill.source.description,
                    standard_format=False,
                )
            )
        try:
            result = await env.rollout(
                sample.task,
                work_dir,
                reward_data=sample.reward_data,
                skill_set=SkillSet({"0": rollout_skill}, max_concurrency=1),
                resume=False,
            )
        except Exception as exc:
            failure = {
                "trajectory_id": identifier,
                "task_id": sample.task.id,
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(work_dir / "error.json", failure)
            return None, failure
        boundary(
            checkpoint,
            identity,
            payload={
                "result": result.model_dump(mode="json"),
                "work_dir": str(work_dir),
                **({"routed_skill_chars": routed_lengths[identifier]} if identifier in routed_lengths else {}),
            },
        )
        return TrainingRollout(trajectory_id=identifier, sample=sample, result=result, work_dir=work_dir), None

    async def monitored_slot(
        slot: tuple[int, DataSample, int],
    ) -> tuple[TrainingRollout | None, dict[str, JsonValue] | None]:
        """Report each completed slot before the batch drains."""
        nonlocal completed
        index, sample, repeat = slot
        identifier = f"task-{index:06d}-sample-{repeat:03d}"
        record, failure = await execute_slot(slot)
        completed += 1
        logger.info(
            f"[{stage_name}] item.completed | sample_id={sample.task.id} repeat={repeat}"
            f" | completed={completed}/{len(slots)} | progress={completed / len(slots):.1%}"
            f" | elapsed={perf_counter() - started:.1f}s"
            f" | success={record.result.reward.success if record else None} reused={identifier in reused_slots}"
            f" | error={failure['error'] if failure else None}",
        )
        return record, failure

    outcomes = await map_bounded(monitored_slot, slots, concurrency=concurrency)
    records = [record for record, _ in outcomes if record is not None]
    records.sort(key=lambda record: (record.result.finished_at, record.trajectory_id))
    failures = [failure for _, failure in outcomes if failure is not None]
    write_json(
        directory / "index.json",
        {
            "completed": [
                {
                    "trajectory_id": record.trajectory_id,
                    "task_id": record.sample.task.id,
                    "work_dir": str(record.work_dir),
                }
                for record in records
            ],
            "failed": failures,
        },
    )
    report = score_report(records, failures, len(slots))
    logger.info(
        f"[{stage_name}] completed | scored={report['completed']} errors={len(failures)}"
        f" accuracy={report['success_rate']} reused={len(reused_slots)}"
        f" | mean_score={report['mean_score']} rollout_duration={report['rollout_duration_s']:.1f}s"
        f" | completed={completed}/{len(slots)} | progress=100.0% | elapsed={perf_counter() - started:.1f}s"
    )
    if routed_lengths:
        lengths = list(routed_lengths.values())
        logger.info(
            f"[{stage_name}] progress | routed_skill_chars_mean={sum(lengths) / len(lengths)}"
            f" | routed_skill_chars_min={min(lengths)} | routed_skill_chars_max={max(lengths)}"
        )
    return records, failures


@Algorithm.register("treeskill")
class TreeSkillTrainer(SkillTrainer[TreeSkillTrainerConfig]):
    """Collect with the full initial Skill, then analyze, locate and fuse once."""

    config_type = TreeSkillTrainerConfig
    """Validated model, sampling and reference algorithm settings."""

    def __init__(
        self,
        config: TreeSkillTrainerConfig,
    ) -> None:
        """Configure training without constructing models or accessing storage.

        Args:
            config: One-round protocol settings.
        """
        super().__init__(config.model_copy(deep=True))

    async def train(
        self,
        *,
        skill: Skill,
        env: Environment,
        train_samples: Sequence[DataSample],
        validation_samples: Sequence[DataSample] = (),
        work_dir: Path,
    ) -> Skill:
        """Run one training round and save its complete local audit.

        Args:
            skill: Initial Skill; its source and resources remain unchanged.
            env: Borrowed task environment, never a trajectory input.
            train_samples: Materialized tasks used for active sampling.
            validation_samples: Disjoint external data, excluded from one-round optimization.
            work_dir: Run directory; compatible committed work is reused automatically.

        Returns:
            Final routable Skill, including an unchanged candidate when no edit applies.

        Raises:
            ValueError: Samples are invalid or existing work has an incompatible identity.
        """
        root = Path(work_dir).expanduser().resolve()
        root.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncFileLock(str(root.parent / f".{root.name}.lock"), timeout=0, run_in_executor=False):
            logger.info("[treeskill.training] started")
            env_adapter = select_env_adapter(env)
            reused = (root / "completed.json").exists()
            selected = await self._run_train(skill, env, train_samples, validation_samples, root, env_adapter)
            report = json.loads((root / "summary.json").read_text())
            training = report["training"]
            logger.info(
                f"[treeskill.training] completed | scored={report['train']['completed']}"
                f" accuracy={report['train']['success_rate']} errors={report['train']['failed']}"
                f" | accepted_patches={training['accepted_patches']} rejected_patches={training['rejected_patches']}"
                f" | skill_chars={len(skill.content)}->{len(selected.content)}"
                f" | elapsed={report['duration_s']:.1f}s reused={reused}"
            )
            return selected

    async def _run_train(
        self,
        skill: Skill,
        env: Environment,
        train_samples: Sequence[DataSample],
        validation_samples: Sequence[DataSample],
        root: Path,
        env_adapter: TreeSkillEnvAdapter,
    ) -> TreeSkill:
        """Validate a run and reuse committed calls under the directory lock."""
        env_adapter.validate_environment(env)
        if not train_samples:
            raise ValueError("TreeSkill requires training tasks")
        if len({sample.task.id for sample in train_samples}) != len(train_samples):
            raise ValueError("Training task IDs must be unique")
        external_ids = [sample.task.id for sample in validation_samples]
        if len(set(external_ids)) != len(external_ids) or set(external_ids) & {
            sample.task.id for sample in train_samples
        }:
            raise ValueError("External validation IDs must be unique and disjoint from training")
        identity = {
            "recovery_protocol": 2,
            "implementation": implementation_identity(self, env),
            "config": {"algorithm": self.config.model_dump(mode="json")},
            "adapter": {
                "class_name": TreeSkillEnvAdapter.name_for(type(env_adapter)),
                "config": env_adapter.config.model_dump(mode="json"),
            },
            "skill": skill.snapshot().model_dump(mode="json"),
            "train": [sample_identity(sample, env) for sample in train_samples],
            "validation": [sample_identity(sample, env) for sample in validation_samples],
            "environment": env.config.model_dump(mode="json"),
        }
        manifest = root / "recovery_manifest.json"
        if root.exists() and any(root.iterdir()) and not manifest.exists():
            raise ValueError("No compatible recovery manifest; legacy runs require a new work_dir")
        boundary(manifest, identity, payload={"protocol": 2})
        for path in chain(root.glob("rollouts/train/*/completed.json"), root.glob("items/**/completed.json")):
            read_boundary(path)
        completed = boundary(root / "completed.json", identity)
        if completed is not None:
            return TreeSkill.from_snapshot(
                SkillSnapshot.model_validate(completed["skill"]),
                router=Agent.from_config(
                    self.config.router.model_copy(
                        update={
                            "temperature": self.config.routing_temperature,
                            "max_completion_tokens": self.config.routing_max_tokens,
                        }
                    ),
                ),
            )
        start = perf_counter()
        config = experiment_config(self.config)
        write_json(root / "config.json", config)
        summary = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages": {},
            "train": None,
            "val": None,
            "test": None,
        }
        record_experiment_report(root, summary)
        try:
            result = await self._train(skill, env, train_samples, root, summary, env_adapter)
            write_json(root / "final_skill.json", result.snapshot().model_dump(mode="json"))
            result.export(root / "final_skill", overwrite=True)
            summary["status"] = (
                "completed" if summary["train"]["complete"] and not summary["training"]["has_failures"] else "partial"
            )
            if summary["status"] == "completed":
                boundary(
                    root / "completed.json", identity, payload={"skill": result.snapshot().model_dump(mode="json")}
                )
            return result
        except BaseException as exc:
            summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            summary["model_usage"] = model_usage(root)
            summary["duration_s"] = perf_counter() - start
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            record_experiment_report(root, summary)

    async def _train(
        self,
        source: Skill,
        env: Environment,
        samples: Sequence[DataSample],
        root: Path,
        summary: dict[str, Any],
        env_adapter: TreeSkillEnvAdapter,
    ) -> TreeSkill:
        """Collect once, analyze, locate on the initial tree and fuse bottom-up."""
        config = self.config
        initial = Skill(source.source)
        start = perf_counter()
        with stage("treeskill.rollout"):
            records, failures = await sample_tasks(
                env=env,
                skill=initial,
                samples=samples,
                directory=root / "rollouts/train",
                repeats=config.samples_per_task,
                concurrency=config.rollout_concurrency,
            )
        summary["stages"]["collection_s"] = perf_counter() - start
        summary["train"] = score_report(records, failures, len(samples) * config.samples_per_task)
        summary["train"]["skill_role"] = "initial_full_skill"
        evidence = prepare_evidence(records, config, initial.snapshot().skill_hash)
        record_training_stage(root, "evidence", evidence)
        tree = parse_skill_markdown(initial.content)
        current = tree
        analyses, analysis_failures = [], []
        located, localization_failures = [], []
        edits, fusion_failures = [], []
        reason = None
        if len(evidence) < config.min_trajectories:
            reason = "below_minimum_trajectory_count"
        elif len(evidence) > config.max_trajectories:
            raise ValueError("Collected evidence exceeds max_trajectories")
        elif not tree.nodes:
            reason = "skill_has_no_content_bearing_markdown_headings"
        if reason is None:
            analysis_config = config.optimizer.model_copy(
                update={
                    "temperature": config.analysis_temperature,
                    "max_completion_tokens": config.analysis_max_tokens,
                    "parse_retry_mode": "feedback",
                }
            )
            failure_config = (config.failure_optimizer or config.optimizer).model_copy(
                update={
                    "temperature": config.analysis_temperature,
                    "max_completion_tokens": config.analysis_max_tokens,
                }
            )
            analyzer = env_adapter.create_analyzer(
                TreeSkillAnalysisRuntime(
                    config=config,
                    agent=Agent.from_config(analysis_config),
                    failure_agent=Agent.from_config(failure_config),
                )
            )
            start = perf_counter()
            with stage("treeskill.analyze"):
                analyses, analysis_failures = await self._analyze(evidence, records, analyzer, env_adapter, root)
            summary["stages"]["analysis_s"] = perf_counter() - start
            record_training_stage(root, "analysis", analyses, analysis_failures)
            if not analyses:
                reason = "no_valid_trajectory_analysis_records"
        if analyses:
            with stage("treeskill.merge"):
                start = perf_counter()
                located, localization_failures = await self._localize(tree, analyses, root)
                summary["stages"]["localization_s"] = perf_counter() - start
                record_training_stage(root, "localization", located, localization_failures)
                if located:
                    start = perf_counter()
                    current, edits, fusion_failures = await self._fuse(tree, located, root)
                    summary["stages"]["fusion_s"] = perf_counter() - start
                    record_training_stage(root, "fusion", edits, fusion_failures)
                else:
                    reason = "no_reusable_evidence_localized"
        for name, executed in (
            ("analysis", "analysis_s" in summary["stages"]),
            ("localization", "localization_s" in summary["stages"]),
            ("fusion", "fusion_s" in summary["stages"]),
        ):
            if not executed:
                logger.info(
                    f"[{f'treeskill.train.{name}'}] skipped | reason={reason}"
                    f" | skill_chars_before={len(initial.content)} | skill_chars_after={len(current.full_content)}"
                    f" | skill_chars_delta={len(current.full_content) - len(initial.content)}"
                )
        changed = normalize_skill_text(current.full_content) != normalize_skill_text(initial.content)
        summary["training"] = {
            "skill_chars_before": len(initial.content),
            "skill_chars_after": len(current.full_content),
            "skill_chars_delta": len(current.full_content) - len(initial.content),
            "extracted_items": sum(len(record.items) for record in analyses),
            "proposed_patches": len(edits),
            "input_trajectories": len(evidence),
            "initial_node_count": len(tree.nodes),
            "final_node_count": len(current.nodes),
            "analysis_records": len(analyses),
            "analysis_failures": analysis_failures,
            "located_evidence": len(located),
            "localization_failures": [item.model_dump(mode="json") for item in localization_failures],
            "fusion_failures": [item.model_dump(mode="json") for item in fusion_failures],
            "accepted_patches": sum(item.accepted and item.operation != "reject" for item in edits),
            "rejected_patches": sum(not item.accepted or item.operation == "reject" for item in edits),
            "changed": changed,
            "reason": reason or (None if changed else "no_effective_change"),
            "has_failures": bool(analysis_failures or localization_failures or fusion_failures),
        }
        metadata = compile_tree_metadata(current)
        write_json(root / "tree_metadata.json", metadata)
        router_config = config.router.model_copy(
            update={
                "temperature": config.routing_temperature,
                "max_completion_tokens": config.routing_max_tokens,
            }
        )
        return TreeSkill(
            TreeSkillConfig(
                router=router_config,
                tree_metadata=metadata,
                env_adapter=EnvAdapterSelection(
                    class_name=TreeSkillEnvAdapter.name_for(type(env_adapter)),
                    config=env_adapter.config.model_dump(mode="json"),
                ),
            ),
            source=initial.with_content(current.full_content).source if changed else initial.source,
            router=Agent.from_config(router_config),
        )

    async def _analyze(
        self,
        evidence: list[TraceEvidence],
        records: list[TrainingRollout],
        analyzer: TrajectoryAnalyzer,
        adapter: TreeSkillEnvAdapter,
        root: Path,
    ) -> tuple[list[TrajectoryAnalysisRecord], list[str]]:
        """Schedule and persist one analysis at a time under trainer ownership."""
        logger.info(f"[treeskill.train.analysis] started | concurrency={self.config.analysis_concurrency}")
        trajectories = {record.trajectory_id: record for record in records}

        async def execute(item: TraceEvidence) -> AnalysisOutcome:
            directory = root / "items/analysis" / f"item-{quote(item.trajectory_id, safe='')}"
            identity = item.model_dump(mode="json")
            saved = _load_item(directory, identity)
            if saved is not None:
                logger.info(f"[treeskill.train.analysis] item.reused | sample_id={item.trajectory_id}")
                return AnalysisOutcome.model_validate(saved)
            attempt = _next_attempt(directory)
            workspace = attempt / "workspace"
            workspace.mkdir(parents=True)
            trajectory = trajectories[item.trajectory_id]
            # Workspace provisioning errors are infrastructure failures, not model failures.
            if item.score is not None and item.score < self.config.success_score_threshold:
                adapter.prepare_analysis_workspace(trajectory, workspace)
            start = perf_counter()
            try:
                outcome = await analyzer(AnalysisInput(evidence=item, trajectory=trajectory, workspace=workspace))
            except BaseException as exc:
                _save_outcome(
                    attempt,
                    StageOutcome(interrupted=True, error=f"{type(exc).__name__}: {exc}"),
                    ANALYSIS_TASK,
                    perf_counter() - start,
                )
                raise
            _save_outcome(attempt, outcome, ANALYSIS_TASK, perf_counter() - start)
            if outcome.error is None and outcome.record is not None:
                boundary(directory / "completed.json", identity, payload=outcome.model_dump(mode="json"))
            error = outcome.error or ("No analysis record" if outcome.record is None else None)
            logger.log(
                logging.WARNING if error else logging.INFO,
                f"[treeskill.train.analysis] item.completed | sample_id={item.trajectory_id}"
                f" | error={error} elapsed={perf_counter() - start:.1f}s",
            )
            return outcome

        outcomes = []
        for phase in adapter.analysis_phases(evidence, self.config.success_score_threshold):
            results = await map_bounded(execute, phase, concurrency=self.config.analysis_concurrency)
            outcomes.extend(zip(phase, results, strict=True))
        valid = [outcome.record for _, outcome in outcomes if outcome.record is not None]
        logger.info(
            f"[treeskill.train.analysis] progress | analysis_records={len(valid)}"
            f" | extracted_items={sum((len(record.items) for record in valid))}"
            f" | success_source_items={sum((len(record.items) for record in valid if record.record_source == 'success'))}"
            f" | failure_source_items={sum((len(record.items) for record in valid if record.record_source == 'error'))}"
            f" | unlabeled_source_items={sum((len(record.items) for record in valid if record.record_source == 'unlabeled'))}"
        )
        return (
            [outcome.record for _, outcome in outcomes if outcome.record is not None],
            [item.trajectory_id for item, outcome in outcomes if outcome.error or outcome.record is None],
        )

    async def _localize(
        self,
        tree: MarkdownSkillTree,
        analyses: list[TrajectoryAnalysisRecord],
        root: Path,
    ) -> tuple[list[LocatedEvidence], list[LocalizationFailure]]:
        """Own localization concurrency, retries across runs and item checkpoints."""
        logger.info(f"[treeskill.train.localization] started | concurrency={self.config.localization_concurrency}")
        agents = tuple(
            Agent.from_config(
                self.config.optimizer.model_copy(
                    update={
                        "temperature": self.config.localization_temperature,
                        "max_completion_tokens": budget,
                    }
                )
            )
            for budget in (self.config.localization_max_tokens, min(self.config.localization_max_tokens * 2, 16384))
        )

        async def execute(record: TrajectoryAnalysisRecord) -> LocalizationOutcome:
            directory = root / "items/localization" / f"item-{quote(record.instance_id, safe='')}"
            identity = {"tree": compile_tree_metadata(tree), "record": record.model_dump(mode="json")}
            saved = _load_item(directory, identity)
            if saved is not None:
                logger.info(f"[treeskill.train.localization] item.reused | sample_id={record.instance_id}")
                return LocalizationOutcome.model_validate(saved)
            attempt = _next_attempt(directory)
            start = perf_counter()
            try:
                outcome = await locate_record(tree, record, agents=agents)
            except BaseException as exc:
                _save_outcome(
                    attempt,
                    StageOutcome(interrupted=True, error=f"{type(exc).__name__}: {exc}"),
                    LOCALIZATION_TASK,
                    perf_counter() - start,
                )
                raise
            _save_outcome(attempt, outcome, LOCALIZATION_TASK, perf_counter() - start)
            if outcome.error is None:
                boundary(directory / "completed.json", identity, payload=outcome.model_dump(mode="json"))
            logger.log(
                logging.WARNING if outcome.error else logging.INFO,
                f"[treeskill.train.localization] item.completed | sample_id={record.instance_id}"
                f" | error={outcome.error} elapsed={perf_counter() - start:.1f}s",
            )
            return outcome

        outcomes = await map_bounded(execute, analyses, concurrency=self.config.localization_concurrency)
        located = [item for outcome in outcomes for item in outcome.evidence]
        logger.info(
            f"[treeskill.train.localization] progress | located_evidence={len(located)}"
            f" | covered_nodes={len({item.target_node_id for item in located})}"
        )
        return (
            [item for outcome in outcomes for item in outcome.evidence],
            [
                LocalizationFailure(instance_id=record.instance_id, error=outcome.error)
                for record, outcome in zip(analyses, outcomes, strict=True)
                if outcome.error
            ],
        )

    async def _fuse(
        self, tree: MarkdownSkillTree, evidence: list[LocatedEvidence], root: Path
    ) -> tuple[MarkdownSkillTree, list[AppliedEditRecord], list[FusionFailure]]:
        """Commit a successful prefix of bottom-up edits with stable tree identities."""
        grouped: dict[str, list[LocatedEvidence]] = defaultdict(list)
        for item in evidence:
            grouped[item.target_node_id].append(item)
        targets = sorted(
            grouped, key=lambda key: (-len(tree.node_by_id[key].heading_path), tree.node_by_id[key].source_line)
        )
        agent = Agent.from_config(
            self.config.optimizer.model_copy(
                update={
                    "temperature": self.config.fusion_temperature,
                    "max_completion_tokens": self.config.fusion_max_tokens,
                }
            )
        )
        patches = root / "patches"
        if patches.exists():
            archived = root / "superseded_patches" / uuid4().hex
            archived.parent.mkdir(parents=True, exist_ok=True)
            patches.replace(archived)
        current, edits, failures = tree, [], []
        committed_prefix = True
        logger.info(
            f"[treeskill.train.fusion] started | skill_chars_before={len(tree.full_content)}"
            f" | node_count_before={len(tree.nodes)} | proposed_patches=0 | accepted_patches=0"
            " | rejected_patches=0"
        )
        for index, target in enumerate(targets):
            directory = root / "items/fusion" / f"{index:06d}-{target}"
            identity = {
                "content": current.full_content,
                "tree": compile_tree_metadata(current),
                "evidence": [item.model_dump(mode="json") for item in grouped[target]],
            }
            saved = _load_item(directory, identity) if committed_prefix else None
            if saved is not None:
                outcome = FusionOutcome.model_validate(saved)
            else:
                attempt = _next_attempt(directory)
                start = perf_counter()
                try:
                    outcome = await fuse_node(current, target, grouped[target], agent=agent)
                except BaseException as exc:
                    _save_outcome(
                        attempt,
                        StageOutcome(interrupted=True, error=f"{type(exc).__name__}: {exc}"),
                        FUSION_TASK,
                        perf_counter() - start,
                    )
                    raise
                _save_outcome(attempt, outcome, FUSION_TASK, perf_counter() - start)
            for patch, edit in zip(outcome.patches, outcome.edits, strict=True):
                record_patch(
                    root / "patches" / f"{len(edits) + 1:04d}-{target}-{patch.payload['result']['operation']}",
                    before=patch.before,
                    after=patch.after,
                    payload=patch.payload,
                )
                edits.append(edit)
            current = parse_tree_with_metadata(outcome.content, outcome.tree_metadata)
            if outcome.error:
                failures.append(FusionFailure(target_node_id=target, error=outcome.error))
                committed_prefix = False
            elif committed_prefix and saved is None:
                boundary(directory / "completed.json", identity, payload=outcome.model_dump(mode="json"))
            accepted = sum(item.accepted and item.operation != "reject" for item in edits)
            logger.log(
                logging.WARNING if outcome.error else logging.INFO,
                f"[treeskill.train.fusion] progress | node_id={target} done={index + 1}/{len(targets)}"
                f" | proposed_patches={len(edits)} accepted_patches={accepted} rejected_patches={len(edits) - accepted}"
                f" | skill_chars={len(tree.full_content)}->{len(current.full_content)}"
                f" | reused={saved is not None} error={outcome.error}",
            )
        return current, edits, failures


def _load_item(directory: Path, identity: dict) -> dict | None:
    """Invalidate dependent work when a retried upstream item produces new inputs."""
    checkpoint = directory / "completed.json"
    if checkpoint.exists():
        saved = read_boundary(checkpoint)
        if saved.identity != fingerprint(identity):
            archived = directory / f"superseded-{uuid4().hex}.json"
            checkpoint.replace(archived)
            return None
        return saved.payload
    return None


def _next_attempt(directory: Path) -> Path:
    """Allocate a fresh item workspace without overwriting interrupted executions."""
    attempts = [int(path.name.removeprefix("attempt-")) for path in directory.glob("attempt-*")]
    attempt = directory / f"attempt-{max(attempts, default=0) + 1:03d}"
    attempt.mkdir(parents=True)
    return attempt


def _save_outcome(directory: Path, outcome: StageOutcome, task: str, duration_s: float) -> None:
    """Persist returned execution evidence; artifact write failures propagate."""
    for name, content in outcome.artifacts.items():
        path = directory / "reports" / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid analysis artifact filename: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    write_json(directory / "outcome.json", {**outcome.model_dump(mode="json"), "task": task, "duration_s": duration_s})


async def run_experiment(
    *,
    trainer: TreeSkillTrainer,
    skill: Skill,
    env: Environment,
    train_samples: Sequence[DataSample],
    work_dir: Path,
    validation_samples: Sequence[DataSample] = (),
    test_samples: Sequence[DataSample] = (),
    preflight: Callable[[Path], Awaitable[dict[str, JsonValue]]] | None = None,
) -> Skill:
    """Train once and report final held-out scores without using them for selection.

    Args:
        trainer: Configured TreeSkill implementation, with no database dependencies.
        skill: Initial Skill and unchanged auxiliary resources.
        env: Caller-owned environment used for collection and final testing.
        train_samples: Tasks from which training evidence is actively sampled.
        work_dir: Fresh root for all experiment artifacts.
        validation_samples: Empty for the pinned one-round protocol.
        test_samples: Optional disjoint tasks, executed once after training.
        preflight: Optional local runtime check receiving the experiment preflight directory.

    Returns:
        The final trained Skill regardless of held-out score.

    Raises:
        ValueError: Splits overlap, validation selection is requested, or root exists.
    """
    start = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if validation_samples:
        raise ValueError("The reference protocol has no validation selection; supply an empty val set")
    SplitManifest(
        {
            "train": [sample.task.id for sample in train_samples],
            "val": [],
            "test": [sample.task.id for sample in test_samples],
        }
    )
    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = experiment_config(trainer.config)
    write_json(root / "config.json", config)
    logger.info(f"[treeskill.experiment] started | tasks={len(train_samples) + len(test_samples)}")
    summary = {"status": "running", "train": None, "val": None, "test": None}
    record_experiment_report(root, summary)
    try:
        if preflight is not None:
            preflight_start = perf_counter()
            write_json(root / "preflight/result.json", await preflight(root / "preflight"))
            summary["preflight_duration_s"] = perf_counter() - preflight_start
        final = await trainer.train(
            skill=skill, env=env, train_samples=train_samples, validation_samples=(), work_dir=root / "training"
        )
        summary.update(json.loads((root / "training/summary.json").read_text()))
        summary["status"] = "testing" if test_samples else "completed"
        record_experiment_report(root, summary)
        if test_samples:
            testing_start = perf_counter()
            with stage("treeskill.test"):
                records, failures = await sample_tasks(
                    env=env,
                    skill=final,
                    samples=test_samples,
                    directory=root / "test",
                    stage_name="treeskill.evolved",
                    repeats=1,
                    concurrency=trainer.config.rollout_concurrency,
                )
                summary["test"] = score_report(records, failures, len(test_samples))
                summary["test"]["skill_role"] = "final_routed_skill"
            summary["stages"]["test_s"] = perf_counter() - testing_start
        summary["status"] = (
            "completed"
            if summary["train"]["complete"]
            and not summary["training"]["has_failures"]
            and (summary["test"] is None or summary["test"]["complete"])
            else "partial"
        )
        summary["final_skill"] = "training/final_skill"
        return final
    except BaseException as exc:
        if (root / "training/summary.json").exists():
            summary.update(json.loads((root / "training/summary.json").read_text()))
        summary.update(
            status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        summary["duration_s"] = perf_counter() - start
        summary["started_at"] = started_at
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["model_usage"] = model_usage(root)
        record_experiment_report(root, summary)
        logger.log(
            logging.ERROR if summary["status"] == "failed" else logging.INFO,
            f"[treeskill.experiment] {summary['status']} | elapsed={summary['duration_s']:.1f}s"
            f" | train_accuracy={summary['train']['success_rate'] if summary['train'] else None}"
            f" test_accuracy={summary['test']['success_rate'] if summary['test'] else None}"
            f" | error={summary.get('error')}",
        )
