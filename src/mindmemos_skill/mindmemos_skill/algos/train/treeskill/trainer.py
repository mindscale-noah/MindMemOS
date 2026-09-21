"""One-round active TreeSkill training with immutable input and local artifacts."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ....agents import OpenAIChatAgent
from ....common.json import write_json
from ....envs import Environment
from ....envs.data import DataSample
from ....skill import Skill, SkillBundle, SkillSet
from ...base import Algorithm
from ..base import SkillTrainer
from .analysis import TreeSkillTrajectoryAnalyzer
from .artifacts import REFERENCE_COMMIT, experiment_config, model_usage, normalize_skill_text
from .config import TreeSkillConfig, TreeSkillTrainerConfig
from .evidence import render_trajectory
from .fusion import TreeSkillNodeFuser
from .fusion_tree import compile_tree_metadata, parse_skill_markdown
from .localization import TreeSkillEvidenceLocator
from .model import RecordedChatModel, RecordedRouter
from .skill import TreeSkill
from .typing import AnnotationMode, TraceEvidence, TrainingRollout


def sample_tasks(*, env, skill, samples, directory, repeats, concurrency):
    """Actively sample independent rollouts, preserving every failed attempt.

    Args:
        env: Borrowed environment with caller-managed lifecycle.
        skill: Immutable Skill supplied to every rollout.
        samples: Materialized public tasks paired with private reward inputs.
        directory: Parent for unique slot and attempt directories.
        repeats: Independent rollouts per task.
        concurrency: Maximum simultaneous rollouts.

    Returns:
        Completed physical trajectories and explicit failed-slot records.
    """
    slots = [(index, sample, repeat) for index, sample in enumerate(samples) for repeat in range(repeats)]

    def execute_slot(slot):
        """Execute slot.

        Args:
            slot: Explicit slot for this operation.
        """
        index, sample, repeat = slot
        identifier = f"task-{index:06d}-sample-{repeat:03d}"
        attempt = 1
        work_dir = directory / identifier / f"attempt-{attempt:02d}"
        try:
            result = env.rollout(
                sample.task,
                work_dir,
                reward_data=sample.reward_data,
                skill_set=SkillSet({"0": skill}, max_concurrency=1),
                resume=False,
            )
            return TrainingRollout(trajectory_id=identifier, sample=sample, result=result, work_dir=work_dir), None
        except Exception as exc:
            failure = {
                "trajectory_id": identifier,
                "task_id": sample.task.id,
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(work_dir / "error.json", failure)
        return None, failure

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        outcomes = list(executor.map(execute_slot, slots))
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
    return records, failures


def score_report(records, failures, expected):
    """Report environment scores with explicit execution coverage.

    Args:
        records: Completed rollouts with actual rewards.
        failures: Failed execution slots, never substituted with zero rewards.
        expected: Requested number of physical sampling slots.

    Returns:
        Mean score and success fraction over completed slots, with denominators.
    """
    count = len(records)
    return {
        "expected": expected,
        "completed": count,
        "failed": len(failures),
        "skipped": len(failures),
        "score_sum": sum(record.result.reward.score for record in records),
        "mean_score": sum(record.result.reward.score for record in records) / count if count else None,
        "success_count": sum(record.result.reward.success for record in records),
        "success_rate": sum(record.result.reward.success for record in records) / count if count else None,
        "complete": count == expected,
        "rollout_duration_s": sum(record.result.duration_s for record in records),
        "usage": {
            "input_tokens": sum(record.result.usage.input_tokens for record in records),
            "output_tokens": sum(record.result.usage.output_tokens for record in records),
            "total_tokens": sum(record.result.usage.total_tokens for record in records),
        },
    }


@Algorithm.register("treeskill")
class TreeSkillTrainer(SkillTrainer[TreeSkillTrainerConfig]):
    """Collect with the full initial Skill, then analyze, locate and fuse once."""

    config_type = TreeSkillTrainerConfig
    """Validated model, sampling and reference algorithm settings."""

    def __init__(self, config: TreeSkillTrainerConfig, *, agent_factory=OpenAIChatAgent):
        """Configure training without constructing models or accessing storage.

        Args:
            config: One-round protocol settings.
            agent_factory: Model transport constructor for injection and offline testing.
        """
        super().__init__(config)
        self.agent_factory = agent_factory

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
            validation_samples: Must be empty; upstream has no validation selection.
            work_dir: Fresh experiment directory; existing directories are rejected.

        Returns:
            Final routable Skill, including an unchanged candidate when no edit applies.

        Raises:
            ValueError: Samples are empty, duplicated, or validation is requested.
        """
        if not train_samples:
            raise ValueError("TreeSkill requires training tasks")
        if len({sample.task.id for sample in train_samples}) != len(train_samples):
            raise ValueError("Training task IDs must be unique")
        root = Path(work_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
        start = perf_counter()
        config = experiment_config(self.config)
        write_json(root / "config.json", config)
        summary = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "reference_commit": REFERENCE_COMMIT,
            "stages": {},
            "train": None,
            "val": None,
            "test": None,
        }
        write_json(root / "summary.json", summary)
        try:
            result = await self._train(skill, env, train_samples, root, summary)
            write_json(root / "final_skill.json", result.snapshot().model_dump(mode="json"))
            result.export(root / "final_skill")
            summary["status"] = (
                "completed" if summary["train"]["complete"] and not summary["training"]["has_failures"] else "partial"
            )
            return result
        except BaseException as exc:
            summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            summary["model_usage"] = model_usage(root / "model_calls")
            summary["duration_s"] = perf_counter() - start
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(root / "summary.json", summary)

    async def _train(self, source, env, samples, root, summary):
        """Execute the upstream stage order without adding selection or extra rounds."""
        config = self.config
        initial = Skill(bundle=source.bundle.model_copy(deep=True), resources=source.resources.model_copy(deep=True))
        start = perf_counter()
        records, failures = await asyncio.to_thread(
            sample_tasks,
            env=env,
            skill=initial,
            samples=samples,
            directory=root / "rollouts/train",
            repeats=config.samples_per_task,
            concurrency=config.rollout_concurrency,
        )
        summary["stages"]["collection_s"] = perf_counter() - start
        summary["train"] = score_report(records, failures, len(samples) * config.samples_per_task)
        if config.require_skill_match:
            import json

            expected = initial.snapshot().local_snapshot_hash
            for record in records:
                request = json.loads((record.work_dir / "audit/request.json").read_text(encoding="utf-8"))
                if request["skills"][0]["local_snapshot_hash"] != expected:
                    raise ValueError("Sampled trajectory does not match the initial Skill")
        summary["train"]["skill_role"] = "initial_full_skill"
        evidence = [
            TraceEvidence(
                trajectory_id=record.trajectory_id,
                task_id=record.sample.task.id,
                transcript=render_trajectory(record.result.trajectory, config.transcript_max_chars),
                score=record.result.reward.score if config.annotation_mode != AnnotationMode.IGNORE else None,
                annotation_detail=record.result.reward.detail
                if config.annotation_mode != AnnotationMode.IGNORE
                else None,
                annotation_metadata=record.result.reward.model_extra or {}
                if config.annotation_mode != AnnotationMode.IGNORE
                else {},
            )
            for record in records
        ]
        write_json(root / "stages/evidence.json", [item.model_dump(mode="json") for item in evidence])
        tree = parse_skill_markdown(initial.bundle.files["SKILL.md"])
        current = tree
        summary["training"] = {
            "input_trajectories": len(evidence),
            "initial_node_count": len(tree.nodes),
            "analysis_records": 0,
            "analysis_failures": [],
            "located_evidence": 0,
            "localization_failures": [],
            "fusion_failures": [],
            "accepted_patches": 0,
            "rejected_patches": 0,
            "changed": False,
        }
        metrics = summary["training"]
        reason = None
        if len(evidence) < config.min_trajectories:
            reason = "below_minimum_trajectory_count"
        elif len(evidence) > config.max_trajectories:
            raise ValueError("Collected evidence exceeds max_trajectories")
        elif not tree.nodes:
            reason = "skill_has_no_content_bearing_markdown_headings"
        else:
            model = RecordedChatModel(config.optimizer, root / "model_calls/optimizer", factory=self.agent_factory)
            if config.analysis_adapter == "spreadsheetbench_reference":
                from .env_specific.spreadsheetbench import SpreadsheetBenchReferenceAnalyzer

                failure_model = RecordedChatModel(
                    config.failure_optimizer or config.optimizer,
                    root / "model_calls/failure",
                    factory=self.agent_factory,
                )
                analyzer = SpreadsheetBenchReferenceAnalyzer(
                    chat_model=model,
                    failure_chat_model=failure_model,
                    task=config.analysis_task,
                    output_root=root / "stages/reference_analysis",
                    concurrency=config.analysis_concurrency,
                    success_score_threshold=config.success_score_threshold,
                    temperature=config.analysis_temperature,
                    max_tokens=config.analysis_max_tokens,
                    max_turns=config.analysis_max_turns,
                    shell_timeout_seconds=config.shell_timeout_seconds,
                )
            else:
                analyzer = TreeSkillTrajectoryAnalyzer(
                    chat_model=model,
                    task=config.analysis_task,
                    concurrency=config.analysis_concurrency,
                    success_score_threshold=config.success_score_threshold,
                    temperature=config.analysis_temperature,
                    max_tokens=config.analysis_max_tokens,
                )
            start = perf_counter()
            analyses, analysis_failures = await analyzer.analyze(
                evidence, trajectories_by_id={record.trajectory_id: record for record in records}
            )
            summary["stages"]["analysis_s"] = perf_counter() - start
            metrics.update(analysis_records=len(analyses), analysis_failures=analysis_failures)
            write_json(
                root / "stages/analysis.json",
                {"records": [item.model_dump(mode="json") for item in analyses], "failures": analysis_failures},
            )
            if not analyses:
                reason = "no_valid_trajectory_analysis_records"
            else:
                start = perf_counter()
                located, localization_failures = await TreeSkillEvidenceLocator(
                    chat_model=model,
                    task=config.localization_task,
                    concurrency=config.localization_concurrency,
                    temperature=config.localization_temperature,
                    max_tokens=config.localization_max_tokens,
                ).locate(tree, analyses)
                summary["stages"]["localization_s"] = perf_counter() - start
                metrics.update(
                    located_evidence=len(located),
                    localization_failures=[item.model_dump(mode="json") for item in localization_failures],
                )
                write_json(
                    root / "stages/localization.json",
                    {
                        "evidence": [item.model_dump(mode="json") for item in located],
                        "failures": metrics["localization_failures"],
                    },
                )
                if not located:
                    reason = "no_reusable_evidence_localized"
                else:
                    start = perf_counter()
                    current, edits, fusion_failures = await TreeSkillNodeFuser(
                        chat_model=model,
                        task=config.fusion_task,
                        temperature=config.fusion_temperature,
                        max_tokens=config.fusion_max_tokens,
                        audit_dir=root / "patches",
                    ).fuse(tree, located)
                    summary["stages"]["fusion_s"] = perf_counter() - start
                    metrics.update(
                        accepted_patches=sum(item.accepted and item.operation != "reject" for item in edits),
                        rejected_patches=sum(not item.accepted or item.operation == "reject" for item in edits),
                        fusion_failures=[item.model_dump(mode="json") for item in fusion_failures],
                    )
                    write_json(
                        root / "stages/fusion.json",
                        {
                            "edits": [item.model_dump(mode="json") for item in edits],
                            "failures": metrics["fusion_failures"],
                        },
                    )
        changed = normalize_skill_text(current.full_content) != normalize_skill_text(initial.bundle.files["SKILL.md"])
        metrics.update(
            changed=changed,
            final_node_count=len(current.nodes),
            reason=reason or (None if changed else "no_effective_change"),
        )
        metrics["has_failures"] = bool(
            metrics["analysis_failures"] or metrics["localization_failures"] or metrics["fusion_failures"]
        )
        write_json(root / "tree_metadata.json", compile_tree_metadata(current))
        bundle = SkillBundle(files={"SKILL.md": current.full_content}) if changed else initial.bundle
        candidate = TreeSkill(
            TreeSkillConfig(router=config.router, reference_routing=True, tree_metadata=compile_tree_metadata(current)),
            bundle=bundle,
            resources=initial.resources,
            router=RecordedRouter(config.router, root / "model_calls/router", factory=self.agent_factory),
        )
        return candidate
