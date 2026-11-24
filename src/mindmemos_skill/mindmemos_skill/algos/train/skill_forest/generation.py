"""Candidate-local reflection and evidence reduction for forest edges."""

import random
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from ....common.json import extract_json_object
from ....envs.data.typing import DataSample
from ....envs.typing import RolloutResult
from ....logging import get_logger
from ....skill import Skill
from .runtime import ForestRuntime, parallel
from .typing import (
    EvidenceBatch,
    Patch,
    SkillForestFinalPatchContext,
    SkillForestPatchMergeContext,
    SkillForestRankingContext,
    SkillForestReflectionContext,
    SkillForestTrajectoryContext,
)

logger = get_logger(__name__)


class InvalidPatchError(ValueError):
    """The optimizer response violates the patch protocol, unlike transport or I/O failures."""


def _parse_reference(
    text: str,
    *,
    reflection: bool = False,
    source: str | None = None,
    edit_budget: int | None = None,
    merge_level: int | None = None,
) -> Patch:
    """Parse reference metadata without converting provider failures into patch failures."""
    try:
        raw = extract_json_object(text)
        support = max(int(raw.get("batch_size", 0) or 0), 1) if reflection else 1
        payload = raw.get("patch") if reflection else raw
        if not isinstance(payload, dict):
            raise ValueError("Reference response requires a patch object")
        return Patch.from_reference(
            payload, source=source, support=support, edit_budget=edit_budget, merge_level=merge_level
        )
    except (ValueError, TypeError) as exc:
        raise InvalidPatchError(str(exc)) from exc


def _concatenate_reference(patches: Sequence[Patch], *, reasoning: str, level: int | None = None) -> Patch:
    """Retain valid evidence when reference aggregation returns malformed model output."""
    edits = []
    for patch in patches:
        for edit in deepcopy(patch.optimizer_input()["edits"]):
            if level is not None:
                edit.setdefault("merge_level", level)
            edits.append(edit)
    logger.warning("[skill_forest.patch_merge] fallback | reason=invalid_reference_response")
    return Patch.from_reference({"reasoning": reasoning, "edits": edits})


async def _reflect_batch(
    batch: EvidenceBatch, *, success: bool, index: int, text: str, runtime: ForestRuntime, directory: Path
) -> Patch:
    """Extract bounded edits from one public evidence batch."""
    config = runtime.config
    name = "success" if success else "failure"
    context = SkillForestReflectionContext(
        skill=text,
        success=success,
        edit_budget=config.reflection_edit_budget,
        trajectories=tuple(
            SkillForestTrajectoryContext(task=sample.task, result=result, index=i)
            for i, (sample, result) in enumerate(batch, 1)
        ),
    )
    logger.info("[skill_forest.reflection] started")
    response = await runtime.chat(
        messages=runtime.env_adapter.build_reflection_messages(context),
        directory=directory / "reflection" / name / f"{index:06d}",
    )
    try:
        patch = _parse_reference(response, reflection=True, source=name, edit_budget=config.reflection_edit_budget)
        logger.info(f"[skill_forest.reflection] progress | extracted_items={len(patch.edits)}")
        return patch
    except InvalidPatchError:
        logger.warning("[skill_forest.reflection] skipped | reason=invalid_reference_response")
        return Patch()


async def _merge_group(
    patches: Sequence[Patch], *, name: str, level: int, index: int, text: str, runtime: ForestRuntime, directory: Path
) -> Patch:
    """Resolve overlapping edits within one bounded reduction group."""
    if len(patches) == 1:
        logger.info(
            f"[skill_forest.patch_merge] skipped | reason=single_patch | patch_edits={len(patches[0].edits)}"
            f" | group_id={str(directory)}"
        )
        return patches[0]
    logger.info("[skill_forest.patch_merge] started")
    response = await runtime.chat(
        messages=runtime.env_adapter.build_patch_merge_messages(
            SkillForestPatchMergeContext(skill=text, patches=tuple(patches), success=name == "success", level=level)
        ),
        directory=directory / name / f"{level:03d}" / f"{index:06d}",
    )
    try:
        patch = _parse_reference(response, merge_level=level)
    except InvalidPatchError:
        patch = _concatenate_reference(patches, reasoning="fallback concatenation", level=level)
    patch = patch.model_copy(update={"edits": [edit.model_copy(update={"merge_level": level}) for edit in patch.edits]})
    logger.info(f"[skill_forest.patch_merge] progress | patch_edits={len(patch.edits)}")
    return patch


async def _reduce_evidence(
    batches: Sequence[EvidenceBatch], *, success: bool, text: str, runtime: ForestRuntime, directory: Path
) -> Patch:
    """Reduce a single outcome stream without owning tree or candidate state."""
    config = runtime.config
    name = "success" if success else "failure"
    if not batches:
        logger.info(
            f"[skill_forest.reflection] skipped | reason=empty_evidence_stream | source={name}"
            f" | extracted_items=0 | group_id={str(directory)}"
        )
    patches = await parallel(
        _reflect_batch(batch, success=success, index=index, text=text, runtime=runtime, directory=directory)
        for index, batch in enumerate(batches)
    )
    patches = [patch for patch in patches if patch.edits]
    level = 0
    while len(patches) > 1:
        level += 1
        groups = [
            patches[start : start + config.merge_batch_size]
            for start in range(0, len(patches), config.merge_batch_size)
        ]
        patches = await parallel(
            _merge_group(batch, name=name, level=level, index=index, text=text, runtime=runtime, directory=directory)
            for index, batch in enumerate(groups)
        )
    return patches[0] if patches else Patch()


async def _rank_patch(patch: Patch, *, text: str, depth: int, runtime: ForestRuntime, directory: Path) -> Patch:
    """Select a bounded edit pool through the shared optimizer transport."""
    config = runtime.config
    budget = config.ranking_edit_budgets[min(depth, len(config.ranking_edit_budgets) - 1)]
    if not config.ranking_enabled or len(patch.edits) <= budget:
        logger.info(
            "[skill_forest.ranking] skipped"
            f" | reason={'disabled' if not config.ranking_enabled else 'within_budget'}"
            f" | patch_edits={len(patch.edits)} | edit_budget={budget} | group_id={str(directory)}"
        )
        return patch
    logger.info("[skill_forest.ranking] started")
    response = await runtime.chat(
        messages=runtime.env_adapter.build_ranking_messages(
            SkillForestRankingContext(skill=text, patch=patch, edit_budget=budget)
        ),
        directory=directory / "ranking",
    )
    try:
        ranking = extract_json_object(response)
        supplied = ranking.get("selected_indices", [])
        if not isinstance(supplied, list):
            supplied = []
        indices = list(
            dict.fromkeys(index for index in supplied if isinstance(index, int) and 0 <= index < len(patch.edits))
        )[:budget]
    except ValueError:
        indices = []
    raw = deepcopy(patch.optimizer_input())
    if indices:
        raw["reasoning"] = patch.reasoning + f" [optimizer-ranked: selected {len(indices)}/{len(patch.edits)} edits]"
        raw["ranking_details"] = ranking
    else:
        indices = list(range(min(budget, len(patch.edits))))
        raw["reasoning"] = patch.reasoning + f" [fallback truncated {len(patch.edits)}->{budget} edits]"
        logger.warning("[skill_forest.ranking] fallback | reason=invalid_reference_response")
    raw["edits"] = [raw["edits"][index] for index in indices]
    return Patch.from_reference(raw)


async def generate_patch(
    *,
    skill: Skill,
    samples: Sequence[DataSample],
    results: Sequence[RolloutResult],
    runtime: ForestRuntime,
    directory: Path,
    seed: int,
    depth: int = 0,
) -> Patch:
    """Derive a candidate patch from shared parent trajectories.

    Args:
        skill: Unchanged parent Skill.
        samples: Tree-local public tasks, ordered like results.
        results: One collected rollout per task, reused across candidate repeats.
        runtime: Forest-wide bounded model transport.
        directory: Unique candidate-generation artifact directory.
        seed: Deterministic candidate-local seed for failure/success minibatch shuffling.
        depth: Nonnegative parent depth used to select the ranking edit budget.

    Returns:
        One combined failure/success patch, possibly empty.
    """
    if depth < 0:
        raise ValueError("Parent depth must be nonnegative")
    if len(samples) != len(results):
        raise ValueError("Reflection requires one result for every sample")
    text = skill.content
    config = runtime.config
    streams = []
    for success in (False, True):
        stream = [
            (sample, result)
            for sample, result in zip(samples, results, strict=True)
            if result.reward.success == success
        ]
        random.Random(seed + int(success)).shuffle(stream)
        streams.append(
            [
                stream[start : start + config.reflection_minibatch_size]
                for start in range(0, len(stream), config.reflection_minibatch_size)
            ]
        )

    failure, success = await parallel(
        _reduce_evidence(stream, success=outcome, text=text, runtime=runtime, directory=directory)
        for stream, outcome in zip(streams, (False, True), strict=True)
    )
    if failure.edits and success.edits:
        logger.info("[skill_forest.final_patch_merge] started")
        response = await runtime.chat(
            messages=runtime.env_adapter.build_final_patch_messages(
                SkillForestFinalPatchContext(skill=text, failure=failure, success=success)
            ),
            directory=directory / "final_merge",
        )
        try:
            patch = _parse_reference(response)
        except InvalidPatchError:
            patch = _concatenate_reference([failure, success], reasoning="fallback: failure first, then success")
        logger.info(f"[skill_forest.final_patch_merge] progress | patch_edits={len(patch.edits)}")
    else:
        patch = failure if failure.edits else success
    patch = await _rank_patch(patch, text=text, depth=depth, runtime=runtime, directory=directory)
    return patch
