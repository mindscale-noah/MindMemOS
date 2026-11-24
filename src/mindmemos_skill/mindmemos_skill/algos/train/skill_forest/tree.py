"""Stateless splitting, patch application, pruning and selection primitives."""

import math
import random
import re
from collections import Counter
from collections.abc import Sequence

from ....skill import Skill
from .typing import Edit, Node, Patch, TreeSplit


def split_tasks(task_ids: Sequence[str], *, train_ratio: float, seed: int) -> TreeSplit:
    """Partition a pool without replacement or additional bootstrap sampling.

    Args:
        task_ids: Unique task IDs in the released training pool.
        train_ratio: Fraction used for reflection.
        seed: Tree-local random seed.

    Returns:
        Ordered, disjoint train and validation memberships covering the pool.
    """
    ids = list(task_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Training task IDs must be unique")
    count = math.floor(len(ids) * train_ratio)
    if not 0 < count < len(ids):
        raise ValueError("Tree partitions must both be nonempty")
    random.Random(seed).shuffle(ids)
    return TreeSplit(train_ids=tuple(ids[:count]), validation_ids=tuple(ids[count:]))


def _apply_reference_edit(text: str, edit: Edit) -> str:
    """Apply the archived SkillOpt line edits, including protected tail sections."""
    markers = (
        ("<!-- SLOW_UPDATE_START -->", "<!-- SLOW_UPDATE_END -->"),
        ("<!-- APPENDIX_START -->", "<!-- APPENDIX_END -->"),
    )
    content = edit.content.strip()
    protected_starts = []
    target_index = text.find(edit.target) if edit.target else -1
    for start_marker, end_marker in markers:
        content = content.replace(start_marker, "").replace(end_marker, "")
        start, end = text.find(start_marker), text.find(end_marker)
        if start >= 0:
            protected_starts.append(start)
        if start >= 0 and end >= 0 and start <= target_index < end + len(end_marker):
            return text
    if edit.op == "append" or (edit.op == "insert_after" and target_index < 0):
        if protected_starts:
            position = min(protected_starts)
            return text[:position].rstrip() + "\n\n" + content + "\n\n" + text[position:]
        return text.rstrip() + "\n\n" + content + "\n"
    if edit.op == "insert_after":
        end = text.find("\n", target_index + len(edit.target))
        position = end + 1 if end != -1 else len(text)
        return text[:position] + "\n" + content + "\n" + text[position:]
    if target_index >= 0:
        return text.replace(edit.target, "" if edit.op == "delete" else content, 1)
    return text


def apply_patch(skill: Skill, patch: Patch, *, reference: bool = False) -> Skill:
    """Apply a complete patch atomically while preserving other files and resources.

    Args:
        skill: Parent Skill, never mutated.
        patch: Sequential exact edits to SKILL.md.
        reference: Preserve historical append, line insertion and missing-target semantics.

    Returns:
        A separate Skill containing the resulting entrypoint.

    Raises:
        ValueError: A target is missing or ambiguous, or the result is empty.
    """
    text = skill.content
    for edit in patch.edits:
        if reference:
            text = _apply_reference_edit(text, edit)
            continue
        if edit.op == "append":
            text = text.rstrip() + "\n\n" + edit.content
            continue
        if text.count(edit.target) != 1:
            raise ValueError("Patch target must occur exactly once")
        replacement = edit.content
        if edit.op == "insert_after":
            replacement = edit.target + "\n" + edit.content
        elif edit.op == "delete":
            replacement = ""
        text = text.replace(edit.target, replacement, 1)
    if not text.strip():
        raise ValueError("A patch cannot erase the complete Skill")
    return skill.with_content(text)


def mean_score(scores: dict[str, float]) -> float:
    """Average a complete score vector.

    Args:
        scores: Nonempty task-level scores.

    Returns:
        Arithmetic mean with its actual denominator.
    """
    if not scores:
        raise ValueError("Cannot score an empty evaluation")
    return math.fsum(scores.values()) / len(scores)


def cannot_improve(scores: dict[str, float], parent_scores: dict[str, float], min_gain: float) -> bool:
    """Compute the exact optimistic bound for unfinished validation.

    Args:
        scores: Observed candidate scores in [0, 1].
        parent_scores: Complete parent vector on the same validation tasks.
        min_gain: Required strict improvement.

    Returns:
        Whether even perfect remaining scores cannot pass the gate.
    """
    if not scores.keys() <= parent_scores.keys() or not parent_scores:
        raise ValueError("Candidate scores must be a subset of the parent vector")
    if any(not 0 <= value <= 1 for value in (*scores.values(), *parent_scores.values())):
        raise ValueError("Exact rejection requires scores in [0, 1]")
    maximum = (math.fsum(scores.values()) + len(parent_scores) - len(scores)) / len(parent_scores)
    return maximum - mean_score(parent_scores) <= min_gain


def expansion_wave(nodes: Sequence[Node], *, limit: int, max_depth: int) -> list[Node]:
    """Choose global best-first parents without modifying the supplied frontier.

    Args:
        nodes: Accepted nodes not yet expanded.
        limit: Available slots after applying the remaining expansion budget.
        max_depth: Nodes at this depth cannot be expanded.

    Returns:
        Highest parent-relative gains first, with stable ties.
    """
    return sorted(
        (node for node in nodes if node.depth < max_depth), key=lambda node: (-node.gain, node.tree_id, node.id)
    )[:limit]


def tree_winners(nodes: Sequence[Node]) -> list[Node]:
    """Freeze one highest-scoring positive-gain non-root node per tree.

    Args:
        nodes: Accepted nodes in deterministic generation order.

    Returns:
        Winners ordered by tree ID; trees with no eligible node are excluded.
    """
    winners: dict[int, Node] = {}
    for node in nodes:
        if node.parent_id is None or node.gain <= 0:
            continue
        previous = winners.get(node.tree_id)
        if previous is None or (-mean_score(node.scores), node.depth, len(node.skill.content), node.id) < (
            -mean_score(previous.scores),
            previous.depth,
            len(previous.skill.content),
            previous.id,
        ):
            winners[node.tree_id] = node
    return [winners[key] for key in sorted(winners)]


def novelty(initial: str, candidate: str) -> float:
    """Compute the paper's two-document TF-IDF cosine distance.

    Args:
        initial: Initial Skill text.
        candidate: Winner text compared with the initial Skill.

    Returns:
        Lexical cosine distance; identical empty vocabularies have distance zero.
    """
    counts = [Counter(re.findall(r"(?u)\b\w\w+\b", text.lower())) for text in (initial, candidate)]
    if not all(counts):
        return 0.0 if counts[0] == counts[1] else 1.0
    vocabulary = counts[0].keys() | counts[1].keys()
    idf = {word: 1 + math.log(3 / (1 + sum(word in count for count in counts))) for word in vocabulary}
    vectors = [{word: count[word] * idf[word] for word in count} for count in counts]
    norms = [math.sqrt(math.fsum(value * value for value in vector.values())) for vector in vectors]
    similarity = math.fsum(value * vectors[1].get(word, 0) for word, value in vectors[0].items())
    return max(0.0, min(1.0, 1 - similarity / (norms[0] * norms[1])))
