"""Small atomic node checkpoints for forest recovery."""

from pathlib import Path

from pydantic import JsonValue

from ....common.checkpoint import fingerprint
from ....common.json import read_json, write_json
from ....skill import Skill
from ....skill.typing import SkillSnapshot
from .typing import Node, NodeCheckpoint


def node_directory(root: Path, tree_id: int, node_id: str) -> Path:
    """Return the run-owned directory for a generated forest node."""
    if not node_id or "/" in node_id or "\\" in node_id or node_id in {".", ".."}:
        raise ValueError("Invalid forest node ID")
    return root / "trees" / f"tree-{tree_id}" / f"node-{node_id}"


def read_node(root: Path, tree_id: int, node_id: str) -> NodeCheckpoint | None:
    """Load one patch-applied node when its atomic checkpoint exists."""
    path = node_directory(root, tree_id, node_id) / "node.json"
    return read_node_file(path) if path.exists() else None


def read_node_file(path: Path) -> NodeCheckpoint:
    """Reject a changed or incomplete node checkpoint before recovery."""
    payload = read_json(path)
    digest = payload.pop("digest", None)
    if digest != fingerprint(payload):
        raise ValueError(f"Skill-Forest node checkpoint mismatch: {path}")
    checkpoint = NodeCheckpoint.model_validate(payload)
    if path.parent.name != f"node-{checkpoint.id}" or path.parent.parent.name != f"tree-{checkpoint.tree_id}":
        raise ValueError(f"Skill-Forest node checkpoint location mismatch: {path}")
    return checkpoint


def write_node(root: Path, checkpoint: NodeCheckpoint) -> None:
    """Atomically save one node without touching its rollout audit files."""
    path = node_directory(root, checkpoint.tree_id, checkpoint.id) / "node.json"
    payload = checkpoint.model_dump(mode="json")
    write_json(path, {**payload, "digest": fingerprint(payload)})


def load_skill(payload: SkillSnapshot | dict[str, JsonValue]) -> Skill:
    """Restore a complete Skill snapshot from a node or final result."""
    return Skill.from_snapshot(SkillSnapshot.model_validate(payload))


def load_node(checkpoint: NodeCheckpoint) -> Node:
    """Reconstruct an accepted node for frontier and winner selection."""
    if checkpoint.status not in {"root", "accepted"}:
        raise ValueError("Only validated nodes can enter the forest")
    return Node(
        id=checkpoint.id,
        tree_id=checkpoint.tree_id,
        parent_id=checkpoint.parent_id,
        depth=checkpoint.depth,
        skill=load_skill(checkpoint.skill),
        scores=checkpoint.scores,
        gain=checkpoint.gain,
    )
