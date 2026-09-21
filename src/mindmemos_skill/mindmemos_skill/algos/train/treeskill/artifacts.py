"""Atomic experiment artifacts and reference text normalization."""

import difflib
import json
from pathlib import Path

from ....common.json import write_json
from ....skill import SkillBundle

REFERENCE_COMMIT = "e83aae887f08eb9fa1596384404dc7757a6a7f2f"
"""Pinned upstream revision used by the prompt and algorithm parity tests."""


def normalize_skill_text(content: str) -> str:
    """Apply the reference algorithm's text normalization.

    Args:
        content: Source or candidate Markdown.

    Returns:
        LF-delimited text with exactly one final newline.
    """
    return content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


def compute_skill_content_hash(blob: dict[str, str]) -> str:
    """Hash a bundle with the package's canonical serialization.

    Args:
        blob: Managed Skill files.

    Returns:
        SHA-256 content identity.
    """
    return SkillBundle(files=blob).content_hash


def record_patch(directory: Path, *, before: str, after: str, payload: dict) -> None:
    """Save a proposed edit, its disposition, full snapshots, and a unified diff.

    Args:
        directory: Unique directory for one node edit.
        before: Complete source before this edit.
        after: Complete source after acceptance or rejection.
        payload: Proposal, evidence and validation result.
    """
    write_json(directory / "patch.json", payload)
    (directory / "before.md").write_text(before, encoding="utf-8")
    (directory / "after.md").write_text(after, encoding="utf-8")
    (directory / "change.diff").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), fromfile="before.md", tofile="after.md"
            )
        ),
        encoding="utf-8",
    )


def experiment_config(trainer_config):
    """Serialize algorithm settings without credentials or artifact metadata.

    Args:
        trainer_config: Validated algorithm and provider settings.

    Returns:
        JSON-compatible algorithm configuration with API keys removed.
    """
    return {
        "algorithm": trainer_config.model_dump(
            mode="json", exclude={"optimizer": {"api_key"}, "router": {"api_key"}, "failure_optimizer": {"api_key"}}
        ),
    }


def model_usage(directory: Path) -> dict:
    """Aggregate recorded algorithm requests without double-counting parser retries.

    Args:
        directory: Root of the per-attempt model audit tree.

    Returns:
        Observed token totals, request count and availability information.
    """
    calls = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("**/attempt-*.json"))]
    return {
        "calls": len(calls),
        "failed_calls": sum(call["status"] == "failed" for call in calls),
        "usage_reported_calls": sum(bool(call.get("response", {}).get("usage")) for call in calls),
        **{
            key: sum(int(call.get("response", {}).get("usage", {}).get(key, 0)) for call in calls)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
    }
