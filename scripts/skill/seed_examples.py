"""Import the bundled examples into a local Skill repository without replacing existing aliases."""

import argparse
import asyncio
import json
from pathlib import Path

from mindmemos_skill.api import create_service
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.pipelines.typing import CommitRequest, HistoryRequest, RegisterRequest, ShowRequest
from mindmemos_skill.skill import Skill

EXAMPLES = Path(__file__).resolve().parents[2] / "resources/mindmemos_skill/examples"
"""Repository-owned example sources and their version manifest."""


async def seed_examples(home: Path | None = None) -> list[dict[str, str]]:
    """Register bundled examples, preserving existing Skills and resuming an incomplete bundled import.

    Args:
        home: Explicit local home; otherwise use the normal UI home resolver.

    Returns:
        Imported or skipped aliases and their central identities.
    """
    manifest = json.loads((EXAMPLES / "manifest.json").read_text())
    prepared = [(item, [Skill.from_path(EXAMPLES / path).snapshot() for path in item["versions"]]) for item in manifest]
    results = []
    async with await create_service(home=home) as service:
        for item, snapshots in prepared:
            provenance = f"bundled-example:{item['alias']}"
            try:
                detail = await service.show_skill(ShowRequest(skill_ref=item["alias"]))
            except SkillNotFoundError:
                detail = await service.register_skill(
                    RegisterRequest(
                        name=item["name"],
                        alias=item["alias"],
                        source=snapshots[0],
                        registration_source=provenance,
                        message="Bundled example: version 1",
                    )
                )
                completed, status = 1, "created"
            else:
                history = await service.skill_history(HistoryRequest(skill_ref=detail.skill.skill_id))
                resumable = (
                    detail.skill.registration_source == provenance
                    and 0 < len(history) < len(snapshots)
                    and detail.skill.current_version_id == history[-1].version_id
                    and all(
                        version.local_snapshot_hash == snapshot.local_snapshot_hash
                        for version, snapshot in zip(history, snapshots)
                    )
                )
                if not resumable:
                    results.append({"alias": item["alias"], "status": "skipped", "skill_id": detail.skill.skill_id})
                    continue
                completed, status = len(history), "resumed"
            for number, snapshot in enumerate(snapshots[completed:], start=completed + 1):
                detail = await service.commit_skill(
                    CommitRequest(
                        skill_ref=detail.skill.skill_id,
                        source=snapshot,
                        base_version_id=detail.skill.current_version_id,
                        message=f"Bundled example: version {number}; add boundary checks and a report template",
                    )
                )
            results.append({"alias": item["alias"], "status": status, "skill_id": detail.skill.skill_id})
    return results


async def _main() -> None:
    """Import examples into the selected home and print a compact result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, help="Local home used by the UI")
    args = parser.parse_args()
    print(json.dumps(await seed_examples(args.home), ensure_ascii=False, indent=2))


def main() -> None:
    """Own the process event loop for the asynchronous entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
