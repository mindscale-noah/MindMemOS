"""Seed static and dynamic versions for real task assembly in an isolated workbench."""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from mindmemos_skill.agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.api import create_service
from mindmemos_skill.errors import SkillNotFoundError
from mindmemos_skill.pipelines.typing import CommitRequest, RegisterRequest, ShowRequest
from mindmemos_skill.skill import Skill, SkillBundle


async def _main() -> None:
    """Seed a new demo once; preserve existing versions on subsequent runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Router model used only when assembly is requested")
    parser.add_argument("--env-file", type=Path, help="Environment file containing the model endpoint and credential")
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("--env-file must be an existing file")
        load_dotenv(args.env_file, override=True)
    source_path = Path(__file__).resolve().parents[2] / "resources/ui/treeskill-demo"
    service = await create_service(home=args.home)
    try:
        try:
            result = await service.show_skill(ShowRequest(skill_ref="treeskill-ui-demo"))
        except SkillNotFoundError:
            source = Skill.from_path(source_path)
            result = await service.register_skill(
                RegisterRequest(
                    source=source.snapshot(),
                    name="Task Assembly Demo",
                    alias="treeskill-ui-demo",
                    message="Synthetic spreadsheet source; static baseline.",
                )
            )
            dynamic = TreeSkill(
                TreeSkillConfig(
                    router=OpenAIChatConfig(
                        model=args.model,
                        max_retries=0,
                        timeout_s=60,
                        max_completion_tokens=512,
                    )
                ),
                bundle=source.bundle,
                resources=source.resources,
            )
            result = await service.commit_skill(
                CommitRequest(
                    skill_ref=result.skill.skill_id,
                    source=dynamic.snapshot(),
                    message="Dynamic task assembly.",
                )
            )
            revised = dynamic.with_bundle(
                SkillBundle(
                    files={
                        "SKILL.md": source.bundle.files["SKILL.md"]
                        + (
                            "\n## Missing-month policy\n"
                            "Keep missing months explicitly marked as missing; never replace them with zero.\n"
                        ),
                    }
                )
            )
            result = await service.commit_skill(
                CommitRequest(
                    skill_ref=result.skill.skill_id,
                    source=revised.snapshot(),
                    message="Clarify missing-month handling.",
                )
            )
        print(f"Demo home: {args.home.resolve()}")
        print(f"Skill: {result.skill.skill_id}")
    finally:
        await service.close()


def main() -> None:
    """Own the process event loop for the asynchronous entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
