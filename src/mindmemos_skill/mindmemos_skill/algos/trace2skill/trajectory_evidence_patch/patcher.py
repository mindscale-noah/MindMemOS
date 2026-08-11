"""Patch proposal, deterministic application, and optional format repair."""

from __future__ import annotations

from .config import TrajectoryEvidencePatchConfig
from .editor import apply_patch_ops, format_numbered
from .models import TrajectorySummary
from .prompts import (
    APPLY_PATCH_SYSTEM,
    PROPOSE_PATCH_ANNOTATED_SYSTEM,
    PROPOSE_PATCH_SYSTEM,
    REWRITE_SKILL_SYSTEM,
    apply_patch_user,
    propose_patch_user,
    rewrite_skill_user,
)
from .summarizer import ChatModel


class TrajectoryEvidencePatcher:
    """Generate and apply one minimal Skill patch from trajectory summaries."""

    def __init__(self, *, chat_model: ChatModel, config: TrajectoryEvidencePatchConfig) -> None:
        self._chat_model = chat_model
        self._config = config

    async def patch(
        self,
        *,
        skill_name: str,
        skill_md: str,
        summaries: list[TrajectorySummary],
    ) -> tuple[str, str]:
        annotated = any(
            item.score is not None or item.annotation_detail or item.annotation_metadata for item in summaries
        )
        proposal = await self._chat_model.chat(
            task=self._config.patch_task,
            messages=[
                {
                    "role": "system",
                    "content": PROPOSE_PATCH_ANNOTATED_SYSTEM if annotated else PROPOSE_PATCH_SYSTEM,
                },
                {
                    "role": "user",
                    "content": propose_patch_user(skill_name, skill_md, summaries),
                },
            ],
        )
        patch_plan = (proposal.content or "").strip()
        applied = await self._chat_model.chat(
            task=self._config.apply_task,
            messages=[
                {"role": "system", "content": APPLY_PATCH_SYSTEM},
                {
                    "role": "user",
                    "content": apply_patch_user(skill_md, patch_plan, format_numbered(skill_md)),
                },
            ],
            format_parser=lambda content: apply_patch_ops(skill_md, content),
            feedback_on_parse_error=True,
        )
        candidate = applied.parsed if applied.parsed is not None else apply_patch_ops(skill_md, applied.content or "")
        if self._config.rewrite_skill:
            rewritten = await self._chat_model.chat(
                task=self._config.rewrite_task,
                messages=[
                    {"role": "system", "content": REWRITE_SKILL_SYSTEM},
                    {"role": "user", "content": rewrite_skill_user(candidate)},
                ],
            )
            candidate = rewritten.content or ""
        return patch_plan, candidate


__all__ = ["TrajectoryEvidencePatcher"]
