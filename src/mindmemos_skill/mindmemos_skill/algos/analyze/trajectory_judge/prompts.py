"""Prompt construction for dynamic trajectory analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SuccessVerdict


def build_success_prompt(*, trace: str, trajectory_count: int, verifier_score: float | None) -> str:
    reference = ""
    if verifier_score is not None:
        reference = (
            f"\nAn official verifier score is {verifier_score}. Treat it as reference evidence, "
            "not as proof; explicitly say whether your assessment agrees.\n"
        )
    return f"""You are evaluating whether an AI agent completed a task end-to-end.

=== TRAJECTORIES ({trajectory_count}) ===
{trace}
=== END TRAJECTORIES ===
{reference}
Judge only what the recorded actions and outputs prove. Verify required processing, final artifacts, format, consistency, and explicit execution status. A claimed success without evidence is not success. A failed attempt can still provide useful evidence, but the overall result must reflect whether the requested task was completed.

Return only JSON:
{{"success": true, "confidence": "low", "reason": "one short paragraph", "evidence": ["specific trajectory or event evidence"]}}
"""


def build_attribution_prompt(
    *,
    task: str,
    segments: str,
    verifier_score: float | None,
    verdict: SuccessVerdict | None,
) -> str:
    reference = f"Official verifier score: {verifier_score}.\n" if verifier_score is not None else ""
    prior = ""
    if verdict is not None:
        prior = (
            f"\nAn LLM judge already assessed this trajectory: success={verdict.success}, "
            f'confidence={verdict.confidence}, reason="{verdict.reason[:300]}"\n'
        )
    return f"""You are attributing a failed AI-agent task to trajectory segments.

The trajectory is split into segments, one per skill invocation (or skill-content injection). A skill's "main" segment spans from its invocation until the next skill's invocation (or the end).

=== TASK ===
{task}
=== END TASK ===

=== SEGMENTED TRACE ===
{segments}
=== END SEGMENTED TRACE ===
{reference}{prior}
For EACH skill segment shown above, assess whether it caused or materially contributed to the failure. Consider:
- Did the skill output wrong or malformed results (wrong values, wrong format, missing key fields)?
- Did the agent mis-use the skill (wrong args, wrong domain, wrong interpretation of results)?
- Did the segment show errors, retries, or a suspiciously low amount of output?
- Was the segment merely informational and clearly not the failure point?

Rules:
- Use the exact segment labels shown above in "skill".
- suspicion ranges from 0 (clearly not the cause) to 1 (almost certainly the cause); use decimals.
- If no skill is clearly responsible (failure happened before any skill, or outside skill usage), set primary_cause.skill to null and explain.
- If the trajectory actually succeeded, still list the segments but with low suspicion and primary_cause.skill null.

Return only JSON:
{{
  "segments": [{{"skill": "exact segment label", "suspicion": 0.0, "reason": "...", "evidence": ["..."]}}],
  "primary_cause": {{"skill": null, "reason": "...", "confidence": "low"}}
}}
"""


__all__ = ["build_attribution_prompt", "build_success_prompt"]
