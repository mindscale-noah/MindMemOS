from __future__ import annotations

from mindmemos.api.mappers import to_feedback_pipeline_input
from mindmemos.api.schemas import FeedbackRequest
from mindmemos.typing.memory import DialogueMessage


def test_feedback_request_maps_explicit_context_fields() -> None:
    request = FeedbackRequest(
        feedback="please keep summaries concise",
        mode="async",
        messages=[DialogueMessage(role="user", content="summarize it")],
        recalled_memories=[
            {
                "id": "m1",
                "memory": "User prefers concise summaries.",
            }
        ],
    )

    inp = to_feedback_pipeline_input(request)

    assert inp.feedback == "please keep summaries concise"
    assert inp.mode == "async"
    assert inp.messages[0].content == "summarize it"
    assert inp.recalled_memories[0].id == "m1"
    assert inp.recalled_memories[0].memory == "User prefers concise summaries."
    assert inp.recalled_memories[0].last_update_at is None
