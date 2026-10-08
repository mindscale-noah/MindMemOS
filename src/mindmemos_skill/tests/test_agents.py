"""Validate shared agent outputs and concrete structured-output serialization."""

import json

import pytest
from mindmemos_skill.agents import AgentResult
from pydantic import BaseModel, ValidationError


def test_structured_output_preserves_model_and_fields():
    """Keep the parsed model instance and its fields when serializing a result."""

    class Answer(BaseModel):
        """Represent a parsed answer with its supporting explanation."""

        value: int
        """Validated numeric answer."""

        explanation: str
        """Explanation accompanying the answer."""

    answer = Answer(value=2, explanation="One plus one")
    result = AgentResult(output=answer)
    assert result.output is answer
    assert result.model_dump()["output"] == answer.model_dump()
    assert json.loads(result.model_dump_json())["output"] == answer.model_dump()


@pytest.mark.parametrize("output", [None, 1, 1.5, True, [], {"value": 2}])
def test_output_rejects_unparsed_values(output):
    """Reject raw JSON values that are neither text nor parsed model instances."""
    with pytest.raises(ValidationError):
        AgentResult(output=output)
