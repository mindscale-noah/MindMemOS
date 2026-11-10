"""Scene transition semantics retained by graph representation construction."""

import pytest
from mindmemos_skill.algos.manage.skill_manage.representation import scene_flow_edges
from mindmemos_skill.scene.typing import SceneInput


@pytest.mark.parametrize(
    ("conditions", "expected"),
    [
        (["ready"], "ready"),
        (["ready", "approved"], "(ready) OR (approved)"),
        (["ready", "ready", "approved"], "(ready) OR (approved)"),
        (["ready", ""], ""),
        (["", "approved"], ""),
    ],
)
def test_scene_flow_merges_conditions_and_excludes_terminal_and_self_targets(conditions, expected):
    """Retain target order, OR semantics, and unconditional transition precedence."""
    scene = SceneInput(
        name="Read input",
        goal="Prepare a report",
        scene_id="read",
        scene_type="processing",
        entry_logic_step_id="step-1",
        contained_logic_steps=("step-1",),
        next_scene_rules=[
            {"target": "END_DONE", "condition": "finished"},
            {"target": "read", "condition": "repeat"},
            *({"target": "report", "condition": condition} for condition in conditions),
            {"target": "review", "condition": "needs review"},
        ],
    )
    node_ids = {"read": "node-read", "report": "node-report", "review": "node-review"}
    before = scene.model_dump()
    assert scene_flow_edges(scene, node_ids, weight=0.65) == [
        {"source": "node-read", "target": "node-report", "type": "flow", "weight": 0.65, "condition": expected},
        {"source": "node-read", "target": "node-review", "type": "flow", "weight": 0.65, "condition": "needs review"},
    ]
    assert scene.model_dump() == before
