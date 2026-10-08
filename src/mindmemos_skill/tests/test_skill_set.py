"""Verify collection selection and bounded task-time assembly."""

import json
from contextvars import ContextVar
from threading import Barrier, Lock

import pytest
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.skill import Skill, SkillBundle, SkillSet
from pydantic import BaseModel


def test_parallel_assembly_bounds_order_context_and_source_isolation():
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    peak = 0
    context = ContextVar("assembly_test_context", default="missing")

    class ParallelSkill(Skill):
        def assemble(self, task):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait(timeout=5)
                assert context.get() == "caller"
                return self.with_bundle(SkillBundle(files={"SKILL.md": self.bundle.files["SKILL.md"] + task.id}))
            finally:
                with lock:
                    active -= 1

    sources = {str(i): ParallelSkill(bundle=SkillBundle(files={"SKILL.md": str(i)})) for i in range(4)}
    collection = SkillSet(sources, max_concurrency=2)
    token = context.set("caller")
    try:
        result = collection.assemble(Task(id="task", instruction="Public task"))
    finally:
        context.reset(token)
    assert peak == 2
    assert [skill.bundle.files["SKILL.md"] for skill in result] == [f"{i}task" for i in range(4)]
    assert [skill.bundle.files["SKILL.md"] for skill in sources.values()] == [str(i) for i in range(4)]


def test_dynamic_selection_assembles_only_selected_members_in_order():
    calls = []

    class RecordingSkill(Skill):
        def assemble(self, task):
            calls.append(self.bundle.files["SKILL.md"])
            return super().assemble(task)

    class RecommendedSet(SkillSet):
        def select(self, task):
            return ["c", "a"] if task.id == "selected" else []

    collection = RecommendedSet(
        {key: RecordingSkill(bundle=SkillBundle(files={"SKILL.md": key})) for key in "abc"},
        max_concurrency=2,
    )
    result = collection.assemble(Task(id="selected", instruction="Choose members"))
    assert [skill.bundle.files["SKILL.md"] for skill in result] == ["c", "a"]
    assert sorted(calls) == ["a", "c"]
    assert collection.assemble(Task(id="empty", instruction="No members")) == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_concurrency(limit):
    with pytest.raises(ValueError, match="positive integer"):
        SkillSet({}, max_concurrency=limit)


@pytest.mark.parametrize("selection", [["missing"], ["a", "a"]])
def test_invalid_selection_rejected_before_assembly(selection):
    class InvalidSet(SkillSet):
        def select(self, task):
            return selection

    collection = InvalidSet({"a": Skill(bundle=SkillBundle(files={"SKILL.md": "a"}))}, max_concurrency=1)
    with pytest.raises(ValueError, match="unique IDs"):
        collection.assemble(Task(id="task", instruction="Task"))


def test_member_failure_propagates():
    class FailedSkill(Skill):
        def assemble(self, task):
            raise RuntimeError("Assembly failed")

    collection = SkillSet({"a": FailedSkill(bundle=SkillBundle(files={"SKILL.md": "a"}))}, max_concurrency=1)
    with pytest.raises(RuntimeError, match="Assembly failed"):
        collection.assemble(Task(id="task", instruction="Task"))


class CollectionOptions(BaseModel):
    """Settings for the collection test environment."""


class CollectionEnvironment(Environment[CollectionOptions]):
    """Expose ordered assembled content without external execution."""

    config_type = CollectionOptions
    fail = False
    calls = 0

    def _run(self, *, task, reward_data, skills, workspace):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Execution failed")
        return RolloutResult(
            output=[skill.bundle.files["SKILL.md"] for skill in skills],
            trajectory=[],
            reward=Reward(score=1, success=True, detail="Assembled"),
        )


class SelectingSet(SkillSet):
    """Count selections while exposing explicit policy configuration."""

    calls = 0
    fail = False

    def __init__(self, skills, *, selected, max_concurrency=2):
        super().__init__(skills, max_concurrency=max_concurrency)
        self.selected = selected

    def select(self, task):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Selection failed")
        return list(self.selected)

    def config_snapshot(self):
        return {"selected": self.selected}


def source_members():
    return {key: Skill(bundle=SkillBundle(files={"SKILL.md": key})) for key in "abc"}


@pytest.mark.parametrize("selection", [["c", "a"], []])
def test_resume_reuses_selection_and_assembly_after_execution_failure(tmp_path, selection):
    collection = SelectingSet(source_members(), selected=selection)
    env = CollectionEnvironment(CollectionOptions())
    task = Task(id="task", instruction="Public task")
    env.fail = True
    with pytest.raises(RuntimeError, match="Execution failed"):
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)
    assert collection.calls == 1
    collection.fail = True
    env.fail = False
    result = env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)
    assert result.output == selection
    assert env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection) == result
    assert collection.calls == 1
    assert env.calls == 2
    saved = json.loads((tmp_path / "audit/request.json").read_text())
    assert saved["selected_skill_ids"] == selection


@pytest.mark.parametrize("change", ["policy", "order", "concurrency", "source", "type"])
def test_resume_rejects_changed_collection_inputs_before_selection(tmp_path, change):
    sources = source_members()
    collection = SelectingSet(sources, selected=["c", "a"])
    env = CollectionEnvironment(CollectionOptions())
    task = Task(id="task", instruction="Public task")
    env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)
    if change == "policy":
        collection.selected = ["a"]
    elif change == "order":
        collection = SelectingSet(dict(reversed(list(sources.items()))), selected=["c", "a"])
    elif change == "concurrency":
        collection = SelectingSet(sources, selected=["c", "a"], max_concurrency=1)
    elif change == "source":
        sources["b"] = Skill(bundle=SkillBundle(files={"SKILL.md": "changed"}))
        collection = SelectingSet(sources, selected=["c", "a"])
    else:
        collection = SkillSet(sources, max_concurrency=2)
    with pytest.raises(ValueError, match="inputs changed"):
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)


def test_forced_rerun_clears_old_selection_before_failure(tmp_path):
    collection = SelectingSet(source_members(), selected=["c", "a"])
    env = CollectionEnvironment(CollectionOptions())
    task = Task(id="task", instruction="Public task")
    env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)
    collection.fail = True
    with pytest.raises(RuntimeError, match="Selection failed"):
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection, resume=False)
    saved = json.loads((tmp_path / "audit/request.json").read_text())
    assert saved["selected_skill_ids"] is None
    assert saved["assembled_bundles"] is None
    assert not (tmp_path / "audit/result.json").exists()
    collection.fail = False
    env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)
    assert collection.calls == 3


@pytest.mark.parametrize("completed", [True, False])
def test_legacy_static_record_reuses_existing_assembly(tmp_path, completed):
    class CountingSkill(Skill):
        calls = 0

        def assemble(self, task):
            self.calls += 1
            return super().assemble(task)

    skill = CountingSkill(bundle=SkillBundle(files={"SKILL.md": "source"}))
    collection = SkillSet({"source": skill}, max_concurrency=2)
    env = CollectionEnvironment(CollectionOptions())
    task = Task(id="task", instruction="Public task")
    result = env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)
    path = tmp_path / "audit/request.json"
    saved = json.loads(path.read_text())
    del saved["skill_set"], saved["selected_skill_ids"]
    path.write_text(json.dumps(saved))
    if not completed:
        (tmp_path / "audit/result.json").unlink()
    assert env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection).output == result.output
    assert skill.calls == 1


@pytest.mark.parametrize("ids", [["c", "c"], ["unknown"], ["a"], None])
def test_corrupt_selection_cannot_resume(tmp_path, ids):
    collection = SelectingSet(source_members(), selected=["c", "a"])
    env = CollectionEnvironment(CollectionOptions())
    task = Task(id="task", instruction="Public task")
    env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)
    path = tmp_path / "audit/request.json"
    saved = json.loads(path.read_text())
    saved["selected_skill_ids"] = ids
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=collection)
    assert collection.calls == 1


def test_rollout_rejects_removed_list_argument(tmp_path):
    env = CollectionEnvironment(CollectionOptions())
    with pytest.raises(TypeError, match="skills"):
        env.rollout(Task(id="task", instruction="Public task"), tmp_path, reward_data=RewardData(), **{"skills": []})
    assert not (tmp_path / "audit").exists()
