# PersonaMem Profile Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make memory-RAG ingest one extractable PersonaMem profile per visible scope without changing conversation batching or session timestamps.

**Architecture:** Add one private profile selector beside the PersonaMem timestamp helpers. `_build_scope` will issue a dedicated profile add request with profile-only metadata, then submit existing non-system conversation batches unchanged.

**Tech Stack:** Python 3.13, asyncio, Pydantic, pytest, pytest-asyncio, Ruff

---

### Task 1: Lock profile-ingestion behavior with regression tests

**Files:**
- Create: `tests/mindmemos_eval/test_personamem_profile_ingestion.py`

- [ ] **Step 1: Write the failing repeated-profile test**

Create recording fakes for the memory client and context store. Build a scope containing the same `Current user persona:` at two session boundaries plus three conversation messages. Assert:

```python
assert len(memory.add_calls) == 2
assert memory.add_calls[0]["messages"] == [{
    "role": "user",
    "content": "Current user persona: Likes quiet outdoor activities.",
    "timestamp": _PERSONAMEM_EPOCH_MS,
}]
assert memory.add_calls[0]["metadata"]["source"] == "personamem_persona"
assert memory.add_calls[0]["metadata"]["content_type"] == "profile"
assert [message["content"] for message in memory.add_calls[1]["messages"]] == ["u1", "a1", "u2"]
assert "source" not in memory.add_calls[1]["metadata"]
assert summary.total_messages == 4
assert summary.added_messages == 4
assert summary.add_calls == 2
```

- [ ] **Step 2: Write the no-profile compatibility test**

Use a non-profile system marker followed by one user message. Assert that only the user message is submitted, ordinary metadata has no profile keys, and the summary is one message/one call.

- [ ] **Step 3: Run the new tests and verify RED**

Run: `uv run pytest -q tests/mindmemos_eval/test_personamem_profile_ingestion.py`

Expected: the repeated-profile test fails because current `_build_scope` filters every system message and performs only the conversation add request.

### Task 2: Add isolated profile selection and ingestion

**Files:**
- Modify: `src/mindmemos_eval/mindmemos_eval/memory/envs/personamem/env.py:500-630`
- Test: `tests/mindmemos_eval/test_personamem_profile_ingestion.py`

- [ ] **Step 1: Add a profile selector**

```python
_PERSONAMEM_PROFILE_PREFIX = "current user persona:"


def _first_visible_personamem_profile(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[int, str] | None:
    for index, message in enumerate(messages):
        if str(message.get("role") or "").strip().lower() != "system":
            continue
        content = str(message.get("content") or "").strip()
        if content.lower().startswith(_PERSONAMEM_PROFILE_PREFIX):
            return index, content
    return None
```

This deliberately selects only the first matching visible profile and ignores unrelated system markers.

- [ ] **Step 2: Submit the profile separately before conversation batches**

Create shared scope metadata once. Before the conversation loop submit:

```python
profile = _first_visible_personamem_profile(visible)
if profile is not None:
    profile_index, profile_content = profile
    await self._memory.add(
        [{
            "role": "user",
            "content": profile_content,
            "timestamp": ts_map.get(profile_index, _PERSONAMEM_EPOCH_MS),
        }],
        user_id=scope.user_id,
        session_id=scope.session_id,
        mode="sync",
        metadata={
            **scope_metadata,
            "source": "personamem_persona",
            "content_type": "profile",
        },
    )
    add_calls += 1
    added_messages += 1
```

Set `total_messages` to `len(messages) + int(profile is not None)`. Keep conversation metadata equal to `scope_metadata` so profile labels never leak into conversation batches.

- [ ] **Step 3: Run the new tests and verify GREEN**

Run: `uv run pytest -q tests/mindmemos_eval/test_personamem_profile_ingestion.py`

Expected: all tests pass.

- [ ] **Step 4: Run PersonaMem regression tests**

Run:

```bash
uv run pytest -q tests/mindmemos_eval/test_personamem_profile_ingestion.py tests/mindmemos_eval/test_personamem_answer_handling.py tests/mindmemos_eval/test_personamem_memory_date.py tests/mindmemos_eval/test_personamem_prompt.py tests/mindmemos_eval/test_personamem_timestamp.py tests/mindmemos_eval/test_personamem_evo.py tests/mindmemos_eval/test_memory_runner.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Run static verification**

Run:

```bash
uv run ruff check src/mindmemos_eval/mindmemos_eval/memory/envs/personamem/env.py tests/mindmemos_eval/test_personamem_profile_ingestion.py
git diff --check
```

Expected: Ruff and diff checks pass.

- [ ] **Step 6: Commit the implementation**

Stage only the PersonaMem environment and profile-ingestion test, then create a Lore-format commit recording the non-extractable-system constraint and verification evidence. Do not stage `config/mindmemos/api_keys.yaml`, `reports/`, or unrelated existing edits.
