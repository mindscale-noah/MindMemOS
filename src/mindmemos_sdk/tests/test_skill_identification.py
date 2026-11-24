"""Verify that trajectory bindings use observed loads and immutable evidence."""

import json
from unittest.mock import Mock

import pytest
from mindmemos_sdk.errors import MindMemOSSDKError
from mindmemos_sdk.skills.client import SkillClient
from mindmemos_sdk.skills.content import content_hash
from mindmemos_sdk.skills.identification import identify_loaded_versions
from mindmemos_sdk.skills.typing import SkillReadEvidence, SkillShellEvidence

OLD_SOURCE = "---\nname: demo\ndescription: Example\n---\nOld instructions.\n"
NEW_SOURCE = OLD_SOURCE.replace("Old", "New")


@pytest.fixture
def client():
    """Model a family activated to a new version after the host loaded the old one."""
    versions = [
        {
            "skill_id": "sk-demo",
            "skill_version_id": version_id,
            "content_hash": content_hash(source),
            "skill_hash": bundle_hash,
        }
        for version_id, source, bundle_hash in [
            ("skv-old", OLD_SOURCE, "a" * 64),
            ("skv-current", NEW_SOURCE, "b" * 64),
        ]
    ]
    fake = Mock(spec=SkillClient)

    def resolve_skill(request):
        if request.skill_ref in {"demo", "sk-demo"}:
            return versions[1]
        raise MindMemOSSDKError("Unknown reference")

    def resolve_version(request):
        filters = request.model_dump(exclude_none=True)
        matches = [item for item in versions if all(item.get(key) == value for key, value in filters.items())]
        if len(matches) != 1:
            raise MindMemOSSDKError("Missing or ambiguous version")
        return matches[0]

    fake.skill_resolve.side_effect = resolve_skill
    fake.version_resolve.side_effect = resolve_version
    fake.list.return_value = {"items": []}
    return fake


@pytest.mark.parametrize("reference", ["demo", "sk-demo", "$REF"])
@pytest.mark.parametrize(
    "output",
    ["Skill source displayed.", '{"version": {}}', '{"version": {"content": ""}}', '{"content_hash": "truncated"}'],
)
def test_incomplete_show_does_not_bind_the_newly_active_version(client, reference, output):
    evidence = SkillShellEvidence(kind="shell", command=f'mindmemos skill show "{reference}"', content=output)

    assert identify_loaded_versions(client, [evidence]) == []
    assert client.mock_calls == []


@pytest.mark.parametrize(
    "command",
    [
        "mindmemos skill commit demo --from source",
        "mindmemos skill register source",
        "mindmemos skill activate demo --version-id skv-old",
        "mindmemos skill resolve demo",
        "mindmemos skill graph recommend task",
        "mindmemos skill show demo --help",
        "wrapper mindmemos skill show demo",
        "echo mindmemos skill show demo",
    ],
)
def test_non_load_commands_cannot_supply_version_evidence(client, command):
    output = json.dumps({"version": {"skill_version_id": "skv-old", "content": OLD_SOURCE}})
    evidence = SkillShellEvidence(kind="shell", command=command, content=f"[stdout]\n{output}\n[done]")

    assert identify_loaded_versions(client, [evidence]) == []
    assert client.mock_calls == []


@pytest.mark.parametrize(
    ("reference", "output"),
    [
        ("skv-old", "Skill source displayed."),
        ("demo", "Loaded version skv-old"),
        ("demo", json.dumps({"version": {"skill_version_id": "skv-old"}})),
        ("demo", json.dumps({"version": {"content": OLD_SOURCE}})),
        ("demo", OLD_SOURCE),
        ("demo", json.dumps({"version": {"content_hash": content_hash(OLD_SOURCE)}})),
        ("demo", json.dumps({"version": {"skill_hash": "a" * 64}})),
    ],
)
def test_immutable_show_evidence_preserves_the_loaded_historical_version(client, reference, output):
    evidence = SkillShellEvidence(kind="shell", command=f"mindmemos skill show {reference}", content=output)

    assert identify_loaded_versions(client, [evidence]) == ["skv-old"]


@pytest.mark.parametrize("version_id", ["skv-old", "skv-missing"])
def test_failed_exact_evidence_does_not_fall_back_to_the_current_version(client, version_id):
    evidence = SkillShellEvidence(
        kind="shell",
        command="mindmemos skill show demo",
        content=json.dumps({"version": {"skill_version_id": version_id, "content": NEW_SOURCE}}),
    )

    assert identify_loaded_versions(client, [evidence]) == []
    assert all(call.args[0].skill_ref == version_id for call in client.skill_resolve.call_args_list)


def test_recognized_chained_shows_keep_exact_versions_in_load_order(client):
    output = "\n".join(json.dumps({"version": {"skill_version_id": version}}) for version in ["skv-old", "skv-current"])
    evidence = SkillShellEvidence(
        kind="shell", command="mindmemos skill show demo && mindmemos skill show demo", content=output
    )

    assert identify_loaded_versions(client, [evidence]) == ["skv-old", "skv-current"]


def test_export_requires_a_matching_read_and_deduplicates_loads(client):
    exported = SkillShellEvidence(
        kind="shell",
        command="mindmemos skill export demo --to /skills/demo",
        content=json.dumps({"directory": "/skills/demo", "skill_version_id": "skv-old"}),
    )
    read = SkillReadEvidence(kind="read", path="/skills/demo/SKILL.md", content=OLD_SOURCE)

    assert identify_loaded_versions(client, [exported]) == []
    assert identify_loaded_versions(client, [exported, read, read]) == ["skv-old"]
    changed = read.model_copy(update={"content": OLD_SOURCE.replace("Old", "Unregistered")})
    assert identify_loaded_versions(client, [exported, changed]) == []


def test_local_read_matches_historical_content_after_activation(client):
    read = SkillReadEvidence(kind="read", path="/skills/demo/SKILL.md", content=OLD_SOURCE)

    assert identify_loaded_versions(client, [read]) == ["skv-old"]
