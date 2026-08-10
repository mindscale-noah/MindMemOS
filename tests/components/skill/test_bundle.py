import pytest
from mindmemos.components.skill import (
    compute_content_hash,
    deserialize_bundle,
    is_whitelisted,
    normalize_bundle,
    serialize_bundle,
)
from mindmemos.errors import SkillBundleError

SKILL_BODY = "---\nname: prd-writer\nversion: 1.2.0\n---\n\n# PRD Writer\n\nWrite product docs.\n"


def test_is_whitelisted_accepts_only_canonical_bundle_paths():
    assert is_whitelisted("SKILL.md")
    assert not is_whitelisted("scripts/check.py")
    assert not is_whitelisted("scripts/nested/helper.sh")
    assert not is_whitelisted("prd-writer/SKILL.md")
    assert not is_whitelisted("scripts/../secret.txt")
    assert not is_whitelisted("scripts\\check.py")
    assert not is_whitelisted("README.md")
    assert not is_whitelisted("prd-writer/reference.md")


def test_normalize_bundle_keeps_only_skill_md():
    normalized = normalize_bundle({"SKILL.md": SKILL_BODY})
    assert set(normalized) == {"SKILL.md"}
    assert normalized["SKILL.md"] == SKILL_BODY


@pytest.mark.parametrize(
    "path",
    ["scripts/check.py", "README.md", "references/guide.md", "assets/logo.txt", "../secret.txt"],
)
def test_normalize_bundle_rejects_every_non_bundle_path(path):
    with pytest.raises(SkillBundleError, match="not allowed|invalid"):
        normalize_bundle({"SKILL.md": SKILL_BODY, path: "private"})


def test_normalize_bundle_normalizes_newlines():
    normalized = normalize_bundle({"SKILL.md": "a\r\nb\rc\n"})
    assert normalized["SKILL.md"] == "a\nb\nc\n"


def test_empty_bundle_raises():
    with pytest.raises(SkillBundleError):
        normalize_bundle({"README.md": "no skill here"})
    with pytest.raises(SkillBundleError):
        compute_content_hash({})


def test_content_hash_is_stable_and_path_independent():
    # Hash pinned: changing the algorithm or canonical form must fail this test.
    expected = compute_content_hash({"SKILL.md": SKILL_BODY})
    # Newline normalization is stable across platforms.
    via_path = compute_content_hash({"SKILL.md": SKILL_BODY.replace("\n", "\r\n")})
    assert via_path == expected
    with pytest.raises(SkillBundleError, match="not allowed"):
        compute_content_hash({"SKILL.md": SKILL_BODY, "scripts/check.py": "anything"})
    # Different content -> different hash.
    assert compute_content_hash({"SKILL.md": SKILL_BODY + "x"}) != expected


def test_content_hash_pinned_value():
    # Nail down the exact canonical form and digest so edge and server agree forever.
    assert serialize_bundle({"SKILL.md": "hello\n"}) == '[{"content":"hello\\n","path":"SKILL.md"}]'
    assert (
        compute_content_hash({"SKILL.md": "hello\n"})
        == "068b720526ff50f193ef393f30638007ea9922f85d88a3c7f1246052884b8708"
    )


def test_serialize_roundtrip():
    files = {"SKILL.md": SKILL_BODY}
    text = serialize_bundle(files)
    assert deserialize_bundle(text) == files


def test_deserialize_rejects_garbage():
    with pytest.raises(SkillBundleError):
        deserialize_bundle("{not json")
    with pytest.raises(SkillBundleError):
        deserialize_bundle('{"path": "SKILL.md"}')


@pytest.mark.parametrize(
    "text",
    [
        '["not an object"]',  # element is not a {path, content} record
        '[{"path": 1, "content": "body"}]',  # non-string path
        '[{"path": "SKILL.md", "content": 2}]',  # non-string content
        '[{"path":"SKILL.md","content":"a","extra":"private"}]',
        '[{"content":"a","path":"SKILL.md"},{"content":"b","path":"SKILL.md"}]',
        '[{"content":"a","path":"SKILL.md"},{"content":"private","path":"references/guide.md"}]',
        '[ {"content":"a","path":"SKILL.md"} ]',  # valid data, noncanonical bytes
    ],
)
def test_deserialize_rejects_malformed_records(text):
    # (-> 400), not a raw TypeError/KeyError (-> 500).
    with pytest.raises(SkillBundleError):
        deserialize_bundle(text)
