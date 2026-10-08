"""Cross-language vectors and integrity checks for the new Skill content hashes."""

import pytest
from mindmemos_skill.skill.content import (
    content_hash,
    decode_resources,
    encode_resources,
    skill_hash,
    validate_resource_paths,
)


@pytest.mark.parametrize(
    ("content", "resources", "expected_content_hash", "expected_skill_hash"),
    [
        (
            "",
            {},
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "d309b0aec2471247d5c7fae40a6c8e8c762223603200600b6c34a56376dde16f",
        ),
        (
            "---\r\nname: Café\r\n---\r\n",
            {},
            "24488d2b8fbd36705f4702c8048ed8f82d29b335e67440ef42033e297abb2dd3",
            "153eeabb772ce4d15cebefe91451ad36f524d8fefec8d352af318c0f037cf770",
        ),
        (
            "你好\n",
            {"assets/logo.bin": b"\x00\xff", "references/note.txt": "Café".encode("utf-8")},
            "4e0826721642ed8e3a27e7147538ac7b7013a08fe5ae343a8ef09749b7e5790f",
            "fb588d7943833f61a1388f6df978b744cdf877a89c339e5fd8df8a6654eb172b",
        ),
    ],
)
def test_hash_vectors(content, resources, expected_content_hash, expected_skill_hash):
    """Keep exact hashes stable across languages and transport implementations."""
    assert content_hash(content) == expected_content_hash
    assert skill_hash(content, resources) == expected_skill_hash
    assert decode_resources(encode_resources(resources)) == resources


def test_resource_order_and_changes():
    """Make hashing independent of insertion order but sensitive to every source byte."""
    content = "Instructions\r\n"
    resources = {"z.txt": b"z", "a.bin": b"\x00"}
    assert skill_hash(content, resources) == skill_hash(content, dict(reversed(list(resources.items()))))
    assert skill_hash(content, resources) != skill_hash(content, {**resources, "z.txt": b"Z"})
    assert skill_hash(content, resources) != skill_hash(content, {"z.txt": b"z", "b.bin": b"\x00"})
    assert skill_hash(content, resources) != skill_hash("Instructions\n", resources)
    assert content_hash(content) != content_hash("Instructions\n")


@pytest.mark.parametrize(
    "path", ["/absolute", "../parent", "./file", "a//b", "SKILL.md", "SKILL.md/x", "a\\b", "x/", "x\x00y"]
)
def test_invalid_resource_paths(path):
    """Keep resources within a canonical Skill directory."""
    with pytest.raises(ValueError):
        validate_resource_paths({path: b"content"})


def test_overlapping_paths_and_noncanonical_base64():
    """Reject ambiguous directory layouts and noncanonical transport bytes."""
    with pytest.raises(ValueError):
        validate_resource_paths({"a": b"file", "a/b": b"nested"})
    with pytest.raises(ValueError):
        decode_resources({"asset.bin": "YQ==\n"})
