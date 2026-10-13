"""Contracts for shared UUID generation and existing identifier formats."""

from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from mindmemos_skill.common.ids import generate_id, generate_stable_id


@pytest.mark.parametrize("prefix,length", [("", 32), ("skg-", 32), ("sk-", 16), ("skv-", 8), ("_", 1)])
def test_random_id_prefix_and_length(prefix: str, length: int) -> None:
    identifier = generate_id(prefix, length)
    assert identifier.startswith(prefix)
    suffix = identifier[len(prefix) :]
    assert len(suffix) == length
    assert all(character in "0123456789abcdef" for character in suffix)


def test_default_id_is_a_random_uuid4() -> None:
    first, second = generate_id(), generate_id()
    assert len(first) == 32
    assert UUID(first).version == 4
    assert first != second


def test_hyphenated_id_preserves_standard_uuid_format() -> None:
    identifier = generate_id(hyphenated=True)
    assert len(identifier) == 36
    assert str(UUID(identifier)) == identifier
    assert UUID(identifier).version == 4


@pytest.mark.parametrize("length", [0, -1, 33])
def test_invalid_length_is_rejected(length: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 32"):
        generate_id(length=length)


@pytest.mark.parametrize("length", [True, 8.5, "8", None])
def test_noninteger_length_is_rejected(length) -> None:
    with pytest.raises(TypeError, match="integer"):
        generate_id(length=length)


def test_hyphenated_id_cannot_be_truncated() -> None:
    with pytest.raises(ValueError, match="require length 32"):
        generate_id(length=12, hyphenated=True)


def test_stable_id_matches_existing_publication_id() -> None:
    name = "training:project:run"
    expected = "skv-" + uuid5(NAMESPACE_URL, name).hex
    assert generate_stable_id(name, "skv-") == expected
    assert generate_stable_id(name, "skv-") == generate_stable_id(name, "skv-")
    assert generate_stable_id(name + "2", "skv-") != expected


def test_stable_id_supports_namespace_length_and_format() -> None:
    namespace = UUID("11111111-2222-3333-4444-555555555555")
    expected = uuid5(namespace, "key")
    assert generate_stable_id("key", "test-", 12, namespace=namespace) == "test-" + expected.hex[:12]
    assert generate_stable_id("key", namespace=namespace, hyphenated=True) == str(expected)
