"""Shared random and deterministic UUID identifier generation."""

from uuid import NAMESPACE_URL, UUID, uuid4, uuid5


def _format_id(value: UUID, prefix: str, length: int, hyphenated: bool) -> str:
    """Format a UUID without changing existing identifier contracts."""
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("UUID length must be an integer")
    if not 1 <= length <= 32:
        raise ValueError("UUID length must be between 1 and 32")
    if hyphenated and length != 32:
        raise ValueError("Hyphenated UUIDs require length 32")
    return prefix + (str(value) if hyphenated else value.hex[:length])


def generate_id(prefix: str = "", length: int = 32, *, hyphenated: bool = False) -> str:
    """Return a prefix followed by a random UUID4 identifier.

    Args:
        prefix: Literal prefix, including any desired separator.
        length: Number of hexadecimal UUID characters, from 1 to 32; excludes prefix.
        hyphenated: Use the standard 36-character UUID format; requires length 32.

    Returns:
        Prefixed UUID string. Shorter UUIDs have fewer possible values.

    Raises:
        TypeError: Length is not an integer.
        ValueError: Length is out of range or incompatible with hyphenated output.
    """
    return _format_id(uuid4(), prefix, length, hyphenated)


def generate_stable_id(
    name: str, prefix: str = "", length: int = 32, *, namespace: UUID = NAMESPACE_URL, hyphenated: bool = False
) -> str:
    """Return a stable UUID5 identifier for a namespace and business key.

    Args:
        name: Business key; the same key and namespace produce the same UUID.
        prefix: Literal prefix, including any desired separator.
        length: Number of hexadecimal UUID characters, from 1 to 32; excludes prefix.
        namespace: UUID namespace used to derive the identifier.
        hyphenated: Use the standard 36-character UUID format; requires length 32.

    Returns:
        Prefixed deterministic UUID string.

    Raises:
        TypeError: Length is not an integer.
        ValueError: Length is out of range or incompatible with hyphenated output.
    """
    return _format_id(uuid5(namespace, name), prefix, length, hyphenated)
