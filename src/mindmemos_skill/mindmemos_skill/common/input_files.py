"""Compute private file identities from materialized input models."""

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel


def file_fingerprints(paths: Iterable[Path]) -> dict[str, str]:
    """Fingerprint input files by their resolved paths and current bytes.

    Args:
        paths: Existing input files; repeated paths are read only once.

    Returns:
        Absolute file paths mapped to SHA-256 digests for internal recovery checks.

    Raises:
        OSError: An input file cannot be read.
    """
    result = {}
    for path in sorted({path.expanduser().resolve() for path in paths}):
        with path.open("rb") as stream:
            result[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def model_file_fingerprints(*models: BaseModel) -> dict[str, str]:
    """Fingerprint Path fields nested in input models, mappings, and lists.

    Args:
        models: Task or scoring models whose Path fields refer to input files.
            String fields remain ordinary business data and are never treated as paths.

    Returns:
        Resolved input paths mapped to internally computed content digests.

    Raises:
        OSError: An input file cannot be read.
    """
    return file_fingerprints(path for model in models for path in _paths(model.model_dump(mode="python")))


def _paths(value: object) -> Iterator[Path]:
    """Find typed paths inside a serialized model without interpreting strings."""
    if isinstance(value, Path):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _paths(item)
