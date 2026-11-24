"""Portable vector declarations; backend codecs live in builtin_backends."""

import math
from array import array

from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType):
    """Declare a fixed or variable-dimensional float32 vector column.

    Args:
        dimensions: Positive element count, or None to allow per-row dimensions.
    """

    cache_ok = True

    def __init__(self, dimensions: int | None = None):
        if dimensions is not None and (
            isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0
        ):
            raise ValueError("Vector dimensions must be a positive integer")
        self.dimensions = dimensions

    def get_col_spec(self, **kwargs):
        """Return the logical SQL declaration.

        Args:
            **kwargs: SQLAlchemy compiler options.
        """
        return "VECTOR" if self.dimensions is None else f"VECTOR({self.dimensions})"

    def validate(self, value):
        """Validate and normalize a cosine-searchable float32 vector.

        Args:
            value: Numeric sequence with exactly the declared dimensions.

        Returns:
            A list of finite float32 values with nonzero norm.
        """
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("Expected a nonempty vector")
        if self.dimensions is not None and len(value) != self.dimensions:
            raise ValueError(f"Expected a vector with {self.dimensions} dimensions")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
            raise ValueError("Vector elements must be numbers")
        try:
            result = list(array("f", value))
        except (OverflowError, TypeError) as exc:
            raise ValueError("Invalid float32 vector") from exc
        if not all(math.isfinite(x) for x in result) or not any(result):
            raise ValueError("Cosine vectors must be finite and nonzero")
        return result
