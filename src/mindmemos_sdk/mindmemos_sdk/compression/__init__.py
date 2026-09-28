"""Stateless working-memory compression, independent of long-term memory APIs."""

from .compressor import compress_memory
from .models import CompressionConfig, CompressionError, CompressionMode, CompressionResult

__all__ = [
    "compress_memory",
    "CompressionConfig",
    "CompressionResult",
    "CompressionMode",
    "CompressionError",
]
