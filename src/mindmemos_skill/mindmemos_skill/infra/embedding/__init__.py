"""Configurable, injectable embedding transport."""

from .client import OpenAIEmbeddingClient
from .config import OpenAIEmbeddingConfig
from .typing import EmbeddingModel

__all__ = ["EmbeddingModel", "OpenAIEmbeddingClient", "OpenAIEmbeddingConfig"]
