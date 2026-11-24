"""Registered embedding backends with configuration-driven construction."""

from .base import EmbeddingModel
from .client import OpenAIEmbeddingClient
from .config import EmbeddingConfig, OpenAIEmbeddingConfig
from .registry import ensure_registered

ensure_registered()

__all__ = ["EmbeddingConfig", "EmbeddingModel", "OpenAIEmbeddingClient", "OpenAIEmbeddingConfig"]
