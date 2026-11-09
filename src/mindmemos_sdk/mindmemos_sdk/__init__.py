"""MindMemOS SDK package."""

from .client import MindMemOSClient
from .config import ConfigManager, SDKConfig
from .errors import (
    ApiError,
    AuthRequiredError,
    ConfigError,
    InvalidRequestError,
    MindMemOSSDKError,
    TransportError,
)
from .memory import (
    AddResult,
    AsyncMemoryClient,
    DialogueMessage,
    FeedbackMode,
    FileMessage,
    GetResult,
    MemoryClient,
    SearchResult,
    StatusResult,
    TextMessage,
    UrlMessage,
)
from .skills import AsyncSkillClient, SkillClient

__all__ = [
    "__version__",
    "MindMemOSClient",
    "MemoryClient",
    "AsyncMemoryClient",
    "SkillClient",
    "AsyncSkillClient",
    "FeedbackMode",
    "ConfigManager",
    "SDKConfig",
    "AddResult",
    "SearchResult",
    "GetResult",
    "StatusResult",
    "TextMessage",
    "DialogueMessage",
    "UrlMessage",
    "FileMessage",
    "MindMemOSSDKError",
    "InvalidRequestError",
    "ConfigError",
    "AuthRequiredError",
    "TransportError",
    "ApiError",
]

__version__ = "1.0.0"
