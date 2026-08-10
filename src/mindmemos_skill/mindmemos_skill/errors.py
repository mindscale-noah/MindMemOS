"""Exception hierarchy for the ``mindmemos_skill`` package.

All package-owned exceptions derive from :class:`MindMemOSSkillError`, so a
caller can handle Skill-runtime failures without catching unrelated provider or
Python exceptions.
"""

from __future__ import annotations


class MindMemOSSkillError(Exception):
    """Base class for all errors raised by ``mindmemos_skill`` itself."""


# A spelling-friendly alias for callers who use the Python package name rather
# than the product name.  The canonical public name follows the other
# MindMemOS packages (for example, ``MindMemOSSDKError``).
MindMemosSkillError = MindMemOSSkillError


class ConfigError(MindMemOSSkillError):
    """Base class for local Skill configuration errors."""


class ConfigNotInitializedError(ConfigError):
    """Raised when configuration is accessed before it is initialized."""

    def __init__(self) -> None:
        super().__init__("Config has not been initialized. Call init_config() first.")


class MissingConfigValueError(ConfigError):
    """Raised when a required configuration value is missing."""

    def __init__(self, field: str, reason: str = "") -> None:
        message = f"Missing required config field: '{field}'"
        if reason:
            message += f" ({reason})"
        super().__init__(message)
        self.field = field
        self.reason = reason


class InvalidConfigError(ConfigError):
    """Raised when a configuration value violates a package contract."""

    def __init__(self, field: str, support: str | None = None) -> None:
        message = f"Invalid config field: '{field}'"
        if support:
            message += f" only support ({support})"
        super().__init__(message)
        self.field = field
        self.support = support


class LLMError(MindMemOSSkillError):
    """Base class for local LLM client failures."""


class ModelEndpointNotConfiguredError(LLMError, RuntimeError):
    """Raised when a chat or embedding request has no model endpoint."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        super().__init__(f"No {model_type} model endpoint configured")


class EmbeddingDimensionError(LLMError):
    """Raised when an embedding vector does not match the configured dimension."""

    def __init__(self, *, expected: int, actual: int, model: str, task: str) -> None:
        self.expected = expected
        self.actual = actual
        self.model = model
        self.task = task
        message = (
            f"Embedding dimension mismatch (task={task}, model={model}): "
            f"expected {expected} (= database.qdrant.vector_size), got {actual}. "
            "This usually means the `dimensions` request param was silently dropped by the "
            "provider or litellm (drop_params=True), or the embedding model was switched to one "
            "with a different native dimension. The Qdrant collection dimension is immutable after "
            "creation; restore the previous model, set endpoints[].dimensions to match vector_size, "
            "or drop and recreate the collection."
        )
        super().__init__(message)


class SkillError(MindMemOSSkillError):
    """Base class for Skill algorithm and runtime errors."""


class SkillConfigurationError(SkillError, ValueError):
    """Raised when a Skill runtime is constructed with invalid components."""


class SkillCapabilityUnavailableError(SkillError, RuntimeError):
    """Raised when the configured runtime does not provide an operation."""


class SkillServiceClosedError(SkillError, RuntimeError):
    """Raised when an operation is attempted after the service is closed."""


class SkillManagementError(SkillError):
    """Base class for local Skill management failures."""


class SkillNotFoundError(SkillManagementError, LookupError):
    """Raised when a Skill family or immutable version cannot be resolved."""


class SkillConflictError(SkillManagementError, ValueError):
    """Raised when a local management invariant or CAS precondition fails."""


class SkillSnapshotError(SkillManagementError, ValueError):
    """Raised when an external Skill snapshot is unsafe or malformed."""


class SkillExportError(SkillManagementError):
    """Raised when a snapshot cannot be safely materialized or restored."""


class SkillRemoteRequestError(SkillManagementError):
    """Transport-neutral failure returned by a configured Skill remote port."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retryable: bool,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.error_code = error_code
        self.retryable = retryable
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(message)


class SkillRemoteOperationError(SkillManagementError):
    """Raised after a remote operation fails and its durable state is recorded."""

    def __init__(
        self,
        operation_id: str,
        error_code: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.operation_id = operation_id
        self.error_code = error_code
        self.retryable = retryable
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(f"remote Skill operation failed: {operation_id} ({error_code})")


__all__ = [
    "MindMemOSSkillError",
    "MindMemosSkillError",
    "ConfigError",
    "ConfigNotInitializedError",
    "MissingConfigValueError",
    "InvalidConfigError",
    "LLMError",
    "ModelEndpointNotConfiguredError",
    "EmbeddingDimensionError",
    "SkillError",
    "SkillConfigurationError",
    "SkillCapabilityUnavailableError",
    "SkillServiceClosedError",
    "SkillManagementError",
    "SkillNotFoundError",
    "SkillConflictError",
    "SkillSnapshotError",
    "SkillExportError",
    "SkillRemoteRequestError",
    "SkillRemoteOperationError",
]
