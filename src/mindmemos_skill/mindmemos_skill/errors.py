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


class SkillManageProfileMismatchError(SkillConflictError):
    """Raised when an embedding profile differs from the one a graph or deployment uses."""

    error_code = "SKILL_MANAGE_EMBEDDING_PROFILE_MISMATCH"

    def __init__(self, *, expected: dict[str, object], actual: dict[str, object]) -> None:
        super().__init__(f"{self.error_code}: expected embedding profile {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


class SkillManageWriterConflictError(SkillConflictError):
    """Raised when the graph writer lease is held by, or was taken over by, another writer."""

    error_code = "SKILL_MANAGE_WRITER_CONFLICT"
    retryable = True

    def __init__(self, *, graph_id: str, owner_operation_id: str | None = None) -> None:
        super().__init__(f"{self.error_code}: writer lease for graph {graph_id!r} is not held by this writer")
        self.graph_id = graph_id
        self.owner_operation_id = owner_operation_id


class SkillManageConsistencyError(SkillManagementError, RuntimeError):
    """Raised when durable skill_manage state violates a frozen invariant."""

    error_code = "SKILL_MANAGE_CONSISTENCY_VIOLATION"
    retryable = False

    def __init__(self, reason: str) -> None:
        super().__init__(f"{self.error_code}: {reason}")
        self.reason = reason


class SkillManageValidationError(SkillManagementError, ValueError):
    """Raised when a staged generation fails pre-publish validation."""

    error_code = "SKILL_MANAGE_VALIDATION_FAILED"
    retryable = False

    def __init__(self, reason: str) -> None:
        super().__init__(f"{self.error_code}: {reason}")
        self.reason = reason


class SkillManageRevisionConflictError(SkillManagementError, RuntimeError):
    """Raised when an explicit ``expected_revision`` no longer matches the head.

    The optimistic-lock contract is that the caller asserts the head revision
    its plan was computed against; a mismatch means another writer advanced the
    graph, so the write must be a conflict rather than a silent rebase.
    """

    error_code = "SKILL_MANAGE_REVISION_CONFLICT"
    retryable = False

    def __init__(self, *, expected_revision: int | None, actual_revision: int | None) -> None:
        super().__init__(
            f"{self.error_code}: expected head revision {expected_revision}, actual revision {actual_revision}"
        )
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision


class SkillManageOperationPendingError(SkillManagementError, TimeoutError):
    """Raised when a same-key operation is still in flight after the configured wait."""

    error_code = "SKILL_MANAGE_OPERATION_PENDING"
    retryable = True

    def __init__(self, operation_id: str, status: str) -> None:
        super().__init__(f"{self.error_code}: operation {operation_id} is still {status}")
        self.operation_id = operation_id
        self.status = status


class SkillManageOperationCancelledError(SkillManagementError, RuntimeError):
    """Raised when a same-key retry hits an operation that was cancelled before publish."""

    error_code = "SKILL_MANAGE_OPERATION_CANCELLED"
    retryable = False

    def __init__(self, operation_id: str) -> None:
        super().__init__(f"{self.error_code}: operation {operation_id} was cancelled; retry with a new key")
        self.operation_id = operation_id


class SkillManageOperationPendingRepairError(SkillManagementError, RuntimeError):
    """Raised when an operation's outcome cannot be determined without explicit repair."""

    error_code = "SKILL_MANAGE_OPERATION_PENDING_REPAIR"
    retryable = False

    def __init__(self, operation_id: str, reason: str | None = None) -> None:
        super().__init__(f"{self.error_code}: operation {operation_id} requires repair ({reason or 'unknown'})")
        self.operation_id = operation_id
        self.reason = reason


class SkillManageGcPartialFailureError(SkillManagementError, RuntimeError):
    """Raised when a retired-generation GC pass deleted some generations but not all.

    Each generation is deleted independently, so the pass carries on past a
    failure; ``removed`` names what is physically gone and ``deferred`` pairs
    each generation left in place with the failing exception's type name, so
    a maintenance report never claims nothing was removed when something was.
    """

    error_code = "SKILL_MANAGE_GC_DEFERRED"
    retryable = True

    def __init__(self, *, removed: list[str], deferred: list[tuple[str, str]]) -> None:
        super().__init__(f"{self.error_code}: {len(deferred)} retired generation(s) could not be deleted")
        self.removed = list(removed)
        self.deferred = list(deferred)


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
    "SkillManageProfileMismatchError",
    "SkillManageWriterConflictError",
    "SkillManageConsistencyError",
    "SkillManageValidationError",
    "SkillManageRevisionConflictError",
    "SkillManageOperationCancelledError",
    "SkillManageOperationPendingError",
    "SkillManageOperationPendingRepairError",
    "SkillManageGcPartialFailureError",
    "SkillSnapshotError",
    "SkillExportError",
    "SkillRemoteRequestError",
    "SkillRemoteOperationError",
]
