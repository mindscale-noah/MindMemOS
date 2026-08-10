"""SDK exception hierarchy.

All SDK errors derive from :class:`MindMemOSSDKError` so callers can catch the
whole family with a single base class. Config-related failures live here because
they are raised before any network transport is involved.
"""

from __future__ import annotations


class MindMemOSSDKError(Exception):
    """Base class for all MindMemOS SDK errors."""


class InvalidRequestError(MindMemOSSDKError):
    """Raised when SDK method arguments cannot form a valid request."""


class ConfigError(MindMemOSSDKError):
    """Raised when local SDK configuration cannot be read, parsed, or written."""


class ConfigNotFoundError(ConfigError):
    """Raised when the SDK config file does not exist yet.

    Typically resolved by running ``mindmemos auth`` to create
    ``~/.mindmemos/config.yaml``.
    """


class ConfigValidationError(ConfigError):
    """Raised when config content is present but fails schema validation."""


class TransportError(MindMemOSSDKError):
    """Raised when an HTTP request cannot complete (network, timeout, connect)."""


class ApiError(MindMemOSSDKError):
    """Raised when the server returns a non-``ok`` envelope or an HTTP error.

    Attributes:
        code: Business error code from the response envelope, when available.
        status_code: HTTP status code, when a response was received.
        request_id: Server-side request id for correlation, when available.
        response_body: Raw decoded response body, for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        response_body: object | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.request_id = request_id
        self.response_body = response_body


class AuthRequiredError(ConfigError):
    """Raised when an API call is attempted without a configured api_key."""


class SkillError(MindMemOSSDKError):
    """Base class for SDK skill-management errors."""


class SkillCapabilityUnavailableError(SkillError):
    """Raised when the configured Skill application lacks an operation."""


class SkillRemoteError(SkillError):
    """Stable SDK projection of a configured Skill remote failure."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retryable: bool,
        status_code: int | None = None,
        request_id: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        self.error_code = error_code
        self.retryable = retryable
        self.status_code = status_code
        self.request_id = request_id
        self.operation_id = operation_id
        super().__init__(message)


class SkillBundleError(SkillError):
    """Raised when a local skill bundle cannot be read or normalized."""


class SkillRegistryError(SkillError):
    """Raised when SDK skill registry content is invalid."""


def translate_skill_error(exc: Exception) -> SkillError:
    """Translate Skill domain errors without exposing package-specific classes."""

    from mindmemos_skill import (
        MindMemOSSkillError,
        SkillRemoteOperationError,
        SkillRemoteRequestError,
    )
    from mindmemos_skill import (
        SkillCapabilityUnavailableError as DomainCapabilityUnavailableError,
    )

    if isinstance(exc, DomainCapabilityUnavailableError):
        return SkillCapabilityUnavailableError(str(exc))
    if isinstance(exc, (SkillRemoteRequestError, SkillRemoteOperationError)):
        return SkillRemoteError(
            str(exc),
            error_code=exc.error_code,
            retryable=exc.retryable,
            status_code=exc.status_code,
            request_id=exc.request_id,
            operation_id=getattr(exc, "operation_id", None),
        )
    if isinstance(exc, MindMemOSSkillError):
        return SkillRegistryError(str(exc))
    raise TypeError(f"not a Skill domain error: {type(exc).__name__}")
