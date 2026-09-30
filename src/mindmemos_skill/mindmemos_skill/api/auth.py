"""Resolve Skill HTTP credentials into a trusted project and operation scopes."""

import hmac
from pathlib import Path
from typing import Annotated, Any

import jwt
import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints

from ..config.config import HttpAuthConfig

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Nonempty user-supplied text after trimming surrounding whitespace."""


class SkillHttpIdentity(BaseModel):
    """Trusted account identity supplied by the deployment authentication adapter."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")
    """Accept the shared account provider's resolved identity object."""

    project_id: Nonempty
    """Authorized project selected by the credential, never by the request body."""

    scopes: set[str] = Field(default_factory=set)
    """Granted Skill operation scopes from the same account system as Memory."""

    request_id: Nonempty | None = None
    """Request trace identity; the HTTP entry generates one when omitted."""

    actor_id: Nonempty | None = None
    """Trusted account or service identity recorded on graph publications."""

    principal: Any = None
    """Original trusted principal passed to a configured project authorizer."""


class SkillHttpError(Exception):
    """Carry a stable HTTP error code without exposing backend internals."""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


class LocalKeyFile(BaseModel):
    """Credentials for one locally deployed Skill project."""

    model_config = ConfigDict(extra="forbid")
    """Reject ambiguous or misspelled key file fields."""

    project_id: str = Field(min_length=1)
    """Project assigned to all requests carrying this key."""

    api_key: SecretStr
    """Bearer credential accepted by this local deployment."""


class LocalKeyResolver:
    """Resolve a single local key while allowing atomic credential rotation."""

    def __init__(self, path: str | Path):
        self._path = Path(path).expanduser().resolve()

    def __call__(self, credential: str) -> SkillHttpIdentity:
        """Authenticate a bearer key against the configured file.

        Args:
            credential: Bearer credential supplied with the request.

        Returns:
            The project authorized by the configured key.

        Raises:
            SkillHttpError: The key file is unavailable or the credential is invalid.
        """
        try:
            key_file = LocalKeyFile.model_validate(yaml.safe_load(self._path.read_text(encoding="utf-8")))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise SkillHttpError(503, "authentication_unavailable", "Local API key file is unavailable") from exc
        expected = key_file.api_key.get_secret_value()
        if not expected or not hmac.compare_digest(credential, expected):
            raise SkillHttpError(401, "auth.invalid_api_key", "Invalid API key")
        return SkillHttpIdentity(project_id=key_file.project_id, scopes={"skill:write"}, actor_id=key_file.project_id)


class GatewayJwtResolver:
    """Verify Gateway-issued JWTs within the Skill service."""

    def __init__(self, *, secret: str, issuer: str, audience: str):
        """Set the Gateway signature key and expected token origin.

        Args:
            secret: HMAC key shared with the Gateway.
            issuer: Required JWT issuer claim.
            audience: Required JWT audience claim.
        """
        self._secret = secret
        self._issuer = issuer
        self._audience = audience

    def __call__(self, credential: str) -> SkillHttpIdentity:
        """Verify a bearer JWT and return its authorized project and scopes.

        Args:
            credential: Gateway-issued bearer JWT supplied with the request.

        Returns:
            Project and scopes from the verified JWT.

        Raises:
            SkillHttpError: The JWT or its required project claim is invalid.
        """
        try:
            payload = jwt.decode(
                credential,
                self._secret,
                algorithms=["HS256"],
                issuer=self._issuer,
                audience=self._audience,
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise SkillHttpError(401, "auth.internal_token_expired", "Gateway token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise SkillHttpError(401, "auth.invalid_internal_token", "Invalid Gateway token") from exc
        project_id = payload.get("project_id")
        scopes = payload.get("scopes", [])
        if (
            not isinstance(project_id, str)
            or not project_id.strip()
            or not isinstance(scopes, list)
            or not all(isinstance(scope, str) for scope in scopes)
            or payload.get("aud") != self._audience
            or type(payload.get("exp")) is not int
        ):
            raise SkillHttpError(401, "auth.invalid_internal_token", "Invalid Gateway token claims")
        actor_id = payload.get("sub") or payload.get("account_id")
        if not isinstance(actor_id, str) or not actor_id.strip():
            actor_id = project_id
        return SkillHttpIdentity(project_id=project_id, scopes=set(scopes), actor_id=actor_id)


def build_identity_resolver(config: HttpAuthConfig) -> LocalKeyResolver | GatewayJwtResolver | None:
    """Build the selected credential resolver without importing the Memory service.

    Args:
        config: Validated Skill HTTP authentication settings.

    Returns:
        A resolver for authenticated modes, or None for local bypass.
    """
    if config.mode == "local":
        assert config.api_key_file is not None
        return LocalKeyResolver(config.api_key_file)
    if config.mode == "gateway_jwt":
        assert config.gateway_jwt_secret is not None
        return GatewayJwtResolver(
            secret=config.gateway_jwt_secret.get_secret_value(),
            issuer=config.gateway_jwt_issuer,
            audience=config.gateway_jwt_audience,
        )
    return None
