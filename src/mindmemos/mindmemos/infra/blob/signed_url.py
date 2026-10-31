"""HMAC-signed, expiring URLs for the image proxy route.

Search results embed these as the no-API-key download link when the S3 backend has no
public mirror and presigning fails (or is not wanted): the signature covers
project/blob/owner/expiry, so a leaked link is a bearer capability only until ``expires``.
The route still re-verifies blob ownership in the project's image records, so a valid
signature alone never widens access beyond what a ``memory:read`` key could fetch.
"""

from __future__ import annotations

import hashlib
import hmac


def resolve_signing_secret(url_signing_secret: str | None, gateway_jwt_secret: str | None) -> str:
    """The HMAC key for image URLs: the dedicated secret, else the gateway JWT secret, else empty.

    An empty result must make callers refuse the signed-URL tier entirely (never mint,
    never verify): an HMAC over an empty key is publicly computable, so "verifying" with
    it would accept signatures anyone can forge.
    """

    return url_signing_secret or gateway_jwt_secret or ""


def image_url_signature(project_id: str, blob_id: str, user_id: str, expires: int, secret: str) -> str:
    """HMAC-SHA256 over the signed-URL identity tuple (hex digest, 64 chars)."""

    message = "\n".join((project_id, blob_id, user_id, str(expires)))
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_image_url_signature(
    project_id: str,
    blob_id: str,
    user_id: str,
    expires: int,
    signature: str,
    secret: str,
    *,
    now: int,
) -> bool:
    """Constant-time check that ``signature`` covers this tuple and has not expired.

    ``now`` (unix seconds) is injected by the caller so tests can travel in time.
    """

    if not secret:
        return False  # no key configured: fail closed instead of accepting a forgeable MAC
    if expires <= now:
        return False
    # hexdigest() is 64 lowercase hex chars; anything else (wrong length, uppercase,
    # non-ASCII) is refused before compare_digest, which only accepts ASCII str pairs.
    if len(signature) != 64 or signature.strip("0123456789abcdef"):
        return False
    expected = image_url_signature(project_id, blob_id, user_id, expires, secret)
    return hmac.compare_digest(expected, signature)
