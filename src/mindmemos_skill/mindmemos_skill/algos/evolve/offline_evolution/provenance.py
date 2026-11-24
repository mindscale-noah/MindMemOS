"""Verbatim input-quote verification shared by both gates."""

import json


def quoted(quote: str, data: bytes) -> bool:
    """Check the quote, or its JSON-escaped form, against raw file bytes.

    CRLF line endings are also compared as LF, since text-mode reads translate them.

    Args:
        quote: Text that should occur in the file.
        data: Raw file content.

    Returns:
        Whether some form of the quote occurs.
    """
    variants = {quote, json.dumps(quote, ensure_ascii=False)[1:-1], json.dumps(quote)[1:-1]}
    texts = {data, data.replace(b"\r\n", b"\n")}
    return any(variant.encode("utf-8") in text for variant in variants for text in texts)
