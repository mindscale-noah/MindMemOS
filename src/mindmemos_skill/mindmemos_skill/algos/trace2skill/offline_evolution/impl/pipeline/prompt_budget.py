"""Provide consistent truncation markers for budgeted text and item views."""

from __future__ import annotations


def truncation_notice(
    rel_path: str,
    *,
    omitted: int | None = None,
    what: str = "content",
) -> str:
    """Render the highlighted truncation marker.

    ``rel_path`` is relative to the READER's cwd (the scope workspace
    root for stage agents). ``omitted``/``what`` name the elided span
    when countable; omit them for byte-capped blobs.
    """
    count = f"{omitted} {what} " if omitted is not None else f"{what} "
    return f"> **[TRUNCATED]** {count}omitted from this view for prompt budget. Read the full text at `{rel_path}`."


def cut_text(
    text: str,
    *,
    rel_path: str,
    char_cap: int | None = None,
    byte_cap: int | None = None,
    tail_fraction: float = 0.0,
    what: str = "content",
) -> tuple[str, bool]:
    """Budget a text blob and mark omitted content.

    Exactly one of ``char_cap`` / ``byte_cap`` must be given.
    ``tail_fraction`` in ``[0, 0.5]`` requires ``char_cap`` and preserves
    both ends of the input. Head-only cuts stop at a line boundary.
    Untruncated input is returned unchanged.
    """
    if (char_cap is None) == (byte_cap is None):
        raise ValueError("give exactly one of char_cap / byte_cap")
    if not 0.0 <= tail_fraction <= 0.5:
        raise ValueError("tail_fraction must be within [0, 0.5]")
    if tail_fraction > 0.0 and byte_cap is not None:
        raise ValueError("head+tail sandwich requires char_cap")

    def _size(s: str) -> int:
        return len(s) if char_cap is not None else len(s.encode("utf-8"))

    cap = char_cap if char_cap is not None else byte_cap
    if cap is not None and cap <= 0:
        raise ValueError("cap must be positive")
    assert cap is not None  # for the type checker; one is always set
    if _size(text) <= cap:
        return text, False

    total = _size(text)
    if tail_fraction > 0.0:
        # Char-domain sandwich: budget by characters, exact.
        head_len = int(cap * (1.0 - tail_fraction))
        tail_len = int(cap * tail_fraction)
        head, tail = text[:head_len], text[len(text) - tail_len :]
        omitted = total - head_len - tail_len
        marker = truncation_notice(rel_path, omitted=omitted, what=what)
        return head + "\n" + marker + "\n" + tail, True

    if byte_cap is not None:
        head = text.encode("utf-8")[:byte_cap].decode("utf-8", errors="ignore")
    else:
        head = text[:cap]
    head = head.rsplit("\n", 1)[0]
    omitted = total - _size(head)
    return head + "\n" + truncation_notice(rel_path, omitted=omitted, what=what), True


def cut_items(
    items: list,
    *,
    cap: int,
    rel_path: str,
    what: str,
) -> tuple[list, str | None]:
    """Keep the first ``cap`` items and return a marker when items were omitted."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    if len(items) <= cap:
        return items, None
    return items[:cap], truncation_notice(rel_path, omitted=len(items) - cap, what=what)


__all__ = ["cut_items", "cut_text", "truncation_notice"]
