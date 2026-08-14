"""Controller-managed OfficeQA custom search client."""

from __future__ import annotations

import json
import os
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_SERPAPI_BASE_URL_ENV = "OFFICEQA_SERPAPI_BASE_URL"
DEFAULT_SERPAPI_API_KEY_ENV = "OFFICEQA_SERPAPI_API_KEY"
DEFAULT_SERPAPI_PROTOCOL_ENV = "OFFICEQA_SERPAPI_PROTOCOL"
DEFAULT_CUSTOM_SEARCH_MAX_RESULTS = 4
DEFAULT_CUSTOM_SEARCH_TIMEOUT = 20
DEFAULT_CUSTOM_SEARCH_MAX_RETRIES = 4
DEFAULT_CUSTOM_SEARCH_INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/135.0 Safari/537.36"
)


def _extract_search_items(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in (
        "results",
        "items",
        "data",
        "organic",
        "organic_results",
        "search_results",
        "webPages",
        "value",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict) and (nested := _extract_search_items(value)):
            return nested
    return []


def _normalize_search_item(item: dict, index: int) -> str:
    title = str(
        item.get("title")
        or item.get("name")
        or item.get("headline")
        or item.get("source")
        or f"Result {index}"
    ).strip()
    url = str(
        item.get("url")
        or item.get("link")
        or item.get("href")
        or item.get("display_url")
        or ""
    ).strip()
    snippet = str(
        item.get("snippet")
        or item.get("description")
        or item.get("body")
        or item.get("text")
        or item.get("content")
        or ""
    ).strip()
    lines = [f"[{index}] {title}"]
    if url:
        lines.append(f"URL: {url}")
    if snippet:
        lines.append(f"Snippet: {snippet}")
    return "\n".join(lines)


def _format_search_payload(query: str, payload: object, *, max_items: int | None = None) -> str:
    items = _extract_search_items(payload)
    if max_items is not None and max_items > 0:
        items = items[:max_items]
    header = f"Query: {query}"
    if not items:
        body = json.dumps(payload, ensure_ascii=False) if payload else "[no results]"
        return f"{header}\n{body}"
    rendered = [_normalize_search_item(item, index) for index, item in enumerate(items, start=1)]
    return f"{header}\n\n" + "\n\n".join(rendered)


def _is_retryable_search_http_error(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500


def custom_search(
    query: str,
    *,
    base_url: str,
    protocol: str = "serper",
    api_key: str | None = None,
    api_key_env: str = DEFAULT_SERPAPI_API_KEY_ENV,
    max_num_results: int = DEFAULT_CUSTOM_SEARCH_MAX_RESULTS,
    timeout: int = DEFAULT_CUSTOM_SEARCH_TIMEOUT,
    max_retries: int = DEFAULT_CUSTOM_SEARCH_MAX_RETRIES,
    initial_backoff_seconds: float = DEFAULT_CUSTOM_SEARCH_INITIAL_BACKOFF_SECONDS,
) -> str:
    query = str(query or "").strip()
    if not query:
        raise ValueError("custom_search query must be non-empty")
    endpoint = str(base_url or "").strip().rstrip("/")
    if not endpoint:
        raise ValueError("custom_search API base URL must be non-empty")
    if not endpoint.lower().endswith("/search"):
        endpoint += "/search"
    token = str(api_key or os.environ.get(api_key_env, "")).strip()
    if not token:
        raise ValueError(f"custom_search API key missing; set {api_key_env}")
    normalized_protocol = str(protocol or "").strip().lower()
    if normalized_protocol == "serper":
        request_data = json.dumps(
            {"q": query, "num": int(max_num_results)},
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            endpoint,
            data=request_data,
            headers={
                "X-API-KEY": token,
                "Content-Type": "application/json",
                "User-Agent": DEFAULT_USER_AGENT,
            },
            method="POST",
        )
    elif normalized_protocol == "yibu":
        separator = "&" if "?" in endpoint else "?"
        request_url = endpoint + separator + urlencode(
            {"q": query, "output": "json", "num": int(max_num_results)}
        )
        authorization = token if token.lower().startswith("bearer ") else f"Bearer {token}"
        request = Request(
            request_url,
            headers={"Authorization": authorization, "User-Agent": DEFAULT_USER_AGENT},
            method="GET",
        )
    else:
        raise ValueError("custom_search protocol must be 'serper' or 'yibu'")
    attempts = max(1, int(max_retries) + 1)
    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                raw_body = response.read().decode("utf-8", errors="ignore")
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"custom_search HTTP {exc.code}: {detail[:1000]}")
            if attempt >= attempts or not _is_retryable_search_http_error(exc.code):
                raise last_error from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            last_error = RuntimeError(f"custom_search connection error: {exc}")
            if attempt >= attempts:
                raise last_error from exc
        backoff = max(0.0, float(initial_backoff_seconds)) * (2 ** (attempt - 1))
        if backoff > 0:
            time.sleep(backoff)
    else:
        raise last_error or RuntimeError("custom_search failed without a captured error")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return f"Query: {query}\n\n{raw_body.strip() or '[empty response]'}"
    return _format_search_payload(query, payload, max_items=int(max_num_results))


__all__ = [
    "DEFAULT_SERPAPI_API_KEY_ENV",
    "DEFAULT_SERPAPI_BASE_URL_ENV",
    "DEFAULT_SERPAPI_PROTOCOL_ENV",
    "custom_search",
]
