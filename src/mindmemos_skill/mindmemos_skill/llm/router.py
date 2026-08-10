"""LiteLLM router construction and response-normalization helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeAlias

from ..errors import SkillCapabilityUnavailableError

if TYPE_CHECKING:
    from litellm import Router

ConfigObject: TypeAlias = Mapping[str, Any] | object

_LITELLM_PARAM_FIELDS: tuple[str, ...] = (
    "model",
    "api_key",
    "api_base",
    "rpm",
    "tpm",
    "timeout",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "encoding_format",
    "dimensions",
    "num_retries",
)


def _load_litellm() -> Any:
    """Load and configure the optional LiteLLM dependency on first use."""

    try:
        litellm = import_module("litellm")
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "litellm"
        raise SkillCapabilityUnavailableError(
            "LLM capability is unavailable because LiteLLM is not installed correctly "
            f"(missing module: {missing_module!r}). "
            "Install it with `pip install 'mindmemos-skill[llm]'`."
        ) from exc

    litellm.drop_params = True
    litellm.suppress_debug_info = True
    litellm.turn_off_message_logging = True
    logging.getLogger("LiteLLM").setLevel(logging.INFO)
    for logger_name in ("LiteLLM Router", "LiteLLM Proxy"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    return litellm


def resolve_model_provider(model: str, *, api_base: str | None = None) -> str:
    """Return the provider LiteLLM resolves for one model endpoint."""

    litellm = _load_litellm()
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model, api_base=api_base)
    except Exception as exc:
        raise ValueError(f"unrecognized LiteLLM model identifier {model!r}") from exc
    if not provider:
        raise ValueError(f"unrecognized LiteLLM model identifier {model!r}")
    return str(provider)


def validate_model_identifier(model: str, *, api_base: str | None = None) -> None:
    """Validate that LiteLLM can resolve a provider from one model identifier."""

    resolve_model_provider(model, api_base=api_base)


@dataclass(slots=True)
class Usage:
    """Provider token counters shared by chat and embedding responses."""

    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None


def _config_value(config: ConfigObject, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _model_supports_dimensions(model: str, whitelist: Sequence[str]) -> bool:
    stripped = model.split("/", 1)[1] if "/" in model else model
    return any(stripped.startswith(prefix) for prefix in whitelist)


def build_litellm_params(
    endpoint: ConfigObject,
    *,
    dimensions_supported_models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Flatten a mapping, dataclass, or Pydantic endpoint into LiteLLM params."""
    params = {key: value for key in _LITELLM_PARAM_FIELDS if (value := _config_value(endpoint, key)) is not None}
    extra_body = _config_value(endpoint, "extra_body")
    if extra_body:
        params["extra_body"] = dict(extra_body) if isinstance(extra_body, Mapping) else extra_body

    model = str(_config_value(endpoint, "model", ""))
    dimensions = _config_value(endpoint, "dimensions")
    if dimensions is not None and _model_supports_dimensions(model, dimensions_supported_models or ()):
        params["allowed_openai_params"] = ["dimensions"]
    return params


def build_router(router_config: ConfigObject, alias: str, *, num_retries: int | None = None) -> tuple[Router, int]:
    """Build a LiteLLM router from a mapping or attribute-based config object."""
    litellm = _load_litellm()
    endpoints = list(_config_value(router_config, "endpoints", ()) or ())
    dimensions_supported_models = tuple(_config_value(router_config, "dimensions_supported_models", ()) or ())
    deployment_counts: dict[str, int] = {}
    model_list: list[dict[str, Any]] = []
    for endpoint in endpoints:
        model = str(_config_value(endpoint, "model", ""))
        api_base = str(_config_value(endpoint, "api_base", ""))
        deployment_key = f"{model}@{api_base}"
        deployment_counts[deployment_key] = deployment_counts.get(deployment_key, 0) + 1
        model_list.append(
            {
                "model_name": alias,
                "litellm_params": build_litellm_params(
                    endpoint,
                    dimensions_supported_models=dimensions_supported_models,
                ),
                "model_info": {"id": f"{deployment_key}#{deployment_counts[deployment_key]}"},
            }
        )

    max_retries = num_retries
    if max_retries is None:
        max_retries = max((int(_config_value(endpoint, "num_retries", 0) or 0) for endpoint in endpoints), default=0)
    router = litellm.Router(
        model_list=model_list,
        routing_strategy=_config_value(router_config, "routing_strategy", "simple-shuffle"),
        num_retries=max_retries,
        allowed_fails=_config_value(router_config, "allowed_fails"),
        cooldown_time=_config_value(router_config, "cool_down"),
    )
    return router, max_retries


_ROUTER_CACHE: dict[str, tuple[Router, int]] = {}


def _router_cache_key(router_config: ConfigObject, alias: str, num_retries: int | None) -> str:
    endpoints = list(_config_value(router_config, "endpoints", ()) or ())
    dimensions_supported_models = tuple(_config_value(router_config, "dimensions_supported_models", ()) or ())
    payload = {
        "alias": alias,
        "routing_strategy": _config_value(router_config, "routing_strategy", "simple-shuffle"),
        "allowed_fails": _config_value(router_config, "allowed_fails"),
        "cool_down": _config_value(router_config, "cool_down"),
        "num_retries": num_retries,
        "dimensions_supported_models": dimensions_supported_models,
        "endpoints": [
            build_litellm_params(endpoint, dimensions_supported_models=dimensions_supported_models)
            for endpoint in endpoints
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialized).hexdigest()


def get_router(router_config: ConfigObject, alias: str, *, num_retries: int | None = None) -> tuple[Router, int]:
    """Return the cached router for an effective configuration."""
    key = _router_cache_key(router_config, alias, num_retries)
    if key not in _ROUTER_CACHE:
        _ROUTER_CACHE[key] = build_router(router_config, alias, num_retries=num_retries)
    return _ROUTER_CACHE[key]


def clear_router_cache() -> None:
    """Drop cached routers so refreshed configuration takes effect."""
    _ROUTER_CACHE.clear()


def dump_response(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of a provider response to a plain dictionary."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return {}
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception:
        return {}


def get_response_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def litellm_response_headers(obj: Any) -> dict[str, Any]:
    hidden = getattr(obj, "_hidden_params", {}) or {}
    if hasattr(hidden, "model_dump"):
        hidden = hidden.model_dump()
    if not isinstance(hidden, Mapping):
        return {}
    headers = hidden.get("additional_headers", {}) or {}
    return dict(headers) if isinstance(headers, Mapping) else {}


def usage_tokens(usage: Any) -> Usage:
    return Usage(
        completion_tokens=get_response_value(usage, "completion_tokens"),
        prompt_tokens=get_response_value(usage, "prompt_tokens"),
        total_tokens=get_response_value(usage, "total_tokens"),
    )
