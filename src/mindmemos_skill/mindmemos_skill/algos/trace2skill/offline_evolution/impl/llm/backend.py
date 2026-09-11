"""Backend-neutral orchestration for local agent invocation channels."""

from __future__ import annotations

import contextlib
import contextvars
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Protocol

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.config import (
    setting_int,
    setting_optional_str,
    setting_raw,
)

from .invocation import InvokeError, InvokeResult, prepare_retry
from .tracing import write_invoke_trace

BackendName = Literal["react",]

_ALLOWED_BACKENDS = {
    "react",
}
# Context-local, not module-global: workers bind backends inside thread
# pools, and each context copy keeps one worker's binding from clobbering
# another's.
_ACTIVE_BACKEND: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_active_backend", default=None)
_ACTIVE_DEFAULT_CWD: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_active_default_cwd", default=None)
# Context-local record hook: a plain synchronous callback (max_attempts +
# final result, no per-attempt traffic) tied to the calling pipeline run. It
# lets the host surface vendored LLM calls into its own audit table without
# the vendored code importing host types.
_ACTIVE_CALL_RECORD: contextvars.ContextVar["CallRecorder | None"] = contextvars.ContextVar(
    "llm_active_call_record", default=None
)


class CallRecorder(Protocol):
    """Receives one completed (or terminally failed) logical LLM call."""

    def record(self, call: "RecordedCall") -> None: ...


@dataclass(frozen=True)
class RecordedCall:
    """A transport-neutral summary of one vendored LLM call for host auditing."""

    task: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    status: str
    error: str | None
    # These are the raw system/request payloads plus the wire response;
    # redaction of secrets happens on the host side when it builds the record.
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    latency_ms: float = 0.0


@contextlib.contextmanager
def bind_call_recorder(recorder: CallRecorder | None) -> Iterator[None]:
    """Temporarily bind a call recorder for the calling context.

    Context-local like the backend binding, so concurrent workers each see
    their own recorder and no module-global state is shared.
    """
    token = _ACTIVE_CALL_RECORD.set(recorder)
    try:
        yield
    finally:
        _ACTIVE_CALL_RECORD.reset(token)


def current_call_recorder() -> CallRecorder | None:
    """Return the recorder bound in the calling context, or ``None``."""
    return _ACTIVE_CALL_RECORD.get()


def normalise_backend(name: str | None) -> BackendName:
    """Normalize and validate an invocation backend name."""
    raw = (name or "").strip()
    if not raw:
        raw = setting_raw("agent.backend").strip()
    if raw not in _ALLOWED_BACKENDS:
        allowed = ", ".join(sorted(_ALLOWED_BACKENDS))
        raise ValueError(f"unsupported LLM backend {raw!r}; expected one of: {allowed}")
    return raw  # type: ignore[return-value]


def get_backend() -> BackendName:
    """Return the backend bound in the calling context."""
    return normalise_backend(_ACTIVE_BACKEND.get())


def get_default_cwd() -> str | None:
    """Return the default subprocess cwd bound in the calling context."""
    return _ACTIVE_DEFAULT_CWD.get()


def stage_tools(tools: list[str] | None, *, with_bash: bool = False) -> list[str] | None:
    """Declare a pipeline stage's tool set, opting Bash in explicitly.

    ``with_bash`` is the stage's intent to expose unrestricted Bash. The
    react backend cannot carry Bash in its tool set (``SandboxedToolRuntime``
    rejects it), so it is stripped there; every other backend passes it
    through. A backend never injects Bash on its own -- a stage must ask.
    """
    if not with_bash or not tools:
        return tools
    if get_backend() == "react":
        return tools
    return list(dict.fromkeys([*tools, "Bash"]))


@contextlib.contextmanager
def use_backend(
    backend: str | None,
    *,
    default_cwd: str | None = None,
) -> Iterator[BackendName]:
    """Temporarily bind the active LLM backend for the calling context.

    Context-local, so concurrent workers each see their own binding.
    """
    resolved = normalise_backend(backend)
    backend_token = _ACTIVE_BACKEND.set(resolved)
    cwd_token = _ACTIVE_DEFAULT_CWD.set(default_cwd)
    try:
        yield resolved
    finally:
        _ACTIVE_DEFAULT_CWD.reset(cwd_token)
        _ACTIVE_BACKEND.reset(backend_token)


def _backend_invoke(name: BackendName) -> Callable[..., InvokeResult]:
    if name == "react":
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react import invoke as backend_invoke

        return backend_invoke
    raise AssertionError(f"unreachable backend: {name}")


def _invoke_max_attempts(explicit: Any) -> int:
    if explicit is None:
        try:
            max_attempts = setting_int("agent.invokes_max_attempts")
        except Exception as e:
            raise InvokeError(
                "llm",
                f"invalid agent.invokes_max_attempts setting: {e}",
                cause=e if isinstance(e, Exception) else None,
                retryable=False,
            )
    else:
        try:
            max_attempts = int(explicit)
        except (TypeError, ValueError) as e:
            raise InvokeError(
                "llm",
                f"max_attempts={explicit!r} is not a valid int",
                cause=e,
                retryable=False,
            )
    if max_attempts < 1:
        raise InvokeError(
            "llm",
            f"max_attempts must be >= 1, got {max_attempts!r}",
            retryable=False,
        )
    return max_attempts


def _normalize_roots(
    backend: BackendName,
    roots: Any,
    cwd: Any,
    label: str,
) -> list[str] | None:
    """Validate cwd-contained file roots, returning resolved paths.

    Shared by read_roots and write_roots. Returns None when ``roots`` is
    None so callers leave the key untouched.
    """
    if roots is None:
        return None
    if cwd is None:
        raise InvokeError(backend, f"{label}_roots requires cwd", retryable=False)
    cwd_path = Path(os.fspath(cwd)).resolve()
    normalized: list[str] = []
    for raw in roots:
        root = Path(os.fspath(raw)).resolve()
        try:
            root.relative_to(cwd_path)
        except ValueError as exc:
            raise InvokeError(
                backend,
                f"{label} root is outside cwd: {root}",
                cause=exc,
                retryable=False,
            ) from exc
        if not root.is_dir():
            raise InvokeError(
                backend,
                f"{label} root does not exist or is not a directory: {root}",
                retryable=False,
            )
        normalized.append(str(root))
    return normalized


def _normalise_kwargs_for_backend(
    backend: BackendName,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return backend-specific kwargs without mutating the caller's dict."""
    out = dict(kwargs)

    if out.get("cwd") is None:
        default_cwd = get_default_cwd()
        if default_cwd:
            out["cwd"] = default_cwd

    cwd = out.get("cwd")
    if cwd is not None and not os.path.isdir(os.fspath(cwd)):
        raise InvokeError(
            backend,
            f"cwd does not exist or is not a directory: {cwd}",
            retryable=False,
        )

    normalized_read = _normalize_roots(backend, out.get("read_roots"), cwd, "read")
    if normalized_read is not None:
        out["read_roots"] = normalized_read
    normalized_write = _normalize_roots(backend, out.get("write_roots"), cwd, "write")
    if normalized_write is not None:
        out["write_roots"] = normalized_write

    if backend == "react":
        # CLI-only permission/process flags have no meaning for the API backend.
        for name in (
            "permission_mode",
            "output_format",
            "model",
            "agent",
            "files",
            "attach",
            "safe_mode",
            "pure",
            "timeout_stage",
        ):
            out.pop(name, None)

    return out


def _record_call(
    recorder: CallRecorder | None,
    *,
    task: str,
    model: str,
    attempt_user: str,
    system: str,
    kwargs: dict[str, Any],
    result: InvokeResult | None,
    error: InvokeError | None,
    latency_ms: float,
) -> None:
    """Forward one terminal LLM call to the bound host recorder, if any."""
    if recorder is None:
        return
    if result is not None:
        status = "succeeded"
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        total_tokens = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        call = RecordedCall(
            task=task,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            status=status,
            error=None,
            request={"system": system, "user": attempt_user, **kwargs},
            response={"text": result.text, "finish_reason": result.finish_reason},
            latency_ms=latency_ms,
        )
    else:
        # Keep the record on a terminal failure as a failed call so the audit
        # table still shows the attempt and its error, not a silent gap.
        call = RecordedCall(
            task=task,
            model=model,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            status="failed",
            error=f"{type(error).__name__}: {error}" if error is not None else None,
            request={"system": system, "user": attempt_user, **kwargs},
            response=None,
            latency_ms=latency_ms,
        )
    recorder.record(call)


def invoke(system: str, user: str, **kwargs: Any) -> InvokeResult:
    """Invoke the active LLM backend with shared retry."""
    try:
        backend = get_backend()
    except Exception as e:
        raise InvokeError(
            "llm",
            f"backend selection failed: {e}",
            cause=e if isinstance(e, Exception) else None,
            retryable=False,
        )

    raw_kwargs = dict(kwargs)
    max_attempts = _invoke_max_attempts(raw_kwargs.pop("max_attempts", None))
    trace_name = raw_kwargs.pop("trace_name", None)
    if trace_name is not None:
        trace_name = str(trace_name)
    # Tracing bookkeeping only; no backend ever sees it.
    logs_root = raw_kwargs.pop("logs_root", None)
    invoke_kwargs = _normalise_kwargs_for_backend(backend, raw_kwargs)
    if backend == "react" and trace_name is not None and invoke_kwargs.get("tools"):
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.tracing import trace_result_store_path

        # Oversized tool results are archived beside this call's trace
        # (logs_root), never under cwd — a store inside the workspace once
        # got mistaken for a scope directory.
        invoke_kwargs["result_store"] = str(trace_result_store_path(trace_name, logs_root=logs_root))

    backend_invoke = _backend_invoke(backend)
    last_err: InvokeError | None = None
    attempt_user = user
    attempt_kwargs = dict(invoke_kwargs)
    recorder = current_call_recorder()
    # The model is resolved by the backend from its own config resolver (which
    # honours the LLM_MODEL env override the host wrapper injects). surface the
    # resolved id on the record so the audit table shows the model actually
    # used, not the caller's maybe-absent kwarg.
    resolved_model = kwargs.get("model")
    if not resolved_model:
        try:
            resolved_model = setting_optional_str("llm.model")
        except Exception:
            resolved_model = None
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        result: InvokeResult | None = None
        try:
            result = backend_invoke(
                system,
                attempt_user,
                **attempt_kwargs,
            )
            if result.permission_denials:
                denied_tools = sorted({str(item.get("tool_name") or "unknown") for item in result.permission_denials})
                denied_inputs = [item.get("tool_input", {}) for item in result.permission_denials[:3]]
                raise InvokeError(
                    backend,
                    f"tool permission denied during invocation: tools={denied_tools}, inputs={denied_inputs}",
                    retryable=True,
                    resume_session=True,
                )
            final_result = replace(
                result,
                attempts=attempt,
            )
            result = final_result
            write_invoke_trace(
                trace_name=trace_name,
                backend=backend,
                system=system,
                user=attempt_user,
                kwargs=attempt_kwargs,
                result=final_result,
                error=None,
                attempt=attempt,
                max_attempts=max_attempts,
                duration_sec=time.perf_counter() - started,
                logs_root=logs_root,
            )
            _record_call(
                recorder,
                task=trace_name or "llm",
                model=resolved_model or "",
                attempt_user=attempt_user,
                system=system,
                kwargs=attempt_kwargs,
                result=final_result,
                error=None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return final_result
        except InvokeError as e:
            if result is None and e.partial_result is not None:
                result = e.partial_result
            if result is not None:
                if not e.session_id:
                    e.session_id = result.session_id
                if e.partial_result is None:
                    e.partial_result = result
            e.attempts = attempt
            last_err = e
            write_invoke_trace(
                trace_name=trace_name,
                backend=backend,
                system=system,
                user=attempt_user,
                kwargs=attempt_kwargs,
                result=result,
                error=e,
                attempt=attempt,
                max_attempts=max_attempts,
                duration_sec=time.perf_counter() - started,
                logs_root=logs_root,
            )
            if not e.retryable or attempt >= max_attempts:
                _record_call(
                    recorder,
                    task=trace_name or "llm",
                    model=resolved_model or "",
                    attempt_user=attempt_user,
                    system=system,
                    kwargs=attempt_kwargs,
                    result=result,
                    error=e,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                raise
            attempt_user, attempt_kwargs = prepare_retry(
                original_user=user,
                current_kwargs=attempt_kwargs,
                result=result,
                error=e,
            )
        except Exception as e:
            last_err = InvokeError(
                backend,
                f"backend invocation failed: {e}",
                cause=e if isinstance(e, Exception) else None,
                retryable=True,
                session_id=result.session_id if result is not None else None,
                partial_result=result,
            )
            last_err.attempts = attempt
            write_invoke_trace(
                trace_name=trace_name,
                backend=backend,
                system=system,
                user=attempt_user,
                kwargs=attempt_kwargs,
                result=result,
                error=last_err,
                attempt=attempt,
                max_attempts=max_attempts,
                duration_sec=time.perf_counter() - started,
                logs_root=logs_root,
            )
            if attempt >= max_attempts:
                _record_call(
                    recorder,
                    task=trace_name or "llm",
                    model=resolved_model or "",
                    attempt_user=attempt_user,
                    system=system,
                    kwargs=attempt_kwargs,
                    result=result,
                    error=last_err,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                raise last_err
            attempt_user, attempt_kwargs = prepare_retry(
                original_user=user,
                current_kwargs=attempt_kwargs,
                result=result,
                error=last_err,
            )

    # Unreachable in practice; kept for type checkers.
    raise InvokeError(
        backend,
        f"backend invocation failed after {max_attempts} attempts: {last_err}",
        cause=last_err,
    )


__all__ = [
    "BackendName",
    "CallRecorder",
    "RecordedCall",
    "bind_call_recorder",
    "current_call_recorder",
    "get_backend",
    "get_default_cwd",
    "invoke",
    "normalise_backend",
    "use_backend",
]
