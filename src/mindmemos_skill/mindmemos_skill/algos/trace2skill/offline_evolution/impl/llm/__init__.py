"""Backend-neutral facade for local agent CLI invocations."""

from .backend import (
    BackendName,
    get_backend,
    invoke,
    normalise_backend,
    stage_tools,
    use_backend,
)
from .invocation import InvokeError, InvokeResult

__all__ = [
    "InvokeResult",
    "InvokeError",
    "BackendName",
    "get_backend",
    "invoke",
    "normalise_backend",
    "stage_tools",
    "use_backend",
]
