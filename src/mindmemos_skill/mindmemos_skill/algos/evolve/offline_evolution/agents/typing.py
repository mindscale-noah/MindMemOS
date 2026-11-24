"""Backend-neutral file scopes and persistence for one offline agent invocation."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class SessionOptions(BaseModel):
    """Describe stage permissions, run-local agent configuration and event output."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    """Keep stage scopes immutable and reject undeclared fields."""

    config_dir: Path
    """Run-local backend configuration and persistent session directory."""

    read_dirs: tuple[Path, ...] = ()
    """Paths readable by the agent; write paths are readable as well."""

    write_dirs: tuple[Path, ...] = ()
    """Files or directory trees where this stage may write."""

    deny_paths: tuple[Path, ...] = ()
    """Host-owned paths unavailable to the agent's tools."""

    readonly_dirs: tuple[Path, ...] = ()
    """Read-only subtrees inside otherwise writable directories."""

    session_id: str | None = None
    """Observed native session or thread identity to resume."""

    log_path: Path | None = None
    """Raw event file retained even when execution is interrupted."""
