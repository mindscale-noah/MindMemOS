"""Workspace-scoped control sockets for stopping only launcher-owned processes."""

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from uuid import uuid4


def _directory(root: Path) -> Path:
    """Keep Unix socket paths short and isolate workspaces and operating-system users."""
    identity = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    path = Path("/tmp") / f"mindmemos-dev-{os.getuid()}-{identity}"
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise RuntimeError("Unsafe development control directory")
    path.chmod(0o700)
    return path


class ControlSession:
    """Serve status and stop requests without storing or trusting reusable process IDs."""

    def __init__(self, root: Path, summary: str):
        """Bind a private control socket.

        Args:
            root: Workspace owning the application processes.
            summary: Non-secret application addresses and selected features.
        """
        self.path = _directory(root) / f"{uuid4().hex[:12]}.sock"
        self.summary = summary
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(self.path))
        self.path.chmod(0o600)
        self.socket.listen(4)
        self.socket.setblocking(False)

    def poll(self, ready: bool) -> bool:
        """Handle one pending request.

        Args:
            ready: Whether all selected applications have passed their startup probes.

        Returns:
            True when an authorized local client requests shutdown.
        """
        try:
            connection, _ = self.socket.accept()
        except BlockingIOError:
            return False
        with connection:
            connection.settimeout(0.2)
            try:
                command = connection.recv(32).decode().strip()
                if command not in {"status", "stop"}:
                    return False
                connection.sendall(json.dumps({"ready": ready, "summary": self.summary}).encode())
                return command == "stop"
            except (OSError, UnicodeError):
                return False

    def close(self) -> None:
        """Remove this run's private control endpoint after child cleanup."""
        self.socket.close()
        self.path.unlink(missing_ok=True)


def request_sessions(root: Path, command: str) -> list[dict]:
    """Read or stop active runs belonging to this workspace.

    Args:
        root: Workspace whose managed processes should be addressed.
        command: Either status or stop; no arbitrary commands are accepted.

    Returns:
        Responses containing readiness and non-secret startup summaries.

    Raises:
        RuntimeError: A managed launcher cannot be contacted or stopped in time.
    """
    if command not in {"status", "stop"}:
        raise ValueError("Unsupported control command")
    responses = []
    stopping = []
    for path in _directory(root).glob("*.sock"):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            try:
                client.connect(str(path))
                client.sendall(command.encode())
                chunks = []
                while chunk := client.recv(4096):
                    chunks.append(chunk)
                responses.append(json.loads(b"".join(chunks)))
                stopping.append(path)
            except (ConnectionRefusedError, FileNotFoundError):
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise RuntimeError("Cannot contact a managed launcher; retry after startup completes") from exc
    if command == "stop":
        deadline = time.monotonic() + 20
        while any(path.exists() for path in stopping):
            if time.monotonic() > deadline:
                raise RuntimeError("Applications are still stopping; Docker has not been stopped")
            time.sleep(0.1)
    return responses
