"""Record an existing development API process without putting shell logic in Make."""

import os
import shutil
import subprocess
from pathlib import Path

from . import constants as env
from .config import prepare_environment


def main() -> None:
    """Resolve the requested process and record a CPU flame graph."""
    root = Path(__file__).resolve().parents[2]
    prepare_environment(root / ".env")
    pid = os.environ.get(env.INPUT_PROFILE_PID)
    if not pid:
        port = os.environ.get(env.INPUT_MEMORY_PORT) or os.environ.get(env.MEMORY_PORT) or "8000"
        result = subprocess.run(["lsof", f"-tiTCP:{int(port)}", "-sTCP:LISTEN"], capture_output=True, text=True)
        matches = result.stdout.split()
        if len(matches) != 1:
            raise SystemExit("Specify PID=<process-id>; no unique Memory listener was found.")
        pid = matches[0]
    output = Path(os.environ.get(env.INPUT_PROFILE_OUT) or "tmp/profile.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("py-spy") or str(root / ".venv/bin/py-spy")
    subprocess.run(
        [
            executable,
            "record",
            "--pid",
            str(int(pid)),
            "--subprocesses",
            "--duration",
            str(int(os.environ.get(env.INPUT_PROFILE_SECONDS) or 60)),
            "--rate",
            str(int(os.environ.get(env.INPUT_PROFILE_RATE) or 100)),
            "--output",
            str(output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
