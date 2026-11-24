"""Fingerprint official evaluator code and validate public asset paths."""

from hashlib import sha256
from pathlib import Path, PurePosixPath


def relative_path(value: str) -> str:
    """Require a normalized relative path contained in a public workspace.

    Args:
        value: Benchmark-provided destination path.

    Returns:
        Validated POSIX path.
    """
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value:
        raise ValueError(f"Unsafe benchmark path: {value!r}")
    if not path.parts or path.parts[0].startswith("."):
        raise ValueError(f"Reserved benchmark path: {value!r}")
    return value


def evaluator_digest(root: Path) -> str:
    """Hash official evaluator sources without importing the upstream package.

    Args:
        root: Official repository checkout containing benchmark/.

    Returns:
        SHA-256 binding relative filenames and source bytes.
    """
    benchmark = root / "benchmark"
    if not (benchmark / "AgentSkillOS_bench" / "orchestrator.py").is_file():
        raise ValueError("benchmark_root must contain the official AgentSkillOS benchmark evaluator")
    digest = sha256()
    for path in sorted(benchmark.rglob("*.py")):
        if path.is_symlink():
            raise ValueError("Evaluator source must not contain symlinks")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
