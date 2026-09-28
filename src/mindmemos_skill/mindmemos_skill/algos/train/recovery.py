"""Source identity checks for resumable trainers."""

import hashlib
import inspect
from pathlib import Path

from ...envs.base import Environment


def implementation_identity(trainer, env) -> dict[str, str]:
    """Fingerprint algorithm, environment and subclass source resources.

    Args:
        trainer: Concrete trainer implementation owning split and proposal hooks.
        env: Concrete environment implementation.

    Returns:
        Source hashes preventing silent reuse after implementation changes.
    """
    try:
        environment_file = inspect.getfile(type(env))
    except (TypeError, OSError):
        environment_file = inspect.getfile(env._run if isinstance(env, Environment) else env.rollout)
    roots = {Path(__file__).parents[2], Path(environment_file).parent}
    result = {}
    for root in sorted(roots):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md"}:
                result[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = Path(inspect.getfile(type(trainer)))
    result[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
