"""Choose built-in prompt adaptation by the execution environment's registered name."""

from .....envs.base import Environment
from .....logging import get_logger
from ..env_adapter import SkillForestEnvAdapter
from ..registry import BUILTIN_ADAPTER_NAMES

logger = get_logger(__name__)


def select_env_adapter(env: Environment) -> SkillForestEnvAdapter:
    """Create and log independent adaptation for one training call.

    Args:
        env: Actual execution environment supplied by the caller.

    Returns:
        The same-name built-in adapter, or generic adaptation for other environments.
    """
    try:
        environment_name = Environment.name_for(type(env))
    except ValueError:
        environment_name = f"{type(env).__module__}.{type(env).__qualname__}"
    adapter_name = environment_name if environment_name in BUILTIN_ADAPTER_NAMES else "generic"
    adapter = SkillForestEnvAdapter.build(adapter_name, {})
    logger.info(f"adapter.selected | algorithm=skill_forest environment={environment_name} adapter={adapter_name}")
    return adapter
