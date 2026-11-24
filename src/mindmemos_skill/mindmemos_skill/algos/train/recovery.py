"""Sample input checks for resumable trainers."""

from ...common.input_files import model_file_fingerprints
from ...envs.base import Environment
from ...envs.data import DataSample


def sample_identity(sample: DataSample, env) -> dict[str, object]:
    """Record materialized data and internally computed external input identities.

    Args:
        sample: Public task paired with its private scoring data.
        env: Environment implementation or a trainer test backend.

    Returns:
        A mapping with serialized sample data and private external file fingerprints.
    """
    identity = (
        env.input_identity(sample.task, sample.reward_data)
        if isinstance(env, Environment)
        else model_file_fingerprints(sample.task, sample.reward_data)
    )
    return {"data": sample.model_dump(mode="json"), "input_identity": identity}
