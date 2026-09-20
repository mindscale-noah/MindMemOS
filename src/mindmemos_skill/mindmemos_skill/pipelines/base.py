"""Registration family for composable Skill business operations."""

from ..registry.base import ConfigT, Registrable


class Pipeline(Registrable[ConfigT]):
    """Compose algorithms, persistence services, and optional environments.

    Register concrete implementations with Pipeline.register, including
    evolution and training implementations. Constructor keyword dependencies
    are supplied by the existing service factory; this base requires no
    particular algorithm, database, or storage layout.

    Pipelines own input selection, materialization, algorithm invocation, and
    result persistence. Trainers and evolvers must receive detached data and
    must not access persistence through injected dependencies. Pipelines borrow
    shared backends and must not close them. Per-call state belongs to the call,
    not the shared pipeline instance. Concrete configuration belongs in the
    implementation's config.py; registration must not create resources.
    """
