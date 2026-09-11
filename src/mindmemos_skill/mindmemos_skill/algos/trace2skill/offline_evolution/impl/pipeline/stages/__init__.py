"""Stage package: importing it registers every concrete Stage via ``@register_stage``."""

from . import (  # noqa: F401 -- side effect: register every concrete Stage
    evolve,
    extract,
    finalize,
    pattern,
    staging,
)
