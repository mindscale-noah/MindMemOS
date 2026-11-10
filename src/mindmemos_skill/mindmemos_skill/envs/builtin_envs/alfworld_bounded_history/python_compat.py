"""Apply the known TextWorld fix for Python's independent locals snapshots."""

import sys
from importlib.metadata import version
from typing import Any


def patch_textworld_locals() -> None:
    """Fix expression evaluation for TextWorld 1.7.0 on Python 3.13 and newer.

    Call inside the simulator worker before creating any games. Other Python
    and TextWorld versions retain their upstream implementation. Repeated calls
    leave the installed patch unchanged.
    """
    if sys.version_info < (3, 13) or version("textworld") != "1.7.0":
        return

    from textworld.envs.pddl.textgen import EvalSymbol, TerminalSymbol

    original = EvalSymbol.derive
    if original.__module__ == __name__:
        return

    def derive(self: EvalSymbol, context: dict[str, Any] | None = None) -> list[TerminalSymbol]:
        """Evaluate with upstream globals and explicitly bound grammar variables.

        Args:
            context: Upstream grammar context, whose variables mapping binds
                expression names to entity information. Other entries retain
                the upstream grammar state. None uses the symbol's context.

        Returns:
            One upstream terminal symbol containing the evaluated value.
        """
        context = context or self.context
        namespace = {"self": self, "context": context}
        namespace.update(context["variables"])
        value = eval(self.expression, original.__globals__, namespace)
        return [TerminalSymbol(value)]

    EvalSymbol.derive = derive
