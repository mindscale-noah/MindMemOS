"""Check version boundaries and expression semantics of the TextWorld shim."""

import sys
from types import ModuleType

import pytest
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history import python_compat


def grammar_helper(value):
    """Represent a helper from the upstream evaluator namespace."""
    return value.upper()


@pytest.fixture
def textgen(monkeypatch):
    module = ModuleType("textworld.envs.pddl.textgen")

    class TerminalSymbol:
        def __init__(self, value):
            self.value = value

    class EvalSymbol:
        def __init__(self, expression, context):
            self.expression = expression
            self.context = context

        def derive(self, context=None):
            raise AssertionError("Upstream method called")

    module.TerminalSymbol = TerminalSymbol
    module.EvalSymbol = EvalSymbol
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(python_compat.sys, "version_info", (3, 13))
    monkeypatch.setattr(python_compat, "version", lambda name: "1.7.0")
    return module


def test_patch_resolves_variables_and_preserves_upstream_globals(textgen):
    python_compat.patch_textworld_locals()
    symbol = textgen.EvalSymbol("grammar_helper(r)", {"variables": {"r": "ok"}})
    assert symbol.derive()[0].value == "OK"
    assert symbol.derive({"variables": {"r": "override"}})[0].value == "OVERRIDE"
    symbol.expression = "self.expression == context['variables']['expected']"
    assert symbol.derive({"variables": {"expected": symbol.expression}})[0].value is True
    patched = textgen.EvalSymbol.derive
    python_compat.patch_textworld_locals()
    assert textgen.EvalSymbol.derive is patched


@pytest.mark.parametrize("python_version,textworld_version", [((3, 12), "1.7.0"), ((3, 13), "1.7.1")])
def test_unaffected_versions_keep_upstream_method(textgen, monkeypatch, python_version, textworld_version):
    monkeypatch.setattr(python_compat.sys, "version_info", python_version)
    monkeypatch.setattr(python_compat, "version", lambda name: textworld_version)
    original = textgen.EvalSymbol.derive
    python_compat.patch_textworld_locals()
    assert textgen.EvalSymbol.derive is original
