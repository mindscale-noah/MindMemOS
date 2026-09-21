"""Exact SpreadsheetBench reference prompt resources."""

from importlib.resources import files

ROUTING_SYSTEM_PROMPT = (
    files(__package__).joinpath("tree_only_skill_routing_system_prompt.txt").read_text(encoding="utf-8").rstrip()
)
"""Pinned spreadsheet subtree-routing instructions, with reference trailing whitespace handling."""
