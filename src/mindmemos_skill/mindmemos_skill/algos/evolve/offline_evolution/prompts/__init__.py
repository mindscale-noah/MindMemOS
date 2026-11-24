"""Stage instructions kept as Markdown files next to this module."""

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent
"""Directory holding the stage prompt files."""

LENS_DIR = PROMPT_DIR / "lenses"
"""Preset perspectives inserted into the stage prompts, kept apart so they can change on their own."""


def render_prompt(name: str, **values: object) -> str:
    """Load a prompt file and fill its placeholders.

    Args:
        name: File stem: evidence, change or retry.
        values: Placeholder values; literal braces in the files are written as `{{ }}`.

    Returns:
        The filled prompt text.
    """
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").format(**values)


def lenses(stage: str) -> str:
    """Load the preset perspectives of a stage.

    Args:
        stage: evidence or change.

    Returns:
        Markdown text inserted verbatim into the stage prompt.
    """
    return (LENS_DIR / f"{stage}.md").read_text(encoding="utf-8").strip()
