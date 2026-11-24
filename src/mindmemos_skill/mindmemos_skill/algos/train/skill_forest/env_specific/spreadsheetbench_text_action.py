"""Spreadsheet text-action evidence uses the shared public spreadsheet renderer."""

from ..env_adapter import SkillForestEnvAdapter
from .spreadsheetbench_tool_calling import SkillForestSpreadsheetBenchToolCallingEnvAdapter


@SkillForestEnvAdapter.register("spreadsheetbench_text_action")
class SkillForestSpreadsheetBenchTextActionEnvAdapter(SkillForestSpreadsheetBenchToolCallingEnvAdapter):
    """Render recorded text actions with spreadsheet task context."""
