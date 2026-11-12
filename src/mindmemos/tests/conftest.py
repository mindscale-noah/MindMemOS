"""Shared pytest fixtures and options for the MindMemOS test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.nodes import Item


def pytest_addoption(parser: "Parser") -> None:
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Enable live LLM tests that call external model APIs. "
        "Without this flag, all tests marked 'llm' are unconditionally skipped "
        "to avoid costs, rate limits, and CI flakiness.",
    )


def pytest_collection_modifyitems(config: "Config", items: list["Item"]) -> None:
    """Auto-skip tests marked 'llm' unless --run-llm is explicitly passed."""
    if config.getoption("--run-llm"):
        return  # explicitly opted in — let them run

    skip_llm = pytest.mark.skip(reason="need --run-llm to run live LLM tests")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip_llm)
