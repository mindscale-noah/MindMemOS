"""Physical table names owned by the observability backend."""

from typing import Final

OBSERVABILITY_RUNS_TABLE: Final[str] = "observability_runs"
"""One algorithm run and its latest execution state."""

OBSERVABILITY_SPANS_TABLE: Final[str] = "observability_spans"
"""Completed OpenTelemetry spans keyed by trace and span identity."""

OBSERVABILITY_LOGS_TABLE: Final[str] = "observability_logs"
"""Trace-correlated structured audit records."""
