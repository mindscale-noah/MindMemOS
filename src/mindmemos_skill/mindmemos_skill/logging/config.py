"""Explicit console and file logging configuration for this package."""

import logging
from pathlib import Path

from .records import ContextFormatter, RecordFilter

_handlers: list[logging.Handler] = []
_capture_handlers: list[logging.Handler] = []
_level_without_capture: int | None = None


def attach_capture_handler(handler: logging.Handler) -> None:
    """Forward package logs to a capture handler without changing console filtering.

    Args:
        handler: Task or OpenTelemetry logging handler.
    """
    global _level_without_capture
    logger = get_logger()
    if not _capture_handlers:
        _level_without_capture = logger.level
    handler.addFilter(RecordFilter())
    logger.addHandler(handler)
    _capture_handlers.append(handler)
    logger.setLevel(logging.DEBUG)


def detach_capture_handler(handler: logging.Handler) -> None:
    """Remove one capture handler and restore the configured package level.

    Args:
        handler: Previously attached capture handler.
    """
    global _level_without_capture
    logger = get_logger()
    logger.removeHandler(handler)
    _capture_handlers.remove(handler)
    handler.close()
    if not _capture_handlers:
        logger.setLevel(_level_without_capture if _level_without_capture is not None else logging.NOTSET)
        _level_without_capture = None


def get_logger(name: str = "mindmemos_skill") -> logging.Logger:
    """Return a logger within the package namespace without configuring it.

    Args:
        name: Module name or short component name. Short names are prefixed
            with ``mindmemos_skill.``.

    Returns:
        A standard logger inheriting the package logging configuration.
    """
    if name != "mindmemos_skill" and not name.startswith("mindmemos_skill."):
        name = f"mindmemos_skill.{name}"
    return logging.getLogger(name)


def configure_logging(*, level: str | int = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure package logging at application startup.

    Console output goes to stderr. Reconfiguration replaces and closes only
    handlers created here. The root logger is unchanged. Call this before
    starting worker threads; concurrent reconfiguration is not supported.

    Args:
        level: Standard logging level name or integer threshold for both outputs.
        log_file: Optional UTF-8 append-only log file. Missing parent directories
            are created. None enables console output only.

    Returns:
        The configured package logger, with propagation to the root disabled.

    Raises:
        ValueError: If level is not a recognized name or integer.
        OSError: If the log directory or file cannot be created or opened.
    """
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper())
    if not isinstance(level, int):
        raise ValueError("level must be a logging level name or integer")

    formatter = ContextFormatter()
    handlers: list[logging.Handler] = []
    if log_file is not None:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))
    handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.addFilter(RecordFilter())
        handler.setFormatter(formatter)
        handler.setLevel(level)

    logger = get_logger()
    for handler in _handlers:
        logger.removeHandler(handler)
        handler.close()
    _handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)
    _handlers.extend(handlers)
    global _level_without_capture
    if _capture_handlers:
        _level_without_capture = level
    logger.setLevel(logging.DEBUG if _capture_handlers else level)
    logger.propagate = False
    return logger
