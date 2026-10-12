"""Behavioral checks for package logging."""

import logging
import subprocess
import sys

import pytest
from mindmemos_skill.logging import config, configure_logging, get_logger
from mindmemos_skill.logging.otel import attach_logs, detach_logs
from opentelemetry.sdk._logs import LoggerProvider, LogRecordProcessor


@pytest.fixture(autouse=True)
def restore_logging():
    """Restore package logging state and release files after each test."""
    logger = get_logger()
    old_level, old_propagate = logger.level, logger.propagate
    yield
    for handler in config._handlers:
        logger.removeHandler(handler)
        handler.close()
    config._handlers.clear()
    logger.setLevel(old_level)
    logger.propagate = old_propagate


def test_console_and_file_output(tmp_path, capsys):
    path = tmp_path / "nested" / "skill.log"
    configure_logging(log_file=path)
    logger = get_logger("algos")
    logger.debug("filtered message")
    logger.info("Unicode: \u4f60\u597d")
    try:
        raise RuntimeError("example failure")
    except RuntimeError:
        logger.exception("operation failed")

    console = capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == console
    assert "[INFO] [mindmemos_skill.algos] | Unicode: \u4f60\u597d" in console
    assert "filtered message" not in console
    assert "Traceback" in console
    assert "RuntimeError: example failure" in console


def test_reconfiguration_appends_without_duplicates(tmp_path, capsys):
    path = tmp_path / "skill.log"
    logger = configure_logging(log_file=path)
    logger.info("first record")
    old_file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    configure_logging(level="debug", log_file=path)
    logger.debug("second record")
    assert old_file_handler.stream is None
    text = path.read_text(encoding="utf-8")
    assert text.count("first record") == 1
    assert text.count("second record") == 1
    assert capsys.readouterr().err.count("second record") == 1


def test_switching_files_and_disabling_file_output(tmp_path, capsys):
    first, second = tmp_path / "first.log", tmp_path / "second.log"
    logger = configure_logging(log_file=first)
    logger.info("first destination")
    configure_logging(log_file=second)
    logger.info("second destination")
    configure_logging()
    logger.info("console only")
    assert "second destination" not in first.read_text()
    assert "console only" not in second.read_text()
    assert "console only" in capsys.readouterr().err


def test_root_and_external_handlers_are_preserved():
    root = logging.getLogger()
    root_handlers, root_level = root.handlers[:], root.level
    logger = get_logger()
    external = logging.NullHandler()
    logger.addHandler(external)
    try:
        configure_logging()
        configure_logging()
        assert external in logger.handlers
        assert logger.propagate is False
        assert root.handlers == root_handlers
        assert root.level == root_level
        assert get_logger("mindmemos_skill.algos") is get_logger("algos")
    finally:
        logger.removeHandler(external)
        external.close()


def test_failed_configuration_preserves_existing_output(tmp_path):
    logger = configure_logging()
    handlers = logger.handlers[:]
    with pytest.raises(ValueError):
        configure_logging(level="invalid", log_file=tmp_path / "unused.log")
    assert not (tmp_path / "unused.log").exists()
    with pytest.raises(OSError):
        configure_logging(log_file=tmp_path)
    assert logger.handlers == handlers


def test_otel_bridge_keeps_console_level_independent(capsys):
    """The OTel handler receives debug records while INFO console output stays filtered."""
    observed = []

    class CaptureLogs(LogRecordProcessor):
        """Collect emitted OTel log records."""

        def on_emit(self, record):
            """Save the emitted record."""
            observed.append(record.log_record)

        def shutdown(self):
            """Release no resources."""

        def force_flush(self, timeout_millis=30000):
            """Report synchronous completion."""
            return True

    provider = LoggerProvider(shutdown_on_exit=False)
    provider.add_log_record_processor(CaptureLogs())
    handler = attach_logs(provider, context=lambda: {"algorithm.run_id": "run"}, capture_level="debug")
    try:
        configure_logging(level="INFO")
        get_logger("algos").debug("debug event")
        assert [item.event_name for item in observed] == ["python.log"]
        assert observed[0].attributes["algorithm.run_id"] == "run"
        assert "debug event" not in capsys.readouterr().err
    finally:
        detach_logs(handler)
        provider.shutdown()


def test_import_has_no_logging_side_effects():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
root = logging.getLogger()
package = logging.getLogger('mindmemos_skill')
before = (root.handlers[:], root.level, package.handlers[:], package.level, package.propagate)
import mindmemos_skill.logging
after = (root.handlers[:], root.level, package.handlers[:], package.level, package.propagate)
assert before == after
""",
        ],
        check=True,
    )
