"""Recalculate one temporary workbook using LibreOffice's own Python runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run(pipe_name: str, workbook: Path, timeout: int) -> int:
    import uno
    from com.sun.star.beans import PropertyValue

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver",
        local_context,
    )
    deadline = time.monotonic() + timeout
    context = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            context = resolver.resolve(f"uno:pipe,name={pipe_name};urp;StarOffice.ComponentContext")
            break
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    if context is None:
        raise RuntimeError(f"could not connect to LibreOffice UNO pipe: {last_error}")

    service_manager = context.ServiceManager
    desktop = service_manager.createInstanceWithContext("com.sun.star.frame.Desktop", context)
    hidden = PropertyValue()
    hidden.Name = "Hidden"
    hidden.Value = True
    document = desktop.loadComponentFromURL(workbook.resolve().as_uri(), "_blank", 0, (hidden,))
    if document is None:
        raise RuntimeError(f"LibreOffice could not open workbook: {workbook}")
    try:
        document.calculateAll()
        document.store()
    finally:
        document.close(True)
        desktop.terminate()
    print(json.dumps({"status": "success", "workbook": str(workbook)}))
    return 0


def main() -> int:
    """Parse the internal worker invocation and run one UNO transaction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--pipe-name", required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    return run(args.pipe_name, args.workbook.resolve(), args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
