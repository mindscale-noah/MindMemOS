"""Open a host-native folder chooser in an isolated process."""

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Lock

from .constants import DIRECTORY_PICKER_INITIAL_ENV

_picker_lock = Lock()

_MAC_SCRIPT = """on run argv
    try
        set chosenFolder to choose folder with prompt "Select a folder" default location (POSIX file (item 1 of argv))
        return POSIX path of chosenFolder
    on error number -128
        return ""
    end try
end run"""

_WINDOWS_SCRIPT = (
    """
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select a folder'
$dialog.SelectedPath = [Environment]::GetEnvironmentVariable('%s')
$dialog.ShowNewFolderButton = $true
try {
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Write($dialog.SelectedPath)
    }
} finally {
    $dialog.Dispose()
}
"""
    % DIRECTORY_PICKER_INITIAL_ENV
)


def choose_directory(initial: Path) -> Path | None:
    """Ask the desktop user to select a directory without changing application state.

    Args:
        initial: Preferred initial location; use its nearest existing parent if absent.

    Returns:
        The selected absolute directory, or None when the user cancels.

    Raises:
        RuntimeError: The native dialog is unavailable, busy or failed.
    """
    if not _picker_lock.acquire(blocking=False):
        raise RuntimeError("A directory chooser is already open")
    try:
        initial = initial.expanduser().resolve()
        while not initial.is_dir() and initial != initial.parent:
            initial = initial.parent
        environment = os.environ.copy()
        if sys.platform == "darwin":
            command = ["/usr/bin/osascript", "-e", _MAC_SCRIPT, str(initial)]
            cancelled = set()
        elif sys.platform == "win32":
            environment[DIRECTORY_PICKER_INITIAL_ENV] = str(initial)
            encoded = base64.b64encode(_WINDOWS_SCRIPT.encode("utf-16-le")).decode("ascii")
            command = ["powershell.exe", "-NoProfile", "-STA", "-EncodedCommand", encoded]
            cancelled = set()
        elif shutil.which("zenity"):
            command = [
                "zenity",
                "--file-selection",
                "--directory",
                "--title=Select a folder",
                "--filename=" + str(initial) + os.sep,
            ]
            cancelled = {1}
        elif shutil.which("kdialog"):
            command = ["kdialog", "--getexistingdirectory", str(initial), "--title", "Select a folder"]
            cancelled = {1}
        else:
            command = [sys.executable, "-m", "mindmemos_ui.directory_picker", str(initial)]
            cancelled = set()
        try:
            result = subprocess.run(command, capture_output=True, encoding="utf-8", env=environment, timeout=300)
        except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
            raise RuntimeError("Native directory chooser is unavailable") from exc
        if result.returncode in cancelled and not result.stderr.strip():
            return None
        if result.returncode != 0:
            raise RuntimeError("Native directory chooser failed")
        selected = result.stdout.rstrip("\r\n")
        if not selected:
            return None
        path = Path(selected).resolve()
        if not path.is_dir():
            raise RuntimeError("Selected directory is no longer available")
        return path
    finally:
        _picker_lock.release()


def _tk_choose(initial: str) -> None:
    """Use Tk as a Linux fallback in its own main-thread process."""
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(parent=root, title="Select a folder", initialdir=initial, mustexist=True)
        sys.stdout.buffer.write(selected.encode("utf-8"))
    finally:
        root.destroy()


if __name__ == "__main__":
    _tk_choose(sys.argv[1])
