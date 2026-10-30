"""Paths and default settings."""

import os
import sys
from pathlib import Path

APP_NAME = "SelectorModelosIA"
DEFAULT_REFRESH_HOURS = 6   # catalogs change daily: refresh at least every 6 h
HTTP_TIMEOUT_SECONDS = 30.0
USER_AGENT = "selector-modelos-ia/0.1"


def data_dir() -> Path:
    """Per-user folder for the SQLite cache. Override with SELECTOR_IA_DATA_DIR."""
    override = os.environ.get("SELECTOR_IA_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        path = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        path = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "selector-modelos-ia"
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_dir() / "cache.sqlite3"
