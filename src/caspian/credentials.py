"""Secret lookup. Values are returned to callers and never printed or logged.

On Windows, ``setx`` writes user environment variables to the registry but does
not update processes that are already running (for example an editor started
before the variable was set). We therefore fall back to the user-scope registry
value when the process environment lacks the variable.
"""

from __future__ import annotations

import os
import sys


def _from_windows_user_env(name: str) -> str | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value) if value else None
    except OSError:
        return None


def get_secret(name: str) -> str | None:
    """Return the secret ``name`` from the process environment or, on Windows,
    the user-scope environment in the registry. Returns None if absent."""
    value = os.environ.get(name)
    if value:
        return value
    return _from_windows_user_env(name)


def earthdata_token() -> str | None:
    return get_secret("EARTHDATA_TOKEN")


def describe(value: str | None) -> str:
    """Safe description of a secret for logs: presence and length only."""
    return "absent" if not value else f"present ({len(value)} chars)"
