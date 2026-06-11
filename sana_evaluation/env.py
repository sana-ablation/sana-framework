"""Environment loading helpers for local evaluation runs."""

from __future__ import annotations

from pathlib import Path


def load_repo_dotenv(root: str | Path = ".") -> None:
    """Load ``.env`` from ``root`` without overriding explicit environment."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv(Path(root) / ".env", override=False)
