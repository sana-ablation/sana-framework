"""
Sandbox lifecycle helpers extracted from agent_runner.py.
"""

import shutil
import time
from pathlib import Path


def _create_isolated_sandbox(task_id: str) -> Path:
    """Create an isolated sandbox directory for a task."""
    sandbox_base = Path(__file__).resolve().parent.parent.parent / ".sandbox_isolated"
    sandbox_base.mkdir(parents=True, exist_ok=True)
    task_sandbox = sandbox_base / f"task_{task_id}_{int(time.time() * 1000)}"
    task_sandbox.mkdir(parents=True, exist_ok=True)
    return task_sandbox


def _cleanup_isolated_sandbox(sandbox_path: Path) -> None:
    """Clean up an isolated sandbox directory."""
    if sandbox_path and sandbox_path.exists():
        shutil.rmtree(sandbox_path)
