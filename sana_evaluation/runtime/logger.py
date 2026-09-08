"""
Logging helpers for the Strands evaluation runner.

Extracted from agent_runner.py.
"""

import atexit
import logging
import os
import re
import traceback
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)  # "sana_evaluation.runtime.logger" — never configured here


# ---------------------------------------------------------------------------
# Async-teardown noise suppression
# ---------------------------------------------------------------------------
#
# On Python 3.14, the OpenAI client's async HTTP streams fail to close cleanly:
# httpcore2's PoolByteStream.__aiter__ is thrown into at teardown and does not
# stop, so contextlib raises RuntimeError("generator didn't stop after
# athrow()"). asyncio's exception handler logs the whole traceback through
# logging.getLogger("asyncio"), and a real OpenAI run drowned in it -- 209 of
# 720 log lines (29%) across 21 tracebacks, with zero frames in
# sana_evaluation. Bedrock runs are unaffected (botocore is synchronous).
#
# The filter below drops *only* that signature. It deliberately does not
# silence the asyncio logger, and it counts what it drops so a genuine asyncgen
# bug cannot become undiscoverable: every run that suppressed anything says so
# in one summary line at exit.

_ASYNCGEN_CLOSE_MESSAGE = "an error occurred during closing of asynchronous generator"
_ASYNCGEN_ATHROW_SIGNATURE = "generator didn't stop after athrow()"


def _record_exception_text(record: logging.LogRecord) -> str:
    """Formatted exception attached to a record, or "" if there is none."""
    if record.exc_text:
        return record.exc_text
    if not record.exc_info:
        return ""
    exc_info = record.exc_info
    if exc_info is True or not isinstance(exc_info, tuple) or len(exc_info) != 3:
        return ""
    try:
        return "".join(traceback.format_exception(*exc_info))
    except Exception:  # pragma: no cover - formatting must never break logging
        return ""


class AsyncgenTeardownFilter(logging.Filter):
    """Drop the httpcore/httpx asyncgen-close tracebacks, and count them.

    A record is dropped only when all of these hold:

    - it was logged through the ``asyncio`` logger,
    - at ERROR or above,
    - its message is asyncio's asyncgen-close report, and
    - the athrow signature appears in the message or the attached traceback.

    Anything else on the ``asyncio`` logger -- including a *different* asyncgen
    failure -- passes through untouched.
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.suppressed = 0

    def matches(self, record: logging.LogRecord) -> bool:
        if record.name != "asyncio" or record.levelno < logging.ERROR:
            return False
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - a broken record is not ours to drop
            return False
        if _ASYNCGEN_CLOSE_MESSAGE not in message:
            return False
        if _ASYNCGEN_ATHROW_SIGNATURE in message:
            return True
        return _ASYNCGEN_ATHROW_SIGNATURE in _record_exception_text(record)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.matches(record):
            return True
        self.suppressed += 1
        return False

    def reset(self) -> int:
        count, self.suppressed = self.suppressed, 0
        return count


asyncgen_teardown_filter = AsyncgenTeardownFilter()

_ASYNCGEN_SUMMARY_REGISTERED = False


def install_asyncgen_teardown_filter(logger_name: str = "asyncio") -> AsyncgenTeardownFilter:
    """Attach the teardown filter to the asyncio logger. Idempotent."""
    global _ASYNCGEN_SUMMARY_REGISTERED
    target = logging.getLogger(logger_name)
    if asyncgen_teardown_filter not in target.filters:
        target.addFilter(asyncgen_teardown_filter)
    if not _ASYNCGEN_SUMMARY_REGISTERED:
        # Registered after the logging module's own shutdown hook, so atexit's
        # LIFO order runs this first, while handlers are still open.
        atexit.register(report_suppressed_asyncgen_errors)
        _ASYNCGEN_SUMMARY_REGISTERED = True
    return asyncgen_teardown_filter


def report_suppressed_asyncgen_errors(*, reset: bool = True) -> int:
    """Emit one summary line naming how many teardown errors were dropped."""
    count = asyncgen_teardown_filter.suppressed
    if count:
        logger.info(
            "suppressed %d asyncgen close error%s during teardown "
            "(httpcore/httpx stream close on Python 3.14; not a sana_evaluation fault)",
            count,
            "" if count == 1 else "s",
        )
    if reset:
        asyncgen_teardown_filter.reset()
    return count


def _slugify(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())


def _safe_path_part(value: Optional[str]) -> str:
    slug = _slugify(value)
    if not slug or slug in {".", ".."}:
        return "_"
    return slug


def _uses_mode_layout(condition: Optional[str]) -> bool:
    normalized = str(condition or "").replace("\\", "/").lstrip("./")
    return normalized == "modes" or normalized.startswith("modes/")


def _build_log_file(
    log_dir: str,
    condition: Optional[str],
    model: Optional[str],
    task_id: Optional[str],
) -> str:
    from pathlib import Path
    # Default layout: logs/{condition}/{model}/{task_dir}/{task_stem}.log
    # Mode runs can also provide a precomposed condition path like:
    # logs/modes/{model}/{variant}/{task_dir}/{task_stem}.log
    subdir = log_dir
    if condition:
        # Allow hierarchical condition labels like "naive_k5/baseline".
        for part in str(condition).replace("\\", "/").split("/"):
            if part and part != ".":
                subdir = os.path.join(subdir, _safe_path_part(part))
    if model and not _uses_mode_layout(condition):
        subdir = os.path.join(subdir, _safe_path_part(model))
    if task_id:
        parts = [
            part
            for part in str(task_id).replace("\\", "/").split("/")
            if part and part != "."
        ]
        if len(parts) > 1:
            for part in parts[:-1]:
                subdir = os.path.join(subdir, _safe_path_part(part))
            filename = Path(parts[-1]).stem + ".log"
        else:
            filename = Path(parts[0]).stem + ".log" if parts else "task.log"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_{timestamp}_pid{os.getpid()}.log"
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, filename)


def _replace_handlers(target: logging.Logger, handlers: list[logging.Handler]) -> None:
    old_handlers = list(target.handlers)
    target.handlers = []
    for handler in handlers:
        target.addHandler(handler)
    for handler in old_handlers:
        if handler not in handlers:
            handler.close()


def configure_logging(
    log_dir: str = "logs",
    run_id: Optional[str] = None,
    model: Optional[str] = None,
    batch: Optional[str] = None,
    condition: Optional[str] = None,
    task_id: Optional[str] = None,
    level: int = logging.DEBUG,
) -> None:
    """Configure file + console handlers on the strands and sana_evaluation root loggers.

    Call once per process before creating any Agent. Worker processes call this inside
    _run_task_worker before constructing DataLakeAgent.
    """
    log_file = _build_log_file(log_dir, condition, model, task_id)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))

    # Application logger — full verbosity to file + console
    app_root = logging.getLogger("sana_evaluation")
    app_root.setLevel(level)
    _replace_handlers(app_root, [file_handler, console_handler])
    app_root.propagate = False

    # SDK logger — WARNING only (suppresses tool registry, plugin discovery, request body dumps)
    sdk_root = logging.getLogger("strands")
    sdk_root.setLevel(logging.WARNING)
    _replace_handlers(sdk_root, [file_handler])  # warnings/errors still go to file
    sdk_root.propagate = False

    # Re-enable retry INFO logs from SDK (throttle/retry events are useful)
    logging.getLogger("strands.event_loop._retry").setLevel(logging.INFO)

    # Drop the OpenAI-client asyncgen teardown tracebacks (counted, and
    # summarised at exit). Every other asyncio record still gets through.
    install_asyncgen_teardown_filter()

    logger.info(f"Logging to: {log_file}")


def configure_worker_logging(
    run_config,
    *,
    model: Optional[str] = None,
    condition: Optional[str] = None,
    task_id: Optional[str] = None,
    level: int = logging.DEBUG,
) -> None:
    """Configure per-task logging using the run-configured log root."""
    configure_logging(
        log_dir=getattr(run_config, "logs_output_dir", "logs") or "logs",
        model=model,
        condition=condition,
        task_id=task_id,
        level=level,
    )
