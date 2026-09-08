"""The asyncgen-teardown filter drops one signature and nothing else.

The records here are built from a real OpenAI run's log (tmp/webarm-web.log),
where 209 of 720 lines were repeated
``RuntimeError: generator didn't stop after athrow()`` tracebacks raised inside
httpcore2's PoolByteStream and stdlib contextlib on Python 3.14 -- no frame in
sana_evaluation. asyncio reports them through ``logging.getLogger("asyncio")``,
which is where the filter sits.
"""

import contextlib
import logging
import unittest

from sana_evaluation.runtime.logger import (
    AsyncgenTeardownFilter,
    asyncgen_teardown_filter,
    install_asyncgen_teardown_filter,
    report_suppressed_asyncgen_errors,
)


_ASYNCGEN_REPR = "<async_generator object PoolByteStream.__aiter__ at 0x130134d40>"

# asyncio's default_exception_handler joins the message and the context keys
# into one string and passes the exception via exc_info.
_REAL_MESSAGE = (
    f"an error occurred during closing of asynchronous generator {_ASYNCGEN_REPR}\n"
    f"asyncgen: {_ASYNCGEN_REPR}"
)


def _athrow_exc_info():
    """The real exception: contextlib raising from a generator that won't stop."""
    try:
        raise RuntimeError("generator didn't stop after athrow()")
    except RuntimeError:
        import sys

        return sys.exc_info()


def _record(
    *,
    name="asyncio",
    level=logging.ERROR,
    msg=_REAL_MESSAGE,
    args=(),
    exc_info=None,
):
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )


class AsyncgenTeardownFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter = AsyncgenTeardownFilter()

    def test_the_real_teardown_record_is_dropped_and_counted(self):
        record = _record(exc_info=_athrow_exc_info())

        self.assertFalse(self.filter.filter(record), "the teardown traceback should be dropped")
        self.assertEqual(self.filter.suppressed, 1)

    def test_a_different_asyncio_error_survives(self):
        """A genuine asyncio failure must still reach the log."""
        record = _record(
            msg="Task exception was never retrieved",
            exc_info=_athrow_exc_info(),
        )

        self.assertTrue(self.filter.filter(record), "unrelated asyncio errors must survive")
        self.assertEqual(self.filter.suppressed, 0)

    def test_a_different_asyncgen_failure_survives(self):
        """Same asyncio message, different cause -- not our signature, keep it.

        This is the case that keeps a real asyncgen bug discoverable.
        """
        try:
            raise ValueError("boom in someone's aclose")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = _record(exc_info=exc_info)

        self.assertTrue(self.filter.filter(record), "a different asyncgen fault must survive")
        self.assertEqual(self.filter.suppressed, 0)

    def test_non_asyncio_loggers_are_untouched(self):
        record = _record(name="sana_evaluation.runner.batch", exc_info=_athrow_exc_info())

        self.assertTrue(self.filter.filter(record))
        self.assertEqual(self.filter.suppressed, 0)

    def test_a_warning_with_the_signature_survives(self):
        record = _record(level=logging.WARNING, exc_info=_athrow_exc_info())

        self.assertTrue(self.filter.filter(record))
        self.assertEqual(self.filter.suppressed, 0)

    def test_the_signature_inside_the_message_is_enough(self):
        """Some handlers pre-render exc_text into the message."""
        record = _record(msg=_REAL_MESSAGE + "\nRuntimeError: generator didn't stop after athrow()")

        self.assertFalse(self.filter.filter(record))
        self.assertEqual(self.filter.suppressed, 1)

    def test_records_with_args_are_formatted_before_matching(self):
        record = _record(
            msg="an error occurred during closing of asynchronous generator %s",
            args=(_ASYNCGEN_REPR,),
            exc_info=_athrow_exc_info(),
        )

        self.assertFalse(self.filter.filter(record))
        self.assertEqual(self.filter.suppressed, 1)


class AsyncgenSuppressionIsVisibleTest(unittest.TestCase):
    """The count is the mitigation: suppression must never be silent."""

    def test_the_filter_drops_records_through_the_real_asyncio_logger(self):
        asyncio_logger = logging.getLogger("asyncio")
        seen = []

        class Capture(logging.Handler):
            def emit(self, record):
                seen.append(record.getMessage())

        handler = Capture()
        asyncio_logger.addHandler(handler)
        previous = asyncgen_teardown_filter.reset()
        try:
            install_asyncgen_teardown_filter()
            asyncio_logger.error(_REAL_MESSAGE, exc_info=_athrow_exc_info())
            asyncio_logger.error("Task exception was never retrieved")

            self.assertEqual(seen, ["Task exception was never retrieved"])
            self.assertEqual(asyncgen_teardown_filter.suppressed, 1)
        finally:
            asyncio_logger.removeHandler(handler)
            asyncgen_teardown_filter.reset()
            asyncgen_teardown_filter.suppressed = previous

    def test_the_summary_line_names_the_count_and_resets(self):
        previous = asyncgen_teardown_filter.reset()
        try:
            asyncgen_teardown_filter.suppressed = 21

            with self.assertLogs("sana_evaluation.runtime.logger", level=logging.INFO) as captured:
                returned = report_suppressed_asyncgen_errors()

            self.assertEqual(returned, 21)
            self.assertEqual(len(captured.output), 1, "exactly one summary line")
            self.assertIn("suppressed 21 asyncgen close errors during teardown", captured.output[0])
            self.assertEqual(asyncgen_teardown_filter.suppressed, 0, "the count resets per run")
        finally:
            asyncgen_teardown_filter.suppressed = previous

    def test_a_clean_run_says_nothing(self):
        previous = asyncgen_teardown_filter.reset()
        try:
            logger = logging.getLogger("sana_evaluation.runtime.logger")
            seen = []

            class Capture(logging.Handler):
                def emit(self, record):
                    seen.append(record.getMessage())

            handler = Capture()
            logger.addHandler(handler)
            try:
                self.assertEqual(report_suppressed_asyncgen_errors(), 0)
            finally:
                logger.removeHandler(handler)

            self.assertEqual(seen, [], "a run that suppressed nothing must stay quiet")
        finally:
            asyncgen_teardown_filter.suppressed = previous

    def test_singular_wording_for_one_error(self):
        previous = asyncgen_teardown_filter.reset()
        try:
            asyncgen_teardown_filter.suppressed = 1
            with self.assertLogs("sana_evaluation.runtime.logger", level=logging.INFO) as captured:
                report_suppressed_asyncgen_errors()
            self.assertIn("suppressed 1 asyncgen close error during teardown", captured.output[0])
        finally:
            asyncgen_teardown_filter.suppressed = previous

    def test_installation_is_idempotent(self):
        asyncio_logger = logging.getLogger("asyncio")
        install_asyncgen_teardown_filter()
        install_asyncgen_teardown_filter()

        matches = [f for f in asyncio_logger.filters if isinstance(f, AsyncgenTeardownFilter)]
        self.assertEqual(len(matches), 1, "the filter must not stack up per configure_logging call")


class ConfigureLoggingInstallsTheFilterTest(unittest.TestCase):
    def test_configure_logging_installs_it(self):
        import tempfile

        from sana_evaluation.runtime import logger as logger_module

        asyncio_logger = logging.getLogger("asyncio")
        with contextlib.suppress(ValueError):
            asyncio_logger.removeFilter(asyncgen_teardown_filter)
        self.assertNotIn(asyncgen_teardown_filter, asyncio_logger.filters)

        app_root = logging.getLogger("sana_evaluation")
        saved_handlers, saved_propagate = list(app_root.handlers), app_root.propagate
        strands_root = logging.getLogger("strands")
        saved_strands, saved_strands_prop = list(strands_root.handlers), strands_root.propagate
        try:
            with tempfile.TemporaryDirectory() as tmp:
                logger_module.configure_logging(log_dir=tmp, task_id="t")
            self.assertIn(asyncgen_teardown_filter, asyncio_logger.filters)
        finally:
            for handler in list(app_root.handlers):
                handler.close()
            app_root.handlers, app_root.propagate = saved_handlers, saved_propagate
            strands_root.handlers, strands_root.propagate = saved_strands, saved_strands_prop


if __name__ == "__main__":
    unittest.main()
