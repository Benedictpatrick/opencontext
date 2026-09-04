"""
Shared test configuration.

Textual's test harness tears an application down while its animation timer is
still pending, and asyncio then logs "Task was destroyed but it is pending!" to
stderr. It is teardown noise from the framework, not a fault in ContextOS or in
the tests, but it buries real failures in the output. Silencing it keeps a failing
run readable.
"""

from __future__ import annotations

import asyncio
import logging

import pytest


@pytest.fixture(scope="session", autouse=True)
def _quiet_textual_teardown_noise():
    """
    Drop asyncio's pending-task warnings for the whole session.

    The message is emitted from `Task.__del__` at garbage-collection time, which
    can land long after the test that created the task finished, so the filter has
    to outlive individual tests.
    """

    class _PendingTaskFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Task was destroyed but it is pending" not in record.getMessage()

    logger = logging.getLogger("asyncio")
    log_filter = _PendingTaskFilter()
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch):
    """
    Keep ambient configuration out of the tests.

    A developer with CONTEXTOS_UPSTREAM or CONTEXTOS_TOKENIZER exported would
    otherwise get different results from CI, and tests that assert on token counts
    would fail for reasons unrelated to the change under test.
    """
    for variable in (
        "CONTEXTOS_TOKENIZER",
        "CONTEXTOS_API_KEY",
        "CONTEXTOS_EXPOSE_CONTENT",
        "CONTEXTOS_UPSTREAM_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    # Point the default upstream at an address nothing is listening on, so a test
    # that reaches the network by accident fails fast instead of contacting a
    # model server that happens to be running on the developer's machine.
    monkeypatch.setenv("CONTEXTOS_UPSTREAM", "http://127.0.0.1:9/v1")

    from contextos.core import tokens

    tokens.reset_tokenizer_cache()
    yield
    tokens.reset_tokenizer_cache()
