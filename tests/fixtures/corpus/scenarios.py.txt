"""
Demonstration scenarios.

These build real pages, run the real compactor and report measured results. They
live outside the kernel because simulation is not a memory-management concern, and
because keeping them separate makes it obvious which numbers in the UI come from a
demo and which come from the user's own workspace.

Nothing here fabricates a figure. Every number returned was produced by running
the code path it describes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from contextos.core.kernel import ContextKernel
from contextos.core.tokens import estimate_tokens
from contextos.core.types import PageTier, PagingEventType

# One failing tool call, as an agent would capture it. Reused verbatim across
# retries so the scenario reproduces a genuine retry loop.
FAILING_TOOL_CALL = """Traceback (most recent call last):
  File "/workspace/agent/executor.py", line 142, in execute_tool
    return await tool.run(args)
  File "/workspace/.venv/lib/python3.11/site-packages/anyio/_core/_tasks.py", line 118, in __aexit__
    raise exc
  File "/workspace/agent/tools/database.py", line 49, in run
    conn = psycopg2.connect(dsn=self.dsn, connect_timeout=5)
  File "/workspace/.venv/lib/python3.11/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
psycopg2.OperationalError: could not connect to server: Connection refused
	Is the server running on host "db.internal" (10.0.4.19) and accepting
	TCP/IP connections on port 5432?
"""


def run_doom_loop(kernel: ContextKernel, iterations: int = 4) -> Dict[str, Any]:
    """
    Reproduce an agent retrying a failing tool call, and measure what compaction saved.

    Each iteration ingests the same failure with a distinct retry marker, exactly as
    a looping agent would. Returns measured token counts plus whether the kernel's
    repeat-failure detector fired.
    """
    raw_tokens = 0
    compacted_tokens = 0
    page_ids: List[str] = []

    warnings_before = sum(
        1 for event in kernel.events if event.event_type == PagingEventType.LEAK_WARNING
    )

    for attempt in range(1, iterations + 1):
        body = f"{FAILING_TOOL_CALL}\n# retry {attempt} at {time.time():.3f}"
        page_id = f"tool_error:db_connect:attempt_{attempt}"
        page_ids.append(page_id)

        raw_tokens += estimate_tokens(body)
        page = kernel.allocate_page(
            page_id=page_id,
            title=f"Tool failure, attempt {attempt}",
            content=body,
            tier=PageTier.L2_EPISODIC,
            auto_compact=True,
        )
        compacted_tokens += page.token_count

    saved = raw_tokens - compacted_tokens
    pct_saved = round(saved / raw_tokens * 100, 1) if raw_tokens else 0.0

    warnings_after = sum(
        1 for event in kernel.events if event.event_type == PagingEventType.LEAK_WARNING
    )
    detected = warnings_after > warnings_before

    kernel._log_event(
        PagingEventType.DOOM_LOOP_PREVENTED,
        "agent:loop_guard",
        saved,
        f"{iterations} repeated failures compacted: {saved:,} tokens kept out of context "
        f"({pct_saved}%)" + ("; repeat-failure alert raised" if detected else ""),
    )

    compacted_sample = ""
    if page_ids:
        last = kernel.pages.get(page_ids[-1])
        if last is not None:
            compacted_sample = last.get_context_representation()

    return {
        "iterations": iterations,
        "raw_tokens": raw_tokens,
        "compacted_tokens": compacted_tokens,
        "tokens_saved": saved,
        "pct_saved": pct_saved,
        "repeat_failure_detected": detected,
        "page_ids": page_ids,
        "raw_sample": "\n\n".join(
            f"=== attempt {i} ===\n{FAILING_TOOL_CALL}" for i in range(1, iterations + 1)
        ),
        "compacted_sample": compacted_sample,
    }
