"""Tests for the context kernel: budgets, eviction, paging and accounting."""

from __future__ import annotations

import os

import pytest

from contextos.core.kernel import ContextKernel
from contextos.core.tokens import estimate_tokens
from contextos.core.types import PageStatus, PageTier, PagingEventType
from contextos.storage.swap import SwapStorage


@pytest.fixture
def kernel(tmp_path):
    return ContextKernel(token_budget=1000, swap_storage=SwapStorage(str(tmp_path / "swap.db")))


# -- allocation and budget --------------------------------------------------------


def test_allocation_and_eviction_under_budget(kernel):
    kernel.allocate_page("rule:invariants", "Invariants", "x = 1\n" * 50, tier=PageTier.L0_PINNED)
    kernel.allocate_page("file:auth.py", "auth.py", "def verify():\n    pass\n" * 60)
    kernel.allocate_page("file:db.py", "db.py", "def connect():\n    pass\n" * 60)
    kernel.allocate_page("file:api.py", "api.py", "def route():\n    pass\n" * 60)

    metrics = kernel.get_metrics()
    assert metrics.working_tokens <= kernel.token_budget
    assert kernel.pages["rule:invariants"].status == PageStatus.ACTIVE, "L0 must never be evicted"
    assert metrics.l3_pages > 0, "budget pressure should have evicted something"


def test_pinned_pages_are_never_evicted(kernel):
    for index in range(6):
        kernel.allocate_page(
            f"rule:{index}", f"Rule {index}", "keep this\n" * 60, tier=PageTier.L0_PINNED
        )
    assert all(page.status == PageStatus.ACTIVE for page in kernel.pages.values())


def test_over_budget_is_reported_when_pinned_content_alone_exceeds_it(kernel):
    """
    Eviction cannot help when everything is pinned, so the kernel must say so.

    0.1.0 exhausted its candidate list and returned silently, leaving the budget
    exceeded with nothing in the telemetry to indicate it.
    """
    for index in range(8):
        kernel.allocate_page(
            f"rule:{index}", f"Rule {index}", "pinned content\n" * 60, tier=PageTier.L0_PINNED
        )

    metrics = kernel.get_metrics()
    assert metrics.over_budget is True
    assert metrics.working_tokens > kernel.token_budget
    assert any(e.event_type == PagingEventType.BUDGET_EXCEEDED for e in kernel.events)


def test_l2_is_evicted_before_l1(kernel):
    """Episodic history is given up before working code, whatever the access order."""
    kernel.allocate_page("turn:1", "Turn 1", "conversation history\n" * 100, tier=PageTier.L2_EPISODIC)
    kernel.allocate_page("file:a.py", "a.py", "def handler(): pass\n" * 100, tier=PageTier.L1_WORKING_RAM)
    # Pushes the total past the 1,000 token budget and forces a choice.
    kernel.allocate_page("file:b.py", "b.py", "def other(): pass\n" * 200, tier=PageTier.L1_WORKING_RAM)

    assert kernel.get_metrics().working_tokens <= kernel.token_budget
    assert kernel.pages["turn:1"].status == PageStatus.SWAPPED, "L2 should be evicted first"
    assert kernel.pages["file:b.py"].status == PageStatus.ACTIVE, "most recent L1 should be kept"


def test_reallocating_an_existing_page_updates_it_in_place(kernel):
    kernel.allocate_page("file:a.py", "a.py", "original")
    page = kernel.allocate_page("file:a.py", "a.py", "replacement content")
    assert len(kernel.pages) == 1
    assert page.content == "replacement content"


# -- swap and page faults ----------------------------------------------------------


def test_swapping_out_frees_memory_and_leaves_a_tombstone(kernel):
    """Swap must reclaim process memory, not only context budget."""
    kernel.allocate_page("file:big.py", "big.py", "payload\n" * 200)
    assert kernel.page_out("file:big.py") is True

    page = kernel.pages["file:big.py"]
    assert page.status == PageStatus.SWAPPED
    assert page.content == "", "content should be released once it is safely on disk"
    assert page.tombstone and "big.py" in page.tombstone
    assert estimate_tokens(page.tombstone) < page.token_count


def test_page_fault_restores_content(kernel):
    original = "payload\n" * 200
    kernel.allocate_page("file:big.py", "big.py", original)
    kernel.page_out("file:big.py")

    restored = kernel.page_fault("file:big.py")
    assert restored is not None
    assert restored.content == original
    assert restored.status == PageStatus.ACTIVE
    assert kernel.total_page_faults == 1


def test_page_fault_latency_is_measured(kernel):
    kernel.allocate_page("file:a.py", "a.py", "content\n" * 50)
    kernel.page_out("file:a.py")

    assert kernel.get_metrics().avg_page_fault_ms == 0.0, "no faults yet means no latency figure"
    kernel.page_fault("file:a.py")
    assert kernel.get_metrics().avg_page_fault_ms > 0.0


def test_page_fault_on_unknown_page_returns_none(kernel):
    assert kernel.page_fault("file:missing.py") is None


def test_swapping_a_pinned_page_is_refused(kernel):
    kernel.allocate_page("rule:x", "Rule", "content", tier=PageTier.L0_PINNED)
    assert kernel.page_out("rule:x") is False


# -- accounting -------------------------------------------------------------------


def test_live_swap_metrics_match_the_database(tmp_path):
    """
    `swapped_tokens`/`l3_pages` describe this kernel; disk figures describe the file.

    0.1.0 read `swapped_tokens` from a SUM over every row ever written while taking
    `l3_pages` from the in-memory table, and never deleted a row on rehydration, so
    the reported swap total only ever grew. A fresh kernel over an existing database
    reported thousands of swapped tokens across zero pages.
    """
    swap = SwapStorage(str(tmp_path / "swap.db"))
    kernel = ContextKernel(token_budget=100_000, swap_storage=swap)

    for index in range(5):
        kernel.allocate_page(f"file:{index}.py", f"{index}.py", "content\n" * 60)
    kernel.swap_all_unpinned()

    metrics = kernel.get_metrics()
    assert metrics.l3_pages == 5
    assert metrics.swap_disk_rows == 5
    assert metrics.swapped_tokens == metrics.swap_disk_tokens

    kernel.page_fault("file:0.py")
    metrics = kernel.get_metrics()
    assert metrics.l3_pages == 4, "rehydrated page is no longer swapped"
    assert metrics.swap_disk_rows == 4, "its swap row must be removed, not left behind"


def test_a_fresh_kernel_does_not_inherit_old_swap_totals(tmp_path):
    db = str(tmp_path / "swap.db")
    first = ContextKernel(token_budget=100_000, swap_storage=SwapStorage(db))
    for index in range(4):
        first.allocate_page(f"file:{index}.py", f"{index}.py", "content\n" * 60)
    first.swap_all_unpinned()

    second = ContextKernel(token_budget=100_000, swap_storage=SwapStorage(db))
    metrics = second.get_metrics()

    assert metrics.l3_pages == 0, "a new kernel has nothing swapped out"
    assert metrics.swapped_tokens == 0, "live swap total must not include earlier runs"
    assert metrics.swap_disk_rows == 4, "the database rows are still reported, separately"


def test_tokens_saved_reflects_real_reductions(kernel):
    kernel.allocate_page("file:a.py", "a.py", "content\n" * 100)
    before = kernel.total_tokens_saved
    kernel.page_out("file:a.py")
    saved = kernel.total_tokens_saved - before

    page = kernel.pages["file:a.py"]
    assert saved == page.token_count - estimate_tokens(page.tombstone)


# -- repeat-failure detection ------------------------------------------------------


def test_repeated_failures_raise_a_warning(kernel):
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/app/tools/db.py", line 49, in run\n'
        "    conn = connect(dsn)\n"
        "psycopg2.OperationalError: could not connect to server: Connection refused"
    )
    for attempt in range(4):
        kernel.allocate_page(
            f"error:{attempt}", f"Attempt {attempt}", f"{trace}\n# retry {attempt}", auto_compact=True
        )

    warnings = [e for e in kernel.events if e.event_type == PagingEventType.LEAK_WARNING]
    assert warnings, "four identical failures should raise a repeat-failure warning"


def test_signature_ignores_volatile_detail(kernel):
    """Timestamps, addresses and retry counters must not defeat the match."""
    first = kernel._error_signature("Root cause: OperationalError: refused at 0xDEADBEEF port 5432")
    second = kernel._error_signature("Root cause: OperationalError: refused at 0xCAFEBABE port 6543")
    assert first == second


def test_distinct_failures_do_not_trigger_a_warning(kernel):
    for index, message in enumerate(
        ["ValueError: bad input", "KeyError: missing", "TypeError: wrong type"]
    ):
        kernel.allocate_page(
            f"error:{index}",
            f"Error {index}",
            f'Traceback (most recent call last):\n  File "/app/x.py", line 1, in f\n    f()\n{message}',
            auto_compact=True,
        )
    assert not [e for e in kernel.events if e.event_type == PagingEventType.LEAK_WARNING]


# -- prompt resolution -------------------------------------------------------------


def test_touch_or_fault_restores_referenced_pages(kernel):
    kernel.allocate_page("file:src/auth.py", "src/auth.py", "def verify():\n    pass\n" * 40)
    kernel.page_out("file:src/auth.py")

    restored = kernel.touch_or_fault("Can you refactor verify() in src/auth.py?")
    assert restored == ["file:src/auth.py"]
    assert kernel.pages["file:src/auth.py"].status == PageStatus.ACTIVE


def test_touch_or_fault_ignores_short_and_incidental_names(kernel):
    """
    A two-character title matches almost any sentence; matching it would page the
    whole workspace back in on an unrelated question.
    """
    kernel.allocate_page("file:a.py", "a.py", "content\n" * 40)
    kernel.allocate_page("file:db.py", "db.py", "content\n" * 40)
    kernel.swap_all_unpinned()

    assert kernel.touch_or_fault("what a day, the database is slow") == []


def test_touch_or_fault_matches_a_bare_basename(kernel):
    kernel.allocate_page("file:src/deep/handler.py", "src/deep/handler.py", "content\n" * 40)
    kernel.page_out("file:src/deep/handler.py")
    assert kernel.touch_or_fault("what does handler.py do?") == ["file:src/deep/handler.py"]


# -- context assembly ---------------------------------------------------------------


def test_assembled_context_shows_tombstones_not_content(kernel):
    kernel.allocate_page("rule:x", "Rules", "Always annotate types.", tier=PageTier.L0_PINNED)
    kernel.allocate_page("file:cold.py", "cold.py", "SECRET_BODY_TEXT\n" * 40)
    kernel.page_out("file:cold.py")

    context = kernel.assemble_context()
    assert "Always annotate types." in context
    assert "SECRET_BODY_TEXT" not in context, "swapped content must not reach the context window"
    assert "cold.py" in context, "its tombstone should still be visible"


# -- page management -----------------------------------------------------------------


def test_pin_unpin_and_delete(kernel):
    kernel.allocate_page("file:c.py", "c.py", "content\n" * 20)
    assert kernel.pin_page("file:c.py") is True
    assert kernel.pages["file:c.py"].tier == PageTier.L0_PINNED
    assert kernel.unpin_page("file:c.py") is True
    assert kernel.pages["file:c.py"].tier == PageTier.L1_WORKING_RAM
    assert kernel.delete_page("file:c.py") is True
    assert "file:c.py" not in kernel.pages
    assert kernel.delete_page("file:c.py") is False


def test_pinning_a_swapped_page_restores_it_first(kernel):
    kernel.allocate_page("file:d.py", "d.py", "content\n" * 40)
    kernel.page_out("file:d.py")
    kernel.pin_page("file:d.py")

    page = kernel.pages["file:d.py"]
    assert page.status == PageStatus.ACTIVE
    assert page.content, "a pinned page must hold its content"


def test_set_budget_reapplies_eviction(kernel):
    for index in range(5):
        kernel.allocate_page(f"file:{index}.py", f"{index}.py", "content\n" * 40)
    metrics = kernel.set_budget(300)
    assert metrics.token_budget == 300
    assert metrics.working_tokens <= 300 or metrics.over_budget


def test_rehydrate_all_and_swap_all(kernel):
    for index in range(4):
        kernel.allocate_page(f"file:{index}.py", f"{index}.py", "content\n" * 20)
    swapped = kernel.swap_all_unpinned()
    assert swapped == 4

    kernel.set_budget(100_000)
    assert kernel.rehydrate_all() == 4


def test_clear_swap_purges_pages_that_have_no_other_copy(kernel):
    kernel.allocate_page("file:a.py", "a.py", "content\n" * 40)
    kernel.page_out("file:a.py")
    kernel.clear_swap()

    assert "file:a.py" not in kernel.pages, "a swapped page must not survive its only copy"
    assert kernel.get_metrics().swap_disk_rows == 0


# -- search --------------------------------------------------------------------------


def test_search_ranks_by_keyword_overlap(kernel):
    kernel.allocate_page(
        "file:database.py", "database.py", "class DatabaseConnection:\n    def connect_postgres(self): ..."
    )
    kernel.allocate_page("file:auth.py", "auth.py", "class TokenValidator:\n    def verify_jwt(self): ...")

    results = kernel.search("postgres database")
    assert results
    assert results[0][0].id == "file:database.py"


def test_search_is_still_available_under_the_old_name(kernel):
    kernel.allocate_page("file:a.py", "a.py", "alpha")
    assert kernel.semantic_search("alpha")


def test_event_log_is_bounded(kernel):
    for index in range(400):
        kernel.allocate_page(f"file:{index}.py", f"{index}.py", "x")
    assert len(kernel.events) <= 200


# -- context window report -----------------------------------------------------------


def test_context_report_reconciles_to_the_measured_window(kernel):
    """
    The breakdown must add up to the window's real size.

    Summing `page.token_count` would not: a tombstone costs a fraction of the page
    it replaces, section headers are real tokens no page owns, and per-segment
    estimates round independently. A breakdown that did not reconcile would be a
    new fabricated figure dressed as a measurement.
    """
    kernel.allocate_page("rule:x", "Rules", "Always annotate types.", tier=PageTier.L0_PINNED)
    for index in range(4):
        kernel.allocate_page(f"file:{index}.py", f"src/{index}.py", "def handler():\n    pass\n" * 30)
    kernel.allocate_page("turn:1", "Turn", "[USER]: why?", tier=PageTier.L2_EPISODIC)

    report = kernel.assemble_context_report()
    rows = sum(segment["tokens"] for segment in report["segments"])

    assert report["total_tokens"] == estimate_tokens(report["context"])
    assert rows + report["overhead_tokens"] == report["total_tokens"]


def test_context_report_marks_swapped_pages_as_tombstones(kernel):
    kernel.allocate_page("file:big.py", "big.py", "payload\n" * 300)
    kernel.page_out("file:big.py")

    report = kernel.assemble_context_report()
    segment = next(s for s in report["segments"] if s["page_id"] == "file:big.py")

    assert segment["included_as"] == "tombstone"
    assert segment["tokens"] < segment["full_tokens"], "a tombstone must cost less than its page"


def test_context_report_matches_assemble_context(kernel):
    kernel.allocate_page("file:a.py", "a.py", "x = 1\n" * 20)
    assert kernel.assemble_context_report()["context"] == kernel.assemble_context()


def test_empty_kernel_reports_an_empty_window(kernel):
    report = kernel.assemble_context_report()
    assert report["context"] == ""
    assert report["segments"] == []
    assert report["total_tokens"] == 0


def test_automatic_eviction_skips_pages_smaller_than_their_tombstone(kernel):
    """
    Evicting a tiny page makes the window bigger, not smaller.

    A tombstone costs about 24 tokens, so swapping a 6-token conversation turn
    would have the kernel paying to save nothing.
    """
    kernel.allocate_page("turn:1", "Turn", "[USER]: hi", tier=PageTier.L2_EPISODIC)
    for index in range(5):
        kernel.allocate_page(f"file:{index}.py", f"src/{index}.py", "def handler():\n    pass\n" * 40)

    assert kernel.pages["turn:1"].status == PageStatus.ACTIVE, (
        "a page smaller than its own tombstone must not be auto-evicted"
    )
    assert kernel.get_metrics().l3_pages > 0, "larger pages should still have been evicted"


def test_an_explicit_swap_still_honours_the_caller(kernel):
    """The size guard governs automatic eviction, not a deliberate page_out."""
    kernel.allocate_page("turn:1", "Turn", "[USER]: hi", tier=PageTier.L2_EPISODIC)
    assert kernel.page_out("turn:1") is True
    assert kernel.pages["turn:1"].status == PageStatus.SWAPPED


def test_a_long_conversation_does_not_starve_the_working_set(tmp_path):
    """
    Conversation turns must not push out the source the session is about.

    Each turn is smaller than the tombstone that would replace it, so
    `_swap_would_help` rejects it and it is never an eviction candidate. Before
    coalescing, the only pages the kernel *could* evict were the files, so a long
    chat emptied the working set to hold history — backwards for a tool whose
    purpose is keeping the relevant code in context.
    """
    kernel = ContextKernel(
        token_budget=2_000, swap_storage=SwapStorage(str(tmp_path / "swap.db"))
    )
    file_ids = []
    for n in range(4):
        body = f"# module {n}\n" + "\n".join(f"def fn_{n}_{i}():\n    return {i}" for i in range(20))
        file_ids.append(kernel.allocate_page(f"file:mod_{n}.py", f"mod_{n}.py", body).id)

    resident_before = sum(1 for pid in file_ids if kernel.pages[pid].content)
    assert resident_before >= 1

    # Turns small enough that evicting one individually would cost more than it
    # saves — the case that made them permanently ineligible.
    for turn in range(400):
        kernel.allocate_page(
            f"turn:{turn}", f"Turn {turn}", f"Turn {turn}: still timing out.",
            tier=PageTier.L2_EPISODIC,
        )

    resident_after = sum(1 for pid in file_ids if kernel.pages[pid].content)
    assert resident_after == resident_before, (
        "conversation history evicted source files instead of being coalesced"
    )

    digests = [p for pid, p in kernel.pages.items() if pid.startswith("episodic:digest")]
    assert digests, "old turns were never coalesced into an evictable page"
    assert sum(d.metadata.get("coalesced_turns", 0) for d in digests) > 100


def test_a_coalesced_history_can_be_read_back(tmp_path):
    """Coalescing must not lose the conversation: the digest pages back in whole."""
    kernel = ContextKernel(
        token_budget=2_000, swap_storage=SwapStorage(str(tmp_path / "swap.db"))
    )
    kernel.allocate_page("file:big.py", "big.py", "x = 1\n" * 4000)
    for turn in range(300):
        kernel.allocate_page(
            f"turn:{turn}", f"Turn {turn}", f"Turn {turn}: the checkout endpoint is slow.",
            tier=PageTier.L2_EPISODIC,
        )

    digest = next(p for pid, p in kernel.pages.items() if pid.startswith("episodic:digest"))
    restored = kernel.page_fault(digest.id)
    assert restored is not None
    assert "Turn 0" in restored.content


def test_a_page_fault_is_not_undone_by_the_budget_it_triggers(tmp_path):
    """
    Faulting a page that is large relative to the budget used to restore it and
    then immediately evict it again inside the same call, so the caller received a
    page with empty content and no indication anything was wrong. The requested
    page is now exempt for that call; the kernel reports being over budget instead.
    """
    kernel = ContextKernel(
        token_budget=1_500, swap_storage=SwapStorage(str(tmp_path / "swap.db"))
    )
    big = kernel.allocate_page("file:big.py", "big.py", "def f():\n    return 1\n" * 400)
    kernel.page_out(big.id)
    kernel.allocate_page("file:other.py", "other.py", "def g():\n    return 2\n" * 200)

    restored = kernel.page_fault("file:big.py")
    assert restored is not None
    assert restored.content, "page fault returned an emptied page"
    assert kernel.pages["file:big.py"].status == PageStatus.ACTIVE


def _drive_turns(kernel, start, count):
    for i in range(start, start + count):
        kernel.allocate_page(
            f"turn:{i}", f"Turn {i}", f"Turn {i}: checkout is slow.",
            tier=PageTier.L2_EPISODIC,
        )


def test_a_restored_session_does_not_overwrite_its_own_history(tmp_path):
    """
    Digest ids are derived from the pages that exist, not from a counter.

    A restored session builds a fresh kernel around pages that already include
    `episodic:digest:0`. With an instance counter starting at zero, the next
    coalesce handed out that id again, replacing the restored history in place and
    then overwriting its swap row — losing the conversation with no event logged.
    """
    db = str(tmp_path / "swap.db")
    first = ContextKernel(token_budget=2_000, swap_storage=SwapStorage(db))
    first.allocate_page("file:big.py", "big.py", "x = 1\n" * 3000)
    _drive_turns(first, 0, 700)
    assert any(pid.startswith("episodic:digest") for pid in first.pages)

    # Restore: a new kernel over the same pages and the same swap database.
    resumed = ContextKernel(token_budget=2_000, swap_storage=SwapStorage(db))
    for page_id, page in first.pages.items():
        resumed.pages[page_id] = page.model_copy(deep=True)

    _drive_turns(resumed, 1000, 700)

    oldest = resumed.pages["episodic:digest:0"]
    if not oldest.content:
        resumed.page_fault("episodic:digest:0")
    assert "Turn 0:" in resumed.pages["episodic:digest:0"].content


def test_over_budget_after_a_protected_fault_reports_the_real_reason(tmp_path):
    """
    The over-budget event used to state that every unpinned page had been evicted
    and blame the pinned set. After a protected page fault neither is true: the
    overshoot is the page the caller asked for, and nothing need be pinned at all.
    """
    kernel = ContextKernel(
        token_budget=1_200, swap_storage=SwapStorage(str(tmp_path / "swap.db"))
    )
    big = kernel.allocate_page("file:big.py", "big.py", "def f():\n    return 1\n" * 400)
    kernel.page_out(big.id)
    kernel.allocate_page("file:other.py", "other.py", "def g():\n    return 2\n" * 200)
    kernel.page_fault("file:big.py")

    budget_events = [e for e in kernel.events if e.event_type == PagingEventType.BUDGET_EXCEEDED]
    assert budget_events
    message = budget_events[-1].description
    assert "big.py" in message
    assert "pinned to L0" not in message


def test_conversation_pressure_at_a_realistic_budget(tmp_path):
    """
    The figure published in the changelog, pinned to the code that produces it.

    Uses the frozen benchmark corpus against a 16k budget, so the numbers are the
    same on any checkout: eight source files are ingested, the budget evicts some
    immediately, and 600 conversation turns then run through the kernel. What
    matters is that the turns cost at most one further file — before coalescing
    they took all but one.
    """
    corpus = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "fixtures", "corpus",
    )
    kernel = ContextKernel(
        token_budget=16_000, swap_storage=SwapStorage(str(tmp_path / "swap.db"))
    )
    file_ids = []
    for name in sorted(os.listdir(corpus))[:8]:
        with open(os.path.join(corpus, name), "r", encoding="utf-8") as handle:
            file_ids.append(
                kernel.allocate_page(f"file:{name[:-4]}", name[:-4], handle.read()).id
            )

    resident_after_scan = sum(1 for pid in file_ids if kernel.pages[pid].content)
    assert resident_after_scan == 4

    _drive_turns(kernel, 0, 600)

    resident_after_chat = sum(1 for pid in file_ids if kernel.pages[pid].content)
    assert resident_after_chat >= 3
    assert kernel.get_metrics().working_tokens <= kernel.token_budget
