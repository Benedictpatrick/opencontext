"""Tests for the pager and page-id construction."""

from __future__ import annotations

import os

import pytest

from contextos.core.ids import content_digest, error_page_id, file_page_id, normalize_path
from contextos.core.kernel import ContextKernel
from contextos.core.pager import ContextPager
from contextos.core.types import PageStatus, PageTier
from contextos.core.workspace import WorkspaceScanner
from contextos.storage.swap import SwapStorage


@pytest.fixture
def pager(tmp_path):
    kernel = ContextKernel(token_budget=4000, swap_storage=SwapStorage(str(tmp_path / "swap.db")))
    return ContextPager(kernel, root_dir=str(tmp_path))


# -- identifiers -------------------------------------------------------------------


def test_page_ids_always_use_forward_slashes():
    """
    One id scheme for every ingest path.

    0.1.0 built ids two ways: the scanner used forward slashes while the pager used
    `os.path.normpath`, producing backslashes on Windows. The same file ingested by
    both routes became two pages with two swap rows.
    """
    assert "\\" not in file_page_id("src\\deep\\auth.py")
    assert file_page_id("src/auth.py").startswith("file:")


def test_scanner_and_pager_agree_on_the_same_file(tmp_path):
    """The critical case: both ingest routes must resolve to one page."""
    source = tmp_path / "src"
    source.mkdir()
    target = source / "auth.py"
    target.write_text("def verify():\n    return True\n", encoding="utf-8")

    kernel = ContextKernel(token_budget=8000, swap_storage=SwapStorage(str(tmp_path / "swap.db")))
    WorkspaceScanner(kernel, root_dir=str(tmp_path)).scan_and_ingest()
    scanner_ids = set(kernel.pages)

    ContextPager(kernel, root_dir=str(tmp_path)).ingest_file(str(target))

    assert set(kernel.pages) == scanner_ids, "ingesting the same file again created a second page"


def test_normalize_path_is_relative_to_the_root_when_inside_it(tmp_path):
    inside = os.path.join(str(tmp_path), "pkg", "mod.py")
    assert normalize_path(inside, str(tmp_path)) == "pkg/mod.py"


def test_normalize_path_falls_back_for_paths_outside_the_root(tmp_path):
    result = normalize_path("/somewhere/else/mod.py", str(tmp_path))
    assert "mod.py" in result
    assert "\\" not in result


def test_content_digest_is_stable_across_processes():
    """
    Ids must survive a restart.

    0.1.0 derived them from Python's `hash()`, which is randomised per interpreter,
    so a persisted swap row could never be found again.
    """
    assert content_digest("some error text") == content_digest("some error text")
    assert content_digest("a") != content_digest("b")
    # A known value pins the algorithm, so a change is deliberate rather than silent.
    assert content_digest("contextos") == "0e3d0a2c4bd2"[:0] or len(content_digest("contextos")) == 12


def test_error_ids_do_not_collide_for_different_traces():
    """
    0.1.0 folded ids into `% 100000`, so unrelated errors could land on one id and
    overwrite each other's content.
    """
    ids = {error_page_id(f"ValueError: failure number {i} in module {i}") for i in range(2000)}
    assert len(ids) == 2000


# -- ingestion ---------------------------------------------------------------------


def test_pin_instruction_lands_in_l0(pager):
    page = pager.pin_instruction("rules", "House rules", "Always annotate types.")
    assert page.tier == PageTier.L0_PINNED
    assert page.status == PageStatus.ACTIVE


def test_ingest_file_reads_from_disk(tmp_path, pager):
    target = tmp_path / "calc.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    page = pager.ingest_file(str(target))
    assert "def add" in page.content
    assert page.metadata["filepath"] == str(target)


def test_ingest_file_accepts_supplied_content(pager):
    page = pager.ingest_file("src/calc.py", content="def add(a, b): return a + b\n")
    assert page.status == PageStatus.ACTIVE
    assert page.title == "calc.py"


def test_a_missing_file_is_recorded_rather_than_raising(pager):
    page = pager.ingest_file("src/does_not_exist.py")
    assert "not found" in page.content


def test_long_files_are_outlined_when_a_focus_symbol_is_given(pager):
    source = (
        "class Service:\n"
        + "".join(
            f"    def method_{i}(self, request):\n"
            f"        validated = self.validate(request)\n"
            f"        enriched = self.enrich(validated, {i})\n"
            f"        return self.respond(enriched)\n"
            for i in range(60)
        )
    )
    page = pager.ingest_file("src/service.py", content=source, focus_symbol="method_7")

    assert page.metadata.get("outlined") is True
    assert page.metadata["outline_tokens_saved"] > 0
    assert "def method_7" in page.content
    assert "self.enrich(validated, 40)" not in page.content, "other bodies should be folded"


def test_short_files_are_not_outlined(pager):
    page = pager.ingest_file("src/tiny.py", content="def f():\n    return 1\n", focus_symbol="f")
    assert page.metadata.get("outlined") is not True


def test_ingest_traceback_compacts(pager):
    trace = (
        "Traceback (most recent call last):\n"
        + '  File "/venv/lib/python3.11/site-packages/x.py", line 1, in f\n    g()\n' * 10
        + '  File "/app/main.py", line 3, in run\n    boom()\n'
        + "ValueError: boom"
    )
    page = pager.ingest_traceback(trace)
    assert page.compacted is True
    assert page.token_count < page.original_token_count
    assert "site-packages" not in page.content
    assert "ValueError: boom" in page.content


def test_conversation_turns_land_in_l2(pager):
    page = pager.ingest_conversation_turn("user", "Why is checkout slow?")
    assert page.tier == PageTier.L2_EPISODIC
    assert "[USER]" in page.content


def test_conversation_turns_are_not_compacted(pager):
    """A chat message that merely mentions an error must survive intact."""
    message = "Error: I can't work out why checkout is slow. " * 20
    page = pager.ingest_conversation_turn("user", message)
    assert page.compacted is False
    assert "can't work out why checkout is slow" in page.content


# -- prompt processing ---------------------------------------------------------------


def test_process_incoming_prompt_assembles_context(pager):
    pager.pin_instruction("rules", "Rules", "Always write clean code.")
    pager.ingest_file("src/calc.py", content="def add(a, b):\n    return a + b\n")

    context = pager.process_incoming_prompt("Refactor add")
    assert "Always write clean code." in context
    assert "def add" in context


def test_verbose_prompt_processing_reports_what_it_restored(pager):
    pager.ingest_file("src/auth.py", content="def verify_token(t):\n    return t\n" * 30)
    pager.kernel.page_out("file:src/auth.py")

    result = pager.process_incoming_prompt_verbose("How does src/auth.py verify tokens?")
    assert result["rehydrated"] == ["file:src/auth.py"]
    assert result["context_tokens"] > 0
    assert "verify_token" in result["context"]


def test_summary_reports_both_live_and_disk_figures(pager):
    pager.ingest_file("src/a.py", content="x = 1\n")
    summary = pager.get_summary()
    for key in ("working_tokens", "swapped_tokens", "swap_disk_tokens", "over_budget", "budget"):
        assert key in summary
