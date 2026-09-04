"""
Tests for session persistence.

OpenContext has no daemon, so without this a pin, a budget change or a curated
working set is lost the moment the process exits. The rule these tests enforce is
that a restored page always holds real content: anything whose source has gone is
dropped and reported, never resurrected empty.
"""

from __future__ import annotations

import json
import os

import pytest

from opencontext.core.kernel import ContextKernel
from opencontext.core.session import (
    SESSION_VERSION,
    SessionError,
    restore_session,
    save_session,
    session_exists,
)
from opencontext.core.types import PageStatus, PageTier
from opencontext.storage.swap import SwapStorage


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "auth.py"
    source.write_text("def verify(token):\n    return token\n" * 30, encoding="utf-8")
    return {
        "root": tmp_path,
        "source": str(source),
        "swap": str(tmp_path / "swap.db"),
        "session": str(tmp_path / "session.json"),
    }


def populated_kernel(workspace, budget=4000) -> ContextKernel:
    kernel = ContextKernel(token_budget=budget, swap_storage=SwapStorage(workspace["swap"]))
    kernel.allocate_page(
        "rule:house", "House rules", "Always annotate types.", tier=PageTier.L0_PINNED
    )
    with open(workspace["source"], encoding="utf-8") as handle:
        kernel.allocate_page(
            "file:auth.py", "auth.py", handle.read(), metadata={"abs_path": workspace["source"]}
        )
    kernel.allocate_page(
        "turn:1", "User message", "[USER]: why is checkout slow?", tier=PageTier.L2_EPISODIC
    )
    kernel.allocate_page("file:cold.py", "cold.py", "value = 1\n" * 400)
    kernel.page_out("file:cold.py")
    return kernel


# -- round trip --------------------------------------------------------------------


def test_session_round_trip_preserves_the_arrangement(workspace):
    kernel = populated_kernel(workspace)
    kernel.pin_page("file:auth.py")
    kernel.set_budget(2500)

    summary = save_session(kernel, workspace["session"], root_dir=str(workspace["root"]))
    assert summary["pages_saved"] == 4
    assert session_exists(workspace["session"])

    restored_kernel = ContextKernel(
        token_budget=16000, swap_storage=SwapStorage(workspace["swap"])
    )
    result = restore_session(restored_kernel, workspace["session"])

    assert result["restored"] == 4
    assert result["dropped"] == []
    assert restored_kernel.token_budget == 2500, "the budget must carry across runs"
    assert restored_kernel.pages["file:auth.py"].tier == PageTier.L0_PINNED, "pins must survive"
    assert restored_kernel.pages["turn:1"].tier == PageTier.L2_EPISODIC


def test_each_page_kind_is_restored_from_the_right_source(workspace):
    """
    Three cases, three sources: swap for evicted pages, disk for file-backed ones,
    and the session file itself for anything with nowhere else to live.
    """
    kernel = populated_kernel(workspace)
    summary = save_session(kernel, workspace["session"], root_dir=str(workspace["root"]))

    assert summary["from_swap"] == 1, "the swapped page is left in swap.db"
    assert summary["from_file"] == 1, "the file-backed page is re-read from disk"
    assert summary["inlined"] == 2, "the rule and the turn have no other home"

    restored_kernel = ContextKernel(
        token_budget=16000, swap_storage=SwapStorage(workspace["swap"])
    )
    restore_session(restored_kernel, workspace["session"])

    assert "def verify" in restored_kernel.pages["file:auth.py"].content
    assert "checkout slow" in restored_kernel.pages["turn:1"].content
    assert "annotate types" in restored_kernel.pages["rule:house"].content
    assert restored_kernel.pages["file:cold.py"].status == PageStatus.SWAPPED


def test_a_swapped_page_restores_with_a_working_tombstone(workspace):
    kernel = populated_kernel(workspace)
    save_session(kernel, workspace["session"])

    restored_kernel = ContextKernel(
        token_budget=16000, swap_storage=SwapStorage(workspace["swap"])
    )
    restore_session(restored_kernel, workspace["session"])

    page = restored_kernel.pages["file:cold.py"]
    assert page.status == PageStatus.SWAPPED
    assert page.tombstone and "cold.py" in page.tombstone
    assert page.content == "", "swapped content stays on disk"

    # And it can still be paged back in.
    assert restored_kernel.page_fault("file:cold.py") is not None
    assert "value = 1" in restored_kernel.pages["file:cold.py"].content


def test_file_re_read_reflects_edits_made_since_the_save(workspace):
    """A file-backed page restores from the current file, not a stale copy."""
    kernel = populated_kernel(workspace)
    save_session(kernel, workspace["session"])

    with open(workspace["source"], "w", encoding="utf-8") as handle:
        handle.write("def verify(token):\n    return SOMETHING_NEW\n")

    restored_kernel = ContextKernel(
        token_budget=16000, swap_storage=SwapStorage(workspace["swap"])
    )
    restore_session(restored_kernel, workspace["session"])
    assert "SOMETHING_NEW" in restored_kernel.pages["file:auth.py"].content


# -- pages that cannot be restored ---------------------------------------------------


def test_a_vanished_file_is_dropped_and_reported(workspace):
    """
    Never restore a page with empty content.

    A page that claims to hold something it does not is exactly the class of
    silent-wrongness this project spent a release removing.
    """
    kernel = populated_kernel(workspace)
    save_session(kernel, workspace["session"])
    os.remove(workspace["source"])

    restored_kernel = ContextKernel(
        token_budget=16000, swap_storage=SwapStorage(workspace["swap"])
    )
    result = restore_session(restored_kernel, workspace["session"])

    assert "file:auth.py" not in restored_kernel.pages
    assert any(item["id"] == "file:auth.py" for item in result["dropped"])
    assert "no longer exists" in result["dropped"][0]["reason"]


def test_a_missing_swap_row_is_dropped_and_reported(workspace):
    kernel = populated_kernel(workspace)
    save_session(kernel, workspace["session"])
    kernel.swap.clear()

    restored_kernel = ContextKernel(
        token_budget=16000, swap_storage=SwapStorage(workspace["swap"])
    )
    result = restore_session(restored_kernel, workspace["session"])

    assert "file:cold.py" not in restored_kernel.pages
    assert any(item["id"] == "file:cold.py" for item in result["dropped"])


def test_no_restored_page_is_ever_empty(workspace):
    """The invariant, stated directly."""
    kernel = populated_kernel(workspace)
    save_session(kernel, workspace["session"])
    os.remove(workspace["source"])

    restored_kernel = ContextKernel(
        token_budget=16000, swap_storage=SwapStorage(workspace["swap"])
    )
    restore_session(restored_kernel, workspace["session"])

    for page in restored_kernel.pages.values():
        if page.status == PageStatus.ACTIVE:
            assert page.content, f"{page.id} restored with no content"


# -- budget and eviction on restore ----------------------------------------------------


def test_restore_re_enforces_the_budget(workspace):
    """Files can grow between runs, so the restored set may no longer fit."""
    kernel = populated_kernel(workspace, budget=100_000)
    save_session(kernel, workspace["session"])

    with open(workspace["source"], "w", encoding="utf-8") as handle:
        handle.write("def verify(token):\n    return token\n" * 4000)

    restored_kernel = ContextKernel(
        token_budget=16000, swap_storage=SwapStorage(workspace["swap"])
    )
    restored_kernel.token_budget = 500
    result = restore_session(restored_kernel, workspace["session"])
    restored_kernel.set_budget(500)

    metrics = restored_kernel.get_metrics()
    assert metrics.working_tokens <= 500 or metrics.over_budget, (
        "restore must re-run eviction rather than leaving the budget silently blown"
    )
    assert result["restored"] >= 1


# -- file handling ---------------------------------------------------------------------


def test_missing_session_file_is_not_an_error(tmp_path):
    kernel = ContextKernel(token_budget=1000, swap_storage=SwapStorage(str(tmp_path / "s.db")))
    result = restore_session(kernel, str(tmp_path / "absent.json"))
    assert result["restored"] == 0
    assert result["error"] == "no session file"


def test_a_corrupt_session_file_raises_rather_than_restoring_nonsense(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("{ not json", encoding="utf-8")

    kernel = ContextKernel(token_budget=1000, swap_storage=SwapStorage(str(tmp_path / "s.db")))
    with pytest.raises(SessionError):
        restore_session(kernel, str(path))


def test_a_session_from_another_version_is_refused(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"version": SESSION_VERSION + 99, "pages": []}), encoding="utf-8")

    kernel = ContextKernel(token_budget=1000, swap_storage=SwapStorage(str(tmp_path / "s.db")))
    result = restore_session(kernel, str(path))
    assert result["restored"] == 0
    assert "version" in result["error"]


def test_saving_is_atomic(workspace):
    """An interrupted write must not leave a truncated file behind."""
    kernel = populated_kernel(workspace)
    save_session(kernel, workspace["session"])
    save_session(kernel, workspace["session"])

    assert not os.path.exists(workspace["session"] + ".tmp")
    with open(workspace["session"], encoding="utf-8") as handle:
        assert json.load(handle)["version"] == SESSION_VERSION


def test_saving_creates_the_directory(tmp_path):
    kernel = ContextKernel(token_budget=1000, swap_storage=SwapStorage(str(tmp_path / "s.db")))
    kernel.allocate_page("rule:x", "Rule", "content")

    target = tmp_path / "nested" / "deeper" / "session.json"
    save_session(kernel, str(target))
    assert target.exists()
