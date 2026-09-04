"""
Tests for swap storage.

This module had no tests in 0.1.0, and it is where the swap-accounting defect
lived. Once a page is swapped its content is released from memory, so this
database holds the only copy — durability and honest failure reporting matter.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from opencontext.core.types import ContextPage, PageStatus, PageTier
from opencontext.storage.swap import SwapStorage, SwapStorageError


@pytest.fixture
def storage(tmp_path):
    return SwapStorage(str(tmp_path / "swap.db"))


def make_page(page_id: str = "file:a.py", content: str = "print('hello')") -> ContextPage:
    return ContextPage(
        id=page_id,
        title=page_id.split(":", 1)[-1],
        tier=PageTier.L1_WORKING_RAM,
        status=PageStatus.ACTIVE,
        content=content,
        token_count=len(content) // 4 or 1,
        original_token_count=len(content) // 4 or 1,
        metadata={"origin": "test"},
    )


def test_store_and_retrieve_round_trip(storage):
    page = make_page(content="def f():\n    return 42\n")
    storage.store(page)

    restored = storage.retrieve(page.id)
    assert restored is not None
    assert restored.id == page.id
    assert restored.content == page.content
    assert restored.token_count == page.token_count
    assert restored.tier == page.tier
    assert restored.metadata == {"origin": "test"}
    assert restored.status == PageStatus.ACTIVE, "a retrieved page is resident again"


def test_retrieve_missing_page_returns_none(storage):
    assert storage.retrieve("file:nope.py") is None


def test_store_replaces_an_existing_row(storage):
    storage.store(make_page(content="first"))
    storage.store(make_page(content="second"))

    assert storage.get_disk_stats()["rows"] == 1
    assert storage.retrieve("file:a.py").content == "second"


def test_delete_reports_whether_a_row_was_removed(storage):
    storage.store(make_page())
    assert storage.delete("file:a.py") is True
    assert storage.delete("file:a.py") is False
    assert storage.retrieve("file:a.py") is None


def test_delete_page_alias_still_works(storage):
    storage.store(make_page())
    assert storage.delete_page("file:a.py") is True


def test_disk_stats_count_rows_and_tokens(storage):
    for index in range(3):
        storage.store(make_page(f"file:{index}.py", content="x" * 400))

    stats = storage.get_disk_stats()
    assert stats["rows"] == 3
    assert stats["tokens"] == 300
    assert storage.get_total_swapped_tokens() == stats["tokens"]


def test_clear_removes_every_row_and_reports_the_count(storage):
    for index in range(5):
        storage.store(make_page(f"file:{index}.py"))
    assert storage.clear() == 5
    assert storage.get_disk_stats()["rows"] == 0


def test_prune_keeps_only_the_named_pages(storage):
    for index in range(5):
        storage.store(make_page(f"file:{index}.py"))

    removed = storage.prune(keep_page_ids=["file:1.py", "file:3.py"])
    assert removed == 3
    assert storage.retrieve("file:1.py") is not None
    assert storage.retrieve("file:0.py") is None


def test_prune_with_no_ids_clears_everything(storage):
    storage.store(make_page())
    assert storage.prune([]) == 1


def test_list_swapped_orders_newest_first(storage):
    for index in range(3):
        storage.store(make_page(f"file:{index}.py"))

    listed = storage.list_swapped()
    assert len(listed) == 3
    assert {row["page_id"] for row in listed} == {"file:0.py", "file:1.py", "file:2.py"}
    assert listed == sorted(listed, key=lambda row: row["swapped_at"], reverse=True)


def test_store_raises_rather_than_reporting_a_silent_success(tmp_path, monkeypatch):
    """
    The kernel drops its in-memory copy the moment `store` returns, so a write that
    quietly failed would lose the page. Failures must raise.
    """
    storage = SwapStorage(str(tmp_path / "swap.db"))

    def explode(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(storage, "_connect", explode)
    with pytest.raises(SwapStorageError, match="Failed to swap out"):
        storage.store(make_page())


def test_unreadable_database_path_raises_on_open(tmp_path):
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    with pytest.raises(SwapStorageError):
        SwapStorage(str(directory))


def test_corrupt_metadata_does_not_break_retrieval(storage):
    storage.store(make_page())
    connection = storage._connect()
    connection.execute("UPDATE swap_pages SET metadata_json = ?", ("{not json",))
    connection.commit()

    restored = storage.retrieve("file:a.py")
    assert restored is not None
    assert restored.metadata == {}


def test_unknown_tier_falls_back_instead_of_raising(storage):
    storage.store(make_page())
    connection = storage._connect()
    connection.execute("UPDATE swap_pages SET tier = ?", ("L9_IMAGINARY",))
    connection.commit()

    restored = storage.retrieve("file:a.py")
    assert restored is not None
    assert restored.tier == PageTier.L1_WORKING_RAM


def test_a_v1_database_is_migrated_in_place(tmp_path):
    """An existing database from 0.1.0 must be upgraded, not misread."""
    path = str(tmp_path / "old.db")
    legacy = sqlite3.connect(path)
    legacy.execute(
        """
        CREATE TABLE swap_pages (
            page_id TEXT PRIMARY KEY, title TEXT NOT NULL, tier TEXT NOT NULL,
            content TEXT NOT NULL, tombstone TEXT, token_count INTEGER NOT NULL,
            access_count INTEGER NOT NULL, metadata_json TEXT,
            created_at REAL NOT NULL, swapped_at REAL NOT NULL
        )
        """
    )
    legacy.execute(
        "INSERT INTO swap_pages VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("file:old.py", "old.py", "L1_WORKING", "legacy content", None, 10, 1, "{}", 0.0, 0.0),
    )
    legacy.commit()
    legacy.close()

    storage = SwapStorage(path)
    restored = storage.retrieve("file:old.py")
    assert restored is not None
    assert restored.content == "legacy content"
    assert restored.compacted is False


def test_connections_are_per_thread(tmp_path):
    """
    sqlite3 connections cannot be shared across threads, and the proxy serves
    requests on a thread pool.
    """
    storage = SwapStorage(str(tmp_path / "swap.db"))
    storage.store(make_page())
    errors = []

    def worker(index: int) -> None:
        try:
            storage.store(make_page(f"file:thread{index}.py"))
            assert storage.retrieve(f"file:thread{index}.py") is not None
        except Exception as error:  # surfaced on the main thread below
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"threaded access failed: {errors}"
    assert storage.get_disk_stats()["rows"] == 7


def test_close_is_idempotent(storage):
    storage.store(make_page())
    storage.close()
    storage.close()
    assert storage.retrieve("file:a.py") is not None, "storage reopens on next use"
