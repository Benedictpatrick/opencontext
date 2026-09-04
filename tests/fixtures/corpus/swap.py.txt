"""
SQLite-backed swap storage.

Once a page is swapped out the kernel releases its in-memory content, so this
database holds the only copy. Writes therefore raise on failure rather than
reporting success, and the schema is versioned so an old database is detected
instead of misread.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from contextos.core.types import ContextPage, PageStatus, PageTier

SCHEMA_VERSION = 2


class SwapStorageError(RuntimeError):
    """Raised when swap storage cannot complete an operation."""


class SwapStorage:
    """Durable store for pages that are not currently in the context window."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.join(os.getcwd(), ".contextos")
            os.makedirs(base_dir, exist_ok=True)
            db_path = os.path.join(base_dir, "swap.db")
        else:
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        self.db_path = db_path
        # One connection per thread, reused across operations. Opening a fresh
        # connection per call (and re-issuing the PRAGMAs each time) dominated
        # page-fault latency; reuse takes a fault from milliseconds to microseconds.
        # sqlite3 connections are not safe to share between threads, and the proxy
        # serves requests on a thread pool, so the handle is thread-local.
        self._local = threading.local()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            return connection

        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        # WAL keeps the dashboard's reads from blocking the kernel's writes.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        self._local.connection = connection
        return connection

    def close(self) -> None:
        """Close this thread's connection. Safe to call more than once."""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            try:
                connection.close()
            finally:
                self._local.connection = None

    def _init_db(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS swap_pages (
                        page_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        tier TEXT NOT NULL,
                        content TEXT NOT NULL,
                        tombstone TEXT,
                        token_count INTEGER NOT NULL,
                        original_token_count INTEGER NOT NULL DEFAULT 0,
                        compacted INTEGER NOT NULL DEFAULT 0,
                        access_count INTEGER NOT NULL,
                        metadata_json TEXT,
                        created_at REAL NOT NULL,
                        swapped_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_swapped_at ON swap_pages (swapped_at)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS swap_meta (key TEXT PRIMARY KEY, value TEXT)"
                )
                self._migrate(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO swap_meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise SwapStorageError(f"Cannot open swap database at {self.db_path}: {error}") from error

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Add columns introduced after v1 to an existing database."""
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(swap_pages)")}
        for column, ddl in (
            ("original_token_count", "INTEGER NOT NULL DEFAULT 0"),
            ("compacted", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if column not in existing:
                connection.execute(f"ALTER TABLE swap_pages ADD COLUMN {column} {ddl}")

    def store(self, page: ContextPage) -> bool:
        """
        Persist a page. Raises `SwapStorageError` if the write does not land.

        The kernel drops the in-memory copy immediately after this returns, so a
        silent failure here would lose data.
        """
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO swap_pages (
                        page_id, title, tier, content, tombstone, token_count,
                        original_token_count, compacted, access_count, metadata_json,
                        created_at, swapped_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        page.id,
                        page.title,
                        page.tier.value,
                        page.content,
                        page.tombstone,
                        page.token_count,
                        page.original_token_count or page.token_count,
                        int(page.compacted),
                        page.access_count,
                        json.dumps(page.metadata),
                        page.created_at,
                        time.time(),
                    ),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise SwapStorageError(f"Failed to swap out page '{page.id}': {error}") from error

        return True

    def retrieve(self, page_id: str) -> Optional[ContextPage]:
        """Read a page back. Returns None when the page is not in swap."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM swap_pages WHERE page_id = ?", (page_id,)
                ).fetchone()
        except sqlite3.Error as error:
            raise SwapStorageError(f"Failed to read page '{page_id}': {error}") from error

        if row is None:
            return None

        try:
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        except json.JSONDecodeError:
            metadata = {}

        try:
            tier = PageTier(row["tier"])
        except ValueError:
            tier = PageTier.L1_WORKING_RAM

        return ContextPage(
            id=row["page_id"],
            title=row["title"],
            tier=tier,
            status=PageStatus.ACTIVE,
            content=row["content"],
            tombstone=None,
            token_count=row["token_count"],
            original_token_count=row["original_token_count"] or row["token_count"],
            compacted=bool(row["compacted"]),
            access_count=row["access_count"],
            created_at=row["created_at"],
            last_accessed_at=time.time(),
            metadata=metadata,
        )

    def delete(self, page_id: str) -> bool:
        """Remove a page from swap. Returns True when a row was removed."""
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM swap_pages WHERE page_id = ?", (page_id,)
                )
                connection.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as error:
            raise SwapStorageError(f"Failed to delete page '{page_id}': {error}") from error

    def delete_page(self, page_id: str) -> bool:
        """Alias for `delete`, kept for callers using the older name."""
        return self.delete(page_id)

    def list_swapped(self) -> List[Dict[str, Any]]:
        """Summarise every row in swap, newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT page_id, title, tier, token_count, access_count, swapped_at
                FROM swap_pages ORDER BY swapped_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_disk_stats(self) -> Dict[str, int]:
        """
        Row count and token total for the database as it exists on disk.

        Reported separately from live kernel metrics: this database can contain
        rows from previous runs, and conflating the two made swap usage appear to
        grow without bound.
        """
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS rows, COALESCE(SUM(token_count), 0) AS tokens FROM swap_pages"
                ).fetchone()
            return {"rows": int(row["rows"]), "tokens": int(row["tokens"])}
        except sqlite3.Error:
            return {"rows": 0, "tokens": 0}

    def get_total_swapped_tokens(self) -> int:
        """Token total across the database. Prefer `get_disk_stats`."""
        return self.get_disk_stats()["tokens"]

    def clear(self) -> int:
        """Delete every row. Returns how many rows were removed."""
        try:
            connection = self._connect()
            cursor = connection.execute("DELETE FROM swap_pages")
            connection.commit()
            removed = cursor.rowcount
            # VACUUM cannot run inside a transaction.
            connection.execute("VACUUM")
            return max(0, removed)
        except sqlite3.Error as error:
            raise SwapStorageError(f"Failed to clear swap storage: {error}") from error

    def prune(self, keep_page_ids: Optional[List[str]] = None) -> int:
        """
        Drop rows for pages the kernel no longer tracks.

        Used to clean a database left behind by an earlier run: pass the ids that
        are currently swapped out and everything else is removed.
        """
        keep = list(keep_page_ids or [])
        try:
            with self._connect() as connection:
                if keep:
                    placeholders = ",".join("?" * len(keep))
                    cursor = connection.execute(
                        f"DELETE FROM swap_pages WHERE page_id NOT IN ({placeholders})", keep
                    )
                else:
                    cursor = connection.execute("DELETE FROM swap_pages")
                connection.commit()
                return max(0, cursor.rowcount)
        except sqlite3.Error as error:
            raise SwapStorageError(f"Failed to prune swap storage: {error}") from error
