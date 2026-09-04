"""
Session persistence.

ContextOS has no daemon: each command builds its own kernel. Without persistence
that means pinning a page, setting a budget or arranging a working set is undone
the moment you quit, which makes those actions close to pointless. A session file
carries that arrangement across runs.

Where each page's content comes from on restore:

  * **Swapped** — the content is in `swap.db`, which already survives the process.
    The session file records only that the page was swapped.
  * **Resident, backed by a file** — re-read from disk, so the restored page
    reflects the current file rather than a stale copy.
  * **Resident, not backed by a file** — a pinned instruction, an ingested
    traceback, a conversation turn. Nothing else holds this, so the content is
    written into the session file itself.

A page whose source has vanished since the session was saved is dropped and
reported. It is never restored with empty content: a page that claims to hold
something it does not is worse than an absent one.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from contextos.core.kernel import ContextKernel
from contextos.core.types import ContextPage, PageStatus, PageTier

SESSION_VERSION = 2
DEFAULT_SESSION_PATH = os.path.join(".contextos", "session.json")

# Content this large is not inlined into the session file; the page is dropped
# instead, with a note. Keeps the file readable and bounded.
MAX_INLINE_CONTENT_CHARS = 256 * 1024


class SessionError(RuntimeError):
    """Raised when a session file cannot be read or written."""


def _source_path(page: ContextPage) -> Optional[str]:
    """The file a page was read from, if it is still on disk."""
    for key in ("abs_path", "filepath"):
        candidate = page.metadata.get(key)
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def save_session(
    kernel: ContextKernel, path: str = DEFAULT_SESSION_PATH, root_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Write the kernel's arrangement to `path`.

    Returns a summary of what was saved. Writes atomically, so an interrupted save
    cannot leave a truncated file that fails to parse on the next start.
    """
    pages: List[Dict[str, Any]] = []
    inlined = 0
    from_disk = 0
    from_swap = 0

    for page in kernel.pages.values():
        record: Dict[str, Any] = {
            "id": page.id,
            "title": page.title,
            "tier": page.tier.value,
            "status": page.status.value,
            "token_count": page.token_count,
            "original_token_count": page.original_token_count,
            "compacted": page.compacted,
            "access_count": page.access_count,
            "created_at": page.created_at,
            "last_accessed_at": page.last_accessed_at,
            "metadata": page.metadata,
        }

        if page.status == PageStatus.SWAPPED:
            record["source"] = "swap"
            from_swap += 1
        elif _source_path(page):
            record["source"] = "file"
            record["source_path"] = _source_path(page)
            from_disk += 1
        elif len(page.content) <= MAX_INLINE_CONTENT_CHARS:
            record["source"] = "inline"
            record["content"] = page.content
            inlined += 1
        else:
            # Too large to inline and with no other home. Recorded as unrecoverable
            # rather than saved as an empty shell that would restore as a lie.
            record["source"] = "dropped"
            record["dropped_reason"] = "content exceeds the inline limit and has no file source"

        pages.append(record)

    payload = {
        "version": SESSION_VERSION,
        "saved_at": time.time(),
        "root_dir": os.path.abspath(root_dir) if root_dir else os.getcwd(),
        "token_budget": kernel.token_budget,
        "total_tokens_saved": kernel.total_tokens_saved,
        "total_page_faults": kernel.total_page_faults,
        "pages": pages,
    }

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
    except OSError as error:
        raise SessionError(f"Cannot write session to {path}: {error}") from error
    finally:
        if os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass

    return {
        "path": path,
        "pages_saved": len(pages),
        "from_file": from_disk,
        "from_swap": from_swap,
        "inlined": inlined,
        "token_budget": kernel.token_budget,
    }


def session_exists(path: str = DEFAULT_SESSION_PATH) -> bool:
    return os.path.isfile(path)


def restore_session(kernel: ContextKernel, path: str = DEFAULT_SESSION_PATH) -> Dict[str, Any]:
    """
    Rebuild a saved arrangement into `kernel`.

    Returns a summary including anything that could not be restored. The budget is
    re-applied and eviction re-run, because files may have grown since the session
    was saved and the restored set can no longer fit.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {"restored": 0, "dropped": [], "error": "no session file"}
    except (OSError, json.JSONDecodeError) as error:
        raise SessionError(f"Cannot read session from {path}: {error}") from error

    if payload.get("version") != SESSION_VERSION:
        return {
            "restored": 0,
            "dropped": [],
            "error": (
                f"session file is version {payload.get('version')}, "
                f"this build writes version {SESSION_VERSION}"
            ),
        }

    kernel.token_budget = max(100, int(payload.get("token_budget", kernel.token_budget)))
    kernel.total_tokens_saved = int(payload.get("total_tokens_saved", 0))
    kernel.total_page_faults = int(payload.get("total_page_faults", 0))

    restored = 0
    dropped: List[Dict[str, str]] = []

    for record in payload.get("pages", []):
        page_id = record.get("id")
        if not page_id:
            continue

        source = record.get("source")
        content = ""
        status = PageStatus.ACTIVE

        if source == "swap":
            stored = kernel.swap.retrieve(page_id)
            if stored is None:
                dropped.append({"id": page_id, "reason": "swapped content is no longer on disk"})
                continue
            # Left swapped, exactly as it was: the tombstone is regenerated below.
            content = stored.content
            status = PageStatus.SWAPPED

        elif source == "file":
            source_path = record.get("source_path", "")
            if not source_path or not os.path.isfile(source_path):
                dropped.append({"id": page_id, "reason": f"file no longer exists: {source_path}"})
                continue
            try:
                with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
            except OSError as error:
                dropped.append({"id": page_id, "reason": f"file unreadable: {error}"})
                continue

        elif source == "inline":
            content = record.get("content", "")
            if not content:
                dropped.append({"id": page_id, "reason": "inline content was empty"})
                continue

        else:
            dropped.append(
                {"id": page_id, "reason": record.get("dropped_reason", "no recoverable source")}
            )
            continue

        try:
            tier = PageTier(record.get("tier", PageTier.L1_WORKING_RAM.value))
        except ValueError:
            tier = PageTier.L1_WORKING_RAM

        page = ContextPage(
            id=page_id,
            title=record.get("title", page_id),
            tier=tier,
            status=PageStatus.ACTIVE,
            content=content,
            token_count=record.get("token_count", 0),
            original_token_count=record.get("original_token_count", 0),
            compacted=bool(record.get("compacted", False)),
            access_count=int(record.get("access_count", 1)),
            created_at=float(record.get("created_at", time.time())),
            last_accessed_at=float(record.get("last_accessed_at", time.time())),
            metadata=record.get("metadata", {}) or {},
        )
        kernel.pages[page_id] = page

        if status == PageStatus.SWAPPED:
            # Re-derive the on-disk state through the normal path so the tombstone
            # and accounting match a page swapped during this run.
            page.status = PageStatus.ACTIVE
            kernel.page_out(page_id)

        restored += 1

    # Files may have grown since the session was written, so the restored set can
    # exceed the budget it was saved under. Re-run eviction; if pinned content alone
    # no longer fits, this raises the BUDGET_EXCEEDED event rather than staying quiet.
    kernel._enforce_budget()

    return {
        "restored": restored,
        "dropped": dropped,
        "token_budget": kernel.token_budget,
        "saved_at": payload.get("saved_at"),
        "root_dir": payload.get("root_dir"),
    }
