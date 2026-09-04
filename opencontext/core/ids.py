"""
Page identifier construction.

Every subsystem that creates a page must build its id here. Two ingest paths
previously disagreed — the workspace scanner used forward slashes while the pager
used `os.path.normpath`, which produces backslashes on Windows — so the same file
ingested twice became two pages with two swap rows.

Ids are also required to be stable across processes. Identifiers derived from
Python's built-in `hash()` change on every interpreter start because of hash
randomisation, which makes a persisted swap row unreachable after a restart, and
folding them into a small integer range let unrelated content collide onto one id
and silently overwrite it.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional


def normalize_path(path: str, root: Optional[str] = None) -> str:
    """
    Canonical path form for ids: forward slashes, relative to `root` when given.

    Falls back to the absolute path when the file lies outside `root` (or on a
    different drive, where `os.path.relpath` raises on Windows).
    """
    if root:
        try:
            relative = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
            if not relative.startswith(".."):
                return relative.replace("\\", "/")
        except ValueError:
            pass
    return os.path.normpath(path).replace("\\", "/")


def file_page_id(path: str, root: Optional[str] = None) -> str:
    """Id for a source file. The single definition used by every ingest path."""
    return f"file:{normalize_path(path, root)}"


def content_digest(text: str, length: int = 12) -> str:
    """
    Stable content fingerprint.

    SHA-1 over the text: identical across processes and machines, and wide enough
    that unrelated content does not collide onto the same page.
    """
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def error_page_id(error_text: str) -> str:
    """Id for an ingested error trace, keyed on its content."""
    return f"error:{content_digest(error_text)}"


def turn_page_id(role: str, message: str) -> str:
    """Id for a conversation turn, keyed on role and content."""
    return f"turn:{role.lower()}:{content_digest(message)}"


def rule_page_id(rule_id: str) -> str:
    """Id for a pinned instruction."""
    return f"rule:{rule_id}"
