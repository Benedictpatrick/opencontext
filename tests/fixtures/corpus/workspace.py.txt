"""
Workspace scanner.

Walks a project tree and ingests text files into the kernel, skipping build
output, dependency directories and binary content.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

from opencontext.core.ids import file_page_id
from opencontext.core.tokens import estimate_tokens
from opencontext.core.kernel import ContextKernel
from opencontext.core.types import PageTier

# Files pinned to L0 only when they sit at the project root. Matching on bare
# filename at any depth pinned every package.json in a monorepo into a tier that
# cannot be evicted, which quietly pushed the kernel over budget.
ROOT_PINNED_FILES = {"readme.md", "pyproject.toml", "package.json", "cargo.toml", "go.mod"}

# Share of the budget the scanner is willing to pin. Beyond this, root files are
# ingested as ordinary evictable pages.
MAX_PINNED_BUDGET_SHARE = 0.25


class WorkspaceScanner:
    """Synchronises a directory tree into the kernel."""

    IGNORE_DIRS = {
        ".git", ".hg", ".svn", ".opencontext", "__pycache__", "node_modules", "venv",
        ".venv", "env", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
        ".idea", ".vscode", "dist", "build", "target", "out", ".next", ".nuxt",
        "coverage", "htmlcov", ".gradle", "vendor", "Pods", ".terraform",
    }

    IGNORE_EXTS = {
        ".pyc", ".pyo", ".pyd", ".class", ".o", ".obj", ".a", ".lib",
        ".db", ".sqlite", ".sqlite3", ".log", ".lock",
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp", ".tiff",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".jar",
        ".mp3", ".mp4", ".mov", ".avi", ".webm", ".wav", ".pdf",
    }

    def __init__(self, kernel: ContextKernel, root_dir: Optional[str] = None):
        self.kernel = kernel
        self.root_dir = os.path.abspath(root_dir or os.getcwd())

    def scan_and_ingest(
        self,
        max_file_size_kb: int = 250,
        max_files: int = 2000,
    ) -> Dict[str, int]:
        """
        Walk the tree and ingest every eligible text file.

        Returns counts for what was scanned, skipped and pinned. `max_files` caps
        the walk so a scan of an unexpectedly large tree terminates.
        """
        scanned = 0
        skipped_binary = 0
        skipped_large = 0
        skipped_error = 0
        pinned = 0
        total_tokens = 0
        pinned_tokens = 0
        pinned_budget = int(self.kernel.token_budget * MAX_PINNED_BUDGET_SHARE)

        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [
                d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")
            ]

            for filename in sorted(files):
                if scanned >= max_files:
                    break

                if os.path.splitext(filename)[1].lower() in self.IGNORE_EXTS:
                    continue

                full_path = os.path.join(root, filename)
                try:
                    if os.path.getsize(full_path) / 1024 > max_file_size_kb:
                        skipped_large += 1
                        continue

                    content = self._read_text(full_path)
                    if content is None:
                        skipped_binary += 1
                        continue

                    at_root = os.path.dirname(os.path.abspath(full_path)) == self.root_dir
                    # The file's own cost counts against the cap. Testing only what
                    # had already been pinned let a single large README blow straight
                    # through the limit, because the running total started at zero.
                    file_tokens = estimate_tokens(content)
                    should_pin = (
                        at_root
                        and filename.lower() in ROOT_PINNED_FILES
                        and pinned_tokens + file_tokens <= pinned_budget
                    )
                    tier = PageTier.L0_PINNED if should_pin else PageTier.L1_WORKING_RAM

                    page = self.kernel.allocate_page(
                        page_id=file_page_id(full_path, self.root_dir),
                        title=os.path.relpath(full_path, self.root_dir).replace("\\", "/"),
                        content=content,
                        tier=tier,
                        auto_compact=False,
                        metadata={
                            "abs_path": full_path,
                            "size_bytes": len(content),
                            "modified_at": os.path.getmtime(full_path),
                        },
                    )

                    scanned += 1
                    total_tokens += page.token_count
                    if should_pin:
                        pinned += 1
                        pinned_tokens += page.token_count

                except (OSError, ValueError):
                    skipped_error += 1
                    continue

            if scanned >= max_files:
                break

        metrics = self.kernel.get_metrics()
        return {
            "files_scanned": scanned,
            "files_skipped_binary": skipped_binary,
            "files_skipped_large": skipped_large,
            "files_skipped_error": skipped_error,
            "files_pinned": pinned,
            "total_tokens": total_tokens,
            "working_tokens": metrics.working_tokens,
            "swapped_tokens": metrics.swapped_tokens,
        }

    @staticmethod
    def _read_text(path: str) -> Optional[str]:
        """
        Read a file as text, or return None if it is binary.

        Extension checks miss extensionless binaries, so the leading bytes are
        sniffed for NUL before the file is treated as source.
        """
        try:
            with open(path, "rb") as handle:
                probe = handle.read(8192)
            if b"\x00" in probe:
                return None
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            return None

    def list_tracked_files(self) -> List[str]:
        """Page ids for every file this scanner has ingested into the kernel."""
        prefix = "file:"
        return sorted(pid for pid in self.kernel.pages if pid.startswith(prefix))
