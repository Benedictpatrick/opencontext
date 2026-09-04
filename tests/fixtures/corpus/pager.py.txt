"""
High-level ingestion API over the kernel.

The pager is what applications talk to: it turns files, instructions, error traces
and conversation turns into pages, and resolves references in an incoming prompt
before handing back the assembled context.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from contextos.core.compactor import CodeOutlineCompactor
from contextos.core.ids import error_page_id, file_page_id, rule_page_id, turn_page_id
from contextos.core.kernel import ContextKernel
from contextos.core.types import ContextPage, PageTier

# Files longer than this are folded to an outline when a focus symbol is given.
OUTLINE_THRESHOLD_LINES = 150


class ContextPager:
    """Ingestion and prompt processing on top of a `ContextKernel`."""

    def __init__(self, kernel: Optional[ContextKernel] = None, root_dir: Optional[str] = None):
        self.kernel = kernel or ContextKernel()
        self.root_dir = os.path.abspath(root_dir) if root_dir else os.getcwd()

    def pin_instruction(self, rule_id: str, title: str, instruction: str) -> ContextPage:
        """Pin a rule or invariant to L0, where it is never evicted."""
        return self.kernel.allocate_page(
            page_id=rule_page_id(rule_id),
            title=title,
            content=instruction,
            tier=PageTier.L0_PINNED,
            auto_compact=False,
        )

    def ingest_file(
        self,
        filepath: str,
        content: Optional[str] = None,
        focus_symbol: str = "",
        tier: PageTier = PageTier.L1_WORKING_RAM,
    ) -> ContextPage:
        """
        Ingest a source file.

        When the file is long and `focus_symbol` is given, it is folded to an
        outline with that symbol left expanded. The page id is built by
        `core.ids.file_page_id`, so a file ingested here and the same file found by
        the workspace scanner resolve to one page.
        """
        if content is None:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
            else:
                content = f"# {filepath} (not found on disk when referenced)"

        title = os.path.basename(filepath) or filepath
        metadata: Dict[str, Any] = {"filepath": filepath, "focus_symbol": focus_symbol}

        if focus_symbol and len(content.splitlines()) > OUTLINE_THRESHOLD_LINES:
            outlined, original_tokens, outline_tokens = CodeOutlineCompactor.compact_code(
                title, content, focus_symbol
            )
            if outline_tokens < original_tokens:
                content = outlined
                metadata["outlined"] = True
                metadata["outline_tokens_saved"] = original_tokens - outline_tokens

        return self.kernel.allocate_page(
            page_id=file_page_id(filepath, self.root_dir),
            title=title,
            content=content,
            tier=tier,
            auto_compact=False,
            metadata=metadata,
        )

    def ingest_traceback(self, error_log: str, title: str = "Execution error") -> ContextPage:
        """
        Ingest an error trace, compacting it to user frames plus root cause.

        This is the one ingest path that opts into compaction, because the caller
        has asserted the content is an error trace.
        """
        return self.kernel.allocate_page(
            page_id=error_page_id(error_log),
            title=title,
            content=error_log,
            tier=PageTier.L1_WORKING_RAM,
            auto_compact=True,
        )

    def ingest_conversation_turn(self, role: str, message: str) -> ContextPage:
        """Ingest a conversation turn into L2, the first tier to be evicted."""
        return self.kernel.allocate_page(
            page_id=turn_page_id(role, message),
            title=f"{role.capitalize()} message",
            content=f"[{role.upper()}]: {message}",
            tier=PageTier.L2_EPISODIC,
            auto_compact=False,
        )

    def process_incoming_prompt(self, user_prompt: str) -> str:
        """Resolve any references to swapped pages, then return the assembled context."""
        self.kernel.touch_or_fault(user_prompt)
        return self.kernel.assemble_context()

    def process_incoming_prompt_verbose(self, user_prompt: str) -> Dict[str, Any]:
        """As `process_incoming_prompt`, but also reports which pages were restored."""
        rehydrated: List[str] = self.kernel.touch_or_fault(user_prompt)
        context = self.kernel.assemble_context()
        return {
            "context": context,
            "rehydrated": rehydrated,
            "context_tokens": self.kernel.get_metrics().working_tokens,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Status summary for external tools and dashboards."""
        metrics = self.kernel.get_metrics()
        return {
            "working_tokens": metrics.working_tokens,
            "swapped_tokens": metrics.swapped_tokens,
            "swap_disk_tokens": metrics.swap_disk_tokens,
            "budget": metrics.token_budget,
            "utilization_pct": metrics.budget_utilization_pct,
            "over_budget": metrics.over_budget,
            "active_pages": metrics.l0_pages + metrics.l1_pages + metrics.l2_pages,
            "swapped_pages": metrics.l3_pages,
            "tokens_saved": metrics.total_tokens_saved,
            "page_faults": metrics.total_page_faults,
        }
