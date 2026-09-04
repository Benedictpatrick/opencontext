"""
OpenContext core types: memory tiers, context pages, paging events and metrics.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PageTier(str, Enum):
    """Where a page sits in the memory hierarchy."""

    L0_PINNED = "L0_PINNED"        # System instructions, invariants, active goal. Never evicted.
    L1_WORKING_RAM = "L1_WORKING"  # Active code, symbols and tool schemas.
    L2_EPISODIC = "L2_EPISODIC"    # Conversation history and decision summaries. Evicted first.
    L3_SWAP = "L3_SWAP"            # Reserved. Swapping changes `status`, not `tier`, so that a
                                   # rehydrated page returns to the tier it came from.


class PageStatus(str, Enum):
    """Current residency of a page."""

    ACTIVE = "ACTIVE"      # Content is in the context window.
    SWAPPED = "SWAPPED"    # Content is on disk; only a tombstone is in context.
    COMPACTED = "COMPACTED"  # Reserved. Compaction is recorded via `ContextPage.compacted`.
    EVICTED = "EVICTED"      # Reserved. `delete_page` removes the page outright.


class ContextPage(BaseModel):
    """An atomic unit of context managed by the kernel."""

    id: str = Field(..., description="Unique identifier, e.g. 'file:src/auth.py'")
    title: str = Field(..., description="Human-readable label")
    tier: PageTier = Field(default=PageTier.L1_WORKING_RAM)
    status: PageStatus = Field(default=PageStatus.ACTIVE)
    content: str = Field(..., description="Raw text or code. Empty while SWAPPED.")
    tombstone: Optional[str] = Field(
        default=None, description="Compact pointer left in context while swapped out"
    )
    token_count: int = Field(default=0, description="Estimated tokens of the full content")
    compacted: bool = Field(
        default=False, description="True when `content` is a compacted form of the original"
    )
    original_token_count: int = Field(
        default=0, description="Token count before compaction; equals token_count when not compacted"
    )
    access_count: int = Field(default=1, description="Times this page has been referenced")
    last_accessed_at: float = Field(default_factory=time.time)
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def mark_access(self) -> None:
        """Record a reference, for LRU ordering."""
        self.access_count += 1
        self.last_accessed_at = time.time()

    def get_context_representation(self) -> str:
        """What the model actually sees: full content when resident, tombstone when swapped."""
        if self.status == PageStatus.SWAPPED and self.tombstone:
            return self.tombstone
        return self.content

    @property
    def is_resident(self) -> bool:
        """True when the page's content occupies the context window."""
        return self.status == PageStatus.ACTIVE


class PagingEventType(str, Enum):
    """Kinds of kernel activity recorded in telemetry."""

    PAGE_ALLOCATED = "PAGE_ALLOCATED"
    PAGE_SWAPPED_OUT = "PAGE_SWAPPED_OUT"
    PAGE_FAULT_HIT = "PAGE_FAULT_HIT"
    PAGE_COMPACTED = "PAGE_COMPACTED"
    PAGE_EVICTED = "PAGE_EVICTED"
    LEAK_WARNING = "LEAK_WARNING"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    TIER_PROMOTED = "TIER_PROMOTED"
    TIER_DEMOTED = "TIER_DEMOTED"
    BUDGET_CHANGED = "BUDGET_CHANGED"
    DOOM_LOOP_PREVENTED = "DOOM_LOOP_PREVENTED"


class PagingEvent(BaseModel):
    """A single telemetry record."""

    timestamp: float = Field(default_factory=time.time)
    event_type: PagingEventType
    page_id: str
    tokens_saved: int = 0
    description: str

    @property
    def formatted_time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class MemoryMetrics(BaseModel):
    """
    Kernel telemetry.

    Live figures describe the *current* process. Disk figures describe the swap
    database, which may contain rows from earlier runs; the two are reported
    separately so they can never be mistaken for one another.
    """

    token_budget: int = 16000
    working_tokens: int = 0
    swapped_tokens: int = Field(
        default=0, description="Tokens held by pages this kernel currently has swapped out"
    )
    swap_disk_tokens: int = Field(
        default=0, description="Tokens across every row in the swap database, including prior runs"
    )
    swap_disk_rows: int = Field(default=0, description="Row count in the swap database")
    total_tokens_saved: int = Field(
        default=0, description="Tokens removed from the context window by this kernel"
    )
    budget_utilization_pct: float = 0.0
    over_budget: bool = Field(
        default=False, description="True when pinned content alone exceeds the budget"
    )
    l0_pages: int = 0
    l1_pages: int = 0
    l2_pages: int = 0
    l3_pages: int = Field(default=0, description="Pages this kernel currently has swapped out")
    total_page_faults: int = 0
    avg_page_fault_ms: float = Field(
        default=0.0,
        description="Mean wall-clock time of recent page faults, measured. 0.0 when none have occurred.",
    )
    recent_events: List[PagingEvent] = Field(default_factory=list)
