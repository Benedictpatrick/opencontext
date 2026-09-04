"""
The ContextOS virtual memory kernel.

Enforces a token budget over a set of context pages: allocates pages, evicts cold
ones to disk under an LRU policy, rehydrates them on demand, and reports what it
did. All accounting is measured, never assumed — the kernel does not report a
saving it did not achieve.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from contextos.core.compactor import TracebackCompactor
from contextos.core.tokens import estimate_tokens
from contextos.core.types import (
    ContextPage,
    MemoryMetrics,
    PageStatus,
    PageTier,
    PagingEvent,
    PagingEventType,
)
from contextos.storage.swap import SwapStorage

MAX_RETAINED_EVENTS = 200
DOOM_LOOP_THRESHOLD = 3

# Automatic eviction skips a page unless swapping it actually frees this many
# tokens. A tombstone is around 24 tokens, so evicting a page smaller than that
# makes the context window *larger* — the kernel would be paying to save nothing.
MIN_AUTO_SWAP_BENEFIT_TOKENS = 16

# Small episodic pages are individually not worth evicting (above), but a long
# conversation is made entirely of them, and they must not be allowed to crowd out
# the source the session is actually about. Once they exceed this share of the
# budget the oldest are coalesced into a single page, which pays one tombstone for
# all of them and is then evictable on the ordinary path.
MAX_EPISODIC_BUDGET_SHARE = 0.35

# The most recent turns stay whole: a digest is for history, not for the exchange
# currently in progress.
EPISODIC_TURNS_KEPT_WHOLE = 6


class ContextKernel:
    """Memory management unit for an LLM context window."""

    def __init__(self, token_budget: int = 16000, swap_storage: Optional[SwapStorage] = None):
        self.token_budget = max(100, int(token_budget))
        self.swap = swap_storage or SwapStorage()
        self.pages: Dict[str, ContextPage] = {}
        self.events: List[PagingEvent] = []
        self.total_page_faults = 0
        self.total_tokens_saved = 0
        self._error_signature_counts: Dict[str, int] = {}
        # Wall-clock duration of recent page faults, so reported latency is measured
        # rather than asserted. Bounded so a long session cannot grow it without limit.
        self._page_fault_durations_ms: List[float] = []

    # -- allocation -------------------------------------------------------------

    def allocate_page(
        self,
        page_id: str,
        title: str,
        content: str,
        tier: PageTier = PageTier.L1_WORKING_RAM,
        auto_compact: bool = False,
        metadata: Optional[Dict] = None,
    ) -> ContextPage:
        """
        Place a page in working context, evicting cold pages if the budget requires it.

        `auto_compact` defaults to False. Compaction is lossy, so it is opt-in:
        callers that know they are handing over an error trace (see
        `ContextPager.ingest_traceback`) request it explicitly. The previous
        default of True, combined with a loose traceback test, silently discarded
        content from ordinary text.
        """
        metadata = metadata or {}
        original_tokens = estimate_tokens(content)
        token_count = original_tokens
        compacted = False

        if auto_compact and TracebackCompactor.is_traceback(content):
            compacted_text, _, compacted_tokens = TracebackCompactor.compact(content)
            if compacted_tokens < original_tokens:
                saved = original_tokens - compacted_tokens
                content = compacted_text
                token_count = compacted_tokens
                compacted = True
                self.total_tokens_saved += saved
                self._log_event(
                    PagingEventType.PAGE_COMPACTED,
                    page_id,
                    saved,
                    f"Compacted trace in '{title}': {original_tokens} -> {compacted_tokens} tokens",
                )

            # Signature tracking is independent of whether compaction won. An agent
            # retrying the same short failure is still looping, and a trace too small
            # to compact is exactly the case where the loop is cheapest to catch.
            self._record_error_signature(page_id, compacted_text)

        existing = self.pages.get(page_id)
        if existing is not None:
            existing.title = title
            existing.content = content
            existing.token_count = token_count
            existing.original_token_count = original_tokens
            existing.compacted = compacted
            existing.status = PageStatus.ACTIVE
            existing.tombstone = None
            if metadata:
                existing.metadata.update(metadata)
            existing.mark_access()
            page = existing
        else:
            page = ContextPage(
                id=page_id,
                title=title,
                tier=tier,
                status=PageStatus.ACTIVE,
                content=content,
                token_count=token_count,
                original_token_count=original_tokens,
                compacted=compacted,
                metadata=metadata,
            )
            self.pages[page_id] = page

        self._log_event(
            PagingEventType.PAGE_ALLOCATED,
            page_id,
            0,
            f"Allocated '{title}' ({token_count:,} tokens) in {page.tier.value}",
        )

        self._enforce_budget()
        return page

    def _record_error_signature(self, page_id: str, compacted_content: str) -> None:
        """
        Track repeated failures so an agent stuck retrying the same call is visible.

        Keyed on the normalised root-cause line rather than a prefix of the
        compacted text, so it survives differences in timestamps, retry counters
        and memory addresses between otherwise identical failures.
        """
        signature = self._error_signature(compacted_content)
        if not signature:
            return

        self._error_signature_counts[signature] = self._error_signature_counts.get(signature, 0) + 1
        occurrences = self._error_signature_counts[signature]
        if occurrences >= DOOM_LOOP_THRESHOLD:
            self._log_event(
                PagingEventType.LEAK_WARNING,
                page_id,
                0,
                f"Repeated failure detected {occurrences}x: {signature[:90]}",
            )

    @staticmethod
    def _error_signature(text: str) -> str:
        """Normalise a compacted trace to a comparable signature."""
        root = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("root cause:"):
                root = stripped.split(":", 1)[1].strip()
                break
        if not root:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            root = lines[-1] if lines else ""

        # Strip volatile detail: addresses, timestamps, ports, retry counters.
        root = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", root)
        root = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ][\d:.]+\b", "TIMESTAMP", root)
        root = re.sub(r"\b\d+\b", "N", root)
        return root.strip()

    # -- accounting -------------------------------------------------------------

    def _get_current_working_tokens(self) -> int:
        """Tokens currently occupying the context window."""
        total = 0
        for page in self.pages.values():
            if page.status == PageStatus.ACTIVE:
                total += page.token_count
            elif page.status == PageStatus.SWAPPED and page.tombstone:
                total += estimate_tokens(page.tombstone)
        return total

    def _enforce_budget(self, protect: Optional[str] = None) -> None:
        """
        Evict cold pages until working tokens fit the budget.

        Eviction order is L2 episodic before L1 working, oldest access first. L0
        is never evicted, so a large enough pinned set can exceed the budget; when
        that happens the kernel says so rather than failing silently.

        `protect` exempts one page for the duration of the call. A page fault uses
        it so that a page which is large relative to the budget cannot be evicted
        by the same call that restored it: the caller asked for that content, and
        handing back an emptied page while reporting success is worse than going
        over budget and saying so.
        """
        if self._get_current_working_tokens() <= self.token_budget:
            return

        self._coalesce_episodic_history()

        tier_weight = {PageTier.L2_EPISODIC: 0, PageTier.L1_WORKING_RAM: 1}
        candidates = [
            p
            for p in self.pages.values()
            if p.status == PageStatus.ACTIVE
            and p.tier != PageTier.L0_PINNED
            and p.id != protect
            and self._swap_would_help(p)
        ]
        candidates.sort(key=lambda p: (tier_weight.get(p.tier, 2), p.last_accessed_at))

        for page in candidates:
            if self._get_current_working_tokens() <= self.token_budget:
                return
            self.page_out(page.id)

        overshoot = self._get_current_working_tokens()
        if overshoot > self.token_budget:
            pinned = sum(
                p.token_count
                for p in self.pages.values()
                if p.tier == PageTier.L0_PINNED and p.status == PageStatus.ACTIVE
            )
            over_by = f"Over budget by {overshoot - self.token_budget:,} tokens"
            if protect is not None and protect in self.pages:
                # Blaming the pinned set here would be wrong, and it may well be
                # empty: the page just faulted in is what does not fit.
                held = self.pages[protect]
                reason = (
                    f"{over_by}: '{held.title}' ({held.token_count:,} tokens) was just paged in "
                    f"and is held in context for this request rather than evicted again."
                )
            else:
                reason = (
                    f"{over_by} after evicting every unpinned page. "
                    f"{pinned:,} tokens are pinned to L0 and cannot be evicted."
                )
            self._log_event(PagingEventType.BUDGET_EXCEEDED, protect or "system:budget", 0, reason)

    def _next_digest_index(self) -> int:
        """
        The next free digest number, derived from the pages themselves.

        Deliberately not an instance counter. A restored session builds a fresh
        kernel around pages that already contain `episodic:digest:0`; a counter
        starting from zero would hand that id out a second time and overwrite the
        restored history in place — and overwrite its swap row with it, losing the
        conversation for good. Reading the number off the live page table cannot
        drift from what actually exists.
        """
        used = []
        for page_id in self.pages:
            if page_id.startswith("episodic:digest:"):
                suffix = page_id.rsplit(":", 1)[-1]
                if suffix.isdigit():
                    used.append(int(suffix))
        return max(used) + 1 if used else 0

    def _coalesce_episodic_history(self) -> None:
        """
        Fold old conversation turns into one page so they can be evicted together.

        Individually, a turn is smaller than the tombstone that would replace it,
        so `_swap_would_help` rejects it and it is never an eviction candidate. A
        long conversation is nothing but such turns, and the effect was that the
        kernel evicted *source files* to make room for chat — starving the working
        set to hold history, which is precisely backwards for this tool.

        Coalescing pays one tombstone for the whole run instead of one per turn.
        The digest is an ordinary page: over budget it swaps to disk like anything
        else, and mentioning it pages the full history back in.
        """
        small_turns = [
            p
            for p in self.pages.values()
            if p.tier == PageTier.L2_EPISODIC
            and p.status == PageStatus.ACTIVE
            and not self._swap_would_help(p)
        ]
        if len(small_turns) <= EPISODIC_TURNS_KEPT_WHOLE:
            return

        small_turns.sort(key=lambda p: p.last_accessed_at)
        stale = small_turns[:-EPISODIC_TURNS_KEPT_WHOLE]
        if sum(p.token_count for p in stale) <= self.token_budget * MAX_EPISODIC_BUDGET_SHARE:
            return

        body = "\n\n".join(f"[{p.title}]\n{p.content}" for p in stale)
        for page in stale:
            del self.pages[page.id]

        digest_id = f"episodic:digest:{self._next_digest_index()}"
        digest = ContextPage(
            id=digest_id,
            title=f"Conversation history ({len(stale)} turns)",
            tier=PageTier.L2_EPISODIC,
            status=PageStatus.ACTIVE,
            content=body,
            token_count=estimate_tokens(body),
            metadata={"coalesced_turns": len(stale)},
        )
        self.pages[digest_id] = digest
        self._log_event(
            PagingEventType.PAGE_ALLOCATED,
            digest_id,
            0,
            f"Coalesced {len(stale)} conversation turns into one evictable page "
            f"({digest.token_count:,} tokens)",
        )

    # -- paging -----------------------------------------------------------------

    def _swap_would_help(self, page: ContextPage) -> bool:
        """
        Whether evicting this page would actually shrink the context window.

        A page smaller than its own tombstone costs more swapped than resident.
        Automatic eviction skips those; an explicit `page_out` still honours the
        caller, who may want the content out of context for reasons other than size.
        """
        tombstone_cost = estimate_tokens(self._tombstone_for(page))
        return (page.token_count - tombstone_cost) >= MIN_AUTO_SWAP_BENEFIT_TOKENS

    @staticmethod
    def _tombstone_for(page: ContextPage) -> str:
        """The marker left in context while a page is on disk."""
        return (
            f"[ContextOS] '{page.title}' ({page.token_count:,} tokens) is swapped to disk. "
            f"Mention {page.id} to page it back in."
        )

    def page_out(self, page_id: str) -> bool:
        """
        Move a page to disk, leaving a tombstone in context.

        The page's content is released from memory once the write to swap has been
        confirmed, so swapping reclaims process memory as well as context budget.
        """
        page = self.pages.get(page_id)
        if page is None:
            return False
        if page.tier == PageTier.L0_PINNED or page.status == PageStatus.SWAPPED:
            return False

        # Persist before dropping the in-memory copy; swap is the only copy after this.
        self.swap.store(page)

        tombstone = self._tombstone_for(page)
        saved_tokens = max(0, page.token_count - estimate_tokens(tombstone))
        self.total_tokens_saved += saved_tokens

        page.status = PageStatus.SWAPPED
        page.tombstone = tombstone
        page.content = ""

        self._log_event(
            PagingEventType.PAGE_SWAPPED_OUT,
            page_id,
            saved_tokens,
            f"Swapped out '{page.title}': freed {saved_tokens:,} tokens of context",
        )
        return True

    def page_fault(self, page_id: str) -> Optional[ContextPage]:
        """
        Bring a swapped page back into the context window.

        Returns None when the page is neither resident nor in swap. The swap row is
        removed once the content is restored, so the database mirrors exactly the
        set of pages that are currently swapped out.
        """
        started_at = time.perf_counter()
        page = self.pages.get(page_id)

        if page is None:
            restored = self.swap.retrieve(page_id)
            if restored is None:
                return None
            self.pages[page_id] = restored
            self.swap.delete(page_id)
            restored.mark_access()
            self.total_page_faults += 1
            self._record_fault_duration(started_at)
            self._log_event(
                PagingEventType.PAGE_FAULT_HIT,
                page_id,
                0,
                f"Page fault: restored '{restored.title}' ({restored.token_count:,} tokens) from disk",
            )
            self._enforce_budget(protect=page_id)
            return restored

        if page.status != PageStatus.SWAPPED:
            page.mark_access()
            return page

        stored = self.swap.retrieve(page_id)
        if stored is None:
            self._log_event(
                PagingEventType.LEAK_WARNING,
                page_id,
                0,
                f"Page '{page.title}' is marked swapped but is missing from disk",
            )
            return None

        page.content = stored.content
        page.token_count = stored.token_count
        page.original_token_count = stored.original_token_count or stored.token_count
        page.compacted = stored.compacted
        page.status = PageStatus.ACTIVE
        page.tombstone = None
        page.mark_access()
        self.swap.delete(page_id)

        self.total_page_faults += 1
        self._record_fault_duration(started_at)
        self._log_event(
            PagingEventType.PAGE_FAULT_HIT,
            page_id,
            0,
            f"Page fault: restored '{page.title}' ({page.token_count:,} tokens) into context",
        )

        self._enforce_budget(protect=page_id)
        return page

    def touch_or_fault(self, text_query: str) -> List[str]:
        """
        Page in anything the incoming text refers to.

        Matches a swapped page's id, its title, or the file's base name, on word
        boundaries. Short names are ignored, because a two-character title matches
        almost any sentence and would page the whole workspace back in.
        """
        if not text_query:
            return []

        normalized = text_query.lower().replace("\\", "/")
        rehydrated: List[str] = []

        for page_id, page in list(self.pages.items()):
            if page.status != PageStatus.SWAPPED:
                continue

            candidates = {
                page_id.lower().replace("\\", "/"),
                page.title.lower().replace("\\", "/"),
            }
            title_base = page.title.replace("\\", "/").rsplit("/", 1)[-1].lower()
            if len(title_base) >= 4:
                candidates.add(title_base)

            for candidate in candidates:
                if len(candidate) < 4:
                    continue
                # A path separator before the name is part of the reference, not a
                # reason to reject it: "src/auth.py" names auth.py. Word characters
                # and dots before it are, so "oauth.py" does not match "auth.py".
                if re.search(rf"(?<![\w.]){re.escape(candidate)}(?![\w])", normalized):
                    if self.page_fault(page_id):
                        rehydrated.append(page_id)
                    break

        return rehydrated

    # -- context assembly -------------------------------------------------------

    def assemble_context(self) -> str:
        """Render the context window: resident pages by tier, then swap tombstones."""
        return self.assemble_context_report()["context"]

    def assemble_context_report(self) -> Dict[str, Any]:
        """
        Render the context window and account for every token in it.

        Returns the assembled text, its measured size, and a per-page breakdown of
        what each page contributes — full content for a resident page, a tombstone
        for a swapped one.

        The breakdown is reconciled: `segments` plus `overhead_tokens` always sums
        to `total_tokens`. That matters because a per-page total computed from
        `page.token_count` would *not* match the window — tombstones cost a fraction
        of the page they stand in for, section headers are real tokens no page owns,
        and per-segment estimates round independently. A breakdown that did not add
        up would be another figure that looks measured and is not.
        """
        sections: List[str] = []
        segments: List[Dict[str, Any]] = []

        for tier in (PageTier.L0_PINNED, PageTier.L1_WORKING_RAM, PageTier.L2_EPISODIC):
            tier_pages = [
                p for p in self.pages.values() if p.tier == tier and p.status == PageStatus.ACTIVE
            ]
            if not tier_pages:
                continue
            sections.append(f"=== [ContextOS {tier.value}] ===")
            for page in tier_pages:
                body = f"--- [{page.title} (id: {page.id})] ---\n{page.content}\n"
                sections.append(body)
                segments.append(
                    {
                        "page_id": page.id,
                        "title": page.title,
                        "tier": page.tier.value,
                        "included_as": "full content",
                        "tokens": estimate_tokens(body),
                    }
                )

        swapped = [p for p in self.pages.values() if p.status == PageStatus.SWAPPED]
        if swapped:
            sections.append("=== [ContextOS swapped to disk] ===")
            for page in swapped:
                line = f"- {page.tombstone}"
                sections.append(line)
                segments.append(
                    {
                        "page_id": page.id,
                        "title": page.title,
                        "tier": page.tier.value,
                        "included_as": "tombstone",
                        "tokens": estimate_tokens(line),
                        "full_tokens": page.token_count,
                    }
                )
            sections.append("")

        context = "\n".join(sections)
        total = estimate_tokens(context)
        accounted = sum(int(segment["tokens"]) for segment in segments)

        return {
            "context": context,
            "total_tokens": total,
            "segments": segments,
            # Section headers, the newlines joining sections, and per-segment
            # rounding. Reported rather than hidden so the rows reconcile exactly.
            "overhead_tokens": total - accounted,
        }

    # -- telemetry --------------------------------------------------------------

    def get_metrics(self) -> MemoryMetrics:
        """
        Current kernel state.

        `swapped_tokens` and `l3_pages` describe pages this kernel has swapped out
        right now. `swap_disk_tokens` and `swap_disk_rows` describe the database on
        disk, which may hold rows written by earlier runs. Mixing the two made the
        dashboard report a swap figure that only ever grew.
        """
        working = self._get_current_working_tokens()
        swapped_pages = [p for p in self.pages.values() if p.status == PageStatus.SWAPPED]
        utilization = (working / self.token_budget * 100.0) if self.token_budget > 0 else 0.0

        disk = self.swap.get_disk_stats()

        return MemoryMetrics(
            token_budget=self.token_budget,
            working_tokens=working,
            swapped_tokens=sum(p.token_count for p in swapped_pages),
            swap_disk_tokens=disk["tokens"],
            swap_disk_rows=disk["rows"],
            total_tokens_saved=self.total_tokens_saved,
            budget_utilization_pct=round(utilization, 1),
            over_budget=working > self.token_budget,
            l0_pages=sum(
                1
                for p in self.pages.values()
                if p.tier == PageTier.L0_PINNED and p.status == PageStatus.ACTIVE
            ),
            l1_pages=sum(
                1
                for p in self.pages.values()
                if p.tier == PageTier.L1_WORKING_RAM and p.status == PageStatus.ACTIVE
            ),
            l2_pages=sum(
                1
                for p in self.pages.values()
                if p.tier == PageTier.L2_EPISODIC and p.status == PageStatus.ACTIVE
            ),
            l3_pages=len(swapped_pages),
            total_page_faults=self.total_page_faults,
            avg_page_fault_ms=self.average_page_fault_ms(),
            recent_events=list(self.events[-12:]),
        )

    def average_page_fault_ms(self) -> float:
        """Mean duration of recent page faults in milliseconds. 0.0 when none measured."""
        if not self._page_fault_durations_ms:
            return 0.0
        return round(sum(self._page_fault_durations_ms) / len(self._page_fault_durations_ms), 3)

    def _record_fault_duration(self, started_at: float) -> None:
        self._page_fault_durations_ms.append((time.perf_counter() - started_at) * 1000.0)
        if len(self._page_fault_durations_ms) > 100:
            del self._page_fault_durations_ms[:-100]

    # -- page management --------------------------------------------------------

    def pin_page(self, page_id: str) -> bool:
        """Promote a page to L0, rehydrating it first if it is on disk."""
        page = self.pages.get(page_id)
        if page is None:
            return False
        if page.status == PageStatus.SWAPPED:
            self.page_fault(page_id)
        page.tier = PageTier.L0_PINNED
        self._log_event(
            PagingEventType.TIER_PROMOTED, page_id, 0, f"Pinned '{page.title}' to L0"
        )
        return True

    def unpin_page(self, page_id: str) -> bool:
        """Demote a page to L1, where it becomes eligible for eviction again."""
        page = self.pages.get(page_id)
        if page is None:
            return False
        page.tier = PageTier.L1_WORKING_RAM
        self._log_event(
            PagingEventType.TIER_DEMOTED, page_id, 0, f"Unpinned '{page.title}' to L1"
        )
        self._enforce_budget()
        return True

    def delete_page(self, page_id: str) -> bool:
        """Remove a page from memory and from swap."""
        page = self.pages.pop(page_id, None)
        if page is None:
            return False
        self.swap.delete(page_id)
        self._log_event(
            PagingEventType.PAGE_EVICTED, page_id, 0, f"Purged '{page.title}'"
        )
        return True

    def set_budget(self, new_budget: int) -> MemoryMetrics:
        """Change the token budget and re-run eviction against it."""
        self.token_budget = max(100, int(new_budget))
        self._log_event(
            PagingEventType.BUDGET_CHANGED,
            "system:budget",
            0,
            f"Token budget set to {self.token_budget:,}",
        )
        self._enforce_budget()
        return self.get_metrics()

    def rehydrate_all(self) -> int:
        """Page every swapped page back in. Returns how many were restored."""
        restored = 0
        for page_id in [pid for pid, p in self.pages.items() if p.status == PageStatus.SWAPPED]:
            if self.page_fault(page_id):
                restored += 1
        return restored

    def swap_all_unpinned(self) -> int:
        """Swap out every unpinned resident page. Returns how many were written."""
        swapped = 0
        for page_id in [
            pid
            for pid, p in self.pages.items()
            if p.status == PageStatus.ACTIVE and p.tier != PageTier.L0_PINNED
        ]:
            if self.page_out(page_id):
                swapped += 1
        return swapped

    def search(self, query: str, top_k: int = 5) -> List[Tuple[ContextPage, float]]:
        """
        Rank pages by keyword overlap with `query`.

        This is lexical scoring over titles, ids and resident content — not a
        semantic embedding search. Named accordingly so it is not mistaken for one.
        Swapped pages are matched on title and id only, since their content is on
        disk.
        """
        keywords = [w.lower() for w in re.split(r"[\s/.:_-]+", query) if len(w) >= 2]
        if not keywords:
            return []

        scored: List[Tuple[ContextPage, float]] = []
        for page in self.pages.values():
            score = 0.0
            title_low = page.title.lower()
            id_low = page.id.lower()
            content_sample = page.content[:4000].lower() if page.content else ""

            for keyword in keywords:
                if keyword in title_low:
                    score += 6.0
                if keyword in id_low:
                    score += 4.0
                if content_sample and keyword in content_sample:
                    score += min(4.0, content_sample.count(keyword) * 0.5)

            if score > 0:
                scored.append((page, round(score, 1)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    # Retained under the previous name so existing callers keep working.
    semantic_search = search

    def clear_swap(self) -> int:
        """
        Drop every row from the swap database, including rows from earlier runs.

        Pages this kernel currently has swapped out lose their only copy, so they
        are purged from the page table too rather than left pointing at nothing.
        """
        lost = [pid for pid, p in self.pages.items() if p.status == PageStatus.SWAPPED]
        for page_id in lost:
            self.pages.pop(page_id, None)
        rows = self.swap.clear()
        self._log_event(
            PagingEventType.PAGE_EVICTED,
            "system:swap",
            0,
            f"Cleared swap: {rows} rows removed, {len(lost)} swapped pages purged",
        )
        return rows

    def _log_event(
        self, event_type: PagingEventType, page_id: str, tokens_saved: int, description: str
    ) -> None:
        self.events.append(
            PagingEvent(
                event_type=event_type,
                page_id=page_id,
                tokens_saved=tokens_saved,
                description=description,
            )
        )
        if len(self.events) > MAX_RETAINED_EVENTS:
            del self.events[: len(self.events) - MAX_RETAINED_EVENTS]
