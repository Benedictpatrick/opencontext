"""
`contextos top` — a read-only terminal monitor for the kernel, in the spirit of htop.

Renders a single self-sizing frame that adapts to the terminal width, and refreshes
in place until interrupted. Every figure shown is read from kernel telemetry; none
is hardcoded.

This is deliberately separate from the interactive Textual application
(`contextos tui`): `top` is for watching, `tui` is for driving.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from contextos import __version__
from contextos.core.kernel import ContextKernel
from contextos.core.types import PageStatus, PageTier, PagingEventType

TIER_LABELS = {
    PageTier.L0_PINNED: ("L0", "bold #a371f7"),
    PageTier.L1_WORKING_RAM: ("L1", "#3fb950"),
    PageTier.L2_EPISODIC: ("L2", "#38bdf8"),
    PageTier.L3_SWAP: ("L3", "dim magenta"),
}

EVENT_LABELS = {
    PagingEventType.PAGE_ALLOCATED: ("alloc", "#3fb950"),
    PagingEventType.PAGE_SWAPPED_OUT: ("swap-out", "#d29922"),
    PagingEventType.PAGE_FAULT_HIT: ("fault", "#f59e0b"),
    PagingEventType.PAGE_COMPACTED: ("compact", "#38bdf8"),
    PagingEventType.PAGE_EVICTED: ("evict", "#f85149"),
    PagingEventType.LEAK_WARNING: ("WARNING", "bold #f85149"),
    PagingEventType.BUDGET_EXCEEDED: ("OVER-BUDGET", "bold #f85149"),
    PagingEventType.BUDGET_CHANGED: ("budget", "#8b949e"),
    PagingEventType.TIER_PROMOTED: ("pin", "#a371f7"),
    PagingEventType.TIER_DEMOTED: ("unpin", "#8b949e"),
    PagingEventType.DOOM_LOOP_PREVENTED: ("loop-guard", "#3fb950"),
}


class ContextTopUI:
    """Live terminal monitor for a `ContextKernel`."""

    def __init__(self, kernel: ContextKernel, console: Optional[Console] = None):
        self.kernel = kernel
        self.console = console or Console()

    @property
    def width(self) -> int:
        return max(40, self.console.size.width)

    # -- panels -----------------------------------------------------------------

    def render_header(self) -> Panel:
        metrics = self.kernel.get_metrics()
        header = Text()
        header.append(f" ContextOS {__version__} ", style="bold black on #f59e0b")
        if self.width >= 70:
            header.append("  context memory monitor  ", style="#8b949e")
        header.append(f"{len(self.kernel.pages)} pages  ", style="#e6edf3")
        header.append(time.strftime("%H:%M:%S"), style="dim #8b949e")
        if metrics.over_budget:
            header.append("   OVER BUDGET ", style="bold white on #b91c1c")
        return Panel(header, border_style="#30363d", padding=(0, 1))

    def render_metrics_panel(self) -> Panel:
        """Budget gauge plus the counters, laid out to fit the terminal width."""
        metrics = self.kernel.get_metrics()

        if metrics.budget_utilization_pct > 85:
            gauge_style = "bold #f85149"
        elif metrics.budget_utilization_pct > 65:
            gauge_style = "bold #f59e0b"
        else:
            gauge_style = "bold #3fb950"

        # The gauge is sized from the live terminal width instead of a fixed 40
        # cells, so it neither wraps on a narrow terminal nor strands space on a wide one.
        gauge_width = max(10, min(48, self.width - 46))
        pct = min(100.0, max(0.0, metrics.budget_utilization_pct))
        filled = int(gauge_width * pct / 100)
        gauge = "█" * filled + "░" * (gauge_width - filled)

        budget_line = Text()
        budget_line.append("context  ", style="bold #e6edf3")
        budget_line.append(gauge, style=gauge_style)
        budget_line.append(f" {metrics.budget_utilization_pct:5.1f}%", style=gauge_style)
        budget_line.append(
            f"  {metrics.working_tokens:,} / {metrics.token_budget:,} tokens", style="dim #8b949e"
        )

        counters = Table.grid(expand=True, padding=(0, 2))
        for _ in range(2 if self.width < 100 else 4):
            counters.add_column()

        cells = [
            self._counter("swapped out", f"{metrics.l3_pages} pages", f"{metrics.swapped_tokens:,} tok", "#d29922"),
            self._counter("removed from context", f"+{metrics.total_tokens_saved:,} tok", "compaction + swap", "#3fb950"),
            self._counter(
                "page faults",
                f"{metrics.total_page_faults}",
                f"{metrics.avg_page_fault_ms:.2f} ms avg" if metrics.total_page_faults else "none yet",
                "#38bdf8",
            ),
            self._counter(
                "swap database",
                f"{metrics.swap_disk_rows} rows",
                f"{metrics.swap_disk_tokens:,} tok on disk",
                "#8b949e",
            ),
        ]

        if self.width < 100:
            counters.add_row(cells[0], cells[1])
            counters.add_row(cells[2], cells[3])
        else:
            counters.add_row(*cells)

        return Panel(
            Group(budget_line, Text(""), counters),
            title="[bold #e6edf3]memory[/bold #e6edf3]",
            border_style="#30363d",
            padding=(0, 1),
        )

    @staticmethod
    def _counter(label: str, value: str, detail: str, color: str) -> Text:
        cell = Text()
        cell.append(f"{label}\n", style="dim #8b949e")
        cell.append(value, style=f"bold {color}")
        cell.append(f"\n{detail}", style="dim #8b949e")
        return cell

    def render_pages_table(self, limit: int = 12) -> Panel:
        """The page table, with columns dropped as the terminal narrows."""
        wide = self.width >= 110
        medium = self.width >= 80

        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("tier", justify="center", width=4)
        table.add_column("state", justify="center", width=8)
        table.add_column("page", style="#e6edf3", ratio=3, no_wrap=True)
        table.add_column("tokens", justify="right", style="bold #f0f6fc", width=9)
        if medium:
            table.add_column("refs", justify="right", style="dim", width=5)
        if wide:
            table.add_column("in context", style="dim #8b949e", ratio=4, no_wrap=True)

        pages = sorted(
            self.kernel.pages.values(),
            key=lambda p: (
                0 if p.tier == PageTier.L0_PINNED else 1,
                0 if p.status == PageStatus.ACTIVE else 1,
                -p.last_accessed_at,
            ),
        )

        if not pages:
            return Panel(
                Text("No pages allocated. Run `contextos scan` to index a project.", style="dim"),
                title="[bold #e6edf3]pages[/bold #e6edf3]",
                border_style="#30363d",
            )

        for page in pages[:limit]:
            tier_label, tier_style = TIER_LABELS.get(page.tier, ("??", "white"))
            resident = page.status == PageStatus.ACTIVE
            state = Text("resident" if resident else "swapped", style="#3fb950" if resident else "#d29922")

            row: List[object] = [
                Text(tier_label, style=tier_style),
                state,
                page.title,
                f"{page.token_count:,}",
            ]
            if medium:
                row.append(str(page.access_count))
            if wide:
                preview = page.get_context_representation().replace("\n", " ")
                row.append(preview[:80] + ("..." if len(preview) > 80 else ""))
            table.add_row(*row)

        hidden = len(pages) - min(limit, len(pages))
        title = f"[bold #e6edf3]pages[/bold #e6edf3] [dim]({len(pages)} total"
        title += f", {hidden} not shown)" if hidden > 0 else ")"
        return Panel(table, title=title, border_style="#30363d")

    def render_events_panel(self, limit: int = 5) -> Panel:
        events = self.kernel.get_metrics().recent_events[-limit:]

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="dim #8b949e", width=9)
        table.add_column(width=12)
        table.add_column(style="#e6edf3", ratio=1, overflow="ellipsis")

        if not events:
            table.add_row(time.strftime("%H:%M:%S"), Text("idle", style="dim"), "No kernel activity yet")
        else:
            for event in events:
                label, style = EVENT_LABELS.get(event.event_type, (event.event_type.value.lower(), "white"))
                table.add_row(event.formatted_time, Text(label, style=style), event.description)

        return Panel(table, title="[bold #e6edf3]kernel activity[/bold #e6edf3]", border_style="#30363d")

    def render_footer(self) -> Text:
        footer = Text()
        footer.append(" Ctrl+C ", style="bold black on #8b949e")
        footer.append(" exit    ", style="dim #8b949e")
        footer.append("read-only monitor — run ", style="dim #8b949e")
        footer.append("contextos tui", style="#f59e0b")
        footer.append(" to page memory in and out", style="dim #8b949e")
        return footer

    # -- composition ------------------------------------------------------------

    def draw(self) -> Group:
        """One frame, sized to the terminal. Panels are dropped if height is tight."""
        height = self.console.size.height
        panels: List[object] = [self.render_header(), self.render_metrics_panel()]

        # Header + metrics + footer cost roughly 14 rows; give the rest to the table.
        remaining = height - 14
        if remaining >= 5:
            panels.append(self.render_pages_table(limit=max(3, min(15, remaining - 2))))
        if remaining >= 12:
            panels.append(self.render_events_panel())
        panels.append(self.render_footer())
        return Group(*panels)

    def print_snapshot(self) -> None:
        """Print one frame and return. Used for non-interactive output and tests."""
        self.console.print(self.draw())

    def run_live(self, refresh_rate: float = 1.0, iterations: Optional[int] = None) -> None:
        """
        Refresh in place until interrupted.

        Ctrl+C exits cleanly rather than surfacing a traceback; `iterations` bounds
        the loop for tests.
        """
        refresh_per_second = max(1, min(10, int(1 / max(0.1, refresh_rate))))
        count = 0
        try:
            with Live(
                self.draw(),
                console=self.console,
                refresh_per_second=refresh_per_second,
                screen=True,
            ) as live:
                while iterations is None or count < iterations:
                    live.update(self.draw())
                    time.sleep(refresh_rate)
                    count += 1
        except KeyboardInterrupt:
            self.console.print("[dim]ContextOS monitor stopped.[/dim]")


def run_top(kernel: ContextKernel, refresh_rate: float = 1.0, once: bool = False) -> None:
    """Entry point for `contextos top`."""
    ui = ContextTopUI(kernel)
    if once:
        ui.print_snapshot()
    else:
        ui.run_live(refresh_rate=refresh_rate)
