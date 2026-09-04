"""
Interactive terminal UI (Textual).

Six tabs:

  1. **Overview** — budget, swap and fault telemetry, and the kernel activity log.
  2. **Pages** — the page table, with a live preview and paging controls.
  3. **Context** — the assembled context window the model would actually receive,
     with every token in it accounted for, and an export to file.
  4. **Chat** — questions against the indexed codebase, streamed as they arrive.
  5. **Compact** — a workbench for reducing stack traces.
  6. **Config** — budget, model server, and the benchmark.

Four rules this module holds to:

  * **Nothing is displayed that was not measured.** An earlier version printed a
    fixed "98.4% cache hit ratio", a fixed "< 500us" fault latency and a
    hand-written retry-loop panel complete with an invented hash. If a figure
    cannot be measured it is not shown.
  * **The Context tab shows the window verbatim**, rendered as literal text rather
    than markup — it contains section headers in square brackets and arbitrary
    source code, either of which Rich would otherwise swallow. Its per-page
    breakdown reconciles exactly to the window's measured size.
  * **The arrangement survives the session.** Pins, tiers and the budget are saved
    on exit and restored on start, because a pin that evaporates when you quit is
    not worth making. Anything whose source has gone is dropped and reported, never
    restored empty.
  * **The layout must survive an 80x24 terminal.** Card titles are short enough not
    to truncate, the four HUD cards share one fixed height so they cannot close
    ragged, and panes collapse rather than squeeze as the terminal narrows.
    `tests/test_tui_layout.py` renders the app at four sizes and fails if any of
    that regresses.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from opencontext import __version__
from opencontext.core.compactor import CodeOutlineCompactor, TracebackCompactor
from opencontext.core.kernel import ContextKernel
from opencontext.core.pager import ContextPager
from opencontext.core.scenarios import run_doom_loop
from opencontext.core.session import (
    DEFAULT_SESSION_PATH,
    SessionError,
    restore_session,
    save_session,
    session_exists,
)
from opencontext.core.types import PageStatus, PageTier
from opencontext.core.workspace import WorkspaceScanner

# Below this width the HUD sheds detail lines and the table sheds columns.
NARROW_WIDTH = 100

LEXERS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".json": "json", ".md": "markdown", ".html": "html",
    ".css": "css", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".sql": "sql", ".sh": "bash", ".rs": "rust", ".go": "go", ".java": "java",
    ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".h": "c",
}


def detect_lexer(filename: str) -> str:
    """Pygments lexer name for a filename."""
    return LEXERS.get(os.path.splitext(filename)[1].lower(), "text")


class CodeViewModal(ModalScreen):
    """Full-screen viewer for one page."""

    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("p", "page_in", "Page in"),
        Binding("s", "swap_out", "Swap out"),
        Binding("P", "toggle_pin", "Pin/unpin"),
    ]

    def __init__(self, kernel: ContextKernel, page_id: str):
        super().__init__()
        self.kernel = kernel
        self.page_id = page_id

    @property
    def page(self):
        return self.kernel.pages.get(self.page_id)

    def compose(self) -> ComposeResult:
        page = self.page
        title = page.title if page else self.page_id

        with Vertical(id="modal-container"):
            with Horizontal(id="modal-header"):
                yield Label(title, id="modal-filename")
                if page:
                    yield Label(
                        f"{page.status.value.lower()}  {page.tier.value}  "
                        f"{page.token_count:,} tok  {page.access_count} refs",
                        id="modal-meta",
                    )

            with VerticalScroll(id="modal-code-scroll"):
                if page is None:
                    yield Label("Page not found.", id="modal-body-text")
                elif page.status == PageStatus.SWAPPED:
                    yield Label(
                        "This page is swapped to disk. Its content is not in the "
                        "context window.\n\n"
                        f"{page.tombstone}\n\n"
                        "Press [p] to page it back in.",
                        id="modal-body-text",
                    )
                elif not page.content:
                    yield Label("Page is empty.", id="modal-body-text")
                else:
                    yield Static(
                        Syntax(
                            page.content,
                            detect_lexer(page.title),
                            theme="monokai",
                            line_numbers=True,
                            word_wrap=True,
                        ),
                        id="modal-syntax-content",
                    )

            with Horizontal(id="modal-actions"):
                yield Label("[p] page in   [s] swap out   [P] pin   [Esc] close", id="modal-hint")
                yield Button("Close", id="btn-modal-close")

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_page_in(self) -> None:
        page = self.kernel.page_fault(self.page_id)
        self.notify(
            f"Paged in {page.title} ({page.token_count:,} tok)" if page else "Page could not be restored",
            severity="information" if page else "warning",
        )
        self.app.pop_screen()

    def action_swap_out(self) -> None:
        page = self.page
        if page and page.tier == PageTier.L0_PINNED:
            self.notify("Pinned pages are protected from eviction", severity="warning")
            return
        if self.kernel.page_out(self.page_id):
            self.notify(f"Swapped {self.page_id} to disk")
        self.app.pop_screen()

    def action_toggle_pin(self) -> None:
        page = self.page
        if page is None:
            return
        if page.tier == PageTier.L0_PINNED:
            self.kernel.unpin_page(self.page_id)
            self.notify(f"Unpinned {page.title}")
        else:
            self.kernel.pin_page(self.page_id)
            self.notify(f"Pinned {page.title} — protected from eviction")
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-modal-close":
            self.app.pop_screen()


class HelpScreen(ModalScreen):
    """Keyboard reference."""

    BINDINGS = [Binding("escape,question_mark,q", "close", "Close", priority=True)]

    HELP = [
        ("Tabs", [
            ("1 - 6", "switch tab"),
        ]),
        ("Pages", [
            ("p", "page in the selected page"),
            ("s", "swap the selected page to disk"),
            ("P", "pin / unpin (pinned pages are never evicted)"),
            ("o", "toggle outline view in the preview"),
            ("enter", "open the full-screen viewer"),
            ("/", "focus the page filter"),
        ]),
        ("Workspace", [
            ("r", "re-index the project from disk"),
            ("w", "write the assembled context to a file"),
        ]),
        ("Session", [
            ("ctrl+s", "save the session now"),
            ("", "pins, tiers and budget are saved on exit and restored on start"),
            ("", "start with --fresh to ignore a saved session"),
        ]),
        ("Chat", [
            ("/help", "list the slash commands"),
            ("", "answers need a model server; see the Config tab"),
        ]),
        ("General", [
            ("?", "this help"),
            ("q", "quit"),
        ]),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Label("OpenContext keys", id="help-title")
            with VerticalScroll(id="help-body"):
                for section, entries in self.HELP:
                    yield Label(section.upper(), classes="panel-title")
                    for key, description in entries:
                        yield Label(f"  {key:<10} {description}")
            yield Label("Esc to close", id="help-hint")

    def action_close(self) -> None:
        self.app.pop_screen()


class OpenContextApp(App):
    """The interactive OpenContext terminal application."""

    TITLE = "OpenContext"
    SUB_TITLE = "context memory kernel"

    CSS = """
    Screen { layout: vertical; background: #0c0e14; color: #e6edf3; }
    Header { background: #131720; color: #f59e0b; text-style: bold; }
    Footer { background: #131720; color: #8b949e; }

    TabbedContent, ContentSwitcher { height: 1fr; }
    TabPane { height: 1fr; padding: 0 1; }
    .tab-body { height: 1fr; }

    Tabs { background: #0c0e14; height: 2; }
    Tab { color: #8b949e; padding: 0 1; }
    Tab.-active, ContentTab.-active {
        color: #ffffff;
        background: #b45309;
        text-style: bold;
    }

    /* HUD. A fixed bar height with full-height cards keeps all four the same
       size; without it the taller RAM card left the others closing early and
       the row rendered ragged. */
    .hud-bar { height: 7; }
    .panel-card {
        height: 100%;
        width: 1fr;
        background: #131720;
        border: round #262c3a;
        padding: 0 1;
    }
    .card-ram { border: round #3fb950; }
    .card-swap { border: round #d29922; }
    .card-saved { border: round #38bdf8; }
    .card-faults { border: round #a371f7; }

    .panel-title { color: #8b949e; text-style: bold; }
    .metric-big { text-style: bold; color: #f0f6fc; }
    .val-green { color: #3fb950; text-style: bold; }
    .val-amber { color: #d29922; text-style: bold; }
    .val-cyan { color: #38bdf8; text-style: bold; }
    .val-purple { color: #a371f7; text-style: bold; }
    .val-red { color: #f85149; text-style: bold; }

    /* Detail lines and the sparkline are hidden on a narrow terminal, where
       there is not enough width to render them without truncating. */
    OpenContextApp.-narrow .hud-detail { display: none; }
    OpenContextApp.-narrow #hud-sparkline { display: none; }
    OpenContextApp.-narrow .wide-only { display: none; }

    ProgressBar { height: 1; }
    Bar > .bar--bar { color: #3fb950; background: #262c3a; }
    Sparkline { height: 1; }
    Sparkline > .sparkline--max-color { color: #3fb950; }
    Sparkline > .sparkline--min-color { color: #1e3a2b; }

    .cmd-bar { height: 3; align: left middle; }
    Button { min-width: 10; height: 3; margin-right: 1;
             border: solid #262c3a; background: #181d28; color: #e6edf3; }
    Button:hover { background: #262c3a; color: #f59e0b; border: solid #f59e0b; }
    Button.btn-primary { background: #b45309; color: #ffffff; border: solid #d97706; }

    DataTable { height: 1fr; border: round #262c3a; background: #0c0e14; }
    DataTable > .datatable--cursor { background: #b45309; color: #ffffff; text-style: bold; }
    DataTable > .datatable--header { background: #131720; color: #8b949e; text-style: bold; }

    Input { background: #131720; border: round #262c3a; color: #f0f6fc; }
    Input:focus { border: round #f59e0b; }

    .split { height: 1fr; }
    #vmm-left { width: 55%; height: 1fr; margin-right: 1; }
    #vmm-right { width: 45%; height: 1fr; background: #131720;
                 border: round #262c3a; padding: 0 1; }
    #vmm-preview-scroll { height: 1fr; background: #0c0e14; border: solid #262c3a; }
    OpenContextApp.-narrow #vmm-left { width: 100%; margin-right: 0; }
    OpenContextApp.-narrow #vmm-right { display: none; }

    #chat-main { width: 65%; height: 1fr; background: #0c0e14;
                 border: round #262c3a; padding: 0 1; }
    #chat-side { width: 35%; height: 1fr; margin-left: 1; background: #131720;
                 border: round #262c3a; padding: 0 1; }
    OpenContextApp.-narrow #chat-main { width: 100%; }
    OpenContextApp.-narrow #chat-side { display: none; }
    #chat-input-row { height: 3; }
    #chat-input { width: 1fr; margin-right: 1; }

    #compactor-status { height: 1; padding: 0 1; background: #131720; }
    .compactor-pane { width: 50%; height: 1fr; background: #131720;
                      border: round #262c3a; padding: 0 1; }
    #pane-raw { margin-right: 1; }
    TextArea { height: 1fr; background: #0c0e14; border: solid #262c3a; color: #e6edf3; }
    TextArea:focus { border: solid #f59e0b; }

    #modal-container { width: 92%; height: 92%; background: #131720;
                       border: thick #d97706; padding: 0 1; }
    #modal-header { height: 2; align: left middle; }
    #modal-filename { color: #f59e0b; text-style: bold; width: 1fr; }
    #modal-meta { color: #3fb950; }
    #modal-code-scroll { height: 1fr; border: round #262c3a; background: #0c0e14; }
    #modal-actions { height: 3; align: left middle; }
    #modal-hint { width: 1fr; color: #8b949e; }

    .config-card { height: auto; background: #131720; border: round #262c3a;
                   padding: 0 1; margin-bottom: 1; }

    #context-left { width: 52%; height: 1fr; margin-right: 1; }
    #context-right { width: 48%; height: 1fr; background: #131720;
                     border: round #262c3a; padding: 0 1; }
    #context-scroll { height: 1fr; background: #0c0e14; border: solid #262c3a; }
    OpenContextApp.-narrow #context-left { width: 100%; margin-right: 0; }
    OpenContextApp.-narrow #context-right { display: none; }
    #context-summary { height: 4; padding: 0 1; background: #131720; }

    #help-container { width: 70%; height: 80%; background: #131720;
                      border: thick #d97706; padding: 0 1; }
    #help-title { color: #f59e0b; text-style: bold; height: 1; }
    #help-body { height: 1fr; }
    #help-hint { color: #8b949e; height: 1; }
    #benchmark-log { height: 14; background: #0c0e14; border: solid #262c3a; }
    """

    # Only the page actions appear in the footer. The number keys are already
    # labelled on the tabs themselves ("1 Overview"), and listing all thirteen
    # bindings made the footer wider than a 100-column terminal.
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("p", "page_in_selected", "Page in"),
        Binding("s", "swap_out_selected", "Swap out"),
        Binding("P", "toggle_pin_selected", "Pin"),
        Binding("o", "toggle_outline", "Outline"),
        Binding("r", "rescan", "Rescan"),
        Binding("1", "switch_tab('tab-overview')", "Overview", show=False),
        Binding("2", "switch_tab('tab-pages')", "Pages", show=False),
        Binding("3", "switch_tab('tab-context')", "Context", show=False),
        Binding("4", "switch_tab('tab-chat')", "Chat", show=False),
        Binding("5", "switch_tab('tab-compactor')", "Compact", show=False),
        Binding("enter", "inspect_selected", "Inspect", show=False),
        Binding("slash", "focus_search", "Search", show=False),
        Binding("question_mark", "help", "Help", show=False),
        Binding("w", "export_context", "Write context", show=False),
        Binding("ctrl+s", "save_session", "Save session", show=False),
        Binding("6", "switch_tab('tab-config')", "Config", show=False),
    ]

    def __init__(
        self,
        kernel: Optional[ContextKernel] = None,
        root_dir: Optional[str] = None,
        session_path: Optional[str] = DEFAULT_SESSION_PATH,
    ):
        super().__init__()
        self.kernel = kernel or ContextKernel()
        self.root_dir = root_dir
        self.pager = ContextPager(self.kernel, root_dir=root_dir)
        self.scanner = WorkspaceScanner(self.kernel, root_dir=root_dir)
        # None disables persistence entirely (used by tests and by --fresh).
        self.session_path = session_path
        self.page_filter = "ALL"
        self.preview_mode = "FULL"
        self.search_mode = "NAME"
        self.selected_page_id: Optional[str] = None
        self.ram_history: List[float] = []
        self.chat_turns = 0
        self._suppress_compactor_events = 0
        self._streaming_reply = ""
        self._last_export_path: Optional[str] = None

    # -- layout -----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with TabbedContent(initial="tab-overview", id="tabs"):
            with TabPane("1 Overview", id="tab-overview"):
                with Vertical(classes="tab-body"):
                    with Horizontal(classes="hud-bar"):
                        with Container(classes="panel-card card-ram", id="card-ram"):
                            yield Label("CONTEXT", classes="panel-title")
                            yield Label("0 / 0", id="hud-ram-tokens", classes="metric-big")
                            yield Label("0%", id="hud-ram-pct", classes="val-green")
                            yield ProgressBar(total=100, show_percentage=False, show_eta=False, id="hud-ram-bar")
                            yield Sparkline(data=[0.0], id="hud-sparkline")

                        with Container(classes="panel-card card-swap", id="card-swap"):
                            yield Label("SWAPPED", classes="panel-title")
                            yield Label("0 tok", id="hud-swap-tokens", classes="val-amber")
                            yield Label("0 pages", id="hud-swap-pages")
                            yield Label("", id="hud-swap-disk", classes="hud-detail")

                        with Container(classes="panel-card card-saved", id="card-saved"):
                            yield Label("KEPT OUT", classes="panel-title")
                            yield Label("+0 tok", id="hud-saved", classes="val-cyan")
                            yield Label("of context", classes="hud-detail")
                            yield Label("", id="hud-saved-detail", classes="hud-detail")

                        with Container(classes="panel-card card-faults", id="card-faults"):
                            yield Label("FAULTS", classes="panel-title")
                            yield Label("0", id="hud-faults", classes="val-purple")
                            yield Label("", id="hud-fault-latency")
                            yield Label("", id="hud-fault-detail", classes="hud-detail")

                    with Horizontal(classes="split"):
                        with Vertical(classes="panel-card", id="arch-card"):
                            yield Label("MEMORY HIERARCHY", classes="panel-title")
                            yield Static("", id="arch-map")
                        with Vertical(classes="panel-card wide-only", id="log-card"):
                            yield Label("KERNEL ACTIVITY", classes="panel-title")
                            yield RichLog(id="kernel-log", max_lines=60, markup=True)

                    with Horizontal(classes="cmd-bar"):
                        yield Button("Rescan", classes="btn-primary", id="btn-rescan")
                        yield Button("Page in all", id="btn-rehydrate")
                        yield Button("Swap cold", id="btn-swap-cold")
                        yield Button("Clear swap", id="btn-clear-swap")

            with TabPane("2 Pages", id="tab-pages"):
                with Vertical(classes="tab-body"):
                    with Horizontal(classes="cmd-bar"):
                        yield Input(
                            placeholder="Filter by name (press / to focus)", id="page-search"
                        )
                        yield Button("Name", id="btn-search-mode")
                        yield Button("All", id="btn-filter-all")
                        yield Button("Resident", id="btn-filter-ram")
                        yield Button("Swapped", id="btn-filter-swap")
                        yield Button("Pinned", id="btn-filter-pinned")

                    with Horizontal(classes="split"):
                        with Vertical(id="vmm-left"):
                            yield DataTable(id="page-table")
                        with Vertical(id="vmm-right"):
                            with Horizontal():
                                yield Label("PREVIEW", classes="panel-title")
                                yield Label("full", id="preview-mode", classes="val-cyan")
                            with VerticalScroll(id="vmm-preview-scroll"):
                                yield Static("Select a page.", id="preview-content")

                    with Horizontal(classes="cmd-bar"):
                        yield Button("Page in", id="btn-page-in")
                        yield Button("Swap out", id="btn-swap-out")
                        yield Button("Pin", id="btn-pin")
                        yield Button("Outline", id="btn-outline")
                        yield Button("Inspect", classes="btn-primary", id="btn-inspect")

            with TabPane("3 Context", id="tab-context"):
                with Vertical(classes="tab-body"):
                    with Horizontal(id="context-summary"):
                        yield Static("", id="context-totals")

                    with Horizontal(classes="split"):
                        with Vertical(id="context-left"):
                            yield Label("WHAT EACH PAGE CONTRIBUTES", classes="panel-title")
                            yield DataTable(id="context-table")
                        with Vertical(id="context-right"):
                            yield Label("ASSEMBLED CONTEXT WINDOW", classes="panel-title")
                            with VerticalScroll(id="context-scroll"):
                                yield Static("", id="context-text")

                    with Horizontal(classes="cmd-bar"):
                        yield Button("Refresh", id="btn-context-refresh")
                        yield Button("Write to file", classes="btn-primary", id="btn-context-export")
                        yield Static("", id="context-export-note")

            with TabPane("4 Chat", id="tab-chat"):
                with Vertical(classes="tab-body"):
                    with Horizontal(classes="split"):
                        with Vertical(id="chat-main"):
                            yield RichLog(id="chat-log", wrap=True, markup=True)
                            # The in-progress reply lands here: a RichLog cannot
                            # rewrite its last line, so streaming needs its own widget.
                            yield Static("", id="chat-streaming")
                        with Vertical(id="chat-side"):
                            yield Label("CONTEXT WINDOW", classes="panel-title")
                            yield Static("", id="chat-context")
                            yield Label("RESIDENT PAGES", classes="panel-title")
                            yield RichLog(id="chat-pages", wrap=True, markup=True)

                    with Horizontal(id="chat-input-row"):
                        yield Input(placeholder="Ask about the codebase, or /help", id="chat-input")
                        yield Button("Send", classes="btn-primary", id="btn-send")

            with TabPane("5 Compact", id="tab-compactor"):
                with Vertical(classes="tab-body"):
                    with Horizontal(classes="cmd-bar"):
                        yield Button("Python", id="btn-sample-py")
                        yield Button("Node", id="btn-sample-node")
                        yield Button("Rust", id="btn-sample-rust")
                        yield Button("Retry loop", id="btn-sample-loop")
                        yield Button("Clear", id="btn-clear-lab")
                        yield Button("Compact", classes="btn-primary", id="btn-compact")

                    with Horizontal(id="compactor-status"):
                        yield Label("Paste a traceback, or load a sample.", id="compactor-status-label")

                    with Horizontal(classes="split"):
                        with Vertical(classes="compactor-pane", id="pane-raw"):
                            yield Label("INPUT", classes="panel-title")
                            yield TextArea(id="compactor-raw", theme="monokai")
                        with Vertical(classes="compactor-pane", id="pane-compacted"):
                            yield Label("COMPACTED", classes="panel-title")
                            yield TextArea(id="compactor-out", read_only=True, theme="monokai")

            with TabPane("6 Config", id="tab-config"):
                with VerticalScroll(classes="tab-body"):
                    with Container(classes="config-card"):
                        yield Label("TOKEN BUDGET", classes="panel-title")
                        yield Label("Pages are evicted to disk when the context exceeds this.")
                        with Horizontal(classes="cmd-bar"):
                            for size in (4, 8, 16, 32, 64):
                                yield Button(f"{size}k", id=f"btn-budget-{size}k")

                    with Container(classes="config-card"):
                        yield Label("MODEL SERVER", classes="panel-title")
                        yield Static("", id="config-llm")

                    with Container(classes="config-card"):
                        yield Label("BENCHMARK", classes="panel-title")
                        yield Label("Measures compaction and page-fault latency on committed fixtures.")
                        with Horizontal(classes="cmd-bar"):
                            yield Button("Run benchmark", classes="btn-primary", id="btn-benchmark")
                        yield RichLog(id="benchmark-log", markup=True)

        yield Footer()

    # -- lifecycle ---------------------------------------------------------------

    def on_mount(self) -> None:
        table = self.query_one("#page-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("STATE", "TIER", "PAGE", "TOKENS")

        context_table = self.query_one("#context-table", DataTable)
        context_table.cursor_type = "row"
        # Share is folded into the tokens cell rather than given its own column: a
        # fifth column is the first thing to be truncated away on a narrow pane.
        context_table.add_columns("TIER", "PAGE", "INCLUDED AS", "TOKENS")

        self._apply_width_class(self.size.width)

        session_note = self._restore_session()

        self.refresh_all()
        self.set_interval(1.5, self._tick)

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"[bold #f59e0b]OpenContext {__version__}[/bold #f59e0b]")
        chat_log.write(f"[dim]{len(self.kernel.pages)} pages indexed. ? for keys, /help for commands.[/dim]")
        if session_note:
            chat_log.write(f"[dim]{session_note}[/dim]")
        self._refresh_llm_config()

    # -- session ------------------------------------------------------------------

    def _restore_session(self) -> str:
        """
        Reload the arrangement saved by a previous run.

        Pinning a page or setting a budget is close to meaningless if it evaporates
        on exit, so those decisions are carried across runs. Anything whose source
        has disappeared is dropped and reported rather than restored empty.
        """
        if not self.session_path or not session_exists(self.session_path):
            return ""

        try:
            result = restore_session(self.kernel, self.session_path)
        except SessionError as error:
            return f"Session not restored: {error}"

        if result.get("error"):
            return f"Session not restored: {result['error']}"

        message = f"Restored {result['restored']} pages from the previous session."
        dropped = result.get("dropped") or []
        if dropped:
            message += f" {len(dropped)} could not be restored: " + "; ".join(
                f"{item['id']} ({item['reason']})" for item in dropped[:3]
            )
        return message

    def action_save_session(self) -> None:
        """Write the session now, without waiting for exit."""
        if not self.session_path:
            self.notify("Session persistence is disabled for this run", severity="warning")
            return
        try:
            result = save_session(self.kernel, self.session_path, root_dir=self.root_dir)
        except SessionError as error:
            self.notify(f"Could not save session: {error}", severity="error")
            return
        self.notify(f"Saved {result['pages_saved']} pages to {result['path']}", title="Session")

    def on_unmount(self) -> None:
        """Persist the arrangement on the way out."""
        if not self.session_path:
            return
        try:
            save_session(self.kernel, self.session_path, root_dir=self.root_dir)
        except Exception:
            # Quitting must not fail because the session could not be written.
            pass

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def on_resize(self, event) -> None:
        self._apply_width_class(event.size.width)

    def _apply_width_class(self, width: int) -> None:
        """Toggle the narrow-layout class so CSS can shed detail that will not fit."""
        self.set_class(width < NARROW_WIDTH, "-narrow")

    def _tick(self) -> None:
        try:
            self.refresh_hud()
        except Exception:
            pass

    def refresh_all(self) -> None:
        for refresher in (
            self.refresh_hud,
            self.refresh_page_table,
            self.refresh_chat_sidebar,
            self.refresh_context_tab,
        ):
            try:
                refresher()
            except Exception:
                pass

    # -- overview ----------------------------------------------------------------

    def refresh_hud(self) -> None:
        metrics = self.kernel.get_metrics()

        self.query_one("#hud-ram-tokens", Label).update(
            f"{metrics.working_tokens:,} / {metrics.token_budget:,}"
        )
        pct_label = self.query_one("#hud-ram-pct", Label)
        pct_label.update(f"{metrics.budget_utilization_pct}%")
        pct_label.set_classes(
            "val-red" if metrics.over_budget
            else "val-amber" if metrics.budget_utilization_pct > 85
            else "val-green"
        )
        self.query_one("#hud-ram-bar", ProgressBar).progress = min(
            100.0, max(0.0, metrics.budget_utilization_pct)
        )

        self.ram_history.append(float(metrics.working_tokens))
        if len(self.ram_history) > 40:
            self.ram_history.pop(0)
        if len(self.ram_history) >= 2:
            self.query_one("#hud-sparkline", Sparkline).data = list(self.ram_history)

        self.query_one("#hud-swap-tokens", Label).update(f"{metrics.swapped_tokens:,} tok")
        self.query_one("#hud-swap-pages", Label).update(f"{metrics.l3_pages} pages")
        # Disk totals can exceed live totals when a previous run left rows behind.
        # Shown separately and only when they differ, so the two are never conflated.
        self.query_one("#hud-swap-disk", Label).update(
            f"disk: {metrics.swap_disk_rows} rows" if metrics.swap_disk_rows != metrics.l3_pages else ""
        )

        self.query_one("#hud-saved", Label).update(f"+{metrics.total_tokens_saved:,} tok")
        compacted_pages = sum(1 for page in self.kernel.pages.values() if page.compacted)
        self.query_one("#hud-saved-detail", Label).update(
            f"{compacted_pages} compacted" if compacted_pages else ""
        )

        self.query_one("#hud-faults", Label).update(str(metrics.total_page_faults))
        # Measured, not asserted: blank until a fault has actually been timed.
        self.query_one("#hud-fault-latency", Label).update(
            f"{metrics.avg_page_fault_ms:.2f} ms avg" if metrics.total_page_faults else "none yet"
        )

        pressure = (
            "over budget" if metrics.over_budget
            else "high" if metrics.budget_utilization_pct >= 85
            else "moderate" if metrics.budget_utilization_pct >= 65
            else "low"
        )
        pressure_color = {
            "over budget": "#f85149", "high": "#f59e0b",
            "moderate": "#d29922", "low": "#3fb950",
        }[pressure]

        self.query_one("#arch-map", Static).update(
            f"[bold #a371f7]L0 pinned[/bold #a371f7]    {metrics.l0_pages} pages, never evicted\n"
            f"[bold #3fb950]L1 working[/bold #3fb950]   {metrics.l1_pages} pages\n"
            f"[bold #38bdf8]L2 episodic[/bold #38bdf8]  {metrics.l2_pages} pages, evicted first\n"
            f"[bold #d29922]on disk[/bold #d29922]      {metrics.l3_pages} pages, {metrics.swapped_tokens:,} tok\n\n"
            f"Memory pressure  [{pressure_color} bold]{pressure}[/{pressure_color} bold]\n"
            f"Eviction policy  LRU, L2 before L1, L0 never\n"
            f"Token counting   {'tiktoken' if os.environ.get('OPENCONTEXT_TOKENIZER') == 'tiktoken' else 'heuristic'}"
        )

        log = self.query_one("#kernel-log", RichLog)
        log.clear()
        for event in metrics.recent_events[-10:]:
            name = event.event_type.value
            color = (
                "#f85149" if "WARNING" in name or "EXCEEDED" in name
                else "#f59e0b" if "FAULT" in name
                else "#d29922" if "SWAP" in name
                else "#38bdf8" if "COMPACT" in name
                else "#3fb950"
            )
            log.write(f"[dim]{event.formatted_time}[/dim] [{color}]{name}[/{color}] {event.description}")

    # -- context window ----------------------------------------------------------

    def refresh_context_tab(self) -> None:
        """
        Show the context window the model would actually receive.

        This is the product's output, so it is shown exactly: the assembled text,
        and a breakdown of every token in it. The per-page figures come from what
        `assemble_context` emitted, not from summing `page.token_count` — a swapped
        page contributes a ~24 token tombstone in place of thousands, and the
        section headers are real tokens no page owns. The rows plus the overhead
        line reconcile to the window's measured size.
        """
        try:
            table = self.query_one("#context-table", DataTable)
            totals = self.query_one("#context-totals", Static)
            body = self.query_one("#context-text", Static)
        except Exception:
            return

        report = self.kernel.assemble_context_report()
        total = report["total_tokens"]
        budget = self.kernel.token_budget
        segments = sorted(report["segments"], key=lambda item: -item["tokens"])

        table.clear()
        for segment in segments:
            share = (segment["tokens"] / total * 100) if total else 0.0
            included = segment["included_as"]
            if included == "tombstone":
                included = f"[#d29922]tombstone[/#d29922] of {segment.get('full_tokens', 0):,}"
            else:
                included = "[#3fb950]full content[/#3fb950]"
            table.add_row(
                {"L0_PINNED": "[#a371f7]L0[/#a371f7]", "L1_WORKING": "[#3fb950]L1[/#3fb950]"}.get(
                    segment["tier"], "[#38bdf8]L2[/#38bdf8]"
                ),
                segment["title"],
                included,
                f"{segment['tokens']:,} [dim]{share:.0f}%[/dim]",
                key=segment["page_id"],
            )

        overhead = report["overhead_tokens"]
        if overhead:
            share = (overhead / total * 100) if total else 0.0
            table.add_row(
                "[dim]--[/dim]",
                "[dim]section headers and framing[/dim]",
                "[dim]structure[/dim]",
                f"[dim]{overhead:,} {share:.0f}%[/dim]",
            )

        pct = (total / budget * 100) if budget else 0.0
        colour = "#f85149" if total > budget else ("#d29922" if pct > 85 else "#3fb950")
        swapped = sum(1 for item in segments if item["included_as"] == "tombstone")
        held_back = sum(
            item.get("full_tokens", 0) - item["tokens"]
            for item in segments
            if item["included_as"] == "tombstone"
        )

        totals.update(
            f"[bold {colour}]{total:,}[/bold {colour}] of {budget:,} tokens "
            f"([bold {colour}]{pct:.1f}%[/bold {colour}])   "
            f"{len(segments) - swapped} pages in full, {swapped} as tombstones\n"
            f"[dim]{held_back:,} tokens held on disk, recoverable on reference. "
            f"Rows below sum to the window's measured size.[/dim]"
        )

        text = report["context"]
        shown = text[:20000]
        if len(text) > 20000:
            shown += f"\n\n... {len(text) - 20000:,} more characters not shown"
        # Rendered as literal text, never as markup. The window contains section
        # headers like "=== [OpenContext L1_WORKING] ===" and arbitrary source code;
        # letting Rich interpret either would silently swallow bracketed content.
        body.update(Text(shown))

    def action_export_context(self) -> None:
        """Write the assembled context to a file and report where it went."""
        report = self.kernel.assemble_context_report()
        if not report["context"].strip():
            self.notify("The context window is empty", severity="warning")
            return

        directory = os.path.abspath(self.root_dir or os.getcwd())
        target = os.path.join(directory, f"opencontext-context-{int(time.time())}.txt")
        try:
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(report["context"])
        except OSError as error:
            self.notify(f"Could not write the file: {error}", severity="error")
            return

        self._last_export_path = target
        try:
            self.query_one("#context-export-note", Static).update(
                f"[#3fb950]{report['total_tokens']:,} tokens written to[/#3fb950] {target}"
            )
        except Exception:
            pass
        self.notify(f"{report['total_tokens']:,} tokens written to {target}", title="Context")

    # -- page table --------------------------------------------------------------

    def refresh_page_table(self, filter_text: str = "") -> None:
        table = self.query_one("#page-table", DataTable)
        table.clear()

        needle = filter_text.lower().strip()

        # In content mode the kernel ranks pages by keyword overlap over titles, ids
        # and resident text; name mode is a plain substring match on the label.
        ranked: Optional[dict] = None
        if needle and self.search_mode == "CONTENT":
            ranked = {page.id: score for page, score in self.kernel.search(needle, top_k=200)}

        pages = sorted(
            self.kernel.pages.values(),
            key=lambda p: (
                0 if p.tier == PageTier.L0_PINNED else (1 if p.status == PageStatus.ACTIVE else 2),
                -p.token_count,
            ),
        )

        first_id = None
        for page in pages:
            resident = page.status == PageStatus.ACTIVE
            if self.page_filter == "RAM" and not resident:
                continue
            if self.page_filter == "SWAP" and resident:
                continue
            if self.page_filter == "PINNED" and page.tier != PageTier.L0_PINNED:
                continue
            if needle:
                if ranked is not None:
                    if page.id not in ranked:
                        continue
                elif needle not in page.id.lower() and needle not in page.title.lower():
                    continue

            first_id = first_id or page.id
            table.add_row(
                "[#3fb950]resident[/#3fb950]" if resident else "[#d29922]swapped[/#d29922]",
                {"L0_PINNED": "[#a371f7]L0[/#a371f7]", "L1_WORKING": "[#3fb950]L1[/#3fb950]"}.get(
                    page.tier.value, "[#38bdf8]L2[/#38bdf8]"
                ),
                page.title,
                f"{page.token_count:,}",
                key=page.id,
            )

        if first_id and self.selected_page_id is None:
            self.selected_page_id = first_id
            self.update_preview(first_id)

        # Keep the cursor on whatever page the user was working with. Swapping a
        # page changes its sort position, so without this the cursor would land on
        # a different row and the next keypress would act on the wrong page.
        self._restore_cursor(table)

    def _restore_cursor(self, table: DataTable) -> None:
        """Move the cursor back to `selected_page_id` if that row is still listed."""
        if not self.selected_page_id or not table.row_count:
            return
        try:
            row_index = table.get_row_index(self.selected_page_id)
        except Exception:
            return
        try:
            table.move_cursor(row=row_index, animate=False)
        except Exception:
            pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key and event.row_key.value:
            self.selected_page_id = str(event.row_key.value)
            self.update_preview(self.selected_page_id)

    def update_preview(self, page_id: str) -> None:
        try:
            target = self.query_one("#preview-content", Static)
        except Exception:
            return

        page = self.kernel.pages.get(page_id)
        if page is None:
            target.update("Page not found.")
            return

        # A swapped page has no content in memory — it was released to disk when it
        # was evicted. Show the tombstone rather than an empty pane.
        if page.status == PageStatus.SWAPPED:
            target.update(
                f"[bold #d29922]Swapped to disk[/bold #d29922]\n\n"
                f"Page     {page.title}\n"
                f"Id       [dim]{page.id}[/dim]\n"
                f"On disk  [bold #d29922]{page.token_count:,} tokens[/bold #d29922]\n"
                f"In context  [bold #3fb950]{len(page.tombstone or '') // 4} tokens[/bold #3fb950] (tombstone)\n\n"
                f"[dim]{page.tombstone}[/dim]\n\n"
                f"[#38bdf8]Press p to page it back in.[/#38bdf8]"
            )
            return

        if not page.content:
            target.update("[dim]Page is empty.[/dim]")
            return

        lexer = detect_lexer(page.title)
        if self.preview_mode == "OUTLINE":
            outline, before, after = CodeOutlineCompactor.compact_code(page.title, page.content)
            if after < before:
                target.update(Syntax(outline, lexer, theme="monokai", line_numbers=True, word_wrap=True))
            else:
                target.update(
                    "[dim]Outline would not be smaller than the file. Showing full content.[/dim]\n\n"
                )
                target.update(Syntax(page.content[:6000], lexer, theme="monokai", line_numbers=True, word_wrap=True))
            return

        target.update(Syntax(page.content[:6000], lexer, theme="monokai", line_numbers=True, word_wrap=True))

    # -- chat --------------------------------------------------------------------

    def refresh_chat_sidebar(self) -> None:
        try:
            target = self.query_one("#chat-context", Static)
            pages_log = self.query_one("#chat-pages", RichLog)
        except Exception:
            return

        metrics = self.kernel.get_metrics()
        target.update(
            f"In context   [bold #3fb950]{metrics.working_tokens:,}[/bold #3fb950] tok\n"
            f"Budget       {metrics.token_budget:,} tok\n"
            f"Headroom     [bold #38bdf8]{max(0, metrics.token_budget - metrics.working_tokens):,}[/bold #38bdf8] tok\n"
            f"On disk      {metrics.swapped_tokens:,} tok\n"
            f"Kept out     [bold #38bdf8]+{metrics.total_tokens_saved:,}[/bold #38bdf8] tok"
        )

        pages_log.clear()
        resident = [p for p in self.kernel.pages.values() if p.status == PageStatus.ACTIVE]
        for page in sorted(resident, key=lambda p: -p.token_count)[:10]:
            pages_log.write(f"[#3fb950]{page.title}[/#3fb950] [dim]{page.token_count:,} tok[/dim]")

    def _refresh_llm_config(self) -> None:
        try:
            target = self.query_one("#config-llm", Static)
        except Exception:
            return
        from opencontext.llm import LLMConfig

        config = LLMConfig.from_env()
        target.update(
            f"Endpoint  {config.base_url}\n"
            f"Model     {config.model}\n"
            f"Location  {'local — nothing leaves this machine' if config.is_local else 'remote'}\n\n"
            f"[dim]Set OPENCONTEXT_UPSTREAM, OPENCONTEXT_MODEL and OPENCONTEXT_UPSTREAM_API_KEY to change.[/dim]"
        )

    def handle_chat_submit(self) -> None:
        chat_input = self.query_one("#chat-input", Input)
        text = chat_input.value.strip()
        if not text:
            return

        log = self.query_one("#chat-log", RichLog)
        log.write(f"\n[dim]{time.strftime('%H:%M:%S')}[/dim] [bold #f59e0b]you[/bold #f59e0b]  {text}")
        chat_input.value = ""

        if text.startswith("/"):
            self.handle_slash_command(text)
            return

        # The question becomes an episodic page. This is the tier OpenContext says it
        # manages, so the UI's own conversation has to live in it rather than beside
        # it — turns then age out under the same budget as everything else.
        self.pager.ingest_conversation_turn("user", text)
        self.chat_turns += 1

        rehydrated = self.kernel.touch_or_fault(text)
        if rehydrated:
            log.write(f"[#f59e0b]paged in[/#f59e0b] [dim]{', '.join(rehydrated)}[/dim]")

        matches = self.kernel.search(text, top_k=3)
        if matches:
            log.write(
                "[dim]most relevant: "
                + ", ".join(f"{page.title} ({score})" for page, score in matches)
                + "[/dim]"
            )

        self._streaming_reply = ""
        try:
            self.query_one("#chat-streaming", Static).update("[dim]waiting for the model...[/dim]")
        except Exception:
            pass

        self.refresh_all()
        self._ask_model(text)

    @work(thread=True, exclusive=True)
    def _ask_model(self, question: str) -> None:
        """
        Stream the answer from the configured model on a worker thread.

        Deltas are pushed to the UI as they arrive rather than after the whole reply
        lands, so a slow model shows progress instead of a frozen pane.

        No answer is fabricated when no model is reachable: the UI says so and points
        at the setting to change. A plausible-looking invented reply is worse than
        none, because the user cannot tell it from a real one.
        """
        from opencontext.llm import LLMClient, LLMUnavailable

        system_prompt = (
            "You are a software engineering assistant answering questions about a "
            "codebase. The context below was assembled by OpenContext from the user's "
            "workspace. Pages marked as swapped are on disk and not shown.\n\n"
            + self.kernel.assemble_context()
        )

        client = LLMClient()
        collected: List[str] = []
        try:
            for delta in client.stream(system_prompt, question):
                collected.append(delta)
                self.call_from_thread(self._write_chat_delta, "".join(collected))
        except LLMUnavailable as error:
            self.call_from_thread(self._write_chat_error, str(error))
            return
        except Exception as error:
            self.call_from_thread(self._write_chat_error, f"Model call failed: {error}")
            return

        answer = "".join(collected).strip()
        if not answer:
            self.call_from_thread(self._write_chat_error, "The model returned an empty response.")
            return
        self.call_from_thread(self._write_chat_answer, answer)

    def _write_chat_delta(self, partial: str) -> None:
        """Render the reply so far into the streaming pane."""
        self._streaming_reply = partial
        try:
            self.query_one("#chat-streaming", Static).update(
                f"[bold #3fb950]model[/bold #3fb950]  {partial}"
            )
        except Exception:
            pass

    def _write_chat_answer(self, answer: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold #3fb950]model[/bold #3fb950]  {answer}")
        self._clear_streaming_pane()

        # Ingested only now that the answer exists. Ingesting before the call would
        # have created an empty page and charged the budget for nothing.
        self.pager.ingest_conversation_turn("assistant", answer)
        self.chat_turns += 1
        self.refresh_all()

    def _clear_streaming_pane(self) -> None:
        self._streaming_reply = ""
        try:
            self.query_one("#chat-streaming", Static).update("")
        except Exception:
            pass

    def _write_chat_error(self, message: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        self._clear_streaming_pane()
        log.write("[bold #f85149]no model available[/bold #f85149]")
        log.write(f"[dim]{message}[/dim]")
        log.write(
            "[dim]OpenContext still assembled and paged the context — only the answer "
            "needs a model. Press 5 to see the current endpoint.[/dim]"
        )

    def handle_slash_command(self, command_text: str) -> None:
        parts = command_text.split()
        command, args = parts[0].lower(), parts[1:]
        log = self.query_one("#chat-log", RichLog)

        def find_page(needle: str):
            matches = [
                p for p in self.kernel.pages.values()
                if needle.lower() in p.title.lower() or needle.lower() in p.id.lower()
            ]
            return matches[0] if matches else None

        if command == "/help":
            log.write(
                "[bold #f59e0b]commands[/bold #f59e0b]\n"
                "  /status          context memory summary\n"
                "  /page-in <file>  restore a page from disk\n"
                "  /swap <file>     move a page to disk\n"
                "  /pin <file>      protect a page from eviction\n"
                "  /unpin <file>    allow a page to be evicted\n"
                "  /budget <n>      set the token budget\n"
                "  /scan            re-index the project\n"
                "  /loop            run the retry-loop demonstration\n"
                "  /clear           clear this log"
            )
        elif command == "/status":
            metrics = self.kernel.get_metrics()
            log.write(
                f"{metrics.working_tokens:,} / {metrics.token_budget:,} tok "
                f"({metrics.budget_utilization_pct}%) - {metrics.l3_pages} pages on disk - "
                f"+{metrics.total_tokens_saved:,} tok kept out - {metrics.total_page_faults} faults"
            )
        elif command in ("/page-in", "/fault") and args:
            restored = self.kernel.touch_or_fault(" ".join(args))
            log.write(
                f"[#3fb950]paged in {', '.join(restored)}[/#3fb950]"
                if restored else f"[dim]nothing swapped matches '{' '.join(args)}'[/dim]"
            )
        elif command == "/swap" and args:
            page = find_page(args[0])
            if page is None:
                log.write(f"[dim]no page matching '{args[0]}'[/dim]")
            elif self.kernel.page_out(page.id):
                log.write(f"[#d29922]swapped {page.title} to disk[/#d29922]")
            else:
                log.write(f"[dim]{page.title} is pinned or already swapped[/dim]")
        elif command == "/pin" and args:
            page = find_page(args[0])
            if page and self.kernel.pin_page(page.id):
                log.write(f"[#a371f7]pinned {page.title}[/#a371f7]")
            else:
                log.write(f"[dim]no page matching '{args[0]}'[/dim]")
        elif command == "/unpin" and args:
            page = find_page(args[0])
            if page and self.kernel.unpin_page(page.id):
                log.write(f"[#3fb950]unpinned {page.title}[/#3fb950]")
            else:
                log.write(f"[dim]no page matching '{args[0]}'[/dim]")
        elif command == "/budget" and args:
            try:
                self.kernel.set_budget(int(args[0]))
                log.write(f"[#38bdf8]budget set to {self.kernel.token_budget:,} tok[/#38bdf8]")
            except ValueError:
                log.write("[#f85149]not a number. example: /budget 12000[/#f85149]")
        elif command == "/scan":
            result = self.scanner.scan_and_ingest()
            log.write(f"[#3fb950]indexed {result['files_scanned']} files[/#3fb950]")
        elif command == "/loop":
            self.action_switch_tab("tab-compactor")
            self.run_retry_loop_demo()
        elif command == "/clear":
            log.clear()
        else:
            log.write(f"[dim]unknown command '{command}'. /help for the list.[/dim]")

        self.refresh_all()

    # -- compactor ---------------------------------------------------------------

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "compactor-raw":
            return
        if self._suppress_compactor_events > 0:
            self._suppress_compactor_events -= 1
            return
        self.run_compaction()

    def run_compaction(self) -> None:
        try:
            raw_area = self.query_one("#compactor-raw", TextArea)
            out_area = self.query_one("#compactor-out", TextArea)
            status = self.query_one("#compactor-status-label", Label)
        except Exception:
            return

        raw = raw_area.text
        if not raw.strip():
            out_area.text = ""
            status.update("Paste a traceback, or load a sample.")
            return

        compacted, before, after = TracebackCompactor.compact(raw)
        language = TracebackCompactor.detect_language(raw)
        out_area.text = compacted

        if before == after:
            # The compactor returns the input untouched when it cannot shrink it.
            # Reported honestly rather than as a saving of zero percent.
            status.update(
                f"[#d29922]Not compacted[/#d29922] - {before:,} tok - detected as {language} - "
                "output would not have been smaller"
            )
        else:
            saved = before - after
            status.update(
                f"[#38bdf8]{before:,}[/#38bdf8] -> [#3fb950]{after:,} tok[/#3fb950]  "
                f"[#3fb950]-{round(saved / before * 100, 1)}%[/#3fb950]  "
                f"[dim]{saved:,} tokens removed, {language}[/dim]"
            )

    def load_sample(self, name: str) -> None:
        """Load a sample trace. Real captured output, not invented text."""
        samples = {
            "py": self._read_fixture("python_traceback.txt"),
            "node": self._read_fixture("node_stacktrace.txt"),
            "rust": (
                "thread 'main' panicked at src/vmm/page_table.rs:182:9:\n"
                "assertion `left == right` failed: page directory checksum mismatch\n"
                "  left: 0xDEADBEEF\n"
                " right: 0xCAFEBABE\n"
                "stack backtrace:\n"
                "   0: std::panicking::begin_panic\n"
                "   1: core::panicking::panic_fmt\n"
                "   2: opencontext::vmm::verify_checksum\n"
                "   3: opencontext::vmm::page_table::commit\n"
                "   4: opencontext::main\n"
                "note: run with `RUST_BACKTRACE=full` for a verbose backtrace"
            ),
        }
        text = samples.get(name, "")
        if not text:
            self.notify(f"Sample '{name}' is unavailable", severity="warning")
            return
        self.query_one("#compactor-raw", TextArea).text = text
        self.run_compaction()

    @staticmethod
    def _read_fixture(name: str) -> str:
        """Read a committed fixture, falling back to a short inline trace."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "tests", "fixtures", name,
        )
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return (
                "Traceback (most recent call last):\n"
                '  File "/srv/app/services/payments.py", line 51, in charge\n'
                '    response = await self._client.post("/v1/charges", json=payload)\n'
                "httpx.ConnectTimeout: timed out connecting to payments-gateway.internal:8443"
            )

    def run_retry_loop_demo(self) -> None:
        """
        Demonstrate the retry loop, showing what the compactor actually produced.

        The output pane is filled from the kernel's own compacted page. An earlier
        version displayed a hand-written summary — including a fabricated tombstone
        hash — that the compactor had never generated.
        """
        result = run_doom_loop(self.kernel, iterations=4)

        self._suppress_compactor_events = 2
        try:
            self.query_one("#compactor-raw", TextArea).text = result["raw_sample"]
            self.query_one("#compactor-out", TextArea).text = result["compacted_sample"]
            detection = (
                "repeat-failure alert raised"
                if result["repeat_failure_detected"]
                else "no repeat-failure alert"
            )
            self.query_one("#compactor-status-label", Label).update(
                f"[#38bdf8]{result['raw_tokens']:,}[/#38bdf8] -> "
                f"[#3fb950]{result['compacted_tokens']:,} tok[/#3fb950]  "
                f"[#3fb950]-{result['pct_saved']}%[/#3fb950]  "
                f"[dim]{result['iterations']} identical failures, {detection}[/dim]"
            )
        except Exception:
            pass

        self.notify(
            f"{result['iterations']} retries compacted: {result['tokens_saved']:,} tokens kept out "
            f"(-{result['pct_saved']}%)",
            title="Retry loop",
        )
        self.refresh_all()

    @work(thread=True, exclusive=True)
    def run_benchmark_worker(self) -> None:
        """Run the real benchmark off the UI thread and stream results back."""
        from opencontext.benchmark import run_benchmark

        try:
            report = run_benchmark()
        except Exception as error:
            self.call_from_thread(self._write_benchmark_line, f"[#f85149]Benchmark failed: {error}[/#f85149]")
            return

        lines = ["[bold #f59e0b]Token reduction[/bold #f59e0b]"]
        for result in report.results:
            lines.append(
                f"  {result.name}\n"
                f"    [#38bdf8]{result.before_tokens:,}[/#38bdf8] -> "
                f"[#3fb950]{result.after_tokens:,} tok[/#3fb950]  "
                f"[bold #3fb950]-{result.reduction_pct}%[/bold #3fb950]"
            )
        latency = report.latency
        if latency:
            lines.append("\n[bold #f59e0b]Page fault latency[/bold #f59e0b]")
            lines.append(
                f"  median [bold #38bdf8]{latency['page_fault_median_ms']:.3f} ms[/bold #38bdf8]  "
                f"mean {latency['page_fault_mean_ms']:.3f} ms  "
                f"p95 {latency['page_fault_p95_ms']:.3f} ms  "
                f"[dim]over {int(latency['samples'])} samples[/dim]"
            )
        lines.append(f"\n[dim]{report.environment['tokenizer']}, Python {report.environment['python']}[/dim]")

        self.call_from_thread(self._write_benchmark_line, "\n".join(lines))

    def _write_benchmark_line(self, text: str) -> None:
        self.query_one("#benchmark-log", RichLog).write(text)

    # -- actions -----------------------------------------------------------------

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id

    def action_focus_search(self) -> None:
        self.action_switch_tab("tab-pages")
        self.query_one("#page-search", Input).focus()

    def action_toggle_search_mode(self) -> None:
        """Switch the page filter between matching names and searching content."""
        self.search_mode = "CONTENT" if self.search_mode == "NAME" else "NAME"
        searching_content = self.search_mode == "CONTENT"

        self.query_one("#btn-search-mode", Button).label = "Content" if searching_content else "Name"
        self.query_one("#page-search", Input).placeholder = (
            "Search page contents" if searching_content else "Filter by name (press / to focus)"
        )
        self.refresh_page_table(self.query_one("#page-search", Input).value)
        self.notify(
            "Searching page contents" if searching_content else "Filtering by name",
            title="Search mode",
        )

    def action_toggle_outline(self) -> None:
        self.preview_mode = "OUTLINE" if self.preview_mode == "FULL" else "FULL"
        self.query_one("#preview-mode", Label).update(self.preview_mode.lower())
        if self.selected_page_id:
            self.update_preview(self.selected_page_id)

    def _current_page_id(self) -> Optional[str]:
        table = self.query_one("#page-table", DataTable)
        if table.row_count and table.cursor_row is not None:
            try:
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                return str(row_key.value)
            except Exception:
                pass
        return self.selected_page_id

    def action_page_in_selected(self) -> None:
        page_id = self._current_page_id()
        if not page_id:
            self.notify("Select a page first", severity="warning")
            return
        self.selected_page_id = page_id
        page = self.kernel.page_fault(page_id)
        self.notify(
            f"Paged in {page.title} ({page.token_count:,} tok)" if page
            else "Page is not on disk", severity="information" if page else "warning",
        )
        self.refresh_all()
        self.update_preview(page_id)

    def action_swap_out_selected(self) -> None:
        page_id = self._current_page_id()
        if not page_id:
            self.notify("Select a page first", severity="warning")
            return
        self.selected_page_id = page_id
        page = self.kernel.pages.get(page_id)
        if page and page.tier == PageTier.L0_PINNED:
            self.notify("Pinned pages are protected from eviction", severity="warning")
            return
        if self.kernel.page_out(page_id):
            self.notify(f"Swapped {page_id} to disk")
        else:
            self.notify("Page is already on disk", severity="warning")
        self.refresh_all()
        self.update_preview(page_id)

    def action_toggle_pin_selected(self) -> None:
        page_id = self._current_page_id()
        page = self.kernel.pages.get(page_id) if page_id else None
        if page is None:
            self.notify("Select a page first", severity="warning")
            return
        self.selected_page_id = page_id
        if page.tier == PageTier.L0_PINNED:
            self.kernel.unpin_page(page_id)
            self.notify(f"Unpinned {page.title}")
        else:
            self.kernel.pin_page(page_id)
            self.notify(f"Pinned {page.title} — protected from eviction")
        self.refresh_all()
        self.update_preview(page_id)

    def action_inspect_selected(self) -> None:
        if self.query_one("#tabs", TabbedContent).active != "tab-pages":
            return
        page_id = self._current_page_id()
        if page_id:
            self.push_screen(CodeViewModal(self.kernel, page_id))
        else:
            self.notify("Select a page first", severity="warning")

    def action_rescan(self) -> None:
        result = self.scanner.scan_and_ingest()
        self.notify(
            f"Indexed {result['files_scanned']} files ({result['total_tokens']:,} tok)",
            title="Workspace",
        )
        self.refresh_all()

    # -- events ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "page-search":
            self.refresh_page_table(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self.handle_chat_submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""

        simple = {
            "btn-rescan": self.action_rescan,
            "btn-page-in": self.action_page_in_selected,
            "btn-swap-out": self.action_swap_out_selected,
            "btn-pin": self.action_toggle_pin_selected,
            "btn-outline": self.action_toggle_outline,
            "btn-inspect": self.action_inspect_selected,
            "btn-send": self.handle_chat_submit,
            "btn-compact": self.run_compaction,
            "btn-sample-loop": self.run_retry_loop_demo,
            "btn-benchmark": self.run_benchmark_worker,
            "btn-context-refresh": self.refresh_context_tab,
            "btn-context-export": self.action_export_context,
            "btn-search-mode": self.action_toggle_search_mode,
        }
        if button_id in simple:
            simple[button_id]()
            return

        if button_id == "btn-rehydrate":
            count = self.kernel.rehydrate_all()
            self.notify(f"Paged in {count} pages")
        elif button_id == "btn-swap-cold":
            count = self.kernel.swap_all_unpinned()
            self.notify(f"Swapped {count} pages to disk")
        elif button_id == "btn-clear-swap":
            rows = self.kernel.clear_swap()
            self.selected_page_id = None
            self.notify(f"Cleared {rows} rows from the swap database")
        elif button_id.startswith("btn-filter-"):
            self.page_filter = button_id.rsplit("-", 1)[-1].upper()
            if self.page_filter == "ALL":
                self.page_filter = "ALL"
            self.refresh_page_table()
            return
        elif button_id.startswith("btn-sample-"):
            self.load_sample(button_id.rsplit("-", 1)[-1])
            return
        elif button_id == "btn-clear-lab":
            self._suppress_compactor_events = 2
            self.query_one("#compactor-raw", TextArea).text = ""
            self.query_one("#compactor-out", TextArea).text = ""
            self.query_one("#compactor-status-label", Label).update("Paste a traceback, or load a sample.")
            return
        elif button_id.startswith("btn-budget-"):
            thousands = int(button_id.rsplit("-", 1)[-1].rstrip("k"))
            self.kernel.set_budget(thousands * 1000)
            self.notify(f"Budget set to {thousands * 1000:,} tokens")
        else:
            return

        self.refresh_all()


def run_interactive_tui(
    kernel: Optional[ContextKernel] = None,
    root_dir: Optional[str] = None,
    session_path: Optional[str] = DEFAULT_SESSION_PATH,
) -> None:
    """Entry point for `opencontext tui`."""
    OpenContextApp(kernel, root_dir=root_dir, session_path=session_path).run()
