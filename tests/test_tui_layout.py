"""
Layout regression tests for the interactive TUI.

The TUI was previously unusable at 80x24: card titles truncated mid-word, the four
HUD cards rendered at different heights so the row closed ragged, and everything
below the HUD was pushed off-screen. These tests render the real application at
several terminal sizes and assert measurable properties, so a future CSS change
that re-breaks the layout fails here rather than in a user's terminal.
"""

from __future__ import annotations

import re

import pytest

from contextos.core.kernel import ContextKernel
from contextos.interfaces.interactive_tui import ContextOSApp
from contextos.storage.swap import SwapStorage

TERMINAL_SIZES = [(80, 24), (100, 30), (120, 40), (160, 50)]

# Text that must appear intact whenever its card is visible.
HUD_TITLES = ["CONTEXT", "SWAPPED", "KEPT OUT", "FAULTS"]
HUD_CARDS = ["#card-ram", "#card-swap", "#card-saved", "#card-faults"]


def build_kernel(tmp_path, pages: int = 6) -> ContextKernel:
    kernel = ContextKernel(
        token_budget=8000, swap_storage=SwapStorage(str(tmp_path / "layout.db"))
    )
    for index in range(pages):
        kernel.allocate_page(
            page_id=f"file:module_{index}.py",
            title=f"src/module_{index}.py",
            content=f"def handler_{index}(request):\n    return process(request)\n" * 12,
        )
    return kernel


def rendered_text(app: ContextOSApp) -> str:
    """Extract the visible text of the current frame from an exported screenshot."""
    svg = app.export_screenshot()
    chunks = re.findall(r"<text[^>]*>(.*?)</text>", svg, re.S)
    text = "".join(chunks)
    for entity, char in (("&#160;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')):
        text = text.replace(entity, char)
    return text


@pytest.mark.asyncio
@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
async def test_hud_cards_share_one_height(tmp_path, width, height):
    """
    All four HUD cards must be the same height.

    The RAM card holds a progress bar and a sparkline, so without an explicit
    shared height it grew taller than its neighbours and the row rendered with
    three cards closing early against one long one.
    """
    app = ContextOSApp(kernel=build_kernel(tmp_path), session_path=None)
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        heights = {
            card: app.query_one(card).region.height
            for card in HUD_CARDS
            if app.query_one(card).display
        }
        assert len(set(heights.values())) <= 1, f"HUD cards differ in height at {width}x{height}: {heights}"


@pytest.mark.asyncio
@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
async def test_no_horizontal_overflow(tmp_path, width, height):
    """No visible widget may extend past the right edge of the terminal."""
    app = ContextOSApp(kernel=build_kernel(tmp_path), session_path=None)
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        overflowing = [
            (node.id or node.__class__.__name__, node.region.right)
            for node in app.screen.walk_children()
            if node.display and node.region.width > 0 and node.region.right > width
        ]
        assert not overflowing, f"Widgets overflow {width} columns: {overflowing}"


@pytest.mark.asyncio
@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
async def test_hud_titles_render_untruncated(tmp_path, width, height):
    """
    Card titles must appear in full.

    At 80 columns the old titles rendered as 'ACTIVE WORKI', 'L3 VIRTUAL SW' and
    'AUTONOMOUS PA'. Titles are now short enough to fit the narrowest supported
    terminal.
    """
    app = ContextOSApp(kernel=build_kernel(tmp_path), session_path=None)
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        text = rendered_text(app)
        missing = [title for title in HUD_TITLES if title not in text]
        assert not missing, f"Titles truncated or missing at {width}x{height}: {missing}"


@pytest.mark.asyncio
async def test_page_table_visible_on_small_terminal(tmp_path):
    """
    The page table must have usable height at 80x24.

    Previously the HUD consumed the entire 24-row viewport and everything below it
    rendered as a bare rule.
    """
    app = ContextOSApp(kernel=build_kernel(tmp_path), session_path=None)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_switch_tab("tab-pages")
        await pilot.pause()

        table = app.query_one("#page-table")
        assert table.display, "Page table is not displayed at 80x24"
        assert table.region.height >= 5, f"Page table only {table.region.height} rows tall at 80x24"
        assert table.row_count >= 6, "Page table did not populate"


@pytest.mark.asyncio
async def test_narrow_layout_hides_side_panes(tmp_path):
    """Below the narrow threshold the split panes collapse instead of squeezing."""
    app = ContextOSApp(kernel=build_kernel(tmp_path), session_path=None)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.has_class("-narrow")
        app.action_switch_tab("tab-pages")
        await pilot.pause()
        assert not app.query_one("#vmm-right").display, "Preview pane should collapse on a narrow terminal"
        assert app.query_one("#vmm-left").region.width >= 70, "Table should take the full width when alone"


@pytest.mark.asyncio
async def test_wide_layout_shows_side_panes(tmp_path):
    """At full width both panes are shown."""
    app = ContextOSApp(kernel=build_kernel(tmp_path), session_path=None)
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert not app.has_class("-narrow")
        app.action_switch_tab("tab-pages")
        await pilot.pause()
        assert app.query_one("#vmm-right").display, "Preview pane should be visible on a wide terminal"


@pytest.mark.asyncio
async def test_resize_updates_layout_class(tmp_path):
    """Resizing between wide and narrow re-applies the layout class both ways."""
    app = ContextOSApp(kernel=build_kernel(tmp_path), session_path=None)
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert not app.has_class("-narrow")

        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.has_class("-narrow")

        await pilot.resize_terminal(140, 44)
        await pilot.pause()
        assert not app.has_class("-narrow")


@pytest.mark.asyncio
@pytest.mark.parametrize("width,height", TERMINAL_SIZES)
async def test_every_tab_renders_at_every_size(tmp_path, width, height):
    """Switching through every tab must not raise at any supported size."""
    app = ContextOSApp(kernel=build_kernel(tmp_path), session_path=None)
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        for tab in (
            "tab-overview",
            "tab-pages",
            "tab-context",
            "tab-chat",
            "tab-compactor",
            "tab-config",
        ):
            app.action_switch_tab(tab)
            await pilot.pause()
            assert app.query_one("#tabs").active == tab
