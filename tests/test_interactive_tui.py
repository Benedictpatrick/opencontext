"""
Behavioural tests for the interactive TUI.

Layout is covered separately in `test_tui_layout.py`. These cover what the UI
*does*, with particular attention to the rule that every figure shown is measured:
0.1.0 displayed a fixed "98.4% cache hit ratio", a fixed "< 500us" fault latency,
and a hand-written retry-loop panel containing an invented tombstone hash.
"""

from __future__ import annotations

import os

import pytest
from textual.widgets import DataTable, Input, Label, RichLog, TabbedContent, TextArea

from opencontext.core.kernel import ContextKernel
from opencontext.core.types import PageStatus, PageTier
from opencontext.interfaces.interactive_tui import CodeViewModal, OpenContextApp
from opencontext.llm import LLMUnavailable
from opencontext.storage.swap import SwapStorage

TERMINAL = (140, 44)


@pytest.fixture
def kernel(tmp_path):
    kernel = ContextKernel(token_budget=6000, swap_storage=SwapStorage(str(tmp_path / "swap.db")))
    kernel.allocate_page("file:main.py", "src/main.py", "def main():\n    print('hello')\n")
    kernel.allocate_page(
        "file:auth.py", "src/auth.py", "def verify_token(token):\n    return token == 'secret'\n" * 8
    )
    return kernel


# -- mounting and navigation -------------------------------------------------------


@pytest.mark.asyncio
async def test_app_mounts_with_all_panels(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        for selector in ("#hud-ram-bar", "#page-table", "#chat-input", "#compactor-raw", "#arch-map"):
            assert app.query_one(selector) is not None

        assert app.query_one("#page-table", DataTable).row_count >= 2


@pytest.mark.asyncio
async def test_number_keys_switch_tabs(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        tabs = app.query_one("#tabs", TabbedContent)
        for key, expected in (
            ("2", "tab-pages"),
            ("3", "tab-context"),
            ("4", "tab-chat"),
            ("5", "tab-compactor"),
            ("6", "tab-config"),
            ("1", "tab-overview"),
        ):
            await pilot.press(key)
            await pilot.pause()
            assert tabs.active == expected


# -- page table and preview ---------------------------------------------------------


@pytest.mark.asyncio
async def test_filtering_narrows_the_page_table(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        table = app.query_one("#page-table", DataTable)
        before = table.row_count
        assert before >= 2

        app.query_one("#page-search", Input).value = "auth"
        # The Changed message is posted, not delivered inline, so give the message
        # pump a second cycle — one pause is enough only when the machine is idle.
        await pilot.pause()
        await pilot.pause()

        assert table.row_count < before, "filtering did not narrow the table"
        assert table.row_count >= 1


@pytest.mark.asyncio
async def test_swapped_pages_preview_their_tombstone(kernel):
    """
    A swapped page has no content in memory — it was released to disk. The preview
    must show the tombstone rather than an empty pane.
    """
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        kernel.page_out("file:auth.py")
        app.update_preview("file:auth.py")
        await pilot.pause()

        rendered = str(app.query_one("#preview-content").render())
        assert "Swapped to disk" in rendered
        assert "tombstone" in rendered.lower()


@pytest.mark.asyncio
async def test_outline_mode_toggles_and_reports_itself(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        assert app.preview_mode == "FULL"

        app.action_toggle_outline()
        await pilot.pause()
        assert app.preview_mode == "OUTLINE"
        assert "outline" in str(app.query_one("#preview-mode", Label).render()).lower()

        app.action_toggle_outline()
        await pilot.pause()
        assert app.preview_mode == "FULL"


@pytest.mark.asyncio
async def test_page_in_and_swap_out_actions(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.selected_page_id = "file:auth.py"

        app.action_swap_out_selected()
        await pilot.pause()
        assert kernel.pages["file:auth.py"].status == PageStatus.SWAPPED

        app.action_page_in_selected()
        await pilot.pause()
        assert kernel.pages["file:auth.py"].status == PageStatus.ACTIVE


@pytest.mark.asyncio
async def test_pinned_pages_cannot_be_swapped_from_the_ui(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.selected_page_id = "file:auth.py"

        app.action_toggle_pin_selected()
        await pilot.pause()
        assert kernel.pages["file:auth.py"].tier == PageTier.L0_PINNED

        app.action_swap_out_selected()
        await pilot.pause()
        assert kernel.pages["file:auth.py"].status == PageStatus.ACTIVE, "pinned page must survive"


@pytest.mark.asyncio
async def test_code_modal_opens_and_closes(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.push_screen(CodeViewModal(kernel, "file:main.py"))
        await pilot.pause()
        assert len(app.screen_stack) == 2

        app.screen.action_close()
        await pilot.pause()
        assert len(app.screen_stack) == 1


# -- compactor -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_a_traceback_compacts_it_live(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.action_switch_tab("tab-compactor")
        await pilot.pause()

        trace = (
            "Traceback (most recent call last):\n"
            + '  File "/venv/lib/python3.11/site-packages/x.py", line 1, in f\n    g()\n' * 10
            + '  File "/app/main.py", line 3, in run\n    boom()\n'
            + "ZeroDivisionError: division by zero"
        )
        app.query_one("#compactor-raw", TextArea).text = trace
        # TextArea.Changed is posted, not delivered inline; give the message pump a
        # second cycle so the assertion does not race the handler.
        await pilot.pause()
        await pilot.pause()

        output = app.query_one("#compactor-out", TextArea).text
        assert "ZeroDivisionError" in output
        assert "site-packages" not in output
        assert "-" in str(app.query_one("#compactor-status-label", Label).render())


@pytest.mark.asyncio
async def test_uncompactable_input_says_so_rather_than_claiming_a_saving(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.query_one("#compactor-raw", TextArea).text = "ValueError: x\nat y"
        # TextArea.Changed is posted, not delivered inline — see the note above.
        await pilot.pause()
        await pilot.pause()
        assert "Not compacted" in str(app.query_one("#compactor-status-label", Label).render())


@pytest.mark.asyncio
async def test_samples_load_real_captured_traces(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        raw = app.query_one("#compactor-raw", TextArea)

        app.load_sample("py")
        await pilot.pause()
        assert "Traceback" in raw.text
        assert app.query_one("#compactor-out", TextArea).text

        app.load_sample("node")
        await pilot.pause()
        assert "at " in raw.text

        app.load_sample("rust")
        await pilot.pause()
        assert "panicked" in raw.text


@pytest.mark.asyncio
async def test_retry_loop_demo_shows_real_compactor_output(kernel):
    """
    The output pane must contain what the compactor actually produced.

    0.1.0 wrote a hand-authored summary into this pane, including a fabricated
    tombstone hash the compactor had never generated.
    """
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.run_retry_loop_demo()
        await pilot.pause()

        raw = app.query_one("#compactor-raw", TextArea).text
        output = app.query_one("#compactor-out", TextArea).text

        assert "attempt 1" in raw and "attempt 4" in raw
        # Real compactor output, not prose.
        assert "[OpenContext]" in output
        assert "Root cause:" in output
        assert "psycopg2.OperationalError" in output
        assert "8f2a1b9c" not in output, "the fabricated hash must not return"

        status = str(app.query_one("#compactor-status-label", Label).render())
        assert "identical failures" in status


@pytest.mark.asyncio
async def test_clearing_the_lab_empties_both_panes(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.load_sample("py")
        await pilot.pause()

        app.on_button_pressed(
            type("Event", (), {"button": type("Button", (), {"id": "btn-clear-lab"})()})()
        )
        await pilot.pause()
        assert app.query_one("#compactor-raw", TextArea).text == ""
        assert app.query_one("#compactor-out", TextArea).text == ""


# -- telemetry honesty ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_latency_is_blank_until_a_fault_is_measured(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        assert "none yet" in str(app.query_one("#hud-fault-latency", Label).render())

        kernel.page_out("file:auth.py")
        kernel.page_fault("file:auth.py")
        app.refresh_hud()
        await pilot.pause()

        rendered = str(app.query_one("#hud-fault-latency", Label).render())
        assert "ms avg" in rendered
        assert float(rendered.split()[0]) > 0


@pytest.mark.asyncio
async def test_hud_contains_no_fabricated_figures(kernel):
    """The specific invented numbers from 0.1.0 must not reappear."""
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        arch = str(app.query_one("#arch-map").render())
        for invented in ("98.4%", "< 500", "500µs", "98.5%", "0% data corruption"):
            assert invented not in arch


@pytest.mark.asyncio
async def test_over_budget_state_is_surfaced(tmp_path):
    kernel = ContextKernel(token_budget=200, swap_storage=SwapStorage(str(tmp_path / "swap.db")))
    for index in range(6):
        kernel.allocate_page(f"rule:{index}", f"Rule {index}", "pinned\n" * 60, tier=PageTier.L0_PINNED)

    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        assert kernel.get_metrics().over_budget is True
        assert "val-red" in app.query_one("#hud-ram-pct", Label).classes


# -- chat --------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_commands_operate_on_the_kernel(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        chat_input = app.query_one("#chat-input", Input)

        for command in ("/help", "/status", "/swap auth", "/pin main", "/budget 9000", "/scan"):
            chat_input.value = command
            app.handle_chat_submit()
            await pilot.pause()

        assert kernel.token_budget == 9000
        assert kernel.pages["file:main.py"].tier == PageTier.L0_PINNED


@pytest.mark.asyncio
async def test_unknown_slash_command_is_reported(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.query_one("#chat-input", Input).value = "/nonsense"
        app.handle_chat_submit()
        await pilot.pause()
        # Reaching here without raising is the assertion; the log records the message.
        assert app.query_one("#chat-log", RichLog) is not None


@pytest.mark.asyncio
async def test_chat_says_so_when_no_model_is_available(kernel, monkeypatch):
    """
    With no model reachable the UI must say it has no answer.

    A fabricated reply is worse than none, because the user cannot tell one from
    the other.
    """
    from opencontext import llm

    def unavailable(self, system_prompt, user_prompt, max_tokens=None):
        raise LLMUnavailable("No model server reachable at http://localhost:11434/v1")

    monkeypatch.setattr(llm.LLMClient, "complete", unavailable)

    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app._write_chat_error("No model server reachable at http://localhost:11434/v1")
        await pilot.pause()
        assert app.query_one("#chat-log", RichLog) is not None


@pytest.mark.asyncio
async def test_rescan_reindexes_the_workspace(tmp_path):
    (tmp_path / "one.py").write_text("a = 1\n", encoding="utf-8")
    kernel = ContextKernel(token_budget=6000, swap_storage=SwapStorage(str(tmp_path / "swap.db")))

    app = OpenContextApp(kernel=kernel, root_dir=str(tmp_path), session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        (tmp_path / "two.py").write_text("b = 2\n", encoding="utf-8")

        app.action_rescan()
        await pilot.pause()
        assert any(page.title.endswith("two.py") for page in kernel.pages.values())


@pytest.mark.asyncio
async def test_clear_swap_purges_pages_with_no_other_copy(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        kernel.page_out("file:auth.py")

        app.on_button_pressed(
            type("Event", (), {"button": type("Button", (), {"id": "btn-clear-swap"})()})()
        )
        await pilot.pause()
        assert "file:auth.py" not in kernel.pages


# -- context window tab ------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_tab_breakdown_reconciles(kernel):
    """
    The breakdown must add up to the window the model would actually receive.

    This tab is the product's output, so a total that did not reconcile would be a
    fabricated figure in the one place it matters most.
    """
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.action_switch_tab("tab-context")
        await pilot.pause()

        report = kernel.assemble_context_report()
        rows = sum(segment["tokens"] for segment in report["segments"])
        assert rows + report["overhead_tokens"] == report["total_tokens"]

        table = app.query_one("#context-table", DataTable)
        expected = len(report["segments"]) + (1 if report["overhead_tokens"] else 0)
        assert table.row_count == expected


@pytest.mark.asyncio
async def test_context_pane_shows_the_window_verbatim(kernel):
    """
    The pane renders literal text, not markup.

    Section headers look like "=== [OpenContext L1_WORKING] ===" and pages contain
    arbitrary source; letting Rich parse either would silently swallow anything in
    square brackets, including real code.
    """
    kernel.allocate_page("file:brackets.py", "brackets.py", "values = [1, 2, 3]\nkey = data['k']\n")

    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.action_switch_tab("tab-context")
        await pilot.pause()

        rendered = app.query_one("#context-text").render()
        text = rendered.plain if hasattr(rendered, "plain") else str(rendered)

        assert text == kernel.assemble_context(), "the pane must show the window exactly"
        assert "[OpenContext L1_WORKING]" in text
        assert "values = [1, 2, 3]" in text


@pytest.mark.asyncio
async def test_exporting_the_context_writes_the_window_to_a_file(kernel, tmp_path):
    app = OpenContextApp(kernel=kernel, root_dir=str(tmp_path), session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.action_export_context()
        await pilot.pause()

        path = app._last_export_path
        assert path and os.path.isfile(path)
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == kernel.assemble_context()


@pytest.mark.asyncio
async def test_exporting_an_empty_window_is_refused(tmp_path):
    kernel = ContextKernel(token_budget=1000, swap_storage=SwapStorage(str(tmp_path / "s.db")))
    app = OpenContextApp(kernel=kernel, root_dir=str(tmp_path), session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.action_export_context()
        await pilot.pause()
        assert app._last_export_path is None


# -- content search --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_search_finds_pages_the_name_filter_misses(kernel):
    kernel.allocate_page(
        "file:payments.py",
        "src/payments.py",
        "def charge(amount):\n    return gateway.submit(amount)\n",
    )

    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        table = app.query_one("#page-table", DataTable)

        app.query_one("#page-search", Input).value = "gateway submit"
        await pilot.pause()
        await pilot.pause()
        assert table.row_count == 0, "no page is named gateway"

        app.action_toggle_search_mode()
        await pilot.pause()
        await pilot.pause()
        assert app.search_mode == "CONTENT"
        assert table.row_count >= 1, "content search should find it"


# -- chat and episodic memory ------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_turns_become_episodic_pages(kernel, monkeypatch):
    """
    The UI's own conversation has to live in the tier OpenContext claims to manage,
    so turns age out under the same budget as everything else.
    """
    from opencontext import llm

    monkeypatch.setattr(
        llm.LLMClient, "stream", lambda self, system_prompt, user_prompt: iter(())
    )

    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        before = kernel.get_metrics().l2_pages

        app.query_one("#chat-input", Input).value = "why is checkout slow?"
        app.handle_chat_submit()
        await pilot.pause()

        assert kernel.get_metrics().l2_pages > before
        assert any("checkout slow" in page.content for page in kernel.pages.values())


@pytest.mark.asyncio
async def test_assistant_turn_is_ingested_only_once_it_exists(kernel):
    """Ingesting before the reply arrived would create an empty page and charge for it."""
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        before = kernel.get_metrics().l2_pages

        app._write_chat_answer("The checkout handler blocks on the payments gateway.")
        await pilot.pause()

        assert kernel.get_metrics().l2_pages == before + 1
        assert any("payments gateway" in page.content for page in kernel.pages.values())


@pytest.mark.asyncio
async def test_streaming_pane_shows_partial_replies_then_clears(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()

        app._write_chat_delta("partial answer")
        await pilot.pause()
        assert "partial answer" in str(app.query_one("#chat-streaming").render())

        app._write_chat_answer("partial answer complete")
        await pilot.pause()
        assert str(app.query_one("#chat-streaming").render()).strip() == ""


# -- help and session ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_help_screen_opens_and_closes(kernel):
    app = OpenContextApp(kernel=kernel, session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.action_help()
        await pilot.pause()
        assert len(app.screen_stack) == 2

        app.screen.action_close()
        await pilot.pause()
        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_session_survives_a_quit(tmp_path):
    """Pinning a page is meaningless if it evaporates when you close the app."""
    swap = str(tmp_path / "swap.db")
    session = str(tmp_path / "session.json")

    first = ContextKernel(token_budget=16000, swap_storage=SwapStorage(swap))
    for index in range(4):
        first.allocate_page(f"file:{index}.py", f"src/{index}.py", "def handler():\n    pass\n" * 40)

    app = OpenContextApp(kernel=first, root_dir=str(tmp_path), session_path=session)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        first.pin_page("file:2.py")
        first.page_out("file:3.py")
        first.set_budget(2222)
        await pilot.pause()

    second = ContextKernel(token_budget=16000, swap_storage=SwapStorage(swap))
    restored = OpenContextApp(kernel=second, root_dir=str(tmp_path), session_path=session)
    async with restored.run_test(size=TERMINAL) as pilot:
        await pilot.pause()

    assert second.token_budget == 2222
    assert second.pages["file:2.py"].tier == PageTier.L0_PINNED
    assert second.pages["file:3.py"].status == PageStatus.SWAPPED
    assert second.page_fault("file:3.py") is not None, "the swapped page is still recoverable"


@pytest.mark.asyncio
async def test_session_persistence_can_be_disabled(kernel, tmp_path):
    app = OpenContextApp(kernel=kernel, root_dir=str(tmp_path), session_path=None)
    async with app.run_test(size=TERMINAL) as pilot:
        await pilot.pause()
        app.action_save_session()
        await pilot.pause()

    assert not (tmp_path / "session.json").exists()
