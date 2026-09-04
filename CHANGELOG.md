# Changelog

## 0.2.1

The terminal UI does the job it was built for, rather than only reporting on it.

### The context window is now visible and extractable

ContextOS exists to produce one thing — the context window an agent is charged for
— and there was no way to see it or get it out. A new **Context** tab shows the
assembled window verbatim, with a per-page breakdown of every token in it.

The breakdown reconciles to the window's measured size rather than summing
`page.token_count`, which would not match: a tombstone costs a fraction of the page
it replaces, and section headers are real tokens no page owns. Header overhead is
its own row, so the figures add up exactly.
→ `test_kernel.py::test_context_report_reconciles_to_the_measured_window`

The pane renders literal text, not Rich markup. Headers look like
`=== [ContextOS L1_WORKING] ===` and pages hold arbitrary source, so markup parsing
silently swallowed anything in square brackets — including real code.
→ `test_interactive_tui.py::test_context_pane_shows_the_window_verbatim`

`w`, or the button, writes the window to a file and reports the path.
→ `test_interactive_tui.py::test_exporting_the_context_writes_the_window_to_a_file`

### The arrangement survives quitting

Pins, tiers and the budget are saved on exit and restored on start. Previously a
pin lasted until you closed the app, which made curating a working set pointless.

Content is recovered from wherever it actually lives: swapped pages from `swap.db`,
file-backed pages re-read from disk (so edits since the save are picked up), and
everything else — pinned instructions, ingested traces, conversation turns — inlined
into the session file. A page whose source has vanished is dropped and reported,
never restored empty.
→ `test_session.py`, `test_interactive_tui.py::test_session_survives_a_quit`

Restoring re-runs eviction, since files may have grown since the save.
`contextos tui --fresh` ignores a saved session.

### Chat does real work

- Replies stream in as the model produces them, instead of the pane sitting frozen
  until the whole answer lands.
- Conversation turns become episodic pages, so the tier ContextOS claims to manage
  now actually holds the UI's own conversation and ages out under the same budget.
  The assistant turn is ingested only once the reply exists — ingesting earlier
  would have created an empty page and charged the budget for it.
  → `test_interactive_tui.py::test_chat_turns_become_episodic_pages`

### Other

- **Content search.** The Pages filter toggles between matching names and searching
  page contents, so you can find a page by what is in it.
  → `test_interactive_tui.py::test_content_search_finds_pages_the_name_filter_misses`
- **Help.** `?` opens a key reference.
- **Automatic eviction no longer makes the window bigger.** A page smaller than its
  own tombstone cost more evicted than resident — a 6-token conversation turn was
  being swapped for a 24-token marker. Automatic eviction now skips those; an
  explicit swap still honours the caller.
  → `test_kernel.py::test_automatic_eviction_skips_pages_smaller_than_their_tombstone`
- **A long conversation no longer starves the working set.** Because each turn is
  smaller than its own tombstone, no turn was ever an eviction candidate — so the
  only pages the kernel could evict were the source files, and a long chat emptied
  the working set to hold history. Exactly backwards for a tool whose job is
  keeping the relevant code in context. Old turns are now coalesced into a single
  page, paying one tombstone for the whole run instead of one per turn, which makes
  the history evictable on the ordinary path and pages back in whole when
  referenced. Over 600 turns against a 16k budget, with 4 of 8 files resident after
  the initial scan, the conversation now costs at most one further file; the 0.2.0
  build was measured leaving 1.
  → `test_kernel.py::test_a_long_conversation_does_not_starve_the_working_set`,
  `test_kernel.py::test_conversation_pressure_at_a_realistic_budget`,
  `test_kernel.py::test_a_coalesced_history_can_be_read_back`
- **A restored session no longer overwrites its own history.** Digest ids came from
  an instance counter, and restore builds a fresh kernel around pages that already
  contain `episodic:digest:0` — so the next coalesce reissued that id, replaced the
  restored history in place, and overwrote its swap row with it. The conversation
  was lost with nothing logged. Ids are now derived from the live page table.
  → `test_kernel.py::test_a_restored_session_does_not_overwrite_its_own_history`
- **A page fault is no longer undone by the budget it triggers.** Restoring a page
  large relative to the budget re-ran eviction, which could swap that very page
  straight back out; the caller got a page with empty content and a success return.
  The requested page is exempt from eviction for that call, and the kernel reports
  being over budget instead of silently returning nothing.
  → `test_kernel.py::test_a_page_fault_is_not_undone_by_the_budget_it_triggers`
- The page cursor follows the page you acted on, so a swap no longer leaves the
  next keypress pointed at a different row.

### The published benchmark is reproducible

`contextos bench` claimed its inputs were committed fixtures, and two of the five
workloads actually read the working tree — the outline benchmark opened
`contextos/core/kernel.py`, and the session benchmark walked the installed package.
Editing ContextOS therefore moved the numbers printed in the README, and they had
already drifted. Both now read frozen snapshots under `tests/fixtures/`, and a test
parses the README table and asserts it matches what the command prints, so the
weaker version of the 0.1.0 failure — a table that was true and quietly went stale
— cannot be merged.
→ `test_benchmark_fixtures.py`

The republished figures are in the README. The session workload changed shape when
its corpus was frozen, so its saving is now 74.8% over 48,578 tokens.


## 0.2.0

A correctness and honesty pass over 0.1.0. Every defect listed below is now pinned
by a regression test; the test that covers each one is named beside it.

### Correctness

**Swap accounting was a high-water mark.** `get_metrics()` read `swapped_tokens`
from a `SUM` over every row the swap database had ever held, while taking
`l3_pages` from the in-memory page table, and `page_fault()` never deleted a row
after restoring it. A fresh kernel over an existing database reported tens of
thousands of swapped tokens across zero pages. Live figures now describe the
current process and disk figures are reported separately as `swap_disk_tokens` /
`swap_disk_rows`; rehydration removes the row.
→ `test_kernel.py::test_live_swap_metrics_match_the_database`

**Focus-mode folding never exited, so it compacted nothing.** The exit condition
was `not line.startswith(" ")`, which is never true for a sibling method inside a
class. Focus stayed open to the end of the file, the output exceeded the input, and
the guard returned the source unchanged — zero reduction on any class. Python is
now parsed with `ast`; other languages use a corrected indentation scanner.
→ `test_compactor.py::test_focus_expands_one_method_and_folds_its_siblings`

**Streaming clients always received a placeholder.** The proxy forwarded
`stream: true` and then called `.json()` on the SSE body. That always raised, the
bare `except` swallowed it, and the client got a synthesised completion. Streaming
is now a real SSE passthrough. Verified against a live model as well as in tests.
→ `test_proxy.py::test_streaming_returns_server_sent_events`

**Ordinary prose was classified as a traceback and truncated.** `is_traceback`
matched the bare substrings `"Error:"`, `"at "` and `"FAILED"`. A 29-line document
containing the word "at" was cut to 18% of its size and the loss booked as a
saving. Detection now requires a structural marker, and `auto_compact` defaults to
off.
→ `test_compactor.py::test_prose_is_not_detected_as_a_traceback`

**Page ids were unstable and collided.** Ids derived from Python's `hash()` change
every interpreter start, so a persisted swap row could never be found again; the
`% 100000` fold let unrelated content share an id and silently overwrite it. Ids
now come from a SHA-1 digest, built in one place (`core/ids.py`).
→ `test_pager.py::test_content_digest_is_stable_across_processes`,
`test_pager.py::test_error_ids_do_not_collide_for_different_traces`

**Two ingest paths disagreed on id format.** The scanner used forward slashes; the
pager used `os.path.normpath`, producing backslashes on Windows. The same file
ingested by both routes became two pages with two swap rows.
→ `test_pager.py::test_scanner_and_pager_agree_on_the_same_file`

**Pinning was filename-blind and uncapped.** Any `package.json` at any depth was
pinned to a tier that cannot be evicted. Pinning is now root-only and capped at a
share of the budget, counting the candidate file's own cost.
→ `test_workspace.py::test_nested_config_files_are_not_pinned`,
`test_workspace.py::test_pinning_is_capped_by_budget_share`

**Exceeding the budget was silent.** When pinned content alone exceeded the budget,
eviction exhausted its candidates and returned with nothing in the telemetry.
A `BUDGET_EXCEEDED` event and an `over_budget` flag now report it.
→ `test_kernel.py::test_over_budget_is_reported_when_pinned_content_alone_exceeds_it`

**Swapping did not free memory.** `page_out` wrote to disk but kept `page.content`
in memory, so a large scan held the whole tree regardless of budget. Content is now
released once the write is confirmed, and `SwapStorage.store` raises rather than
reporting a silent success — swap holds the only copy.
→ `test_kernel.py::test_swapping_out_frees_memory_and_leaves_a_tombstone`,
`test_storage.py::test_store_raises_rather_than_reporting_a_silent_success`

**MCP protocol violations.** The server replied to `notifications/initialized` —
a notification, which must not be answered — and returned `"id": null` on every
error, so clients could not correlate failures. `context_ingest_file` set
`"status"` twice in one dict, overwriting its own result. All fixed; `ping`,
`resources/list` and `prompts/list` are now handled.
→ `test_mcp_server.py::test_notifications_receive_no_response`,
`test_mcp_server.py::test_errors_echo_the_request_id`

**Repeat-failure detection keyed on a text prefix** of the compacted output, so it
only fired by coincidence. It now keys on a normalised root cause, with addresses,
timestamps and counters stripped, and fires even when compaction did not shrink the
input.
→ `test_kernel.py::test_signature_ignores_volatile_detail`

**Conversation history was discarded** by the proxy, which forwarded only the last
user turn, and client fields (`tools`, `response_format`, …) were dropped by a
closed pydantic model.
→ `test_proxy.py::test_conversation_history_is_preserved`,
`test_proxy.py::test_unknown_fields_are_passed_through`

**Page-fault latency was ~14 ms** because SQLite was reopened, with its PRAGMAs
re-issued, on every operation. Connections are now reused per thread: median fault
latency is 0.21 ms, a 70× improvement.

### Fabricated output removed

Nothing displays a figure it did not measure.

- The retry-loop panel was a hand-written string containing an invented tombstone
  hash (`8f2a1b9c`) the compactor never produced. It now shows real compactor output.
  → `test_interactive_tui.py::test_retry_loop_demo_shows_real_compactor_output`
- Fixed strings `98.4% cache hit ratio`, `Latency: < 500µs`, `~20 tokens (98.5%
  context savings)` and `0% data corruption, sub-millisecond retrieval confirmed`
  are gone. Fault latency is now measured and blank until a fault has occurred.
  → `test_interactive_tui.py::test_hud_contains_no_fabricated_figures`
- The proxy no longer synthesises a completion when the upstream is unreachable; it
  returns 502 with a reason.
  → `test_proxy.py::test_upstream_failure_returns_an_error_not_a_fabricated_reply`
- `contextos chat` and the TUI chat call a real model, or say plainly that none is
  configured.
- The README's benchmark table had no code behind it. `contextos bench` now
  produces every published figure from fixtures committed in `tests/fixtures/`.

### Terminal UI

The interactive UI was unusable at 80×24: card titles truncated mid-word
(`ACTIVE WORKI`, `AUTONOMOUS PA`), the four HUD cards rendered at different heights
so the row closed ragged, tab 1's label was clipped out of the tab bar, and
everything below the HUD was pushed off-screen.

Rebuilt with a fixed HUD height and full-height cards, titles short enough for the
narrowest supported terminal, and a narrow-layout mode that collapses split panes
rather than squeezing them. `tests/test_tui_layout.py` renders the app at four
terminal sizes and asserts card heights match, titles are not truncated, nothing
overflows horizontally, and the page table keeps usable height.

Also fixed: after swapping a page the table re-sorted and the cursor landed on a
different row, so the next keypress acted on the wrong page.

### Other

- `contextos top` launched the Textual app; the rich monitor it advertised was
  unreachable dead code. `top` and `tui` are now distinct.
  → `test_cli_and_bench.py::test_top_and_tui_are_distinct_commands`
- `demo.py` imported a function that did not exist and raised on run. Removed.
- Interface imports are lazy: `contextos mcp` no longer loads FastAPI, uvicorn and
  Textual before serving, and an import error in one interface no longer breaks the
  others.
- `/api/pages` returned full file content for every page, unauthenticated. Content
  is now withheld unless explicitly enabled, and `CONTEXTOS_API_KEY` enables auth.
- New: `contextos bench`, `contextos doctor`, `context_search`,
  `context_outline_file`, `context_scan_workspace`.
- Token estimation moved to `core/tokens.py` with a documented, tested accuracy
  against tiktoken, and opt-in exact counting.
- Dependencies split into extras so a library or MCP-only install does not pull in
  a web framework.
- Version reported consistently as 0.2.0 (0.1.0 shipped a README badge saying
  0.1.0 while the CLI printed 0.3.0).
- Test suite: 19 tests → 220, covering `mcp_server.py`, `storage/swap.py`,
  `cli.py` and `tokens.py`, which previously had none.

## 0.1.0

Initial release. See `CODEBASE_ANALYSIS.md` for an audit of this version.
