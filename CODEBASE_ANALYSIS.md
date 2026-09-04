> **Historical document.** This is the audit of OpenContext **0.1.0** that preceded
> the 0.2.0 rewrite. Every defect described below has since been fixed and pinned
> by a regression test; see `CHANGELOG.md` for what changed and where. It is kept
> because the reasoning explains why several design decisions in 0.2.0 are what
> they are.

# OpenContext — Codebase Analysis

Analysed 2026-09-04. ~2,900 lines of Python + an 833-line single-file dashboard, no git history, 19 tests passing.

## 1. What it is, and how it actually works

OpenContext presents itself as a "virtual memory kernel" for LLM context windows. The metaphor maps onto real code:

| Concept | Reality |
|---|---|
| Page | `ContextPage` pydantic model (`core/types.py:28`) — id, title, tier, status, content, token_count |
| RAM tiers | `PageTier` enum: L0_PINNED / L1_WORKING / L2_EPISODIC / L3_SWAP |
| MMU | `ContextKernel` (`core/kernel.py`) — allocate, LRU evict, page-fault, metrics |
| Swap disk | `SwapStorage` (`storage/swap.py`) — one SQLite table, `.opencontext/swap.db` |
| Token accounting | `len(text) // 4` (`kernel.py:34`) |
| Compaction | Regex/line heuristics over tracebacks and code (`core/compactor.py`) |

The actual mechanism is honest and quite small: pages live in a dict; `_enforce_budget()` (`kernel.py:121`) sorts evictable pages by (tier, last_accessed) and calls `page_out()` until the estimated token total fits the budget; `page_out()` writes the page to SQLite and sets a one-line tombstone string; `assemble_context()` (`kernel.py:228`) emits full content for ACTIVE pages and tombstones for SWAPPED ones. `touch_or_fault()` (`kernel.py:211`) does a substring scan of the incoming prompt for swapped page ids/titles and rehydrates matches. That core loop works and is covered by tests.

Four front-ends sit on top: a Textual TUI (`interfaces/interactive_tui.py`, 1,552 lines — over half the codebase), a FastAPI app serving both a dashboard and an OpenAI-compatible proxy (`interfaces/proxy.py`), a hand-rolled MCP stdio server (`interfaces/mcp_server.py`), and a rich-based HUD (`interfaces/tui.py`) that nothing reaches.

**Architecturally the important fact: there is no daemon.** `cmd_status`, `cmd_chat` and `cmd_tui` each call `get_or_create_workspace_kernel()` (`cli.py:40`), which builds a fresh in-memory kernel and rescans the whole tree. Nothing survives a process except swap rows. Every README reference to a "KERNEL DAEMON" or "background daemon" describes something that doesn't exist; `opencontext serve` is a normal foreground uvicorn process.

## 2. Correctness findings, most severe first

### 2.1 Metrics mix persistent and ephemeral state; swap rows are never reclaimed

`get_metrics()` (`kernel.py:254`) reads `swapped_tokens` from SQLite (`SUM(token_count)` over every row ever written) but derives `l3_pages` from the in-process dict. `page_fault()` rehydrates without deleting the swap row, so the number only ever grows.

Verified against the real `.opencontext/swap.db` in this repo:

- `opencontext status` reports **52,726 swapped tokens / 16 pages**
- the DB actually holds **36 rows**, including `file:src\legacy_engine.py` for a file that doesn't exist
- a fresh `ContextKernel()` in an empty process reports `l3_pages=0, swapped_tokens=52726`

Every "Total Tokens Saved" figure the dashboard, TUI and CLI display is a cross-run high-water mark, not current state.

### 2.2 Focus-mode code folding never exits — it delivers zero compaction on classes

`CodeOutlineCompactor.compact_code` (`compactor.py:151`) exits focus on `indent <= focus_indent and not line.startswith(" ")`. Every sibling method of a class *does* start with a space, so once focus opens inside a class it never closes; the whole remainder is emitted verbatim, `comp_tokens >= orig_tokens` trips the guard at line 166, and the function returns the input unchanged.

Confirmed: `compact_code(..., focus_symbol="delete_user")` on a 25-line class returns the original, 104 → 104 tokens. This is the path `ContextPager.ingest_file` uses for files over 150 lines (`pager.py:52`) and the one the MCP `context_ingest_file` tool advertises as "AST outline compaction."

`test_compactor.py:79` passes anyway: its `"folded"` assertion is satisfied by a `def __init__` line *before* the focus point, and its only size check is a bare `comp_tok < orig_tok`.

### 2.3 The streaming path always returns the fake response

`/v1/chat/completions` (`proxy.py:194`) forwards `stream: true` upstream, then calls `upstream_resp.json()` on what is an SSE byte stream. That raises, the bare `except Exception: pass` at line 235 swallows it, and the client receives the canned `"[OpenContext Proxy Active]"` completion. Cursor and Continue.dev stream by default, so the advertised drop-in integration returns a placeholder for exactly the clients the README targets. Two adjacent problems on the same handler:

- `payload = req.model_dump()` (line 224) is built from a closed pydantic model, so `tools`, `tool_choice`, `response_format` and every other field are silently dropped — tool-calling clients break.
- Conversation history is discarded: only `latest_user_prompt` plus the assembled context is forwarded (lines 213-216). Multi-turn chat loses prior assistant turns entirely.

### 2.4 Loose traceback detection routes ordinary prose through the compactor

`TracebackCompactor.is_traceback` (`compactor.py:15`) matches on the bare substrings `"Error:"`, `"at "` and `"FAILED"`. `allocate_page` defaults to `auto_compact=True`, and `proxy.py:202` routes any message containing `"Traceback"` or `"Error:"` into `ingest_traceback`.

Confirmed: a 29-line document containing the word "at" compacts 684 → 125 tokens, discarding 82% of the text, and the loss is booked as `total_tokens_saved`. A user message beginning "Error: I can't work out why…" takes this path. The `comp_tokens >= orig_tokens` guard protects short inputs only; longer text is silently truncated.

Scope note: `workspace.scan_and_ingest`, `pin_instruction`, `ingest_file` and `ingest_conversation_turn` all pass `auto_compact=False`, so ingested source files are not affected today — but the default is the wrong way round.

### 2.5 Page ids collide and are unstable across processes

`ingest_traceback` and `ingest_conversation_turn` (`pager.py:67`, `pager.py:78`) build ids as `abs(hash(text[:100])) % 100000`. Two consequences: `PYTHONHASHSEED` randomisation means the same error gets a different id every process, so a persisted swap row can never be found again; and the `% 100000` fold means two unrelated errors can produce the same id, at which point `allocate_page` overwrites the first page's content in place (`kernel.py:80-87`).

Separately, the two ingest paths disagree on id format: `WorkspaceScanner` writes `file:{rel_path}` with forward slashes (`workspace.py:59`), while `ContextPager.ingest_file` writes `file:{os.path.normpath(...)}` with backslashes on Windows (`pager.py:48`). The same file scanned and then ingested via MCP becomes two pages. The stale `file:src\legacy_engine.py` row in the live swap DB is this bug's fingerprint. `test_workspace.py:29` asserts `"file:src/app.py" in pages or f"file:src{os.sep}app.py" in pages`, which accepts either and hides it.

### 2.6 L0 pinning is filename-blind and unbounded

`workspace.py:64` pins on bare filename: `file.lower() in {"readme.md", "pyproject.toml", "package.json"}`. In a monorepo every `package.json` at every depth lands in L0, where nothing can evict it. `_enforce_budget` handles the resulting over-budget state by exhausting its candidate list and returning silently — no event, no warning, no `LEAK_WARNING`. The budget is simply exceeded with nothing in the telemetry saying so.

### 2.7 Swap-out is a context abstraction, not a memory one

`page_out` (`kernel.py:141`) writes to SQLite but never clears `page.content`. Token accounting is correct — `_get_current_working_tokens` charges the tombstone estimate and `assemble_context` emits only the tombstone — but process RSS never drops. A scan of a large tree holds the entire tree in RAM regardless of budget, and `page_fault` re-reads from SQLite content that was in memory the whole time. Worth stating plainly in the docs: the budget governs what the model sees, not what the process holds.

### 2.8 MCP server bugs (the one module with no tests)

- `execute_tool` for `context_ingest_file` (`mcp_server.py:149-155`) sets `"status"` twice in one dict literal; `"ingested"` is overwritten by `page.status.value`. Callers never see the ingest confirmation.
- `run_stdio_loop` (`mcp_server.py:211`) replies `-32601 Method not found` to anything it doesn't recognise, including `notifications/initialized` — which is a notification with no `id`. Emitting a response (with `"id": null`) to a notification violates JSON-RPC 2.0 and some clients will reject the handshake. `ping`, `resources/list` and `prompts/list` are likewise unhandled.
- The error handler at line 220 returns `"id": null` for every failure, so a client can never correlate an error to its request.

### 2.9 Import weight and coupling

`cli.py` imports `run_proxy_server` and `run_mcp_server` at module top (lines 36-37) and `run_interactive_tui` at line 139, and `interfaces/__init__.py` eagerly imports all four interfaces. `opencontext mcp` — a stdio server that needs neither — therefore loads FastAPI, uvicorn, httpx and Textual before serving a byte, and an import error in any interface breaks all of them.

### 2.10 Smaller items

- `demo.py:15` imports `cmd_demo` from `opencontext.cli`, which doesn't exist. `python demo.py` raises `ImportError`. The file is dead.
- Half the state model is declared and never assigned: `PageTier.L3_SWAP`, `PageStatus.COMPACTED` and `PageStatus.EVICTED` appear only in `tui.py`'s render maps (lines 119, 125, 126). Swapping changes `status` but not `tier`; compaction leaves status `ACTIVE`; `delete_page` pops the page rather than marking it EVICTED.
- `/api/pages` (`proxy.py:78`) returns full file content for every scanned page, unauthenticated. The default bind is `127.0.0.1`, which contains it — but `--host 0.0.0.0` exposes the whole indexed tree, and there is no CORS policy or auth anywhere.
- `api_chat` (`proxy.py:157`) posts the assembled workspace context to `api.openai.com` whenever `OPENAI_API_KEY` is set, with no opt-in prompt. `GEMINI_API_KEY` is read at line 140 and never used.
- Doom-loop detection keys on `content[:80]` of the *compacted* text (`kernel.py:67`), so it only fires when the compactor happens to produce identical prefixes — it is a prefix match, not the semantic signature the docs describe.

## 3. Documentation vs implementation

| README claim | Reality |
|---|---|
| "11 passed" test block | 19 tests pass |
| Version 0.1.0 (badge, pyproject) | `cli.py:51` and `tui.py:51` print "v0.3.0" |
| "Launch the VMM Daemon" / "KERNEL DAEMON" | No daemon; every command builds a fresh kernel and rescans |
| `opencontext chat "How does auth work?"` | `cmd_chat` (`cli.py:112`) never calls an LLM — it prints "Workspace indexed and required files loaded" |
| `opencontext top` = "htop-style HUD" | `cli.py:179` maps `top` to the Textual app; the rich HUD `ContextTopUI` (214 lines) is unreachable from the CLI |
| Benchmarks table (90.2% / 88.0% / 99.1% / 80.5%) | No code produces these. `run_micro_benchmark` (`interactive_tui.py:1500`) measures page-fault *latency* only, and prints "0% data corruption, sub-millisecond retrieval confirmed" as a hardcoded string |
| MCP `context_ingest_file` "with outline folding" | Outline folding is a no-op for class methods (§2.2) |
| Proxy as drop-in for Cursor / Continue.dev | Streaming clients always get the placeholder response (§2.3) |

## 4. Test coverage

19 tests, ~525 lines, all passing in ~3s. Coverage is shaped by module, and the shape explains where the bugs are:

- **Well covered:** kernel allocate/evict/page-fault/pin/budget (`test_kernel.py`, `test_kernel_extended.py`), Textual TUI mount and interactions (`test_interactive_tui.py`, the most thorough file).
- **Thin:** compactor (3 tests, two asserting only `comp_tok < orig_tok` — the assertion that let §2.2 through); proxy (3 tests, all on the fallback path, none exercising an upstream, streaming, or `tools`); workspace (1 test, with the `or` that hides §2.5).
- **Zero:** `mcp_server.py`, `storage/swap.py`, `interfaces/tui.py`, `cli.py`. Two confirmed bugs (§2.8) live in the untested MCP module; the swap-accounting bug (§2.1) lives in the untested storage module.

Every test constructs its own `SwapStorage(tmp_path)`, which is good hygiene — but it also means no test ever observes the shared `.opencontext/swap.db` accumulation that §2.1 describes.

Suite hygiene: `pytest-asyncio` emits a config deprecation warning on every run, and the Textual tests leave "Task was destroyed but it is pending" noise at teardown. There is no `pytest.ini` / `[tool.pytest]` config, no `.gitignore`, and `__pycache__`, `.pytest_cache`, `opencontext.egg-info` and `.opencontext/swap.db` are all sitting in the working tree.

## 5. If I were prioritising

1. §2.1 swap accounting and §2.2 focus folding — both make headline features report success while doing nothing.
2. §2.3 streaming — the primary advertised integration is broken for its primary clients.
3. §2.4 / §2.5 — flip the `auto_compact` default, tighten `is_traceback`, replace `hash()` ids with a stable digest, unify the `file:` id scheme on forward slashes.
4. §2.8 plus tests for `mcp_server.py` and `storage/swap.py`.
5. Reconcile the README with the implementation, or implement the daemon it describes.
