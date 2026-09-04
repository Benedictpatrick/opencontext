# ContextOS

**A virtual memory kernel for LLM context windows.**

Keeps the active working set inside a token budget, moves cold pages to local disk behind a one-line tombstone, and pages them back in when the model refers to them again.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

---

## The problem

A coding agent's context window fills with material it is no longer using: files it opened twenty steps ago, the same stack trace retried four times, conversation turns that are no longer relevant. Quality degrades and cost rises, because everything in the window is re-sent on every call.

## What ContextOS does

Four mechanisms, each measurable:

1. **Token budget with LRU eviction.** You set a budget. When the working set exceeds it, the coldest pages are written to a local SQLite file and replaced in-context by a one-line tombstone. Episodic history is given up before working code; pinned instructions are never evicted.
2. **Page faults.** When an incoming prompt names a page that has been swapped out, ContextOS restores it before the request goes to the model. Median 0.21 ms.
3. **Traceback compaction.** Stack traces are reduced to their user frames and root cause. Python, Node.js, Rust, Go and test-runner output are recognised. Framework noise is dropped; the diagnostic content is kept.
4. **Repeat-failure detection.** When the same root cause arrives three times — normalised so timestamps, addresses and retry counters do not defeat the match — ContextOS raises a warning.

```
                    your editor / agent
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │                  ContextOS                    │
    │                                               │
    │   L0  pinned instructions   never evicted     │
    │   L1  working files         evicted 2nd       │
    │   L2  conversation history  evicted 1st       │
    │                    │                          │
    │                    │  swap out ↓  ↑ page in   │
    │            SQLite  ▼                          │
    │            .contextos/swap.db                 │
    └───────────────────────────────────────────────┘
                            │
                            ▼
                  model (local or hosted)
```

---

## Install

```bash
pip install -e ".[all]"
```

The kernel itself depends only on `pydantic`, `rich` and `httpx`. Interfaces are optional extras, so an MCP-only or library-only install does not pull in a web framework:

| Extra | Adds | For |
| :--- | :--- | :--- |
| *(base)* | — | library use, `contextos mcp`, `scan`, `status`, `compact`, `bench` |
| `[tui]` | textual | `contextos tui` |
| `[server]` | fastapi, uvicorn | `contextos serve` |
| `[exact-tokens]` | tiktoken | exact token counting |
| `[all]` | all of the above | everything |

Check the installation at any time:

```bash
contextos doctor
```

---

## Quick start

```bash
contextos scan .        # index the project
contextos status        # what is in context, what is on disk
contextos top           # live monitor
contextos tui           # interactive UI
```

### The interactive UI

`contextos tui` is where the work happens. Six tabs:

| Tab | What it is for |
| :--- | :--- |
| **Overview** | Budget, swap and page-fault telemetry, and the live kernel activity log |
| **Pages** | The page table, with a syntax-highlighted preview. Page in, swap out, pin, outline, or open a file full-screen. Filter by name, or search page contents |
| **Context** | The context window the model would actually receive, shown verbatim, with every token accounted for — and a one-key export to file |
| **Chat** | Questions against the indexed codebase, streamed as the model answers. Turns become episodic pages, so the conversation ages out under the same budget as everything else |
| **Compact** | Paste a stack trace and watch it reduce. Samples for Python, Node and Rust, plus a retry-loop demonstration |
| **Config** | Budget, model server, and the benchmark |

Press `?` for the key list.

**Your arrangement survives closing the app.** Pins, tiers and the budget are saved
on exit and restored on start, so curating a working set is worth doing. A page
whose file has since been deleted is dropped and reported rather than restored
empty. Start with `contextos tui --fresh` to ignore a saved session.

**The Context tab is the point.** ContextOS exists to produce one string — the
window your agent will be charged for — and that tab shows it exactly, alongside a
breakdown of what every page contributes. A swapped page appears as its ~24-token
tombstone next to the thousands it stands in for. The rows reconcile to the
window's measured size, including the section headers no page owns, so the figures
can be trusted rather than merely displayed.

---

## Measured results

Every figure below is produced by `contextos bench`, which runs over fixtures committed in `tests/fixtures/`. The inputs are frozen snapshots, not the working tree, so the reduction figures are the same on any checkout and a test asserts this table matches what the command prints. Run it yourself:

```bash
contextos bench
```

| Workload | Before | After | Reduction |
| :--- | ---: | ---: | ---: |
| Python traceback (FastAPI, 24 frames) | 846 tok | 138 tok | **83.7%** |
| Node.js stack trace (Next.js, 17 frames) | 376 tok | 93 tok | **75.3%** |
| Source file folded to outline (656 lines) | 5,784 tok | 1,049 tok | **81.9%** |
| Inactive file swapped to disk (tombstone in context) | 5,784 tok | 27 tok | **99.5%** |
| Mixed agent session (files, retries, turns, rules) | 48,578 tok | 12,262 tok | **74.8%** |

**Latency**, over 50 pages and 25 sampled faults:

| | |
| :--- | ---: |
| Page fault, median | 0.22 ms |
| Page fault, p95 | 0.35 ms |
| Swap out, per page | 0.08 ms |

Measured on Python 3.11 / Windows with the built-in token heuristic. Unlike the reduction figures, these are wall-clock and will differ with your hardware and disk; the command reports whatever it measures on your machine. The mean is omitted because it is dominated by the first fault, which pays for opening the database.

**What is not claimed.** ContextOS does not publish an end-to-end saving against a specific model's bill. That depends on the model, the prompt and the provider's caching policy, none of which this project controls. The figures above are token counts before and after, on stated inputs.

---

## Integration

### As a library

```python
from contextos import ContextKernel, ContextPager

kernel = ContextKernel(token_budget=16_000)
pager = ContextPager(kernel)

pager.pin_instruction("rules", "House rules", "Always write type-annotated code.")
pager.ingest_file("src/auth.py", focus_symbol="verify_token")
pager.ingest_traceback(crash_output)

result = pager.process_incoming_prompt_verbose("Refactor verify_token in src/auth.py")
result["context"]         # the assembled context window
result["rehydrated"]      # pages restored to answer this prompt
result["context_tokens"]  # what it costs
```

### As an OpenAI-compatible proxy

For Cursor, Continue.dev, LibreChat, or anything speaking the chat completions API.

```bash
contextos serve --port 9090
```

Point the client at `http://localhost:9090/v1`. Requests are routed through the kernel and forwarded upstream.

* **Streaming works.** `stream: true` is proxied as server-sent events.
* **Client fields are preserved.** `tools`, `tool_choice`, `response_format` and anything else pass through untouched.
* **Conversation history is preserved.** The assembled context is prepended as a system message; your turns are not replaced.
* **Failures are reported, never faked.** If the upstream is unreachable you get a 502 explaining why. ContextOS will not synthesise a completion, because you could not tell it from a real one.

Configure the upstream by environment:

```bash
CONTEXTOS_UPSTREAM=http://localhost:11434/v1   # default: local Ollama
CONTEXTOS_UPSTREAM_API_KEY=sk-...              # if the endpoint needs one
CONTEXTOS_MODEL=qwen2.5-coder:7b
```

Anything OpenAI-compatible works, including local runtimes (Ollama, llama.cpp, LM Studio, vLLM) and hosted providers.

### As an MCP server

For Claude Code, Cursor Agent and Antigravity.

```json
{
  "mcpServers": {
    "contextos": {
      "command": "contextos",
      "args": ["mcp"]
    }
  }
}
```

| Tool | Does |
| :--- | :--- |
| `context_inspect` | Current budget usage, page table, warnings |
| `context_page_in` | Restore a swapped page |
| `context_force_swap` | Move a page to disk |
| `context_compact_error` | Reduce a stack trace to user frames and root cause |
| `context_ingest_file` | Read a file into context, optionally outlined |
| `context_outline_file` | Outline a file without ingesting it |
| `context_search` | Find pages by keyword |
| `context_scan_workspace` | Index a directory |

### Web dashboard

`contextos serve` also serves a dashboard at `http://localhost:9090/`: live budget usage, the page table with page-in/swap-out controls, a compaction workbench, and chat against the indexed workspace.

---

## Commands

| Command | Does |
| :--- | :--- |
| `contextos` | Interactive terminal UI (`--fresh` ignores a saved session) |
| `contextos top` | Live read-only monitor (`--once` for a single frame) |
| `contextos status` | One-shot summary |
| `contextos scan [dir]` | Index a project (`--prune-swap` to clear stale swap rows) |
| `contextos chat "..."` | Ask about the codebase, with context assembled by the kernel |
| `contextos compact -f trace.txt` | Compact a trace (also reads stdin) |
| `contextos serve` | Dashboard and proxy |
| `contextos mcp` | MCP stdio server |
| `contextos bench` | Run the benchmark (`--json`, `--markdown`) |
| `contextos doctor` | Check installation and configuration |

---

## Security

The dashboard and page APIs expose information about the indexed source tree.

* The default bind is `127.0.0.1`. Binding elsewhere without a key prints a warning.
* **File contents are withheld by default.** `/api/pages` returns metadata and a short preview. Set `CONTEXTOS_EXPOSE_CONTENT=1` to serve full contents.
* Set `CONTEXTOS_API_KEY` to require `Authorization: Bearer <key>` or `X-API-Key`. `/health` stays open for probes.
* A server-configured upstream key takes precedence over one presented by a client, so the operator controls which credential leaves the machine.
* Nothing is sent anywhere unless you configure an upstream. With a local model, no content leaves the machine.

---

## Token counting

The default estimator is a deterministic heuristic, so budgets are reproducible across machines. Measured against `tiktoken` (cl100k_base) over this repository's source:

| Estimator | Mean error | Worst case |
| :--- | ---: | ---: |
| characters / 4 | 10.8% | 32.2% |
| ContextOS heuristic | 7.1% | 21.1% |

`tests/test_tokens.py` re-runs that measurement and fails if the estimator drifts outside the documented tolerance.

For exact counts:

```bash
pip install "contextos[exact-tokens]"
export CONTEXTOS_TOKENIZER=tiktoken
```

---

## Testing

```bash
python -m pytest
```

The suite covers the kernel, compactors, storage, pager, workspace scanner, MCP
server, proxy (including streaming), CLI, benchmark, and TUI behaviour and layout.

The layout tests render the terminal UI at 80×24, 100×30, 120×40 and 160×50 and assert that card titles are not truncated, that the HUD cards share one height, and that the page table stays usable — so a CSS change cannot quietly break the UI at small terminal sizes.

---

## Known limitations

Stated plainly, because a tool that reports its own limits is easier to trust than one that does not.

* **No daemon.** Each command builds its own kernel and re-indexes. The interactive UI persists its arrangement — pins, tiers and budget — to `.contextos/session.json` and restores it on start, but there is no background process: two ContextOS commands running at once do not share live state.
* **Token counts are estimates by default.** See the accuracy table above, or opt into exact counting.
* **Search is lexical, not semantic.** `context_search`, the Pages tab's content search and the chat ranking all score by keyword overlap. There are no embeddings.
* **Page-fault detection is name-based.** A prompt must mention a page's id, title or base name for it to be restored automatically. Names shorter than four characters are ignored, because they match almost any sentence.
* **Outline folding is exact for Python only.** Python is parsed with `ast`. Other languages use an indentation scanner, which is approximate.
* **`PageTier.L3_SWAP`, `PageStatus.COMPACTED` and `PageStatus.EVICTED` are reserved.** Swapping changes a page's `status`, not its `tier`, so it returns to the tier it came from; compaction is recorded on `ContextPage.compacted`. The enum members are retained for API compatibility.
* **Long conversations are kept as a digest, not turn by turn.** Once old turns exceed a share of the budget they are coalesced into one page, which then swaps to disk like any other. Nothing is lost — referencing it pages the whole history back — but individual turns stop being separately addressable at that point. This is what stops a long chat from evicting the code you are working on.
* **The swap database can outlive a run.** Rows from an earlier session remain until pruned. Live and on-disk figures are reported separately, and `contextos scan --prune-swap` clears them.

---

## License

MIT.
