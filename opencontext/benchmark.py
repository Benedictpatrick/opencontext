"""
Reproducible benchmark for OpenContext.

Every number OpenContext publishes comes from here. The inputs are committed under
`tests/fixtures/`, so anyone can run `opencontext bench` and get the same figures on
their own machine. Nothing here reads the working tree: an input that changed when
OpenContext itself was edited would make the published numbers unreproducible.

What is measured:

  * traceback / stack-trace reduction — the compactor on real captured traces
  * outline folding — a real source file (frozen snapshot) reduced to its skeleton
  * tombstone reduction — what an evicted file costs in context once swapped
  * mixed session — a realistic sequence of agent activity through the kernel
  * page-fault latency — wall-clock time to restore a page from disk

What is deliberately *not* measured: an end-to-end saving against a specific
model's billing. That depends on the model, the prompt and the caching policy,
so OpenContext does not claim it.
"""

from __future__ import annotations

import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from opencontext.core.compactor import CodeOutlineCompactor, TracebackCompactor
from opencontext.core.kernel import ContextKernel
from opencontext.core.pager import ContextPager
from opencontext.core.scenarios import FAILING_TOOL_CALL
from opencontext.core.tokens import estimate_tokens
from opencontext.core.types import PageTier
from opencontext.storage.swap import SwapStorage

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")


@dataclass
class BenchmarkResult:
    """One measured row of the benchmark table."""

    name: str
    before_tokens: int
    after_tokens: int
    detail: str = ""

    @property
    def saved_tokens(self) -> int:
        return self.before_tokens - self.after_tokens

    @property
    def reduction_pct(self) -> float:
        if self.before_tokens <= 0:
            return 0.0
        return round(self.saved_tokens / self.before_tokens * 100, 1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "saved_tokens": self.saved_tokens,
            "reduction_pct": self.reduction_pct,
            "detail": self.detail,
        }


@dataclass
class BenchmarkReport:
    """Full benchmark output."""

    results: List[BenchmarkResult] = field(default_factory=list)
    latency: Dict[str, float] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [result.as_dict() for result in self.results],
            "latency": self.latency,
            "environment": self.environment,
        }

    def to_markdown(self) -> str:
        """Render the table exactly as it appears in the README."""
        lines = [
            "| Workload | Before | After | Reduction |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for result in self.results:
            lines.append(
                f"| {result.name} | {result.before_tokens:,} tok | "
                f"{result.after_tokens:,} tok | **{result.reduction_pct}%** |"
            )
        return "\n".join(lines)


def _read_fixture(name: str) -> str:
    path = os.path.join(FIXTURE_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


CORPUS_DIR = os.path.join(FIXTURE_DIR, "corpus")


def _sample_source_file() -> str:
    """
    A real source file for the outline and tombstone benchmarks.

    This is a frozen snapshot of the kernel's own source, committed as a fixture
    rather than read from `opencontext/core/kernel.py` at run time. Reading the live
    file made every published figure a function of the working tree: editing the
    kernel silently moved the numbers in the README, which is the one thing a
    benchmark must not do. The snapshot is real production code, and it is the
    same bytes on every machine and every checkout.
    """
    return _read_fixture("sample_source.py.txt")


def _corpus_files() -> List[Tuple[str, str]]:
    """
    The frozen source corpus behind the mixed-session benchmark, as
    `(filename, content)` pairs sorted by name.

    Committed for the same reason as the snapshot above: the session benchmark
    used to walk the installed package, so adding a module to OpenContext changed
    the published saving. Fixtures are stored with a `.txt` suffix so they are
    inert — not importable, not collected by pytest, not linted as project code.
    """
    files: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(CORPUS_DIR)):
        if not name.endswith(".py.txt"):
            continue
        with open(os.path.join(CORPUS_DIR, name), "r", encoding="utf-8") as handle:
            files.append((name[: -len(".txt")], handle.read()))
    return files


def bench_traceback() -> BenchmarkResult:
    raw = _read_fixture("python_traceback.txt")
    _, before, after = TracebackCompactor.compact(raw)
    frames = raw.count('File "')
    return BenchmarkResult(
        "Python traceback (FastAPI, 24 frames)",
        before,
        after,
        f"{frames} frames in, user frames and root cause kept",
    )


def bench_node_stack() -> BenchmarkResult:
    raw = _read_fixture("node_stacktrace.txt")
    _, before, after = TracebackCompactor.compact(raw)
    return BenchmarkResult(
        "Node.js stack trace (Next.js, 17 frames)",
        before,
        after,
        "node_modules and node internals dropped",
    )


def bench_outline() -> BenchmarkResult:
    source = _sample_source_file()
    _, before, after = CodeOutlineCompactor.compact_code("sample_source.py", source)
    return BenchmarkResult(
        f"Source file folded to outline ({len(source.splitlines())} lines)",
        before,
        after,
        "signatures and docstrings kept, bodies folded",
    )


def bench_tombstone() -> BenchmarkResult:
    """What a file costs in context once it has been swapped out."""
    source = _sample_source_file()
    workdir = tempfile.mkdtemp(prefix="opencontext-bench-")
    try:
        kernel = ContextKernel(
            token_budget=100_000, swap_storage=SwapStorage(os.path.join(workdir, "swap.db"))
        )
        page = kernel.allocate_page("file:sample_source.py", "sample_source.py", source)
        before = page.token_count
        kernel.page_out("file:sample_source.py")
        after = estimate_tokens(kernel.pages["file:sample_source.py"].tombstone or "")
        return BenchmarkResult(
            "Inactive file swapped to disk (tombstone in context)",
            before,
            after,
            "full content recoverable on reference",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def bench_mixed_session() -> BenchmarkResult:
    """
    A realistic agent session through the kernel.

    Deliberately mixed rather than four copies of one failure: source files the
    agent opened, a retry loop, conversation turns and pinned instructions. A
    benchmark built only from repeated identical tracebacks would measure the
    compactor several times over and call it a session.
    """
    workdir = tempfile.mkdtemp(prefix="opencontext-session-")
    try:
        kernel = ContextKernel(
            token_budget=16_000, swap_storage=SwapStorage(os.path.join(workdir, "swap.db"))
        )
        pager = ContextPager(kernel)

        raw_tokens = 0

        instructions = (
            "Always write type-annotated Python. Never edit files under migrations/. "
            "Run the test suite before reporting completion."
        )
        raw_tokens += estimate_tokens(instructions)
        pager.pin_instruction("house_rules", "House rules", instructions)

        # Files the agent opened while working, from the frozen corpus. A real
        # session ranges over far more source than fits the budget — that is the
        # condition OpenContext exists for, so the fixture has to reach it rather
        # than stopping just above the line.
        for filename, content in _corpus_files():
            raw_tokens += estimate_tokens(content)
            pager.ingest_file(os.path.join("src", filename), content=content)

        # A failing tool call retried three times.
        for attempt in range(1, 4):
            body = f"{FAILING_TOOL_CALL}\n# retry {attempt}"
            raw_tokens += estimate_tokens(body)
            pager.ingest_traceback(body, title=f"Tool failure {attempt}")

        # Conversation turns.
        for turn in range(6):
            message = (
                f"Turn {turn}: the checkout endpoint is timing out under load, "
                "walk me through where the request is spending its time."
            )
            raw_tokens += estimate_tokens(message)
            pager.ingest_conversation_turn("user" if turn % 2 == 0 else "assistant", message)

        after = kernel.get_metrics().working_tokens
        return BenchmarkResult(
            "Mixed agent session (files, retries, turns, rules)",
            raw_tokens,
            after,
            f"held to a {kernel.token_budget:,} token budget",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def bench_page_fault_latency(pages: int = 50, samples: int = 25) -> Dict[str, float]:
    """Measure the wall-clock cost of restoring a page from disk."""
    workdir = tempfile.mkdtemp(prefix="opencontext-latency-")
    try:
        kernel = ContextKernel(
            token_budget=1_000_000, swap_storage=SwapStorage(os.path.join(workdir, "swap.db"))
        )
        body = "def handler(request):\n    return process(request)\n" * 40
        page_ids = []
        for index in range(pages):
            page_id = f"file:module_{index}.py"
            page_ids.append(page_id)
            kernel.allocate_page(page_id, f"module_{index}.py", body)

        swap_start = time.perf_counter()
        swapped = kernel.swap_all_unpinned()
        swap_ms = (time.perf_counter() - swap_start) * 1000

        durations = []
        for page_id in page_ids[:samples]:
            start = time.perf_counter()
            kernel.page_fault(page_id)
            durations.append((time.perf_counter() - start) * 1000)

        durations.sort()
        return {
            "pages_swapped": float(swapped),
            "swap_out_total_ms": round(swap_ms, 3),
            "swap_out_per_page_ms": round(swap_ms / max(1, swapped), 3),
            "page_fault_mean_ms": round(statistics.mean(durations), 3),
            "page_fault_median_ms": round(statistics.median(durations), 3),
            "page_fault_p95_ms": round(durations[int(len(durations) * 0.95) - 1], 3),
            "samples": float(len(durations)),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_benchmark(include_latency: bool = True) -> BenchmarkReport:
    """Run every benchmark and return the measured report."""
    import platform
    import sys

    from opencontext import __version__
    from opencontext.core.tokens import _get_encoder

    report = BenchmarkReport(
        results=[
            bench_traceback(),
            bench_node_stack(),
            bench_outline(),
            bench_tombstone(),
            bench_mixed_session(),
        ],
        environment={
            "opencontext_version": __version__,
            "python": platform.python_version(),
            "platform": platform.system(),
            "tokenizer": "tiktoken cl100k_base" if _get_encoder() else "built-in heuristic",
        },
    )

    if include_latency:
        report.latency = bench_page_fault_latency()

    return report
