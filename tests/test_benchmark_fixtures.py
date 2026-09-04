"""
The published figures must come from the benchmark, and the benchmark must not
read the working tree.

Both of these were broken once. `bench` outlined `opencontext/core/kernel.py` and
walked the installed package for its session workload, so editing OpenContext moved
the numbers printed in the README — and they drifted apart silently, because
nothing compared them.
"""

import os
import re

from opencontext import benchmark

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_reduction_figures_are_identical_across_runs():
    """
    The whole point of freezing the inputs. If a workload ever reads something
    mutable — a live source file, a timestamp, a directory listing of the package
    — this is what catches it.
    """
    first = {r.name: (r.before_tokens, r.after_tokens) for r in benchmark.run_benchmark().results}
    second = {r.name: (r.before_tokens, r.after_tokens) for r in benchmark.run_benchmark().results}
    assert first == second


def test_benchmark_inputs_live_under_fixtures():
    """
    The corpus is committed, non-empty, and stored inert: `.py.txt` rather than
    `.py`, so pytest does not collect it and nothing imports it by accident.
    """
    corpus = benchmark._corpus_files()
    assert len(corpus) >= 8
    assert all(name.endswith(".py") and content.strip() for name, content in corpus)
    assert benchmark._sample_source_file().strip()


def test_the_session_workload_actually_exceeds_the_budget():
    """
    A session that fits in the budget measures nothing: no eviction runs, and the
    reduction figure would describe compaction alone. The corpus has to be
    comfortably larger than the budget it is held to, which is the situation
    OpenContext exists for.
    """
    session = next(
        r for r in benchmark.run_benchmark().results if r.name.startswith("Mixed agent session")
    )
    assert session.after_tokens <= 16_000, "the session workload is held to a 16k budget"
    assert session.before_tokens > 2 * 16_000, "the corpus must comfortably exceed that budget"


def test_readme_table_matches_what_the_command_prints():
    """
    Pins every published reduction figure to the code that produces it. The README
    once carried a benchmark table with nothing behind it at all; this makes the
    weaker version of that failure — a table that was true and quietly went stale —
    impossible to merge.
    """
    with open(os.path.join(REPO_ROOT, "README.md"), "r", encoding="utf-8") as handle:
        readme = handle.read()

    published = {}
    for line in readme.splitlines():
        match = re.match(
            r"^\| (.+?) \| ([\d,]+) tok \| ([\d,]+) tok \| \*\*([\d.]+)%\*\* \|$", line.strip()
        )
        if match:
            name, before, after, pct = match.groups()
            published[name] = (int(before.replace(",", "")), int(after.replace(",", "")), pct)

    measured = {
        r.name: (r.before_tokens, r.after_tokens, f"{r.reduction_pct:.1f}")
        for r in benchmark.run_benchmark().results
    }

    assert published, "no benchmark table found in README.md"
    assert published == measured
