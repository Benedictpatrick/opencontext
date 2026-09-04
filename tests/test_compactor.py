"""
Tests for the traceback and code compactors.

Several of these pin specific defects that shipped in 0.1.0, noted where relevant,
so a regression fails here rather than in a user's context window.
"""

from __future__ import annotations

import os

import pytest

from opencontext.core.compactor import CodeOutlineCompactor, TracebackCompactor

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture(name: str) -> str:
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as handle:
        return handle.read()


# -- detection -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The cat sat on the mat.\nWe looked at it for a while.",
        "Error: I can't work out why my login page is slow. Any ideas?",
        "Meet me at 5pm at the cafe on the corner.",
        "\n".join(f"Step {i}: the pipeline pauses at stage {i}." for i in range(40)),
        "The deployment FAILED to impress anyone at the review.",
        "",
        "short",
    ],
)
def test_prose_is_not_detected_as_a_traceback(text):
    """
    Ordinary text must never be classified as an error trace.

    0.1.0 matched the bare substrings "Error:", "at " and "FAILED", so a 29-line
    document containing the word "at" was compacted to 18% of its size and the
    discarded content was reported as a saving.
    """
    assert TracebackCompactor.is_traceback(text) is False


@pytest.mark.parametrize(
    "name,expected_language",
    [("python_traceback.txt", "python"), ("node_stacktrace.txt", "node")],
)
def test_real_traces_are_detected(name, expected_language):
    text = fixture(name)
    assert TracebackCompactor.is_traceback(text) is True
    assert TracebackCompactor.detect_language(text) == expected_language


def test_rust_and_test_output_are_detected():
    rust = (
        "thread 'main' panicked at src/lib.rs:12:9:\n"
        "assertion failed: checksum mismatch\n"
        "stack backtrace:\n   0: std::panicking::begin_panic"
    )
    assert TracebackCompactor.is_traceback(rust)
    assert TracebackCompactor.detect_language(rust) == "rust"

    pytest_output = "=========== FAILURES ===========\nE   AssertionError: assert 1 == 2"
    assert TracebackCompactor.is_traceback(pytest_output)


def test_single_quoted_frames_are_recognised():
    """Some tools emit File '...' with single quotes rather than double."""
    trace = (
        "Traceback (most recent call last):\n"
        "  File 'a.py', line 1, in run\n"
        "ZeroDivisionError: division by zero"
    )
    assert TracebackCompactor.is_traceback(trace)


# -- compaction ------------------------------------------------------------------


def test_python_traceback_keeps_user_frames_and_root_cause():
    raw = fixture("python_traceback.txt")
    compacted, before, after = TracebackCompactor.compact(raw)

    assert after < before
    assert "site-packages" not in compacted
    assert "/srv/app/api/routes/checkout.py" in compacted
    assert "/srv/app/services/payments.py" in compacted
    assert "httpx.ConnectTimeout" in compacted
    assert "payments-gateway.internal:8443" in compacted


def test_node_stack_drops_node_modules():
    raw = fixture("node_stacktrace.txt")
    compacted, before, after = TracebackCompactor.compact(raw)

    assert after < before
    assert "node_modules" not in compacted
    assert "node:internal" not in compacted
    assert "src/server/tenancy.ts" in compacted
    assert "organizationId" in compacted


def test_compaction_never_reports_a_saving_it_did_not_achieve():
    """
    When the compacted form is not smaller the original is returned unchanged and
    the two token counts are equal, so callers cannot book a phantom saving.
    """
    tiny = "Traceback (most recent call last):\n  File \"a.py\", line 1\nValueError: x"
    compacted, before, after = TracebackCompactor.compact(tiny)
    if after == before:
        assert compacted == tiny


def test_generic_compaction_keeps_a_tail_window():
    """
    With no recognisable stack format and no error lines, a head and tail window is
    kept. 0.1.0 emitted a negative count ("-3 lines truncated") for short input.
    """
    lines = [f"line {i}" for i in range(20)]
    result = TracebackCompactor._compact_generic_error(lines)
    assert "-" not in result.split("lines omitted")[0].split()[-1]
    assert "line 19" in result, "tail must be preserved"
    assert "line 0" in result, "head must be preserved"


# -- outline folding -------------------------------------------------------------

SAMPLE_CLASS = '''"""Account handling."""
import os


class AccountManager:
    """Manages accounts."""

    def __init__(self):
        self.users = {}
        self.sessions = []

    def create_user(self, name: str, email: str) -> bool:
        """Create a user."""
        if name in self.users:
            return False
        self.users[name] = {"email": email, "roles": ["read"]}
        return True

    def delete_user(self, name: str) -> None:
        """Delete a user."""
        self.users.pop(name, None)
        self.sessions = [s for s in self.sessions if s != name]

    def process_batch(self, items: list) -> list:
        """Process a batch."""
        results = []
        for item in items:
            results.append(str(item).upper())
        return results


def module_level_helper(value):
    """Helper."""
    return value * 2
'''


def test_outline_folds_bodies_and_keeps_signatures():
    outline, before, after = CodeOutlineCompactor.compact_code("account.py", SAMPLE_CLASS)

    assert after < before
    assert "class AccountManager" in outline
    assert "def create_user" in outline
    assert "def delete_user" in outline
    assert "folded" in outline
    # Bodies must be gone.
    assert "results.append" not in outline
    assert 'self.users[name] = {"email"' not in outline


def test_focus_expands_one_method_and_folds_its_siblings():
    """
    Focus mode must end when the focused block ends.

    0.1.0 exited focus on `not line.startswith(" ")`, which is never true for a
    sibling method inside a class. Focus therefore never closed, nothing after the
    focused method was folded, and the function returned the file unchanged —
    zero compaction on any class.
    """
    outline, before, after = CodeOutlineCompactor.compact_code(
        "account.py", SAMPLE_CLASS, focus_symbol="delete_user"
    )

    assert after < before, "focus mode achieved no compaction at all"
    # The focused method keeps its body.
    assert "self.users.pop(name, None)" in outline
    # Its siblings do not.
    assert "results.append" not in outline
    assert 'self.users[name] = {"email"' not in outline
    # But their signatures survive.
    assert "def create_user" in outline
    assert "def process_batch" in outline


@pytest.mark.parametrize("symbol", ["create_user", "delete_user", "process_batch", "module_level_helper"])
def test_every_focus_target_compacts(symbol):
    outline, before, after = CodeOutlineCompactor.compact_code(
        "account.py", SAMPLE_CLASS, focus_symbol=symbol
    )
    assert after < before, f"focus on {symbol} produced no reduction"
    assert symbol in outline


def test_outline_uses_the_parse_tree_for_python():
    """Signatures come from `ast`, so annotations and defaults survive intact."""
    source = "def handler(request: Request, *, retries: int = 3) -> Response:\n" + "    pass\n" * 30
    outline, _, _ = CodeOutlineCompactor.compact_code("h.py", source)
    assert "def handler(request: Request, *, retries: int=3) -> Response:" in outline.replace(
        "retries: int = 3", "retries: int=3"
    )


def test_unparseable_python_falls_back_without_raising():
    broken = "def broken(:\n    this is not python\n" + "x\n" * 60
    outline, before, after = CodeOutlineCompactor.compact_code("broken.py", broken)
    assert isinstance(outline, str) and outline


def test_non_python_falls_back_to_indent_scanner():
    typescript = (
        "import { Request } from 'express';\n\n"
        "export class Router {\n"
        "  handle(req: Request) {\n"
        "    const a = 1;\n"
        "    const b = 2;\n"
        "    return a + b;\n"
        "  }\n\n"
        "  dispatch(req: Request) {\n"
        "    const c = 3;\n"
        "    return c;\n"
        "  }\n"
        "}\n"
    ) * 4
    outline, before, after = CodeOutlineCompactor.compact_code("router.ts", typescript)
    assert after < before
    assert "class Router" in outline


def test_empty_input_is_returned_unchanged():
    for text in ("", "   \n\n"):
        outline, before, after = CodeOutlineCompactor.compact_code("x.py", text)
        assert outline == text
        assert before == after
