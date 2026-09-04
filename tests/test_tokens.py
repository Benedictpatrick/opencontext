"""
Tests for token estimation.

`core/tokens.py` documents a measured accuracy against tiktoken. These tests
re-run that measurement so the published claim is checked rather than asserted; if
the estimator drifts outside the documented tolerance, this fails.
"""

from __future__ import annotations

import glob
import os
import statistics

import pytest

from opencontext.core import tokens as tokens_module
from opencontext.core.tokens import estimate_tokens, heuristic_tokens, reset_tokenizer_cache

# The tolerance documented in core/tokens.py.
DOCUMENTED_MEAN_ERROR_PCT = 7.1
DOCUMENTED_WORST_ERROR_PCT = 21.1
# Headroom so ordinary edits do not fail the suite, while real drift does.
MEAN_TOLERANCE = DOCUMENTED_MEAN_ERROR_PCT + 3.0
WORST_TOLERANCE = DOCUMENTED_WORST_ERROR_PCT + 8.0


def test_empty_string_is_zero_tokens():
    assert estimate_tokens("") == 0
    assert heuristic_tokens("") == 0


def test_any_non_empty_string_is_at_least_one_token():
    for text in ("a", " ", "\n", "hi"):
        assert estimate_tokens(text) >= 1


def test_estimates_grow_with_length():
    short = estimate_tokens("def handler(): pass")
    long = estimate_tokens("def handler(): pass\n" * 50)
    assert long > short


def test_the_heuristic_is_deterministic():
    """
    Budgets must be reproducible across machines, so the default estimator cannot
    vary by environment.
    """
    text = "def handler(request):\n    return process(request)\n" * 20
    assert len({heuristic_tokens(text) for _ in range(10)}) == 1


@pytest.fixture(autouse=True)
def _reset_tokenizer():
    reset_tokenizer_cache()
    yield
    reset_tokenizer_cache()


def test_exact_tokenizer_is_opt_in(monkeypatch):
    monkeypatch.delenv("OPENCONTEXT_TOKENIZER", raising=False)
    reset_tokenizer_cache()
    assert tokens_module._get_encoder() is None
    text = "def handler(): pass\n" * 10
    assert estimate_tokens(text) == heuristic_tokens(text)


def test_opting_in_uses_the_exact_tokenizer(monkeypatch):
    pytest.importorskip("tiktoken")
    monkeypatch.setenv("OPENCONTEXT_TOKENIZER", "tiktoken")
    reset_tokenizer_cache()

    assert tokens_module._get_encoder() is not None
    import tiktoken

    text = "def handler(request):\n    return process(request)\n" * 5
    assert estimate_tokens(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))


def test_a_broken_tokenizer_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("OPENCONTEXT_TOKENIZER", "tiktoken")
    reset_tokenizer_cache()

    class Broken:
        def encode(self, text):
            raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr(tokens_module, "_encoder", Broken())
    monkeypatch.setattr(tokens_module, "_encoder_resolved", True)

    text = "hello world"
    assert estimate_tokens(text) == heuristic_tokens(text)


def test_heuristic_accuracy_stays_within_the_documented_tolerance():
    """
    Measure the heuristic against tiktoken over this repository's own source.

    `core/tokens.py` publishes a mean absolute error of 7.1% and a worst case of
    21.1%. If an edit moves the estimator well outside that, the documentation is
    no longer true and this test says so.
    """
    tiktoken = pytest.importorskip("tiktoken")
    encoder = tiktoken.get_encoding("cl100k_base")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = glob.glob(os.path.join(root, "opencontext", "**", "*.py"), recursive=True)
    paths.append(os.path.join(root, "README.md"))
    paths = [path for path in paths if os.path.exists(path)]
    assert len(paths) >= 8, "expected a meaningful sample of files to measure against"

    errors = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        if not text.strip():
            continue
        actual = len(encoder.encode(text))
        estimated = heuristic_tokens(text)
        errors.append(abs(estimated - actual) / actual * 100)

    mean_error = statistics.mean(errors)
    worst_error = max(errors)

    assert mean_error <= MEAN_TOLERANCE, (
        f"mean error {mean_error:.1f}% exceeds the documented {DOCUMENTED_MEAN_ERROR_PCT}% "
        "— update core/tokens.py or fix the estimator"
    )
    assert worst_error <= WORST_TOLERANCE, (
        f"worst-case error {worst_error:.1f}% exceeds the documented {DOCUMENTED_WORST_ERROR_PCT}%"
    )


def test_the_heuristic_beats_naive_character_division():
    """The whitespace correction has to earn its place."""
    tiktoken = pytest.importorskip("tiktoken")
    encoder = tiktoken.get_encoding("cl100k_base")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = glob.glob(os.path.join(root, "opencontext", "**", "*.py"), recursive=True)

    naive_errors, tuned_errors = [], []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        if not text.strip():
            continue
        actual = len(encoder.encode(text))
        naive_errors.append(abs(len(text) // 4 - actual) / actual * 100)
        tuned_errors.append(abs(heuristic_tokens(text) - actual) / actual * 100)

    assert statistics.mean(tuned_errors) < statistics.mean(naive_errors)
