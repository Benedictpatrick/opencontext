"""
Token estimation for OpenContext.

By default OpenContext uses a deterministic, dependency-free heuristic rather than a
real BPE tokenizer:

  * Budgets must be reproducible across machines and Python versions. A tokenizer
    that is present on one box and absent on another silently changes eviction
    behaviour and benchmark output.
  * The kernel only needs *relative* sizes to decide what to evict.

The heuristic is characters / 4 with a correction for whitespace-heavy text
(indented source code), where the raw ratio over-counts.

Measured against `tiktoken` (cl100k_base) over this repository's own source files:

    characters / 4          mean abs error 10.8%   worst 32.2%
    OpenContext heuristic     mean abs error  7.1%   worst 21.1%

`tests/test_tokens.py` re-runs that measurement and fails if the error moves
outside the documented tolerance, so the claim above is checked, not asserted.

Exact counting is available opt-in for users who want budgets to match a specific
model's tokenizer and accept the extra dependency:

    pip install "opencontext[exact-tokens]"
    export OPENCONTEXT_TOKENIZER=tiktoken

When enabled and importable, `estimate_tokens` uses cl100k_base. If the import
fails, OpenContext falls back to the heuristic rather than crashing.
"""

from __future__ import annotations

import os
from typing import Any, Optional

CHARS_PER_TOKEN = 4.0
WHITESPACE_DISCOUNT = 0.3

_encoder: Optional[Any] = None
_encoder_resolved = False


def _get_encoder() -> Optional[Any]:
    """Return a tiktoken encoder if the user opted in and it is importable."""
    global _encoder, _encoder_resolved
    if _encoder_resolved:
        return _encoder

    _encoder_resolved = True
    if os.environ.get("OPENCONTEXT_TOKENIZER", "").lower() != "tiktoken":
        _encoder = None
        return None

    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _encoder = None
    return _encoder


def reset_tokenizer_cache() -> None:
    """Forget the resolved tokenizer. Used by tests that toggle the env var."""
    global _encoder, _encoder_resolved
    _encoder = None
    _encoder_resolved = False


def heuristic_tokens(text: str) -> int:
    """The deterministic estimator. Always available, never varies by machine."""
    if not text:
        return 0

    whitespace = sum(1 for ch in text if ch in " \t\n\r")
    effective = len(text) - (whitespace * WHITESPACE_DISCOUNT)
    return max(1, int(effective / CHARS_PER_TOKEN))


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of LLM tokens in `text`.

    Uses the exact tokenizer when the user has opted in via OPENCONTEXT_TOKENIZER,
    otherwise the deterministic heuristic. Returns 0 for an empty string and at
    least 1 for anything non-empty.
    """
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(text)))
        except Exception:
            pass
    return heuristic_tokens(text)


def format_tokens(count: int) -> str:
    """Human-readable token count used across the CLI, TUI and dashboard."""
    return f"{count:,}"
