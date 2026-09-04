"""
ContextOS: a virtual memory kernel for LLM context windows.

Keeps the active working set inside a token budget, moves cold pages to local
disk behind a one-line tombstone, and pages them back in when the model refers
to them.
"""

__version__ = "0.2.1"

from contextos.core.kernel import ContextKernel
from contextos.core.pager import ContextPager
from contextos.core.compactor import CodeOutlineCompactor, TracebackCompactor
from contextos.core.types import ContextPage, MemoryMetrics, PageStatus, PageTier
from contextos.core.workspace import WorkspaceScanner

__all__ = [
    "CodeOutlineCompactor",
    "ContextKernel",
    "ContextPage",
    "ContextPager",
    "MemoryMetrics",
    "PageStatus",
    "PageTier",
    "TracebackCompactor",
    "WorkspaceScanner",
    "__version__",
]
