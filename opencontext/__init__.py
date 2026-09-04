"""
OpenContext: a virtual memory kernel for LLM context windows.

Keeps the active working set inside a token budget, moves cold pages to local
disk behind a one-line tombstone, and pages them back in when the model refers
to them.
"""

__version__ = "0.3.0"

from opencontext.core.kernel import ContextKernel
from opencontext.core.pager import ContextPager
from opencontext.core.compactor import CodeOutlineCompactor, TracebackCompactor
from opencontext.core.types import ContextPage, MemoryMetrics, PageStatus, PageTier
from opencontext.core.workspace import WorkspaceScanner

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
