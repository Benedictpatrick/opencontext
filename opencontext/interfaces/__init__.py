"""
OpenContext interfaces: terminal UI, monitor, web dashboard/proxy and MCP server.

Imports are resolved lazily. Importing all four eagerly meant that `opencontext mcp`
— a stdio server that needs neither a web framework nor a terminal UI — pulled in
FastAPI, uvicorn and Textual before it could serve a byte, and an import error in
any one interface broke every other one.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "OpenContextApp",
    "OpenContextMCPServer",
    "ContextTopUI",
    "create_proxy_app",
    "run_interactive_tui",
    "run_mcp_server",
    "run_proxy_server",
    "run_top",
]

_EXPORTS = {
    "ContextTopUI": ("opencontext.interfaces.tui", "ContextTopUI"),
    "run_top": ("opencontext.interfaces.tui", "run_top"),
    "OpenContextApp": ("opencontext.interfaces.interactive_tui", "OpenContextApp"),
    "run_interactive_tui": ("opencontext.interfaces.interactive_tui", "run_interactive_tui"),
    "create_proxy_app": ("opencontext.interfaces.proxy", "create_proxy_app"),
    "run_proxy_server": ("opencontext.interfaces.proxy", "run_proxy_server"),
    "OpenContextMCPServer": ("opencontext.interfaces.mcp_server", "OpenContextMCPServer"),
    "run_mcp_server": ("opencontext.interfaces.mcp_server", "run_mcp_server"),
}


def __getattr__(name: str) -> Any:
    """Import an interface only when it is actually referenced (PEP 562)."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list:
    return sorted(__all__)
