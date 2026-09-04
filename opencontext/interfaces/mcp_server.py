"""
Model Context Protocol server (JSON-RPC 2.0 over stdio).

Exposes the kernel to MCP clients such as Claude Code, Cursor and Antigravity.

Protocol notes that the previous implementation got wrong, and which clients do
notice:

  * A JSON-RPC *notification* carries no `id` and must receive no response. The
    old loop replied "Method not found" to `notifications/initialized`, which is
    part of the standard handshake.
  * An error response must echo the request's `id`. Returning `null` left the
    client unable to match an error to the call that caused it.
  * Only `stdout` may carry protocol traffic. Anything diagnostic goes to stderr,
    or it corrupts the stream.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, List, Optional, TextIO

from opencontext import __version__
from opencontext.core.compactor import CodeOutlineCompactor, TracebackCompactor
from opencontext.core.kernel import ContextKernel
from opencontext.core.pager import ContextPager
from opencontext.core.workspace import WorkspaceScanner

PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class OpenContextMCPServer:
    """Serves the OpenContext kernel over MCP stdio."""

    def __init__(self, kernel: Optional[ContextKernel] = None, root_dir: Optional[str] = None):
        self.kernel = kernel or ContextKernel()
        self.pager = ContextPager(self.kernel, root_dir=root_dir)
        self.scanner = WorkspaceScanner(self.kernel, root_dir=root_dir)
        self._initialized = False

    # -- tool surface -----------------------------------------------------------

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Schemas for every tool this server exposes."""
        return [
            {
                "name": "context_inspect",
                "description": (
                    "Show current context memory usage: working tokens against budget, "
                    "the page table, swap state and any repeat-failure warnings."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "context_page_in",
                "description": (
                    "Restore a swapped-out page from disk into the context window."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page id, e.g. 'file:src/auth.py'",
                        }
                    },
                    "required": ["page_id"],
                },
            },
            {
                "name": "context_force_swap",
                "description": "Move a page out to disk to free space in the context window.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {"type": "string", "description": "Page id to swap out"}
                    },
                    "required": ["page_id"],
                },
            },
            {
                "name": "context_compact_error",
                "description": (
                    "Reduce a stack trace to its user frames and root cause. Supports "
                    "Python, Node.js, Rust, Go and test-runner output. Returns the "
                    "original text unchanged if compaction would not shrink it."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "error_log": {"type": "string", "description": "Raw error text"}
                    },
                    "required": ["error_log"],
                },
            },
            {
                "name": "context_ingest_file",
                "description": (
                    "Read a file into context memory. With focus_symbol, long files are "
                    "folded to an outline with that symbol left expanded."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to the file"},
                        "focus_symbol": {
                            "type": "string",
                            "description": "Function or class to keep expanded",
                        },
                    },
                    "required": ["filepath"],
                },
            },
            {
                "name": "context_outline_file",
                "description": (
                    "Return a structural outline of a file without ingesting it: "
                    "signatures and docstrings, bodies folded."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to the file"},
                        "focus_symbol": {
                            "type": "string",
                            "description": "Symbol to keep expanded",
                        },
                    },
                    "required": ["filepath"],
                },
            },
            {
                "name": "context_search",
                "description": (
                    "Find pages matching a query by keyword over titles, ids and "
                    "resident content. Lexical, not semantic."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search terms"},
                        "top_k": {"type": "integer", "description": "Results to return (default 5)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "context_scan_workspace",
                "description": "Index a project directory into context memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory to scan (default: working directory)",
                        }
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Run a tool and return its result payload."""
        handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "context_inspect": self._tool_inspect,
            "context_page_in": self._tool_page_in,
            "context_force_swap": self._tool_force_swap,
            "context_compact_error": self._tool_compact_error,
            "context_ingest_file": self._tool_ingest_file,
            "context_outline_file": self._tool_outline_file,
            "context_search": self._tool_search,
            "context_scan_workspace": self._tool_scan_workspace,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        return handler(arguments or {})

    def _tool_inspect(self, _arguments: Dict[str, Any]) -> Dict[str, Any]:
        metrics = self.kernel.get_metrics()
        warnings = [
            event.description
            for event in self.kernel.events
            if event.event_type.value in ("LEAK_WARNING", "BUDGET_EXCEEDED")
        ]
        return {
            "working_tokens": metrics.working_tokens,
            "token_budget": metrics.token_budget,
            "utilization_pct": metrics.budget_utilization_pct,
            "over_budget": metrics.over_budget,
            "swapped_tokens": metrics.swapped_tokens,
            "swapped_pages": metrics.l3_pages,
            "tokens_saved": metrics.total_tokens_saved,
            "page_faults": metrics.total_page_faults,
            "warnings": warnings[-5:],
            "pages": [
                {
                    "id": page.id,
                    "title": page.title,
                    "tier": page.tier.value,
                    "status": page.status.value,
                    "tokens": page.token_count,
                }
                for page in self.kernel.pages.values()
            ],
        }

    def _tool_page_in(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        page_id = arguments.get("page_id", "")
        if not page_id:
            return {"error": "page_id is required"}
        page = self.kernel.page_fault(page_id)
        if page is None:
            return {"error": f"Page '{page_id}' is not in memory or swap"}
        return {
            "result": "paged_in",
            "page_id": page.id,
            "tokens": page.token_count,
            "content_preview": page.content[:300]
            + ("..." if len(page.content) > 300 else ""),
        }

    def _tool_force_swap(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        page_id = arguments.get("page_id", "")
        if not page_id:
            return {"error": "page_id is required"}
        page = self.kernel.pages.get(page_id)
        if page is None:
            return {"error": f"Page '{page_id}' is not in memory"}
        if not self.kernel.page_out(page_id):
            reason = (
                "page is pinned to L0"
                if page.tier.value == "L0_PINNED"
                else "page is already swapped out"
            )
            return {"error": f"Cannot swap '{page_id}': {reason}"}
        return {"result": "swapped_out", "page_id": page_id, "tombstone": page.tombstone}

    def _tool_compact_error(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = arguments.get("error_log", "")
        if not raw:
            return {"error": "error_log is required"}
        compacted, original, remaining = TracebackCompactor.compact(raw)
        saved = original - remaining
        return {
            "compacted_text": compacted,
            "language": TracebackCompactor.detect_language(raw),
            "original_tokens": original,
            "compacted_tokens": remaining,
            "tokens_saved": saved,
            "reduction_pct": round(saved / original * 100, 1) if original else 0.0,
            "unchanged": saved == 0,
        }

    def _tool_ingest_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        filepath = arguments.get("filepath", "")
        if not filepath:
            return {"error": "filepath is required"}
        page = self.pager.ingest_file(filepath, focus_symbol=arguments.get("focus_symbol", ""))
        return {
            "result": "ingested",
            "page_id": page.id,
            "title": page.title,
            "tokens": page.token_count,
            "page_status": page.status.value,
            "outlined": bool(page.metadata.get("outlined")),
        }

    def _tool_outline_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        filepath = arguments.get("filepath", "")
        if not filepath:
            return {"error": "filepath is required"}
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
                source = handle.read()
        except OSError as error:
            return {"error": f"Cannot read '{filepath}': {error}"}

        import os

        outline, original, remaining = CodeOutlineCompactor.compact_code(
            os.path.basename(filepath), source, arguments.get("focus_symbol", "")
        )
        return {
            "outline": outline,
            "original_tokens": original,
            "outline_tokens": remaining,
            "reduction_pct": round((original - remaining) / original * 100, 1) if original else 0.0,
        }

    def _tool_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = arguments.get("query", "")
        if not query:
            return {"error": "query is required"}
        top_k = arguments.get("top_k", 5)
        try:
            top_k = max(1, min(50, int(top_k)))
        except (TypeError, ValueError):
            top_k = 5

        return {
            "query": query,
            "matches": [
                {
                    "page_id": page.id,
                    "title": page.title,
                    "score": score,
                    "status": page.status.value,
                    "tokens": page.token_count,
                }
                for page, score in self.kernel.search(query, top_k=top_k)
            ],
        }

    def _tool_scan_workspace(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        directory = arguments.get("directory")
        scanner = (
            WorkspaceScanner(self.kernel, root_dir=directory) if directory else self.scanner
        )
        return scanner.scan_and_ingest()

    # -- JSON-RPC ---------------------------------------------------------------

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process one JSON-RPC message.

        Returns the response to send, or None when the message is a notification
        (no `id`), which the protocol says must not be answered.
        """
        method = message.get("method")
        message_id = message.get("id")
        is_notification = "id" not in message

        if not isinstance(method, str):
            if is_notification:
                return None
            return self._error(message_id, INVALID_REQUEST, "Missing or invalid 'method'")

        if method == "notifications/initialized" or method.startswith("notifications/"):
            self._initialized = True
            return None

        if method == "initialize":
            return self._result(
                message_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "opencontext", "version": __version__},
                },
            )

        if method == "ping":
            return self._result(message_id, {})

        if method == "tools/list":
            return self._result(message_id, {"tools": self.get_tool_definitions()})

        if method == "tools/call":
            params = message.get("params") or {}
            tool_name = params.get("name")
            if not isinstance(tool_name, str):
                return self._error(message_id, INVALID_PARAMS, "Missing tool name")

            try:
                result = self.execute_tool(tool_name, params.get("arguments") or {})
            except Exception as error:  # a tool fault must not kill the server
                return self._result(
                    message_id,
                    {
                        "content": [{"type": "text", "text": f"Tool '{tool_name}' failed: {error}"}],
                        "isError": True,
                    },
                )

            is_error = isinstance(result, dict) and "error" in result
            return self._result(
                message_id,
                {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
                    "isError": is_error,
                },
            )

        # Advertised as unsupported rather than left to time out.
        if method in ("resources/list", "prompts/list"):
            key = "resources" if method.startswith("resources") else "prompts"
            return self._result(message_id, {key: []})

        if is_notification:
            return None
        return self._error(message_id, METHOD_NOT_FOUND, f"Method not found: {method}")

    @staticmethod
    def _result(message_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    @staticmethod
    def _error(message_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}

    def run_stdio_loop(self, stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None) -> None:
        """Read requests from stdin and write responses to stdout until EOF."""
        source = stdin or sys.stdin
        sink = stdout or sys.stdout

        for line in source:
            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError as error:
                self._write(sink, self._error(None, PARSE_ERROR, f"Invalid JSON: {error}"))
                continue

            if not isinstance(message, dict):
                self._write(sink, self._error(None, INVALID_REQUEST, "Request must be an object"))
                continue

            try:
                response = self.handle_message(message)
            except Exception as error:
                response = self._error(message.get("id"), INTERNAL_ERROR, str(error))

            if response is not None:
                self._write(sink, response)

    @staticmethod
    def _write(sink: TextIO, payload: Dict[str, Any]) -> None:
        sink.write(json.dumps(payload, default=str) + "\n")
        sink.flush()


def run_mcp_server(root_dir: Optional[str] = None) -> None:
    """Entry point for `opencontext mcp`."""
    # Diagnostics go to stderr: stdout is the protocol channel.
    print("OpenContext MCP server ready on stdio", file=sys.stderr, flush=True)
    OpenContextMCPServer(root_dir=root_dir).run_stdio_loop()
