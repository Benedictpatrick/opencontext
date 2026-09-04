"""
Tests for the MCP stdio server.

This module had no tests in 0.1.0, which is why two protocol defects survived in
it: a response sent to a notification, and a duplicate key that overwrote a tool's
result. Both are pinned below.
"""

from __future__ import annotations

import io
import json

import pytest

from contextos.core.kernel import ContextKernel
from contextos.interfaces.mcp_server import METHOD_NOT_FOUND, PARSE_ERROR, ContextOSMCPServer
from contextos.storage.swap import SwapStorage


@pytest.fixture
def server(tmp_path):
    kernel = ContextKernel(token_budget=8000, swap_storage=SwapStorage(str(tmp_path / "swap.db")))
    kernel.allocate_page("file:auth.py", "auth.py", "def verify_token(t):\n    return t\n" * 10)
    return ContextOSMCPServer(kernel=kernel, root_dir=str(tmp_path))


def call_tool(server: ContextOSMCPServer, name: str, arguments=None):
    response = server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": name, "arguments": arguments or {}}}
    )
    return json.loads(response["result"]["content"][0]["text"])


# -- protocol ---------------------------------------------------------------------


def test_initialize_returns_a_handshake(server):
    response = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert response["id"] == 1
    assert response["result"]["protocolVersion"]
    assert response["result"]["serverInfo"]["name"] == "contextos"


def test_notifications_receive_no_response(server):
    """
    A JSON-RPC notification has no `id` and must not be answered.

    0.1.0 replied "Method not found" to `notifications/initialized`, which is part
    of the standard handshake, with `"id": null`.
    """
    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None


def test_unknown_notification_is_still_silent(server):
    assert server.handle_message({"jsonrpc": "2.0", "method": "some/unknown/notification"}) is None


def test_errors_echo_the_request_id(server):
    """0.1.0 returned `"id": null` for every error, so clients could not correlate."""
    response = server.handle_message({"jsonrpc": "2.0", "id": 77, "method": "no/such/method"})
    assert response["id"] == 77
    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_ping_is_answered(server):
    response = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert response["result"] == {}


def test_resources_and_prompts_are_advertised_as_empty(server):
    for method, key in (("resources/list", "resources"), ("prompts/list", "prompts")):
        response = server.handle_message({"jsonrpc": "2.0", "id": 3, "method": method})
        assert response["result"][key] == []


def test_tools_list_is_well_formed(server):
    response = server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
    tools = response["result"]["tools"]
    assert len(tools) >= 5
    for tool in tools:
        assert tool["name"] and tool["description"]
        assert tool["inputSchema"]["type"] == "object"


def test_malformed_json_is_reported_over_stdio(server):
    stdin = io.StringIO('{"not valid json\n')
    stdout = io.StringIO()
    server.run_stdio_loop(stdin=stdin, stdout=stdout)

    response = json.loads(stdout.getvalue().strip())
    assert response["error"]["code"] == PARSE_ERROR


def test_stdio_loop_answers_requests_and_stays_silent_on_notifications(server):
    stdin = io.StringIO(
        '{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'
        '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'
    )
    stdout = io.StringIO()
    server.run_stdio_loop(stdin=stdin, stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().strip().splitlines()]
    assert len(responses) == 2, "the notification must not produce a response"
    assert [r["id"] for r in responses] == [1, 2]


def test_a_failing_tool_does_not_kill_the_server(server, monkeypatch):
    monkeypatch.setattr(
        server, "_tool_inspect", lambda _args: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    response = server.handle_message(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "context_inspect"}}
    )
    assert response["result"]["isError"] is True
    assert "boom" in response["result"]["content"][0]["text"]


# -- tools -------------------------------------------------------------------------


def test_inspect_reports_kernel_state(server):
    result = call_tool(server, "context_inspect")
    assert result["token_budget"] == 8000
    assert result["working_tokens"] > 0
    assert any(page["id"] == "file:auth.py" for page in result["pages"])


def test_ingest_file_reports_that_it_ingested(server, tmp_path):
    """
    0.1.0 set `"status"` twice in one dict literal, so `"ingested"` was overwritten
    by the page status and callers never saw the confirmation.
    """
    target = tmp_path / "sample.py"
    target.write_text("def sample():\n    return 1\n", encoding="utf-8")

    result = call_tool(server, "context_ingest_file", {"filepath": str(target)})
    assert result["result"] == "ingested"
    assert result["page_status"] == "ACTIVE"
    assert result["tokens"] > 0


def test_page_in_and_force_swap_round_trip(server):
    swapped = call_tool(server, "context_force_swap", {"page_id": "file:auth.py"})
    assert swapped["result"] == "swapped_out"
    assert swapped["tombstone"]

    restored = call_tool(server, "context_page_in", {"page_id": "file:auth.py"})
    assert restored["result"] == "paged_in"
    assert "verify_token" in restored["content_preview"]


def test_force_swap_on_a_pinned_page_explains_itself(server):
    server.kernel.pin_page("file:auth.py")
    result = call_tool(server, "context_force_swap", {"page_id": "file:auth.py"})
    assert "error" in result
    assert "pinned" in result["error"]


def test_compact_error_reports_the_measured_reduction(server):
    trace = (
        "Traceback (most recent call last):\n"
        + '  File "/venv/lib/python3.11/site-packages/x/y.py", line 1, in f\n    g()\n' * 8
        + '  File "/app/main.py", line 9, in run\n    boom()\n'
        + "ValueError: boom"
    )
    result = call_tool(server, "context_compact_error", {"error_log": trace})
    assert result["language"] == "python"
    assert result["tokens_saved"] > 0
    assert result["unchanged"] is False
    assert "site-packages" not in result["compacted_text"]


def test_compact_error_says_so_when_it_cannot_shrink(server):
    result = call_tool(server, "context_compact_error", {"error_log": "ValueError: x\nat y"})
    assert result["unchanged"] is True
    assert result["tokens_saved"] == 0


def test_outline_file_returns_a_skeleton(server, tmp_path):
    target = tmp_path / "big.py"
    # Bodies need real substance: a one-line body cannot be folded into something
    # smaller than itself, and the compactor correctly declines to try.
    target.write_text(
        "class Service:\n"
        + "".join(
            f"    def method_{i}(self, request):\n"
            f'        """Handle request {i}."""\n'
            f"        validated = self.validate(request)\n"
            f"        enriched = self.enrich(validated, {i})\n"
            f"        self.audit.record(enriched)\n"
            f"        return self.respond(enriched)\n"
            for i in range(40)
        ),
        encoding="utf-8",
    )
    result = call_tool(server, "context_outline_file", {"filepath": str(target)})
    assert result["outline_tokens"] < result["original_tokens"]
    assert "class Service" in result["outline"]


def test_search_returns_ranked_matches(server):
    result = call_tool(server, "context_search", {"query": "verify token auth"})
    assert result["matches"]
    assert result["matches"][0]["page_id"] == "file:auth.py"


def test_scan_workspace_indexes_a_directory(server, tmp_path):
    (tmp_path / "extra.py").write_text("value = 1\n", encoding="utf-8")
    result = call_tool(server, "context_scan_workspace", {"directory": str(tmp_path)})
    assert result["files_scanned"] >= 1


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("context_page_in", {}),
        ("context_force_swap", {}),
        ("context_compact_error", {}),
        ("context_ingest_file", {}),
        ("context_search", {}),
    ],
)
def test_missing_required_arguments_are_reported(server, tool, arguments):
    assert "error" in call_tool(server, tool, arguments)


def test_unknown_tool_is_reported(server):
    assert "error" in call_tool(server, "context_does_not_exist", {})
