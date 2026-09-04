"""
Tests for the dashboard and OpenAI-compatible proxy.

The upstream is stubbed with a local transport, so these exercise the real request
path — including streaming — without needing a model server.
"""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.testclient import TestClient

from contextos.core.kernel import ContextKernel
from contextos.interfaces import proxy as proxy_module
from contextos.interfaces.proxy import ProxyConfig, create_proxy_app
from contextos.storage.swap import SwapStorage


@pytest.fixture
def kernel(tmp_path):
    kernel = ContextKernel(token_budget=5000, swap_storage=SwapStorage(str(tmp_path / "swap.db")))
    kernel.allocate_page("file:auth.py", "auth.py", "def verify_token(token):\n    return token\n")
    return kernel


@pytest.fixture
def client(kernel):
    app = create_proxy_app(kernel=kernel, scan_on_start=False, config=ProxyConfig(upstream_url="http://upstream.test/v1"))
    return TestClient(app, raise_server_exceptions=False)


class _ChunkStream(httpx.AsyncByteStream):
    """
    An SSE body delivered chunk by chunk.

    httpx requires an AsyncByteStream here. A response built with `content=bytes`
    arrives already consumed, so `aiter_raw` raises StreamConsumed — an artefact of
    MockTransport, not of the proxy, which streams correctly against a real server.
    """

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class StubUpstream:
    """Captures what the proxy forwards and returns a canned response."""

    def __init__(self, status_code=200, payload=None, stream_chunks=None):
        self.status_code = status_code
        self.payload = payload or {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "upstream reply"}}],
        }
        self.stream_chunks = stream_chunks
        self.captured_request = None
        self.captured_headers = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.captured_request = json.loads(request.content)
        self.captured_headers = dict(request.headers)
        if self.stream_chunks is not None:
            # Yielded from an async iterator rather than passed as `content=`.
            # A response built from bytes is already consumed, and `aiter_raw`
            # then raises StreamConsumed — a MockTransport artefact, not proxy
            # behaviour; a real server streams these chunks as they arrive.
            return httpx.Response(
                self.status_code,
                stream=_ChunkStream(self.stream_chunks),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(self.status_code, json=self.payload)

    def install(self, monkeypatch):
        install_transport(monkeypatch, self.handler)
        return self


def install_transport(monkeypatch, handler):
    """
    Route the proxy's outbound requests through a stub transport.

    `proxy_module.httpx` *is* the httpx module, so patching AsyncClient on it is a
    global change. The real class must be captured before patching — a factory that
    looks up `httpx.AsyncClient` at call time would find its own replacement and
    recurse until the stack runs out.
    """
    real_client_cls = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", factory)


def unreachable_upstream(request):
    raise httpx.ConnectError("connection refused")


# -- basic surface ------------------------------------------------------------------


def test_health_is_unauthenticated(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]


def test_models_endpoint_lists_contextos_models(client):
    ids = [model["id"] for model in client.get("/v1/models").json()["data"]]
    assert "contextos/vkernel" in ids


def test_metrics_separates_live_and_disk_swap_figures(client):
    metrics = client.get("/metrics").json()
    for field in ("working_tokens", "swapped_tokens", "swap_disk_tokens", "swap_disk_rows", "over_budget"):
        assert field in metrics


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ContextOS" in response.text


# -- page content exposure ------------------------------------------------------------


def test_page_content_is_withheld_by_default(client):
    """
    A plain GET must not dump the indexed source tree.

    0.1.0 returned full file content for every page from an unauthenticated
    endpoint.
    """
    body = client.get("/api/pages").json()
    assert body["content_included"] is False
    assert "content" not in body["pages"][0]
    assert "preview" in body["pages"][0]


def test_content_is_still_withheld_when_asked_but_not_enabled(client):
    body = client.get("/api/pages?include_content=true").json()
    assert body["content_included"] is False


def test_content_is_served_when_the_operator_enables_it(kernel):
    app = create_proxy_app(
        kernel=kernel, scan_on_start=False, config=ProxyConfig(expose_page_content=True)
    )
    body = TestClient(app).get("/api/pages?include_content=true").json()
    assert body["content_included"] is True
    assert "verify_token" in body["pages"][0]["content"]


# -- authentication ---------------------------------------------------------------------


def test_api_key_is_enforced_when_configured(kernel):
    app = create_proxy_app(kernel=kernel, scan_on_start=False, config=ProxyConfig(api_key="secret"))
    guarded = TestClient(app, raise_server_exceptions=False)

    assert guarded.get("/metrics").status_code == 401
    assert guarded.get("/health").status_code == 200, "health must stay open for probes"
    assert guarded.get("/metrics", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert guarded.get("/metrics", headers={"X-API-Key": "secret"}).status_code == 200


# -- paging endpoints ----------------------------------------------------------------------


def test_swap_out_and_page_in(client):
    swapped = client.post("/api/swap_out", json={"page_id": "file:auth.py"})
    assert swapped.status_code == 200
    assert swapped.json()["result"] == "swapped_out"

    restored = client.post("/api/page_in", json={"page_id": "file:auth.py"})
    assert restored.status_code == 200
    assert restored.json()["result"] == "paged_in"


def test_paging_a_missing_page_is_a_404(client):
    assert client.post("/api/page_in", json={"page_id": "file:nope.py"}).status_code == 404


def test_swapping_a_pinned_page_is_a_409(client, kernel):
    kernel.pin_page("file:auth.py")
    response = client.post("/api/swap_out", json={"page_id": "file:auth.py"})
    assert response.status_code == 409
    assert "pinned" in response.json()["detail"]


def test_compact_endpoint_reports_measured_reduction(client):
    trace = (
        "Traceback (most recent call last):\n"
        + '  File "/venv/lib/python3.11/site-packages/x.py", line 1, in f\n    g()\n' * 10
        + '  File "/app/main.py", line 3, in run\n    boom()\n'
        + "ValueError: boom"
    )
    body = client.post("/api/compact", json={"error_log": trace}).json()
    assert body["language"] == "python"
    assert body["tokens_saved"] > 0
    assert body["unchanged"] is False


# -- chat completions ------------------------------------------------------------------------


def test_completions_forwards_to_upstream(client, monkeypatch):
    upstream = StubUpstream().install(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "contextos/gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "upstream reply"
    assert upstream.captured_request["model"] == "gpt-4o", "the contextos/ prefix is stripped"


def test_unknown_fields_are_passed_through(client, monkeypatch):
    """
    0.1.0 rebuilt the payload from a closed model, silently dropping `tools`,
    `response_format` and everything else, which broke tool-calling clients.
    """
    upstream = StubUpstream().install(monkeypatch)

    client.post(
        "/v1/chat/completions",
        json={
            "model": "contextos/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
            "seed": 42,
        },
    )

    forwarded = upstream.captured_request
    assert forwarded["tools"][0]["function"]["name"] == "get_weather"
    assert forwarded["tool_choice"] == "auto"
    assert forwarded["response_format"] == {"type": "json_object"}
    assert forwarded["seed"] == 42


def test_conversation_history_is_preserved(client, monkeypatch):
    """0.1.0 forwarded only the final user turn, discarding the conversation."""
    upstream = StubUpstream().install(monkeypatch)

    client.post(
        "/v1/chat/completions",
        json={
            "model": "contextos/vkernel",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
        },
    )

    forwarded = upstream.captured_request["messages"]
    contents = [message["content"] for message in forwarded]
    assert any("first question" in text for text in contents)
    assert any("first answer" in text for text in contents)
    assert any("second question" in text for text in contents)
    assert forwarded[0]["role"] == "system", "assembled context leads as a system message"


def test_upstream_failure_returns_an_error_not_a_fabricated_reply(client, monkeypatch):
    """
    0.1.0 answered with a synthesised completion when the upstream was unreachable.
    A client cannot tell an invented answer from a real one, so this must be an error.
    """
    install_transport(monkeypatch, unreachable_upstream)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "contextos/vkernel", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "could not reach" in detail.lower()
    assert "CONTEXTOS_UPSTREAM" in detail
    assert "choices" not in response.text, "no completion may be synthesised"


def test_upstream_error_status_is_relayed(client, monkeypatch):
    StubUpstream(status_code=429, payload={"error": {"message": "rate limited"}}).install(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "contextos/vkernel", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 429


# -- streaming -----------------------------------------------------------------------------


def test_streaming_returns_server_sent_events(client, monkeypatch):
    """
    The streaming path must actually stream.

    0.1.0 forwarded `stream: true` and then called `.json()` on the SSE body, which
    always raised and fell through to a synthesised reply — so every streaming
    client, which is most editor integrations, received a placeholder.
    """
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    StubUpstream(stream_chunks=chunks).install(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "contextos/vkernel", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    deltas = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and "[DONE]" not in line:
            deltas.append(json.loads(line[6:])["choices"][0]["delta"]["content"])
    assert "".join(deltas) == "Hello world"


def test_streaming_failure_is_reported_inside_the_sse_envelope(client, monkeypatch):
    """The client is already parsing SSE, so the error is delivered in that shape."""
    install_transport(monkeypatch, unreachable_upstream)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "contextos/vkernel", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.status_code == 200
    assert "upstream_unavailable" in response.text
    assert "[DONE]" in response.text


# -- workspace chat ---------------------------------------------------------------------------


def test_api_chat_reports_when_no_model_is_reachable(client, monkeypatch):
    install_transport(monkeypatch, unreachable_upstream)

    response = client.post("/api/chat", json={"message": "how does auth work?"})
    assert response.status_code == 502
    assert "No model server reachable" in response.json()["detail"]


def test_api_chat_returns_the_model_answer(client, monkeypatch):
    StubUpstream().install(monkeypatch)
    body = client.post("/api/chat", json={"message": "how does auth work?"}).json()
    assert body["response"] == "upstream reply"
    assert "context_tokens" in body


def test_traces_in_messages_are_compacted_on_ingest(client, monkeypatch, kernel):
    StubUpstream().install(monkeypatch)
    trace = (
        "Traceback (most recent call last):\n"
        + '  File "/venv/lib/python3.11/site-packages/x.py", line 1, in f\n    g()\n' * 10
        + '  File "/app/main.py", line 3, in run\n    boom()\n'
        + "ValueError: boom"
    )
    client.post(
        "/v1/chat/completions",
        json={"model": "contextos/vkernel", "messages": [{"role": "user", "content": trace}]},
    )
    assert any(page.compacted for page in kernel.pages.values())
