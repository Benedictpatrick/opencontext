"""
OpenAI-compatible reverse proxy and web dashboard.

Sits between an editor (Cursor, Continue.dev, LibreChat) and an upstream model
server. Incoming requests are routed through the kernel — traces compacted, cold
pages swapped out, referenced pages paged back in — and the resulting context is
forwarded upstream.

Design rules this module holds to:

  * Streaming requests stream. `stream: true` is proxied as server-sent events
    rather than buffered and re-parsed as JSON.
  * Fields ContextOS does not understand (`tools`, `response_format`, and
    anything else the client sends) are passed through untouched.
  * When the upstream is unreachable the proxy returns an error. It never
    synthesises a completion, because a client cannot tell a fabricated answer
    from a real one.
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from contextos import __version__
from contextos.core.compactor import TracebackCompactor
from contextos.core.kernel import ContextKernel
from contextos.core.pager import ContextPager
from contextos.core.workspace import WorkspaceScanner

DEFAULT_UPSTREAM = "http://localhost:11434/v1"
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    # extra="allow" keeps tools, tool_choice, response_format, seed and any other
    # client field intact. A closed model silently dropped them, which broke
    # tool-calling clients in a way that looked like a model failure.
    model_config = ConfigDict(extra="allow")

    model: str = "contextos/vkernel"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class PageActionRequest(BaseModel):
    page_id: str


class CompactRequest(BaseModel):
    error_log: str


class UserChatRequest(BaseModel):
    message: str


class ProxyConfig(BaseModel):
    """Runtime configuration, resolved from arguments then environment."""

    upstream_url: str = Field(default_factory=lambda: os.environ.get("CONTEXTOS_UPSTREAM", DEFAULT_UPSTREAM))
    api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("CONTEXTOS_API_KEY") or None)
    upstream_api_key: Optional[str] = Field(
        default_factory=lambda: os.environ.get("CONTEXTOS_UPSTREAM_API_KEY") or None
    )
    expose_page_content: bool = Field(
        default_factory=lambda: os.environ.get("CONTEXTOS_EXPOSE_CONTENT", "").lower()
        in ("1", "true", "yes")
    )


def _message_text(content: Any) -> str:
    """Flatten OpenAI message content, which may be a string or a content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content or "")


def create_proxy_app(
    kernel: Optional[ContextKernel] = None,
    upstream_url: Optional[str] = None,
    config: Optional[ProxyConfig] = None,
    scan_on_start: bool = True,
    root_dir: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI application."""
    kernel = kernel or ContextKernel()
    settings = config or ProxyConfig()
    if upstream_url:
        settings.upstream_url = upstream_url

    pager = ContextPager(kernel, root_dir=root_dir)
    scanner = WorkspaceScanner(kernel, root_dir=root_dir)

    app = FastAPI(
        title="ContextOS",
        version=__version__,
        description="Virtual memory kernel for LLM context windows",
    )
    app.state.kernel = kernel
    app.state.config = settings

    if scan_on_start and not kernel.pages:
        scanner.scan_and_ingest()

    static_html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")

    def require_api_key(request: Request) -> None:
        """
        Enforce a shared secret when one is configured.

        The dashboard and page APIs expose the contents of the indexed workspace.
        Binding to anything other than localhost without CONTEXTOS_API_KEY set
        would publish that source tree to the network.
        """
        if not settings.api_key:
            return
        header = request.headers.get("Authorization", "")
        presented = header[7:] if header.startswith("Bearer ") else request.headers.get("X-API-Key", "")
        if presented != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    guarded = [Depends(require_api_key)]

    # -- dashboard --------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def serve_dashboard() -> HTMLResponse:
        if os.path.exists(static_html_path):
            with open(static_html_path, "r", encoding="utf-8") as handle:
                return HTMLResponse(content=handle.read())
        return HTMLResponse(
            content="<h1>ContextOS</h1><p>Dashboard assets are missing from this install.</p>",
            status_code=500,
        )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__, "pages": len(kernel.pages)}

    @app.get("/metrics", dependencies=guarded)
    async def get_metrics() -> Dict[str, Any]:
        return kernel.get_metrics().model_dump()

    @app.get("/api/pages", dependencies=guarded)
    async def list_pages(include_content: bool = False) -> Dict[str, Any]:
        """
        List the page table.

        File contents are omitted unless explicitly requested and the server was
        started with content exposure enabled, so a casual GET does not dump the
        entire indexed source tree.
        """
        allow_content = include_content and settings.expose_page_content
        pages = []
        for page in kernel.pages.values():
            entry: Dict[str, Any] = {
                "id": page.id,
                "title": page.title,
                "tier": page.tier.value,
                "status": page.status.value,
                "tokens": page.token_count,
                "access_count": page.access_count,
                "compacted": page.compacted,
                "tombstone": page.tombstone,
                "preview": (page.content[:200] if page.content else ""),
            }
            if allow_content:
                entry["content"] = page.content
            pages.append(entry)
        return {
            "pages": pages,
            "content_included": allow_content,
            "content_available": settings.expose_page_content,
        }

    @app.post("/api/page_in", dependencies=guarded)
    async def api_page_in(request: PageActionRequest) -> Dict[str, Any]:
        page = kernel.page_fault(request.page_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"Page '{request.page_id}' not found")
        return {"result": "paged_in", "page_id": page.id, "tokens": page.token_count}

    @app.post("/api/swap_out", dependencies=guarded)
    async def api_swap_out(request: PageActionRequest) -> Dict[str, Any]:
        page = kernel.pages.get(request.page_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"Page '{request.page_id}' not found")
        if not kernel.page_out(request.page_id):
            reason = (
                "page is pinned to L0"
                if page.tier.value == "L0_PINNED"
                else "page is already swapped out"
            )
            raise HTTPException(status_code=409, detail=f"Cannot swap out: {reason}")
        return {"result": "swapped_out", "page_id": request.page_id, "tombstone": page.tombstone}

    @app.post("/api/compact", dependencies=guarded)
    async def api_compact(request: CompactRequest) -> Dict[str, Any]:
        compacted, original, remaining = TracebackCompactor.compact(request.error_log)
        saved = original - remaining
        return {
            "compacted_text": compacted,
            "language": TracebackCompactor.detect_language(request.error_log),
            "original_tokens": original,
            "compacted_tokens": remaining,
            "tokens_saved": saved,
            "reduction_pct": round(saved / original * 100, 1) if original else 0.0,
            "unchanged": saved == 0,
        }

    @app.post("/api/scan", dependencies=guarded)
    async def api_scan() -> Dict[str, int]:
        return scanner.scan_and_ingest()

    @app.post("/api/chat", dependencies=guarded)
    async def api_chat(request: UserChatRequest) -> Dict[str, Any]:
        """
        Answer a question about the workspace using the assembled context.

        Requires a configured upstream model. When none is reachable this reports
        the failure; it does not invent an answer.
        """
        resolved = pager.process_incoming_prompt_verbose(request.message)

        payload = {
            "model": os.environ.get("CONTEXTOS_MODEL", "qwen2.5-coder:7b"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a software engineering assistant. The following context "
                        "was assembled by ContextOS from the user's workspace.\n\n"
                        f"{resolved['context']}"
                    ),
                },
                {"role": "user", "content": request.message},
            ],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                response = await client.post(
                    f"{settings.upstream_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=_upstream_headers(settings, None),
                )
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"No model server reachable at {settings.upstream_url} ({error}). "
                    "Start a local model (for example `ollama serve`) or set "
                    "CONTEXTOS_UPSTREAM and CONTEXTOS_UPSTREAM_API_KEY."
                ),
            ) from error

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream returned {response.status_code}: {response.text[:300]}",
            )

        try:
            answer = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as error:
            raise HTTPException(
                status_code=502, detail=f"Malformed upstream response: {error}"
            ) from error

        return {
            "response": answer,
            "rehydrated": resolved["rehydrated"],
            "context_tokens": resolved["context_tokens"],
        }

    # -- OpenAI-compatible surface ---------------------------------------------

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": f"contextos/{name}", "object": "model", "created": created, "owned_by": "contextos"}
                for name in ("vkernel", "qwen2.5-coder", "llama3.1", "gpt-4o")
            ],
        }

    @app.post("/v1/chat/completions", dependencies=guarded)
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        """
        Route a chat completion through the kernel and on to the upstream model.

        Conversation history is preserved. The kernel's assembled context is
        prepended as a system message rather than replacing the turns, so the
        model still sees what was said.
        """
        payload = request.model_dump(exclude_none=True)
        incoming = payload.get("messages", [])

        latest_user_prompt = ""
        for index, message in enumerate(request.messages):
            text = _message_text(message.content)
            if not text:
                continue
            if message.role == "system":
                pager.pin_instruction(f"sys_{index}", "System instructions", text)
            elif TracebackCompactor.is_traceback(text):
                pager.ingest_traceback(text, title=f"Error log (turn {index})")
            elif message.role == "user":
                latest_user_prompt = text
                pager.ingest_conversation_turn("user", text)

        rehydrated = kernel.touch_or_fault(latest_user_prompt)
        assembled = kernel.assemble_context()

        forwarded: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Context assembled by ContextOS from the user's workspace. Pages marked "
                    "as swapped are available on request.\n\n" + assembled
                ),
            }
        ]
        # Keep the client's own turns, minus their system messages (folded above).
        forwarded.extend(
            message for message in incoming if message.get("role") != "system"
        )

        payload["messages"] = forwarded
        payload["model"] = str(payload.get("model", "")).replace("contextos/", "")

        upstream = f"{settings.upstream_url.rstrip('/')}/chat/completions"
        headers = _upstream_headers(settings, raw_request.headers.get("Authorization"))

        if request.stream:
            return StreamingResponse(
                _stream_upstream(upstream, payload, headers),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-ContextOS-Rehydrated": str(len(rehydrated)),
                },
            )

        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                response = await client.post(upstream, json=payload, headers=headers)
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"ContextOS could not reach the upstream model at {settings.upstream_url}: "
                    f"{error}. Set CONTEXTOS_UPSTREAM to a running OpenAI-compatible endpoint."
                ),
            ) from error

        if response.status_code != 200:
            return JSONResponse(status_code=response.status_code, content=_safe_json(response))

        return JSONResponse(
            content=_safe_json(response),
            headers={"X-ContextOS-Rehydrated": str(len(rehydrated))},
        )

    return app


def _upstream_headers(settings: ProxyConfig, client_authorization: Optional[str]) -> Dict[str, str]:
    """
    Build upstream headers.

    A key configured on the server wins over one presented by the client, so the
    operator controls which credential leaves the machine.
    """
    headers = {"Content-Type": "application/json"}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    elif client_authorization:
        headers["Authorization"] = client_authorization
    return headers


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text[:1000], "type": "upstream_error"}}


async def _stream_upstream(
    url: str, payload: Dict[str, Any], headers: Dict[str, str]
) -> AsyncIterator[bytes]:
    """
    Relay an upstream SSE stream to the client verbatim.

    The previous implementation forwarded `stream: true` and then called `.json()`
    on the response, which always raised and fell through to a synthesised reply —
    so every streaming client, which is most of them, received a placeholder.
    """
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    yield _sse_error(f"Upstream returned {response.status_code}: {body[:300]}")
                    return
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
    except httpx.HTTPError as error:
        yield _sse_error(f"ContextOS could not reach the upstream model: {error}")


def _sse_error(message: str) -> bytes:
    """Emit an error inside the SSE envelope the client is already parsing."""
    import json

    payload = json.dumps({"error": {"message": message, "type": "upstream_unavailable"}})
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode("utf-8")


def run_proxy_server(
    kernel: Optional[ContextKernel] = None,
    host: str = "127.0.0.1",
    port: int = 9090,
    auto_open: bool = False,
    upstream_url: Optional[str] = None,
) -> None:
    """Launch the server. Called by `contextos serve`."""
    import webbrowser

    app = create_proxy_app(kernel, upstream_url=upstream_url)

    if auto_open:
        try:
            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass

    uvicorn.run(app, host=host, port=port, log_level="info")
