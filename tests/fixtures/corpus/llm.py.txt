"""
Minimal client for OpenAI-compatible chat endpoints.

OpenContext does not bundle a provider SDK. Anything that speaks the OpenAI chat
completions shape works, which covers local runtimes (Ollama, llama.cpp, LM Studio,
vLLM) and hosted providers alike — including Anthropic, via its compatibility
endpoint. Configuration is entirely by environment:

    OPENCONTEXT_UPSTREAM          base URL, default http://localhost:11434/v1
    OPENCONTEXT_UPSTREAM_API_KEY  bearer token, if the endpoint needs one
    OPENCONTEXT_MODEL             model name to request

When no model server is reachable, callers get `LLMUnavailable` and are expected
to say so. OpenContext never substitutes generated-looking text for a real answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import httpx

DEFAULT_UPSTREAM = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5-coder:7b"


class LLMUnavailable(RuntimeError):
    """Raised when no model server could be reached, or it returned an error."""


@dataclass
class LLMConfig:
    """Resolved client settings."""

    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.environ.get("OPENCONTEXT_UPSTREAM", DEFAULT_UPSTREAM).rstrip("/"),
            model=os.environ.get("OPENCONTEXT_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("OPENCONTEXT_UPSTREAM_API_KEY") or None,
        )

    @property
    def is_local(self) -> bool:
        """True when the endpoint is on this machine, so no data leaves it."""
        return any(host in self.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))

    def describe(self) -> str:
        location = "local" if self.is_local else "remote"
        return f"{self.model} at {self.base_url} ({location})"


class LLMClient:
    """Blocking chat client. One request per call, no session state."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def is_reachable(self) -> bool:
        """Check whether the configured endpoint answers, without sending a prompt."""
        try:
            response = httpx.get(
                f"{self.config.base_url}/models", headers=self._headers(), timeout=5.0
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def available_models(self) -> List[str]:
        """Model ids the endpoint advertises. Empty when it is unreachable."""
        try:
            response = httpx.get(
                f"{self.config.base_url}/models", headers=self._headers(), timeout=5.0
            )
            response.raise_for_status()
            return [entry["id"] for entry in response.json().get("data", [])]
        except (httpx.HTTPError, KeyError, ValueError):
            return []

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Send one completion and return the assistant's text.

        Raises `LLMUnavailable` with an actionable message on any failure.
        """
        payload: Dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            response = httpx.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.config.timeout,
            )
        except httpx.HTTPError as error:
            raise LLMUnavailable(
                f"No model server reachable at {self.config.base_url} ({error}).\n"
                "Start one locally (`ollama serve`) or point OpenContext at a provider:\n"
                "  set OPENCONTEXT_UPSTREAM, OPENCONTEXT_UPSTREAM_API_KEY and OPENCONTEXT_MODEL"
            ) from error

        if response.status_code != 200:
            raise LLMUnavailable(
                f"{self.config.base_url} returned {response.status_code}: {response.text[:300]}"
            )

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as error:
            raise LLMUnavailable(f"Unexpected response shape from upstream: {error}") from error

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream a completion, yielding text deltas as they arrive."""
        import json

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
        }

        try:
            with httpx.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.config.timeout,
            ) as response:
                if response.status_code != 200:
                    raise LLMUnavailable(
                        f"{self.config.base_url} returned {response.status_code}"
                    )
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        return
                    try:
                        delta = json.loads(body)["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError, ValueError):
                        continue
                    if delta:
                        yield delta
        except httpx.HTTPError as error:
            raise LLMUnavailable(
                f"No model server reachable at {self.config.base_url} ({error})."
            ) from error
