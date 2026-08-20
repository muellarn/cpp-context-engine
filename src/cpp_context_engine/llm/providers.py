"""LLM providers with explicit network boundaries and no implicit retries."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cpp_context_engine.llm.protocols import ToolDefinition


class LLMProviderError(RuntimeError):
    """A sanitized provider failure safe to return through a service boundary."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleProvider:
    """Call an OpenAI-compatible ``/chat/completions`` HTTP endpoint once."""

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0
    max_response_bytes: int = 10 * 1024 * 1024
    _opener: Callable[..., Any] = field(default=urlopen, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout_seconds <= 0 or self.max_response_bytes <= 0:
            raise ValueError("timeout and response limit must be positive")

    def complete(self, prompt: str, *, tools: Sequence[ToolDefinition] = ()) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.input_schema),
                    },
                }
                for tool in tools
            ]

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self._endpoint_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            raise LLMProviderError(f"LLM endpoint returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise LLMProviderError("LLM endpoint was unavailable or timed out") from None

        if len(raw) > self.max_response_bytes:
            raise LLMProviderError("LLM response exceeded the configured size limit")
        try:
            document = json.loads(raw)
            content = document["choices"][0]["message"]["content"]
        except (JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM endpoint returned an invalid response") from exc

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "text"
            ]
            if text_parts:
                return "".join(text_parts)
        raise LLMProviderError("LLM endpoint response contained no text")

    def _endpoint_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


@dataclass(slots=True)
class DeterministicFakeProvider:
    """Return predefined responses for repeatable, network-free service tests."""

    responses: tuple[str, ...]
    calls: list[tuple[str, tuple[ToolDefinition, ...]]] = field(default_factory=list, init=False)
    _cursor: int = field(default=0, init=False)

    def __init__(self, responses: str | Sequence[str]) -> None:
        normalized = (responses,) if isinstance(responses, str) else tuple(responses)
        if not normalized:
            raise ValueError("at least one fake response is required")
        self.responses = normalized
        self.calls = []
        self._cursor = 0

    def complete(self, prompt: str, *, tools: Sequence[ToolDefinition] = ()) -> str:
        self.calls.append((prompt, tuple(tools)))
        index = min(self._cursor, len(self.responses) - 1)
        self._cursor += 1
        return self.responses[index]
