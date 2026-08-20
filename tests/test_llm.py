from __future__ import annotations

import json
from typing import Any
from urllib.request import Request

import pytest

from cpp_context_engine.llm import (
    DeterministicFakeProvider,
    LLMProviderError,
    OpenAICompatibleProvider,
    ToolDefinition,
)


class ResponseStub:
    def __init__(self, document: dict[str, Any]) -> None:
        self.body = json.dumps(document).encode()

    def __enter__(self) -> ResponseStub:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


def test_openai_compatible_provider_sends_one_bounded_request() -> None:
    captured: dict[str, Any] = {}

    def opener(request: Request, *, timeout: float) -> ResponseStub:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data or b"{}")
        captured["timeout"] = timeout
        return ResponseStub({"choices": [{"message": {"content": "answer"}}]})

    provider = OpenAICompatibleProvider(
        "https://llm.example/v1",
        "test-model",
        api_key="sensitive-key",
        timeout_seconds=2.5,
        _opener=opener,
    )
    tool = ToolDefinition("lookup", "Find a symbol", {"type": "object"})

    assert provider.complete("question", tools=[tool]) == "answer"
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer sensitive-key"
    assert captured["timeout"] == 2.5
    assert captured["payload"]["tools"][0]["function"]["name"] == "lookup"
    assert "sensitive-key" not in repr(provider)


def test_provider_timeout_is_sanitized_and_is_not_retried() -> None:
    calls = 0

    def opener(request: Request, *, timeout: float) -> ResponseStub:
        nonlocal calls
        calls += 1
        raise TimeoutError("sensitive-key appeared in a low-level error")

    provider = OpenAICompatibleProvider(
        "http://localhost:1234/v1",
        "local",
        api_key="sensitive-key",
        _opener=opener,
    )

    with pytest.raises(LLMProviderError, match="unavailable or timed out") as caught:
        provider.complete("question")

    assert calls == 1
    assert "sensitive-key" not in str(caught.value)


def test_provider_rejects_invalid_response_shape() -> None:
    provider = OpenAICompatibleProvider(
        "http://localhost:1234/v1",
        "local",
        _opener=lambda request, timeout: ResponseStub({"unexpected": True}),
    )

    with pytest.raises(LLMProviderError, match="invalid response"):
        provider.complete("question")


def test_deterministic_fake_records_calls_and_holds_last_response() -> None:
    provider = DeterministicFakeProvider(["first", "second"])

    assert provider.complete("one") == "first"
    assert provider.complete("two") == "second"
    assert provider.complete("three") == "second"
    assert [prompt for prompt, _ in provider.calls] == ["one", "two", "three"]
