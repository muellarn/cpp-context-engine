from __future__ import annotations

import io
import json
from urllib.error import URLError
from urllib.request import Request

import pytest

from cpp_context_engine.search.embeddings import (
    DeterministicLocalEmbeddingProvider,
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingProvider,
)


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self._body.read(size)


def test_local_embeddings_are_deterministic_and_identifier_sensitive() -> None:
    provider = DeterministicLocalEmbeddingProvider(64)

    first, second, different = provider.embed(
        ["PacketParser validateHeader", "PacketParser validateHeader", "Database executeSql"]
    )

    assert first == second
    assert first != different
    assert len(first) == 64
    assert provider.model_id == "local-feature-hash-v1-64"


def test_openai_embedding_provider_orders_results_and_hides_secret() -> None:
    captured: list[Request] = []

    def opener(request: Request, *, timeout: float):
        captured.append(request)
        assert timeout == 4
        return _Response(
            {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}
        )

    provider = OpenAICompatibleEmbeddingProvider(
        "http://localhost:11434/v1", "code-model", "top-secret", 4, _opener=opener
    )

    assert provider.embed(["one", "two"]) == ((1.0, 0.0), (0.0, 1.0))
    assert captured[0].full_url == "http://localhost:11434/v1/embeddings"
    assert captured[0].get_header("Authorization") == "Bearer top-secret"
    assert "top-secret" not in repr(provider)


def test_openai_embedding_provider_sanitizes_network_failure() -> None:
    def opener(_request: Request, *, timeout: float):
        raise URLError("contains-sensitive-upstream-details")

    provider = OpenAICompatibleEmbeddingProvider(
        "https://example.invalid/v1", "model", "secret", _opener=opener
    )

    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed(["text"])

    assert "sensitive" not in str(captured.value)
    assert "secret" not in str(captured.value)
