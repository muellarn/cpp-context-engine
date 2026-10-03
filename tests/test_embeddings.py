from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from collections import Counter
from urllib.error import URLError
from urllib.request import Request

import pytest

from cpp_context_engine.search import embeddings
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


@pytest.fixture
def empty_local_token_cache():
    cached = embeddings._cached_token_contributions  # noqa: SLF001
    cached.cache_clear()
    yield
    cached.cache_clear()


@pytest.mark.usefixtures("empty_local_token_cache")
def test_local_embeddings_hash_repeated_short_features_once(monkeypatch) -> None:
    calls: Counter[bytes] = Counter()
    original = embeddings.hashlib.blake2b

    def counted(data: bytes, **kwargs):
        calls[data] += 1
        return original(data, **kwargs)

    monkeypatch.setattr(embeddings.hashlib, "blake2b", counted)
    provider = DeterministicLocalEmbeddingProvider(32)
    provider.embed(["cache_112_probe repeat repeat", "second repeat cache_112_probe"])

    assert calls
    assert max(calls.values()) == 1


def _direct_local_vector(text: str, dimensions: int) -> bytes:
    values = [0.0] * dimensions
    for token in embeddings._tokens(text) or ("<empty>",):  # noqa: SLF001
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        weight = 1.0 / math.sqrt(max(1, len(token)))
        for offset in (0, 4, 8, 12):
            bucket = int.from_bytes(digest[offset : offset + 4], "little") % dimensions
            sign = 1.0 if digest[offset] & 1 else -1.0
            values[bucket] += sign * weight
    return struct.pack(f"<{dimensions}d", *values)


@pytest.mark.parametrize("dimensions", (16, 32, 64, 384))
def test_local_cached_contributions_preserve_every_vector_bit(dimensions: int) -> None:
    texts = (
        "",
        ";::{} ++",
        "PacketParser parseHTTP2Header std::vector<int>",
        "Straße MÜNCHEN naïve 東京 HTTP2",
        "001 12345 _identifier identifier2 identifier2",
        "same shared repeated_tokens " * 40,
        "a" * 64,
        "z" * 65,
        "long_identifier_" * 100,
    )
    expected = tuple(_direct_local_vector(text, dimensions) for text in texts)
    for _ in range(2):
        provider = DeterministicLocalEmbeddingProvider(dimensions)
        actual = tuple(struct.pack(f"<{dimensions}d", *v) for v in provider.embed(texts))
        assert actual == expected
        assert provider.configuration_id == f"local-feature-hash-v1-{dimensions}"


def test_local_token_cache_is_bounded_and_recomputes_evicted_entries() -> None:
    cached = embeddings._cached_token_contributions  # noqa: SLF001
    cached.cache_clear()
    try:
        first = cached("eviction_probe_112", 32)
        for number in range(4096):
            cached(f"cache_entry_{number}", 32)
        info = cached.cache_info()
        assert info.maxsize == info.currsize == 4096
        assert cached("eviction_probe_112", 32) == first
        assert cached.cache_info().misses == info.misses + 1
    finally:
        cached.cache_clear()


@pytest.mark.usefixtures("empty_local_token_cache")
def test_local_token_cache_does_not_retain_oversized_tokens(monkeypatch) -> None:
    calls: Counter[bytes] = Counter()
    original = embeddings.hashlib.blake2b

    def counted(data: bytes, **kwargs):
        calls[data] += 1
        return original(data, **kwargs)

    monkeypatch.setattr(embeddings.hashlib, "blake2b", counted)
    short = "q" * 64
    long = "w" * 65
    provider = DeterministicLocalEmbeddingProvider(32)
    provider.embed([f"{short} {long}", f"{short} {long}"])

    assert calls[short.encode()] == 1
    assert calls[long.encode()] == 2
    assert calls[b"qqq"] == calls[b"www"] == 1


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


def test_openai_embedding_configuration_identity_includes_endpoint_but_not_secret() -> None:
    first = OpenAICompatibleEmbeddingProvider(
        "https://first.invalid/v1", "code-model", "first-secret"
    )
    second = OpenAICompatibleEmbeddingProvider(
        "https://second.invalid/v1", "code-model", "second-secret"
    )
    rotated_secret = OpenAICompatibleEmbeddingProvider(
        "https://first.invalid/v1", "code-model", "rotated-secret"
    )

    assert first.model_id == second.model_id
    assert first.configuration_id != second.configuration_id
    assert first.configuration_id == rotated_secret.configuration_id
    assert "secret" not in first.configuration_id


def test_openai_embedding_provider_rejects_duplicate_response_indexes() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        "http://localhost:11434/v1",
        "code-model",
        _opener=lambda _request, timeout: _Response(
            {
                "data": [
                    {"index": 0, "embedding": [1, 0]},
                    {"index": 0, "embedding": [0, 1]},
                ]
            }
        ),
    )

    with pytest.raises(EmbeddingProviderError, match="invalid response"):
        provider.embed(["one", "two"])


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
