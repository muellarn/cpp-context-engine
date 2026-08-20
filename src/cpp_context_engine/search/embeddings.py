"""Deterministic local and OpenAI-compatible embedding providers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingProviderError(RuntimeError):
    """A provider failure whose message is safe to show to a CLI or API caller."""


@dataclass(frozen=True, slots=True)
class DeterministicLocalEmbeddingProvider:
    """Network-free feature hashing for repeatable lexical-semantic fallback search.

    This intentionally small model captures identifier and token overlap. It is useful
    offline but does not have the semantic understanding of a trained embedding model.
    """

    dimensions: int = 384

    def __post_init__(self) -> None:
        if self.dimensions < 16:
            raise ValueError("local embedding dimensions must be at least 16")

    @property
    def model_id(self) -> str:
        return f"local-feature-hash-v1-{self.dimensions}"

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimensions
        tokens = _tokens(text)
        if not tokens:
            tokens = ("<empty>",)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            weight = 1.0 / math.sqrt(max(1, len(token)))
            for offset in (0, 4, 8, 12):
                bucket = int.from_bytes(digest[offset : offset + 4], "little") % self.dimensions
                sign = 1.0 if digest[offset] & 1 else -1.0
                values[bucket] += sign * weight
        return tuple(values)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEmbeddingProvider:
    """Call an OpenAI-compatible ``/embeddings`` endpoint without hidden retries."""

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0
    max_response_bytes: int = 32 * 1024 * 1024
    batch_size: int = 64
    _opener: Callable[..., Any] = field(default=urlopen, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("embedding base URL must use http or https")
        if not self.model.strip():
            raise ValueError("embedding model must not be empty")
        if self.timeout_seconds <= 0 or self.max_response_bytes <= 0 or self.batch_size <= 0:
            raise ValueError("embedding timeout, response limit, and batch size must be positive")

    @property
    def model_id(self) -> str:
        return f"openai-compatible:{self.model}"

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._request_batch(tuple(texts[start : start + self.batch_size])))
        return tuple(vectors)

    def _request_batch(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        payload = {"model": self.model, "input": list(texts)}
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
            raise EmbeddingProviderError(f"embedding endpoint returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise EmbeddingProviderError(
                "embedding endpoint was unavailable or timed out"
            ) from None
        if len(raw) > self.max_response_bytes:
            raise EmbeddingProviderError("embedding response exceeded the configured size limit")
        try:
            document = json.loads(raw)
            data = document["data"]
            # Duplicate or missing indexes silently attach vectors to the wrong symbols.
            indexes = [item["index"] for item in data]
            if any(type(index) is not int for index in indexes) or sorted(indexes) != list(
                range(len(texts))
            ):
                raise ValueError("invalid embedding indexes")
            indexed = sorted(data, key=lambda item: item["index"])
            vectors = tuple(tuple(float(value) for value in item["embedding"]) for item in indexed)
        except (AttributeError, JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError("embedding endpoint returned an invalid response") from exc
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise EmbeddingProviderError("embedding endpoint returned the wrong number of vectors")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or any(
            not math.isfinite(value) for vector in vectors for value in vector
        ):
            raise EmbeddingProviderError("embedding endpoint returned invalid vectors")
        return vectors

    def _endpoint_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/embeddings") else f"{base}/embeddings"


def _tokens(text: str) -> tuple[str, ...]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", expanded.casefold())
    features: list[str] = []
    for word in words:
        features.append(word)
        features.extend(word[index : index + 3] for index in range(max(0, len(word) - 2)))
    return tuple(features)
