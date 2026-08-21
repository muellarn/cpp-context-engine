"""A provider-neutral embedding adapter over the SQLite vector repository."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from cpp_context_engine.models import BuildScope, SearchHit, SearchQuery
from cpp_context_engine.storage.sqlite import SQLiteStore


class EmbeddingProvider(Protocol):
    """Minimal interface implemented by local or hosted embedding models."""

    @property
    def model_id(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SQLiteVectorSearch:
    def __init__(
        self,
        store: SQLiteStore,
        provider: EmbeddingProvider,
        *,
        project_root: Path | None = None,
        max_text_chars: int = 32_000,
        build_scope: BuildScope | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._project_root = project_root
        self._build_scope = build_scope or BuildScope.single()
        if max_text_chars <= 0:
            raise ValueError("max embedding text size must be positive")
        self._max_text_chars = max_text_chars

    def index(self, symbol_ids: Sequence[str]) -> None:
        symbols = self._store.get_symbols(
            symbol_ids, self._project_root, build_scope=self._build_scope
        )
        present = [symbol for symbol in symbols if symbol is not None]
        if not present:
            return
        texts = [
            "\n".join(
                part
                for part in (
                    symbol.qualified_name,
                    symbol.signature,
                    symbol.documentation,
                    symbol.source_text,
                )
                if part
            )[: self._max_text_chars]
            for symbol in present
        ]
        vectors = self._provider.embed(texts)
        if len(vectors) != len(present):
            raise ValueError("embedding provider returned a different number of vectors than texts")
        self._store.put_embeddings(
            (
                (symbol.variant_id or symbol.id, vector)
                for symbol, vector in zip(present, vectors, strict=True)
            ),
            self._provider.model_id,
            self._project_root,
            build_scope=self._build_scope,
        )

    def search(self, query: SearchQuery) -> Sequence[SearchHit]:
        vectors = self._provider.embed([query.text])
        if len(vectors) != 1:
            raise ValueError("embedding provider must return exactly one vector for one query")
        return self._store.search_vector(
            vectors[0],
            model=self._provider.model_id,
            limit=query.limit,
            project_root=self._project_root,
            build_scope=self._build_scope,
        )
