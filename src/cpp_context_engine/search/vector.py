"""A provider-neutral embedding adapter over the SQLite vector repository."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import islice
from pathlib import Path
from typing import Protocol

from cpp_context_engine.models import BuildScope, SearchHit, SearchQuery
from cpp_context_engine.storage.sqlite import EMBEDDING_BATCH_SIZE, SQLiteStore, _embedding_text


class EmbeddingProvider(Protocol):
    """Minimal interface implemented by local or hosted embedding models."""

    @property
    def model_id(self) -> str: ...

    @property
    def configuration_id(self) -> str: ...

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
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._store = store
        self._provider = provider
        self._project_root = project_root
        self._build_scope = build_scope or BuildScope.single()
        if max_text_chars <= 0:
            raise ValueError("max embedding text size must be positive")
        if batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        self._max_text_chars = max_text_chars
        self._batch_size = batch_size

    @property
    def _configuration_id(self) -> str:
        # Third-party providers written for the original protocol remain isolated
        # by their public model ID until they opt into the stronger identity.
        return getattr(self._provider, "configuration_id", self._provider.model_id)

    def index(self, symbol_ids: Iterable[str]) -> int:
        indexed = 0
        identifiers = iter(symbol_ids)
        with self._store.embedding_write_session(self._project_root):
            while chunk := tuple(islice(identifiers, self._batch_size)):
                symbols = self._store.get_symbols(
                    chunk, self._project_root, build_scope=self._build_scope
                )
                present = [symbol for symbol in symbols if symbol is not None]
                if not present:
                    continue
                records = tuple(
                    (
                        symbol.variant_id or symbol.id,
                        _embedding_text(symbol, self._max_text_chars),
                    )
                    for symbol in present
                )
                missing = self._store.attach_existing_embeddings(
                    records,
                    self._provider.model_id,
                    self._project_root,
                    build_scope=self._build_scope,
                    configuration_id=self._configuration_id,
                )
                unique_texts = tuple(dict.fromkeys(text for _, text in missing))
                if unique_texts:
                    vectors = self._provider.embed(unique_texts)
                    if len(vectors) != len(unique_texts):
                        raise ValueError(
                            "embedding provider returned a different number of vectors than texts"
                        )
                    by_text = dict(zip(unique_texts, vectors, strict=True))
                    self._store.put_content_embeddings(
                        ((variant_id, text, by_text[text]) for variant_id, text in missing),
                        self._provider.model_id,
                        self._project_root,
                        build_scope=self._build_scope,
                        configuration_id=self._configuration_id,
                    )
                indexed += len(present)
        return indexed

    def index_missing(self) -> int:
        return self.index(
            variant_id
            for batch in self._store.iter_missing_embedding_variant_id_batches(
                self._provider.model_id,
                self._project_root,
                build_scope=self._build_scope,
                configuration_id=self._configuration_id,
                batch_size=self._batch_size,
            )
            for variant_id in batch
        )

    def search(self, query: SearchQuery) -> Sequence[SearchHit]:
        vectors = self._provider.embed([query.text])
        if len(vectors) != 1:
            raise ValueError("embedding provider must return exactly one vector for one query")
        return self._store.search_vector(
            vectors[0],
            model=self._provider.model_id,
            configuration_id=self._configuration_id,
            limit=query.limit,
            project_root=self._project_root,
            build_scope=self._build_scope,
        )
