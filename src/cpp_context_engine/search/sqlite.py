"""Search adapters backed by one project-scoped SQLite store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cpp_context_engine.models import SearchHit, SearchQuery
from cpp_context_engine.storage.sqlite import SQLiteStore


@dataclass(frozen=True, slots=True)
class SQLiteLexicalSearch:
    store: SQLiteStore
    project_root: Path

    def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        return tuple(self.store.search(query, self.project_root))


@dataclass(frozen=True, slots=True)
class SQLiteSymbolSearch:
    store: SQLiteStore
    project_root: Path

    def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        return tuple(self.store.search_symbols(query, self.project_root))
