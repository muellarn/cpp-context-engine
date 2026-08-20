"""Interfaces for independent lexical and semantic candidate generators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cpp_context_engine.models import SearchHit, SearchQuery


class LexicalSearch(Protocol):
    def search(self, query: SearchQuery) -> Sequence[SearchHit]:
        """Search exact names, signatures, documentation, and source text."""
        ...


class SymbolSearch(Protocol):
    def search(self, query: SearchQuery) -> Sequence[SearchHit]:
        """Search qualified names and compiler-resolved symbols."""
        ...


class VectorSearch(Protocol):
    def search(self, query: SearchQuery) -> Sequence[SearchHit]:
        """Search semantically similar symbols or source regions."""
        ...
