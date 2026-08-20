"""Interfaces for persisting and traversing code relationships."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from cpp_context_engine.models import GraphEdge, GraphRelation


class CodeGraph(Protocol):
    def put_edges(self, edges: Iterable[GraphEdge]) -> None:
        """Insert or replace normalized graph edges."""
        ...

    def neighbors(
        self,
        symbol_id: str,
        *,
        relations: frozenset[GraphRelation] | None = None,
        depth: int = 1,
    ) -> Sequence[GraphEdge]:
        """Return bounded neighboring edges around one symbol."""
        ...
