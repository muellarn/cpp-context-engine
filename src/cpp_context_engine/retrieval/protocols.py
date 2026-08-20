"""Interfaces for candidate fusion, graph expansion, and context packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cpp_context_engine.models import SearchHit


@dataclass(frozen=True, slots=True)
class ContextBundle:
    query: str
    hits: tuple[SearchHit, ...]
    rendered_context: str
    estimated_tokens: int


class Retriever(Protocol):
    def retrieve(self, query: str, *, max_tokens: int) -> ContextBundle:
        """Return connected, ranked source context within a hard token budget."""
        ...
