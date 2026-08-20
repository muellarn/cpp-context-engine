"""Interfaces for candidate fusion, graph expansion, and context packing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cpp_context_engine.models import GraphRelation, SearchHit


@dataclass(frozen=True, slots=True)
class ContextPathStep:
    """One validated graph hop explaining why context was selected."""

    source_id: str
    target_id: str
    relation: GraphRelation


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One source excerpt together with provenance used for citations."""

    hit: SearchHit
    source_text: str
    reason: str
    path: tuple[ContextPathStep, ...] = ()

    @property
    def path_name(self) -> Path:
        return self.hit.symbol.span.path


@dataclass(frozen=True, slots=True)
class ContextBundle:
    query: str
    hits: tuple[SearchHit, ...]
    rendered_context: str
    estimated_tokens: int
    items: tuple[ContextItem, ...] = ()
    diagnostics: tuple[str, ...] = ()
    truncated: bool = False


class Retriever(Protocol):
    def retrieve(self, query: str, *, max_tokens: int) -> ContextBundle:
        """Return connected, ranked source context within a hard token budget."""
        ...
