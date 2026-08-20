"""Request, response, and service boundaries for future HTTP or RPC adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cpp_context_engine.retrieval import ContextBundle


@dataclass(frozen=True, slots=True)
class QueryRequest:
    query: str
    max_context_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class QueryResponse:
    context: ContextBundle


class RetrievalService(Protocol):
    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute one bounded code-context query."""
        ...
