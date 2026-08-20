"""Request, response, and service boundaries for future HTTP or RPC adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    query: str
    max_context_tokens: int | None = None
    max_steps: int | None = None


@dataclass(frozen=True, slots=True)
class SourceCitation:
    symbol_id: str
    qualified_name: str
    path: Path
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class AnswerResponse:
    answer: str
    sources: tuple[SourceCitation, ...]
    steps: int
    complete: bool
    diagnostics: tuple[str, ...] = ()


class AnswerService(Protocol):
    def answer(self, request: AnswerRequest) -> AnswerResponse:
        """Answer one question through a hard-bounded retrieval loop."""
        ...
