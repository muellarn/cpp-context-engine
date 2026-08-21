from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpp_context_engine.api import (
    AnswerRequest,
    ContextRetrievalService,
    IterativeAnswerService,
    QueryRequest,
)
from cpp_context_engine.llm import DeterministicFakeProvider
from cpp_context_engine.models import CodeSymbol, SearchHit, SourceSpan, SymbolKind
from cpp_context_engine.retrieval import ContextBundle, ContextItem


def bundle(query: str, symbol_id: str = "parse") -> ContextBundle:
    symbol = CodeSymbol(
        id=symbol_id,
        qualified_name=f"net::{symbol_id}",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(Path("net/parser.cpp"), 10, 20),
    )
    hit = SearchHit(symbol, 1.0, "hybrid")
    item = ContextItem(hit, "return validate(packet);", "lexical rank 1")
    rendered = f"### {symbol.qualified_name}\n```cpp\n{item.source_text}\n```"
    return ContextBundle(query, (hit,), rendered, 20, (item,))


class RetrieverStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, max_tokens: int) -> ContextBundle:
        self.calls.append((query, max_tokens))
        return bundle(query, "validate" if "validation" in query else "parse")


def test_context_service_validates_hard_token_limit() -> None:
    service = ContextRetrievalService(
        RetrieverStub(), default_max_context_tokens=50, max_context_tokens=100
    )

    with pytest.raises(ValueError, match=r"\[1, 100\]"):
        service.query(QueryRequest("find parser", 101))
    with pytest.raises(ValueError, match=r"\[1, 100\]"):
        service.query(QueryRequest("find parser", 0))
    with pytest.raises(ValueError, match=r"max_results must be in \[1, 100\]"):
        service.query(QueryRequest("find parser", max_results=0))


def test_iterative_answer_searches_again_and_validates_citations() -> None:
    retriever = RetrieverStub()
    retrieval = ContextRetrievalService(retriever, default_max_context_tokens=500)
    llm = DeterministicFakeProvider(
        [
            json.dumps({"action": "search", "query": "packet validation", "source_ids": []}),
            json.dumps(
                {
                    "action": "answer",
                    "answer": "The packet is validated before parsing.",
                    "source_ids": ["validate", "invented"],
                }
            ),
        ]
    )
    service = IterativeAnswerService(retrieval, llm)

    response = service.answer(AnswerRequest("How is the packet parsed?", max_steps=3))

    assert response.answer == "The packet is validated before parsing."
    assert response.complete is True
    assert response.steps == 2
    assert [source.symbol_id for source in response.sources] == ["validate"]
    assert retriever.calls == [
        ("How is the packet parsed?", 500),
        ("packet validation", 500),
    ]


def test_iterative_answer_stops_at_hard_step_limit() -> None:
    retrieval = ContextRetrievalService(RetrieverStub())
    llm = DeterministicFakeProvider(
        json.dumps({"action": "search", "query": "keep searching", "source_ids": []})
    )
    service = IterativeAnswerService(retrieval, llm, default_max_steps=1)

    response = service.answer(AnswerRequest("question"))

    assert response.complete is False
    assert response.steps == 1
    assert len(llm.calls) == 1
    assert "step limit" in response.diagnostics[-1]
