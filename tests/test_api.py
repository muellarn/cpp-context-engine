from __future__ import annotations

import json

from fastapi.testclient import TestClient
from test_service import RetrieverStub

from cpp_context_engine.api import ContextRetrievalService, IterativeAnswerService
from cpp_context_engine.api.http import create_app
from cpp_context_engine.llm import DeterministicFakeProvider
from cpp_context_engine.llm.providers import LLMProviderError


def test_context_endpoint_returns_provenance() -> None:
    retrieval = ContextRetrievalService(RetrieverStub(), default_max_context_tokens=500)
    client = TestClient(create_app(retrieval_service=retrieval))

    response = client.post("/v1/context", json={"query": "find parser", "max_context_tokens": 100})

    assert response.status_code == 200
    document = response.json()
    assert document["items"][0]["symbol_id"] == "parse"
    assert document["items"][0]["path"] == "net/parser.cpp"
    assert document["items"][0]["reason"] == "lexical rank 1"


def test_answer_endpoint_returns_structured_citations() -> None:
    retrieval = ContextRetrievalService(RetrieverStub(), default_max_context_tokens=500)
    llm = DeterministicFakeProvider(
        json.dumps(
            {
                "action": "answer",
                "answer": "It validates the packet.",
                "source_ids": ["parse"],
            }
        )
    )
    answer = IterativeAnswerService(retrieval, llm)
    client = TestClient(create_app(retrieval_service=retrieval, answer_service=answer))

    response = client.post("/v1/answer", json={"query": "How?", "max_steps": 2})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "It validates the packet.",
        "scope": {
            "kind": "single",
            "label": "build:default",
            "variants": ["default"],
        },
        "sources": [
            {
                "symbol_id": "parse",
                "qualified_name": "net::parse",
                "path": "net/parser.cpp",
                "start_line": 10,
                "end_line": 20,
                "build_variant": "default",
            }
        ],
        "steps": 1,
        "complete": True,
        "diagnostics": [],
    }


def test_answer_endpoint_reports_unconfigured_service() -> None:
    retrieval = ContextRetrievalService(RetrieverStub())
    client = TestClient(create_app(retrieval_service=retrieval))

    response = client.post("/v1/answer", json={"query": "How?"})

    assert response.status_code == 503


def test_answer_endpoint_does_not_expose_provider_error_details(tmp_path) -> None:
    retrieval = ContextRetrievalService(RetrieverStub())

    class FailingAnswerService:
        def answer(self, _request):
            raise LLMProviderError(f"secret-token at {tmp_path}/private")

    client = TestClient(
        create_app(retrieval_service=retrieval, answer_service=FailingAnswerService())
    )

    response = client.post("/v1/answer", json={"query": "How?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "configured LLM provider failed"}
    assert str(tmp_path) not in response.text
