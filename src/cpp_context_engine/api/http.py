"""Optional FastAPI transport for the retrieval and answer services."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cpp_context_engine.api.contracts import AnswerRequest, QueryRequest
from cpp_context_engine.api.service import ContextRetrievalService, IterativeAnswerService
from cpp_context_engine.llm import LLMProviderError
from cpp_context_engine.retrieval import ContextBundle


class RetrieveBody(BaseModel):
    query: str = Field(min_length=1)
    max_context_tokens: int | None = Field(default=None, ge=1)


class AnswerBody(RetrieveBody):
    max_steps: int | None = Field(default=None, ge=1)


def create_app(
    *,
    retrieval_service: ContextRetrievalService,
    answer_service: IterativeAnswerService | None = None,
) -> FastAPI:
    """Create an app around injected services; no global state or network setup."""

    app = FastAPI(title="C++ Context Engine", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/context")
    def context(body: RetrieveBody) -> dict[str, Any]:
        try:
            response = retrieval_service.query(QueryRequest(body.query, body.max_context_tokens))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialize_context(response.context)

    @app.post("/v1/answer")
    def answer(body: AnswerBody) -> dict[str, Any]:
        if answer_service is None:
            raise HTTPException(status_code=503, detail="answer service is not configured")
        try:
            response = answer_service.answer(
                AnswerRequest(body.query, body.max_context_tokens, body.max_steps)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "answer": response.answer,
            "sources": [
                {
                    "symbol_id": source.symbol_id,
                    "qualified_name": source.qualified_name,
                    "path": str(source.path),
                    "start_line": source.start_line,
                    "end_line": source.end_line,
                }
                for source in response.sources
            ],
            "steps": response.steps,
            "complete": response.complete,
            "diagnostics": list(response.diagnostics),
        }

    return app


def _serialize_context(bundle: ContextBundle) -> dict[str, Any]:
    return {
        "query": bundle.query,
        "items": [
            {
                "symbol_id": item.hit.symbol.id,
                "variant_id": item.hit.symbol.variant_id,
                "build_variant": item.hit.symbol.build_variant,
                "qualified_name": item.hit.symbol.qualified_name,
                "kind": item.hit.symbol.kind.value,
                "path": str(item.hit.symbol.span.path),
                "start_line": item.hit.symbol.span.start_line,
                "end_line": item.hit.symbol.span.end_line,
                "score": item.hit.score,
                "source": item.hit.source,
                "source_text": item.source_text,
                "reason": item.reason,
                "graph_path": [
                    {
                        "source_id": step.source_id,
                        "target_id": step.target_id,
                        "relation": step.relation.value,
                    }
                    for step in item.path
                ],
            }
            for item in bundle.items
        ],
        "rendered_context": bundle.rendered_context,
        "estimated_tokens": bundle.estimated_tokens,
        "diagnostics": list(bundle.diagnostics),
        "truncated": bundle.truncated,
    }
