"""Optional FastAPI transport for the retrieval and answer services."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cpp_context_engine.api.analysis import (
    MAX_BUILD_VARIANTS,
    AnalysisQueryService,
    BuildName,
    CallRequest,
    CfgRequest,
    FlowRequest,
)
from cpp_context_engine.api.contracts import (
    AnswerRequest,
    AnswerResponse,
    QueryRequest,
    QueryResponse,
)
from cpp_context_engine.api.service import ContextRetrievalService, IterativeAnswerService
from cpp_context_engine.llm import LLMProviderError
from cpp_context_engine.retrieval import ContextBundle


class RetrieveBody(BaseModel):
    query: str = Field(min_length=1, max_length=2_048)
    max_context_tokens: int | None = Field(default=None, ge=1)
    builds: Annotated[list[BuildName] | None, Field(max_length=MAX_BUILD_VARIANTS)] = None


class ContextBody(RetrieveBody):
    max_results: int | None = Field(default=None, ge=1, le=100)


class AnswerBody(RetrieveBody):
    max_steps: int | None = Field(default=None, ge=1)


def create_app(
    *,
    retrieval_service: ContextRetrievalService,
    answer_service: IterativeAnswerService | None = None,
    analysis_service: AnalysisQueryService | None = None,
    scoped_query: Callable[[QueryRequest], QueryResponse] | None = None,
    scoped_answer: Callable[[AnswerRequest], AnswerResponse] | None = None,
) -> FastAPI:
    """Create an app around injected services; no global state or network setup."""

    app = FastAPI(title="C++ Context Engine", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/context")
    def context(body: ContextBody) -> dict[str, Any]:
        try:
            request = QueryRequest(
                body.query,
                body.max_context_tokens,
                tuple(body.builds) if body.builds is not None else None,
                body.max_results,
            )
            response = (
                scoped_query(request)
                if scoped_query is not None
                else retrieval_service.query(request)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialize_context(
            response.context,
            project_root=analysis_service.project_root if analysis_service else None,
        )

    @app.post("/v1/answer")
    def answer(body: AnswerBody) -> dict[str, Any]:
        if answer_service is None:
            raise HTTPException(status_code=503, detail="answer service is not configured")
        try:
            request = AnswerRequest(
                body.query,
                body.max_context_tokens,
                body.max_steps,
                tuple(body.builds) if body.builds is not None else None,
            )
            response = (
                scoped_answer(request)
                if scoped_answer is not None
                else answer_service.answer(request)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMProviderError as exc:
            # Provider errors can contain credentials, URLs, or absolute host paths.
            raise HTTPException(status_code=502, detail="configured LLM provider failed") from exc
        return {
            "answer": response.answer,
            "scope": {
                "kind": "union" if len(response.build_variants) > 1 else "single",
                "label": response.scope_label,
                "variants": list(response.build_variants),
            },
            "sources": [
                {
                    "symbol_id": source.symbol_id,
                    "qualified_name": source.qualified_name,
                    "path": _safe_path(
                        source.path,
                        analysis_service.project_root if analysis_service else None,
                    ),
                    "start_line": source.start_line,
                    "end_line": source.end_line,
                    "build_variant": source.build_variant,
                }
                for source in response.sources
            ],
            "steps": response.steps,
            "complete": response.complete,
            "diagnostics": list(response.diagnostics),
        }

    @app.get("/v1/builds")
    def builds() -> dict[str, Any]:
        service = _require_analysis(analysis_service)
        return service.list_builds().model_dump(mode="json")

    @app.post("/v1/cfg")
    def cfg(body: CfgRequest) -> dict[str, Any]:
        service = _require_analysis(analysis_service)
        try:
            return service.control_flow(body).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/flow")
    def flow(body: FlowRequest) -> dict[str, Any]:
        service = _require_analysis(analysis_service)
        try:
            return service.data_flow(body).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/calls")
    def calls(body: CallRequest) -> dict[str, Any]:
        service = _require_analysis(analysis_service)
        try:
            return service.calls(body).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def _require_analysis(service: AnalysisQueryService | None) -> AnalysisQueryService:
    if service is None:
        raise HTTPException(status_code=503, detail="analysis service is not configured")
    return service


def _serialize_context(
    bundle: ContextBundle, *, project_root: Path | None = None
) -> dict[str, Any]:
    return {
        "query": bundle.query,
        "items": [
            {
                "symbol_id": item.hit.symbol.id,
                "variant_id": item.hit.symbol.variant_id,
                "build_variant": item.hit.symbol.build_variant,
                "qualified_name": item.hit.symbol.qualified_name,
                "kind": item.hit.symbol.kind.value,
                "path": _safe_path(item.hit.symbol.span.path, project_root),
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
        "scope": {
            "kind": "union" if len(bundle.build_variants) > 1 else "single",
            "label": bundle.scope_label or "build:default",
            "variants": list(bundle.build_variants or ("default",)),
        },
    }


def _safe_path(path: Path, project_root: Path | None) -> str:
    if project_root is None:
        return path.as_posix() if not path.is_absolute() else "<absolute-path-redacted>"
    root = project_root.resolve(strict=False)
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    return (
        resolved.relative_to(root).as_posix()
        if resolved.is_relative_to(root)
        else "<outside-project>"
    )
