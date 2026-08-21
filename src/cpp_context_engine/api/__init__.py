"""Transport-neutral API contracts and application services."""

from cpp_context_engine.api.analysis import (
    AnalysisQueryService,
    BuildListResult,
    CallGraphResult,
    CallRequest,
    CfgRequest,
    ControlFlowResult,
    DataFlowResult,
    FlowRequest,
)
from cpp_context_engine.api.contracts import (
    AnswerRequest,
    AnswerResponse,
    AnswerService,
    QueryRequest,
    QueryResponse,
    RetrievalService,
    SourceCitation,
)
from cpp_context_engine.api.service import ContextRetrievalService, IterativeAnswerService

__all__ = [
    "AnalysisQueryService",
    "AnswerRequest",
    "AnswerResponse",
    "AnswerService",
    "BuildListResult",
    "CallGraphResult",
    "CallRequest",
    "CfgRequest",
    "ContextRetrievalService",
    "ControlFlowResult",
    "DataFlowResult",
    "FlowRequest",
    "IterativeAnswerService",
    "QueryRequest",
    "QueryResponse",
    "RetrievalService",
    "SourceCitation",
]
