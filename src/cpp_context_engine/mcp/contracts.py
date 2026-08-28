"""Strict MCP tool inputs and structured outputs."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from cpp_context_engine.models import (
    CallTargetCertainty,
    GraphDirection,
    GraphRelation,
    IndexProfile,
    SymbolKind,
)

MAX_QUERY_CHARS = 2_048
MAX_SYMBOL_ID_CHARS = 2_048
MAX_SEARCH_RESULTS = 20
MAX_GRAPH_DEPTH = 3
MAX_GRAPH_FANOUT = 25
MAX_GRAPH_RESULTS = 100
MAX_CONTEXT_TOKENS = 32_000
MIN_CONTEXT_TOKENS = 256
MAX_ANSWER_STEPS = 6
MAX_SOURCE_CHARS = 50_000
MIN_SOURCE_CHARS = 256
MAX_DIAGNOSTICS = 20
MAX_ANSWER_CHARS = 50_000
MAX_BUILD_VARIANTS = 16
MAX_BUILD_NAME_CHARS = 128
MAX_ANALYSIS_GRAPHS = 20
MAX_ANALYSIS_BLOCKS = 500
MAX_ANALYSIS_ITEMS = 2_000

QueryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)
]
SymbolId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_SYMBOL_ID_CHARS)
]
SearchResultLimit = Annotated[int, Field(ge=1, le=MAX_SEARCH_RESULTS)]
GraphDepth = Annotated[int, Field(ge=1, le=MAX_GRAPH_DEPTH)]
GraphFanout = Annotated[int, Field(ge=1, le=MAX_GRAPH_FANOUT)]
GraphResultLimit = Annotated[int, Field(ge=1, le=MAX_GRAPH_RESULTS)]
ContextTokens = Annotated[int, Field(ge=MIN_CONTEXT_TOKENS, le=MAX_CONTEXT_TOKENS)]
AnswerSteps = Annotated[int, Field(ge=1, le=MAX_ANSWER_STEPS)]
SourceChars = Annotated[int, Field(ge=MIN_SOURCE_CHARS, le=MAX_SOURCE_CHARS)]
Relations = Annotated[list[GraphRelation] | None, Field(max_length=len(GraphRelation))]
Builds = Annotated[
    list[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_BUILD_NAME_CHARS),
        ]
    ]
    | None,
    Field(max_length=MAX_BUILD_VARIANTS),
]
AnalysisGraphs = Annotated[int, Field(ge=1, le=MAX_ANALYSIS_GRAPHS)]
AnalysisBlocks = Annotated[int, Field(ge=1, le=MAX_ANALYSIS_BLOCKS)]
AnalysisItems = Annotated[int, Field(ge=1, le=MAX_ANALYSIS_ITEMS)]


class ToolOutput(BaseModel):
    """Forbid accidental, undocumented fields in every tool result."""

    model_config = ConfigDict(extra="forbid")


class SourceLocation(ToolOutput):
    """A project-relative, POSIX source location with one-based lines."""

    path: str = Field(description="Project-relative POSIX path; never an absolute host path.")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class SymbolReference(ToolOutput):
    symbol_id: str
    variant_id: str = ""
    build_variant: str = "default"
    qualified_name: str
    kind: SymbolKind
    signature: str
    location: SourceLocation


class GraphPathStepResult(ToolOutput):
    source_id: str
    target_id: str
    relation: GraphRelation


class SearchCodeItem(ToolOutput):
    symbol: SymbolReference
    source_text: str
    score: float
    reason: str
    graph_path: Annotated[list[GraphPathStepResult], Field(max_length=MAX_GRAPH_DEPTH)]


class SearchCodeResult(ToolOutput):
    query: str
    items: Annotated[list[SearchCodeItem], Field(max_length=MAX_SEARCH_RESULTS)]
    estimated_tokens: int = Field(ge=0)
    truncated: bool
    diagnostics: Annotated[list[str], Field(max_length=MAX_DIAGNOSTICS)]
    scope_kind: str = "single"
    scope_label: str = "build:default"
    scope_variants: list[str] = Field(default_factory=lambda: ["default"])


class ReadSymbolResult(ToolOutput):
    symbol: SymbolReference
    source_text: str
    truncated: bool
    scope_kind: str = "single"
    scope_label: str = "build:default"
    scope_variants: list[str] = Field(default_factory=lambda: ["default"])


class GraphEdgeResult(ToolOutput):
    edge_id: str = ""
    build_variant: str = "default"
    source: SymbolReference
    target: SymbolReference
    relation: GraphRelation
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    callsite_id: str | None = None
    certainty: CallTargetCertainty | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_reason: str | None = None
    derivation: str | None = None
    target_set_complete: bool | None = None


class GraphResult(ToolOutput):
    symbol: SymbolReference
    direction: GraphDirection
    depth: int = Field(ge=1, le=MAX_GRAPH_DEPTH)
    edges: Annotated[list[GraphEdgeResult], Field(max_length=MAX_GRAPH_RESULTS)]
    truncated: bool
    scope_kind: str = "single"
    scope_label: str = "build:default"
    scope_variants: list[str] = Field(default_factory=lambda: ["default"])


class IndexProjectResult(ToolOutput):
    indexed_translation_units: int = Field(ge=0)
    skipped_translation_units: int = Field(ge=0)
    removed_translation_units: int = Field(ge=0)
    indexed_symbols: int = Field(ge=0)
    indexed_occurrences: int = Field(ge=0)
    indexed_edges: int = Field(ge=0)
    embedded_symbols: int = Field(ge=0)
    embedding_model: str
    analysis_backend: str
    advanced_facts_complete: bool
    analyzer_capabilities: list[str] = Field(default_factory=list)
    index_profile: IndexProfile = IndexProfile.FULL


class AnswerSource(ToolOutput):
    symbol_id: str
    qualified_name: str
    build_variant: str = "default"
    location: SourceLocation


class AskCodeResult(ToolOutput):
    answer: str = Field(max_length=MAX_ANSWER_CHARS)
    complete: bool
    steps: int = Field(ge=1, le=MAX_ANSWER_STEPS)
    sources: Annotated[list[AnswerSource], Field(max_length=MAX_SEARCH_RESULTS)]
    diagnostics: Annotated[list[str], Field(max_length=MAX_DIAGNOSTICS)]
    scope_kind: str = "single"
    scope_label: str = "build:default"
    scope_variants: list[str] = Field(default_factory=lambda: ["default"])
