"""Stable, bounded contracts and queries for compiler-derived analysis evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from cpp_context_engine.models import (
    BuildScope,
    CallTargetCertainty,
    DataFlowCertainty,
    DataFlowRelation,
    GraphDirection,
    SourceSpan,
)
from cpp_context_engine.storage import SQLiteStore

MAX_BUILD_VARIANTS = 16
MAX_BUILD_NAME_CHARS = 128
MAX_ID_CHARS = 2_048
MAX_CFG_GRAPHS = 20
MAX_CFG_BLOCKS = 500
MAX_CFG_ELEMENTS = 2_000
MAX_CFG_EDGES = 2_000
MAX_FLOW_ANALYSES = 20
MAX_FLOW_LOCATIONS = 1_000
MAX_FLOW_ACCESSES = 2_000
MAX_FLOW_EVIDENCE = 2_000
MAX_CALL_RESULTS = 100

Identifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_ID_CHARS)
]
BuildName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_BUILD_NAME_CHARS)
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScopeResult(Contract):
    variants: Annotated[list[str], Field(min_length=1, max_length=MAX_BUILD_VARIANTS)]
    kind: Literal["single", "union"]
    label: str


class SourceLocation(Contract):
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_column: int = Field(ge=1)
    end_column: int = Field(ge=1)


class Provenance(Contract):
    build_variant: str
    build_configuration_id: str
    translation_unit_id: str


class BuildInfo(Contract):
    name: str
    target: str
    platform: str
    reindex_required: bool


class BuildListResult(Contract):
    builds: Annotated[list[BuildInfo], Field(max_length=MAX_BUILD_VARIANTS)]
    active_scope: ScopeResult
    truncated: bool


class CfgRequest(Contract):
    function_symbol_id: Identifier
    builds: Annotated[list[BuildName] | None, Field(max_length=MAX_BUILD_VARIANTS)] = None
    max_graphs: int = Field(default=5, ge=1, le=MAX_CFG_GRAPHS)
    max_blocks: int = Field(default=100, ge=1, le=MAX_CFG_BLOCKS)
    max_elements: int = Field(default=500, ge=1, le=MAX_CFG_ELEMENTS)
    max_edges: int = Field(default=500, ge=1, le=MAX_CFG_EDGES)


class CfgBlockResult(Contract):
    block_id: str
    index: int = Field(ge=0)
    role: str
    reachable: bool
    terminator_kind: str
    terminator_text: str
    terminator_location: SourceLocation | None


class CfgElementResult(Contract):
    element_id: str
    block_id: str
    index: int = Field(ge=0)
    kind: str
    statement_class: str
    text: str
    location: SourceLocation | None


class CfgEdgeResult(Contract):
    edge_id: str
    source_block_id: str
    target_block_id: str
    kind: str
    successor_index: int = Field(ge=0)
    feasible: bool


class CfgGraphResult(Contract):
    graph_id: str
    function_symbol_id: str
    entry_block_id: str
    normal_exit_block_id: str
    exceptional_exit_block_id: str | None
    complete: bool
    provenance: Provenance
    blocks: Annotated[list[CfgBlockResult], Field(max_length=MAX_CFG_BLOCKS)]
    elements: Annotated[list[CfgElementResult], Field(max_length=MAX_CFG_ELEMENTS)]
    edges: Annotated[list[CfgEdgeResult], Field(max_length=MAX_CFG_EDGES)]


class ControlFlowResult(Contract):
    function_symbol_id: str
    scope: ScopeResult
    graphs: Annotated[list[CfgGraphResult], Field(max_length=MAX_CFG_GRAPHS)]
    truncated: bool


class FlowRequest(Contract):
    function_symbol_id: Identifier
    builds: Annotated[list[BuildName] | None, Field(max_length=MAX_BUILD_VARIANTS)] = None
    max_analyses: int = Field(default=5, ge=1, le=MAX_FLOW_ANALYSES)
    max_locations: int = Field(default=200, ge=1, le=MAX_FLOW_LOCATIONS)
    max_accesses: int = Field(default=500, ge=1, le=MAX_FLOW_ACCESSES)
    max_evidence: int = Field(default=500, ge=1, le=MAX_FLOW_EVIDENCE)


class MemoryLocationResult(Contract):
    location_id: str
    kind: str
    name: str
    type_name: str
    declaration_symbol_id: str | None
    base_location_id: str | None
    access_path: list[str]
    is_volatile: bool
    is_atomic: bool


class DataAccessResult(Contract):
    access_id: str
    block_id: str
    cfg_element_id: str | None
    location_id: str
    kind: str
    sequence: int = Field(ge=0)
    expression: str
    location: SourceLocation | None
    pointee_symbol_ids: list[str]
    points_to_complete: bool


class FlowEvidenceResult(Contract):
    evidence_id: str
    relation: DataFlowRelation
    certainty: DataFlowCertainty
    reason: str
    source_access_id: str | None
    target_access_id: str | None
    source_location_id: str | None
    target_location_id: str | None
    location: SourceLocation | None


class SummaryEffectResult(Contract):
    effect_id: str
    kind: str
    location_kind: str
    certainty: DataFlowCertainty
    reason: str
    parameter_index: int | None
    access_path: list[str]
    is_local: bool
    via_callsite_id: str | None
    target_symbol_id: str | None


class SummaryReturnOriginResult(Contract):
    origin_id: str
    kind: str
    certainty: DataFlowCertainty
    reason: str
    location_kind: str | None
    parameter_index: int | None
    access_path: list[str]
    location_id: str | None
    callsite_id: str | None
    is_local: bool
    via_callsite_id: str | None
    target_symbol_id: str | None


class InterproceduralFlowResult(Contract):
    flow_id: str
    kind: str
    caller_summary_id: str
    callee_summary_id: str
    callsite_id: str
    target_symbol_id: str
    target_certainty: CallTargetCertainty
    certainty: DataFlowCertainty
    reason: str
    argument_index: int | None


class DataFlowAnalysisResult(Contract):
    analysis_id: str
    graph_id: str
    complete: bool
    incomplete_reasons: list[str]
    iteration_count: int = Field(ge=0)
    provenance: Provenance
    locations: Annotated[list[MemoryLocationResult], Field(max_length=MAX_FLOW_LOCATIONS)]
    accesses: Annotated[list[DataAccessResult], Field(max_length=MAX_FLOW_ACCESSES)]
    evidence: Annotated[list[FlowEvidenceResult], Field(max_length=MAX_FLOW_EVIDENCE)]
    summary_complete: bool | None
    summary_incomplete_reasons: list[str]
    effects: Annotated[list[SummaryEffectResult], Field(max_length=MAX_FLOW_EVIDENCE)]
    return_origins: Annotated[list[SummaryReturnOriginResult], Field(max_length=MAX_FLOW_EVIDENCE)]
    interprocedural: Annotated[list[InterproceduralFlowResult], Field(max_length=MAX_FLOW_EVIDENCE)]


class DataFlowResult(Contract):
    function_symbol_id: str
    scope: ScopeResult
    analyses: Annotated[list[DataFlowAnalysisResult], Field(max_length=MAX_FLOW_ANALYSES)]
    truncated: bool


class CallRequest(Contract):
    symbol_id: Identifier
    direction: GraphDirection
    builds: Annotated[list[BuildName] | None, Field(max_length=MAX_BUILD_VARIANTS)] = None
    max_results: int = Field(default=20, ge=1, le=MAX_CALL_RESULTS)


class CallEvidenceResult(Contract):
    target_evidence_id: str
    callsite_id: str
    caller_symbol_id: str
    target_symbol_id: str
    dispatch_kind: str
    certainty: CallTargetCertainty
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_reason: str
    derivation: str
    target_set_complete: bool
    unresolved_reason: str
    location: SourceLocation
    provenance: Provenance


class CallGraphResult(Contract):
    symbol_id: str
    direction: GraphDirection
    scope: ScopeResult
    calls: Annotated[list[CallEvidenceResult], Field(max_length=MAX_CALL_RESULTS)]
    truncated: bool


@dataclass(frozen=True, slots=True)
class AnalysisQueryService:
    """Validate scope and budgets before any store query expands analysis facts."""

    store: SQLiteStore
    project_root: Path
    allowed_scope: BuildScope

    def list_builds(self) -> BuildListResult:
        reindex = set(self.store.reindex_required_variants(self.project_root))
        variants = self.store.build_variants(self.project_root, limit=MAX_BUILD_VARIANTS + 1)
        return BuildListResult(
            builds=[
                BuildInfo(
                    name=item.name,
                    target=item.target,
                    platform=item.platform,
                    reindex_required=item.name in reindex,
                )
                for item in variants[:MAX_BUILD_VARIANTS]
            ],
            active_scope=self._scope_result(self.allowed_scope),
            truncated=len(variants) > MAX_BUILD_VARIANTS,
        )

    def control_flow(self, request: CfgRequest) -> ControlFlowResult:
        scope = self.resolve_scope(request.builds)
        self._require_symbol(request.function_symbol_id, scope)
        graphs = self.store.cfg_graphs(
            request.function_symbol_id,
            self.project_root,
            build_scope=scope,
            limit=request.max_graphs,
        )
        remaining_blocks = request.max_blocks
        remaining_elements = request.max_elements
        remaining_edges = request.max_edges
        rendered: list[CfgGraphResult] = []
        truncated = graphs.truncated
        for graph in graphs.items:
            blocks = self.store.cfg_blocks(
                graph.id,
                self.project_root,
                build_scope=BuildScope.single(graph.build_variant),
                limit=max(1, remaining_blocks),
            )
            elements = self.store.cfg_elements(
                graph.id,
                self.project_root,
                build_scope=BuildScope.single(graph.build_variant),
                limit=max(1, remaining_elements),
            )
            edges = self.store.cfg_edges(
                graph.id,
                self.project_root,
                build_scope=BuildScope.single(graph.build_variant),
                limit=max(1, remaining_edges),
            )
            selected_blocks = list(blocks.items[:remaining_blocks])
            selected_elements = list(elements.items[:remaining_elements])
            selected_edges = list(edges.items[:remaining_edges])
            remaining_blocks -= len(selected_blocks)
            remaining_elements -= len(selected_elements)
            remaining_edges -= len(selected_edges)
            truncated |= blocks.truncated or elements.truncated or edges.truncated
            rendered.append(
                CfgGraphResult(
                    graph_id=graph.id,
                    function_symbol_id=graph.function_symbol_id,
                    entry_block_id=graph.entry_block_id,
                    normal_exit_block_id=graph.normal_exit_block_id,
                    exceptional_exit_block_id=graph.exceptional_exit_block_id,
                    complete=all(
                        (not blocks.truncated, not elements.truncated, not edges.truncated)
                    ),
                    provenance=_provenance(graph),
                    blocks=[
                        CfgBlockResult(
                            block_id=item.id,
                            index=item.index,
                            role=item.role.value,
                            reachable=item.reachable,
                            terminator_kind=item.terminator_kind,
                            terminator_text=item.terminator_text,
                            terminator_location=self._location(
                                item.terminator_expansion_span or item.terminator_spelling_span
                            ),
                        )
                        for item in selected_blocks
                    ],
                    elements=[
                        CfgElementResult(
                            element_id=item.id,
                            block_id=item.block_id,
                            index=item.index,
                            kind=item.kind,
                            statement_class=item.statement_class,
                            text=item.text,
                            location=self._location(item.expansion_span or item.spelling_span),
                        )
                        for item in selected_elements
                    ],
                    edges=[
                        CfgEdgeResult(
                            edge_id=item.id,
                            source_block_id=item.source_block_id,
                            target_block_id=item.target_block_id,
                            kind=item.kind.value,
                            successor_index=item.successor_index,
                            feasible=item.feasible,
                        )
                        for item in selected_edges
                    ],
                )
            )
            if min(remaining_blocks, remaining_elements, remaining_edges) == 0:
                truncated |= len(rendered) < len(graphs.items)
                break
        return ControlFlowResult(
            function_symbol_id=request.function_symbol_id,
            scope=self._scope_result(scope),
            graphs=rendered,
            truncated=truncated,
        )

    def data_flow(self, request: FlowRequest) -> DataFlowResult:
        scope = self.resolve_scope(request.builds)
        self._require_symbol(request.function_symbol_id, scope)
        graphs = self.store.cfg_graphs(
            request.function_symbol_id,
            self.project_root,
            build_scope=scope,
            limit=request.max_analyses,
        )
        remaining_locations = request.max_locations
        remaining_accesses = request.max_accesses
        remaining_evidence = request.max_evidence
        rendered: list[DataFlowAnalysisResult] = []
        summaries = self.store.function_summaries(
            request.function_symbol_id,
            self.project_root,
            build_scope=scope,
            limit=request.max_analyses,
        )
        summaries_by_graph = {item.graph_id: item for item in summaries.items}
        truncated = graphs.truncated or summaries.truncated
        for graph in graphs.items:
            graph_scope = BuildScope.single(graph.build_variant)
            analyses = self.store.data_flow_analyses(
                graph.id, self.project_root, build_scope=graph_scope, limit=1
            )
            if not analyses.items:
                continue
            analysis = analyses.items[0]
            locations = self.store.memory_locations(
                analysis.id,
                self.project_root,
                build_scope=graph_scope,
                limit=max(1, remaining_locations),
            )
            accesses = self.store.data_accesses(
                analysis.id,
                self.project_root,
                build_scope=graph_scope,
                limit=max(1, remaining_accesses),
            )
            evidence = self.store.data_flow_evidence(
                analysis.id,
                self.project_root,
                build_scope=graph_scope,
                limit=max(1, remaining_evidence),
            )
            selected_locations = list(locations.items[:remaining_locations])
            selected_accesses = list(accesses.items[:remaining_accesses])
            selected_evidence = list(evidence.items[:remaining_evidence])
            remaining_locations -= len(selected_locations)
            remaining_accesses -= len(selected_accesses)
            remaining_evidence -= len(selected_evidence)
            summary = summaries_by_graph.get(graph.id)
            effects = ()
            return_origins = ()
            cross_flows = ()
            if summary is not None and remaining_evidence > 0:
                effects = self.store.summary_effects(
                    summary.id,
                    self.project_root,
                    build_scope=graph_scope,
                    limit=max(1, remaining_evidence),
                )
                remaining_evidence -= len(effects.items)
            if summary is not None and remaining_evidence > 0:
                return_origins = self.store.summary_return_origins(
                    summary.id,
                    self.project_root,
                    build_scope=graph_scope,
                    limit=max(1, remaining_evidence),
                )
                remaining_evidence -= len(return_origins.items)
            if summary is not None and remaining_evidence > 0:
                cross_flows = self.store.interprocedural_flows(
                    summary.id,
                    self.project_root,
                    build_scope=graph_scope,
                    limit=max(1, remaining_evidence),
                )
                remaining_evidence -= len(cross_flows.items)
            truncated |= any(item.truncated for item in (analyses, locations, accesses, evidence))
            if effects:
                truncated |= effects.truncated
            if return_origins:
                truncated |= return_origins.truncated
            if cross_flows:
                truncated |= cross_flows.truncated
            rendered.append(
                DataFlowAnalysisResult(
                    analysis_id=analysis.id,
                    graph_id=analysis.graph_id,
                    complete=analysis.complete,
                    incomplete_reasons=list(analysis.incomplete_reasons),
                    iteration_count=analysis.iteration_count,
                    provenance=_provenance(analysis),
                    locations=[
                        MemoryLocationResult(
                            location_id=item.id,
                            kind=item.kind.value,
                            name=item.name,
                            type_name=item.type_name,
                            declaration_symbol_id=item.declaration_symbol_id,
                            base_location_id=item.base_location_id,
                            access_path=list(item.access_path),
                            is_volatile=item.is_volatile,
                            is_atomic=item.is_atomic,
                        )
                        for item in selected_locations
                    ],
                    accesses=[
                        DataAccessResult(
                            access_id=item.id,
                            block_id=item.block_id,
                            cfg_element_id=item.cfg_element_id,
                            location_id=item.location_id,
                            kind=item.kind.value,
                            sequence=item.sequence,
                            expression=item.expression,
                            location=self._location(item.span),
                            pointee_symbol_ids=list(item.pointee_symbol_ids),
                            points_to_complete=item.points_to_complete,
                        )
                        for item in selected_accesses
                    ],
                    evidence=[
                        FlowEvidenceResult(
                            evidence_id=item.id,
                            relation=item.relation,
                            certainty=item.certainty,
                            reason=item.reason,
                            source_access_id=item.source_access_id,
                            target_access_id=item.target_access_id,
                            source_location_id=item.source_location_id,
                            target_location_id=item.target_location_id,
                            location=self._location(item.evidence_span),
                        )
                        for item in selected_evidence
                    ],
                    summary_complete=summary.complete if summary else None,
                    summary_incomplete_reasons=list(summary.incomplete_reasons) if summary else [],
                    effects=[
                        SummaryEffectResult(
                            effect_id=item.id,
                            kind=item.kind.value,
                            location_kind=item.location_kind.value,
                            certainty=item.certainty,
                            reason=item.reason,
                            parameter_index=item.parameter_index,
                            access_path=list(item.access_path),
                            is_local=item.is_local,
                            via_callsite_id=item.via_callsite_id,
                            target_symbol_id=item.target_symbol_id,
                        )
                        for item in (effects.items if effects else ())
                    ],
                    return_origins=[
                        SummaryReturnOriginResult(
                            origin_id=item.id,
                            kind=item.kind.value,
                            certainty=item.certainty,
                            reason=item.reason,
                            location_kind=(
                                item.location_kind.value if item.location_kind else None
                            ),
                            parameter_index=item.parameter_index,
                            access_path=list(item.access_path),
                            location_id=item.location_id,
                            callsite_id=item.callsite_id,
                            is_local=item.is_local,
                            via_callsite_id=item.via_callsite_id,
                            target_symbol_id=item.target_symbol_id,
                        )
                        for item in (return_origins.items if return_origins else ())
                    ],
                    interprocedural=[
                        InterproceduralFlowResult(
                            flow_id=item.id,
                            kind=item.kind.value,
                            caller_summary_id=item.caller_summary_id,
                            callee_summary_id=item.callee_summary_id,
                            callsite_id=item.callsite_id,
                            target_symbol_id=item.target_symbol_id,
                            target_certainty=item.target_certainty,
                            certainty=item.certainty,
                            reason=item.reason,
                            argument_index=item.argument_index,
                        )
                        for item in (cross_flows.items if cross_flows else ())
                    ],
                )
            )
            if min(remaining_locations, remaining_accesses, remaining_evidence) == 0:
                truncated = True
                break
        return DataFlowResult(
            function_symbol_id=request.function_symbol_id,
            scope=self._scope_result(scope),
            analyses=rendered,
            truncated=truncated,
        )

    def calls(self, request: CallRequest) -> CallGraphResult:
        if request.direction not in {GraphDirection.INCOMING, GraphDirection.OUTGOING}:
            raise ValueError("call direction must be incoming or outgoing")
        scope = self.resolve_scope(request.builds)
        self._require_symbol(request.symbol_id, scope)
        evidence = self.store.call_evidence(
            request.symbol_id,
            incoming=request.direction == GraphDirection.INCOMING,
            project_root=self.project_root,
            build_scope=scope,
            limit=request.max_results,
        )
        return CallGraphResult(
            symbol_id=request.symbol_id,
            direction=request.direction,
            scope=self._scope_result(scope),
            calls=[
                CallEvidenceResult(
                    target_evidence_id=target.id,
                    callsite_id=site.id,
                    caller_symbol_id=site.owner_symbol_id,
                    target_symbol_id=target.target_symbol_id,
                    dispatch_kind=site.dispatch_kind.value,
                    certainty=target.certainty,
                    confidence=target.confidence,
                    confidence_reason=target.confidence_reason,
                    derivation=target.derivation,
                    target_set_complete=site.target_set_complete,
                    unresolved_reason=site.unresolved_reason,
                    location=self._required_location(site.expansion_span),
                    provenance=_provenance(target),
                )
                for site, target in evidence.items
            ],
            truncated=evidence.truncated,
        )

    def resolve_scope(self, requested: list[str] | None) -> BuildScope:
        """Resolve a caller subset without permitting expansion beyond operator scope."""
        scope = self.allowed_scope if requested is None else BuildScope(tuple(requested))
        allowed = set(self.allowed_scope.variants)
        denied = set(scope.variants) - allowed
        if denied:
            raise ValueError("build scope is not operator-enabled: " + ", ".join(sorted(denied)))
        indexed = {
            item.name
            for item in self.store.build_variants(self.project_root, limit=MAX_BUILD_VARIANTS)
        }
        missing = set(scope.variants) - indexed
        if missing:
            raise ValueError("build scope is not indexed: " + ", ".join(sorted(missing)))
        return scope

    def _require_symbol(self, symbol_id: str, scope: BuildScope) -> None:
        if self.store.get_symbol(symbol_id, self.project_root, build_scope=scope) is None:
            raise ValueError("requested symbol ID is not present in the selected build scope")

    @staticmethod
    def _scope_result(scope: BuildScope) -> ScopeResult:
        kind: Literal["single", "union"] = "union" if scope.is_union else "single"
        prefix = "union" if scope.is_union else "build"
        return ScopeResult(
            variants=list(scope.variants), kind=kind, label=f"{prefix}:{','.join(scope.variants)}"
        )

    def _location(self, span: SourceSpan | None) -> SourceLocation | None:
        return self._required_location(span) if span is not None else None

    def _required_location(self, span: SourceSpan) -> SourceLocation:
        root = self.project_root.resolve(strict=False)
        path = (span.path if span.path.is_absolute() else root / span.path).resolve(strict=False)
        if not path.is_relative_to(root):
            raise ValueError("indexed source location is outside the configured project")
        return SourceLocation(
            path=path.relative_to(root).as_posix(),
            start_line=span.start_line,
            end_line=span.end_line,
            start_column=span.start_column,
            end_column=span.end_column,
        )


class _HasProvenance(Protocol):
    build_variant: str
    build_configuration_id: str
    translation_unit_id: str


def _provenance(item: _HasProvenance) -> Provenance:
    return Provenance(
        build_variant=item.build_variant,
        build_configuration_id=item.build_configuration_id,
        translation_unit_id=item.translation_unit_id,
    )
