"""Project-bound MCP v2 server with bounded, structured code-navigation tools."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from cpp_context_engine import __version__
from cpp_context_engine.api import (
    AnswerRequest,
    BuildListResult,
    CallRequest,
    CfgRequest,
    ControlFlowResult,
    DataFlowResult,
    FlowRequest,
    QueryRequest,
)
from cpp_context_engine.config import AppConfig
from cpp_context_engine.models import CodeSymbol, GraphDirection, GraphEdge, GraphRelation
from cpp_context_engine.runtime import (
    Runtime,
    build_runtime,
)
from cpp_context_engine.runtime import (
    index_project as run_project_index,
)
from cpp_context_engine.storage import FilesystemSourceReader

from .contracts import (
    MAX_ANSWER_CHARS,
    MAX_DIAGNOSTICS,
    MAX_SEARCH_RESULTS,
    AnalysisBlocks,
    AnalysisGraphs,
    AnalysisItems,
    AnswerSource,
    AnswerSteps,
    AskCodeResult,
    Builds,
    ContextTokens,
    GraphDepth,
    GraphEdgeResult,
    GraphFanout,
    GraphPathStepResult,
    GraphResult,
    GraphResultLimit,
    IndexProjectResult,
    QueryText,
    ReadSymbolResult,
    Relations,
    SearchCodeItem,
    SearchCodeResult,
    SearchResultLimit,
    SourceChars,
    SourceLocation,
    SymbolId,
    SymbolReference,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class PublicToolFailure(RuntimeError):
    """A deliberately non-sensitive failure that may be returned to an MCP client."""


@dataclass(slots=True)
class ProjectServerState:
    """One operator-configured runtime shared by all calls for the server lifetime."""

    config: AppConfig
    lock: anyio.Lock
    runtime: Runtime | None = None

    async def open(self) -> None:
        assert self.config.database_path is not None
        if not self.config.database_path.is_file():
            return

        def load_existing() -> Runtime | None:
            try:
                return build_runtime(self.config)
            except Exception as exc:
                logger.warning(
                    "Configured MCP index is not ready (%s); index_project remains available",
                    type(exc).__name__,
                )
                return None

        self.runtime = await anyio.to_thread.run_sync(load_existing, abandon_on_cancel=False)

    async def close(self) -> None:
        async with self.lock:
            runtime, self.runtime = self.runtime, None
            if runtime is not None:
                await anyio.to_thread.run_sync(runtime.close, abandon_on_cancel=False)

    async def execute(self, operation: Callable[[], T]) -> T:
        """Serialize SQLite/index work and never abandon a cancelled worker thread."""

        async with self.lock:
            return await anyio.to_thread.run_sync(operation, abandon_on_cancel=False)

    def require_runtime(self) -> Runtime:
        if self.runtime is None:
            raise PublicToolFailure(
                "The configured project has no usable index; call index_project first."
            )
        return self.runtime


def create_mcp_server(config: AppConfig) -> MCPServer[ProjectServerState]:
    """Build a server permanently bound to the operator-supplied project configuration."""

    local_read = _annotations(read_only=True, idempotent=True, open_world=False)
    hosted_embeddings = config.embedding_provider == "openai"
    embedding_read = _annotations(
        read_only=True,
        idempotent=True,
        open_world=hosted_embeddings,
    )
    indexing_write = _annotations(
        read_only=False,
        idempotent=True,
        open_world=hosted_embeddings,
    )
    llm_read = _annotations(
        read_only=True,
        idempotent=False,
        open_world=bool(config.llm_base_url and config.llm_model),
    )

    @asynccontextmanager
    async def lifespan(_server: MCPServer[ProjectServerState]) -> AsyncIterator[ProjectServerState]:
        state = ProjectServerState(config=config, lock=anyio.Lock())
        await state.open()
        try:
            yield state
        finally:
            await state.close()

    server: MCPServer[ProjectServerState] = MCPServer(
        "cpp-context-engine",
        title="C++ Context Engine",
        description="Compiler-aware, project-bound C++ code retrieval and graph navigation.",
        instructions=(
            "This server is bound to one operator-configured project. Tool callers cannot select "
            "project, database, compilation database, or filesystem paths. Start with search_code, "
            "then use read_symbol and graph tools for exact connected context. "
            "If hosted embeddings or an LLM are configured, queries, indexed symbol text, or "
            "selected code excerpts are "
            "sent to that external provider."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    @server.tool(
        title="List indexed build variants",
        description=(
            "List safe build labels and the operator-enabled single or union query scope. "
            "Compilation-database paths and build metadata are not exposed."
        ),
        annotations=local_read,
    )
    async def list_builds(ctx: Context[ProjectServerState]) -> BuildListResult:
        state = _state(ctx)
        return await _call_tool(
            state,
            "list_builds",
            lambda: state.require_runtime().analysis_service.list_builds(),
            "Build listing failed for the configured project index.",
        )

    @server.tool(
        title="Read bounded control-flow evidence",
        description=(
            "Return compiler-derived CFG blocks, elements, and typed edges for one stable "
            "function symbol ID, within total pre-query budgets."
        ),
        annotations=local_read,
    )
    async def control_flow(
        symbol_id: SymbolId,
        ctx: Context[ProjectServerState],
        builds: Builds = None,
        max_graphs: AnalysisGraphs = 5,
        max_blocks: AnalysisBlocks = 100,
        max_elements: AnalysisItems = 500,
        max_edges: AnalysisItems = 500,
    ) -> ControlFlowResult:
        state = _state(ctx)
        return await _call_tool(
            state,
            "control_flow",
            lambda: state.require_runtime().analysis_service.control_flow(
                CfgRequest(
                    function_symbol_id=symbol_id,
                    builds=builds,
                    max_graphs=max_graphs,
                    max_blocks=max_blocks,
                    max_elements=max_elements,
                    max_edges=max_edges,
                )
            ),
            "Control-flow lookup failed for the configured project index.",
        )

    @server.tool(
        title="Read bounded data-flow evidence",
        description=(
            "Return intraprocedural def-use and alias evidence plus bounded function-summary "
            "and interprocedural-flow evidence; no dead-code verdict is inferred."
        ),
        annotations=local_read,
    )
    async def data_flow(
        symbol_id: SymbolId,
        ctx: Context[ProjectServerState],
        builds: Builds = None,
        max_analyses: AnalysisGraphs = 5,
        max_locations: AnalysisItems = 200,
        max_accesses: AnalysisItems = 500,
        max_evidence: AnalysisItems = 500,
    ) -> DataFlowResult:
        state = _state(ctx)
        return await _call_tool(
            state,
            "data_flow",
            lambda: state.require_runtime().analysis_service.data_flow(
                FlowRequest(
                    function_symbol_id=symbol_id,
                    builds=builds,
                    max_analyses=max_analyses,
                    max_locations=max_locations,
                    max_accesses=max_accesses,
                    max_evidence=max_evidence,
                )
            ),
            "Data-flow lookup failed for the configured project index.",
        )

    @server.tool(
        title="Index configured C++ project",
        description=(
            "Incrementally index the operator-configured project and compilation database. "
            "No caller-controlled path is accepted. Hosted embeddings, when configured, receive "
            "bounded symbol text."
        ),
        annotations=indexing_write,
    )
    async def index_project(ctx: Context[ProjectServerState]) -> IndexProjectResult:
        state = _state(ctx)

        def operation() -> IndexProjectResult:
            previous, state.runtime = state.runtime, None
            if previous is not None:
                previous.close()
            result = run_project_index(state.config)
            state.runtime = build_runtime(state.config)
            indexing = result.indexing
            return IndexProjectResult(
                indexed_translation_units=indexing.indexed_translation_units,
                skipped_translation_units=indexing.skipped_translation_units,
                removed_translation_units=indexing.removed_translation_units,
                indexed_symbols=indexing.indexed_symbols,
                indexed_occurrences=indexing.indexed_occurrences,
                indexed_edges=indexing.indexed_edges,
                embedded_symbols=result.embedded_symbols,
                embedding_model=result.embedding_model,
                analysis_backend=result.analysis_backend,
                advanced_facts_complete=result.advanced_facts_complete,
                analyzer_capabilities=list(result.analyzer_capabilities),
            )

        return await _call_tool(
            state,
            "index_project",
            operation,
            "Indexing failed; verify the server's project and compilation database configuration.",
        )

    @server.tool(
        title="Search connected C++ code",
        description=(
            "Hybrid lexical, symbol, and embedding search followed by bounded graph expansion and "
            "source packing. Hosted embeddings, when configured, receive the query."
        ),
        annotations=embedding_read,
    )
    async def search_code(
        query: QueryText,
        ctx: Context[ProjectServerState],
        max_results: SearchResultLimit = 10,
        max_context_tokens: ContextTokens = 8_000,
        builds: Builds = None,
    ) -> SearchCodeResult:
        state = _state(ctx)

        def operation() -> SearchCodeResult:
            runtime, temporary = _runtime_for_builds(state, builds)
            try:
                bundle = runtime.retrieval_service.query(
                    QueryRequest(query.strip(), max_context_tokens, max_results=max_results)
                ).context
                selected_scope = runtime.config.build_scope
            finally:
                if temporary:
                    runtime.close()
            items = [
                SearchCodeItem(
                    symbol=_symbol_reference(item.hit.symbol, state.config.project_root),
                    source_text=item.source_text,
                    score=item.hit.score,
                    reason=item.reason,
                    graph_path=[
                        GraphPathStepResult(
                            source_id=step.source_id,
                            target_id=step.target_id,
                            relation=step.relation,
                        )
                        for step in item.path
                    ],
                )
                for item in bundle.items[:max_results]
            ]
            return SearchCodeResult(
                query=query.strip(),
                items=items,
                estimated_tokens=bundle.estimated_tokens,
                truncated=bundle.truncated or len(bundle.items) > max_results,
                diagnostics=list(bundle.diagnostics[:MAX_DIAGNOSTICS]),
                scope_kind="union" if selected_scope.is_union else "single",
                scope_label=(
                    f"union:{','.join(selected_scope.variants)}"
                    if selected_scope.is_union
                    else f"build:{selected_scope.variants[0]}"
                ),
                scope_variants=list(selected_scope.variants),
            )

        return await _call_tool(
            state,
            "search_code",
            operation,
            "Code search failed; check the configured index and embedding provider.",
        )

    @server.tool(
        title="Read exact indexed symbol",
        description=(
            "Read the indexed source span for one exact symbol ID. The source path is taken only "
            "from the configured index and must remain inside the configured project."
        ),
        annotations=local_read,
    )
    async def read_symbol(
        symbol_id: SymbolId,
        ctx: Context[ProjectServerState],
        max_source_chars: SourceChars = 20_000,
        builds: Builds = None,
    ) -> ReadSymbolResult:
        state = _state(ctx)

        def operation() -> ReadSymbolResult:
            runtime = state.require_runtime()
            scope = runtime.analysis_service.resolve_scope(builds)
            symbol = _get_symbol(runtime, symbol_id, scope.variants)
            source = FilesystemSourceReader(state.config.project_root).read_symbol(symbol)
            truncated = len(source) > max_source_chars
            return ReadSymbolResult(
                symbol=_symbol_reference(symbol, state.config.project_root),
                source_text=source[:max_source_chars],
                truncated=truncated,
                scope_kind="union" if scope.is_union else "single",
                scope_label=(
                    f"union:{','.join(scope.variants)}"
                    if scope.is_union
                    else f"build:{scope.variants[0]}"
                ),
                scope_variants=list(scope.variants),
            )

        return await _call_tool(
            state,
            "read_symbol",
            operation,
            "The symbol could not be read from the configured project.",
        )

    @server.tool(
        title="Traverse indexed code relationships",
        description=(
            "Traverse compiler-derived graph edges from one exact symbol ID with direction, "
            "relation, depth, per-node fanout, and total-result limits enforced in the store."
        ),
        annotations=local_read,
    )
    async def neighbors(
        symbol_id: SymbolId,
        ctx: Context[ProjectServerState],
        direction: GraphDirection = GraphDirection.BOTH,
        relations: Relations = None,
        depth: GraphDepth = 1,
        max_results: GraphResultLimit = 50,
        per_node_fanout: GraphFanout = 20,
        builds: Builds = None,
    ) -> GraphResult:
        return await _graph_tool(
            _state(ctx),
            "neighbors",
            symbol_id,
            direction=direction,
            relations=relations,
            depth=depth,
            max_results=max_results,
            per_node_fanout=per_node_fanout,
            builds=builds,
        )

    @server.tool(
        title="Find callers",
        description="Return bounded incoming compiler-derived CALLS edges for one exact symbol ID.",
        annotations=local_read,
    )
    async def callers(
        symbol_id: SymbolId,
        ctx: Context[ProjectServerState],
        max_results: GraphFanout = 20,
        builds: Builds = None,
    ) -> GraphResult:
        return await _graph_tool(
            _state(ctx),
            "callers",
            symbol_id,
            direction=GraphDirection.INCOMING,
            relations=[GraphRelation.CALLS],
            depth=1,
            max_results=max_results,
            per_node_fanout=max_results,
            builds=builds,
        )

    @server.tool(
        title="Find callees",
        description="Return bounded outgoing compiler-derived CALLS edges for one exact symbol ID.",
        annotations=local_read,
    )
    async def callees(
        symbol_id: SymbolId,
        ctx: Context[ProjectServerState],
        max_results: GraphFanout = 20,
        builds: Builds = None,
    ) -> GraphResult:
        return await _graph_tool(
            _state(ctx),
            "callees",
            symbol_id,
            direction=GraphDirection.OUTGOING,
            relations=[GraphRelation.CALLS],
            depth=1,
            max_results=max_results,
            per_node_fanout=max_results,
            builds=builds,
        )

    @server.tool(
        title="Answer a C++ code question",
        description=(
            "Answer from bounded retrieved source context with validated citations. This tool "
            "requires an operator-configured LLM and sends the query plus selected code excerpts "
            "to that provider."
        ),
        annotations=llm_read,
    )
    async def ask_code(
        query: QueryText,
        ctx: Context[ProjectServerState],
        max_context_tokens: ContextTokens = 8_000,
        max_steps: AnswerSteps = 3,
        builds: Builds = None,
    ) -> AskCodeResult:
        state = _state(ctx)

        def operation() -> AskCodeResult:
            runtime, temporary = _runtime_for_builds(state, builds)
            try:
                if runtime.answer_service is None:
                    raise PublicToolFailure(
                        "Code answering is unavailable; configure an LLM when starting the server."
                    )
                answer = runtime.answer_service.answer(
                    AnswerRequest(query.strip(), max_context_tokens, max_steps)
                )
            finally:
                if temporary:
                    runtime.close()
            rendered_answer = _redact_project_root(answer.answer, state.config.project_root)
            diagnostics = list(answer.diagnostics[:MAX_DIAGNOSTICS])
            complete = answer.complete
            if len(rendered_answer) > MAX_ANSWER_CHARS:
                rendered_answer = rendered_answer[:MAX_ANSWER_CHARS]
                complete = False
                diagnostics.append("answer stopped at the configured output budget")
            return AskCodeResult(
                answer=rendered_answer,
                complete=complete,
                steps=answer.steps,
                sources=[
                    AnswerSource(
                        symbol_id=source.symbol_id,
                        qualified_name=source.qualified_name,
                        build_variant=source.build_variant,
                        location=_source_location(
                            source.path,
                            source.start_line,
                            source.end_line,
                            state.config.project_root,
                        ),
                    )
                    for source in answer.sources[:MAX_SEARCH_RESULTS]
                ],
                diagnostics=diagnostics[:MAX_DIAGNOSTICS],
                scope_kind="union" if len(answer.build_variants) > 1 else "single",
                scope_label=answer.scope_label,
                scope_variants=list(answer.build_variants),
            )

        return await _call_tool(
            state,
            "ask_code",
            operation,
            "Code answering failed; check the configured index and LLM provider.",
        )

    return server


def _state(ctx: Context[ProjectServerState]) -> ProjectServerState:
    return ctx.request_context.lifespan_context


def _annotations(*, read_only: bool, idempotent: bool, open_world: bool) -> ToolAnnotations:
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=False,
        idempotent_hint=idempotent,
        open_world_hint=open_world,
    )


async def _call_tool(
    state: ProjectServerState,
    tool_name: str,
    operation: Callable[[], T],
    public_error: str,
) -> T:
    try:
        return await state.execute(operation)
    except PublicToolFailure as exc:
        raise ToolError(str(exc)) from None
    except Exception as exc:
        # Provider/database exception messages may contain credentials or host paths.
        logger.error("MCP tool %s failed (%s)", tool_name, type(exc).__name__)
        raise ToolError(public_error) from None


async def _graph_tool(
    state: ProjectServerState,
    tool_name: str,
    symbol_id: str,
    *,
    direction: GraphDirection,
    relations: list[GraphRelation] | None,
    depth: int,
    max_results: int,
    per_node_fanout: int,
    builds: list[str] | None,
) -> GraphResult:
    def operation() -> GraphResult:
        runtime = state.require_runtime()
        scope = runtime.analysis_service.resolve_scope(builds)
        origin = _get_symbol(runtime, symbol_id, scope.variants)
        if (
            relations == [GraphRelation.CALLS]
            and depth == 1
            and direction in {GraphDirection.INCOMING, GraphDirection.OUTGOING}
        ):
            calls = runtime.analysis_service.calls(
                CallRequest(
                    symbol_id=origin.id,
                    direction=direction,
                    builds=list(scope.variants),
                    max_results=max_results,
                )
            )
            rendered_calls: list[GraphEdgeResult] = []
            for call in calls.calls:
                fact_scope = (call.provenance.build_variant,)
                source = runtime.store.get_symbol(
                    call.caller_symbol_id,
                    state.config.project_root,
                    build_scope=fact_scope,
                )
                target = runtime.store.get_symbol(
                    call.target_symbol_id,
                    state.config.project_root,
                    build_scope=fact_scope,
                )
                if source is None or target is None:
                    continue
                rendered_calls.append(
                    GraphEdgeResult(
                        edge_id=call.target_evidence_id,
                        build_variant=call.provenance.build_variant,
                        source=_symbol_reference(source, state.config.project_root),
                        target=_symbol_reference(target, state.config.project_root),
                        relation=GraphRelation.CALLS,
                        translation_unit_id=call.provenance.translation_unit_id,
                        build_configuration_id=call.provenance.build_configuration_id,
                        callsite_id=call.callsite_id,
                        certainty=call.certainty,
                        confidence=call.confidence,
                        confidence_reason=call.confidence_reason,
                        derivation=call.derivation,
                        target_set_complete=call.target_set_complete,
                    )
                )
            if rendered_calls:
                return GraphResult(
                    symbol=_symbol_reference(origin, state.config.project_root),
                    direction=direction,
                    depth=depth,
                    edges=rendered_calls,
                    truncated=calls.truncated,
                    scope_kind=calls.scope.kind,
                    scope_label=calls.scope.label,
                    scope_variants=calls.scope.variants,
                )
        edges = tuple(
            runtime.store.neighbors(
                origin.id,
                relations=frozenset(relations) if relations else None,
                depth=depth,
                direction=direction,
                max_edges=max_results + 1,
                per_node_limit=per_node_fanout,
                project_root=state.config.project_root,
                build_scope=scope,
            )
        )
        truncated = len(edges) > max_results
        if not truncated and len(edges) >= per_node_fanout:
            # The primary query cannot reveal when its per-node SQL limit hid one more edge.
            probe = runtime.store.neighbors(
                origin.id,
                relations=frozenset(relations) if relations else None,
                depth=depth,
                direction=direction,
                max_edges=max_results + 1,
                per_node_limit=per_node_fanout + 1,
                project_root=state.config.project_root,
                build_scope=scope,
            )
            known_edge_ids = {edge.id for edge in edges}
            truncated = any(edge.id not in known_edge_ids for edge in probe)
        call_details = {}
        details_by_id = {}
        evidence_edges: list[GraphEdge] = []
        if relations is None or GraphRelation.CALLS in relations:
            call_directions = (
                (GraphDirection.OUTGOING, GraphDirection.INCOMING)
                if direction == GraphDirection.BOTH
                else (direction,)
            )
            for call_direction in call_directions:
                evidence = runtime.analysis_service.calls(
                    CallRequest(
                        symbol_id=origin.id,
                        direction=call_direction,
                        builds=list(scope.variants),
                        max_results=max_results,
                    )
                )
                truncated |= evidence.truncated
                for call in evidence.calls:
                    pair = (
                        call.caller_symbol_id,
                        call.target_symbol_id,
                        call.provenance.build_variant,
                    )
                    existing = call_details.get(pair)
                    if existing is None or call.certainty.value == "certain":
                        call_details[pair] = call
                    details_by_id[call.target_evidence_id] = call
                    evidence_edges.append(
                        GraphEdge(
                            source_id=call.caller_symbol_id,
                            target_id=call.target_symbol_id,
                            relation=GraphRelation.CALLS,
                            translation_unit_id=call.provenance.translation_unit_id,
                            id=call.target_evidence_id,
                            build_configuration_id=call.provenance.build_configuration_id,
                            build_variant=call.provenance.build_variant,
                        )
                    )
        evidence_pairs = {
            (edge.source_id, edge.target_id, edge.build_variant) for edge in evidence_edges
        }
        legacy_and_non_call_edges = tuple(
            edge
            for edge in edges
            if edge.relation != GraphRelation.CALLS
            or (edge.source_id, edge.target_id, edge.build_variant) not in evidence_pairs
        )
        origin_edges = [
            edge
            for edge in (*legacy_and_non_call_edges, *evidence_edges)
            if origin.id in {edge.source_id, edge.target_id}
        ]
        deeper_edges = [
            edge
            for edge in legacy_and_non_call_edges
            if origin.id not in {edge.source_id, edge.target_id}
        ]

        def edge_sort_key(edge: GraphEdge) -> tuple[object, ...]:
            detail = details_by_id.get(edge.id) or call_details.get(
                (edge.source_id, edge.target_id, edge.build_variant)
            )
            return (
                0 if detail is not None and detail.certainty.value == "certain" else 1,
                edge.relation.value,
                edge.source_id,
                edge.target_id,
                edge.build_variant,
                edge.id,
            )

        origin_edges.sort(key=edge_sort_key)
        if len(origin_edges) > per_node_fanout:
            # Advanced callsite facts augment legacy edges but still share the public fanout budget.
            truncated = True
        edges = tuple((*deeper_edges, *origin_edges[:per_node_fanout]))
        truncated |= len(edges) > max_results
        edges = tuple(sorted(edges, key=edge_sort_key))
        rendered: list[GraphEdgeResult] = []
        for edge in edges[:max_results]:
            edge_scope = (edge.build_variant,)
            source = runtime.store.get_symbol(
                edge.source_id, state.config.project_root, build_scope=edge_scope
            )
            target = runtime.store.get_symbol(
                edge.target_id, state.config.project_root, build_scope=edge_scope
            )
            if source is None or target is None:
                continue
            detail = details_by_id.get(edge.id) or call_details.get(
                (edge.source_id, edge.target_id, edge.build_variant)
            )
            rendered.append(
                GraphEdgeResult(
                    edge_id=edge.id,
                    build_variant=edge.build_variant,
                    source=_symbol_reference(source, state.config.project_root),
                    target=_symbol_reference(target, state.config.project_root),
                    relation=edge.relation,
                    translation_unit_id=edge.translation_unit_id,
                    build_configuration_id=edge.build_configuration_id,
                    callsite_id=detail.callsite_id if detail else None,
                    certainty=detail.certainty if detail else None,
                    confidence=detail.confidence if detail else None,
                    confidence_reason=detail.confidence_reason if detail else None,
                    derivation=detail.derivation if detail else None,
                    target_set_complete=detail.target_set_complete if detail else None,
                )
            )
        return GraphResult(
            symbol=_symbol_reference(origin, state.config.project_root),
            direction=direction,
            depth=depth,
            edges=rendered,
            truncated=truncated,
            scope_kind="union" if scope.is_union else "single",
            scope_label=(
                f"union:{','.join(scope.variants)}"
                if scope.is_union
                else f"build:{scope.variants[0]}"
            ),
            scope_variants=list(scope.variants),
        )

    return await _call_tool(
        state,
        tool_name,
        operation,
        "Graph navigation failed for the configured project index.",
    )


def _get_symbol(
    runtime: Runtime, symbol_id: str, build_scope: Sequence[str] | None = None
) -> CodeSymbol:
    symbol = runtime.store.get_symbol(
        symbol_id,
        runtime.config.project_root,
        build_scope=tuple(build_scope) if build_scope else None,
    )
    if symbol is None:
        raise PublicToolFailure("The requested symbol ID is not present in the configured index.")
    return symbol


def _runtime_for_builds(
    state: ProjectServerState, builds: list[str] | None
) -> tuple[Runtime, bool]:
    runtime = state.require_runtime()
    scope = runtime.analysis_service.resolve_scope(builds)
    if scope == runtime.config.build_scope:
        return runtime, False
    return build_runtime(replace(runtime.config, build_scope=scope)), True


def _symbol_reference(symbol: CodeSymbol, project_root: Path) -> SymbolReference:
    return SymbolReference(
        symbol_id=symbol.id,
        variant_id=symbol.variant_id,
        build_variant=symbol.build_variant,
        qualified_name=symbol.qualified_name,
        kind=symbol.kind,
        signature=symbol.signature,
        location=_source_location(
            symbol.span.path,
            symbol.span.start_line,
            symbol.span.end_line,
            project_root,
        ),
    )


def _source_location(path: Path, start_line: int, end_line: int, root: Path) -> SourceLocation:
    resolved_root = root.resolve(strict=False)
    resolved = (path if path.is_absolute() else resolved_root / path).resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise PublicToolFailure("An indexed source location is outside the configured project.")
    return SourceLocation(
        path=resolved.relative_to(resolved_root).as_posix(),
        start_line=start_line,
        end_line=end_line,
    )


def _redact_project_root(text: str, root: Path) -> str:
    result = text
    for representation in {str(root.resolve(strict=False)), root.resolve(strict=False).as_posix()}:
        result = result.replace(f"{representation}/", "")
        result = result.replace(representation, ".")
    return result


def _standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpp-context-mcp",
        description="Serve one operator-configured C++ project over MCP.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", help="HTTP bind address (default: configured localhost)")
    parser.add_argument("--port", type=int, help="HTTP port")
    return parser


def run_server(
    config: AppConfig,
    *,
    transport: str = "stdio",
    host: str | None = None,
    port: int | None = None,
) -> int:
    """Run stdio by default or an explicitly selected Streamable HTTP transport."""

    server = create_mcp_server(config)
    if transport == "stdio":
        server.run("stdio")
        return 0
    if transport != "streamable-http":
        raise ValueError("MCP transport must be 'stdio' or 'streamable-http'")
    selected_host = host or config.serve_host
    selected_port = port or config.serve_port
    if selected_port < 1 or selected_port > 65_535:
        raise ValueError("MCP HTTP port must be in [1, 65535]")
    if selected_host not in {"127.0.0.1", "localhost", "::1"}:
        logger.warning(
            "Streamable HTTP is binding beyond localhost; use authentication and a trusted proxy"
        )
    # Uvicorn's built-in access logger targets stdout. HTTP has no protocol use for
    # stdout, so keep every diagnostic stream-consistent with the stdio server.
    with redirect_stdout(sys.stderr):
        server.run(
            "streamable-http",
            host=selected_host,
            port=selected_port,
            streamable_http_path="/mcp",
            json_response=True,
            stateless_http=True,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point using environment-owned project/provider configuration."""

    parser = _standalone_parser()
    args = parser.parse_args(argv)
    try:
        config = AppConfig.from_environment()
        return run_server(
            config,
            transport=args.transport,
            host=args.host,
            port=args.port,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
