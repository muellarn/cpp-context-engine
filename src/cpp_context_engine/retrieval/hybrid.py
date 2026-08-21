"""Hybrid ranking, bounded graph expansion, and connected context packing."""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from cpp_context_engine.graph import CodeGraph
from cpp_context_engine.models import CodeSymbol, GraphRelation, SearchHit, SearchQuery
from cpp_context_engine.retrieval.protocols import ContextBundle, ContextItem, ContextPathStep
from cpp_context_engine.search import LexicalSearch, SymbolSearch, VectorSearch
from cpp_context_engine.storage import SourceReader, SymbolStore

_WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Hard limits and ranking controls for one retrieval operation."""

    search_limit: int = 30
    candidate_limit: int = 80
    seed_limit: int = 6
    rrf_k: int = 60
    graph_depth: int = 2
    max_expansion_steps: int = 2
    graph_node_budget: int = 30
    graph_edge_budget: int = 100
    per_node_edge_budget: int = 20
    graph_decay: float = 0.72
    hub_threshold: int = 12
    chars_per_token: float = 4.0
    max_context_chars: int | None = None
    relations: frozenset[GraphRelation] = frozenset(
        {
            GraphRelation.CALLS,
            GraphRelation.REFERENCES,
            GraphRelation.INHERITS,
            GraphRelation.OVERRIDES,
            GraphRelation.USES_TYPE,
            GraphRelation.CONTAINS,
        }
    )
    source_weights: Mapping[str, float] = field(
        default_factory=lambda: {"lexical": 1.0, "symbol": 1.15, "vector": 0.9}
    )

    def __post_init__(self) -> None:
        integer_limits = (
            self.search_limit,
            self.candidate_limit,
            self.seed_limit,
            self.rrf_k,
            self.graph_node_budget,
            self.graph_edge_budget,
            self.per_node_edge_budget,
        )
        if min(integer_limits) <= 0:
            raise ValueError("retrieval limits must be greater than zero")
        if self.graph_depth < 0 or self.max_expansion_steps < 0:
            raise ValueError("graph depth and expansion steps must not be negative")
        if not 0 < self.graph_decay <= 1:
            raise ValueError("graph decay must be in (0, 1]")
        if self.hub_threshold <= 0 or self.chars_per_token <= 0:
            raise ValueError("hub threshold and chars per token must be positive")
        if self.max_context_chars is not None and self.max_context_chars <= 0:
            raise ValueError("max context chars must be positive when set")
        if any(weight <= 0 for weight in self.source_weights.values()):
            raise ValueError("source weights must be positive")
        object.__setattr__(self, "source_weights", MappingProxyType(dict(self.source_weights)))


@dataclass(slots=True)
class _Candidate:
    hit: SearchHit
    reasons: list[str]
    path: tuple[ContextPathStep, ...] = ()
    is_seed: bool = False
    parent_id: str | None = None
    hub_degree: int = 0

    @property
    def symbol(self) -> CodeSymbol:
        return self.hit.symbol


class HybridRetriever:
    """Fuse independent searches and expand only high-value code-graph neighborhoods."""

    def __init__(
        self,
        *,
        lexical_search: LexicalSearch,
        symbol_search: SymbolSearch,
        vector_search: VectorSearch,
        symbol_store: SymbolStore,
        source_reader: SourceReader,
        graph: CodeGraph | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._searches = (
            ("lexical", lexical_search),
            ("symbol", symbol_search),
            ("vector", vector_search),
        )
        self._symbol_store = symbol_store
        self._source_reader = source_reader
        self._graph = graph
        self._config = config or RetrievalConfig()

    def retrieve(self, query: str, *, max_tokens: int) -> ContextBundle:
        """Run bounded search, expansion, source reading, and context assembly."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        diagnostics: list[str] = []
        candidates = self._fuse(query, diagnostics)
        if not candidates:
            return ContextBundle(query, (), "", 0, diagnostics=tuple(diagnostics))

        ranked = self._rerank(query, candidates, update_scores=False)
        seeds = ranked[: min(self._config.seed_limit, len(ranked))]
        for candidate in seeds:
            candidate.is_seed = True
            candidate.reasons.append("selected as a fused search seed")

        expanded = self._expand(seeds, candidates, diagnostics)
        reranked = self._rerank(query, expanded, update_scores=True)
        return self._pack(query, reranked, max_tokens, diagnostics)

    def _fuse(self, text: str, diagnostics: list[str]) -> dict[str, _Candidate]:
        query = SearchQuery(text, limit=self._config.search_limit)
        fused_scores: dict[str, float] = defaultdict(float)
        symbols: dict[str, CodeSymbol] = {}
        reasons: dict[str, list[str]] = defaultdict(list)

        successful_searches = 0
        for name, backend in self._searches:
            try:
                hits = list(backend.search(query))[: self._config.search_limit]
            except Exception as exc:  # adapters are an explicit failure boundary
                diagnostics.append(f"{name} search failed: {type(exc).__name__}")
                continue
            successful_searches += 1
            seen_in_backend: set[str] = set()
            weight = self._config.source_weights.get(name, 1.0)
            for rank, hit in enumerate(hits, start=1):
                symbol_id = _evidence_key(hit.symbol)
                if symbol_id in seen_in_backend:
                    continue
                seen_in_backend.add(symbol_id)
                fused_scores[symbol_id] += weight / (self._config.rrf_k + rank)
                symbols[symbol_id] = hit.symbol
                reasons[symbol_id].append(f"{name} rank {rank}")

        if successful_searches == 0:
            diagnostics.append("all candidate searches failed")
        ordered_ids = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
        ordered_ids = ordered_ids[: self._config.candidate_limit]
        return {
            symbol_id: _Candidate(
                hit=SearchHit(
                    symbol=symbols[symbol_id],
                    score=fused_scores[symbol_id],
                    source="hybrid",
                ),
                reasons=reasons[symbol_id],
            )
            for symbol_id in ordered_ids
        }

    def _expand(
        self,
        seeds: Sequence[_Candidate],
        candidates: dict[str, _Candidate],
        diagnostics: list[str],
    ) -> dict[str, _Candidate]:
        if self._graph is None or self._config.graph_depth == 0:
            return candidates

        queue = deque((seed, 0) for seed in seeds)
        visited = {_evidence_key(seed.symbol) for seed in seeds}
        edge_count = 0
        expanded_nodes = 0
        budget_exhausted = False
        hard_depth = min(self._config.graph_depth, self._config.max_expansion_steps)

        while queue and edge_count < self._config.graph_edge_budget:
            current, depth = queue.popleft()
            if depth >= hard_depth:
                continue
            if expanded_nodes >= self._config.graph_node_budget:
                budget_exhausted = True
                continue
            try:
                raw_edges = list(
                    self._graph.neighbors(
                        current.symbol.id,
                        relations=self._config.relations,
                        depth=1,
                    )
                )
            except Exception as exc:  # graph adapters are another explicit boundary
                diagnostics.append(
                    f"graph expansion failed for {current.symbol.id}: {type(exc).__name__}"
                )
                continue

            # Adapter row/insertion order must not decide which neighbors survive a hard budget.
            relevant = sorted(
                (edge for edge in raw_edges if edge.relation in self._config.relations),
                key=lambda edge: (edge.relation.value, edge.source_id, edge.target_id),
            )
            degree = len(relevant)
            if degree > self._config.per_node_edge_budget:
                budget_exhausted = True
            for edge in relevant[: self._config.per_node_edge_budget]:
                if (
                    edge_count >= self._config.graph_edge_budget
                    or expanded_nodes >= self._config.graph_node_budget
                ):
                    budget_exhausted = True
                    break
                edge_count += 1
                if edge.source_id == current.symbol.id:
                    neighbor_id = edge.target_id
                elif edge.target_id == current.symbol.id:
                    neighbor_id = edge.source_id
                else:
                    continue
                neighbor_key = f"{edge.build_variant}:{neighbor_id}"
                if neighbor_key in visited:
                    continue

                try:
                    symbol = self._symbol_store.get_symbol(  # type: ignore[call-arg]
                        neighbor_id, build_scope=(edge.build_variant,)
                    )
                except TypeError:
                    symbol = self._symbol_store.get_symbol(neighbor_id)
                if symbol is None:
                    continue
                neighbor_key = _evidence_key(symbol)
                if neighbor_key in visited:
                    continue
                visited.add(neighbor_key)
                expanded_nodes += 1
                # Neighbor traversal is bidirectional, but provenance must retain the
                # compiler-derived edge orientation instead of inventing a reverse edge.
                step = ContextPathStep(edge.source_id, edge.target_id, edge.relation)
                path = (*current.path, step)
                graph_score = current.hit.score * self._config.graph_decay
                candidate = candidates.get(neighbor_key)
                if candidate is None:
                    candidate = _Candidate(
                        hit=SearchHit(symbol, graph_score, "graph"),
                        reasons=[
                            f"{edge.relation.value} neighbor at depth {depth + 1}",
                            f"hub degree {degree}",
                        ],
                        path=path,
                        parent_id=_evidence_key(current.symbol),
                        hub_degree=degree,
                    )
                    candidates[neighbor_key] = candidate
                else:
                    candidate.path = path
                    candidate.parent_id = _evidence_key(current.symbol)
                    candidate.hub_degree = degree
                    candidate.reasons.append(f"reached through {edge.relation.value}")
                    if graph_score > candidate.hit.score:
                        candidate.hit = SearchHit(symbol, graph_score, candidate.hit.source)
                queue.append((candidate, depth + 1))

        if queue or budget_exhausted:
            diagnostics.append("graph expansion stopped at configured budget")
        return candidates

    def _rerank(
        self,
        query: str,
        candidates: dict[str, _Candidate],
        *,
        update_scores: bool,
    ) -> list[_Candidate]:
        query_terms = {term.casefold() for term in _WORD_PATTERN.findall(query)}
        reranked: list[tuple[float, _Candidate]] = []
        for candidate in candidates.values():
            symbol = candidate.symbol
            symbol_terms = {
                term.casefold()
                for term in _WORD_PATTERN.findall(
                    f"{symbol.qualified_name} {symbol.signature} {symbol.documentation}"
                )
            }
            overlap = len(query_terms & symbol_terms) / max(1, len(query_terms))
            exact_name = 1.0 if symbol.qualified_name.casefold() in query.casefold() else 0.0
            score = candidate.hit.score * (1.0 + 0.35 * overlap + 0.2 * exact_name)
            if candidate.hub_degree > self._config.hub_threshold:
                score /= math.sqrt(candidate.hub_degree / self._config.hub_threshold)
            reranked.append((score, candidate))
        ordered = sorted(
            reranked,
            key=lambda item: (item[0], item[1].is_seed, item[1].symbol.qualified_name),
            reverse=True,
        )
        if update_scores:
            for score, candidate in ordered:
                candidate.hit = SearchHit(candidate.symbol, score, candidate.hit.source)
        return [candidate for _, candidate in ordered]

    def _pack(
        self,
        query: str,
        ranked: Sequence[_Candidate],
        max_tokens: int,
        diagnostics: list[str],
    ) -> ContextBundle:
        char_budget = math.floor(max_tokens * self._config.chars_per_token)
        if self._config.max_context_chars is not None:
            char_budget = min(char_budget, self._config.max_context_chars)

        by_parent: dict[str, list[_Candidate]] = defaultdict(list)
        roots: list[_Candidate] = []
        for candidate in ranked:
            if candidate.parent_id is None or candidate.is_seed:
                roots.append(candidate)
            else:
                by_parent[candidate.parent_id].append(candidate)
        roots.sort(key=lambda item: item.hit.score, reverse=True)
        for children in by_parent.values():
            children.sort(key=lambda item: item.hit.score, reverse=True)

        ordered: list[_Candidate] = []
        seen: set[str] = set()
        for root in roots:
            component_queue = deque([root])
            while component_queue:
                candidate = component_queue.popleft()
                key = _evidence_key(candidate.symbol)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(candidate)
                component_queue.extend(by_parent.get(key, ()))

        rendered_parts: list[str] = []
        items: list[ContextItem] = []
        used_chars = 0
        truncated = False
        for candidate in ordered:
            try:
                source = self._source_reader.read_symbol(candidate.symbol)
            except Exception as exc:  # malformed or unavailable files must not abort a query
                diagnostics.append(
                    f"source read failed for {candidate.symbol.id}: {type(exc).__name__}"
                )
                continue

            reason = "; ".join(dict.fromkeys(candidate.reasons))
            prefix, suffix = self._render_shell(candidate, reason)
            remaining = char_budget - used_chars
            full = f"{prefix}{source}{suffix}"
            if len(full) > remaining:
                available_source = remaining - len(prefix) - len(suffix) - len("\n… [truncated]")
                if available_source <= 0:
                    truncated = True
                    break
                source = f"{source[:available_source]}\n… [truncated]"
                full = f"{prefix}{source}{suffix}"
                truncated = True

            path = candidate.path
            items.append(ContextItem(candidate.hit, source, reason, path))
            rendered_parts.append(full)
            used_chars += len(full)
            if truncated:
                break

        if len(items) < len(ordered):
            truncated = True
        rendered = "".join(rendered_parts)
        estimated_tokens = math.ceil(len(rendered) / self._config.chars_per_token)
        return ContextBundle(
            query=query,
            hits=tuple(item.hit for item in items),
            rendered_context=rendered,
            estimated_tokens=estimated_tokens,
            items=tuple(items),
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            truncated=truncated,
        )

    def _render_shell(self, candidate: _Candidate, reason: str) -> tuple[str, str]:
        symbol = candidate.symbol
        span = symbol.span
        display_path = span.path.as_posix()
        project_root = getattr(self._source_reader, "project_root", None)
        if project_root is not None:
            root = project_root.resolve(strict=False)
            resolved = (span.path if span.path.is_absolute() else root / span.path).resolve(
                strict=False
            )
            if resolved.is_relative_to(root):
                display_path = resolved.relative_to(root).as_posix()
        if candidate.path:
            path = " -> ".join(
                f"{step.source_id} -[{step.relation.value}]-> {step.target_id}"
                for step in candidate.path
            )
        else:
            path = "search seed"
        prefix = (
            f"### {symbol.qualified_name}\n"
            f"Symbol-ID: {symbol.id}\n"
            f"Build: {symbol.build_variant}\n"
            f"Location: {display_path}:{span.start_line}-{span.end_line}\n"
            f"Selected: {reason}\n"
            f"Path: {path}\n"
            "```cpp\n"
        )
        return prefix, "\n```\n\n"


def _evidence_key(symbol: CodeSymbol) -> str:
    return symbol.variant_id or f"{symbol.build_variant}:{symbol.id}"
