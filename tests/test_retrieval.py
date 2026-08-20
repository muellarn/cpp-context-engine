from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cpp_context_engine.models import (
    CodeSymbol,
    GraphEdge,
    GraphRelation,
    SearchHit,
    SearchQuery,
    SourceSpan,
    SymbolKind,
)
from cpp_context_engine.retrieval.hybrid import HybridRetriever, RetrievalConfig
from cpp_context_engine.storage.source import FilesystemSourceReader, SourceReadError


def symbol(symbol_id: str, name: str | None = None, line: int = 1) -> CodeSymbol:
    return CodeSymbol(
        id=symbol_id,
        qualified_name=name or f"ns::{symbol_id}",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(Path(f"{symbol_id}.cpp"), line, line + 2),
        signature=f"void {symbol_id}()",
    )


class SearchStub:
    def __init__(self, symbols: Sequence[CodeSymbol], *, failure: Exception | None = None) -> None:
        self.symbols = symbols
        self.failure = failure

    def search(self, query: SearchQuery) -> Sequence[SearchHit]:
        if self.failure:
            raise self.failure
        return [SearchHit(item, 1.0 / rank, "stub") for rank, item in enumerate(self.symbols, 1)]


class StoreStub:
    def __init__(self, symbols: Sequence[CodeSymbol]) -> None:
        self.symbols = {item.id: item for item in symbols}

    def get_symbol(self, symbol_id: str) -> CodeSymbol | None:
        return self.symbols.get(symbol_id)


class SourceStub:
    def __init__(self, text: str = "return validate(packet);") -> None:
        self.text = text

    def read_symbol(self, requested: CodeSymbol) -> str:
        return f"// {requested.id}\n{self.text}"


class GraphStub:
    def __init__(self, edges: Sequence[GraphEdge]) -> None:
        self.edges = edges

    def neighbors(
        self,
        symbol_id: str,
        *,
        relations: frozenset[GraphRelation] | None = None,
        depth: int = 1,
    ) -> Sequence[GraphEdge]:
        assert depth == 1
        return [
            edge
            for edge in self.edges
            if symbol_id in (edge.source_id, edge.target_id)
            and (relations is None or edge.relation in relations)
        ]

    def put_edges(self, edges: Sequence[GraphEdge]) -> None:
        self.edges = [*self.edges, *edges]


def test_rrf_deduplicates_and_expands_a_connected_component() -> None:
    parser = symbol("parser", "net::PacketParser::parse")
    unrelated = symbol("unrelated")
    validate = symbol("validate", "net::validateHeader")
    decode = symbol("decode", "net::MessageDecoder::decode")
    all_symbols = [parser, unrelated, validate, decode]
    graph = GraphStub(
        [
            GraphEdge("parser", "validate", GraphRelation.CALLS),
            GraphEdge("validate", "decode", GraphRelation.CALLS),
        ]
    )
    retriever = HybridRetriever(
        lexical_search=SearchStub([parser, unrelated]),
        symbol_search=SearchStub([parser]),
        vector_search=SearchStub([parser, validate, unrelated]),
        symbol_store=StoreStub(all_symbols),
        source_reader=SourceStub(),
        graph=graph,
        config=RetrievalConfig(seed_limit=1, graph_depth=2, max_expansion_steps=2),
    )

    bundle = retriever.retrieve("PacketParser parse validation decode", max_tokens=1_000)

    ids = [item.hit.symbol.id for item in bundle.items]
    assert ids[:3] == ["parser", "validate", "decode"]
    assert ids.count("parser") == 1
    assert bundle.items[1].path[0].relation is GraphRelation.CALLS
    assert bundle.items[2].path[-1].target_id == "decode"
    assert "Location: parser.cpp:1-3" in bundle.rendered_context
    assert "selected as a fused search seed" in bundle.items[0].reason


def test_graph_hub_is_penalized_and_expansion_respects_node_budget() -> None:
    root = symbol("root")
    neighbors = [symbol(f"child-{index}") for index in range(20)]
    graph = GraphStub([GraphEdge("root", item.id, GraphRelation.REFERENCES) for item in neighbors])
    retriever = HybridRetriever(
        lexical_search=SearchStub([root]),
        symbol_search=SearchStub([]),
        vector_search=SearchStub([]),
        symbol_store=StoreStub([root, *neighbors]),
        source_reader=SourceStub(),
        graph=graph,
        config=RetrievalConfig(
            seed_limit=1,
            graph_depth=1,
            graph_node_budget=2,
            hub_threshold=4,
            per_node_edge_budget=20,
        ),
    )

    bundle = retriever.retrieve("root", max_tokens=1_000)

    graph_items = [item for item in bundle.items if item.hit.source == "graph"]
    assert len(graph_items) == 2
    assert all("hub degree 20" in item.reason for item in graph_items)
    assert all(item.hit.score < bundle.items[0].hit.score for item in graph_items)
    assert "graph expansion stopped at configured budget" in bundle.diagnostics


def test_context_packing_enforces_character_derived_token_budget() -> None:
    root = symbol("root")
    retriever = HybridRetriever(
        lexical_search=SearchStub([root]),
        symbol_search=SearchStub([]),
        vector_search=SearchStub([]),
        symbol_store=StoreStub([root]),
        source_reader=SourceStub("x" * 2_000),
        config=RetrievalConfig(seed_limit=1, chars_per_token=4),
    )

    bundle = retriever.retrieve("root", max_tokens=100)

    assert len(bundle.rendered_context) <= 400
    assert bundle.estimated_tokens <= 100
    assert bundle.truncated is True
    assert bundle.items[0].source_text.endswith("… [truncated]")


def test_partial_search_failure_is_reported_without_hiding_other_results() -> None:
    root = symbol("root")
    retriever = HybridRetriever(
        lexical_search=SearchStub([], failure=RuntimeError("private backend detail")),
        symbol_search=SearchStub([root]),
        vector_search=SearchStub([]),
        symbol_store=StoreStub([root]),
        source_reader=SourceStub(),
    )

    bundle = retriever.retrieve("root", max_tokens=500)

    assert [hit.symbol.id for hit in bundle.hits] == ["root"]
    assert bundle.diagnostics == ("lexical search failed: RuntimeError",)
    assert "private backend detail" not in bundle.rendered_context


def test_filesystem_source_reader_rejects_project_escape(tmp_path: Path) -> None:
    reader = FilesystemSourceReader(tmp_path)
    escaped = CodeSymbol(
        id="escaped",
        qualified_name="escaped",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(Path("../secret.cpp"), 1, 1),
    )

    try:
        reader.read_symbol(escaped)
    except SourceReadError as exc:
        assert "escapes project root" in str(exc)
    else:
        raise AssertionError("project escape was not rejected")


def test_filesystem_source_reader_returns_only_symbol_lines(tmp_path: Path) -> None:
    source_file = tmp_path / "parser.cpp"
    source_file.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    requested = CodeSymbol(
        id="slice",
        qualified_name="slice",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(Path("parser.cpp"), 2, 3),
    )

    assert FilesystemSourceReader(tmp_path).read_symbol(requested) == "two\nthree"
