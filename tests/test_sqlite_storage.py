from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    CodeSymbol,
    GraphEdge,
    GraphRelation,
    OccurrenceKind,
    SearchQuery,
    SourceSpan,
    SymbolKind,
    SymbolOccurrence,
    TranslationUnit,
)
from cpp_context_engine.storage.sqlite import SQLiteStore


def _batch(root: Path) -> IngestionBatch:
    path = root / "source.cpp"
    path.write_text("int alpha() { return 7; }\n", encoding="utf-8")
    configuration = BuildConfiguration(
        id="build-a",
        source_path=path,
        directory=root,
        arguments=("c++", "source.cpp"),
        command_hash="command-hash",
    )
    unit = TranslationUnit(
        id="unit-a",
        build_configuration_id=configuration.id,
        source_path=path,
        content_hash="content-hash",
        dependencies=((path, "content-hash"),),
    )
    parent = CodeSymbol(
        id="file-a",
        qualified_name="source.cpp",
        kind=SymbolKind.FILE,
        span=SourceSpan(path, 1, 1, 1, 26),
        signature="source.cpp",
        build_configuration_id=configuration.id,
        translation_unit_id=unit.id,
    )
    symbol = CodeSymbol(
        id="symbol-alpha",
        qualified_name="alpha",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(path, 1, 1, 1, 26),
        signature="int alpha()",
        documentation="Returns the important answer.",
        source_text="int alpha() { return 7; }",
        source_hash="source-hash",
        build_configuration_id=configuration.id,
        translation_unit_id=unit.id,
        metadata={"is_definition": True},
    )
    occurrence = SymbolOccurrence(
        id="occurrence-alpha",
        symbol_id=symbol.id,
        span=symbol.span,
        kind=OccurrenceKind.DEFINITION,
        translation_unit_id=unit.id,
    )
    edge = GraphEdge(parent.id, symbol.id, GraphRelation.CONTAINS, unit.id)
    return IngestionBatch((configuration,), (unit,), (parent, symbol), (occurrence,), (edge,))


def test_schema_round_trip_fts_graph_and_occurrences(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch, current_translation_unit_ids=frozenset({"unit-a"}))

        symbol = store.get_symbol("symbol-alpha")
        assert symbol is not None
        assert symbol.source_text == "int alpha() { return 7; }"
        assert store.search(SearchQuery("important answer"))[0].symbol.id == symbol.id
        assert store.search(SearchQuery("alpha return"))[0].symbol.id == symbol.id
        assert store.search_symbols(SearchQuery("alpha"))[0].symbol.id == symbol.id
        assert store.occurrences(symbol.id)[0].kind == OccurrenceKind.DEFINITION
        assert store.neighbors("file-a") == (
            GraphEdge("file-a", symbol.id, GraphRelation.CONTAINS),
        )


def test_symbol_search_weights_names_above_signature_only_matches(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)
    source = root / "source.cpp"
    name_match = CodeSymbol(
        id="name-match",
        qualified_name="zzz::needle::handler",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(source, 1, 1),
        signature="void handler()",
    )
    signature_match = CodeSymbol(
        id="signature-match",
        qualified_name="aaa",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(source, 1, 1),
        signature="needle",
    )

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        store.put_symbols((name_match, signature_match))

        hits = store.search_symbols(SearchQuery("needle"))

    assert [hit.symbol.id for hit in hits[:2]] == ["name-match", "signature-match"]
    assert hits[0].score > hits[1].score


def test_cosine_search_is_mathematically_ordered_and_validated(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        store.put_embedding("symbol-alpha", "fixture", [1.0, 1.0])
        store.put_embedding("file-a", "fixture", [-1.0, 0.0])

        assert store.missing_embedding_symbol_ids("fixture") == ()
        assert store.embedding_count("fixture") == 2

        hits = store.search_vector([2.0, 2.0], model="fixture")

        assert hits[0].symbol.id == "symbol-alpha"
        assert hits[0].score == pytest.approx(1.0)
        assert hits[1].score == pytest.approx(-(2**-0.5))
        with pytest.raises(ValueError, match="magnitude"):
            store.search_vector([0.0, 0.0], model="fixture")
        with pytest.raises(ValueError, match="dimension"):
            store.put_embedding("file-a", "fixture", [1.0, 2.0, 3.0])


def test_changed_source_invalidates_its_stored_embedding(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        store.put_embedding("symbol-alpha", "fixture", [1.0, 0.0])
        changed_symbol = replace(
            batch.symbols[1], source_hash="changed-hash", source_text="int alpha() { return 8; }"
        )
        changed_batch = replace(batch, symbols=(batch.symbols[0], changed_symbol))
        store.apply_ingestion(root, changed_batch)

        assert store.search_vector([1.0, 0.0], model="fixture") == ()


def test_changed_embedding_input_invalidates_vector_even_when_source_hash_is_same(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        store.put_embedding("symbol-alpha", "fixture", [1.0, 0.0])
        documented = replace(batch.symbols[1], documentation="New semantic description")
        store.apply_ingestion(root, replace(batch, symbols=(batch.symbols[0], documented)))

        assert "symbol-alpha" in store.missing_embedding_symbol_ids("fixture")


def test_stale_unit_removal_cascades_symbols_and_search_rows(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch, current_translation_unit_ids=frozenset({"unit-a"}))
        store.apply_ingestion(
            root,
            IngestionBatch((), (), (), (), ()),
            current_translation_unit_ids=frozenset(),
        )

        assert store.symbols() == ()
        assert store.translation_unit_states() == {}
        assert store.search(SearchQuery("alpha")) == ()
