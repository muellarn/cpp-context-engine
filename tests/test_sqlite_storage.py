from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    CodeSymbol,
    GraphDirection,
    GraphEdge,
    GraphRelation,
    OccurrenceKind,
    SearchQuery,
    SourceSpan,
    SymbolKind,
    SymbolOccurrence,
    TranslationUnit,
)
from cpp_context_engine.storage.sqlite import _TRANSLATION_UNIT_DELETE_ORDER, SQLiteStore


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


def test_symbol_refresh_lookup_uses_v9_preference_index(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        indexes = {
            row[1] for row in store._connection.execute("PRAGMA index_list(symbol_variants)")
        }  # noqa: SLF001 - verify the performance migration itself
        plan = store._connection.execute(  # noqa: SLF001 - inspect SQLite planner evidence
            """
            EXPLAIN QUERY PLAN
            SELECT symbol_id, snapshot_json FROM symbol_variants
            WHERE project_id = ? AND symbol_id IN (?)
            ORDER BY symbol_id, is_definition DESC, build_variant, translation_unit_id
            """,
            (1, "symbol"),
        ).fetchall()

    assert "symbol_variants_symbol_preference" in indexes
    assert any("symbol_variants_symbol_preference" in row[3] for row in plan)


def test_symbol_refresh_batches_preference_reads(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        project_id = store._project_id(root)  # noqa: SLF001
        statements: list[str] = []
        store._connection.set_trace_callback(statements.append)  # noqa: SLF001
        store._refresh_symbols(project_id, {"file-a", "symbol-alpha"})  # noqa: SLF001
        store._connection.set_trace_callback(None)  # noqa: SLF001

        assert store.get_symbol("symbol-alpha") is not None

    assert (
        sum(
            "SELECT symbol_id, snapshot_json FROM symbol_variants" in statement
            for statement in statements
        )
        == 1
    )


def test_canonical_symbol_batch_skips_identical_upserts(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        project_id = store._project_id(root)  # noqa: SLF001
        before = store._connection.total_changes  # noqa: SLF001
        store._put_canonical_symbols(  # noqa: SLF001
            project_id, batch.symbols, prefer_definition=False
        )

        assert store._connection.total_changes == before  # noqa: SLF001


def test_translation_unit_cascade_tables_have_v9_lookup_indexes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    expected = {
        "edges": "edges_tu",
        "symbol_variants": "symbol_variants_tu",
        "cfg_blocks": "cfg_blocks_tu",
        "cfg_elements": "cfg_elements_tu",
        "cfg_edges": "cfg_edges_tu",
        "call_targets": "call_targets_tu",
        "data_flow_analyses": "data_flow_analyses_tu",
        "memory_locations": "memory_locations_tu",
        "data_accesses": "data_accesses_tu",
        "data_flow_evidence": "data_flow_evidence_tu",
        "function_summaries": "function_summaries_tu",
    }

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        indexes = {
            table: {row[1] for row in store._connection.execute(f"PRAGMA index_list({table})")}
            for table in expected
        }  # noqa: SLF001 - verify foreign-key cascade performance indexes

    assert all(index in indexes[table] for table, index in expected.items())


def test_v10_removes_redundant_translation_unit_symbol_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        columns = {
            row[1]
            for row in store._connection.execute(  # noqa: SLF001 - schema regression
                "PRAGMA table_info(translation_unit_symbols)"
            )
        }

    assert "snapshot_json" not in columns


def test_every_foreign_key_has_a_child_lookup_index(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    missing: list[tuple[str, tuple[str, ...]]] = []

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        tables = [
            row[0]
            for row in store._connection.execute(  # noqa: SLF001 - schema performance audit
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            foreign_keys: dict[int, list[tuple[int, str]]] = {}
            for row in store._connection.execute(  # noqa: SLF001 - schema performance audit
                f"PRAGMA foreign_key_list({table})"
            ):
                foreign_keys.setdefault(row[0], []).append((row[1], row[3]))
            indexes = [
                tuple(
                    column[2]
                    for column in store._connection.execute(  # noqa: SLF001
                        f"PRAGMA index_info({index[1]})"
                    )
                )
                for index in store._connection.execute(  # noqa: SLF001
                    f"PRAGMA index_list({table})"
                )
            ]
            for key in foreign_keys.values():
                columns = tuple(column for _, column in sorted(key))
                if not any(index[: len(columns)] == columns for index in indexes):
                    missing.append((table, columns))

    assert missing == []


def test_bulk_tu_delete_order_covers_every_tu_scoped_fact_table(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        scoped = {
            row[0]
            for row in store._connection.execute(  # noqa: SLF001 - schema audit
                """
                SELECT tables.name
                FROM sqlite_master tables
                JOIN pragma_table_info(tables.name) columns
                WHERE tables.type = 'table' AND columns.name = 'translation_unit_id'
                  AND tables.name != 'translation_units'
                """
            )
        }

    assert set(_TRANSLATION_UNIT_DELETE_ORDER) == scoped


def test_graph_traversal_enforces_direction_depth_and_limits(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)
    source = root / "source.cpp"
    beta = CodeSymbol(
        id="symbol-beta",
        qualified_name="beta",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(source, 1, 1),
        signature="int beta()",
    )
    gamma = CodeSymbol(
        id="symbol-gamma",
        qualified_name="gamma",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(source, 1, 1),
        signature="int gamma()",
    )

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        store.put_symbols((beta, gamma))
        store.put_edges(
            (
                GraphEdge("symbol-alpha", beta.id, GraphRelation.CALLS, "unit-a"),
                GraphEdge(beta.id, gamma.id, GraphRelation.CALLS, "unit-a"),
                GraphEdge(gamma.id, "symbol-alpha", GraphRelation.REFERENCES, "unit-a"),
            )
        )

        assert store.neighbors(
            beta.id,
            relations=frozenset({GraphRelation.CALLS}),
            direction=GraphDirection.INCOMING,
        ) == (GraphEdge("symbol-alpha", beta.id, GraphRelation.CALLS),)
        assert store.neighbors(
            beta.id,
            relations=frozenset({GraphRelation.CALLS}),
            direction=GraphDirection.OUTGOING,
        ) == (GraphEdge(beta.id, gamma.id, GraphRelation.CALLS),)
        assert store.neighbors(
            "symbol-alpha",
            relations=frozenset({GraphRelation.CALLS}),
            direction=GraphDirection.OUTGOING,
            depth=2,
        ) == (
            GraphEdge("symbol-alpha", beta.id, GraphRelation.CALLS),
            GraphEdge(beta.id, gamma.id, GraphRelation.CALLS),
        )
        assert (
            len(
                store.neighbors(
                    beta.id,
                    direction=GraphDirection.BOTH,
                    max_edges=1,
                    per_node_limit=1,
                )
            )
            == 1
        )

        with pytest.raises(ValueError, match="edge limit"):
            store.neighbors(beta.id, max_edges=0)


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


def test_cosine_search_is_mathematically_ordered_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        store.put_embedding("symbol-alpha", "fixture", [1.0, 1.0])
        store.put_embedding("file-a", "fixture", [-1.0, 0.0])

        assert store.missing_embedding_symbol_ids("fixture") == ()
        assert store.embedding_count("fixture") == 2

        all_hits = store.search_vector([2.0, 2.0], model="fixture")
        assert [hit.symbol.id for hit in all_hits] == ["symbol-alpha", "file-a"]
        assert all_hits[1].score == pytest.approx(-(2**-0.5))

        decoded = 0
        original = SQLiteStore._variant_row_to_symbol

        def count_decode(cls: type[SQLiteStore], row: object) -> CodeSymbol:
            nonlocal decoded
            decoded += 1
            return original(row)  # type: ignore[arg-type]

        monkeypatch.setattr(SQLiteStore, "_variant_row_to_symbol", classmethod(count_decode))
        statements: list[str] = []
        store._connection.set_trace_callback(statements.append)  # noqa: SLF001
        hits = store.search_vector([2.0, 2.0], model="fixture", limit=1)
        store._connection.set_trace_callback(None)  # noqa: SLF001

        assert [hit.symbol.id for hit in hits] == ["symbol-alpha"]
        assert hits[0].score == pytest.approx(1.0)
        assert decoded == 1
        scoring = next(statement for statement in statements if "_cpp_context_cosine" in statement)
        assert "snapshot_json" not in scoring
        with pytest.raises(ValueError, match="magnitude"):
            store.search_vector([0.0, 0.0], model="fixture")
        with pytest.raises(ValueError, match="dimension"):
            store.put_embedding("file-a", "fixture", [1.0, 2.0, 3.0])


def test_embedding_batch_validates_atomically(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        with pytest.raises(ValueError, match="dimension"):
            store.put_embeddings(
                (("symbol-alpha", [1.0, 0.0]), ("file-a", [1.0, 0.0, 0.0])),
                "fixture",
            )
        assert store.embedding_count("fixture") == 0

        store.put_embeddings((("symbol-alpha", [1.0, 0.0]), ("file-a", [0.0, 1.0])), "fixture")
        assert store.embedding_count("fixture") == 2


def test_bulk_symbol_lookup_preserves_order_duplicates_and_missing(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        statements: list[str] = []
        store._connection.set_trace_callback(statements.append)  # noqa: SLF001
        symbols = store.get_symbols(("symbol-alpha", "missing", "file-a", "symbol-alpha"))
        store._connection.set_trace_callback(None)  # noqa: SLF001

    assert [symbol.id if symbol is not None else None for symbol in symbols] == [
        "symbol-alpha",
        None,
        "file-a",
        "symbol-alpha",
    ]
    assert sum("FROM symbol_variants" in statement for statement in statements) == 1


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
