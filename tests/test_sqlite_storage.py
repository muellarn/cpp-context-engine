from __future__ import annotations

import math
import sqlite3
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import cpp_context_engine.storage.sqlite as sqlite_storage
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    BuildScope,
    CodeSymbol,
    GraphDirection,
    GraphEdge,
    GraphRelation,
    IndexProfile,
    OccurrenceKind,
    SearchQuery,
    SourceSpan,
    SymbolKind,
    SymbolOccurrence,
    TranslationUnit,
)
from cpp_context_engine.search.vector import SQLiteVectorSearch
from cpp_context_engine.storage.sqlite import (
    _TRANSLATION_UNIT_DELETE_ORDER,
    VECTOR_ENCODING_RAW_F64LE_V1,
    VECTOR_ENCODING_ZLIB_F64LE_V1,
    SQLiteStore,
    _decode_stored_vector,
    _encode_vector_blob,
    _TrustedEmbeddingAttachment,
)


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


class _RecordingEmbeddingProvider:
    def __init__(
        self,
        *,
        model_id: str = "fixture",
        configuration_id: str = "fixture-config",
        fail_on_call: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.configuration_id = configuration_id
        self.fail_on_call = fail_on_call
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: list[str]) -> tuple[tuple[float, float], ...]:
        self.calls.append(tuple(texts))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected embedding failure")
        return tuple((float(index + 1), 1.0) for index, _text in enumerate(texts))


@pytest.mark.parametrize(
    "values",
    [
        (0.0, -0.0, 5e-324, -5e-324),
        tuple(float(index) / 7.0 for index in range(128)),
        (1.0,) * 128,
    ],
)
def test_vector_storage_encoding_round_trips_exact_float64_bytes(
    values: tuple[float, ...],
) -> None:
    raw = struct.pack(f"<{len(values)}d", *values)
    encoding, stored = _encode_vector_blob(raw)

    assert encoding in {VECTOR_ENCODING_RAW_F64LE_V1, VECTOR_ENCODING_ZLIB_F64LE_V1}
    assert _decode_stored_vector(stored, encoding, len(values)) == raw
    assert _encode_vector_blob(raw) == (encoding, stored)


def test_vector_storage_encoding_falls_back_when_compression_is_not_smaller() -> None:
    raw = bytes(range(64))
    assert _encode_vector_blob(raw) == (VECTOR_ENCODING_RAW_F64LE_V1, raw)


@pytest.mark.parametrize(
    ("blob", "encoding", "dimensions"),
    [
        (b"short", VECTOR_ENCODING_RAW_F64LE_V1, 1),
        (zlib.compress(b"short"), VECTOR_ENCODING_ZLIB_F64LE_V1, 1),
        (zlib.compress(b"12345678")[:-1], VECTOR_ENCODING_ZLIB_F64LE_V1, 1),
        (zlib.compress(b"12345678") + b"trailing", VECTOR_ENCODING_ZLIB_F64LE_V1, 1),
        (zlib.compress(b"123456789"), VECTOR_ENCODING_ZLIB_F64LE_V1, 1),
        (zlib.compress(b"0" * 1_000_000), VECTOR_ENCODING_ZLIB_F64LE_V1, 1),
        (b"12345678", 99, 1),
    ],
)
def test_vector_storage_decoder_rejects_malformed_or_oversized_payloads(
    blob: bytes, encoding: int, dimensions: int
) -> None:
    with pytest.raises(RuntimeError):
        _decode_stored_vector(blob, encoding, dimensions)


def _downgrade_embedding_schema_to_v13(database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = OFF")
    vectors = connection.execute(
        """
        SELECT project_id, model, configuration_id, dimensions, content_hash,
               content_text, magnitude, vector_encoding, vector
        FROM embedding_vectors
        """
    ).fetchall()
    references = connection.execute("SELECT * FROM variant_embeddings").fetchall()
    connection.executescript(
        """
        DROP TABLE variant_embeddings;
        DROP TABLE embedding_vectors;
        CREATE TABLE embedding_vectors (
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            model TEXT NOT NULL,
            configuration_id TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            content_text TEXT NOT NULL,
            magnitude REAL NOT NULL,
            vector BLOB NOT NULL,
            PRIMARY KEY (project_id, model, configuration_id, dimensions, content_hash),
            CHECK (dimensions > 0), CHECK (magnitude > 0),
            CHECK (typeof(vector) = 'blob'), CHECK (length(vector) = dimensions * 8)
        );
        CREATE TABLE variant_embeddings (
            project_id INTEGER NOT NULL,
            variant_id TEXT NOT NULL,
            model TEXT NOT NULL,
            configuration_id TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            PRIMARY KEY (project_id, variant_id, model, configuration_id),
            FOREIGN KEY (project_id, variant_id)
                REFERENCES symbol_variants(project_id, id) ON DELETE CASCADE,
            FOREIGN KEY (project_id, model, configuration_id, dimensions, content_hash)
                REFERENCES embedding_vectors(
                    project_id, model, configuration_id, dimensions, content_hash
                )
        );
        CREATE INDEX variant_embeddings_search ON variant_embeddings(
            project_id, model, configuration_id, dimensions, variant_id
        );
        CREATE INDEX variant_embeddings_content ON variant_embeddings(
            project_id, model, configuration_id, dimensions, content_hash
        );
        PRAGMA user_version = 13;
        """
    )
    connection.executemany(
        "INSERT INTO embedding_vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            (
                row["project_id"],
                row["model"],
                row["configuration_id"],
                row["dimensions"],
                row["content_hash"],
                row["content_text"],
                row["magnitude"],
                _decode_stored_vector(row["vector"], row["vector_encoding"], row["dimensions"]),
            )
            for row in vectors
        ),
    )
    connection.executemany(
        "INSERT INTO variant_embeddings VALUES (?, ?, ?, ?, ?, ?)",
        (tuple(row) for row in references),
    )
    connection.commit()
    connection.close()


def _open_migration_store(database: Path, root: Path) -> SQLiteStore:
    store = SQLiteStore.__new__(SQLiteStore)
    store.path = database
    store.project_root = root
    store.build_scope = BuildScope.single()
    store._connection = sqlite3.connect(database)  # noqa: SLF001 - migration fixture
    store._connection.row_factory = sqlite3.Row  # noqa: SLF001
    store._connection.execute("PRAGMA foreign_keys = ON")  # noqa: SLF001
    return store


def test_v13_embedding_migration_preserves_canonical_vectors_and_ranking(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "index.db"
    values = [1.0] * 128
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        store.put_embedding("symbol-alpha", "fixture", values)
    _downgrade_embedding_schema_to_v13(database)

    with SQLiteStore(database, project_root=root) as store:
        row = store._connection.execute(  # noqa: SLF001
            "SELECT dimensions, magnitude, vector_encoding, vector FROM embedding_vectors"
        ).fetchone()
        assert row is not None
        assert row["vector_encoding"] == VECTOR_ENCODING_ZLIB_F64LE_V1
        assert _decode_stored_vector(row["vector"], row["vector_encoding"], 128) == struct.pack(
            "<128d", *values
        )
        assert row["magnitude"] == math.sqrt(128.0)
        hits = store.search_vector(values, model="fixture")
        assert [hit.symbol.id for hit in hits] == ["symbol-alpha"]
        assert hits[0].score == pytest.approx(1.0)
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001


@pytest.mark.parametrize("stage", ["create", "copy", "index", "foreign-key", "publication"])
def test_v13_embedding_migration_failure_rolls_back_every_stage(tmp_path: Path, stage: str) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        store.put_embedding("symbol-alpha", "fixture", [1.0] * 128)
    _downgrade_embedding_schema_to_v13(database)

    store = _open_migration_store(database, root)

    def fail_at(candidate: str) -> None:
        if candidate == stage:
            raise RuntimeError(f"injected {stage} failure")

    store._embedding_migration_checkpoint = fail_at  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match=f"injected {stage} failure"):
        store._migrate_v14()  # noqa: SLF001
    assert not store._connection.in_transaction  # noqa: SLF001
    assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 13  # noqa: SLF001
    assert "vector_encoding" not in {  # noqa: SLF001
        row[1] for row in store._connection.execute("PRAGMA table_info(embedding_vectors)")
    }
    assert store._connection.execute("SELECT count(*) FROM embedding_vectors").fetchone()[0] == 1  # noqa: SLF001
    store._connection.close()  # noqa: SLF001


def test_v13_embedding_migration_rejects_embedding_foreign_key_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        store.put_embedding("symbol-alpha", "fixture", [1.0] * 128)
    _downgrade_embedding_schema_to_v13(database)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("UPDATE variant_embeddings SET content_hash = 'missing'")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="embedding migration foreign-key failure"):
        SQLiteStore(database, project_root=root)

    unchanged = sqlite3.connect(database)
    try:
        assert unchanged.execute("PRAGMA user_version").fetchone()[0] == 13
        assert "vector_encoding" not in {
            row[1] for row in unchanged.execute("PRAGMA table_info(embedding_vectors)")
        }
    finally:
        unchanged.close()


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


def _create_v12_profile_fixture(database: Path, root: Path) -> None:
    """Create a structurally accurate v12 database with representative TU rows."""

    batch = _batch(root)
    native_unit = replace(
        batch.translation_units[0],
        analysis_backend="clang-libtooling",
        advanced_facts_complete=True,
    )
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, replace(batch, translation_units=(native_unit,)))
        with store._connection:  # noqa: SLF001 - construct the legacy schema fixture
            store._connection.execute(  # noqa: SLF001
                """
                INSERT INTO translation_units(
                    project_id, id, build_configuration_id, source_path, content_hash,
                    diagnostics_json, indexed_at, build_variant, analysis_backend,
                    advanced_facts_complete, index_profile, navigation_facts_complete,
                    cfg_facts_complete, data_flow_facts_complete, summary_facts_complete
                )
                SELECT project_id, 'unit-baseline', build_configuration_id, source_path,
                       content_hash, diagnostics_json, indexed_at, build_variant,
                       'baseline', 0, 'full', 0, 0, 0, 0
                FROM translation_units WHERE id = ?
                """,
                (native_unit.id,),
            )
            for column in (
                "summary_facts_complete",
                "data_flow_facts_complete",
                "cfg_facts_complete",
                "navigation_facts_complete",
                "index_profile",
            ):
                store._connection.execute(  # noqa: SLF001
                    f'ALTER TABLE translation_units DROP COLUMN "{column}"'
                )
            store._connection.execute(  # noqa: SLF001
                'ALTER TABLE build_variants DROP COLUMN "index_profile"'
            )
            store._connection.execute("PRAGMA user_version = 12")  # noqa: SLF001


def test_v13_migration_upgrades_real_v12_profile_and_coverage_rows(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "legacy-v12.db"
    _create_v12_profile_fixture(database, root)

    with sqlite3.connect(database) as legacy:
        assert legacy.execute("PRAGMA user_version").fetchone()[0] == 12
        assert "index_profile" not in {
            row[1] for row in legacy.execute("PRAGMA table_info(build_variants)")
        }
        assert {
            "index_profile",
            "navigation_facts_complete",
            "cfg_facts_complete",
            "data_flow_facts_complete",
            "summary_facts_complete",
        }.isdisjoint(row[1] for row in legacy.execute("PRAGMA table_info(translation_units)"))

    with SQLiteStore(database, project_root=root) as migrated:
        assert migrated._connection.execute("PRAGMA user_version").fetchone()[0] == 14  # noqa: SLF001
        assert "vector_encoding" in {  # noqa: SLF001
            row[1] for row in migrated._connection.execute("PRAGMA table_info(embedding_vectors)")
        }
        states = migrated.translation_unit_states(root)
        build_profiles = migrated.build_index_profiles(root)

    assert build_profiles == {"default": IndexProfile.FULL}
    native = states["unit-a"]
    assert native.index_profile is IndexProfile.FULL
    assert native.navigation_facts_complete
    assert native.cfg_facts_complete
    assert native.data_flow_facts_complete
    assert native.summary_facts_complete
    baseline = states["unit-baseline"]
    assert baseline.index_profile is IndexProfile.FULL
    assert not baseline.navigation_facts_complete
    assert not baseline.cfg_facts_complete
    assert not baseline.data_flow_facts_complete
    assert not baseline.summary_facts_complete


def test_v13_migration_failure_rolls_back_all_profile_columns(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "legacy-v12.db"
    _create_v12_profile_fixture(database, root)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    altered = 0

    def deny_second_alter(
        action: int, _arg1: str | None, _arg2: str | None, _db: str | None, _source: str | None
    ) -> int:
        nonlocal altered
        if action == sqlite3.SQLITE_ALTER_TABLE:
            altered += 1
            if altered == 2:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(deny_second_alter)
    store = SQLiteStore.__new__(SQLiteStore)
    store._connection = connection  # noqa: SLF001 - exercise migration transaction directly
    with pytest.raises(sqlite3.DatabaseError):
        store._migrate_v13()  # noqa: SLF001
    connection.set_authorizer(None)

    assert connection.execute("PRAGMA user_version").fetchone()[0] == 12
    assert "index_profile" not in {
        row[1] for row in connection.execute("PRAGMA table_info(build_variants)")
    }
    assert "index_profile" not in {
        row[1] for row in connection.execute("PRAGMA table_info(translation_units)")
    }
    connection.close()


def test_combined_v12_to_v14_failure_rolls_back_profile_and_vector_schema(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "legacy-v12.db"
    _create_v12_profile_fixture(database, root)
    _downgrade_embedding_schema_to_v13(database)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 12")
    connection.commit()
    connection.close()

    store = _open_migration_store(database, root)

    def fail_publication(stage: str) -> None:
        if stage == "publication":
            raise RuntimeError("injected publication failure")

    store._embedding_migration_checkpoint = fail_publication  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected publication failure"):
        store._migrate_v13_and_v14()  # noqa: SLF001

    assert not store._connection.in_transaction  # noqa: SLF001
    assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 12  # noqa: SLF001
    assert "vector_encoding" not in {  # noqa: SLF001
        row[1] for row in store._connection.execute("PRAGMA table_info(embedding_vectors)")
    }
    assert "index_profile" not in {  # noqa: SLF001
        row[1] for row in store._connection.execute("PRAGMA table_info(translation_units)")
    }
    store._connection.close()  # noqa: SLF001


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
        with pytest.raises(ValueError, match="finite magnitude"):
            store.put_embedding("file-a", "overflow", [1e308, 1e308])
        with pytest.raises(ValueError, match="finite magnitude"):
            store.search_vector([1e308, 1e308], model="fixture")
        with pytest.raises(ValueError, match="dimension"):
            store.put_embedding("file-a", "fixture", [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("encoding", "blob", "magnitude"),
    [
        (VECTOR_ENCODING_RAW_F64LE_V1, struct.pack("<2d", float("nan"), 1.0), 1.0),
        (
            VECTOR_ENCODING_ZLIB_F64LE_V1,
            zlib.compress(struct.pack("<2d", float("inf"), 1.0)),
            1.0,
        ),
        (VECTOR_ENCODING_RAW_F64LE_V1, struct.pack("<2d", 1.0, 1.0), float("inf")),
        (VECTOR_ENCODING_RAW_F64LE_V1, struct.pack("<2d", 1.0, 1.0), 9.0),
    ],
)
def test_cosine_search_rejects_corrupt_stored_values(
    tmp_path: Path, encoding: int, blob: bytes, magnitude: float
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        store.put_embedding("symbol-alpha", "fixture", [1.0, 1.0])
        store._connection.execute("PRAGMA ignore_check_constraints = ON")  # noqa: SLF001
        store._connection.execute(  # noqa: SLF001
            "UPDATE embedding_vectors SET vector_encoding = ?, vector = ?, magnitude = ?",
            (encoding, blob, magnitude),
        )

        with pytest.raises(sqlite3.OperationalError, match="user-defined function"):
            store.search_vector([1.0, 1.0], model="fixture")


def test_cosine_search_rejects_corrupt_query_before_sql(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        store.put_embedding("symbol-alpha", "fixture", [1.0, 1.0])
        statements: list[str] = []
        store._connection.set_trace_callback(statements.append)  # noqa: SLF001
        with pytest.raises(ValueError, match="finite"):
            store.search_vector([float("nan"), 1.0], model="fixture")
        store._connection.set_trace_callback(None)  # noqa: SLF001

    assert not any("_cpp_context_cosine" in statement for statement in statements)


def test_cosine_search_validates_query_once_and_unpacks_each_candidate_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        store.put_embedding("symbol-alpha", "fixture", [1.0, 1.0])
        store.put_embedding("file-a", "fixture", [-1.0, 0.0])
        validations = 0
        unpacks = 0
        original_validate = sqlite_storage._validate_vector
        original_decode = sqlite_storage._decode_vector_blob

        def count_validate(values: object) -> tuple[float, ...]:
            nonlocal validations
            validations += 1
            return original_validate(values)  # type: ignore[arg-type]

        def count_decode(blob: bytes) -> tuple[float, ...]:
            nonlocal unpacks
            unpacks += 1
            return original_decode(blob)

        monkeypatch.setattr(sqlite_storage, "_validate_vector", count_validate)
        monkeypatch.setattr(sqlite_storage, "_decode_vector_blob", count_decode)
        hits = store.search_vector([1.0, 1.0], model="fixture")

    assert [hit.symbol.id for hit in hits] == ["symbol-alpha", "file-a"]
    assert validations == 1
    assert unpacks == 2


def test_concurrent_vector_searches_are_isolated_and_release_lock_on_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        store.put_embedding("symbol-alpha", "fixture", [1.0, 0.0])
        store.put_embedding("file-a", "fixture", [0.0, 1.0])
        barrier = threading.Barrier(2)

        def search(query: list[float]) -> tuple[list[str], bool]:
            barrier.wait(timeout=2)
            hits = store.search_vector(query, model="fixture")
            return [hit.symbol.id for hit in hits], hasattr(
                store._vector_query_state,
                "current",  # noqa: SLF001
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            alpha = executor.submit(search, [1.0, 0.0])
            file_symbol = executor.submit(search, [0.0, 1.0])
            assert alpha.result(timeout=5) == (["symbol-alpha", "file-a"], False)
            assert file_symbol.result(timeout=5) == (["file-a", "symbol-alpha"], False)

        with store._vector_search_lock:  # noqa: SLF001 - prove the lock is reentrant
            assert store.search_vector([1.0, 0.0], model="fixture")[0].symbol.id == "symbol-alpha"

        barrier = threading.Barrier(2)

        def fail_dimension() -> str:
            barrier.wait(timeout=2)
            with pytest.raises(ValueError, match="dimension"):
                store.search_vector([1.0, 0.0, 0.0], model="fixture")
            return "failed-cleanly"

        def search_after_failure() -> str:
            barrier.wait(timeout=2)
            return store.search_vector([0.0, 1.0], model="fixture")[0].symbol.id

        with ThreadPoolExecutor(max_workers=2) as executor:
            failed = executor.submit(fail_dimension)
            successful = executor.submit(search_after_failure)
            assert failed.result(timeout=5) == "failed-cleanly"
            assert successful.result(timeout=5) == "file-a"

        assert store.search_vector([1.0, 0.0], model="fixture")[0].symbol.id == "symbol-alpha"


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


def test_vector_index_shares_equal_text_and_processes_bounded_batches(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)
    duplicate = replace(batch.symbols[1], id="symbol-alpha-copy")
    batch = replace(batch, symbols=(*batch.symbols, duplicate))
    provider = _RecordingEmbeddingProvider()

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        search = SQLiteVectorSearch(store, provider, project_root=root, batch_size=2)
        missing = store.missing_embedding_variant_ids(
            provider.model_id,
            configuration_id=provider.configuration_id,
        )

        assert search.index(missing) == 3
        assert max(map(len, provider.calls)) <= 2
        assert sum(map(len, provider.calls)) == 2
        assert (
            store.embedding_count(
                provider.model_id,
                configuration_id=provider.configuration_id,
            )
            == 3
        )
        assert (
            store.embedding_vector_count(
                provider.model_id,
                configuration_id=provider.configuration_id,
            )
            == 2
        )
        shared = store._connection.execute(  # noqa: SLF001 - physical sharing evidence
            """
            SELECT count(DISTINCT content_hash) FROM variant_embeddings
            WHERE project_id = ? AND configuration_id = ?
              AND variant_id IN (
                  SELECT id FROM symbol_variants WHERE symbol_id LIKE 'symbol-alpha%'
              )
            """,
            (store._project_id(root), provider.configuration_id),  # noqa: SLF001
        ).fetchone()[0]
        calls_after_first_index = tuple(provider.calls)
        assert search.index_missing() == 0
        assert tuple(provider.calls) == calls_after_first_index
        alpha_hits = [
            hit
            for hit in store.search_vector(
                [1.0, 1.0],
                model=provider.model_id,
                configuration_id=provider.configuration_id,
            )
            if hit.symbol.id.startswith("symbol-alpha")
        ]
        assert len(alpha_hits) == 2
        assert alpha_hits[0].score == pytest.approx(alpha_hits[1].score)
        assert [hit.symbol.variant_id for hit in alpha_hits] == sorted(
            hit.symbol.variant_id for hit in alpha_hits
        )

    assert shared == 1


def test_missing_vector_index_streams_snapshot_text_without_symbol_hydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    provider = _RecordingEmbeddingProvider()

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        search = SQLiteVectorSearch(store, provider, project_root=root, batch_size=1)
        monkeypatch.setattr(
            store,
            "get_symbols",
            lambda *_args, **_kwargs: pytest.fail("missing-index path hydrated symbols"),
        )

        assert search.index_missing() == 2
        assert provider.calls == [
            ("source.cpp\nsource.cpp",),
            ("alpha\nint alpha()\nReturns the important answer.\nint alpha() { return 7; }",),
        ]


def test_trusted_embedding_batches_reject_forged_stale_and_cross_store_use(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = SQLiteStore(tmp_path / "first.db", project_root=first_root)
    second = SQLiteStore(tmp_path / "second.db", project_root=second_root)
    try:
        first.apply_ingestion(first_root, _batch(first_root))
        second.apply_ingestion(second_root, _batch(second_root))
        with first.embedding_write_session(first_root):
            batch = next(first.iter_missing_embedding_record_batches("fixture", first_root))
            forged = replace(batch, capability=object())
            with pytest.raises(RuntimeError, match="stale or foreign"):
                first._attach_existing_embeddings_trusted(  # noqa: SLF001
                    forged, "fixture", first_root
                )
            with pytest.raises(RuntimeError, match="stale or foreign"):
                first._attach_existing_embeddings_trusted(  # noqa: SLF001
                    replace(batch), "fixture", first_root
                )
            with pytest.raises(RuntimeError, match="stale or foreign"):
                first._attach_existing_embeddings_trusted(  # noqa: SLF001
                    batch, "fixture", first_root, build_scope=BuildScope(("other",))
                )
            put_before_attach = _TrustedEmbeddingAttachment(batch.capability, (), 2)
            with pytest.raises(RuntimeError, match="stale or foreign"):
                first._put_content_embeddings_trusted(  # noqa: SLF001
                    put_before_attach, (), "fixture", first_root
                )
            with (
                second.embedding_write_session(second_root),
                pytest.raises(RuntimeError, match="stale or foreign"),
            ):
                second._attach_existing_embeddings_trusted(  # noqa: SLF001
                    batch, "fixture", second_root
                )
        with (
            first.embedding_write_session(first_root),
            pytest.raises(RuntimeError, match="stale or foreign"),
        ):
            first._attach_existing_embeddings_trusted(  # noqa: SLF001
                batch, "fixture", first_root
            )
    finally:
        first.close()
        second.close()


def test_trusted_embedding_batch_is_single_use_and_canonical(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        with store.embedding_write_session(root):
            batch = next(store.iter_missing_embedding_record_batches("fixture", root))
            attachment = store._attach_existing_embeddings_trusted(  # noqa: SLF001
                batch, "fixture", root
            )
            with pytest.raises(RuntimeError, match="stale or foreign"):
                store._attach_existing_embeddings_trusted(batch, "fixture", root)  # noqa: SLF001
            with pytest.raises(RuntimeError, match="stale or foreign"):
                store._put_content_embeddings_trusted(  # noqa: SLF001
                    replace(attachment), (), "fixture", root
                )
            forged = tuple(
                (variant_id, text + " forged", [1.0, 0.0])
                for variant_id, text in attachment.records
            )
            with pytest.raises(RuntimeError, match="do not match"):
                store._put_content_embeddings_trusted(  # noqa: SLF001
                    attachment, forged, "fixture", root
                )
            valid = tuple(
                (variant_id, text, [1.0, float(index + 1)])
                for index, (variant_id, text) in enumerate(attachment.records)
            )
            store._put_content_embeddings_trusted(  # noqa: SLF001
                attachment, valid, "fixture", root
            )
            with pytest.raises(RuntimeError, match="stale or foreign"):
                store._put_content_embeddings_trusted(  # noqa: SLF001
                    attachment, valid, "fixture", root
                )
            with pytest.raises(RuntimeError, match="stale or foreign"):
                store._attach_existing_embeddings_trusted(batch, "fixture", root)  # noqa: SLF001


def test_consumed_trusted_batches_leave_registry_bounded(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)
    additions = tuple(
        replace(
            batch.symbols[1],
            id=f"symbol-{index}",
            qualified_name=f"symbol_{index}",
            source_text=f"int symbol_{index}();",
        )
        for index in range(4)
    )
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, replace(batch, symbols=(*batch.symbols, *additions)))
        consumed = 0
        with store.embedding_write_session(root):
            for trusted in store.iter_missing_embedding_record_batches(
                "fixture", root, batch_size=1
            ):
                assert len(store._trusted_embedding_registry or {}) == 1  # noqa: SLF001
                attachment = store._attach_existing_embeddings_trusted(  # noqa: SLF001
                    trusted, "fixture", root
                )
                entries = tuple(
                    (variant_id, text, [1.0, float(consumed + 1)])
                    for variant_id, text in attachment.records
                )
                store._put_content_embeddings_trusted(  # noqa: SLF001
                    attachment, entries, "fixture", root
                )
                consumed += 1
                assert len(store._trusted_embedding_registry or {}) == 0  # noqa: SLF001
                with pytest.raises(RuntimeError, match="stale or foreign"):
                    store._attach_existing_embeddings_trusted(  # noqa: SLF001
                        trusted, "fixture", root
                    )

        assert consumed > 2


def test_trusted_iterator_rejects_a_second_outstanding_batch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)
    addition = replace(batch.symbols[1], id="symbol-extra", qualified_name="extra")
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, replace(batch, symbols=(*batch.symbols, addition)))
        with (
            pytest.raises(RuntimeError, match="still outstanding"),
            store.embedding_write_session(root),
        ):
            iterator = store.iter_missing_embedding_record_batches("fixture", root, batch_size=1)
            next(iterator)
            assert len(store._trusted_embedding_registry or {}) == 1  # noqa: SLF001
            next(iterator)

        assert store._trusted_embedding_registry is None  # noqa: SLF001
        assert not store._connection.in_transaction  # noqa: SLF001
        assert (
            SQLiteVectorSearch(
                store, _RecordingEmbeddingProvider(), project_root=root, batch_size=1
            ).index_missing()
            == 3
        )


def test_no_miss_trusted_attach_immediately_releases_capability(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)
    provider = _RecordingEmbeddingProvider()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        SQLiteVectorSearch(store, provider, project_root=root).index_missing()
        missing_variant = store._connection.execute(  # noqa: SLF001
            "SELECT id FROM symbol_variants ORDER BY id LIMIT 1"
        ).fetchone()[0]
        with store._connection:  # noqa: SLF001
            store._connection.execute(  # noqa: SLF001
                "DELETE FROM variant_embeddings WHERE variant_id = ?", (missing_variant,)
            )

        with store.embedding_write_session(root):
            trusted = next(
                store.iter_missing_embedding_record_batches(
                    "fixture", root, configuration_id=provider.configuration_id
                )
            )
            attachment = store._attach_existing_embeddings_trusted(  # noqa: SLF001
                trusted, "fixture", root, configuration_id=provider.configuration_id
            )
            assert attachment.records == ()
            assert len(store._trusted_embedding_registry or {}) == 0  # noqa: SLF001
            with pytest.raises(RuntimeError, match="stale or foreign"):
                store._attach_existing_embeddings_trusted(  # noqa: SLF001
                    trusted, "fixture", root, configuration_id=provider.configuration_id
                )


def test_embedding_commit_failure_rolls_back_and_allows_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    provider = _RecordingEmbeddingProvider()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        original_commit = store._commit_embedding_session  # noqa: SLF001
        monkeypatch.setattr(
            store,
            "_commit_embedding_session",
            lambda: (_ for _ in ()).throw(RuntimeError("injected commit failure")),
        )
        with pytest.raises(RuntimeError, match="injected commit failure"):
            SQLiteVectorSearch(store, provider, project_root=root).index_missing()

        assert not store._connection.in_transaction  # noqa: SLF001
        assert store.embedding_count(provider.model_id) == 0
        assert store.embedding_vector_count(provider.model_id) == 0
        monkeypatch.setattr(store, "_commit_embedding_session", original_commit)
        assert SQLiteVectorSearch(store, provider, project_root=root).index_missing() == 2


def test_public_embedding_writes_keep_variant_scope_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        calls = 0
        original = store._validate_embedding_variants  # noqa: SLF001

        def recording_validation(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "_validate_embedding_variants", recording_validation)
        variant = store.missing_embedding_variant_ids("fixture")[0]
        with store.embedding_write_session(root):
            assert store.attach_existing_embeddings(((variant, "text"),), "fixture", root) == (
                (variant, "text"),
            )
            store.put_content_embeddings(((variant, "text", [1.0, 0.0]),), "fixture", root)

        assert calls == 2


def test_embedding_configuration_identity_isolated_with_stable_public_model(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)
    first = _RecordingEmbeddingProvider(configuration_id="endpoint-a")
    second = _RecordingEmbeddingProvider(configuration_id="endpoint-b")

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        variants = store.missing_embedding_variant_ids(
            first.model_id, configuration_id=first.configuration_id
        )
        SQLiteVectorSearch(store, first, project_root=root).index(variants)
        SQLiteVectorSearch(store, second, project_root=root).index(variants)

        assert store.embedding_count("fixture", configuration_id="endpoint-a") == 2
        assert store.embedding_count("fixture", configuration_id="endpoint-b") == 2
        assert (
            store.search_vector([1.0, 1.0], model="fixture", configuration_id="endpoint-a")[
                0
            ].source
            == "vector:fixture"
        )


def test_embedding_hash_collision_is_rejected_against_retained_exact_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        variants = {
            row["symbol_id"]: row["id"]
            for row in store._connection.execute(  # noqa: SLF001 - physical collision evidence
                "SELECT id, symbol_id FROM symbol_variants"
            )
        }
        monkeypatch.setattr(sqlite_storage, "_embedding_content_hash", lambda _text: "collision")
        with store.embedding_write_session(root):
            store.put_content_embeddings(
                ((variants["symbol-alpha"], "first exact text", [1.0, 0.0]),),
                "fixture",
                root,
            )
        with (
            pytest.raises(ValueError, match="content hash collision"),
            store.embedding_write_session(root),
        ):
            store.put_content_embeddings(
                ((variants["file-a"], "different exact text", [0.0, 1.0]),),
                "fixture",
                root,
            )

        assert store.embedding_count("fixture") == 1
        assert store.embedding_vector_count("fixture") == 1
        assert (
            store._connection.execute(  # noqa: SLF001 - retained collision witness
                "SELECT content_text FROM embedding_vectors"
            ).fetchone()[0]
            == "first exact text"
        )


def test_batched_embedding_failure_rolls_back_all_new_vectors_and_references(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    batch = _batch(root)
    additions = tuple(
        replace(
            batch.symbols[1],
            id=f"symbol-{index}",
            qualified_name=f"symbol_{index}",
            source_text=f"int symbol_{index}();",
        )
        for index in range(4)
    )
    batch = replace(batch, symbols=(*batch.symbols, *additions))
    provider = _RecordingEmbeddingProvider(fail_on_call=2)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, batch)
        search = SQLiteVectorSearch(store, provider, project_root=root, batch_size=2)
        missing = store.missing_embedding_variant_ids(
            provider.model_id,
            configuration_id=provider.configuration_id,
        )

        with pytest.raises(RuntimeError, match="injected embedding failure"):
            search.index(missing)

        assert (
            store.embedding_count(
                provider.model_id,
                configuration_id=provider.configuration_id,
            )
            == 0
        )
        assert (
            store.embedding_vector_count(
                provider.model_id,
                configuration_id=provider.configuration_id,
            )
            == 0
        )


@pytest.mark.parametrize("failure", ["wrong-count", "cancel"])
def test_missing_embedding_session_rolls_back_provider_failure(
    tmp_path: Path, failure: str
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    class FailingProvider(_RecordingEmbeddingProvider):
        def embed(self, _texts: list[str]) -> tuple[tuple[float, ...], ...]:
            if failure == "cancel":
                raise KeyboardInterrupt
            return ()

    provider = FailingProvider()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        search = SQLiteVectorSearch(store, provider, project_root=root, batch_size=1)
        expected = KeyboardInterrupt if failure == "cancel" else ValueError
        with pytest.raises(expected):
            search.index_missing()

        assert store.embedding_count(provider.model_id) == 0
        assert store.embedding_vector_count(provider.model_id) == 0


def test_equal_embedding_input_with_different_vector_rolls_back(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        variants = store.missing_embedding_variant_ids("fixture")
        with store.embedding_write_session(root):
            store.put_content_embeddings(((variants[0], "same", [1.0, 0.0]),), "fixture", root)
        with (
            pytest.raises(ValueError, match="equal embedding inputs"),
            store.embedding_write_session(root),
        ):
            store.put_content_embeddings(((variants[1], "same", [0.0, 1.0]),), "fixture", root)

        assert store.embedding_count("fixture") == 1
        assert store.embedding_vector_count("fixture") == 1


def test_v12_migrates_legacy_variant_vectors_into_shared_content_pool(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "index.db"
    batch = _batch(root)
    duplicate = replace(batch.symbols[1], id="symbol-alpha-copy")
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, replace(batch, symbols=(*batch.symbols, duplicate)))
        variants = tuple(
            row[0]
            for row in store._connection.execute(  # noqa: SLF001 - migration fixture
                """
                SELECT id FROM symbol_variants
                WHERE symbol_id LIKE 'symbol-alpha%'
                ORDER BY id
                """
            )
        )

    vector = struct.pack("<2d", 1.0, 1.0)
    legacy = sqlite3.connect(database)
    legacy.executescript(
        """
        DROP TABLE variant_embeddings;
        DROP TABLE embedding_vectors;
        CREATE TABLE variant_embeddings (
            project_id INTEGER NOT NULL,
            variant_id TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            magnitude REAL NOT NULL,
            vector BLOB NOT NULL,
            PRIMARY KEY (project_id, variant_id, model),
            FOREIGN KEY (project_id, variant_id)
                REFERENCES symbol_variants(project_id, id) ON DELETE CASCADE
        );
        PRAGMA user_version = 10;
        """
    )
    legacy.executemany(
        "INSERT INTO variant_embeddings VALUES (1, ?, 'fixture', 2, ?, ?)",
        ((variant_id, math.sqrt(2.0), vector) for variant_id in variants),
    )
    legacy.executemany(
        """
        INSERT INTO variant_embeddings
        VALUES (1, ?, 'openai-compatible:legacy', 2, ?, ?)
        """,
        ((variant_id, math.sqrt(2.0), vector) for variant_id in variants),
    )
    legacy.commit()
    legacy.close()

    with SQLiteStore(database, project_root=root) as store:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 14  # noqa: SLF001
        assert store.embedding_count("fixture") == 2
        assert store.embedding_vector_count("fixture") == 1
        assert store.embedding_count("openai-compatible:legacy") == 0
        assert len(store.missing_embedding_variant_ids("openai-compatible:legacy")) >= 2
        hits = store.search_vector([1.0, 1.0], model="fixture")

    assert [hit.symbol.variant_id for hit in hits] == sorted(variants)
    assert {hit.symbol.id for hit in hits} == {"symbol-alpha", "symbol-alpha-copy"}
    assert all(hit.score == pytest.approx(1.0) for hit in hits)


@pytest.mark.parametrize(
    ("dimensions", "magnitude", "vector"),
    (
        (2, math.sqrt(2.0), struct.pack("<d", 1.0)),
        (2, 1.0, struct.pack("<2d", 1.0, float("nan"))),
        (2, 9.0, struct.pack("<2d", 1.0, 1.0)),
    ),
)
def test_v12_migration_rejects_corrupt_legacy_vectors_atomically(
    tmp_path: Path,
    dimensions: int,
    magnitude: float,
    vector: bytes,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, _batch(root))
        variant_id = store._connection.execute(  # noqa: SLF001 - migration fixture
            "SELECT id FROM symbol_variants WHERE symbol_id = 'symbol-alpha'"
        ).fetchone()[0]

    legacy = sqlite3.connect(database)
    legacy.executescript(
        """
        DROP TABLE variant_embeddings;
        DROP TABLE embedding_vectors;
        CREATE TABLE variant_embeddings (
            project_id INTEGER NOT NULL,
            variant_id TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            magnitude REAL NOT NULL,
            vector BLOB NOT NULL,
            PRIMARY KEY (project_id, variant_id, model)
        );
        PRAGMA user_version = 11;
        """
    )
    legacy.execute(
        "INSERT INTO variant_embeddings VALUES (1, ?, 'fixture', ?, ?, ?)",
        (variant_id, dimensions, magnitude, vector),
    )
    legacy.commit()
    legacy.close()

    with pytest.raises(RuntimeError, match="invalid legacy embedding vector"):
        SQLiteStore(database, project_root=root)

    unchanged = sqlite3.connect(database)
    try:
        assert unchanged.execute("PRAGMA user_version").fetchone()[0] == 11
        assert {row[1] for row in unchanged.execute("PRAGMA table_info(variant_embeddings)")} == {
            "project_id",
            "variant_id",
            "model",
            "dimensions",
            "magnitude",
            "vector",
        }
        assert (
            unchanged.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'embedding_vectors'"
            ).fetchone()
            is None
        )
    finally:
        unchanged.close()


def test_v12_migration_accepts_minimal_v11_database(tmp_path: Path) -> None:
    database = tmp_path / "minimal-v11.db"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 11")
    connection.commit()
    connection.close()

    with SQLiteStore(database) as store:
        tables = {
            row[0]
            for row in store._connection.execute(  # noqa: SLF001 - migration boundary
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 14  # noqa: SLF001
        assert "vector_encoding" in {
            row[1]
            for row in store._connection.execute(  # noqa: SLF001
                "PRAGMA table_info(embedding_vectors)"
            )
        }

    assert {"embedding_vectors", "variant_embeddings"} <= tables


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
        assert store.embedding_vector_count("fixture") == 0


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
        assert store.embedding_vector_count("fixture") == 0


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
