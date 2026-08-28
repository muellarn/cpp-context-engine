from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zlib
from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary

from cpp_context_engine.ingestion import NativeAnalyzerClient, NativeClangIngestor
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.models import BuildScope, BuildVariant, DataFlowCertainty
from cpp_context_engine.storage.sqlite import (
    MAX_SUMMARY_PAYLOAD_RECORDS,
    SCHEMA_VERSION,
    SUMMARY_PAYLOAD_ENCODING,
    SQLiteStore,
    SummaryPayloadError,
    _decode_summary_payload,
    _encode_summary_payload,
)

FIXTURE = Path(__file__).parent / "fixtures" / "interprocedural_project"
pytestmark = pytest.mark.native


def _ingestor() -> NativeClangIngestor:
    return NativeClangIngestor(
        NativeAnalyzerClient(analyzer_binary(), timeout_seconds=90), max_workers=2
    )


@pytest.fixture(scope="module")
def solved_batch():
    return _ingestor().ingest(FIXTURE, FIXTURE / "compile_commands.json")


def _effect_order(item) -> tuple[int, int, str, str]:
    return (
        0 if item.certainty == DataFlowCertainty.CERTAIN else 1,
        0 if item.is_local else 1,
        item.kind.value,
        item.id,
    )


def _solution_rows(store: SQLiteStore) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in store._connection.execute(  # noqa: SLF001 - deterministic payload evidence
            """
            SELECT summaries.function_symbol_id, summaries.build_variant,
                   summaries.solution_hash, payloads.effect_count,
                   payloads.origin_count, payloads.uncompressed_bytes,
                   payloads.payload_hash, hex(payloads.payload)
            FROM summary_solution_payloads payloads
            JOIN function_summaries summaries
              ON summaries.project_id = payloads.project_id
             AND summaries.id = payloads.summary_id
            ORDER BY summaries.function_symbol_id, summaries.build_variant, summaries.id
            """
        )
    )


def _insert_legacy_propagated_rows(store: SQLiteStore) -> None:
    project_id = store._project_id()  # noqa: SLF001 - construct a real v10 database
    summary_ids = tuple(
        row[0]
        for row in store._connection.execute(  # noqa: SLF001
            "SELECT summary_id FROM summary_solution_payloads ORDER BY summary_id"
        )
    )
    for summary_id in summary_ids:
        effects, origins = store._summary_solution_payload(  # noqa: SLF001
            project_id, summary_id, ("default",)
        )
        store._connection.executemany(  # noqa: SLF001
            """
            INSERT INTO summary_effects(
                project_id, id, summary_id, kind, location_kind, certainty,
                reason, parameter_index, access_path_json, location_id,
                source_access_id, is_local, via_callsite_id, target_symbol_id,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.summary_id,
                    item.kind.value,
                    item.location_kind.value,
                    item.certainty.value,
                    item.reason,
                    item.parameter_index,
                    json.dumps(item.access_path),
                    item.location_id,
                    item.source_access_id,
                    0,
                    item.via_callsite_id,
                    item.target_symbol_id,
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in effects
            ),
        )
        store._connection.executemany(  # noqa: SLF001
            """
            INSERT INTO summary_return_origins(
                project_id, id, summary_id, kind, certainty, reason,
                location_kind, parameter_index, access_path_json, location_id,
                callsite_id, is_local, via_callsite_id, target_symbol_id,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.summary_id,
                    item.kind.value,
                    item.certainty.value,
                    item.reason,
                    item.location_kind.value if item.location_kind else None,
                    item.parameter_index,
                    json.dumps(item.access_path),
                    item.location_id,
                    item.callsite_id,
                    0,
                    item.via_callsite_id,
                    item.target_symbol_id,
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in origins
            ),
        )
    store._connection.execute("DROP TABLE summary_solution_payloads")  # noqa: SLF001
    store._connection.execute("PRAGMA user_version = 10")  # noqa: SLF001
    store._connection.commit()  # noqa: SLF001


def _downgrade_embedding_schema_to_v11(store: SQLiteStore) -> None:
    store._connection.executescript(  # noqa: SLF001 - construct a real v11 boundary
        """
        CREATE TABLE variant_embeddings_v11 (
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
        INSERT INTO variant_embeddings_v11(
            project_id, variant_id, model, dimensions, magnitude, vector
        )
        SELECT references_.project_id, references_.variant_id, references_.model,
               references_.dimensions, vectors.magnitude, vectors.vector
        FROM variant_embeddings references_
        JOIN embedding_vectors vectors
          ON vectors.project_id = references_.project_id
         AND vectors.model = references_.model
         AND vectors.configuration_id = references_.configuration_id
         AND vectors.dimensions = references_.dimensions
         AND vectors.content_hash = references_.content_hash;
        DROP TABLE variant_embeddings;
        DROP TABLE embedding_vectors;
        ALTER TABLE variant_embeddings_v11 RENAME TO variant_embeddings;
        PRAGMA user_version = 11;
        """
    )
    store._connection.commit()  # noqa: SLF001


def test_schema_v11_stores_propagated_summary_payloads_separately(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        tables = {
            row[0]
            for row in store._connection.execute(  # noqa: SLF001 - schema regression
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert SCHEMA_VERSION == 12
    assert {"summary_solution_payloads", "embedding_vectors", "variant_embeddings"} <= tables


def test_v11_summary_database_upgrades_to_v12_without_changing_solution_payloads(
    tmp_path: Path, solved_batch
) -> None:
    database = tmp_path / "v11.db"
    with SQLiteStore(database, project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, solved_batch)
        before = _solution_rows(store)
        _downgrade_embedding_schema_to_v11(store)

    with SQLiteStore(database, project_root=FIXTURE) as migrated:
        assert migrated._connection.execute("PRAGMA user_version").fetchone()[0] == 12  # noqa: SLF001
        assert _solution_rows(migrated) == before
        assert migrated._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001


def test_propagated_payloads_preserve_exact_solution_and_bounded_api(
    tmp_path: Path, solved_batch
) -> None:
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=FIXTURE) as store:
        store.apply_ingestion(
            FIXTURE,
            solved_batch,
            current_translation_unit_ids=frozenset(
                unit.id for unit in solved_batch.translation_units
            ),
        )
        assert (
            store._connection.execute(  # noqa: SLF001 - relational materialization regression
                "SELECT count(*) FROM summary_effects WHERE is_local = 0"
            ).fetchone()[0]
            == 0
        )
        assert (
            store._connection.execute(  # noqa: SLF001
                "SELECT count(*) FROM summary_return_origins WHERE is_local = 0"
            ).fetchone()[0]
            == 0
        )
        assert store._connection.execute(  # noqa: SLF001
            "SELECT count(*) FROM summary_solution_payloads"
        ).fetchone()[0]

        expected_summaries = {item.id: item for item in solved_batch.function_summaries}
        actual_summaries = {
            row["id"]: store._row_to_function_summary(row)  # noqa: SLF001
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT * FROM function_summaries ORDER BY id"
            )
        }
        assert actual_summaries == expected_summaries
        for summary_id in sorted(expected_summaries):
            expected_effects = tuple(
                sorted(
                    (
                        item
                        for item in solved_batch.summary_effects
                        if item.summary_id == summary_id
                    ),
                    key=_effect_order,
                )
            )
            expected_origins = tuple(
                sorted(
                    (
                        item
                        for item in solved_batch.summary_return_origins
                        if item.summary_id == summary_id
                    ),
                    key=_effect_order,
                )
            )
            assert store.summary_effects(summary_id, limit=10_000).items == expected_effects
            assert store.summary_return_origins(summary_id, limit=10_000).items == expected_origins
            if expected_effects:
                bounded = store.summary_effects(summary_id, limit=1)
                assert bounded.items == expected_effects[:1]
                assert bounded.truncated == (len(expected_effects) > 1)


def test_payload_codec_is_deterministic_and_rejects_malformed_or_oversized_data(
    solved_batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    propagated_effects = tuple(item for item in solved_batch.summary_effects if not item.is_local)
    propagated_origins = tuple(
        item for item in solved_batch.summary_return_origins if not item.is_local
    )
    summary_id = (propagated_effects or propagated_origins)[0].summary_id
    effects = tuple(item for item in propagated_effects if item.summary_id == summary_id)
    origins = tuple(item for item in propagated_origins if item.summary_id == summary_id)
    encoded = _encode_summary_payload(summary_id, effects, origins)
    assert _encode_summary_payload(summary_id, effects, origins) == encoded
    assert _decode_summary_payload(
        summary_id,
        encoding=SUMMARY_PAYLOAD_ENCODING,
        effect_count=encoded[0],
        origin_count=encoded[1],
        uncompressed_bytes=encoded[2],
        payload_hash=encoded[3],
        payload=encoded[4],
    ) == (effects, origins)

    with pytest.raises(SummaryPayloadError, match="record limit"):
        _decode_summary_payload(
            summary_id,
            encoding=SUMMARY_PAYLOAD_ENCODING,
            effect_count=MAX_SUMMARY_PAYLOAD_RECORDS + 1,
            origin_count=0,
            uncompressed_bytes=encoded[2],
            payload_hash=encoded[3],
            payload=encoded[4],
        )
    with pytest.raises(SummaryPayloadError, match="compression stream"):
        _decode_summary_payload(
            summary_id,
            encoding=SUMMARY_PAYLOAD_ENCODING,
            effect_count=encoded[0],
            origin_count=encoded[1],
            uncompressed_bytes=encoded[2],
            payload_hash=encoded[3],
            payload=encoded[4] + b"trailing-data",
        )

    import cpp_context_engine.storage.sqlite as sqlite_module

    compressed_limit = sqlite_module.MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES
    monkeypatch.setattr(sqlite_module, "MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES", len(encoded[4]) - 1)
    with pytest.raises(SummaryPayloadError, match="compressed-size limit"):
        _decode_summary_payload(
            summary_id,
            encoding=SUMMARY_PAYLOAD_ENCODING,
            effect_count=encoded[0],
            origin_count=encoded[1],
            uncompressed_bytes=encoded[2],
            payload_hash=encoded[3],
            payload=encoded[4],
        )
    monkeypatch.setattr(sqlite_module, "MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES", compressed_limit)
    monkeypatch.setattr(sqlite_module, "MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES", 32)
    with pytest.raises(SummaryPayloadError, match="decompressed-size limit"):
        _decode_summary_payload(
            summary_id,
            encoding=SUMMARY_PAYLOAD_ENCODING,
            effect_count=encoded[0],
            origin_count=encoded[1],
            uncompressed_bytes=encoded[2],
            payload_hash=encoded[3],
            payload=encoded[4],
        )

    monkeypatch.setattr(sqlite_module, "MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES", encoded[2] + 1024)
    with pytest.raises(SummaryPayloadError, match="unsupported"):
        _decode_summary_payload(
            summary_id,
            encoding="unknown",
            effect_count=encoded[0],
            origin_count=encoded[1],
            uncompressed_bytes=encoded[2],
            payload_hash=encoded[3],
            payload=encoded[4],
        )
    with pytest.raises(SummaryPayloadError, match="malformed"):
        _decode_summary_payload(
            summary_id,
            encoding=SUMMARY_PAYLOAD_ENCODING,
            effect_count=encoded[0],
            origin_count=encoded[1],
            uncompressed_bytes=encoded[2],
            payload_hash=encoded[3],
            payload=encoded[4][:-1],
        )
    with pytest.raises(SummaryPayloadError, match="hash"):
        _decode_summary_payload(
            summary_id,
            encoding=SUMMARY_PAYLOAD_ENCODING,
            effect_count=encoded[0],
            origin_count=encoded[1],
            uncompressed_bytes=encoded[2],
            payload_hash="0" * 64,
            payload=encoded[4],
        )
    with pytest.raises(SummaryPayloadError, match="record counts"):
        _decode_summary_payload(
            summary_id,
            encoding=SUMMARY_PAYLOAD_ENCODING,
            effect_count=encoded[0] + 1,
            origin_count=encoded[1],
            uncompressed_bytes=encoded[2],
            payload_hash=encoded[3],
            payload=encoded[4],
        )

    invalid_raw = json.dumps([{}, []], separators=(",", ":")).encode()
    with pytest.raises(SummaryPayloadError, match="record groups"):
        _decode_summary_payload(
            summary_id,
            encoding=SUMMARY_PAYLOAD_ENCODING,
            effect_count=0,
            origin_count=0,
            uncompressed_bytes=len(invalid_raw),
            payload_hash=hashlib.sha256(invalid_raw).hexdigest(),
            payload=zlib.compress(invalid_raw),
        )


def test_corrupt_payload_fails_safely_through_public_api(tmp_path: Path, solved_batch) -> None:
    with SQLiteStore(tmp_path / "index.db", project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, solved_batch)
        row = store._connection.execute(  # noqa: SLF001
            "SELECT summary_id, payload FROM summary_solution_payloads ORDER BY summary_id LIMIT 1"
        ).fetchone()
        store._connection.execute(  # noqa: SLF001
            "UPDATE summary_solution_payloads SET payload = ? WHERE summary_id = ?",
            (bytes(row["payload"]) + b"trailing-data", row["summary_id"]),
        )
        with pytest.raises(SummaryPayloadError, match="compression stream"):
            store.summary_effects(row["summary_id"], limit=10_000)


def test_public_api_rejects_oversized_payload_before_materializing_blob(
    tmp_path: Path, solved_batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cpp_context_engine.storage.sqlite as sqlite_module

    with SQLiteStore(tmp_path / "index.db", project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, solved_batch)
        summary_id = store._connection.execute(  # noqa: SLF001
            "SELECT summary_id FROM summary_solution_payloads ORDER BY summary_id LIMIT 1"
        ).fetchone()[0]
        store._connection.execute(  # noqa: SLF001
            "UPDATE summary_solution_payloads SET payload = zeroblob(64) WHERE summary_id = ?",
            (summary_id,),
        )
        monkeypatch.setattr(sqlite_module, "MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES", 32)

        def reject_large_blobs(cursor: sqlite3.Cursor, values: tuple[object, ...]) -> sqlite3.Row:
            if any(isinstance(value, bytes) and len(value) > 32 for value in values):
                raise AssertionError("oversized BLOB reached the Python process")
            return sqlite3.Row(cursor, values)

        store._connection.row_factory = reject_large_blobs  # noqa: SLF001
        with pytest.raises(SummaryPayloadError, match="compressed-size limit"):
            store.summary_effects(summary_id, limit=10_000)


def test_v10_migration_compacts_existing_rows_without_changing_results(
    tmp_path: Path, solved_batch
) -> None:
    database = tmp_path / "v10.db"
    with SQLiteStore(database, project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, solved_batch)
        summary_ids = tuple(
            row[0]
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT id FROM function_summaries ORDER BY id"
            )
        )
        expected = {
            summary_id: (
                store.summary_effects(summary_id, limit=10_000),
                store.summary_return_origins(summary_id, limit=10_000),
            )
            for summary_id in summary_ids
        }
        _insert_legacy_propagated_rows(store)

    with SQLiteStore(database, project_root=FIXTURE) as migrated:
        assert migrated._connection.execute("PRAGMA user_version").fetchone()[0] == 12  # noqa: SLF001
        assert (
            migrated._connection.execute(  # noqa: SLF001
                "SELECT count(*) FROM summary_effects WHERE is_local = 0"
            ).fetchone()[0]
            == 0
        )
        assert {
            summary_id: (
                migrated.summary_effects(summary_id, limit=10_000),
                migrated.summary_return_origins(summary_id, limit=10_000),
            )
            for summary_id in summary_ids
        } == expected


def test_v10_direct_upgrade_preserves_summaries_and_local_vector_bytes(
    tmp_path: Path, solved_batch
) -> None:
    database = tmp_path / "combined-v10.db"
    with SQLiteStore(database, project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, solved_batch)
        summary_rows = _solution_rows(store)
        variant_id = store._connection.execute(  # noqa: SLF001 - migration fixture
            "SELECT id FROM symbol_variants ORDER BY id LIMIT 1"
        ).fetchone()[0]
        store.put_embedding(variant_id, "local-feature-hash-v1-2", [1.0, 1.0])
        expected_hits = tuple(
            (hit.symbol.variant_id, hit.score)
            for hit in store.search_vector([1.0, 1.0], model="local-feature-hash-v1-2")
        )
        vector_bytes = store._connection.execute(  # noqa: SLF001 - exact byte regression
            "SELECT hex(vector) FROM embedding_vectors"
        ).fetchone()[0]
        _downgrade_embedding_schema_to_v11(store)
        _insert_legacy_propagated_rows(store)

    with SQLiteStore(database, project_root=FIXTURE) as migrated:
        assert migrated._connection.execute("PRAGMA user_version").fetchone()[0] == 12  # noqa: SLF001
        assert _solution_rows(migrated) == summary_rows
        assert (
            migrated._connection.execute(  # noqa: SLF001
                "SELECT hex(vector) FROM embedding_vectors"
            ).fetchone()[0]
            == vector_bytes
        )
        assert (
            tuple(
                (hit.symbol.variant_id, hit.score)
                for hit in migrated.search_vector([1.0, 1.0], model="local-feature-hash-v1-2")
            )
            == expected_hits
        )
        assert migrated._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001


def test_v11_migration_and_payload_persistence_failures_roll_back_atomically(
    tmp_path: Path, solved_batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "migration.db"
    with SQLiteStore(database, project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, solved_batch)
        _insert_legacy_propagated_rows(store)
    connection = sqlite3.connect(database)
    try:
        before_nonlocal = connection.execute(
            "SELECT count(*) FROM summary_effects WHERE is_local = 0"
        ).fetchone()[0]
    finally:
        connection.close()

    original = SQLiteStore._write_summary_solution_payload
    calls = 0

    def fail_second(self, project_id, summary_id, effects, origins):
        nonlocal calls
        calls += 1
        original(self, project_id, summary_id, effects, origins)
        if calls == 2:
            raise RuntimeError("injected summary migration failure")

    monkeypatch.setattr(SQLiteStore, "_write_summary_solution_payload", fail_second)
    with pytest.raises(RuntimeError, match="summary migration failure"):
        SQLiteStore(database)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 10
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'summary_solution_payloads'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM summary_effects WHERE is_local = 0"
            ).fetchone()[0]
            == before_nonlocal
        )
    finally:
        connection.close()

    monkeypatch.setattr(SQLiteStore, "_write_summary_solution_payload", original)
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    database = tmp_path / "rollback.db"
    with SQLiteStore(database, project_root=project) as store:
        indexer = ProjectIndexer(_ingestor(), store)
        indexer.index(project, project / "compile_commands.json")
        before = _solution_rows(store)
        changed = project / "src" / "leaf.cpp"
        changed.write_text(changed.read_text() + "\n// payload rollback\n", encoding="utf-8")

        def fail_after_write(project_id, summary_id, effects, origins):
            original(store, project_id, summary_id, effects, origins)
            raise RuntimeError("injected summary payload persistence failure")

        monkeypatch.setattr(store, "_write_summary_solution_payload", fail_after_write)
        with pytest.raises(RuntimeError, match="payload persistence failure"):
            indexer.index(project, project / "compile_commands.json")
        assert _solution_rows(store) == before
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001


def test_v10_direct_upgrade_rolls_back_v11_when_v12_fails(
    tmp_path: Path, solved_batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "combined-rollback.db"
    with SQLiteStore(database, project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, solved_batch)
        _insert_legacy_propagated_rows(store)

    def fail_v12(self: SQLiteStore, *, manage_transaction: bool = True) -> None:
        assert not manage_transaction
        assert self._connection.execute(  # noqa: SLF001 - observe intermediate v11 state
            "SELECT 1 FROM sqlite_master WHERE name = 'summary_solution_payloads'"
        ).fetchone()
        raise RuntimeError("injected v12 migration failure")

    monkeypatch.setattr(SQLiteStore, "_migrate_v12", fail_v12)
    with pytest.raises(RuntimeError, match="v12 migration failure"):
        SQLiteStore(database, project_root=FIXTURE)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 10
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'summary_solution_payloads'"
            ).fetchone()
            is None
        )
        assert connection.execute(
            "SELECT count(*) FROM summary_effects WHERE is_local = 0"
        ).fetchone()[0]
    finally:
        connection.close()


def test_noop_header_refresh_stale_cleanup_and_build_variants_are_isolated(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    default = BuildVariant("default", project / "compile_commands.json")
    alternative = BuildVariant("alternative", project / "compile_commands_alt.json")

    with SQLiteStore(tmp_path / "index.db", project_root=project) as store:
        indexer = ProjectIndexer(_ingestor(), store)
        indexer.index(project, default.compilation_database, build_variant=default)
        indexer.index(project, alternative.compilation_database, build_variant=alternative)
        baseline = _solution_rows(store)
        alternative_before = tuple(row for row in baseline if row[1] == "alternative")

        noop = indexer.index(project, default.compilation_database, build_variant=default)
        assert noop.indexed_translation_units == 0
        assert _solution_rows(store) == baseline

        default_summary_count = store._connection.execute(  # noqa: SLF001
            "SELECT count(*) FROM function_summaries WHERE build_variant = 'default'"
        ).fetchone()[0]
        unrelated_payload = store._connection.execute(  # noqa: SLF001
            """
            SELECT summaries.solution_hash, payloads.payload_hash, payloads.payload
            FROM function_summaries summaries
            JOIN symbols
              ON symbols.project_id = summaries.project_id
             AND symbols.id = summaries.function_symbol_id
            LEFT JOIN summary_solution_payloads payloads
              ON payloads.project_id = summaries.project_id
             AND payloads.summary_id = summaries.id
            WHERE summaries.build_variant = 'default'
              AND symbols.qualified_name LIKE '%::unrelated'
            ORDER BY summaries.id LIMIT 1
            """
        ).fetchone()
        assert unrelated_payload is not None
        leaf = project / "src" / "leaf.cpp"
        leaf.write_text(leaf.read_text() + "\n// targeted refresh\n", encoding="utf-8")
        targeted = indexer.index(project, default.compilation_database, build_variant=default)
        assert targeted.indexed_translation_units == 1
        assert 0 < targeted.invalidated_function_summaries < default_summary_count
        assert (
            store._connection.execute(  # noqa: SLF001
                """
            SELECT summaries.solution_hash, payloads.payload_hash, payloads.payload
            FROM function_summaries summaries
            JOIN symbols
              ON symbols.project_id = summaries.project_id
             AND symbols.id = summaries.function_symbol_id
            LEFT JOIN summary_solution_payloads payloads
              ON payloads.project_id = summaries.project_id
             AND payloads.summary_id = summaries.id
            WHERE summaries.build_variant = 'default'
              AND symbols.qualified_name LIKE '%::unrelated'
            ORDER BY summaries.id LIMIT 1
            """
            ).fetchone()
            == unrelated_payload
        )

        header = project / "include" / "interprocedural.hpp"
        header.write_text(header.read_text() + "\n// header refresh\n", encoding="utf-8")
        refreshed = indexer.index(project, default.compilation_database, build_variant=default)
        assert refreshed.indexed_translation_units > 1
        assert refreshed.invalidated_function_summaries > 0
        assert tuple(row for row in _solution_rows(store) if row[1] == "alternative") == (
            alternative_before
        )

        commands = json.loads(default.compilation_database.read_text())
        default.compilation_database.write_text(json.dumps(commands[:-1]), encoding="utf-8")
        stale = indexer.index(project, default.compilation_database, build_variant=default)
        assert stale.removed_translation_units == 1
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001
        assert (
            store._connection.execute(  # noqa: SLF001
                """
            SELECT count(*) FROM summary_solution_payloads payloads
            WHERE NOT EXISTS (
                SELECT 1 FROM function_summaries summaries
                WHERE summaries.project_id = payloads.project_id
                  AND summaries.id = payloads.summary_id
            )
            """
            ).fetchone()[0]
            == 0
        )
        assert tuple(row for row in _solution_rows(store) if row[1] == "alternative") == (
            alternative_before
        )

        default_summary = store._connection.execute(  # noqa: SLF001
            "SELECT id FROM function_summaries WHERE build_variant = 'default' ORDER BY id LIMIT 1"
        ).fetchone()[0]
        assert (
            store.summary_effects(
                default_summary, build_scope=BuildScope.single("alternative")
            ).items
            == ()
        )
