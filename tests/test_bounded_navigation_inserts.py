from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest
from test_sqlite_storage import _batch
from test_streaming_indexer import _semantic_dump

import cpp_context_engine.storage.sqlite as storage
from cpp_context_engine.models import (
    CallDispatchKind,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    GraphRelation,
    IndexProfile,
    SearchQuery,
)
from cpp_context_engine.storage.sqlite import SQLiteStore


class _ExecutemanyStore(SQLiteStore):
    """Unbatched reference for the same complete storage/public contracts."""

    def _insert_rows(self, sql: str, rows: Iterable[tuple[object, ...]], *, columns: int) -> None:
        values = "(" + ",".join("?" for _ in range(columns)) + ")"
        self._connection.executemany(sql.format(values=values), rows)


def test_navigation_occurrences_use_bounded_multirow_statements(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    batch = replace(
        batch,
        translation_units=(
            replace(batch.translation_units[0], index_profile=IndexProfile.NAVIGATION),
        ),
        occurrences=tuple(replace(batch.occurrences[0], id=f"occ-{i}") for i in range(1025)),
    )
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        statements: list[str] = []
        store._connection.set_trace_callback(statements.append)
        try:
            store.apply_ingestion(tmp_path, batch, index_profile=IndexProfile.NAVIGATION)
        finally:
            store._connection.set_trace_callback(None)
        inserts = [
            sql
            for sql in statements
            if sql.lstrip().startswith("INSERT OR REPLACE INTO occurrences(")
        ]
        assert len(inserts) == 3, f"expected 512/512/1 row batches, got {len(inserts)} statements"
        assert store._connection.execute("SELECT count(*) FROM occurrences").fetchone()[0] == 1025


@pytest.mark.parametrize("profile", [IndexProfile.FULL, IndexProfile.NAVIGATION])
def test_batched_ingestion_matches_unbatched_rows_and_public_results(
    tmp_path: Path, profile: IndexProfile
) -> None:
    batch = _batch(tmp_path)
    symbol = batch.symbols[1]
    sites = tuple(
        CallSite(
            id=f"site-{i}",
            owner_symbol_id=symbol.id,
            dispatch_kind=CallDispatchKind.DIRECT,
            spelling_span=symbol.span,
            expansion_span=symbol.span,
            target_set_complete=True,
            static_target_symbol_id=symbol.id,
            callee_text="alpha()",
            translation_unit_id="unit-a",
            build_configuration_id="build-a",
        )
        for i in range(1025)
    )
    targets = tuple(
        CallTarget(
            id=f"target-{i}",
            callsite_id=site.id,
            target_symbol_id=symbol.id,
            certainty=CallTargetCertainty.CERTAIN,
            confidence=1.0,
            confidence_reason="direct",
            derivation="direct",
            evidence_span=symbol.span,
            translation_unit_id="unit-a",
            build_configuration_id="build-a",
        )
        for i, site in enumerate(sites)
    )
    # Duplicate IDs inside and across 512-row chunks: REPLACE/UPSERT last wins,
    # IGNORE first wins. All rows pass through the actual ingestion path.
    batch = replace(
        batch,
        translation_units=(replace(batch.translation_units[0], index_profile=profile),),
        symbols=(batch.symbols[0],)
        + tuple(replace(symbol, source_text=f"int alpha() {{ return {i}; }}") for i in range(1025)),
        occurrences=tuple(
            replace(batch.occurrences[0], id=f"occ-{(i // 2) % 32}", metadata={"order": i})
            for i in range(1025)
        ),
        edges=tuple(
            replace(
                batch.edges[0],
                id=f"edge-{(i // 2) % 32}",
                relation=GraphRelation.CONTAINS if i % 2 == 0 else GraphRelation.CALLS,
            )
            for i in range(1025)
        ),
        callsites=sites,
        call_targets=targets,
    )
    results = []
    for name, cls in (("reference", _ExecutemanyStore), ("candidate", SQLiteStore)):
        with cls(tmp_path / f"{name}.db", project_root=tmp_path) as store:
            store.apply_ingestion(tmp_path, batch, index_profile=profile)
            store.put_embeddings((("symbol-alpha", [1.0, 0.0]), ("file-a", [0.0, 1.0])), "test")
            results.append(
                (
                    _semantic_dump(store),
                    store.get_symbols(("symbol-alpha", "file-a")),
                    tuple(store.search(SearchQuery("alpha"))),
                    tuple(store.search_vector([1.0, 0.0], model="test")),
                    tuple(store._connection.execute("PRAGMA foreign_key_check")),
                )
            )
    assert results[0] == results[1]
    assert results[1][1][0].source_text == "int alpha() { return 1024; }"


@pytest.mark.parametrize("failure", ["generator", "sql"])
def test_ingestion_failure_after_written_chunk_rolls_back_generation(
    tmp_path: Path, failure: str
) -> None:
    batch = _batch(tmp_path)
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        store.apply_ingestion(tmp_path, batch)
        before = _semantic_dump(store)
        if failure == "sql":
            store._connection.execute("""
                CREATE TEMP TRIGGER fail_late_occurrence BEFORE INSERT ON occurrences
                WHEN NEW.id = 'new-513' BEGIN SELECT RAISE(ABORT, 'late insert'); END
            """)
        first_chunk_written = False

        def occurrences():
            nonlocal first_chunk_written
            for i in range(1025):
                if i == 513:
                    first_chunk_written = (
                        store._connection.execute(
                            "SELECT count(*) FROM occurrences WHERE id LIKE 'new-%'"
                        ).fetchone()[0]
                        == 512
                    )
                    if failure == "generator":
                        raise RuntimeError("late generator")
                yield replace(batch.occurrences[0], id=f"new-{i}")

        with pytest.raises((RuntimeError, sqlite3.IntegrityError), match="late"):
            store.apply_ingestion(tmp_path, replace(batch, occurrences=occurrences()))
        assert first_chunk_written
        assert not store._connection.in_transaction
        assert _semantic_dump(store) == before


def test_insert_chunks_obey_connection_variables_and_payload_without_rejecting_large_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStore(tmp_path / "index.db") as store:
        connection = store._connection
        connection.execute("CREATE TEMP TABLE probe(id INTEGER PRIMARY KEY, value TEXT)")
        old_limit = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 6)
        monkeypatch.setattr(storage, "_INSERT_BATCH_BYTES", 64)
        statements = []
        connection.set_trace_callback(statements.append)
        try:
            with connection:
                sql = "INSERT INTO probe VALUES {values}"
                store._insert_rows(sql, (), columns=2)
                assert statements == []
                # Variable limit: 3 rows; payload (8 + 4*4) each permits only 2.
                store._insert_rows(sql, ((i, "🙂" * 4) for i in range(5)), columns=2)
                store._insert_rows(sql, ((5, "x" * 1000), (6, "tail")), columns=2)
            inserts = [s for s in statements if s.startswith("INSERT INTO probe")]
            assert len(inserts) == 5  # 2/2/1 plus oversize alone and tail.
            assert (
                connection.execute("SELECT length(value) FROM probe WHERE id=5").fetchone()[0]
                == 1000
            )
            statements.clear()
            monkeypatch.setattr(storage, "_INSERT_BATCH_BYTES", 1024 * 1024)
            with connection:
                store._insert_rows(sql, ((i, "v") for i in range(7, 14)), columns=2)
            assert len([s for s in statements if s.startswith("INSERT INTO probe")]) == 3
        finally:
            connection.set_trace_callback(None)
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old_limit)
