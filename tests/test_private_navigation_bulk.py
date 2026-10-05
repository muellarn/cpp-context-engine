from __future__ import annotations

import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.kicad_canary import semantic_snapshot
from cpp_context_engine.models import (
    BuildConfiguration,
    CallDispatchKind,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    CodeSymbol,
    IndexProfile,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.storage.sqlite import SQLiteStore


def _batch(root: Path) -> IngestionBatch:
    source = root / "source.cpp"
    configuration = BuildConfiguration("build", source, root, ("c++", str(source)), "command")
    unit = TranslationUnit(
        "unit", "build", source, "content", index_profile=IndexProfile.NAVIGATION
    )
    symbol = CodeSymbol(
        "alpha",
        "alpha",
        SymbolKind.FUNCTION,
        SourceSpan(source, 1, 1),
        source_text="int alpha();",
        build_configuration_id="build",
        translation_unit_id="unit",
    )
    return IngestionBatch((configuration,), (unit,), (symbol,), (), ())


def _apply(store, root, batches, *, known=True):
    return store.apply_ingestion_batches(
        root,
        batches,
        index_profile=IndexProfile.NAVIGATION,
        current_translation_unit_ids=frozenset({"unit"}) if known else None,
        changed_translation_unit_ids=frozenset({"unit"}) if known else None,
    )


def test_private_known_navigation_checks_constraints_once_before_publication(tmp_path: Path):
    database = tmp_path / "index.db"
    root = tmp_path / "project"
    with SQLiteStore.indexing_generation(database) as store:
        connection = store._connection
        original = store._classify_schema_indexes()
        statements = []
        connection.set_trace_callback(statements.append)

        def batches():
            assert not database.exists()
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0
            remaining = store._classify_schema_indexes()
            assert all(
                item.unique
                or item.origin != "c"
                or item.table
                not in {"edges", "occurrences", "symbol_variants", "callsites", "call_targets"}
                for item in remaining.values()
            )
            yield _batch(root)

        _apply(store, root, batches())
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert store._classify_schema_indexes() == original
        assert "PRAGMA integrity_check" in statements
        assert "PRAGMA foreign_key_check" in statements
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert database.exists()


@pytest.mark.parametrize("preexisting,known", [(True, True), (False, False)])
def test_noneligible_streams_keep_online_foreign_keys(tmp_path: Path, preexisting, known):
    database = tmp_path / "index.db"
    root = tmp_path / "project"
    if preexisting:
        SQLiteStore(database).close()
    with SQLiteStore.indexing_generation(database) as store:

        def batches():
            assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            yield _batch(root)

        _apply(store, root, batches(), known=known)


@pytest.mark.parametrize("failure", ["foreign_key", "duplicate", "generator"])
def test_private_failure_rolls_back_and_restores_enforcement(tmp_path: Path, failure):
    database = tmp_path / "index.db"
    root = tmp_path / "project"
    with (
        pytest.raises((RuntimeError, ValueError, sqlite3.IntegrityError)),
        SQLiteStore.indexing_generation(database) as store,
    ):

        def batches():
            batch = _batch(root)
            if failure == "foreign_key":
                batch = replace(
                    batch,
                    translation_units=(
                        replace(batch.translation_units[0], build_configuration_id="missing"),
                    ),
                )
            yield batch
            if failure == "duplicate":
                yield batch
            if failure == "generator":
                raise RuntimeError("injected generator failure")

        try:
            _apply(store, root, batches())
        finally:
            assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert store._connection.execute("SELECT count(*) FROM projects").fetchone()[0] == 0
    assert not database.exists()


def test_second_private_application_uses_normal_constraints(tmp_path: Path):
    root = tmp_path / "project"
    with SQLiteStore.indexing_generation(tmp_path / "index.db") as store:
        _apply(store, root, (_batch(root),))

        def batches():
            assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            yield _batch(root)

        _apply(store, root, batches())


def test_private_rows_match_enforced_generation(tmp_path: Path):
    root = tmp_path / "project"
    fast, baseline = tmp_path / "fast.db", tmp_path / "baseline.db"
    with SQLiteStore.indexing_generation(fast) as store:
        _apply(store, root, (_batch(root),))
    with SQLiteStore(baseline) as store:
        _apply(store, root, (_batch(root),))
    assert semantic_snapshot(fast) == semantic_snapshot(baseline)


def test_replacing_unknown_stream_keeps_cross_unit_delete_cascade(tmp_path: Path):
    root = tmp_path / "project"
    batch_a = _batch(root)
    span = batch_a.symbols[0].span
    site = CallSite(
        "site",
        "alpha",
        CallDispatchKind.DIRECT,
        span,
        span,
        True,
        static_target_symbol_id="alpha",
        translation_unit_id="unit",
        build_configuration_id="build",
    )
    batch_a = replace(batch_a, callsites=(site,))
    unit_b = replace(batch_a.translation_units[0], id="unit-b")
    symbol_b = replace(batch_a.symbols[0], translation_unit_id="unit-b")
    target = CallTarget(
        "target",
        "site",
        "alpha",
        CallTargetCertainty.CERTAIN,
        1.0,
        "fixture",
        "fixture",
        span,
        translation_unit_id="unit-b",
        build_configuration_id="build",
    )
    batch_b = replace(
        batch_a,
        translation_units=(unit_b,),
        symbols=(symbol_b,),
        callsites=(),
        call_targets=(target,),
    )
    with SQLiteStore.indexing_generation(tmp_path / "index.db") as store:
        _apply(store, root, (batch_a, batch_b, batch_a), known=False)
        assert store._connection.execute("SELECT count(*) FROM callsites").fetchone()[0] == 1
        assert store._connection.execute("SELECT count(*) FROM call_targets").fetchone()[0] == 0


@pytest.mark.parametrize("failure", ["index", "validation", "commit", "cancel"])
def test_private_finalization_failure_never_publishes(tmp_path: Path, failure):
    root = tmp_path / "project"
    database = tmp_path / "index.db"
    cancelled = threading.Event()

    class FailingStore(SQLiteStore):
        def _restore_fresh_generation_indexes(self, indexes):
            if failure == "index":
                raise RuntimeError("injected index failure")
            super()._restore_fresh_generation_indexes(indexes)

        def _validate_fresh_generation(self):
            super()._validate_fresh_generation()
            if failure == "validation":
                raise RuntimeError("injected validation failure")
            if failure == "cancel":
                cancelled.set()
            if failure == "commit":
                self._connection.set_authorizer(
                    lambda action, value, *_: (
                        sqlite3.SQLITE_DENY
                        if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT"
                        else sqlite3.SQLITE_OK
                    )
                )

    with (
        pytest.raises((RuntimeError, sqlite3.DatabaseError)),
        FailingStore.indexing_generation(database) as store,
    ):
        original = store._classify_schema_indexes()
        try:
            store.apply_ingestion_batches(
                root,
                (_batch(root),),
                index_profile=IndexProfile.NAVIGATION,
                current_translation_unit_ids=frozenset({"unit"}),
                changed_translation_unit_ids=frozenset({"unit"}),
                cancelled=cancelled,
            )
        finally:
            store._connection.set_authorizer(None)
            assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert store._connection.execute("PRAGMA cache_size").fetchone()[0] == -2000
            assert store._connection.execute("SELECT count(*) FROM projects").fetchone()[0] == 0
            assert store._classify_schema_indexes() == original
    assert not database.exists()


def test_enforcement_restore_failure_closes_private_connection(tmp_path: Path):
    root = tmp_path / "project"
    database = tmp_path / "index.db"

    class FailingStore(SQLiteStore):
        def _set_foreign_key_enforcement(self, enabled):
            if enabled:
                raise RuntimeError("injected enforcement restoration failure")
            super()._set_foreign_key_enforcement(enabled)

    with (
        pytest.raises(RuntimeError, match="injected enforcement"),
        FailingStore.indexing_generation(database) as store,
    ):
        _apply(store, root, (_batch(root),))
    assert not database.exists()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        store._connection.execute("SELECT 1")


def test_rollback_failure_cannot_leave_reusable_private_store(tmp_path: Path):
    database = tmp_path / "index.db"
    root = tmp_path / "project"
    with (
        pytest.raises(RuntimeError, match="closed before publication"),
        SQLiteStore.indexing_generation(database) as store,
    ):

        def batches():
            yield _batch(root)
            store._connection.set_authorizer(
                lambda action, value, *_: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_TRANSACTION and value == "ROLLBACK"
                    else sqlite3.SQLITE_OK
                )
            )
            raise RuntimeError("trigger rollback")

        with pytest.raises(sqlite3.DatabaseError):
            _apply(store, root, batches())
        assert store._closed
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            store._connection.execute("SELECT 1")
    assert not database.exists()
