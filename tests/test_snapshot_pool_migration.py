import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from test_symbol_snapshot_compression import _batch, _materialize_v18_snapshots, _put

import cpp_context_engine.storage.sqlite as storage
from cpp_context_engine.kicad_canary import semantic_snapshot
from cpp_context_engine.models import BuildScope, SearchQuery
from cpp_context_engine.storage.sqlite import SQLiteStore


@pytest.mark.parametrize("legacy", ("canonical", "mismatch", "missing", "format"))
def test_v18_pool_migration_preserves_exact_legacy_snapshot(tmp_path: Path, legacy: str):
    database = tmp_path / "legacy.db"
    batch = _batch(tmp_path, "one", "alternate")
    with SQLiteStore(database, project_root=tmp_path) as store:
        _put(store, tmp_path, batch)
        before = store.symbols(build_scope=("alternate",))
        hits = store.search(SearchQuery("repeated_value"), build_scope=("alternate",))
        _materialize_v18_snapshots(store)
        row = store._connection.execute(
            "SELECT id,snapshot_json FROM symbol_variants WHERE symbol_id='shared-symbol'"
        ).fetchone()
        data = json.loads(row["snapshot_json"])
        if legacy == "mismatch":
            data["build_configuration_id"] = "original-json-configuration"
        elif legacy == "missing":
            del data["build_variant"]
            del data["variant_id"]
        raw = json.dumps(data, sort_keys=True, indent=2 if legacy == "format" else None)
        store._connection.execute(
            "UPDATE symbol_variants SET snapshot_json=? WHERE id=?",
            (storage._encode_symbol_snapshot(raw), row["id"]),
        )
        store._connection.commit()
    with SQLiteStore(database, project_root=tmp_path) as store:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 22
        migrated = store._connection.execute(
            "SELECT * FROM symbol_variant_snapshots WHERE id=?", (row["id"],)
        ).fetchone()
        assert storage._full_variant_snapshot(migrated) == raw
        assert bool(migrated["provenance_removed"]) is (legacy == "canonical")
        assert store.symbols(build_scope=("alternate",)) == before
        assert store.search(SearchQuery("repeated_value"), build_scope=("alternate",)) == hits
        store._refresh_symbols(store._project_id(), {"shared-symbol"})
        canonical = store._connection.execute(
            "SELECT * FROM symbols WHERE id='shared-symbol'"
        ).fetchone()
        assert (
            store._row_to_symbol(canonical).build_configuration_id == data["build_configuration_id"]
        )
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert [tuple(row) for row in store._connection.execute("PRAGMA integrity_check")] == [
            ("ok",)
        ]


@pytest.mark.parametrize("stage", ("row", "publication"))
def test_pool_migration_failure_restores_schema_rows_and_foreign_keys(tmp_path, monkeypatch, stage):
    database = tmp_path / "legacy.db"
    with SQLiteStore(database, project_root=tmp_path) as store:
        _put(store, tmp_path, _batch(tmp_path, "one"))
        _materialize_v18_snapshots(store)
        before = tuple(tuple(r) for r in store._connection.execute("SELECT * FROM symbol_variants"))
    captured = []

    def fail(store, step):
        captured.append(store)
        if step == stage:
            raise RuntimeError("injected pool migration failure")

    with monkeypatch.context() as patch:
        patch.setattr(SQLiteStore, "_snapshot_pool_migration_checkpoint", fail)
        with pytest.raises(RuntimeError, match="injected pool migration failure"):
            SQLiteStore(database, project_root=tmp_path)
    connection = captured[-1]._connection
    try:
        assert not connection.in_transaction
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 18
        assert (
            tuple(tuple(r) for r in connection.execute("SELECT * FROM symbol_variants")) == before
        )
        assert not connection.execute(
            "SELECT name FROM sqlite_schema "
            "WHERE name IN ('symbol_snapshot_contents','symbol_variant_snapshots')"
        ).fetchall()
    finally:
        captured[-1].close()
    with SQLiteStore(database, project_root=tmp_path) as store:
        assert len(store.symbols()) == 2


def test_pool_hash_collision_and_corrupt_payload_fail_atomically(tmp_path, monkeypatch):
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=tmp_path) as store:
        batch = _batch(tmp_path, "one")
        _put(store, tmp_path, batch)
        before = semantic_snapshot(database, _connection=store._connection)
        key = store._connection.execute(
            "SELECT content_hash FROM symbol_snapshot_contents LIMIT 1"
        ).fetchone()[0]
        with monkeypatch.context() as patch:
            patch.setattr(storage, "_symbol_content_hash", lambda *_: key)
            with pytest.raises(RuntimeError, match="hash collision"):
                _put(store, tmp_path, _batch(tmp_path, "two"))
        assert semantic_snapshot(database, _connection=store._connection) == before
        store._connection.execute(
            "UPDATE symbol_snapshot_contents SET snapshot_json='corrupt' WHERE content_hash=?",
            (key,),
        )
        store._connection.commit()
        with pytest.raises((RuntimeError, ValueError)):
            _put(store, tmp_path, _batch(tmp_path, "two"))
        assert (
            store._connection.execute("SELECT count(*) FROM translation_units").fetchone()[0] == 1
        )


def test_semantic_snapshot_ignores_pool_allocation_but_detects_payload_changes(tmp_path):
    digests = []
    for reverse in (False, True):
        database = tmp_path / f"{reverse}.db"
        with SQLiteStore(database, project_root=tmp_path) as store:
            batch = _batch(tmp_path, "one")
            if reverse:
                batch = replace(batch, symbols=tuple(reversed(batch.symbols)))
            _put(store, tmp_path, batch)
            digests.append(semantic_snapshot(database, _connection=store._connection))
    assert digests[0] == digests[1]
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE symbol_snapshot_contents SET snapshot_json='corrupt' WHERE id=1")
    assert semantic_snapshot(database) != digests[1]


def test_snapshot_foreign_key_cannot_cross_projects(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _put(store, tmp_path, _batch(tmp_path, "one"))
        other = tmp_path / "other"
        _put(store, other, _batch(other, "one"))
        other_id = store._connection.execute(
            "SELECT id FROM symbol_snapshot_contents WHERE project_id=? LIMIT 1",
            (store._project_id(other),),
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError), store._connection:
            store._connection.execute(
                "UPDATE symbol_variants SET snapshot_id=? WHERE project_id=?",
                (other_id, store._project_id(tmp_path)),
            )
        assert len(store.symbols(build_scope=BuildScope.single())) == 2


@pytest.mark.parametrize("failure", ("commit", "rollback", "restore"))
def test_pool_migration_transaction_failures_cannot_leave_unsafe_connection(tmp_path, failure):
    database = tmp_path / "legacy.db"
    with SQLiteStore(database, project_root=tmp_path) as store:
        _put(store, tmp_path, _batch(tmp_path, "one"))
        _materialize_v18_snapshots(store)
    captured = []

    class FailingStore(SQLiteStore):
        def _snapshot_pool_migration_checkpoint(self, stage):
            if stage != "publication":
                return
            captured.append(self)

            def deny(action, name, value, *_):
                transaction = action == sqlite3.SQLITE_TRANSACTION and name == failure.upper()
                restore = (
                    failure == "restore"
                    and action == sqlite3.SQLITE_PRAGMA
                    and name == "foreign_keys"
                    and value is not None
                )
                return sqlite3.SQLITE_DENY if transaction or restore else sqlite3.SQLITE_OK

            self._connection.set_authorizer(deny)
            if failure == "rollback":
                raise RuntimeError("original migration failure")

    expected = RuntimeError if failure == "rollback" else sqlite3.DatabaseError
    message = "original migration failure" if failure == "rollback" else "not authorized"
    with pytest.raises(expected, match=message):
        FailingStore(database, project_root=tmp_path)
    failed = captured[-1]
    try:
        if failure == "commit":
            assert not failed._connection.in_transaction
            assert failed._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert failed._connection.execute("PRAGMA user_version").fetchone()[0] == 18
        else:
            assert failed._closed
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                failed._connection.execute("SELECT 1")
    finally:
        failed.close()
    with SQLiteStore(database, project_root=tmp_path) as reopened:
        assert len(reopened.symbols()) == 2
        assert reopened._connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_pool_interning_bounds_decoded_bytes_not_only_document_count(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_SNAPSHOT_BATCH_BYTES", 1024, raising=False)
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _put(store, tmp_path, _batch(tmp_path, "one"))
        consumed = []
        at_insert = []

        def documents():
            for index in range(20):
                consumed.append(index)
                yield 0, str(index) + "x" * 800

        store._connection.set_trace_callback(
            lambda sql: (
                at_insert.append(len(consumed))
                if sql.startswith("INSERT INTO symbol_snapshot_contents ")
                else None
            )
        )
        identifiers = store._intern_symbol_snapshots(store._project_id(), documents())
        assert len(set(identifiers)) == 20
        assert at_insert[0] <= 2, "one bounded chunk plus at most one lookahead document"
