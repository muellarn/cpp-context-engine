import hashlib
import sqlite3
from dataclasses import replace

import pytest
from test_compact_occurrences import _legacy_occurrence_layout, _seed
from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.kicad_canary import _encode_digest_value, semantic_snapshot
from cpp_context_engine.storage.sqlite import SQLiteStore


def _reference(connection):
    count = connection.execute("SELECT count(*) FROM occurrences").fetchone()[0]
    digest = hashlib.sha256(f"occurrences\0{count}\0".encode())
    for row in connection.execute(
        "SELECT * FROM occurrences ORDER BY " + ",".join(map(str, range(1, 15)))
    ):
        for value in row:
            digest.update(_encode_digest_value(value))
        digest.update(b"\xff")
    return count, digest.hexdigest()


def _populate(store, root):
    for project in (root / "z-root", root / "a-root"):
        for unit in ("z-unit", "a-unit", "ä-unit"):
            occurrence = _seed(store, project, unit)
            with store._connection:
                store._put_occurrences(
                    store._project_id(project),
                    (replace(occurrence, id=identity) for identity in ("z", "a", "名前")),
                )
        _put(store, project, _batch(project, "empty-unit"))
        with store._connection:
            store._connection.execute(
                "INSERT INTO occurrence_units(project_id,translation_unit_id) VALUES(?,?)",
                (store._project_id(project), "empty-unit"),
            )


def test_normalized_digest_uses_unsorted_payload_seeks(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _populate(store, tmp_path)
        expected = _reference(store._connection)
        statements = []
        store._connection.set_trace_callback(statements.append)
        result = semantic_snapshot(path, _connection=store._connection)
        store._connection.set_trace_callback(None)
        assert (result["counts"]["occurrences"], result["table_digests"]["occurrences"]) == expected
        reads = [
            sql
            for sql in statements
            if 'FROM "occurrences"' in sql and not sql.startswith("SELECT count(*)")
        ]
        assert len(reads) == 8, "read each public project/TU through its existing index"
        for query in reads:
            plan = [row[3] for row in store._connection.execute("EXPLAIN QUERY PLAN " + query)]
            assert not any("TEMP B-TREE" in step for step in plan), plan
            assert any(
                "sqlite_autoindex_occurrence_records_1 (project_id=? AND unit_key=?)" in step
                for step in plan
            ), plan


@pytest.mark.parametrize("legacy", [False, True])
def test_digest_preserves_public_order_all_values_and_legacy(tmp_path, legacy):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _populate(store, tmp_path)
        expected = _reference(store._connection)
        before = semantic_snapshot(path, _connection=store._connection)
        with store._connection:
            store._connection.execute("PRAGMA defer_foreign_keys=ON")
            store._connection.execute("UPDATE occurrence_units SET key=10000-key")
            store._connection.execute("UPDATE occurrence_contexts SET unit_key=10000-unit_key")
            store._connection.execute("UPDATE occurrence_records SET unit_key=10000-unit_key")
        assert semantic_snapshot(path, _connection=store._connection) == before
        if legacy:
            _legacy_occurrence_layout(store._connection)
        result = semantic_snapshot(path, _connection=store._connection)
        assert (result["counts"]["occurrences"], result["table_digests"]["occurrences"]) == expected
        with store._connection:
            table = "occurrences" if legacy else "occurrence_records"
            store._connection.execute(f"UPDATE {table} SET metadata_json='changed'")
        changed = semantic_snapshot(path, _connection=store._connection)
        assert changed["table_digests"]["occurrences"] != result["table_digests"]["occurrences"]
        assert (
            changed["counts"]["occurrences"],
            changed["table_digests"]["occurrences"],
        ) == _reference(store._connection)


@pytest.mark.parametrize("units", [1, 2])
def test_digest_owns_consistent_read_snapshot_across_count_and_units(tmp_path, units):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        for unit in range(units):
            _seed(store, tmp_path, str(unit))
        before = semantic_snapshot(path, _connection=store._connection)
        writer = sqlite3.connect(path)
        triggered = []

        def change_after_count(query):
            if (
                'FROM "occurrences"' in query
                and not query.startswith("SELECT count(*)")
                and not triggered
            ):
                triggered.append(True)
                writer.execute("UPDATE occurrence_records SET metadata_json='concurrent-change'")
                writer.commit()

        store._connection.set_trace_callback(change_after_count)
        try:
            observed = semantic_snapshot(path, _connection=store._connection)
        finally:
            store._connection.set_trace_callback(None)
            writer.close()
        assert triggered
        assert observed == before
        assert not store._connection.in_transaction
        assert semantic_snapshot(path, _connection=store._connection) != before


def test_digest_preserves_callers_transaction_on_success_and_failure(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        connection = store._connection
        connection.execute("BEGIN")
        connection.execute("UPDATE occurrence_records SET metadata_json='caller-owned'")
        semantic_snapshot(path, _connection=connection)
        assert connection.in_transaction
        connection.set_progress_handler(lambda: 1, 1)
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            semantic_snapshot(path, _connection=connection)
        connection.set_progress_handler(None, 0)
        assert connection.in_transaction
        assert (
            connection.execute("SELECT metadata_json FROM occurrence_records").fetchone()[0]
            == "caller-owned"
        )
        connection.rollback()


def test_digest_cleans_up_its_read_transaction_after_failure(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        connection = store._connection
        # Reject an occurrence read after the snapshot transaction has begun.
        connection.set_authorizer(
            lambda action, name, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and name == "occurrence_records"
                else sqlite3.SQLITE_OK
            )
        )
        with pytest.raises(sqlite3.DatabaseError):
            semantic_snapshot(path, _connection=connection)
        connection.set_authorizer(None)
        assert not connection.in_transaction
        assert semantic_snapshot(path, _connection=connection)["counts"]["occurrences"] == 1


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_digest_closes_unit_and_row_cursors_after_consumer_failure(
    tmp_path, monkeypatch, caller_transaction
):
    from cpp_context_engine import kicad_canary

    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)

    cursors = []

    class TrackedCursor(sqlite3.Cursor):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    class TrackedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("SELECT project_id, translation_unit_id FROM occurrence_units") or (
                'FROM "occurrences" WHERE' in sql
            ):
                cursor = self.cursor(factory=TrackedCursor)
                cursors.append(cursor)
                return cursor.execute(sql, parameters)
            return super().execute(sql, parameters)

    original = kicad_canary._encode_digest_value

    def interrupted_consumer(value):
        if len(cursors) == 2:
            raise RuntimeError("consumer interrupted after row delivery")
        return original(value)

    connection = sqlite3.connect(path, factory=TrackedConnection)
    try:
        if caller_transaction:
            connection.execute("BEGIN")
        monkeypatch.setattr(kicad_canary, "_encode_digest_value", interrupted_consumer)
        with pytest.raises(RuntimeError, match="consumer interrupted") as caught:
            semantic_snapshot(path, _connection=connection)
        # Keep the traceback alive: collection must not be required for cleanup.
        assert caught.value.__traceback__ is not None
        assert len(cursors) == 2
        assert all(cursor.closed for cursor in cursors)
        assert connection.in_transaction == caller_transaction
    finally:
        connection.close()
