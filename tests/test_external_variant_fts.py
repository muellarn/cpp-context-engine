import sqlite3
from dataclasses import replace

import pytest
from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.models import BuildScope, SearchQuery
from cpp_context_engine.storage.sqlite import SQLiteStore


def _check(store):
    connection = store._connection
    connection.execute(
        "INSERT INTO symbol_variant_fts(symbol_variant_fts,rank) VALUES('integrity-check',1)"
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def _documents(connection):
    return [
        tuple(row)
        for row in connection.execute("SELECT rowid,* FROM symbol_variant_fts ORDER BY rowid")
    ]


def _legacy_layout(connection):
    documents = _documents(connection)
    connection.execute("DROP TABLE symbol_variant_fts")
    connection.execute("DROP VIEW IF EXISTS symbol_variant_fts_source")
    connection.execute("DROP TABLE IF EXISTS symbol_variant_fts_rows")
    connection.execute("""
        CREATE VIRTUAL TABLE symbol_variant_fts USING fts5(
            project_id UNINDEXED,variant_id UNINDEXED,symbol_id UNINDEXED,
            build_variant UNINDEXED,qualified_name,signature,documentation,source_text,
            tokenize='unicode61')
    """)
    connection.executemany(
        "INSERT INTO symbol_variant_fts(rowid,project_id,variant_id,"
        "symbol_id,build_variant,qualified_name,signature,documentation,"
        "source_text) VALUES(?,?,?,?,?,?,?,?,?)",
        documents,
    )
    connection.execute("PRAGMA user_version=20")
    connection.commit()
    return documents


def test_fts_does_not_duplicate_snapshot_text_and_preserves_rank(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        for index in range(5):
            _put(store, tmp_path, _batch(tmp_path, str(index)))
        connection = store._connection
        assert not connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='symbol_variant_fts_content'"
        ).fetchone(), "FTS duplicates already preserved snapshot text"
        before = store.search(SearchQuery("repeated_value"))
        assert len(before) == 5
        documents = _documents(connection)
        ids = connection.execute(
            "SELECT record_id,variant_id FROM symbol_variant_fts_rows"
        ).fetchall()
        connection.execute("VACUUM")
        assert (
            connection.execute(
                "SELECT record_id,variant_id FROM symbol_variant_fts_rows"
            ).fetchall()
            == ids
        )
        assert _documents(connection) == documents
        store._rebuild_variant_fts()
        assert store.search(SearchQuery("repeated_value")) == before
        _check(store)


def test_fts_migration_exact_documents_and_atomic_failure(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _put(store, tmp_path, _batch(tmp_path, "one"))
        expected = store.search(SearchQuery("repeated_value"))
        documents = _legacy_layout(store._connection)
    original = SQLiteStore._migrate_v21

    def fail(store):
        store._connection.set_authorizer(
            lambda action, name, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_CREATE_VTABLE and name == "symbol_variant_fts"
                else sqlite3.SQLITE_OK
            )
        )
        try:
            original(store)
        finally:
            store._connection.set_authorizer(None)
            store.close()

    with monkeypatch.context() as patch:
        patch.setattr(SQLiteStore, "_migrate_v21", fail)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            SQLiteStore(path, project_root=tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 20
        assert _documents(connection) == documents
    with SQLiteStore(path, project_root=tmp_path) as store:
        assert _documents(store._connection) == documents
        assert store.search(SearchQuery("repeated_value")) == expected
        _check(store)


def test_fts_update_delete_and_reopen_preserve_documents(tmp_path):
    path = tmp_path / "index.db"
    scope = BuildScope(("alpha", "beta"))
    with SQLiteStore(path, project_root=tmp_path) as store:
        first = _batch(tmp_path, "one", "alpha")
        second = _batch(tmp_path, "two", "beta")
        _put(store, tmp_path, first)
        _put(store, tmp_path, second)
        changed = replace(
            first, symbols=(replace(first.symbols[0], source_text="OnlyNew"), first.symbols[1])
        )
        _put(store, tmp_path, changed)
        assert len(store.search(SearchQuery("OnlyNew"), build_scope=scope)) == 1
        assert len(store.search(SearchQuery("repeated_value"), build_scope=scope)) == 1
        _check(store)
        assert store.remove_build_variant("alpha")
        assert not store.search(SearchQuery("OnlyNew"), build_scope=scope)
        _check(store)
        expected = store.search(SearchQuery("repeated_value"), build_scope=scope)
    with SQLiteStore(path, project_root=tmp_path) as store:
        assert store.search(SearchQuery("repeated_value"), build_scope=scope) == expected
        _check(store)


def test_fts_migration_rejects_missing_document_without_repair(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _put(store, tmp_path, _batch(tmp_path, "one"))
        _legacy_layout(store._connection)
        store._connection.execute("DELETE FROM symbol_variant_fts WHERE rowid=1")
        store._connection.commit()
    with pytest.raises(RuntimeError, match="FTS.*document"):
        SQLiteStore(path, project_root=tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 20
        assert connection.execute("SELECT count(*) FROM symbol_variant_fts").fetchone()[0] == 1
