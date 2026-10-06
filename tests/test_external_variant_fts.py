import json
import sqlite3
from dataclasses import replace

import pytest
from test_symbol_snapshot_compression import _batch, _put

import cpp_context_engine.storage.sqlite as storage
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


def test_internal_external_fts_scores_and_public_ranking_are_identical(tmp_path):
    path = tmp_path / "index.db"
    terms = ("needle", "café")
    with SQLiteStore(path, project_root=tmp_path) as store:
        for index in range(6):
            batch = _batch(tmp_path, str(index), "alpha" if index % 2 else "beta")
            _put(
                store,
                tmp_path,
                replace(
                    batch,
                    symbols=(
                        replace(
                            batch.symbols[0],
                            id=f"s{index}",
                            qualified_name=f"ns::needle{index}",
                            signature="int needle(café)",
                            documentation="needle " * index,
                            source_text="café needle body " * (1 + index * 20),
                        ),
                    ),
                ),
            )
        other = tmp_path / "other"
        _put(store, other, _batch(other, "other", "alpha"))
        _legacy_layout(store._connection)
        expected = {}
        for scope in (("alpha",), ("alpha", "beta")):
            placeholders = ",".join("?" for _ in scope)
            for symbols_only in (False, True):
                weights = "12,6,0,0" if symbols_only else "8,4,2,1"
                for term in terms:
                    expression = (
                        f'qualified_name : ("{term}") OR signature : ("{term}")'
                        if symbols_only
                        else f'"{term}"'
                    )
                    rows = store._connection.execute(
                        f"SELECT variants.id, variants.snapshot_json, "
                        f"bm25(symbol_variant_fts,0,0,0,0,{weights}) AS rank "
                        "FROM symbol_variant_fts JOIN symbol_variant_snapshots variants "
                        "ON variants.project_id=CAST(symbol_variant_fts.project_id AS INTEGER) "
                        "AND variants.id=symbol_variant_fts.variant_id "
                        "WHERE symbol_variant_fts MATCH ? AND variants.project_id=? "
                        f"AND variants.build_variant IN ({placeholders}) "
                        "ORDER BY rank,variants.build_variant,variants.id LIMIT 50",
                        (expression, store._project_id(tmp_path), *scope),
                    ).fetchall()
                    results = [
                        (
                            row["id"],
                            -row["rank"],
                            json.loads(storage._decode_symbol_snapshot(row["snapshot_json"])),
                        )
                        for row in rows
                    ]
                    if symbols_only:
                        results = [
                            (
                                identifier,
                                score
                                + 2 * (doc["qualified_name"].casefold() == term)
                                + (doc["qualified_name"].casefold().endswith("::" + term)),
                                doc,
                            )
                            for identifier, score, doc in results
                        ]
                        results.sort(
                            key=lambda item: (-item[1], item[2]["qualified_name"], item[2]["id"])
                        )
                    expected[scope, symbols_only, term] = [
                        (identifier, score) for identifier, score, _ in results
                    ]
    with SQLiteStore(path, project_root=tmp_path) as store:
        for (scope, symbols_only, term), want in expected.items():
            method = store.search_symbols if symbols_only else store.search
            got = method(SearchQuery(term, limit=50), build_scope=scope)
            assert [(hit.symbol.variant_id, hit.score) for hit in got] == want
        _check(store)


def test_fts_snapshot_cache_is_one_bounded_document(tmp_path, monkeypatch):
    decode = storage._decode_symbol_snapshot
    calls = []

    def counted(blob):
        calls.append(len(blob))
        return decode(blob)

    monkeypatch.setattr(storage, "_decode_symbol_snapshot", counted)
    fields = ("qualified_name", "signature", "documentation", "source_text")
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:

        def read(source, *, encode=True):
            document = dict(zip(fields, ("name", "signature", "docs", source), strict=True))
            raw = json.dumps(document)
            blob = storage._encode_symbol_snapshot(raw) if encode else raw
            assert tuple(store._sqlite_snapshot_field(blob, field) for field in fields) == tuple(
                document.values()
            )
            return blob

        first = read("first")
        assert len(calls) == 1
        read("second")
        assert len(calls) == 2
        store._sqlite_snapshot_field(first, "source_text")
        assert len(calls) == 3
        read("x" * 65537, encode=False)
        assert len(calls) == 7 and store._fts_snapshot_cache is None
        read("x" * 262145)
        assert len(calls) == 11 and store._fts_snapshot_cache is None


def test_fresh_validation_rejects_variant_missing_from_map_and_index(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _put(store, tmp_path, _batch(tmp_path, "one"))
        c = store._connection
        rowid = c.execute("SELECT record_id FROM symbol_variant_fts_rows LIMIT 1").fetchone()[0]
        c.execute("DELETE FROM symbol_variant_fts WHERE rowid=?", (rowid,))
        c.execute("DELETE FROM symbol_variant_fts_rows WHERE record_id=?", (rowid,))
        c.execute(
            "INSERT INTO symbol_variant_fts(symbol_variant_fts,rank) VALUES('integrity-check',1)"
        )
        with pytest.raises(RuntimeError, match="FTS.*coverage"):
            store._validate_fresh_generation()
