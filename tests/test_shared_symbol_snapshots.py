from dataclasses import replace
from pathlib import Path

from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.models import BuildScope, SearchQuery
from cpp_context_engine.storage.sqlite import SQLiteStore


def test_shared_contents_preserve_each_variant_and_fts_parity(tmp_path: Path):
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=tmp_path) as store:
        for index in range(5):
            _put(store, tmp_path, _batch(tmp_path, str(index)))
        connection = store._connection
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema")}
        assert "symbol_snapshot_contents" in tables, "identical snapshots must share one payload"
        assert (
            connection.execute("SELECT count(*) FROM symbol_snapshot_contents").fetchone()[0] == 2
        )
        assert connection.execute("SELECT count(*) FROM symbol_variants").fetchone()[0] == 10
        symbols = store.symbols()
        hits = store.search(SearchQuery("repeated_value"))
        assert len(hits) == 5
        connection.execute("VACUUM")
        store._rebuild_variant_fts()
        assert store.symbols() == symbols
        assert store.search(SearchQuery("repeated_value")) == hits
        connection.execute(
            "INSERT INTO symbol_variant_fts(symbol_variant_fts,rank) VALUES('integrity-check',1)"
        )
    with SQLiteStore(database, project_root=tmp_path) as store:
        assert store.symbols() == symbols
        assert store.search(SearchQuery("repeated_value")) == hits


def test_shared_content_update_delete_and_project_scope(tmp_path: Path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        first = _batch(tmp_path, "one", "alpha")
        second = _batch(tmp_path, "two", "beta")
        _put(store, tmp_path, first)
        _put(store, tmp_path, second)
        assert (
            store._connection.execute("SELECT count(*) FROM symbol_snapshot_contents").fetchone()[0]
            == 2
        )
        scope = BuildScope(("alpha", "beta"))
        before = store.symbols(build_scope=scope)
        changed = replace(
            first,
            symbols=(
                replace(first.symbols[0], source_text="int UpdatedOnly = 9;"),
                first.symbols[1],
            ),
        )
        _put(store, tmp_path, changed)
        assert len(store.search(SearchQuery("UpdatedOnly"), build_scope=scope)) == 1
        assert len(store.search(SearchQuery("repeated_value"), build_scope=scope)) == 1
        store.validate_deep_navigation_parity(tmp_path, (changed, second))
        assert store.remove_build_variant("alpha")
        assert store.symbols(build_scope=scope) == tuple(
            s for s in before if s.build_variant == "beta"
        )
        assert (
            store._connection.execute("SELECT count(*) FROM symbol_snapshot_contents").fetchone()[0]
            == 2
        )
        other = tmp_path / "other"
        _put(store, other, _batch(other, "two", "beta"))
        assert (
            store._connection.execute("SELECT count(*) FROM symbol_snapshot_contents").fetchone()[0]
            == 4
        )
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []
