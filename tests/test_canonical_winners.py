import sqlite3
from dataclasses import replace

import pytest
from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.storage.sqlite import SQLiteStore


def _seed(store, root, fanout, count=2):
    for index in reversed(range(fanout)):
        batch = _batch(root, f"unit-{index:03}", "beta" if index % 2 else "alpha")
        symbols = tuple(
            replace(
                batch.symbols[0],
                id=f"symbol-{number:04}",
                source_text=f"int value_{index}_{number};",
                metadata={"is_definition": index != 0},
            )
            for number in range(count)
        )
        _put(store, root, replace(batch, symbols=symbols))


def _legacy_selection(store, project, ids):
    preferred = {}
    for row in store._connection.execute(
        "SELECT * FROM symbol_variant_snapshots WHERE project_id=? "
        "ORDER BY symbol_id,is_definition DESC,build_variant,translation_unit_id",
        (project,),
    ):
        if row["symbol_id"] in ids and row["symbol_id"] not in preferred:
            preferred[row["symbol_id"]] = store._snapshot_symbol(
                row["snapshot_json"], row if row["provenance_removed"] else None
            )
    return list(preferred.values())


@pytest.mark.parametrize("fanout,count", [(1, 2), (7, 2), (2, 503)])
def test_refresh_fetches_only_winning_payloads(tmp_path, monkeypatch, fanout, count):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path, fanout, count)
        project = store._project_id()
        ids = {f"symbol-{number:04}" for number in range(count)} | {"missing"}
        expected = _legacy_selection(store, project, ids)
        payload_rows = []
        selections = []
        original = store._put_canonical_symbols

        def track_selection(project_id, symbols, *, prefer_definition):
            symbols = list(symbols)
            selections.extend(symbols)
            return original(project_id, symbols, prefer_definition=prefer_definition)

        def track_payload(cursor, values):
            row = sqlite3.Row(cursor, values)
            if "snapshot_json" in row.keys():  # noqa: SIM118 - Row membership checks values
                payload_rows.append(row["id"])
            return row

        monkeypatch.setattr(store, "_put_canonical_symbols", track_selection)
        connection = store._connection
        old_limit = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        connection.row_factory = track_payload
        try:
            store._refresh_symbols(project, ids)
        finally:
            connection.row_factory = sqlite3.Row
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old_limit)
        assert selections == expected
        assert len(payload_rows) == count, "discarded variants must not materialize payloads"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_refresh_isolates_projects_and_preserves_empty_requests(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=first) as store:
        _seed(store, first, 3)
        _seed(store, second, 1)
        project = store._project_id(first)
        expected = _legacy_selection(store, project, {"symbol-0000", "symbol-0001"})
        store._refresh_symbols(project, {"symbol-0000", "symbol-0001", "missing"})
        rows = list(store._connection.execute("SELECT * FROM symbols ORDER BY project_id,id"))
        changes = store._connection.total_changes
        store._refresh_symbols(project, set())
        store._refresh_symbols(project, {"missing"})
        assert store._connection.total_changes == changes
        assert (
            list(store._connection.execute("SELECT * FROM symbols ORDER BY project_id,id")) == rows
        )
        assert [s.build_configuration_id for s in expected] == ["configuration-unit-002"] * 2


def test_refresh_failure_rolls_back_ingestion(tmp_path, monkeypatch):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path, 2)
        tables = ("symbols", "symbol_variants", "symbol_snapshot_contents", "translation_units")
        before = {
            table: list(store._connection.execute(f"SELECT * FROM {table} ORDER BY rowid"))
            for table in tables
        }
        original = store._snapshot_symbol
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("injected winner decode failure")
            return original(*args)

        monkeypatch.setattr(store, "_snapshot_symbol", fail_second)
        with pytest.raises(ValueError, match="injected winner decode failure"):
            _put(store, tmp_path, _batch(tmp_path, "new-unit"))
        assert not store._connection.in_transaction
        assert {
            table: list(store._connection.execute(f"SELECT * FROM {table} ORDER BY rowid"))
            for table in tables
        } == before


def test_refresh_plan_limits_preference_seek_before_snapshot_fetch(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path, 3)
        statements = []
        store._connection.set_trace_callback(statements.append)
        store._refresh_symbols(store._project_id(), {f"symbol-{i:04}" for i in range(500)})
        store._connection.set_trace_callback(None)
        (selection,) = [statement for statement in statements if "WITH requested" in statement]
        plan = [row[3] for row in store._connection.execute("EXPLAIN QUERY PLAN " + selection)]
        assert "MATERIALIZE preferred" in plan
        assert any("CORRELATED SCALAR SUBQUERY" in step for step in plan)
        assert any(
            "symbol_variants_symbol_preference (project_id=? AND symbol_id=?)" in step
            for step in plan
        )
        assert any(
            "SEARCH variants USING INDEX" in step and "(project_id=? AND id=?)" in step
            for step in plan
        ), plan
        assert "SEARCH contents USING INTEGER PRIMARY KEY (rowid=?)" in plan
