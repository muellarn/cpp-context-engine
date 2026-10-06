import sqlite3
from dataclasses import replace

import pytest
from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.kicad_canary import semantic_snapshot
from cpp_context_engine.models import GraphEdge, GraphRelation
from cpp_context_engine.storage.sqlite import SQLiteStore


def _legacy_graph_layout(connection):
    connection.execute("""CREATE TABLE edges_legacy (
        project_id INTEGER NOT NULL, id TEXT NOT NULL, translation_unit_id TEXT NOT NULL,
        build_configuration_id TEXT NOT NULL, build_variant TEXT NOT NULL,
        source_id TEXT NOT NULL, target_id TEXT NOT NULL, relation TEXT NOT NULL,
        PRIMARY KEY(project_id,id),
        FOREIGN KEY(project_id,translation_unit_id)
            REFERENCES translation_units(project_id,id) ON DELETE CASCADE,
        FOREIGN KEY(project_id,source_id) REFERENCES symbols(project_id,id) ON DELETE CASCADE,
        FOREIGN KEY(project_id,target_id) REFERENCES symbols(project_id,id) ON DELETE CASCADE)""")
    connection.execute("INSERT INTO edges_legacy SELECT * FROM edges")
    connection.execute("DROP VIEW edges")
    for table in ("edge_records", "edge_provenance", "edge_symbols"):
        connection.execute(f"DROP TABLE {table}")
    connection.execute("ALTER TABLE edges_legacy RENAME TO edges")
    for endpoint in ("source", "target"):
        connection.execute(
            f"CREATE INDEX edges_scope_{endpoint} "
            f"ON edges(project_id,{endpoint}_id,build_variant,relation)"
        )
    connection.execute("CREATE INDEX edges_tu ON edges(project_id,translation_unit_id)")
    connection.execute(
        "CREATE INDEX edges_overrides_scope "
        "ON edges(project_id,build_variant,target_id,source_id) WHERE relation='overrides'"
    )
    connection.execute("PRAGMA user_version=22")
    connection.commit()


def _seed(store, root, unit="unit"):
    batch = _batch(root, unit)
    edge = GraphEdge(
        batch.symbols[0].id,
        batch.symbols[1].id,
        GraphRelation.CALLS,
        unit,
        id="unusual public ID: λ",
        build_configuration_id="exact legacy mismatch",
    )
    _put(store, root, replace(batch, edges=(edge,)))
    return edge


def test_repeated_graph_facts_use_bounded_physical_storage(tmp_path):
    count = 8192
    unit = "tu_" + "a" * 32
    source = "symbol_" + "b" * 32
    target = "symbol_" + "c" * 32
    batch = _batch(tmp_path, unit)
    symbols = tuple(
        replace(symbol, id=identifier)
        for symbol, identifier in zip(batch.symbols, (source, target), strict=True)
    )
    edges = tuple(
        GraphEdge(
            source,
            target,
            GraphRelation.CALLS,
            unit,
            id=f"edge_{index:032x}",
            build_configuration_id=batch.build_configurations[0].id,
        )
        for index in range(count)
    )
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _put(store, tmp_path, replace(batch, symbols=symbols, edges=edges))
        assert store._connection.execute("SELECT count(*) FROM edges").fetchone()[0] == count
        assert len(store.neighbors(source)) == count
        occupied = store._connection.execute(
            "SELECT sum(pgsize) FROM dbstat WHERE name='edges' OR name LIKE 'edges_%' "
            "OR name LIKE 'edge_%' OR name LIKE 'sqlite_autoindex_edges_%' "
            "OR name LIKE 'sqlite_autoindex_edge_%'"
        ).fetchone()[0]
        assert occupied <= count * 260, (occupied, count, occupied / count)


def test_graph_migration_preserves_exact_ids_provenance_and_cascades(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        edge = _seed(store, tmp_path)
        before = store.neighbors(edge.source_id)
        assert before == (edge,)
        _legacy_graph_layout(store._connection)
        old = [tuple(row) for row in store._connection.execute("SELECT * FROM edges")]
    with SQLiteStore(path, project_root=tmp_path) as store:
        assert [tuple(row) for row in store._connection.execute("SELECT * FROM edges")] == old
        assert store.neighbors(edge.source_id) == before
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []
        store._connection.execute("DELETE FROM symbols WHERE id=?", (edge.target_id,))
        assert store._connection.execute("SELECT count(*) FROM edges").fetchone()[0] == 0
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_graph_duplicate_ids_ignore_new_invalid_parents_and_keep_first(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        edge = _seed(store, tmp_path)
        store.put_edges((replace(edge, source_id="absent", translation_unit_id="absent"),))
        fresh = replace(edge, id="new")
        store.put_edges((fresh, replace(fresh, target_id="absent")))
        assert store.neighbors(edge.source_id) == (fresh, edge)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store.put_edges((replace(edge, id="invalid", target_id="absent"),))
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_graph_migration_rolls_back_all_ddl_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        _legacy_graph_layout(store._connection)
        before = [tuple(row) for row in store._connection.execute("SELECT * FROM edges")]
    original = SQLiteStore._migrate_v23

    def fail(store):
        store._connection.set_authorizer(
            lambda action, name, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_CREATE_INDEX and name == "edges_scope_target"
                else sqlite3.SQLITE_OK
            )
        )
        try:
            original(store)
        finally:
            store._connection.set_authorizer(None)
            store.close()

    monkeypatch.setattr(SQLiteStore, "_migrate_v23", fail)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        SQLiteStore(path, project_root=tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 22
        assert connection.execute("SELECT * FROM edges").fetchall() == before
        assert not connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='edge_records'"
        ).fetchone()
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


def test_graph_references_cannot_cross_projects(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        edge = _seed(store, tmp_path)
        other = tmp_path / "other"
        _seed(store, other)
        project = store._project_id(tmp_path)
        other_project = store._project_id(other)
        other_key = store._connection.execute(
            "SELECT key FROM edge_symbols WHERE project_id=? AND symbol_id=?",
            (other_project, edge.source_id),
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store._connection.execute(
                "UPDATE edge_records SET source_key=? WHERE project_id=?", (other_key, project)
            )
        store._connection.rollback()
        with store._connection:
            store._delete_translation_units(project, ("unit",))
        assert store.neighbors(edge.source_id, project_root=other) == (edge,)
        assert (
            store._connection.execute(
                "SELECT count(*) FROM edges WHERE project_id=?", (project,)
            ).fetchone()[0]
            == 0
        )
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_semantic_snapshot_covers_edges_without_local_mapping_keys(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        before = semantic_snapshot(path)
        assert before["counts"]["edges"] == 1
        assert "edges" in before["table_digests"]
        with store._connection:
            store._connection.execute("PRAGMA defer_foreign_keys=ON")
            store._connection.execute("UPDATE edge_symbols SET key=key+1000")
            store._connection.execute("UPDATE edge_provenance SET key=key+1000")
            store._connection.execute(
                "UPDATE edge_records SET source_key=source_key+1000, "
                "target_key=target_key+1000, provenance_key=provenance_key+1000"
            )
        assert semantic_snapshot(path) == before
        with store._connection:
            store._connection.execute("UPDATE edge_records SET relation='overrides'")
        after = semantic_snapshot(path)
        assert after["counts"] == before["counts"]
        assert after["digest"] != before["digest"]
        assert after["table_digests"]["edges"] != before["table_digests"]["edges"]


def test_edge_batches_obey_small_variable_limit_and_rollback_iterator_failure(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        edge = _seed(store, tmp_path)
        connection = store._connection
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        new_edges = tuple(replace(edge, id=f"edge-{index}") for index in range(600))
        store.put_edges((*new_edges, replace(new_edges[0], target_id="missing")))
        assert connection.execute("SELECT count(*) FROM edges").fetchone()[0] == 601
        before = {
            table: tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table}"))
            for table in ("edge_records", "edge_provenance", "edge_symbols")
        }

        def failing():
            for item in new_edges:
                yield replace(item, id="attempt-" + item.id, build_configuration_id="new-config")
            raise ValueError("iterator failed after complete chunks")

        with pytest.raises(ValueError, match="iterator failed"):
            store.put_edges(failing())
        assert {
            table: tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table}"))
            for table in before
        } == before
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
