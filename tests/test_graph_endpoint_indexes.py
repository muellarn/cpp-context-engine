import sqlite3
from dataclasses import replace

import pytest
from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.models import GraphDirection, GraphEdge, GraphRelation
from cpp_context_engine.storage.sqlite import SQLiteStore


def _seed_graph(store, root):
    batch = _batch(root, "unit")
    symbols = tuple(
        replace(symbol, id=name) for symbol, name in zip(batch.symbols, ("A", "B"), strict=True)
    )
    edge = GraphEdge(
        "A",
        "B",
        GraphRelation.CALLS,
        "unit",
        id="edge",
        build_configuration_id=batch.build_configurations[0].id,
    )
    _put(store, root, replace(batch, symbols=symbols, edges=(edge,)))


def _indexes(connection):
    return {
        row[1]: tuple(item[2] for item in connection.execute(f'PRAGMA index_info("{row[1]}")'))
        for row in connection.execute("PRAGMA index_list(edges)")
    }


def _assert_endpoint_indexes(connection):
    indexes = _indexes(connection)
    assert "edges_source" not in indexes
    assert "edges_target" not in indexes
    for endpoint in ("source", "target"):
        assert indexes[f"edges_scope_{endpoint}"] == (
            "project_id",
            f"{endpoint}_id",
            "build_variant",
            "relation",
        )
        for suffix in ("", " AND build_variant='default' AND relation='calls'"):
            plan = [
                row[3]
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN SELECT rowid FROM edges "
                    f"WHERE project_id=1 AND {endpoint}_id='B'{suffix}"
                )
            ]
            assert any(f"{endpoint}_id=?" in step for step in plan), plan
            assert not any("SCAN" in step for step in plan), plan


def _legacy_layout(connection):
    connection.execute("DROP INDEX IF EXISTS edges_overrides_scope")
    for name in ("edges_source", "edges_target", "edges_scope_source", "edges_scope_target"):
        connection.execute(f"DROP INDEX IF EXISTS {name}")
    for endpoint in ("source", "target"):
        connection.execute(
            f"CREATE INDEX edges_{endpoint} ON edges(project_id,{endpoint}_id,relation)"
        )
        connection.execute(
            f"CREATE INDEX edges_scope_{endpoint} "
            f"ON edges(project_id,build_variant,{endpoint}_id,relation)"
        )
    connection.execute("PRAGMA user_version=19")
    connection.commit()


def test_graph_endpoint_indexes_cover_fk_and_scoped_seeks(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed_graph(store, tmp_path)
        _assert_endpoint_indexes(store._connection)


def test_graph_endpoint_migration_preserves_facts_and_queries(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed_graph(store, tmp_path)
        before = tuple(
            tuple(row)
            for row in store._connection.execute("SELECT * FROM edges ORDER BY project_id,id")
        )
        expected = store.neighbors("B", build_scope=("default", "alternate"))
        _legacy_layout(store._connection)
    with SQLiteStore(path, project_root=tmp_path) as store:
        _assert_endpoint_indexes(store._connection)
        assert (
            tuple(
                tuple(row)
                for row in store._connection.execute("SELECT * FROM edges ORDER BY project_id,id")
            )
            == before
        )
        assert store.neighbors("B", build_scope=("default", "alternate")) == expected
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []
        store._connection.execute("DELETE FROM symbols WHERE id='B'")
        assert store._connection.execute("SELECT count(*) FROM edges").fetchone()[0] == 0


def test_graph_endpoint_migration_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed_graph(store, tmp_path)
        _legacy_layout(store._connection)
        expected = _indexes(store._connection)
    original = SQLiteStore._migrate_v20

    def fail_second_index(store):
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

    monkeypatch.setattr(SQLiteStore, "_migrate_v20", fail_second_index)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        SQLiteStore(path, project_root=tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 19
        assert _indexes(connection) == expected
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_override_seed_avoids_excluded_build_edges(tmp_path):
    costs = []
    for extra in (32, 1024):
        root = tmp_path / str(extra)
        root.mkdir()
        with SQLiteStore(root / "index.db", project_root=root) as store:
            _seed_graph(store, root)
            store.put_edges(
                (
                    GraphEdge(
                        "B",
                        "A",
                        GraphRelation.OVERRIDES,
                        "unit",
                        id="override",
                        build_configuration_id="configuration-unit",
                    ),
                )
            )
            batch = _batch(root, "excluded-unit", "excluded")
            symbols = tuple(
                replace(symbol, id=name)
                for symbol, name in zip(batch.symbols, ("A", "B"), strict=True)
            )
            edges = tuple(
                GraphEdge(
                    "A",
                    "B",
                    GraphRelation.CALLS,
                    "excluded-unit",
                    id=f"noise{index}",
                    build_configuration_id="configuration-excluded-unit",
                    build_variant="excluded",
                )
                for index in range(extra)
            )
            _put(store, root, replace(batch, symbols=symbols, edges=edges))
            steps = 0

            def progress():
                nonlocal steps
                steps += 1
                return 0

            query = (
                "SELECT target_id,source_id FROM edges WHERE project_id=? "
                "AND build_variant=? AND relation='overrides'"
            )
            parameters = (store._project_id(root), "default")
            store._connection.set_progress_handler(progress, 1)
            try:
                rows = [tuple(row) for row in store._connection.execute(query, parameters)]
            finally:
                store._connection.set_progress_handler(None, 0)
            assert rows == [("A", "B")]
            costs.append(steps)
    assert costs[1] <= costs[0] + 100, costs


@pytest.mark.parametrize("direction", list(GraphDirection))
@pytest.mark.parametrize(
    "relations", [None, frozenset({GraphRelation.CALLS}), frozenset({GraphRelation.OVERRIDES})]
)
@pytest.mark.parametrize("limit", [None, 100])
def test_ordered_neighbors_keep_endpoint_seeks(tmp_path, direction, relations, limit):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed_graph(store, tmp_path)
        statements = []
        store._connection.set_trace_callback(statements.append)
        try:
            store.neighbors("B", direction=direction, relations=relations, per_node_limit=limit)
        finally:
            store._connection.set_trace_callback(None)
        query = next(sql for sql in statements if "relation FROM edges" in sql)
        plan = [row[3] for row in store._connection.execute("EXPLAIN QUERY PLAN " + query)]
        assert not any("SCAN edges" in step for step in plan), plan
        edges = [step for step in plan if "edges USING" in step]
        assert edges
        assert all("source_id=?" in step or "target_id=?" in step for step in edges), plan


@pytest.mark.parametrize("direction", list(GraphDirection))
@pytest.mark.parametrize(
    "relations", [None, frozenset({GraphRelation.CALLS}), frozenset({GraphRelation.OVERRIDES})]
)
def test_neighbor_work_does_not_grow_with_unrelated_edges(tmp_path, direction, relations):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed_graph(store, tmp_path)
        costs = []
        results = []
        for count in (32, 1024):
            store.put_edges(
                tuple(
                    GraphEdge(
                        "A",
                        "A",
                        GraphRelation.CALLS,
                        "unit",
                        id=f"noise{index}",
                        build_configuration_id="configuration-unit",
                    )
                    for index in range(count)
                )
            )
            steps = 0

            def progress():
                nonlocal steps
                steps += 1
                return 0

            store._connection.set_progress_handler(progress, 1)
            try:
                results.append(
                    store.neighbors(
                        "B", direction=direction, relations=relations, per_node_limit=100
                    )
                )
            finally:
                store._connection.set_progress_handler(None, 0)
            costs.append(steps)
        assert results[0] == results[1]
        assert costs[1] <= costs[0] + 100, costs
