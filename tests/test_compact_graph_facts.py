from dataclasses import replace

from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.models import GraphEdge, GraphRelation
from cpp_context_engine.storage.sqlite import SQLiteStore


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
