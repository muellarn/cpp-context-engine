from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.models import (
    CallDispatchKind,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    GraphEdge,
    GraphRelation,
    SourceSpan,
)
from cpp_context_engine.storage.sqlite import SQLiteStore


def _seed_graph(store, root):
    project = store._ensure_project(str(root))
    other = store._ensure_project(str(root / "other"))
    connection = store._connection
    connection.commit()
    # Only the graph tables are relevant to these query plans. Like the summary
    # SQL fixtures, omit unrelated AST parents without changing tables or indexes.
    connection.execute("PRAGMA foreign_keys = OFF")
    edges = tuple(
        GraphEdge(child, base, GraphRelation.OVERRIDES, f"tu-{index}", id=f"e{index}")
        for index, (base, child) in enumerate(
            (
                ("B", "D1"),
                ("D1", "D2"),
                ("B", "D3"),
                ("D1", "D3"),
                ("X", "Y"),
                ("Y", "X"),
                ("B", "D1"),
            )
        )
    )
    span = SourceSpan(root / "calls.cpp", 1, 1)
    sites = tuple(
        CallSite(
            id=name,
            owner_symbol_id=owner,
            static_target_symbol_id=target,
            dispatch_kind=dispatch,
            spelling_span=span,
            expansion_span=span,
            target_set_complete=True,
            translation_unit_id=f"tu-{name}",
            build_configuration_id=f"config-{name}",
        )
        for name, target, owner, dispatch in (
            ("sb", "B", "callerB", CallDispatchKind.VIRTUAL),
            ("sd", "D1", "callerD", CallDispatchKind.VIRTUAL),
            ("sx", "X", "callerX", CallDispatchKind.VIRTUAL),
            ("direct", "B", "directOwner", CallDispatchKind.DIRECT),
            ("cycle", "callerB", "D2", CallDispatchKind.DIRECT),
        )
    )

    def target(site, symbol):
        return CallTarget(
            id=f"target-{site.id}-{symbol}",
            callsite_id=site.id,
            target_symbol_id=symbol,
            certainty=CallTargetCertainty.CERTAIN,
            confidence=1.0,
            confidence_reason="fixture",
            derivation="fixture",
            evidence_span=span,
            translation_unit_id=site.translation_unit_id,
            build_configuration_id=site.build_configuration_id,
        )

    targets = tuple(target(site, site.static_target_symbol_id) for site in sites)
    targets += (target(sites[0], "D3"), target(sites[0], "D2"), target(sites[1], "D2"))
    # A small unrelated component makes project-wide scans visible in VM counts.
    for index in range(32):
        site = replace(sites[0], id=f"noise{index}", static_target_symbol_id=f"N{index}")
        sites += (site,)
        targets += (target(site, f"N{index}"),)
        edges += (
            GraphEdge(
                f"M{index}",
                f"N{index}",
                GraphRelation.OVERRIDES,
                f"noise-tu{index}",
                id=f"noise-edge{index}",
            ),
        )
    store._put_edges(project, edges)
    store._put_edges(
        project, (replace(edges[0], id="alt-edge", source_id="AltOnly", build_variant="alternate"),)
    )
    store._put_edges(other, (replace(edges[0], source_id="OtherOnly"),))
    store._put_call_facts(project, sites, targets)
    store._put_call_facts(
        project,
        (replace(sites[0], id="alt", owner_symbol_id="AltCaller", build_variant="alternate"),),
        (
            replace(
                targets[-1],
                id="alt-target",
                callsite_id="alt",
                target_symbol_id="D2",
                build_variant="alternate",
            ),
        ),
    )
    store._put_call_facts(
        other,
        (replace(sites[0], owner_symbol_id="OtherCaller"),),
        (targets[0], target(sites[0], "D2")),
    )
    connection.commit()
    return project


def _measured_rows(connection, sql):
    steps = 0

    def count():
        nonlocal steps
        steps += 1
        return 0

    connection.set_progress_handler(count, 1)
    try:
        rows = tuple(tuple(row) for row in connection.execute(sql))
    finally:
        connection.set_progress_handler(None, 0)
    return rows, steps


@pytest.mark.parametrize("query", ["override", "reverse"])
def test_graph_queries_use_target_lookups_and_preserve_results(tmp_path: Path, query: str):
    with SQLiteStore(Path(":memory:")) as store:
        project = _seed_graph(store, tmp_path)
        connection = store._connection
        statements = []
        connection.execute("SAVEPOINT observe_graph_query")
        connection.set_trace_callback(statements.append)
        try:
            if query == "override":
                store._refresh_indexed_override_candidates(project, "default")
            else:
                assert store._reverse_summary_callers(project, "default", {"D2"}) == {
                    "callerB",
                    "callerD",
                }
        finally:
            connection.set_trace_callback(None)
            connection.execute("ROLLBACK TO observe_graph_query")
            connection.execute("RELEASE observe_graph_query")
        prefix = "WITH RECURSIVE" if query == "override" else "SELECT DISTINCT sites"
        sql = next(item for item in statements if item.lstrip().startswith(prefix))
        plan = [row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + sql)]
        print(query, "PLAN", plan)
        if query == "override":
            assert any(
                "SEARCH edges" in step and "target_id=?" in step and "relation=?" in step
                for step in plan
            )
            assert any(
                "SEARCH sites" in step
                and "static_target_symbol_id=?" in step
                and "build_variant=?" in step
                for step in plan
            )
            legacy = sql.replace("CROSS JOIN edges", "JOIN edges").replace(
                "FROM override_closure AS closure\n            CROSS JOIN callsites AS sites",
                "FROM callsites AS sites\n            JOIN override_closure AS closure",
            )
        else:
            assert any(
                "SEARCH targets" in step
                and "target_symbol_id=?" in step
                and "build_variant=?" in step
                for step in plan
            )
            assert any("SEARCH sites" in step and "id=?" in step for step in plan)
            legacy = sql.replace("CROSS JOIN callsites", "JOIN callsites")
            # Production consumes a set. Compare both result sequences under the
            # same explicit order without imposing that sort on production.
            sql += " ORDER BY sites.owner_symbol_id"
            legacy += " ORDER BY sites.owner_symbol_id"
        actual, actual_steps = _measured_rows(connection, sql)
        expected, legacy_steps = _measured_rows(connection, legacy)
        print(query, "VM_STEPS", {"legacy": legacy_steps, "current": actual_steps})
        assert actual == expected
        assert actual_steps <= legacy_steps
        if query == "override":
            assert tuple(row[:2] for row in actual) == (
                *((f"noise{i}", f"M{i}") for i in sorted(range(32), key=lambda i: f"noise{i}")),
                ("sb", "D1"),
                ("sd", "D3"),
                ("sx", "Y"),
            )
        else:
            assert actual == (("callerB",), ("callerD",))
