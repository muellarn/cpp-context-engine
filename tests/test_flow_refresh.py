import sqlite3
import threading
from pathlib import Path

import pytest
from test_summary_refresh import _seed_star, _solution_snapshot

from cpp_context_engine.storage.sqlite import SQLiteStore


def _seed_flows(store, root, *, prefix="", build_variant="default"):
    project_id = _seed_star(store, root, caller_count=2, prefix=prefix, build_variant=build_variant)
    connection = store._connection
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute(
        "UPDATE function_summaries SET parameter_modes_json='[\"value\"]', "
        "parameter_location_ids_json='[\"parameter\"]' WHERE id=?",
        (f"{prefix}s0",),
    )
    connection.executemany(
        "INSERT INTO call_argument_bindings VALUES "
        "(?, ?, ?, ?, 0, NULL, 'unknown', NULL, '[]', 0, 1, '', ?, ?, ?)",
        (
            (
                project_id,
                f"{prefix}binding{i}",
                f"{prefix}s{i}",
                f"{prefix}c{i}",
                f"{prefix}tu{i}",
                f"{prefix}cfg{i}",
                build_variant,
            )
            for i in (1, 2)
        ),
    )
    connection.commit()
    return project_id


def _rows(connection):
    return tuple(
        tuple(row)
        for row in connection.execute("SELECT * FROM interprocedural_flows ORDER BY project_id, id")
    )


def test_identical_refresh_does_not_rewrite_flows(tmp_path: Path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        project_id = _seed_flows(store, tmp_path)
        connection = store._connection
        with connection:
            store._refresh_summary_solutions(project_id, "default", {"f0", "f1", "f2"})
        before = _rows(connection)
        assert len(before) == 2 and all(len(row) == 17 for row in before)
        # Identical flow IDs in a different project must neither participate nor be mutated.
        other_project = store._ensure_project(str(tmp_path / "other-project"))
        foreign_row = (other_project, *before[0][1:])
        with connection:
            connection.execute(
                "INSERT INTO interprocedural_flows VALUES (" + ",".join("?" * 17) + ")",
                foreign_row,
            )
        before = _rows(connection)
        summary_before = _solution_snapshot(store)
        connection.execute("CREATE TEMP TABLE flow_mutations(kind TEXT)")
        for operation in ("DELETE", "INSERT", "UPDATE"):
            connection.execute(
                f"CREATE TEMP TRIGGER observe_flow_{operation.lower()} "
                f"AFTER {operation} ON interprocedural_flows BEGIN "
                f"INSERT INTO flow_mutations VALUES ('{operation}'); END"
            )
        statements = []
        connection.set_trace_callback(statements.append)
        with connection:
            store._refresh_summary_solutions(project_id, "default", {"f0", "f1", "f2"})
        connection.set_trace_callback(None)
        assert _rows(connection) == before
        assert _solution_snapshot(store) == summary_before
        assert list(connection.execute("SELECT kind FROM flow_mutations")) == []
        comparisons = [
            sql
            for sql in statements
            if sql.lstrip().startswith("SELECT") and "FROM interprocedural_flows" in sql
        ]
        assert len(comparisons) == 1
        plan = [row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + comparisons[0])]
        assert any(
            "SEARCH interprocedural_flows" in step and "project_id=?" in step for step in plan
        )
        assert not any("SCAN interprocedural_flows" in step for step in plan)
        assert not any("TEMP B-TREE" in step for step in plan)


@pytest.mark.parametrize(
    "column,value",
    [
        ("id", "zzz-changed-id"),
        ("kind", "writeback"),
        ("caller_summary_id", "s2"),
        ("callee_summary_id", "s1"),
        ("callsite_id", "c2"),
        ("target_symbol_id", "f1"),
        ("target_certainty", "possible"),
        ("certainty", "possible"),
        ("reason", "different reason with unchanged IDs and payloads"),
        ("argument_index", 2),
        ("caller_location_id", "changed-caller"),
        ("callee_location_id", None),
        ("caller_access_id", "changed-access"),
        ("translation_unit_id", "changed-tu"),
        ("build_configuration_id", "changed-cfg"),
        ("build_variant", "changed-build"),
        ("missing", None),
        ("extra", None),
    ],
)
def test_refresh_repairs_flow_fields_and_sequence_length(tmp_path, column, value):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        project_id = _seed_flows(store, tmp_path)
        connection = store._connection
        with connection:
            store._refresh_summary_solutions(project_id, "default", {"f0", "f1", "f2"})
        before = _rows(connection)
        last_id = before[-1][1]
        with connection:
            if column == "missing":
                connection.execute("DELETE FROM interprocedural_flows WHERE id=?", (last_id,))
            elif column == "extra":
                extra = list(before[-1])
                extra[1] = "zzz-extra"
                connection.execute(
                    "INSERT INTO interprocedural_flows VALUES (" + ",".join("?" * 17) + ")", extra
                )
            else:
                connection.execute(
                    f"UPDATE interprocedural_flows SET {column}=? WHERE id=?", (value, last_id)
                )
        assert _rows(connection) != before
        with connection:
            store._refresh_summary_solutions(project_id, "default", {"f0", "f1", "f2"})
        assert _rows(connection) == before


def test_removed_flows_preserve_other_callers_and_builds(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        project_id = _seed_flows(store, tmp_path)
        _seed_flows(store, tmp_path, prefix="alt-", build_variant="alternative")
        connection = store._connection
        with connection:
            for variant in ("default", "alternative"):
                store._refresh_summary_solutions(project_id, variant, {"f0", "f1", "f2"})
        before = _rows(connection)
        assert len(before) == 4
        with connection:
            connection.execute("DELETE FROM call_argument_bindings WHERE caller_summary_id='s1'")
            store._refresh_summary_solutions(project_id, "default", {"f1"})
        assert _rows(connection) == tuple(row for row in before if row[3] != "s1")
        with connection:
            connection.execute("DELETE FROM call_argument_bindings WHERE caller_summary_id='s2'")
            store._refresh_summary_solutions(project_id, "default", {"f0", "f1", "f2"})
        assert _rows(connection) == tuple(row for row in before if row[-1] == "alternative")


@pytest.mark.parametrize("phase", ["comparison", "delete", "insert_failure"])
def test_flow_reconciliation_cancellation_and_failure_roll_back(tmp_path, phase):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        project_id = _seed_flows(store, tmp_path)
        connection = store._connection
        with connection:
            store._refresh_summary_solutions(project_id, "default", {"f0", "f1", "f2"})
            if phase != "comparison":
                connection.execute("UPDATE interprocedural_flows SET reason='requires replacement'")
        before = _rows(connection)
        summary_before = _solution_snapshot(store)
        cancelled = threading.Event()
        if phase == "insert_failure":
            connection.execute(
                "CREATE TEMP TRIGGER fail_flow BEFORE INSERT ON interprocedural_flows "
                "BEGIN SELECT RAISE(ABORT, 'flow write failed'); END"
            )
            error, message = sqlite3.IntegrityError, "flow write failed"
        else:

            def cancel(sql):
                if (
                    phase == "comparison"
                    and sql.lstrip().startswith("SELECT")
                    and "FROM interprocedural_flows" in sql
                ) or (phase == "delete" and sql.startswith("DELETE FROM interprocedural_flows")):
                    cancelled.set()

            connection.set_trace_callback(cancel)
            error, message = RuntimeError, "indexing was cancelled"
        with pytest.raises(error, match=message), connection:
            store._refresh_summary_solutions(
                project_id, "default", {"f0", "f1", "f2"}, cancelled=cancelled
            )
        connection.set_trace_callback(None)
        assert not connection.in_transaction
        assert _rows(connection) == before
        assert _solution_snapshot(store) == summary_before
