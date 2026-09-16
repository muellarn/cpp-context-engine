from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from cpp_context_engine.analysis import interprocedural
from cpp_context_engine.analysis.interprocedural import InterproceduralLimits
from cpp_context_engine.models import (
    CallDispatchKind,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    DataFlowCertainty,
    FunctionSummary,
    MemoryLocationKind,
    SourceSpan,
    SummaryEffect,
    SummaryEffectKind,
)
from cpp_context_engine.storage import sqlite as sqlite_storage
from cpp_context_engine.storage.sqlite import SQLiteStore, _encode_summary_payload


def _summary(identifier: str, function_id: str) -> FunctionSummary:
    return FunctionSummary(
        id=identifier,
        function_symbol_id=function_id,
        graph_id=f"graph-{identifier}",
        analysis_id=f"analysis-{identifier}",
        parameter_modes=(),
        parameter_location_ids=(),
        local_complete=True,
        local_incomplete_reasons=(),
        complete=True,
        incomplete_reasons=(),
        recursive=False,
        iteration_count=0,
        max_scc_iterations=32,
        max_scc_size=128,
        max_summary_effects=1024,
        translation_unit_id=f"tu-{identifier}",
        build_configuration_id=f"config-{identifier}",
    )


def test_acyclic_summary_transfer_runs_once_without_changing_solution(monkeypatch) -> None:
    callee = _summary("s0", "f0")
    caller = _summary("s1", "f1")
    span = SourceSpan(Path("calls.cpp"), 1, 1, 1, 5)
    site = CallSite(
        id="c1",
        owner_symbol_id="f1",
        dispatch_kind=CallDispatchKind.DIRECT,
        spelling_span=span,
        expansion_span=span,
        target_set_complete=True,
        static_target_symbol_id="f0",
        callee_text="leaf()",
        translation_unit_id=caller.translation_unit_id,
        build_configuration_id=caller.build_configuration_id,
    )
    target = CallTarget(
        id="t1",
        callsite_id=site.id,
        target_symbol_id=callee.function_symbol_id,
        certainty=CallTargetCertainty.CERTAIN,
        confidence=1.0,
        confidence_reason="direct target",
        derivation="direct",
        evidence_span=span,
        translation_unit_id=site.translation_unit_id,
        build_configuration_id=site.build_configuration_id,
    )
    effect = SummaryEffect(
        id="e0",
        summary_id=callee.id,
        kind=SummaryEffectKind.WRITE,
        location_kind=MemoryLocationKind.GLOBAL,
        certainty=DataFlowCertainty.CERTAIN,
        reason="local write",
        source_access_id="a0",
        translation_unit_id=callee.translation_unit_id,
        build_configuration_id=callee.build_configuration_id,
    )
    propagate_calls = 0
    original = interprocedural._propagate_effect

    def counted_propagation(*args, **kwargs):
        nonlocal propagate_calls
        propagate_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(interprocedural, "_propagate_effect", counted_propagation)
    solution = interprocedural.solve_interprocedural(
        (callee, caller), (effect,), (), (), (), (site,), (target,)
    )

    solved = {item.id: item for item in solution.summaries}
    assert propagate_calls == 1
    # Acyclic callers historically converge on the second equality check. Keep
    # this persisted metadata and the exact solution hash while skipping the work.
    assert solved["s1"].iteration_count == 2
    assert solved["s1"].solution_hash == "solution_968dfa8f2da4f0f019727d3b5863ef11"
    assert tuple(item.id for item in solution.effects) == (
        "e0",
        "summary_effect_0c341cde8221dc5d57170ea6f19bf7d7",
    )

    limited = interprocedural.solve_interprocedural(
        (callee, caller),
        (effect,),
        (),
        (),
        (),
        (site,),
        (target,),
        limits=InterproceduralLimits(max_scc_iterations=1),
    )
    limited_caller = next(item for item in limited.summaries if item.id == "s1")
    assert limited_caller.iteration_count == 1
    assert limited_caller.incomplete_reasons == ("scc_iteration_cap_exceeded",)
    assert limited_caller.solution_hash == "solution_e0c2fe81d048e208b8a00472baeb167f"


def test_cancellable_chunked_payload_encoding_is_byte_exact() -> None:
    effect = SummaryEffect(
        id="encoded-effect",
        summary_id="encoded-summary",
        kind=SummaryEffectKind.WRITE,
        location_kind=MemoryLocationKind.GLOBAL,
        certainty=DataFlowCertainty.CERTAIN,
        reason="x" * (1024 * 1024 + 1),
        source_access_id="source-access",
        is_local=False,
        via_callsite_id="callsite",
        target_symbol_id="target",
    )
    expected = _encode_summary_payload("encoded-summary", (effect,), ())
    polls = 0

    def poll() -> None:
        nonlocal polls
        polls += 1

    assert (
        _encode_summary_payload("encoded-summary", (effect,), (), check_cancelled=poll) == expected
    )
    assert polls >= 3


def _seed_star(
    store: SQLiteStore,
    root: Path,
    *,
    caller_count: int = 3,
    effect_count: int = 2,
    prefix: str = "",
    build_variant: str = "default",
) -> int:
    project_id = store._ensure_project(str(root.resolve()))  # noqa: SLF001
    store._connection.commit()  # noqa: SLF001
    store._connection.execute("PRAGMA foreign_keys = OFF")  # noqa: SLF001
    summaries = []
    for index in range(caller_count + 1):
        summaries.append(
            (
                project_id,
                f"{prefix}s{index}",
                f"f{index}",
                f"{prefix}g{index}",
                f"{prefix}a{index}",
                "[]",
                "[]",
                1,
                "[]",
                1,
                "[]",
                0,
                0,
                32,
                128,
                1024,
                "old",
                f"{prefix}tu{index}",
                f"{prefix}cfg{index}",
                build_variant,
            )
        )
    store._connection.executemany(  # noqa: SLF001
        """
        INSERT INTO function_summaries(
            project_id, id, function_symbol_id, graph_id, analysis_id,
            parameter_modes_json, parameter_location_ids_json, local_complete,
            local_incomplete_reasons_json, complete, incomplete_reasons_json,
            recursive, iteration_count, max_scc_iterations, max_scc_size,
            max_summary_effects, solution_hash, translation_unit_id,
            build_configuration_id, build_variant
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        summaries,
    )
    store._connection.executemany(  # noqa: SLF001
        """
        INSERT INTO summary_effects(
            project_id, id, summary_id, kind, location_kind, certainty, reason,
            parameter_index, access_path_json, location_id, source_access_id,
            is_local, via_callsite_id, target_symbol_id, translation_unit_id,
            build_configuration_id, build_variant
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                project_id,
                f"{prefix}e{index}",
                f"{prefix}s0",
                "write",
                "global",
                "certain",
                "local write",
                None,
                json.dumps((f"field{index}",)),
                None,
                f"{prefix}access{index}",
                1,
                None,
                None,
                f"{prefix}tu0",
                f"{prefix}cfg0",
                build_variant,
            )
            for index in range(effect_count)
        ),
    )
    span = json.dumps(
        {
            "path": str(root / "calls.cpp"),
            "start_line": 1,
            "start_column": 1,
            "end_line": 1,
            "end_column": 5,
        }
    )
    store._connection.executemany(  # noqa: SLF001
        """
        INSERT INTO callsites(
            project_id, id, owner_symbol_id, dispatch_kind, spelling_span_json,
            expansion_span_json, expansion_stack_json, static_target_symbol_id,
            target_set_complete, unresolved_reason, callee_text,
            translation_unit_id, build_configuration_id, build_variant
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                project_id,
                f"{prefix}c{index}",
                f"f{index}",
                "direct",
                "null",
                span,
                "[]",
                "f0",
                1,
                "",
                "leaf()",
                f"{prefix}tu{index}",
                f"{prefix}cfg{index}",
                build_variant,
            )
            for index in range(1, caller_count + 1)
        ),
    )
    store._connection.executemany(  # noqa: SLF001
        """
        INSERT INTO call_targets(
            project_id, id, callsite_id, target_symbol_id, certainty,
            confidence, confidence_reason, derivation, evidence_span_json,
            translation_unit_id, build_configuration_id, build_variant
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                project_id,
                f"{prefix}t{index}",
                f"{prefix}c{index}",
                "f0",
                "certain",
                1.0,
                "direct target",
                "direct",
                span,
                f"{prefix}tu{index}",
                f"{prefix}cfg{index}",
                build_variant,
            )
            for index in range(1, caller_count + 1)
        ),
    )
    store._connection.commit()  # noqa: SLF001
    store._connection.execute("PRAGMA foreign_keys = ON")  # noqa: SLF001
    return project_id


def _solution_snapshot(store: SQLiteStore) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in store._connection.execute(  # noqa: SLF001
            """
            SELECT id, complete, incomplete_reasons_json, recursive,
                   iteration_count, solution_hash
            FROM function_summaries ORDER BY id
            """
        )
    ) + tuple(
        tuple(row)
        for row in store._connection.execute(  # noqa: SLF001
            """
            SELECT summary_id, effect_count, origin_count, uncompressed_bytes,
                   payload_hash, hex(payload)
            FROM summary_solution_payloads ORDER BY summary_id
            """
        )
    )


def test_refresh_uses_projected_inputs_and_deterministic_bounded_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        project_id = _seed_star(store, root)
        # Produce an independent baseline through the former general row converters.
        summaries = tuple(
            store._row_to_function_summary(row)  # noqa: SLF001
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT * FROM function_summaries ORDER BY id"
            )
        )
        effects = tuple(
            store._row_to_summary_effect(row)  # noqa: SLF001
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT * FROM summary_effects WHERE is_local = 1 ORDER BY id"
            )
        )
        sites = tuple(
            store._row_to_callsite(row)  # noqa: SLF001
            for row in store._connection.execute("SELECT * FROM callsites ORDER BY id")  # noqa: SLF001
        )
        targets = tuple(
            store._row_to_call_target(row)  # noqa: SLF001
            for row in store._connection.execute("SELECT * FROM call_targets ORDER BY id")  # noqa: SLF001
        )
        expected = interprocedural.solve_interprocedural(
            summaries, effects, (), (), (), sites, targets
        )
        expected_summaries = {
            item.id: (
                item.complete,
                item.incomplete_reasons,
                item.recursive,
                item.iteration_count,
                item.solution_hash,
            )
            for item in expected.summaries
        }
        expected_payloads = {}
        for summary_id in ("s1", "s2", "s3"):
            expected_payloads[summary_id] = _encode_summary_payload(
                summary_id,
                tuple(
                    item
                    for item in expected.effects
                    if item.summary_id == summary_id and not item.is_local
                ),
                (),
            )

        def reject_general_converter(*_args, **_kwargs):
            raise AssertionError("summary refresh reconstructed an unused full domain row")

        for name in (
            "_row_to_function_summary",
            "_row_to_summary_effect",
            "_row_to_summary_return_origin",
            "_row_to_call_argument_binding",
            "_row_to_call_result_binding",
            "_row_to_callsite",
            "_row_to_call_target",
        ):
            monkeypatch.setattr(store, name, reject_general_converter)
        monkeypatch.setattr(sqlite_storage, "SUMMARY_PAYLOAD_WRITE_BATCH_SIZE", 2)
        batch_sizes = []
        original_batch = store._write_summary_solution_batch  # noqa: SLF001

        def record_batch(rows):
            batch_sizes.append(len(rows))
            original_batch(rows)

        monkeypatch.setattr(store, "_write_summary_solution_batch", record_batch)
        with store._connection:  # noqa: SLF001
            assert (
                store._refresh_summary_solutions(  # noqa: SLF001
                    project_id, "default", {"f0", "f1", "f2", "f3"}
                )
                == 4
            )
        assert batch_sizes == [2, 1]
        actual_summaries = {
            row["id"]: (
                bool(row["complete"]),
                tuple(json.loads(row["incomplete_reasons_json"])),
                bool(row["recursive"]),
                row["iteration_count"],
                row["solution_hash"],
            )
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT * FROM function_summaries ORDER BY id"
            )
        }
        assert actual_summaries == expected_summaries
        actual_payloads = {
            row["summary_id"]: (
                row["effect_count"],
                row["origin_count"],
                row["uncompressed_bytes"],
                row["payload_hash"],
                row["payload"],
            )
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT * FROM summary_solution_payloads ORDER BY summary_id"
            )
        }
        assert actual_payloads == expected_payloads
        first = _solution_snapshot(store)
        batch_sizes.clear()
        with store._connection:  # noqa: SLF001
            store._refresh_summary_solutions(  # noqa: SLF001
                project_id, "default", {"f0", "f1", "f2", "f3"}
            )
        assert batch_sizes == [2, 1]
        assert _solution_snapshot(store) == first


def test_refresh_batch_failure_and_cancellation_roll_back_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        project_id = _seed_star(store, root)
        monkeypatch.setattr(sqlite_storage, "SUMMARY_PAYLOAD_WRITE_BATCH_SIZE", 2)
        with store._connection:  # noqa: SLF001
            store._refresh_summary_solutions(  # noqa: SLF001
                project_id, "default", {"f0", "f1", "f2", "f3"}
            )
        baseline = _solution_snapshot(store)
        original_batch = store._write_summary_solution_batch  # noqa: SLF001

        def fail_batch(rows):
            original_batch(rows)
            raise RuntimeError("injected summary batch failure")

        monkeypatch.setattr(store, "_write_summary_solution_batch", fail_batch)
        with pytest.raises(RuntimeError, match="summary batch failure"), store._connection:  # noqa: SLF001
            store._refresh_summary_solutions(  # noqa: SLF001
                project_id, "default", {"f0", "f1", "f2", "f3"}
            )
        assert _solution_snapshot(store) == baseline

        cancelled = threading.Event()

        def cancel_after_batch(rows):
            original_batch(rows)
            cancelled.set()

        monkeypatch.setattr(store, "_write_summary_solution_batch", cancel_after_batch)
        with pytest.raises(RuntimeError, match="indexing was cancelled"), store._connection:  # noqa: SLF001
            store._refresh_summary_solutions(  # noqa: SLF001
                project_id,
                "default",
                {"f0", "f1", "f2", "f3"},
                cancelled=cancelled,
            )
        assert _solution_snapshot(store) == baseline


def test_projected_refresh_preserves_bindings_origins_flows_and_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        project_id = _seed_star(store, root, caller_count=1)
        connection = store._connection  # noqa: SLF001
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            UPDATE function_summaries
            SET parameter_modes_json = '["reference"]',
                parameter_location_ids_json = '["callee-param"]'
            WHERE id = 's0'
            """
        )
        connection.execute(
            """
            INSERT INTO summary_effects(
                project_id, id, summary_id, kind, location_kind, certainty, reason,
                parameter_index, access_path_json, location_id, source_access_id,
                is_local, via_callsite_id, target_symbol_id, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, 'parameter-write', 's0', 'write', 'parameter', 'certain',
                      'parameter write', 0, '["member"]', 'callee-param',
                      'parameter-access', 1, NULL, NULL, 'tu0', 'cfg0', 'default')
            """,
            (project_id,),
        )
        connection.executemany(
            """
            INSERT INTO summary_return_origins(
                project_id, id, summary_id, kind, certainty, reason, location_kind,
                parameter_index, access_path_json, location_id, callsite_id,
                is_local, via_callsite_id, target_symbol_id, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, 'certain', ?, ?, ?, ?, ?, ?, 1, NULL, NULL, ?, ?, 'default')
            """,
            (
                (
                    project_id,
                    "leaf-origin",
                    "s0",
                    "location",
                    "parameter return",
                    "parameter",
                    0,
                    "[]",
                    "callee-param",
                    None,
                    "tu0",
                    "cfg0",
                ),
                (
                    project_id,
                    "caller-origin",
                    "s1",
                    "call_result",
                    "call return",
                    None,
                    None,
                    "[]",
                    None,
                    "c1",
                    "tu1",
                    "cfg1",
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO call_argument_bindings(
                project_id, id, caller_summary_id, callsite_id, argument_index,
                location_id, location_kind, parameter_index, access_path_json,
                writeback_candidate, complete, incomplete_reason,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, 'binding', 's1', 'c1', 0, 'caller-global', 'global',
                      NULL, '["base"]', 1, 1, '', 'tu1', 'cfg1', 'default')
            """,
            (project_id,),
        )
        connection.execute(
            """
            INSERT INTO call_result_bindings(
                project_id, id, caller_summary_id, callsite_id, location_id,
                definition_access_id, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, 'result', 's1', 'c1', 'result-location',
                      'result-access', 'tu1', 'cfg1', 'default')
            """,
            (project_id,),
        )
        connection.commit()

        inputs = store._load_summary_solver_inputs(  # noqa: SLF001
            project_id, "default", {"f0", "f1"}
        )
        retained = interprocedural.solve_interprocedural(*inputs)
        emitted_effects = []
        emitted_origins = []

        def capture_summary(_summary, effects, origins):
            emitted_effects.extend(effects)
            emitted_origins.extend(origins)

        streamed = interprocedural.solve_interprocedural(
            *inputs,
            emit_summary=capture_summary,
            retain_emitted_facts=False,
        )
        assert streamed.summaries == retained.summaries
        assert streamed.flows == retained.flows
        assert streamed.effects == streamed.return_origins == ()
        assert tuple(sorted(emitted_effects, key=lambda item: item.id)) == retained.effects
        assert tuple(sorted(emitted_origins, key=lambda item: item.id)) == retained.return_origins

        with connection:
            store._refresh_summary_solutions(  # noqa: SLF001
                project_id, "default", {"f0", "f1"}
            )
        flow_rows = tuple(
            tuple(row)
            for row in connection.execute(
                """
                SELECT kind, caller_summary_id, callee_summary_id, callsite_id,
                       target_symbol_id, translation_unit_id,
                       build_configuration_id, build_variant
                FROM interprocedural_flows ORDER BY kind, id
                """
            )
        )
        assert {row[0] for row in flow_rows} == {
            "argument_to_parameter",
            "return_to_caller",
            "writeback",
        }
        assert all(
            row[1:] == ("s1", "s0", "c1", "f0", "tu1", "cfg1", "default") for row in flow_rows
        )
        effects, origins = store._summary_solution_payload(  # noqa: SLF001
            project_id, "s1", ("default",)
        )
        parameter_effect = next(
            item for item in effects if item.source_access_id == "parameter-access"
        )
        propagated_origin = next(item for item in origins if not item.is_local)
        assert (
            parameter_effect.location_kind.value,
            parameter_effect.access_path,
            parameter_effect.location_id,
            parameter_effect.via_callsite_id,
            parameter_effect.target_symbol_id,
            parameter_effect.translation_unit_id,
            parameter_effect.build_configuration_id,
        ) == ("global", ("base", "member"), "caller-global", "c1", "f0", "tu1", "cfg1")
        assert (
            propagated_origin.location_kind.value,
            propagated_origin.access_path,
            propagated_origin.location_id,
            propagated_origin.via_callsite_id,
            propagated_origin.target_symbol_id,
        ) == ("global", ("base",), "caller-global", "c1", "f0")


def test_incremental_reverse_callers_and_build_variants_remain_isolated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        project_id = _seed_star(store, root, caller_count=2)
        _seed_star(
            store,
            root,
            caller_count=2,
            prefix="alt-",
            build_variant="alternative",
        )
        callers = store._reverse_summary_callers(  # noqa: SLF001
            project_id, "default", {"f0"}
        )
        assert callers == {"f1", "f2"}
        with store._connection:  # noqa: SLF001
            assert (
                store._refresh_summary_solutions(  # noqa: SLF001
                    project_id, "default", {"f0"} | callers
                )
                == 3
            )
        assert (
            store._connection.execute(  # noqa: SLF001
                """
            SELECT count(*) FROM function_summaries
            WHERE build_variant = 'alternative' AND solution_hash = 'old'
            """
            ).fetchone()[0]
            == 3
        )
        assert (
            store._connection.execute(  # noqa: SLF001
                """
            SELECT count(*) FROM summary_solution_payloads payloads
            JOIN function_summaries summaries
              ON summaries.project_id = payloads.project_id
             AND summaries.id = payloads.summary_id
            WHERE summaries.build_variant = 'alternative'
            """
            ).fetchone()[0]
            == 0
        )


def test_projected_recursive_refresh_applies_effect_cap(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        project_id = _seed_star(store, root, caller_count=0)
        connection = store._connection  # noqa: SLF001
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.executemany(
            """
            INSERT INTO summary_effects(
                project_id, id, summary_id, kind, location_kind, certainty, reason,
                parameter_index, access_path_json, location_id, source_access_id,
                is_local, via_callsite_id, target_symbol_id, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, ?, 's0', 'write', 'global', 'certain', 'local write',
                      NULL, '[]', NULL, ?, 1, NULL, NULL, 'tu0', 'cfg0', 'default')
            """,
            (
                (project_id, f"extra-{index:04d}", f"extra-access-{index:04d}")
                for index in range(1023)
            ),
        )
        span = json.dumps(
            {
                "path": str(root / "recursive.cpp"),
                "start_line": 1,
                "start_column": 1,
                "end_line": 1,
                "end_column": 5,
            }
        )
        connection.execute(
            """
            INSERT INTO callsites VALUES(
                ?, 'c0', 'f0', 'direct', 'null', ?, '[]', 'f0', 1, '',
                'self()', 'tu0', 'cfg0', 'default')
            """,
            (project_id, span),
        )
        connection.execute(
            """
            INSERT INTO call_targets VALUES(
                ?, 't0', 'c0', 'f0', 'certain', 1.0, 'direct target',
                'direct', ?, 'tu0', 'cfg0', 'default')
            """,
            (project_id, span),
        )
        connection.commit()
        connection.execute("PRAGMA foreign_keys = ON")
        inputs = store._load_summary_solver_inputs(  # noqa: SLF001
            project_id, "default", {"f0"}
        )
        expected = interprocedural.solve_interprocedural(*inputs).summaries[0]
        with connection:
            store._refresh_summary_solutions(project_id, "default", {"f0"})  # noqa: SLF001
        row = connection.execute(
            """
            SELECT recursive, incomplete_reasons_json, iteration_count,
                   max_summary_effects FROM function_summaries WHERE id = 's0'
            """
        ).fetchone()
        assert row["recursive"] == 1
        assert "summary_effect_cap_exceeded" in json.loads(row["incomplete_reasons_json"])
        assert 0 < row["iteration_count"] <= 32
        assert row["max_summary_effects"] == 1024
        assert (
            bool(row["recursive"]),
            tuple(json.loads(row["incomplete_reasons_json"])),
            row["iteration_count"],
            row["max_summary_effects"],
            connection.execute(
                "SELECT solution_hash FROM function_summaries WHERE id = 's0'"
            ).fetchone()[0],
        ) == (
            expected.recursive,
            expected.incomplete_reasons,
            expected.iteration_count,
            expected.max_summary_effects,
            expected.solution_hash,
        )


def test_cancellation_during_solver_rolls_back_before_payload_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        project_id = _seed_star(store, root, caller_count=1)
        connection = store._connection  # noqa: SLF001
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.executemany(
            """
            INSERT INTO summary_effects(
                project_id, id, summary_id, kind, location_kind, certainty, reason,
                parameter_index, access_path_json, location_id, source_access_id,
                is_local, via_callsite_id, target_symbol_id, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, ?, 's0', 'write', 'global', 'certain', 'local write',
                      NULL, '[]', NULL, ?, 1, NULL, NULL, 'tu0', 'cfg0', 'default')
            """,
            ((project_id, f"more-{index:03d}", f"more-access-{index:03d}") for index in range(300)),
        )
        connection.commit()
        connection.execute("PRAGMA foreign_keys = ON")
        baseline = _solution_snapshot(store)
        cancelled = threading.Event()
        original = interprocedural._propagate_effect
        calls = 0

        def cancel_from_transfer(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                cancelled.set()
            return original(*args, **kwargs)

        monkeypatch.setattr(interprocedural, "_propagate_effect", cancel_from_transfer)
        with pytest.raises(RuntimeError, match="indexing was cancelled"), connection:
            store._refresh_summary_solutions(  # noqa: SLF001
                project_id, "default", {"f0", "f1"}, cancelled=cancelled
            )
        assert 1 <= calls <= 256
        assert _solution_snapshot(store) == baseline


def test_payload_batches_honor_aggregate_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        monkeypatch.setattr(sqlite_storage, "SUMMARY_PAYLOAD_WRITE_BATCH_BYTES", 10)
        batches = []
        monkeypatch.setattr(
            store,
            "_write_summary_solution_batch",
            lambda rows: batches.append(tuple(row[1] for row in rows)),
        )
        rows = (
            (1, summary_id, "encoding", 1, 0, 6, "hash", b"123456")
            for summary_id in ("s1", "s2", "s3")
        )
        store._write_summary_solution_rows_batched(rows)  # noqa: SLF001
        assert batches == [("s1",), ("s2",), ("s3",)]
