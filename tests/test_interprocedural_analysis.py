from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from native_cache import cached_native_client, fresh_native_client

from cpp_context_engine.analysis.interprocedural import (
    InterproceduralLimits,
    _solve_variant,
    solve_interprocedural,
)
from cpp_context_engine.ingestion import (
    AnalyzerProtocolError,
    NativeClangIngestor,
)
from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.ingestion.native import _FactBatchBuilder
from cpp_context_engine.models import (
    BuildScope,
    BuildVariant,
    CallArgumentBinding,
    CallDispatchKind,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    DataFlowCertainty,
    FunctionSummary,
    InterproceduralFlowKind,
    MemoryLocationKind,
    SearchQuery,
    SourceSpan,
    SummaryEffect,
    SummaryEffectKind,
)
from cpp_context_engine.storage import SQLiteStore
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION

FIXTURE = Path(__file__).parent / "fixtures" / "interprocedural_project"
pytestmark = pytest.mark.native


def _binary() -> Path:
    configured = os.getenv("CPP_CONTEXT_TEST_ANALYZER")
    candidate = (
        Path(configured)
        if configured
        else Path(__file__).parents[1] / "build" / "clang-analyzer" / "cpp-context-clang-analyzer"
    )
    if not candidate.is_file():
        pytest.skip("Clang analyzer companion has not been built")
    return candidate.resolve()


def _client():
    return cached_native_client(_binary(), timeout_seconds=90)


def _fresh_client():
    return fresh_native_client(_binary(), timeout_seconds=90)


def _batch(*, database: str = "compile_commands.json", variant: str = "default"):
    return NativeClangIngestor(_client()).ingest(FIXTURE, FIXTURE / database, build_variant=variant)


def _fresh_batch(*, database: str = "compile_commands.json", variant: str = "default"):
    return NativeClangIngestor(_fresh_client()).ingest(
        FIXTURE, FIXTURE / database, build_variant=variant
    )


def _summary(batch, name: str):
    names = {symbol.id: symbol.qualified_name for symbol in batch.symbols}
    return next(
        summary
        for summary in batch.function_summaries
        if names[summary.function_symbol_id] == f"interprocedural_fixture::{name}"
    )


def _effects(batch, summary):
    return [effect for effect in batch.summary_effects if effect.summary_id == summary.id]


def test_solver_groups_local_facts_in_one_input_pass() -> None:
    class CountingTuple(tuple):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    summaries = tuple(
        FunctionSummary(
            id=f"summary-{index}",
            function_symbol_id=f"function-{index}",
            graph_id=f"graph-{index}",
            analysis_id=f"analysis-{index}",
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
            translation_unit_id=f"tu-{index}",
            build_configuration_id=f"configuration-{index}",
        )
        for index in range(20)
    )
    effects = CountingTuple()
    origins = CountingTuple()

    _solve_variant(
        summaries,
        effects,
        origins,
        (),
        (),
        (),
        (),
        InterproceduralLimits(),
    )

    assert effects.iterations == 1
    assert origins.iterations == 1


def test_body_variants_consume_only_their_own_callsites() -> None:
    span = SourceSpan(Path("synthetic.cpp"), 1, 1)

    def summary(identity: str, function: str, unit: str, configuration: str) -> FunctionSummary:
        return FunctionSummary(
            id=identity,
            function_symbol_id=function,
            graph_id=f"graph-{identity}",
            analysis_id=f"analysis-{identity}",
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
            translation_unit_id=unit,
            build_configuration_id=configuration,
        )

    caller_a = summary("caller-a", "shared-caller", "tu-a", "config-a")
    caller_b = summary("caller-b", "shared-caller", "tu-b", "config-b")
    callee_a = summary("callee-a", "target-a", "tu-target-a", "config-target-a")
    callee_b = summary("callee-b", "target-b", "tu-target-b", "config-target-b")
    site_a = CallSite(
        "site-a",
        "shared-caller",
        CallDispatchKind.DIRECT,
        span,
        span,
        True,
        translation_unit_id="tu-a",
        build_configuration_id="config-a",
    )
    site_b = CallSite(
        "site-b",
        "shared-caller",
        CallDispatchKind.DIRECT,
        span,
        span,
        True,
        translation_unit_id="tu-b",
        build_configuration_id="config-b",
    )
    target_a = CallTarget(
        "target-edge-a",
        site_a.id,
        callee_a.function_symbol_id,
        CallTargetCertainty.CERTAIN,
        1.0,
        "direct target",
        "direct",
        span,
        translation_unit_id="tu-a",
        build_configuration_id="config-a",
    )
    target_b = CallTarget(
        "target-edge-b",
        site_b.id,
        callee_b.function_symbol_id,
        CallTargetCertainty.CERTAIN,
        1.0,
        "direct target",
        "direct",
        span,
        translation_unit_id="tu-b",
        build_configuration_id="config-b",
    )
    effect_a = SummaryEffect(
        "effect-a",
        callee_a.id,
        SummaryEffectKind.WRITE,
        MemoryLocationKind.GLOBAL,
        DataFlowCertainty.CERTAIN,
        "writes global a",
    )
    effect_b = SummaryEffect(
        "effect-b",
        callee_b.id,
        SummaryEffectKind.WRITE,
        MemoryLocationKind.GLOBAL,
        DataFlowCertainty.CERTAIN,
        "writes global b",
    )

    solution = solve_interprocedural(
        (caller_a, caller_b, callee_a, callee_b),
        (effect_a, effect_b),
        (),
        (),
        (),
        (site_a, site_b),
        (target_a, target_b),
    )
    propagated = {
        summary_id: {
            effect.target_symbol_id
            for effect in solution.effects
            if effect.summary_id == summary_id and not effect.is_local
        }
        for summary_id in (caller_a.id, caller_b.id)
    }

    assert propagated == {caller_a.id: {"target-a"}, caller_b.id: {"target-b"}}


def test_callee_body_resolution_prefers_same_tu_and_downgrades_odr_ambiguity() -> None:
    span = SourceSpan(Path("synthetic.cpp"), 1, 1)

    def summary(identity: str, function: str, unit: str, configuration: str) -> FunctionSummary:
        return FunctionSummary(
            id=identity,
            function_symbol_id=function,
            graph_id=f"graph-{identity}",
            analysis_id=f"analysis-{identity}",
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
            translation_unit_id=unit,
            build_configuration_id=configuration,
        )

    same_tu_caller = summary("same-tu-caller", "same-tu-caller", "tu-a", "config-a")
    ambiguous_caller = summary("ambiguous-caller", "ambiguous-caller", "tu-c", "config-c")
    body_a = summary("body-a", "shared-target", "tu-a", "config-a")
    body_b = summary("body-b", "shared-target", "tu-b", "config-b")
    same_tu_site = CallSite(
        "same-tu-site",
        same_tu_caller.function_symbol_id,
        CallDispatchKind.DIRECT,
        span,
        span,
        True,
        translation_unit_id="tu-a",
        build_configuration_id="config-a",
    )
    ambiguous_site = CallSite(
        "ambiguous-site",
        ambiguous_caller.function_symbol_id,
        CallDispatchKind.DIRECT,
        span,
        span,
        True,
        translation_unit_id="tu-c",
        build_configuration_id="config-c",
    )
    targets = tuple(
        CallTarget(
            f"target-{site.id}",
            site.id,
            "shared-target",
            CallTargetCertainty.CERTAIN,
            1.0,
            "direct target",
            "direct",
            span,
            translation_unit_id=site.translation_unit_id,
            build_configuration_id=site.build_configuration_id,
        )
        for site in (same_tu_site, ambiguous_site)
    )
    effects = (
        SummaryEffect(
            "effect-a",
            body_a.id,
            SummaryEffectKind.WRITE,
            MemoryLocationKind.GLOBAL,
            DataFlowCertainty.CERTAIN,
            "body a",
            source_access_id="source-a",
        ),
        SummaryEffect(
            "effect-b",
            body_b.id,
            SummaryEffectKind.WRITE,
            MemoryLocationKind.GLOBAL,
            DataFlowCertainty.CERTAIN,
            "body b",
            source_access_id="source-b",
        ),
    )

    solution = solve_interprocedural(
        (same_tu_caller, ambiguous_caller, body_a, body_b),
        effects,
        (),
        (),
        (),
        (same_tu_site, ambiguous_site),
        targets,
    )
    summaries = {item.id: item for item in solution.summaries}
    same_tu_effects = [item for item in solution.effects if item.summary_id == same_tu_caller.id]
    ambiguous_effects = [
        item for item in solution.effects if item.summary_id == ambiguous_caller.id
    ]

    assert {item.source_access_id for item in same_tu_effects} == {"source-a"}
    assert summaries[same_tu_caller.id].complete
    assert {item.source_access_id for item in ambiguous_effects} == {"source-a", "source-b"}
    assert all(item.certainty == DataFlowCertainty.POSSIBLE for item in ambiguous_effects)
    assert "multiple_callee_body_variants" in summaries[ambiguous_caller.id].incomplete_reasons


def test_incomplete_argument_binding_makes_the_caller_summary_incomplete() -> None:
    span = SourceSpan(Path("synthetic.cpp"), 1, 1)
    caller = FunctionSummary(
        "caller",
        "caller-function",
        "caller-graph",
        "caller-analysis",
        (),
        (),
        True,
        (),
        True,
        (),
        False,
        0,
        32,
        128,
        1024,
        translation_unit_id="caller-tu",
        build_configuration_id="caller-config",
    )
    callee = FunctionSummary(
        "callee",
        "callee-function",
        "callee-graph",
        "callee-analysis",
        ("reference",),
        ("callee-parameter",),
        True,
        (),
        True,
        (),
        False,
        0,
        32,
        128,
        1024,
        translation_unit_id="callee-tu",
        build_configuration_id="callee-config",
    )
    site = CallSite(
        "site",
        caller.function_symbol_id,
        CallDispatchKind.DIRECT,
        span,
        span,
        True,
        translation_unit_id=caller.translation_unit_id,
        build_configuration_id=caller.build_configuration_id,
    )
    target = CallTarget(
        "target",
        site.id,
        callee.function_symbol_id,
        CallTargetCertainty.CERTAIN,
        1.0,
        "direct target",
        "direct",
        span,
        translation_unit_id=site.translation_unit_id,
        build_configuration_id=site.build_configuration_id,
    )
    binding = CallArgumentBinding(
        "binding",
        caller.id,
        site.id,
        0,
        None,
        MemoryLocationKind.UNKNOWN,
        None,
        (),
        True,
        False,
        "unknown_argument_storage",
        translation_unit_id=site.translation_unit_id,
        build_configuration_id=site.build_configuration_id,
    )
    effect = SummaryEffect(
        "callee-write",
        callee.id,
        SummaryEffectKind.WRITE,
        MemoryLocationKind.PARAMETER,
        DataFlowCertainty.CERTAIN,
        "writes parameter",
        parameter_index=0,
        location_id="callee-parameter",
    )

    solution = solve_interprocedural(
        (caller, callee),
        (effect,),
        (),
        (binding,),
        (),
        (site,),
        (target,),
    )
    solved_caller = next(item for item in solution.summaries if item.id == caller.id)

    assert not solved_caller.complete
    assert "incomplete_call_argument_binding" in solved_caller.incomplete_reasons


def test_real_two_hop_summaries_propagate_arguments_returns_and_side_effects() -> None:
    batch = _batch()
    leaf = _summary(batch, "leaf")
    middle = _summary(batch, "middle")
    top = _summary(batch, "top")

    assert leaf.complete and middle.complete and top.complete
    leaf_effects = _effects(batch, leaf)
    assert {
        (effect.kind, effect.parameter_index)
        for effect in leaf_effects
        if effect.location_kind == MemoryLocationKind.PARAMETER
    } >= {
        (SummaryEffectKind.READ, 0),
        (SummaryEffectKind.WRITE, 1),
    }
    assert any(
        effect.kind == SummaryEffectKind.WRITE
        and effect.parameter_index == 2
        and "*" in effect.access_path
        for effect in leaf_effects
    )
    assert any(
        effect.kind == SummaryEffectKind.WRITE
        and effect.location_kind == MemoryLocationKind.FIELD
        and effect.access_path[-1:] == ("value",)
        for effect in leaf_effects
    )
    assert any(
        effect.kind == SummaryEffectKind.WRITE and effect.location_kind == MemoryLocationKind.GLOBAL
        for effect in _effects(batch, top)
    )
    assert any(not effect.is_local and effect.via_callsite_id for effect in _effects(batch, top))

    flows = batch.interprocedural_flows
    assert {flow.kind for flow in flows} >= {
        InterproceduralFlowKind.ARGUMENT_TO_PARAMETER,
        InterproceduralFlowKind.RETURN_TO_CALLER,
        InterproceduralFlowKind.WRITEBACK,
    }
    assert any(
        flow.caller_summary_id == top.id and flow.callee_summary_id == middle.id for flow in flows
    )
    assert any(
        flow.caller_summary_id == middle.id and flow.callee_summary_id == leaf.id for flow in flows
    )
    assert all(
        flow.callsite_id and flow.translation_unit_id and flow.build_configuration_id
        for flow in flows
    )
    assert all(flow.build_variant == "default" for flow in flows)


def test_recursion_external_calls_and_virtual_targets_never_overclaim() -> None:
    batch = _batch()
    even = _summary(batch, "recursive_even")
    odd = _summary(batch, "recursive_odd")
    external = _summary(batch, "call_external")
    virtual = _summary(batch, "virtual_caller")

    assert even.recursive and odd.recursive
    assert even.complete and odd.complete
    assert 0 < even.iteration_count <= even.max_scc_iterations
    assert not external.complete
    assert "unknown_or_external_call_target" in external.incomplete_reasons
    assert not virtual.complete
    possible = [
        flow for flow in batch.interprocedural_flows if flow.caller_summary_id == virtual.id
    ]
    assert possible
    assert all(flow.target_certainty == CallTargetCertainty.POSSIBLE for flow in possible)
    assert all(flow.certainty.value == "possible" for flow in possible)


def test_scc_and_effect_caps_terminate_deterministically() -> None:
    first = _fresh_batch()
    second = _fresh_batch()
    assert first.function_summaries == second.function_summaries
    assert first.summary_effects == second.summary_effects
    assert first.interprocedural_flows == second.interprocedural_flows
    assert all(
        summary.max_scc_size == InterproceduralLimits().max_scc_size
        and summary.max_summary_effects == InterproceduralLimits().max_summary_effects
        for summary in first.function_summaries
    )

    bounded = solve_interprocedural(
        first.function_summaries,
        tuple(effect for effect in first.summary_effects if effect.is_local),
        tuple(origin for origin in first.summary_return_origins if origin.is_local),
        first.call_argument_bindings,
        first.call_result_bindings,
        first.callsites,
        first.call_targets,
        limits=InterproceduralLimits(
            max_scc_iterations=1,
            max_scc_size=1,
            max_summary_effects=1,
        ),
    )
    names = {symbol.id: symbol.qualified_name for symbol in first.symbols}
    capped = {names[item.function_symbol_id]: item for item in bounded.summaries}
    assert (
        "scc_size_cap_exceeded"
        in capped["interprocedural_fixture::recursive_even"].incomplete_reasons
    )
    assert "scc_iteration_cap_exceeded" in capped["interprocedural_fixture::top"].incomplete_reasons
    assert (
        "summary_effect_cap_exceeded" in capped["interprocedural_fixture::leaf"].incomplete_reasons
    )


def test_build_variants_are_isolated_and_sqlite_refresh_is_targeted(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    default = BuildVariant("default", project / "compile_commands.json")
    alternative = BuildVariant("alternative", project / "compile_commands_alt.json")
    database = tmp_path / "index.db"

    with SQLiteStore(database, project_root=project) as store:
        indexer = ProjectIndexer(NativeClangIngestor(_fresh_client()), store)
        default_result = indexer.index(project, default.compilation_database, build_variant=default)
        alt_result = indexer.index(
            project, alternative.compilation_database, build_variant=alternative
        )
        assert default_result.indexed_function_summaries > 0
        assert alt_result.indexed_function_summaries > 0
        project_id = store._project_id(project)  # noqa: SLF001
        assert {
            row[0]
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT DISTINCT build_variant FROM interprocedural_flows WHERE project_id = ?",
                (project_id,),
            )
        } == {"alternative", "default"}

        before = {
            row["function_symbol_id"]: row["solution_hash"]
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT function_symbol_id, solution_hash FROM function_summaries "
                "WHERE project_id = ? AND build_variant = 'default'",
                (project_id,),
            )
        }
        source = project / "src" / "leaf.cpp"
        source.write_text(source.read_text() + "\n// targeted summary refresh\n", encoding="utf-8")
        changed = indexer.index(project, default.compilation_database, build_variant=default)
        assert changed.indexed_translation_units == 1
        assert changed.invalidated_function_summaries >= 3
        assert changed.invalidated_function_summaries < len(before)
        after = {
            row["function_symbol_id"]: row["solution_hash"]
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT function_symbol_id, solution_hash FROM function_summaries "
                "WHERE project_id = ? AND build_variant = 'default'",
                (project_id,),
            )
        }
        unrelated = next(
            hit.symbol.id
            for hit in store.search_symbols(
                SearchQuery("unrelated", 5),
                build_scope=BuildScope.single("default"),
            )
            if hit.symbol.qualified_name.endswith("::unrelated")
        )
        assert after[unrelated] == before[unrelated]


def test_v5_adapter_rejects_cross_analysis_summary_references() -> None:
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]
    facts = list(_client().analyze(FIXTURE, configuration))
    effect = next(fact for fact in facts if fact.get("fact") == "summary_effect_v1")
    foreign_location = next(
        fact
        for fact in facts
        if fact.get("fact") == "memory_location_v1"
        and fact["analysis_key"] != effect["summary_key"].replace("function-summary:", "data-flow:")
    )
    malformed = dict(effect)
    malformed["location_key"] = foreign_location["key"]
    facts[facts.index(effect)] = malformed

    with pytest.raises(AnalyzerProtocolError, match="inconsistent analysis references"):
        _FactBatchBuilder(FIXTURE.resolve(), configuration).build(facts)


def test_v5_adapter_rejects_malformed_summary_body_and_parameter_references() -> None:
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]
    original = list(_client().analyze(FIXTURE, configuration))
    summaries = [fact for fact in original if fact.get("fact") == "function_summary_v1"]
    assert len(summaries) >= 2

    # Streaming no longer sorts protocol records, so select the relationship
    # under test by its keys instead of relying on incidental output positions.
    parameter_target = next(fact for fact in summaries if fact["parameter_location_keys"])
    foreign_location = next(
        fact
        for fact in original
        if fact.get("fact") == "memory_location_v1"
        and fact["analysis_key"] != parameter_target["analysis_key"]
    )
    foreign_parameter = dict(parameter_target)
    foreign_parameter["parameter_location_keys"] = [
        foreign_location["key"],
        *foreign_parameter["parameter_location_keys"][1:],
    ]
    facts = list(original)
    facts[facts.index(parameter_target)] = foreign_parameter
    with pytest.raises(AnalyzerProtocolError, match="inconsistent analysis references"):
        _FactBatchBuilder(FIXTURE.resolve(), configuration).build(facts)

    body_target = summaries[0]
    body_source = next(fact for fact in summaries if fact["graph_key"] != body_target["graph_key"])
    foreign_body = dict(body_target)
    foreign_body["function_key"] = body_source["function_key"]
    facts = list(original)
    facts[facts.index(body_target)] = foreign_body
    with pytest.raises(AnalyzerProtocolError, match="inconsistent graph references"):
        _FactBatchBuilder(FIXTURE.resolve(), configuration).build(facts)

    effect = next(
        fact
        for fact in original
        if fact.get("fact") == "summary_effect_v1" and "parameter_index" in fact
    )
    invalid_index = dict(effect)
    invalid_index["parameter_index"] = 10_000
    facts = list(original)
    facts[facts.index(effect)] = invalid_index
    with pytest.raises(AnalyzerProtocolError, match="parameter index"):
        _FactBatchBuilder(FIXTURE.resolve(), configuration).build(facts)


def test_v8_migration_is_atomic_and_forces_v7_native_reindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v7.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE translation_units(
            analysis_backend TEXT NOT NULL,
            advanced_facts_complete INTEGER NOT NULL
        );
        INSERT INTO translation_units VALUES ('clang-libtooling', 1);
        PRAGMA user_version=7;
        """
    )
    connection.close()

    import cpp_context_engine.storage.sqlite as sqlite_module

    original = sqlite_module._execute_script

    def fail_after_first(connection: sqlite3.Connection, script: str) -> None:
        original(connection, script.split(";", 1)[0] + ";")
        raise RuntimeError("injected v8 migration failure")

    monkeypatch.setattr(sqlite_module, "_execute_script", fail_after_first)
    with pytest.raises(RuntimeError, match="injected v8"):
        SQLiteStore(database)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
        assert (
            connection.execute("SELECT advanced_facts_complete FROM translation_units").fetchone()[
                0
            ]
            == 1
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='function_summaries'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()

    monkeypatch.setattr(sqlite_module, "_execute_script", original)
    with SQLiteStore(database) as store:
        assert SCHEMA_VERSION == 10
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 10  # noqa: SLF001
        assert (
            store._connection.execute(
                "SELECT advanced_facts_complete FROM translation_units"
            ).fetchone()[0]
            == 0
        )  # noqa: SLF001
