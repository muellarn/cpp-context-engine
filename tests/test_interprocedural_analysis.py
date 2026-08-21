from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from cpp_context_engine.analysis.interprocedural import (
    InterproceduralLimits,
    solve_interprocedural,
)
from cpp_context_engine.ingestion import (
    AnalyzerProtocolError,
    NativeAnalyzerClient,
    NativeClangIngestor,
)
from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.ingestion.native import _FactBatchBuilder
from cpp_context_engine.models import (
    BuildScope,
    BuildVariant,
    CallTargetCertainty,
    InterproceduralFlowKind,
    MemoryLocationKind,
    SearchQuery,
    SummaryEffectKind,
)
from cpp_context_engine.storage import SQLiteStore
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION

FIXTURE = Path(__file__).parent / "fixtures" / "interprocedural_project"


def _binary() -> Path:
    candidate = (
        Path(__file__).parents[1] / "build" / "clang-analyzer" / ("cpp-context-clang-analyzer")
    )
    if not candidate.is_file():
        pytest.skip("Clang analyzer companion has not been built")
    return candidate.resolve()


def _client() -> NativeAnalyzerClient:
    return NativeAnalyzerClient(_binary(), timeout_seconds=90)


def _batch(*, database: str = "compile_commands.json", variant: str = "default"):
    return NativeClangIngestor(_client()).ingest(FIXTURE, FIXTURE / database, build_variant=variant)


def _summary(batch, name: str):
    names = {symbol.id: symbol.qualified_name for symbol in batch.symbols}
    return next(
        summary
        for summary in batch.function_summaries
        if names[summary.function_symbol_id] == f"interprocedural_fixture::{name}"
    )


def _effects(batch, summary):
    return [effect for effect in batch.summary_effects if effect.summary_id == summary.id]


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
    first = _batch()
    second = _batch()
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
        indexer = ProjectIndexer(NativeClangIngestor(_client()), store)
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
        assert SCHEMA_VERSION == 8
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 8  # noqa: SLF001
        assert (
            store._connection.execute(
                "SELECT advanced_facts_complete FROM translation_units"
            ).fetchone()[0]
            == 0
        )  # noqa: SLF001
