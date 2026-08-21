from __future__ import annotations

import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

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
    DataAccessKind,
    DataFlowCertainty,
    DataFlowRelation,
    MemoryLocationKind,
)
from cpp_context_engine.storage import SQLiteStore
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION

FIXTURE = Path(__file__).parent / "fixtures" / "dataflow_project"


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


def _names(batch) -> dict[str, str]:
    return {symbol.id: symbol.qualified_name for symbol in batch.symbols}


def _analysis(batch, function_name: str):
    names = _names(batch)
    graph = next(
        graph
        for graph in batch.cfg_graphs
        if names[graph.function_symbol_id] == f"dataflow_fixture::{function_name}"
    )
    return next(item for item in batch.data_flow_analyses if item.graph_id == graph.id)


def _owned_sites(batch, function_name: str):
    names = _names(batch)
    qualified = f"dataflow_fixture::{function_name}"
    return [site for site in batch.callsites if names[site.owner_symbol_id] == qualified]


def _targets(batch, site):
    return [target for target in batch.call_targets if target.callsite_id == site.id]


def test_real_fixed_point_records_definitions_uses_joins_loops_and_aliases() -> None:
    batch = _batch()
    analysis = _analysis(batch, "definitions_and_join")
    locations = [item for item in batch.memory_locations if item.analysis_id == analysis.id]
    accesses = [item for item in batch.data_accesses if item.analysis_id == analysis.id]
    evidence = [item for item in batch.data_flow_evidence if item.analysis_id == analysis.id]

    assert analysis.complete
    assert 1 < analysis.iteration_count <= analysis.max_iterations == 64
    assert analysis.max_alias_targets == 64
    assert analysis.max_access_path_depth == 8
    assert analysis.max_locations == 4096
    assert {location.kind for location in locations} >= {
        MemoryLocationKind.PARAMETER,
        MemoryLocationKind.LOCAL,
        MemoryLocationKind.RETURN,
        MemoryLocationKind.UNKNOWN,
    }
    assert {access.kind for access in accesses} >= {
        DataAccessKind.PARAMETER_DEFINITION,
        DataAccessKind.INITIALIZATION,
        DataAccessKind.ASSIGNMENT,
        DataAccessKind.COMPOUND_ASSIGNMENT,
        DataAccessKind.INCREMENT,
        DataAccessKind.READ,
        DataAccessKind.RETURN_VALUE,
        DataAccessKind.CONDITION,
    }
    relation_counts = Counter(item.relation for item in evidence)
    assert relation_counts[DataFlowRelation.REACHING_DEFINITION] > 0
    assert relation_counts[DataFlowRelation.OVERWRITES] > 0
    assert any(
        item.relation == DataFlowRelation.REACHING_DEFINITION
        and item.certainty == DataFlowCertainty.POSSIBLE
        for item in evidence
    )
    assert all("dead" not in item.reason.lower() for item in evidence)

    alias_analysis = _analysis(batch, "aliases_and_fields")
    alias_locations = [
        item for item in batch.memory_locations if item.analysis_id == alias_analysis.id
    ]
    alias_evidence = [
        item for item in batch.data_flow_evidence if item.analysis_id == alias_analysis.id
    ]
    assert {item.kind for item in alias_locations} >= {
        MemoryLocationKind.FIELD,
        MemoryLocationKind.DEREFERENCE,
    }
    assert {item.relation for item in alias_evidence} >= {
        DataFlowRelation.MUST_ALIAS,
        DataFlowRelation.MAY_ALIAS,
    }


def test_function_and_member_pointer_targets_preserve_certainty_and_completeness() -> None:
    batch = _batch()
    names = _names(batch)

    singleton = _owned_sites(batch, "singleton_pointer")[0]
    assert singleton.target_set_complete and not singleton.unresolved_reason
    singleton_targets = _targets(batch, singleton)
    assert [
        (names[item.target_symbol_id], item.certainty, item.derivation)
        for item in singleton_targets
    ] == [
        (
            "dataflow_fixture::target_a",
            CallTargetCertainty.CERTAIN,
            "intraprocedural_singleton_points_to",
        )
    ]

    conditional = _owned_sites(batch, "conditional_pointer")[0]
    assert conditional.target_set_complete
    assert {
        (names[item.target_symbol_id], item.certainty) for item in _targets(batch, conditional)
    } == {
        ("dataflow_fixture::target_a", CallTargetCertainty.POSSIBLE),
        ("dataflow_fixture::target_b", CallTargetCertainty.POSSIBLE),
    }
    assert all("probability" in item.confidence_reason for item in _targets(batch, conditional))

    copied_access = next(
        access
        for access in batch.data_accesses
        if access.analysis_id == _analysis(batch, "singleton_pointer").id
        and access.pointee_symbol_ids
        and any(
            names[target] == "dataflow_fixture::target_a" for target in access.pointee_symbol_ids
        )
    )
    assert copied_access.points_to_complete

    member = _owned_sites(batch, "singleton_member")[0]
    member_target = _targets(batch, member)[0]
    assert member.target_set_complete
    assert names[member_target.target_symbol_id] == "dataflow_fixture::Handler::first"
    assert member_target.certainty == CallTargetCertainty.CERTAIN

    conditional_member = _owned_sites(batch, "conditional_member")[0]
    assert conditional_member.target_set_complete
    assert {names[item.target_symbol_id] for item in _targets(batch, conditional_member)} == {
        "dataflow_fixture::Handler::first",
        "dataflow_fixture::Handler::second",
    }
    assert all(
        item.certainty == CallTargetCertainty.POSSIBLE
        for item in _targets(batch, conditional_member)
    )

    null = _owned_sites(batch, "null_pointer")[0]
    assert null.target_set_complete and not _targets(batch, null)
    unknown = _owned_sites(batch, "unknown_pointer")[0]
    assert not unknown.target_set_complete
    assert unknown.unresolved_reason == "points_to_set_incomplete"
    assert not _targets(batch, unknown)


def test_conservative_cases_and_budget_metadata_are_explicit_and_deterministic() -> None:
    first = _batch()
    second = _batch()
    for attribute in (
        "data_flow_analyses",
        "memory_locations",
        "data_accesses",
        "data_flow_evidence",
    ):
        assert getattr(first, attribute) == getattr(second, attribute)

    conservative = _analysis(first, "conservative_cases")
    assert not conservative.complete
    assert set(conservative.incomplete_reasons) >= {
        "external_parameter_points_to",
        "inline_assembly",
        "pointer_arithmetic",
        "reinterpret_cast",
        "volatile_access",
    }
    deep = _analysis(first, "deep_access_path")
    assert not deep.complete
    assert "access_path_cap_exceeded" in deep.incomplete_reasons
    union = _analysis(first, "union_access")
    assert not union.complete
    assert "union_storage" in union.incomplete_reasons
    assert tuple(sorted(conservative.incomplete_reasons)) == conservative.incomplete_reasons


def test_data_flow_adapter_rejects_cross_graph_references() -> None:
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]
    facts = list(_client().analyze(FIXTURE, configuration))
    accesses = [fact for fact in facts if fact.get("fact") == "data_access_v1"]
    graph_keys = {fact["key"] for fact in facts if fact.get("fact") == "cfg_graph_v1"}
    malformed = dict(accesses[0])
    malformed["graph_key"] = next(key for key in graph_keys if key != malformed["graph_key"])
    facts[facts.index(accesses[0])] = malformed

    with pytest.raises(AnalyzerProtocolError, match="inconsistent graph references"):
        _FactBatchBuilder(FIXTURE.resolve(), configuration).build(facts)


def test_sqlite_v7_round_trip_multi_build_and_incremental_cleanup(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    default = BuildVariant("default", project / "compile_commands.json")
    alternative = BuildVariant("alternative", project / "compile_commands_alt.json")
    database = tmp_path / "index.db"

    with SQLiteStore(database, project_root=project) as store:
        indexer = ProjectIndexer(NativeClangIngestor(_client()), store)
        default_result = indexer.index(project, default.compilation_database, build_variant=default)
        alternative_result = indexer.index(
            project, alternative.compilation_database, build_variant=alternative
        )
        assert default_result.indexed_data_flow_analyses > 0
        assert default_result.indexed_memory_locations > 0
        assert default_result.indexed_data_accesses > 0
        assert default_result.indexed_data_flow_evidence > 0
        assert alternative_result.indexed_data_flow_analyses > 0
        project_id = store._project_id(project)  # noqa: SLF001
        counts = {
            row[0]: row[1]
            for row in store._connection.execute(  # noqa: SLF001
                """
                SELECT build_variant, count(*) FROM data_flow_analyses
                WHERE project_id = ? GROUP BY build_variant ORDER BY build_variant
                """,
                (project_id,),
            )
        }
        assert set(counts) == {"alternative", "default"}
        assert all(count > 0 for count in counts.values())
        assert store.remove_build_variant("alternative")
        assert not store.translation_unit_states(
            project, build_scope=BuildScope.single("alternative")
        )
        assert (
            store._connection.execute(  # noqa: SLF001
                "SELECT count(*) FROM data_flow_analyses WHERE project_id = ? "
                "AND build_variant = 'alternative'",
                (project_id,),
            ).fetchone()[0]
            == 0
        )

        source = project / "src" / "dataflow.cpp"
        source.write_text(source.read_text() + "\n// incremental change\n", encoding="utf-8")
        changed = indexer.index(project, default.compilation_database, build_variant=default)
        assert changed.indexed_translation_units == 1
        assert (
            store._connection.execute(  # noqa: SLF001
                "SELECT count(DISTINCT translation_unit_id) FROM data_flow_analyses "
                "WHERE project_id = ? AND build_variant = 'default'",
                (project_id,),
            ).fetchone()[0]
            == 1
        )


def test_v7_migration_is_atomic_and_forces_old_native_reindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v6.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE translation_units(
            analysis_backend TEXT NOT NULL,
            advanced_facts_complete INTEGER NOT NULL
        );
        INSERT INTO translation_units VALUES ('clang-libtooling', 1);
        PRAGMA user_version=6;
        """
    )
    connection.close()

    import cpp_context_engine.storage.sqlite as sqlite_module

    original = sqlite_module._execute_script

    def fail_after_first(connection: sqlite3.Connection, script: str) -> None:
        original(connection, script.split(";", 1)[0] + ";")
        raise RuntimeError("injected v7 migration failure")

    monkeypatch.setattr(sqlite_module, "_execute_script", fail_after_first)
    with pytest.raises(RuntimeError, match="injected v7"):
        SQLiteStore(database)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='data_flow_analyses'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute("SELECT advanced_facts_complete FROM translation_units").fetchone()[
                0
            ]
            == 1
        )
    finally:
        connection.close()

    monkeypatch.setattr(sqlite_module, "_execute_script", original)
    with SQLiteStore(database) as store:
        assert SCHEMA_VERSION == 7
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 7  # noqa: SLF001
        assert (
            store._connection.execute(  # noqa: SLF001
                "SELECT advanced_facts_complete FROM translation_units"
            ).fetchone()[0]
            == 0
        )
