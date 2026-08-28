from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary
from native_cache import cached_native_client, fresh_native_client

from cpp_context_engine.ingestion import NativeClangIngestor
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.models import (
    BuildScope,
    BuildVariant,
    CallDispatchKind,
    CallTargetCertainty,
    GraphRelation,
)
from cpp_context_engine.storage import SQLiteStore
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION

FIXTURE = Path(__file__).parent / "fixtures" / "dispatch_project"
pytestmark = pytest.mark.native


def _ingestor() -> NativeClangIngestor:
    return NativeClangIngestor(  # type: ignore[arg-type]
        cached_native_client(analyzer_binary(), timeout_seconds=90)
    )


def _fresh_ingestor() -> NativeClangIngestor:
    return NativeClangIngestor(fresh_native_client(analyzer_binary(), timeout_seconds=90))


def _batch(*, database: str = "compile_commands.json", variant: str = "default"):
    return _ingestor().ingest(FIXTURE, FIXTURE / database, build_variant=variant)


def _fresh_batch(*, database: str = "compile_commands.json", variant: str = "default"):
    return _fresh_ingestor().ingest(FIXTURE, FIXTURE / database, build_variant=variant)


def _symbol_names(batch) -> dict[str, str]:
    return {symbol.id: symbol.qualified_name for symbol in batch.symbols}


def _owned_sites(batch, owner: str):
    names = _symbol_names(batch)
    return [site for site in batch.callsites if names[site.owner_symbol_id] == owner]


def _targets(batch, callsite):
    return [target for target in batch.call_targets if target.callsite_id == callsite.id]


def test_virtual_candidates_final_dispatch_and_multiple_inheritance_are_honest() -> None:
    batch = _batch()
    names = _symbol_names(batch)
    site = _owned_sites(batch, "dispatch_fixture::virtual_call")[0]

    assert site.dispatch_kind == CallDispatchKind.VIRTUAL
    assert not site.target_set_complete
    assert site.unresolved_reason == "open_world_external_overrides_possible"
    targets = {names[target.target_symbol_id]: target for target in _targets(batch, site)}
    assert set(targets) == {
        "dispatch_fixture::Root::run",
        "dispatch_fixture::OverrideA::run",
        "dispatch_fixture::OverrideB::run",
        "dispatch_fixture::Multi::run",
        "dispatch_fixture::FinalLeaf::run",
    }
    assert targets["dispatch_fixture::Root::run"].derivation == "static_virtual_candidate"
    assert targets["dispatch_fixture::Root::run"].confidence == 0.75
    assert all(target.certainty == CallTargetCertainty.POSSIBLE for target in targets.values())
    assert all("probability" in target.confidence_reason for target in targets.values())

    final = next(
        site
        for site in _owned_sites(batch, "dispatch_fixture::final_call")
        if site.dispatch_kind == CallDispatchKind.DEVIRTUALIZED
    )
    assert final.target_set_complete
    target = _targets(batch, final)[0]
    assert names[target.target_symbol_id] == "dispatch_fixture::FinalLeaf::run"
    assert (target.certainty, target.confidence, target.derivation) == (
        CallTargetCertainty.CERTAIN,
        1.0,
        "final_dispatch",
    )


def test_callable_template_macro_duplicate_and_dependent_forms() -> None:
    batch = _batch()
    names = _symbol_names(batch)
    callable_sites = _owned_sites(batch, "dispatch_fixture::callable_forms")
    assert {site.dispatch_kind for site in callable_sites} >= {
        CallDispatchKind.LAMBDA,
        CallDispatchKind.GENERIC_LAMBDA,
        CallDispatchKind.FUNCTOR,
    }
    for site in callable_sites:
        if site.dispatch_kind in {
            CallDispatchKind.LAMBDA,
            CallDispatchKind.GENERIC_LAMBDA,
            CallDispatchKind.FUNCTOR,
        }:
            assert site.target_set_complete
            assert _targets(batch, site)[0].certainty == CallTargetCertainty.CERTAIN

    dependent = _owned_sites(batch, "dispatch_fixture::dependent_uninstantiated")[0]
    assert dependent.dispatch_kind == CallDispatchKind.DEPENDENT_TEMPLATE
    assert not dependent.target_set_complete
    assert dependent.unresolved_reason == "dependent_or_uninstantiated_template"
    assert not _targets(batch, dependent)

    repeated = [
        site
        for site in _owned_sites(batch, "dispatch_fixture::repeated_direct_calls")
        if site.callee_text == "direct_target(1)"
    ]
    assert len(repeated) == 2
    assert len({site.id for site in repeated}) == 2
    assert len({site.expansion_span.start_line for site in repeated}) == 2
    assert {names[_targets(batch, site)[0].target_symbol_id] for site in repeated} == {
        "dispatch_fixture::direct_target"
    }

    macro = _owned_sites(batch, "dispatch_fixture::macro_generated_call")[0]
    assert [frame.name for frame in macro.expansion_stack] == [
        "GENERATED_CALL",
        "FORWARD_GENERATED",
    ]
    assert macro.spelling_span != macro.expansion_span
    assert macro.expansion_stack[0].spelling_span != macro.expansion_stack[0].expansion_span
    assert any(
        edge.relation == GraphRelation.GENERATED_BY_MACRO
        and edge.source_id == macro.owner_symbol_id
        and names[edge.target_id] == "GENERATED_CALL"
        for edge in batch.edges
    )

    template_edges = [
        edge
        for edge in batch.edges
        if edge.relation in {GraphRelation.INSTANTIATES, GraphRelation.SPECIALIZES}
    ]
    assert {edge.relation for edge in template_edges} == {
        GraphRelation.INSTANTIATES,
        GraphRelation.SPECIALIZES,
    }
    assert all(edge.source_id != edge.target_id for edge in template_edges)
    template_symbols = [
        symbol
        for symbol in batch.symbols
        if symbol.qualified_name == "dispatch_fixture::transform"
        and symbol.metadata.get("template_kind") != "primary"
    ]
    assert {symbol.metadata["template_kind"] for symbol in template_symbols} >= {
        "explicit_specialization",
        "implicit_instantiation",
        "explicit_instantiation_definition",
    }
    assert any(
        symbol.metadata["template_kind"] == "implicit_instantiation"
        and "point_of_instantiation" in symbol.metadata
        for symbol in template_symbols
    )


def test_call_ids_provenance_storage_bounds_and_incremental_cleanup(tmp_path: Path) -> None:
    first = _fresh_batch()
    second = _fresh_batch()
    assert [site.id for site in first.callsites] == [site.id for site in second.callsites]
    assert [target.id for target in first.call_targets] == [
        target.id for target in second.call_targets
    ]
    assert all(
        site.translation_unit_id and site.build_configuration_id and site.build_variant == "default"
        for site in first.callsites
    )
    assert all(
        target.translation_unit_id
        and target.build_configuration_id
        and target.evidence_span == first_site.expansion_span
        for first_site in first.callsites
        for target in _targets(first, first_site)
    )

    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    default = BuildVariant("default", project / "compile_commands.json")
    extra = BuildVariant("extra", project / "compile_commands_extra.json")
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=project) as store:
        indexer = ProjectIndexer(_fresh_ingestor(), store)
        indexer.index(project, default.compilation_database, build_variant=default)
        indexer.index(project, extra.compilation_database, build_variant=extra)
        default_sites = store.callsites(build_scope=BuildScope.single("default"), limit=1)
        assert len(default_sites.items) == 1 and default_sites.truncated
        with pytest.raises(ValueError, match="call page limit"):
            store.callsites(limit=0)
        extra_sites = store.callsites(build_scope=BuildScope.single("extra"), limit=100)
        assert extra_sites.items
        virtual = next(
            site for site in extra_sites.items if site.dispatch_kind == CallDispatchKind.VIRTUAL
        )
        extra_target_names = {
            store.get_symbol(
                target.target_symbol_id, build_scope=BuildScope.single("extra")
            ).qualified_name
            for target in store.call_targets(
                virtual.id, build_scope=BuildScope.single("extra")
            ).items
        }
        assert "dispatch_fixture::BuildOnlyOverride::run" in extra_target_names
        assert store.remove_build_variant("extra")
        assert not store.callsites(build_scope=BuildScope.single("extra"), limit=100).items
        assert store.callsites(build_scope=BuildScope.single("default"), limit=100).items


def test_incremental_override_change_refreshes_unchanged_callsite_targets(tmp_path: Path) -> None:
    project = tmp_path / "project"
    include = project / "include"
    source = project / "src"
    include.mkdir(parents=True)
    source.mkdir()
    (include / "base.hpp").write_text(
        "struct Root { virtual int run() const { return 0; } };\n",
        encoding="utf-8",
    )
    (source / "caller.cpp").write_text(
        '#include "base.hpp"\nint invoke(const Root& root) { return root.run(); }\n',
        encoding="utf-8",
    )
    overrides = source / "overrides.cpp"
    overrides.write_text(
        '#include "base.hpp"\nstruct First : Root { int run() const override { return 1; } };\n',
        encoding="utf-8",
    )
    compile_commands = project / "compile_commands.json"
    compile_commands.write_text(
        json.dumps(
            [
                {
                    "directory": str(project),
                    "file": str(path),
                    "arguments": [
                        "clang++",
                        "-std=c++20",
                        f"-I{include}",
                        "-c",
                        str(path),
                    ],
                }
                for path in (source / "caller.cpp", overrides)
            ]
        ),
        encoding="utf-8",
    )
    database = tmp_path / "index.db"
    variant = BuildVariant("default", compile_commands)
    with SQLiteStore(database, project_root=project) as store:
        indexer = ProjectIndexer(_fresh_ingestor(), store)
        first = indexer.index(project, compile_commands, build_variant=variant)
        assert first.indexed_translation_units == 2

        overrides.write_text(
            '#include "base.hpp"\n'
            "struct First : Root { int run() const override { return 1; } };\n"
            "struct Second : Root { int run() const override { return 2; } };\n",
            encoding="utf-8",
        )
        second = indexer.index(project, compile_commands, build_variant=variant)
        assert second.indexed_translation_units == 1
        assert second.skipped_translation_units == 1

        callsite = next(
            site
            for site in store.callsites(build_scope=BuildScope.single("default")).items
            if site.dispatch_kind == CallDispatchKind.VIRTUAL
        )
        target_names = {
            store.get_symbol(
                target.target_symbol_id,
                build_scope=BuildScope.single("default"),
            ).qualified_name
            for target in store.call_targets(
                callsite.id,
                build_scope=BuildScope.single("default"),
            ).items
        }
        assert target_names == {"Root::run", "First::run", "Second::run"}


def test_v6_migration_is_atomic_and_marks_old_native_rows_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v5.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE translation_units(
            analysis_backend TEXT NOT NULL,
            advanced_facts_complete INTEGER NOT NULL
        );
        INSERT INTO translation_units VALUES ('clang-libtooling', 1);
        PRAGMA user_version=5;
        """
    )
    connection.close()

    import cpp_context_engine.storage.sqlite as sqlite_module

    original = sqlite_module._execute_script

    def fail_after_first(connection: sqlite3.Connection, script: str) -> None:
        first = script.split(";", 1)[0] + ";"
        original(connection, first)
        raise RuntimeError("injected v6 migration failure")

    monkeypatch.setattr(sqlite_module, "_execute_script", fail_after_first)
    with pytest.raises(RuntimeError, match="injected v6"):
        SQLiteStore(database)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        assert (
            connection.execute("SELECT 1 FROM sqlite_master WHERE name='callsites'").fetchone()
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
        assert SCHEMA_VERSION == 12
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 12  # noqa: SLF001
        assert (
            store._connection.execute(  # noqa: SLF001
                "SELECT advanced_facts_complete FROM translation_units"
            ).fetchone()[0]
            == 0
        )
