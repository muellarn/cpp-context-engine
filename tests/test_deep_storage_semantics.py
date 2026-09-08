from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.analysis.interprocedural import solve_interprocedural
from cpp_context_engine.api.analysis import AnalysisQueryService, CfgRequest, FlowRequest
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion.deep import (
    DeepCancellation,
    DeepMaterializer,
    DeepRequestControl,
    MaterializeDeepRequest,
    _digest,
    _file_digest,
)
from cpp_context_engine.ingestion.native import NativeClangIngestor
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    BuildScope,
    BuildVariant,
    CallDispatchKind,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    CfgBlock,
    CfgBlockRole,
    CfgGraph,
    CodeSymbol,
    DataFlowAnalysis,
    DataFlowCertainty,
    FunctionSummary,
    GraphEdge,
    GraphRelation,
    IndexProfile,
    MemoryLocationKind,
    OccurrenceKind,
    SourceSpan,
    SummaryEffect,
    SummaryEffectKind,
    SymbolKind,
    SymbolOccurrence,
    TranslationUnit,
)
from cpp_context_engine.storage.sqlite import SQLiteStore


def _two_tu_summary_batch(root: Path) -> tuple[IngestionBatch, IngestionBatch]:
    caller_path = root / "caller.cpp"
    callee_path = root / "callee.cpp"
    caller_path.write_text("void callee(); void caller() { callee(); }\n", encoding="utf-8")
    callee_path.write_text("int state; void callee() { state = 1; }\n", encoding="utf-8")
    configurations = tuple(
        BuildConfiguration(
            id=f"config-{name}",
            source_path=path,
            directory=root,
            arguments=("c++", path.name),
            command_hash=f"command-{name}",
        )
        for name, path in (("caller", caller_path), ("callee", callee_path))
    )
    deep_units = tuple(
        TranslationUnit(
            id=f"tu-{name}",
            build_configuration_id=f"config-{name}",
            source_path=path,
            content_hash=f"content-{name}",
            dependencies=((path, f"content-{name}"),),
            analysis_backend="clang-libtooling",
            advanced_facts_complete=True,
            index_profile=IndexProfile.FULL,
            navigation_facts_complete=True,
            cfg_facts_complete=True,
            data_flow_facts_complete=True,
            summary_facts_complete=True,
        )
        for name, path in (("caller", caller_path), ("callee", callee_path))
    )
    symbols = tuple(
        CodeSymbol(
            id=name,
            qualified_name=name,
            kind=SymbolKind.FUNCTION,
            span=SourceSpan(path, 1, 1),
            signature=f"void {name}()",
            source_text=f"void {name}()",
            source_hash=f"source-{name}",
            build_configuration_id=f"config-{name}",
            translation_unit_id=f"tu-{name}",
            metadata={"is_definition": True},
        )
        for name, path in (("caller", caller_path), ("callee", callee_path))
    )
    occurrences = tuple(
        SymbolOccurrence(
            id=f"occurrence-{symbol.id}",
            symbol_id=symbol.id,
            span=symbol.span,
            kind=OccurrenceKind.DEFINITION,
            translation_unit_id=symbol.translation_unit_id,
            build_configuration_id=symbol.build_configuration_id,
        )
        for symbol in symbols
    )
    span = SourceSpan(caller_path, 1, 1)
    site = CallSite(
        id="call-callee",
        owner_symbol_id="caller",
        dispatch_kind=CallDispatchKind.DIRECT,
        spelling_span=span,
        expansion_span=span,
        target_set_complete=True,
        static_target_symbol_id="callee",
        callee_text="callee",
        translation_unit_id="tu-caller",
        build_configuration_id="config-caller",
    )
    target = CallTarget(
        id="target-callee",
        callsite_id=site.id,
        target_symbol_id="callee",
        certainty=CallTargetCertainty.CERTAIN,
        confidence=1.0,
        confidence_reason="direct target",
        derivation="direct",
        evidence_span=span,
        translation_unit_id="tu-caller",
        build_configuration_id="config-caller",
    )
    graphs = tuple(
        CfgGraph(
            id=f"graph-{name}",
            function_symbol_id=name,
            entry_block_id=f"block-{name}",
            normal_exit_block_id=f"block-{name}",
            translation_unit_id=f"tu-{name}",
            build_configuration_id=f"config-{name}",
        )
        for name in ("caller", "callee")
    )
    blocks = tuple(
        CfgBlock(
            id=f"block-{name}",
            graph_id=f"graph-{name}",
            index=0,
            role=CfgBlockRole.ENTRY,
            reachable=True,
            translation_unit_id=f"tu-{name}",
            build_configuration_id=f"config-{name}",
        )
        for name in ("caller", "callee")
    )
    analyses = tuple(
        DataFlowAnalysis(
            id=f"analysis-{name}",
            graph_id=f"graph-{name}",
            complete=True,
            incomplete_reasons=(),
            iteration_count=1,
            max_iterations=16,
            max_alias_targets=16,
            max_access_path_depth=8,
            max_locations=128,
            translation_unit_id=f"tu-{name}",
            build_configuration_id=f"config-{name}",
        )
        for name in ("caller", "callee")
    )
    summaries = tuple(
        FunctionSummary(
            id=f"summary-{name}",
            function_symbol_id=name,
            graph_id=f"graph-{name}",
            analysis_id=f"analysis-{name}",
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
            translation_unit_id=f"tu-{name}",
            build_configuration_id=f"config-{name}",
        )
        for name in ("caller", "callee")
    )
    local_effect = SummaryEffect(
        id="callee-global-write",
        summary_id="summary-callee",
        kind=SummaryEffectKind.WRITE,
        location_kind=MemoryLocationKind.GLOBAL,
        certainty=DataFlowCertainty.CERTAIN,
        reason="writes global state",
        translation_unit_id="tu-callee",
        build_configuration_id="config-callee",
    )
    solution = solve_interprocedural(
        summaries,
        (local_effect,),
        (),
        (),
        (),
        (site,),
        (target,),
    )
    deep = IngestionBatch(
        configurations,
        deep_units,
        symbols,
        occurrences,
        (),
        cfg_graphs=graphs,
        cfg_blocks=blocks,
        callsites=(site,),
        call_targets=(target,),
        data_flow_analyses=analyses,
        function_summaries=solution.summaries,
        summary_effects=solution.effects,
        summary_return_origins=solution.return_origins,
        interprocedural_flows=solution.flows,
    )
    navigation = replace(
        deep,
        translation_units=tuple(
            replace(
                unit,
                advanced_facts_complete=False,
                index_profile=IndexProfile.NAVIGATION,
                cfg_facts_complete=False,
                data_flow_facts_complete=False,
                summary_facts_complete=False,
            )
            for unit in deep_units
        ),
    )
    return navigation, deep


def _records_for_unit(records: tuple, unit_id: str) -> tuple:
    return tuple(item for item in records if item.translation_unit_id == unit_id)


def _batch_for_unit(batch: IngestionBatch, unit_id: str) -> IngestionBatch:
    unit = next(item for item in batch.translation_units if item.id == unit_id)
    return IngestionBatch(
        build_configurations=tuple(
            item for item in batch.build_configurations if item.id == unit.build_configuration_id
        ),
        translation_units=(unit,),
        symbols=_records_for_unit(batch.symbols, unit_id),
        occurrences=_records_for_unit(batch.occurrences, unit_id),
        edges=_records_for_unit(batch.edges, unit_id),
        cfg_graphs=_records_for_unit(batch.cfg_graphs, unit_id),
        cfg_blocks=_records_for_unit(batch.cfg_blocks, unit_id),
        cfg_elements=_records_for_unit(batch.cfg_elements, unit_id),
        cfg_edges=_records_for_unit(batch.cfg_edges, unit_id),
        callsites=_records_for_unit(batch.callsites, unit_id),
        call_targets=_records_for_unit(batch.call_targets, unit_id),
        data_flow_analyses=_records_for_unit(batch.data_flow_analyses, unit_id),
        memory_locations=_records_for_unit(batch.memory_locations, unit_id),
        data_accesses=_records_for_unit(batch.data_accesses, unit_id),
        data_flow_evidence=_records_for_unit(batch.data_flow_evidence, unit_id),
        function_summaries=_records_for_unit(batch.function_summaries, unit_id),
        summary_effects=_records_for_unit(batch.summary_effects, unit_id),
        summary_return_origins=_records_for_unit(batch.summary_return_origins, unit_id),
        call_argument_bindings=_records_for_unit(batch.call_argument_bindings, unit_id),
        call_result_bindings=_records_for_unit(batch.call_result_bindings, unit_id),
        interprocedural_flows=_records_for_unit(batch.interprocedural_flows, unit_id),
    )


def test_cross_tu_override_change_invalidates_unchanged_virtual_caller_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    navigation, deep = _two_tu_summary_batch(root)
    virtual_site = replace(
        navigation.callsites[0],
        dispatch_kind=CallDispatchKind.VIRTUAL,
        static_target_symbol_id="caller",
        target_set_complete=False,
        unresolved_reason="open_world_external_overrides_possible",
    )
    static_target = replace(
        navigation.call_targets[0],
        target_symbol_id="caller",
        derivation="static_virtual_candidate",
        certainty=CallTargetCertainty.POSSIBLE,
        confidence=0.75,
    )
    navigation = replace(
        navigation,
        callsites=(virtual_site,),
        call_targets=(static_target,),
    )
    caller_deep = _batch_for_unit(
        replace(deep, callsites=(virtual_site,), call_targets=(static_target,)),
        "tu-caller",
    )
    reverse_configuration = replace(
        navigation.build_configurations[0],
        id="config-reverse",
        command_hash="command-reverse",
    )
    reverse_navigation_unit = replace(
        navigation.translation_units[0],
        id="tu-reverse",
        build_configuration_id="config-reverse",
        content_hash="content-reverse",
    )
    reverse_symbol = replace(
        navigation.symbols[0],
        id="reverse",
        qualified_name="reverse",
        source_hash="source-reverse",
        translation_unit_id="tu-reverse",
        build_configuration_id="config-reverse",
        variant_id="variant-reverse",
    )
    reverse_occurrence = replace(
        navigation.occurrences[0],
        id="occurrence-reverse",
        symbol_id="reverse",
        translation_unit_id="tu-reverse",
        build_configuration_id="config-reverse",
    )
    reverse_site = replace(
        deep.callsites[0],
        id="call-reverse-caller",
        owner_symbol_id="reverse",
        static_target_symbol_id="caller",
        translation_unit_id="tu-reverse",
        build_configuration_id="config-reverse",
    )
    reverse_target = replace(
        deep.call_targets[0],
        id="target-reverse-caller",
        callsite_id="call-reverse-caller",
        target_symbol_id="caller",
        translation_unit_id="tu-reverse",
        build_configuration_id="config-reverse",
    )
    reverse_graph = replace(
        deep.cfg_graphs[0],
        id="graph-reverse",
        function_symbol_id="reverse",
        entry_block_id="block-reverse",
        normal_exit_block_id="block-reverse",
        translation_unit_id="tu-reverse",
        build_configuration_id="config-reverse",
    )
    reverse_block = replace(
        deep.cfg_blocks[0],
        id="block-reverse",
        graph_id="graph-reverse",
        translation_unit_id="tu-reverse",
        build_configuration_id="config-reverse",
    )
    reverse_analysis = replace(
        deep.data_flow_analyses[0],
        id="analysis-reverse",
        graph_id="graph-reverse",
        translation_unit_id="tu-reverse",
        build_configuration_id="config-reverse",
    )
    reverse_summary = replace(
        deep.function_summaries[0],
        id="summary-reverse",
        function_symbol_id="reverse",
        graph_id="graph-reverse",
        analysis_id="analysis-reverse",
        translation_unit_id="tu-reverse",
        build_configuration_id="config-reverse",
    )
    reverse_navigation = IngestionBatch(
        build_configurations=(reverse_configuration,),
        translation_units=(reverse_navigation_unit,),
        symbols=(reverse_symbol,),
        occurrences=(reverse_occurrence,),
        edges=(),
        callsites=(reverse_site,),
        call_targets=(reverse_target,),
    )
    reverse_deep = replace(
        reverse_navigation,
        translation_units=(
            replace(
                reverse_navigation_unit,
                advanced_facts_complete=True,
                index_profile=IndexProfile.FULL,
                cfg_facts_complete=True,
                data_flow_facts_complete=True,
                summary_facts_complete=True,
            ),
        ),
        cfg_graphs=(reverse_graph,),
        cfg_blocks=(reverse_block,),
        data_flow_analyses=(reverse_analysis,),
        function_summaries=(reverse_summary,),
    )
    override_edge = GraphEdge(
        source_id="callee",
        target_id="caller",
        relation=GraphRelation.OVERRIDES,
        translation_unit_id="tu-callee",
        build_configuration_id="config-callee",
    )
    changed_override = replace(_batch_for_unit(navigation, "tu-callee"), edges=(override_edge,))
    removed_override = replace(changed_override, edges=())
    neighbor_configuration = replace(
        navigation.build_configurations[1],
        id="config-neighbor",
        command_hash="command-neighbor",
    )
    neighbor_navigation_unit = replace(
        navigation.translation_units[1],
        id="tu-neighbor",
        build_configuration_id="config-neighbor",
        content_hash="content-neighbor",
        advanced_facts_complete=False,
        index_profile=IndexProfile.NAVIGATION,
        cfg_facts_complete=False,
        data_flow_facts_complete=False,
        summary_facts_complete=False,
    )
    navigation = replace(
        navigation,
        build_configurations=(
            *navigation.build_configurations,
            neighbor_configuration,
            reverse_configuration,
        ),
        translation_units=(
            *navigation.translation_units,
            neighbor_navigation_unit,
            reverse_navigation_unit,
        ),
        symbols=(*navigation.symbols, reverse_symbol),
        occurrences=(*navigation.occurrences, reverse_occurrence),
        callsites=(*navigation.callsites, reverse_site),
        call_targets=(*navigation.call_targets, reverse_target),
    )
    neighbor_deep = IngestionBatch(
        build_configurations=(neighbor_configuration,),
        translation_units=(
            replace(
                neighbor_navigation_unit,
                advanced_facts_complete=True,
                index_profile=IndexProfile.FULL,
                cfg_facts_complete=True,
                data_flow_facts_complete=True,
                summary_facts_complete=True,
            ),
        ),
        symbols=(),
        occurrences=(),
        edges=(),
    )
    alternate_configuration = replace(
        neighbor_configuration,
        id="config-alternate",
        command_hash="command-alternate",
        build_variant="alternate",
    )
    alternate_navigation_unit = replace(
        neighbor_navigation_unit,
        id="tu-alternate",
        build_configuration_id="config-alternate",
        content_hash="content-alternate",
        build_variant="alternate",
    )
    alternate_navigation = IngestionBatch(
        build_configurations=(alternate_configuration,),
        translation_units=(alternate_navigation_unit,),
        symbols=(),
        occurrences=(),
        edges=(),
    )
    alternate_deep = replace(
        alternate_navigation,
        translation_units=(
            replace(
                alternate_navigation_unit,
                advanced_facts_complete=True,
                index_profile=IndexProfile.FULL,
                cfg_facts_complete=True,
                data_flow_facts_complete=True,
                summary_facts_complete=True,
            ),
        ),
    )
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, navigation, index_profile=IndexProfile.NAVIGATION)
        store.apply_deep_overlay(
            root,
            (neighbor_deep,),
            root_symbol_id="caller",
            materialization_id="neighbor-token",
            identities={"tu-neighbor": "neighbor-identity"},
            command_hashes={"tu-neighbor": "command-neighbor"},
            distances={"tu-neighbor": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
        )
        store.apply_deep_overlay(
            root,
            (reverse_deep,),
            root_symbol_id="reverse",
            materialization_id="reverse-token",
            identities={"tu-reverse": "reverse-identity"},
            command_hashes={"tu-reverse": "command-reverse"},
            distances={"tu-reverse": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
        )
        store.apply_ingestion(
            root,
            alternate_navigation,
            build_variant=BuildVariant("alternate", Path("alternate.json")),
            index_profile=IndexProfile.NAVIGATION,
        )
        store.apply_deep_overlay(
            root,
            (alternate_deep,),
            root_symbol_id="caller",
            materialization_id="alternate-token",
            identities={"tu-alternate": "alternate-identity"},
            command_hashes={"tu-alternate": "command-alternate"},
            distances={"tu-alternate": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
            build_scope=BuildScope.single("alternate"),
        )
        store.apply_deep_overlay(
            root,
            (caller_deep,),
            root_symbol_id="caller",
            materialization_id="caller-before-override",
            identities={"tu-caller": "caller-identity"},
            command_hashes={"tu-caller": "command-caller"},
            distances={"tu-caller": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
        )
        assert "tu-caller" in store.deep_cache_states(root)
        store.apply_ingestion(root, changed_override, index_profile=IndexProfile.NAVIGATION)

        assert [
            target.target_symbol_id
            for target in store.call_targets(virtual_site.id).items
            if target.derivation == "indexed_override_candidate"
        ] == ["callee"]
        assert {"tu-caller", "tu-reverse"}.isdisjoint(store.deep_cache_states(root))

    with SQLiteStore(database, project_root=root) as restarted:
        assert not restarted.deep_materialization_matches(
            "caller-before-override", {"tu-caller": "caller-identity"}, root
        )
        assert not restarted.deep_materialization_matches(
            "reverse-token", {"tu-reverse": "reverse-identity"}, root
        )
        assert {
            "tu-neighbor",
            "tu-alternate",
        }.issubset(restarted.deep_cache_states(root))
        derived = restarted.deep_navigation_derived_targets(("tu-caller",), root)
        assert [target.target_symbol_id for target in derived] == ["callee"]
        merged = NativeClangIngestor._merge_batches(  # noqa: SLF001
            [caller_deep],
            profile=IndexProfile.FULL,
            additional_call_targets=derived,
        )
        restarted.apply_deep_overlay(
            root,
            (merged,),
            root_symbol_id="caller",
            materialization_id="caller-with-override",
            identities={"tu-caller": "caller-identity"},
            command_hashes={"tu-caller": "command-caller"},
            distances={"tu-caller": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
        )
        flow = AnalysisQueryService(restarted, root, BuildScope.single()).data_flow(
            FlowRequest(
                function_symbol_id="caller",
                materialization_id="caller-with-override",
            )
        )
        assert flow.available and flow.analyses
        assert not flow.analyses[0].summary_complete
        assert "external_or_unindexed_callee_body" in flow.analyses[0].summary_incomplete_reasons
        assert {
            target.target_symbol_id for target in restarted.call_targets(virtual_site.id).items
        } == {"caller", "callee"}

        restarted.apply_ingestion(root, removed_override, index_profile=IndexProfile.NAVIGATION)
        assert "tu-caller" not in restarted.deep_cache_states(root)
        assert {
            "tu-neighbor",
            "tu-alternate",
        }.issubset(restarted.deep_cache_states(root))

    with SQLiteStore(database, project_root=root) as restarted:
        assert not restarted.deep_materialization_matches(
            "caller-with-override", {"tu-caller": "caller-identity"}, root
        )
        assert restarted.deep_navigation_derived_targets(("tu-caller",), root) == ()
        restarted.apply_deep_overlay(
            root,
            (caller_deep,),
            root_symbol_id="caller",
            materialization_id="caller-after-removal",
            identities={"tu-caller": "caller-identity"},
            command_hashes={"tu-caller": "command-caller"},
            distances={"tu-caller": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
        )
        flow = AnalysisQueryService(restarted, root, BuildScope.single()).data_flow(
            FlowRequest(
                function_symbol_id="caller",
                materialization_id="caller-after-removal",
            )
        )
        assert flow.available and flow.analyses
        assert flow.analyses[0].summary_complete
        assert {
            target.target_symbol_id for target in restarted.call_targets(virtual_site.id).items
        } == {"caller"}


def test_deep_overlay_persists_solved_payload_and_invalidates_reverse_caller(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    navigation, deep = _two_tu_summary_batch(root)
    database = tmp_path / "index.db"
    identities = {"tu-caller": "identity-caller", "tu-callee": "identity-callee"}
    commands = {"tu-caller": "command-caller", "tu-callee": "command-callee"}
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, navigation, index_profile=IndexProfile.NAVIGATION)
        store.apply_deep_overlay(
            root,
            (deep,),
            root_symbol_id="caller",
            materialization_id="caller-generation",
            identities=identities,
            command_hashes=commands,
            distances={"tu-caller": 0, "tu-callee": 1},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
        )
        assert (
            store._connection.execute(  # noqa: SLF001 - persisted solver evidence
                "SELECT count(*) FROM summary_solution_payloads WHERE summary_id = 'summary-caller'"
            ).fetchone()[0]
            == 1
        )

    with SQLiteStore(database, project_root=root) as store:
        propagated = tuple(
            item
            for item in store.summary_effects("summary-caller", limit=100).items
            if not item.is_local
        )
        assert len(propagated) == 1
        assert propagated[0].target_symbol_id == "callee"
        callee_only = replace(
            deep,
            build_configurations=(deep.build_configurations[1],),
            translation_units=(deep.translation_units[1],),
            symbols=(deep.symbols[1],),
            occurrences=(deep.occurrences[1],),
            cfg_graphs=(deep.cfg_graphs[1],),
            cfg_blocks=(deep.cfg_blocks[1],),
            callsites=(),
            call_targets=(),
            data_flow_analyses=(deep.data_flow_analyses[1],),
            function_summaries=(
                next(item for item in deep.function_summaries if item.id == "summary-callee"),
            ),
            summary_effects=(
                next(item for item in deep.summary_effects if item.id == "callee-global-write"),
            ),
            interprocedural_flows=(),
        )
        store.apply_deep_overlay(
            root,
            (callee_only,),
            root_symbol_id="callee",
            materialization_id="callee-generation",
            identities={"tu-callee": "identity-callee-2"},
            command_hashes={"tu-callee": "command-callee"},
            distances={"tu-callee": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
        )
        assert not store.deep_materialization_matches("caller-generation", identities, root)
        assert store.deep_materialization_matches(
            "callee-generation", {"tu-callee": "identity-callee-2"}, root
        )


def test_full_profile_materialization_persists_token_bound_cfg_and_flow_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _navigation, deep = _two_tu_summary_batch(root)
    analyzer = tmp_path / "fake-analyzer"
    analyzer.write_bytes(b"stable analyzer identity")
    compilation_database = root / "compile_commands.json"
    compilation_database.write_text("[]", encoding="utf-8")
    database = tmp_path / "index.db"
    config = AppConfig(
        project_root=root,
        index_directory=tmp_path,
        database_path=database,
        compilation_database=compilation_database,
        clang_analyzer_path=analyzer,
        index_profile=IndexProfile.FULL,
        embedding_dimensions=32,
    )

    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, deep, index_profile=IndexProfile.FULL)
        materializer = DeepMaterializer(config, store)
        monkeypatch.setattr(materializer, "_revalidate", lambda *_args: None)
        monkeypatch.setattr(
            materializer,
            "_load_configurations",
            lambda *_args: {item.id: item for item in deep.build_configurations},
        )
        monkeypatch.setattr(
            "cpp_context_engine.ingestion.deep.NativeAnalyzerClient.probe",
            lambda *_args, **_kwargs: pytest.fail("full profile must not probe Clang"),
        )

        result = materializer.materialize(
            MaterializeDeepRequest(symbol_id="caller", max_tus=2, max_wall_seconds=10)
        )
        service = AnalysisQueryService(store, root, BuildScope.single())
        cfg = service.control_flow(
            CfgRequest(function_symbol_id="caller", materialization_id=result.materialization_id)
        )
        flow = service.data_flow(
            FlowRequest(function_symbol_id="caller", materialization_id=result.materialization_id)
        )
        assert cfg.available and cfg.graphs
        assert flow.available and flow.analyses
        assert store.deep_materialization_matches(
            result.materialization_id,
            {item.translation_unit_id: item.identity_hash for item in result.units},
            root,
        )

    with SQLiteStore(database, project_root=root) as restarted_store:
        service = AnalysisQueryService(restarted_store, root, BuildScope.single())
        assert service.control_flow(
            CfgRequest(function_symbol_id="caller", materialization_id=result.materialization_id)
        ).available
        assert service.data_flow(
            FlowRequest(function_symbol_id="caller", materialization_id=result.materialization_id)
        ).available


def test_partial_token_exposes_local_data_flow_with_truthful_closure_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    navigation, deep = _two_tu_summary_batch(root)
    foreign_symbol = replace(
        deep.symbols[0],
        span=SourceSpan(root / "callee.cpp", 1, 1),
        source_hash="source-caller-foreign",
        build_configuration_id="config-callee",
        translation_unit_id="tu-callee",
        variant_id="variant-caller-foreign",
    )
    foreign_occurrence = replace(
        deep.occurrences[0],
        id="occurrence-caller-foreign",
        span=foreign_symbol.span,
        translation_unit_id="tu-callee",
        build_configuration_id="config-callee",
    )
    navigation = replace(
        navigation,
        symbols=(*navigation.symbols, foreign_symbol),
        occurrences=(*navigation.occurrences, foreign_occurrence),
    )
    caller_summary = next(item for item in deep.function_summaries if item.id == "summary-caller")
    caller_only = replace(
        deep,
        build_configurations=(deep.build_configurations[0],),
        translation_units=(deep.translation_units[0],),
        symbols=(deep.symbols[0],),
        occurrences=(deep.occurrences[0],),
        cfg_graphs=(deep.cfg_graphs[0],),
        cfg_blocks=(deep.cfg_blocks[0],),
        data_flow_analyses=(deep.data_flow_analyses[0],),
        function_summaries=(caller_summary,),
        summary_effects=tuple(
            item for item in deep.summary_effects if item.summary_id == caller_summary.id
        ),
        interprocedural_flows=(),
    )
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, navigation, index_profile=IndexProfile.NAVIGATION)
        foreign_graph = replace(
            deep.cfg_graphs[0],
            id="graph-caller-foreign",
            entry_block_id="block-caller-foreign",
            normal_exit_block_id="block-caller-foreign",
            translation_unit_id="tu-callee",
            build_configuration_id="config-callee",
        )
        foreign_block = replace(
            deep.cfg_blocks[0],
            id="block-caller-foreign",
            graph_id=foreign_graph.id,
            translation_unit_id="tu-callee",
            build_configuration_id="config-callee",
        )
        foreign_analysis = replace(
            deep.data_flow_analyses[0],
            id="analysis-caller-foreign",
            graph_id=foreign_graph.id,
            translation_unit_id="tu-callee",
            build_configuration_id="config-callee",
        )
        foreign_summary = replace(
            caller_summary,
            id="summary-caller-foreign",
            graph_id=foreign_graph.id,
            analysis_id=foreign_analysis.id,
            translation_unit_id="tu-callee",
            build_configuration_id="config-callee",
        )
        foreign_batch = replace(
            deep,
            build_configurations=(deep.build_configurations[1],),
            translation_units=(deep.translation_units[1],),
            symbols=(deep.symbols[1], foreign_symbol),
            occurrences=(deep.occurrences[1], foreign_occurrence),
            cfg_graphs=(foreign_graph,),
            cfg_blocks=(foreign_block,),
            callsites=(),
            call_targets=(),
            data_flow_analyses=(foreign_analysis,),
            function_summaries=(foreign_summary,),
            summary_effects=(),
            interprocedural_flows=(),
        )
        store.apply_deep_overlay(
            root,
            (foreign_batch,),
            root_symbol_id="caller",
            materialization_id="foreign-caller",
            identities={"tu-callee": "identity-foreign"},
            command_hashes={"tu-callee": "command-callee"},
            distances={"tu-callee": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=True,
        )
        store.apply_deep_overlay(
            root,
            (caller_only,),
            root_symbol_id="caller",
            materialization_id="partial-caller",
            identities={"tu-caller": "identity-caller"},
            command_hashes={"tu-caller": "command-caller"},
            distances={"tu-caller": 0},
            analyzer_identity="analyzer",
            protocol_version=5,
            closure_complete=False,
            known_tus=2,
            omitted_tus=1,
            limit_reason="max_tus",
        )
        service = AnalysisQueryService(store, root, BuildScope.single())
        result = service.data_flow(
            FlowRequest(function_symbol_id="caller", materialization_id="partial-caller")
        )
        assert {item.id for item in store.cfg_graphs("caller", root).items} == {
            "graph-caller",
            "graph-caller-foreign",
        }

    assert result.available
    assert [item.graph_id for item in result.analyses] == ["graph-caller"]
    assert result.analyses[0].summary_complete is False
    assert "partial_materialization_closure" in result.analyses[0].summary_incomplete_reasons
    assert len(result.coverage) == 1
    assert result.coverage[0].control_flow
    assert result.coverage[0].data_flow
    assert not result.coverage[0].summaries
    assert not result.coverage[0].bindings
    assert not result.coverage[0].closure_complete
    assert result.coverage[0].materialization_id == "partial-caller"
    assert result.coverage[0].limit_reason == "max_tus"


def test_bounded_merge_restores_project_override_target_and_preserves_indirect_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    navigation, deep = _two_tu_summary_batch(root)
    virtual_site = replace(
        deep.callsites[0],
        dispatch_kind=CallDispatchKind.VIRTUAL,
        target_set_complete=False,
        unresolved_reason="open_world_external_overrides_possible",
    )
    static_target = replace(
        deep.call_targets[0],
        derivation="static_virtual_candidate",
        certainty=CallTargetCertainty.POSSIBLE,
        confidence=0.75,
    )
    override_symbol = replace(
        deep.symbols[1],
        id="override",
        qualified_name="Override::callee",
        signature="void Override::callee()",
        source_hash="source-override",
        variant_id="variant-override",
    )
    override_occurrence = replace(
        deep.occurrences[1], id="occurrence-override", symbol_id="override"
    )
    override_target = replace(
        static_target,
        id="target-override",
        target_symbol_id="override",
        derivation="indexed_override_candidate",
        confidence_reason="project index override candidate",
    )
    indirect_site = replace(
        virtual_site,
        id="call-indirect",
        dispatch_kind=CallDispatchKind.UNRESOLVED_INDIRECT,
        static_target_symbol_id=None,
        unresolved_reason="function_pointer_target_set_incomplete",
        callee_text="callback",
    )
    indirect_target = replace(
        static_target,
        id="target-indirect",
        callsite_id=indirect_site.id,
        derivation="function_pointer_points_to",
        confidence_reason="compiler points-to candidate",
    )
    navigation = replace(
        navigation,
        symbols=(*navigation.symbols, override_symbol),
        occurrences=(*navigation.occurrences, override_occurrence),
        callsites=(virtual_site, indirect_site),
        call_targets=(static_target, override_target, indirect_target),
    )
    analyzer_batch = replace(
        deep,
        symbols=(*deep.symbols, override_symbol),
        occurrences=(*deep.occurrences, override_occurrence),
        callsites=(virtual_site, indirect_site),
        call_targets=(static_target, indirect_target),
    )
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion(root, navigation, index_profile=IndexProfile.NAVIGATION)
        # Project-wide override expansion runs after individual TU parsing.  Insert
        # that already-derived navigation fact to model the bounded FULL request,
        # whose analyzer output intentionally contains only its selected TUs.
        store._put_call_facts(  # noqa: SLF001 - exact post-ingestion derivation fixture
            store._project_id(root),
            (),
            (override_target,),  # noqa: SLF001
        )
        store._connection.commit()  # noqa: SLF001
        derived = store.deep_navigation_derived_targets(
            tuple(unit.id for unit in deep.translation_units), root
        )
        assert [item.id for item in derived] == ["target-override"]
        conflicting_override = replace(
            override_target, confidence_reason="conflicting analyzer provenance"
        )
        conflicting = NativeClangIngestor._merge_batches(  # noqa: SLF001
            [
                replace(
                    analyzer_batch,
                    call_targets=(*analyzer_batch.call_targets, conflicting_override),
                )
            ],
            profile=IndexProfile.FULL,
            additional_call_targets=derived,
        )
        assert (
            next(
                item for item in conflicting.call_targets if item.id == "target-override"
            ).confidence_reason
            == "conflicting analyzer provenance"
        )
        with pytest.raises(RuntimeError, match="call_targets differ"):
            store.validate_deep_navigation_parity(root, (conflicting,))
        merged = NativeClangIngestor._merge_batches(  # noqa: SLF001 - bounded merge contract
            [analyzer_batch],
            profile=IndexProfile.FULL,
            additional_call_targets=derived,
        )
        store.validate_deep_navigation_parity(root, (merged,))

    assert {item.id for item in merged.call_targets} == {
        "target-callee",
        "target-override",
        "target-indirect",
    }
    solved_caller = next(item for item in merged.function_summaries if item.id == "summary-caller")
    assert not solved_caller.complete
    assert set(solved_caller.incomplete_reasons) == {
        "external_or_unindexed_callee_body",
        "unknown_or_external_call_target",
    }


@pytest.mark.parametrize("identity_change", ["callee", "command", "header"])
def test_restart_reuses_exact_closure_generation_across_root_symbols_without_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity_change: str
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    navigation, deep = _two_tu_summary_batch(root)
    alias_symbol = replace(
        deep.symbols[0],
        id="caller-alias",
        qualified_name="caller_alias",
        signature="void caller_alias()",
        source_hash="source-caller-alias",
        variant_id="variant-caller-alias",
    )
    alias_occurrence = replace(
        deep.occurrences[0], id="occurrence-caller-alias", symbol_id=alias_symbol.id
    )
    alias_site = replace(deep.callsites[0], id="call-callee-alias", owner_symbol_id=alias_symbol.id)
    alias_target = replace(
        deep.call_targets[0], id="target-callee-alias", callsite_id=alias_site.id
    )
    navigation = replace(
        navigation,
        symbols=(*navigation.symbols, alias_symbol),
        occurrences=(*navigation.occurrences, alias_occurrence),
        callsites=(*navigation.callsites, alias_site),
        call_targets=(*navigation.call_targets, alias_target),
    )
    deep = replace(
        deep,
        symbols=(*deep.symbols, alias_symbol),
        occurrences=(*deep.occurrences, alias_occurrence),
        callsites=(*deep.callsites, alias_site),
        call_targets=(*deep.call_targets, alias_target),
    )
    analyzer = tmp_path / "fake-analyzer"
    analyzer.write_bytes(b"stable analyzer identity")
    compilation_database = root / "compile_commands.json"
    compilation_database.write_text("[]", encoding="utf-8")
    database = tmp_path / "index.db"
    config = AppConfig(
        project_root=root,
        index_directory=tmp_path,
        database_path=database,
        compilation_database=compilation_database,
        clang_analyzer_path=analyzer,
        index_profile=IndexProfile.NAVIGATION,
        embedding_dimensions=32,
    )
    with SQLiteStore(database, project_root=root) as store:
        store.apply_ingestion(root, navigation, index_profile=IndexProfile.NAVIGATION)
        materializer = DeepMaterializer(config, store)
        control = DeepRequestControl(0.0, 10**12, DeepCancellation())
        roots = store.deep_definition_targets("caller", root)
        selected, known, omitted = materializer._closure(  # noqa: SLF001
            roots, "caller", 2, control
        )
        analyzer_identity = _file_digest(analyzer, control)
        identities = {
            item.translation_unit_id: materializer._identity(  # noqa: SLF001
                item, analyzer_identity, control
            )
            for item in selected
        }

        def generation(values: dict[str, str]) -> str:
            return _digest(
                [
                    "deep-closure-generation-v1",
                    "default",
                    "True",
                    *(f"{unit_id}\0{values[unit_id]}" for unit_id in sorted(values)),
                    analyzer_identity,
                    "cpp-context-clang-facts",
                    "5",
                    "15",
                    "full",
                ],
                control,
            )

        closure_generation_id = generation(identities)
        by_id = {item.translation_unit_id: item for item in selected}
        if identity_change == "callee":
            changed_unit_id = "tu-callee"
            changed_target = replace(
                by_id[changed_unit_id],
                content_hash="changed-callee-content",
                dependencies=((root / "callee.cpp", "changed-callee-content"),),
            )
        elif identity_change == "command":
            changed_unit_id = "tu-caller"
            changed_target = replace(by_id[changed_unit_id], command_hash="changed-command")
        else:
            changed_unit_id = "tu-caller"
            changed_target = replace(
                by_id[changed_unit_id],
                dependencies=(
                    *by_id[changed_unit_id].dependencies,
                    (root / "shared.hpp", "changed-header"),
                ),
            )
        changed_identity = materializer._identity(  # noqa: SLF001
            changed_target,
            analyzer_identity,
            control,
        )
        assert changed_identity != identities[changed_unit_id]
        store.apply_deep_overlay(
            root,
            (deep,),
            root_symbol_id="caller",
            materialization_id="original-caller-token",
            closure_generation_id=closure_generation_id,
            identities=identities,
            command_hashes={item.translation_unit_id: item.command_hash for item in selected},
            distances={item.translation_unit_id: item.distance for item in selected},
            analyzer_identity=analyzer_identity,
            analyzer_version="fixture-1",
            protocol="cpp-context-clang-facts",
            protocol_version=5,
            closure_complete=True,
            known_tus=known,
            omitted_tus=omitted,
        )

    def forbidden_probe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact persisted closure must not spawn or probe the analyzer")

    monkeypatch.setattr(
        "cpp_context_engine.ingestion.deep.NativeAnalyzerClient.probe", forbidden_probe
    )
    with SQLiteStore(database, project_root=root) as restarted_store:
        restarted = DeepMaterializer(config, restarted_store)
        configurations = {item.id: item for item in deep.build_configurations}
        monkeypatch.setattr(restarted, "_load_configurations", lambda *_args: configurations)
        monkeypatch.setattr(restarted, "_revalidate", lambda *_args: None)
        result = restarted.materialize(
            MaterializeDeepRequest(symbol_id="caller-alias", max_tus=2, max_wall_seconds=30)
        )
        assert result.cache_hit and result.status == "cache_hit"
        assert result.materialization_id != "original-caller-token"
        assert restarted_store.deep_materialization_matches(
            result.materialization_id, identities, root
        )
        assert (
            restarted_store._connection.execute(  # noqa: SLF001
                "SELECT count(*) FROM deep_materializations"
            ).fetchone()[0]
            == 2
        )
        persisted_provenance = tuple(
            restarted_store._connection.execute(  # noqa: SLF001
                """
                SELECT analyzer_identity, analyzer_version, protocol, protocol_version,
                       fact_schema_version, profile, closure_generation_id, unit_count,
                       known_tus, omitted_tus
                FROM deep_materializations WHERE id = ?
                """,
                (result.materialization_id,),
            ).fetchone()
        )
        assert persisted_provenance == (
            analyzer_identity,
            "fixture-1",
            "cpp-context-clang-facts",
            5,
            15,
            "full",
            closure_generation_id,
            2,
            2,
            0,
        )
        assert (
            restarted_store.deep_cached_closure(
                closure_generation_id,
                {"tu-caller": identities["tu-caller"]},
                root,
            )
            is None
        )
        subset_identities = {"tu-caller": identities["tu-caller"]}
        assert generation(subset_identities) != closure_generation_id
        assert (
            restarted_store.deep_cached_closure(
                generation(subset_identities), subset_identities, root
            )
            is None
        )
        changed_identities = {**identities, changed_unit_id: changed_identity}
        assert generation(changed_identities) != closure_generation_id
        assert (
            restarted_store.deep_cached_closure(closure_generation_id, changed_identities, root)
            is None
        )

        changed_configurations = tuple(
            replace(item, command_hash=changed_target.command_hash)
            if item.id == changed_target.build_configuration_id
            else item
            for item in deep.build_configurations
        )
        changed_units = tuple(
            replace(
                item,
                content_hash=changed_target.content_hash,
                dependencies=changed_target.dependencies,
            )
            if item.id == changed_unit_id
            else item
            for item in deep.translation_units
        )
        changed_deep = replace(
            deep,
            build_configurations=changed_configurations,
            translation_units=changed_units,
        )
        changed_navigation_units = tuple(
            replace(
                item,
                content_hash=changed_target.content_hash,
                dependencies=changed_target.dependencies,
            )
            if item.id == changed_unit_id
            else item
            for item in navigation.translation_units
        )
        changed_navigation = replace(
            navigation,
            build_configurations=changed_configurations,
            translation_units=changed_navigation_units,
        )

        def changed_unit_records(records: tuple) -> tuple:
            return tuple(item for item in records if item.translation_unit_id == changed_unit_id)

        changed_navigation_unit = next(
            item for item in changed_navigation.translation_units if item.id == changed_unit_id
        )
        refresh = IngestionBatch(
            build_configurations=tuple(
                item
                for item in changed_configurations
                if item.id == changed_navigation_unit.build_configuration_id
            ),
            translation_units=(changed_navigation_unit,),
            symbols=changed_unit_records(changed_navigation.symbols),
            occurrences=changed_unit_records(changed_navigation.occurrences),
            edges=changed_unit_records(changed_navigation.edges),
            callsites=changed_unit_records(changed_navigation.callsites),
            call_targets=changed_unit_records(changed_navigation.call_targets),
        )
        restarted_store.apply_ingestion(root, refresh, index_profile=IndexProfile.NAVIGATION)
        neighbor_id = "tu-callee" if changed_unit_id == "tu-caller" else "tu-caller"
        assert (
            restarted_store.deep_cache_states(root)[neighbor_id].identity_hash
            == identities[neighbor_id]
        )
        assert (
            restarted_store._connection.execute(  # noqa: SLF001
                "SELECT count(*) FROM deep_materializations"
            ).fetchone()[0]
            == 0
        )
        assert (
            restarted_store.analysis_coverage(
                "caller-alias", root, materialization_id=result.materialization_id
            )
            == ()
        )
        if changed_unit_id == "tu-callee":
            # Recreate retained caller navigation after its target-symbol FK was
            # cascaded; a project refresh supplies it, while this fixture parses
            # only the changed callee so its neighboring deep cache stays intact.
            restarted_store._put_call_facts(  # noqa: SLF001
                restarted_store._project_id(root),  # noqa: SLF001
                changed_navigation.callsites,
                changed_navigation.call_targets,
            )
            restarted_store._connection.commit()  # noqa: SLF001
        restarted_store.apply_deep_overlay(
            root,
            (changed_deep,),
            root_symbol_id="caller-alias",
            materialization_id="replacement-token",
            closure_generation_id=generation(changed_identities),
            identities=changed_identities,
            command_hashes={
                item.translation_unit_id: (
                    changed_target.command_hash
                    if item.translation_unit_id == changed_unit_id
                    else item.command_hash
                )
                for item in selected
            },
            distances={item.translation_unit_id: item.distance for item in selected},
            analyzer_identity=analyzer_identity,
            analyzer_version="fixture-1",
            protocol="cpp-context-clang-facts",
            protocol_version=5,
            closure_complete=True,
        )
        assert not restarted_store.deep_materialization_matches(
            "original-caller-token", identities, root
        )
        assert not restarted_store.deep_materialization_matches(
            result.materialization_id, identities, root
        )
        assert restarted_store.deep_materialization_matches(
            "replacement-token", changed_identities, root
        )
        assert (
            restarted_store.analysis_coverage(
                "caller-alias", root, materialization_id=result.materialization_id
            )
            == ()
        )

    assert result.provenance is not None
    assert result.provenance.analyzer_identity == analyzer_identity
    assert result.provenance.analyzer_version == "fixture-1"
    assert result.provenance.protocol == "cpp-context-clang-facts"
    assert result.provenance.protocol_version == 5
    assert result.provenance.fact_schema_version == 15
    assert result.provenance.profile is IndexProfile.FULL
    assert result.provenance.build_scope == ["default"]
    assert result.provenance.closure_generation_id == closure_generation_id
    assert {item.identity_hash for item in result.units} == set(identities.values())
