from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from pydantic import ValidationError

from cpp_context_engine.api import CallRequest, CfgRequest, FlowRequest
from cpp_context_engine.api.http import create_app
from cpp_context_engine.cli import main
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.mcp.server import create_mcp_server
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
    CfgEdge,
    CfgEdgeKind,
    CfgElement,
    CfgGraph,
    CodeSymbol,
    DataAccess,
    DataAccessKind,
    DataFlowAnalysis,
    DataFlowCertainty,
    DataFlowEvidence,
    DataFlowRelation,
    GraphDirection,
    GraphEdge,
    GraphRelation,
    MemoryLocation,
    MemoryLocationKind,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.runtime import build_runtime
from cpp_context_engine.storage import SQLiteStore


def _config(project: Path, database: Path) -> AppConfig:
    variants = (
        BuildVariant("alpha", project / "alpha" / "compile_commands.json"),
        BuildVariant("beta", project / "beta" / "compile_commands.json"),
    )
    return AppConfig(
        project_root=project,
        index_directory=database.parent,
        database_path=database,
        build_variants=variants,
        build_scope=BuildScope(("alpha", "beta")),
        embedding_dimensions=16,
    )


def _seed(config: AppConfig) -> None:
    source = config.project_root / "src" / "fixture.cpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "int target() { return 1; }\nint analyze() { int x = 1; return target() + x; }\n",
        encoding="utf-8",
    )
    assert config.database_path is not None
    with SQLiteStore(config.database_path, project_root=config.project_root) as store:
        for variant in config.build_variants:
            name = variant.name
            configuration = BuildConfiguration(
                id=f"config-{name}",
                source_path=source,
                directory=config.project_root,
                arguments=("c++", str(source)),
                command_hash=f"command-{name}",
                build_variant=name,
            )
            unit = TranslationUnit(
                id=f"unit-{name}",
                build_configuration_id=configuration.id,
                source_path=source,
                content_hash=f"content-{name}",
                build_variant=name,
                analysis_backend="clang-libtooling",
                advanced_facts_complete=True,
            )
            function = CodeSymbol(
                id="cxx:analyze",
                qualified_name="analyze",
                kind=SymbolKind.FUNCTION,
                span=SourceSpan(source, 2, 2),
                signature="int analyze()",
                source_hash=f"analyze-{name}",
                source_text="int analyze() { int x = 1; return target() + x; }",
                build_configuration_id=configuration.id,
                translation_unit_id=unit.id,
                build_variant=name,
                metadata={"is_definition": True},
            )
            target = CodeSymbol(
                id="cxx:target",
                qualified_name="target",
                kind=SymbolKind.FUNCTION,
                span=SourceSpan(source, 1, 1),
                signature="int target()",
                source_hash=f"target-{name}",
                source_text="int target() { return 1; }",
                build_configuration_id=configuration.id,
                translation_unit_id=unit.id,
                build_variant=name,
                metadata={"is_definition": True},
            )
            graph_id = f"graph-{name}"
            analysis_id = f"analysis-{name}"
            entry_id = f"block-entry-{name}"
            exit_id = f"block-exit-{name}"
            location_id = f"location-x-{name}"
            definition_id = f"access-definition-{name}"
            read_id = f"access-read-{name}"
            callsite_id = f"callsite-{name}"
            batch = IngestionBatch(
                build_configurations=(configuration,),
                translation_units=(unit,),
                symbols=(function, target),
                occurrences=(),
                edges=(
                    GraphEdge(
                        function.id,
                        target.id,
                        GraphRelation.CALLS,
                        unit.id,
                        f"edge-{name}",
                        configuration.id,
                        name,
                    ),
                ),
                build_variants=(variant,),
                cfg_graphs=(
                    CfgGraph(
                        graph_id,
                        function.id,
                        entry_id,
                        exit_id,
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
                cfg_blocks=(
                    CfgBlock(
                        entry_id,
                        graph_id,
                        0,
                        CfgBlockRole.ENTRY,
                        True,
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                    CfgBlock(
                        exit_id,
                        graph_id,
                        1,
                        CfgBlockRole.NORMAL_EXIT,
                        True,
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
                cfg_elements=(
                    CfgElement(
                        f"element-{name}",
                        graph_id,
                        entry_id,
                        0,
                        "statement",
                        statement_class="DeclStmt",
                        text="int x = 1",
                        expansion_span=SourceSpan(source, 2, 2, 17, 25),
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
                cfg_edges=(
                    CfgEdge(
                        f"cfg-edge-{name}",
                        graph_id,
                        entry_id,
                        exit_id,
                        CfgEdgeKind.RETURN,
                        0,
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
                callsites=(
                    CallSite(
                        callsite_id,
                        function.id,
                        CallDispatchKind.DIRECT,
                        SourceSpan(source, 2, 2, 40, 47),
                        SourceSpan(source, 2, 2, 40, 47),
                        True,
                        static_target_symbol_id=target.id,
                        callee_text="target",
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
                call_targets=(
                    CallTarget(
                        f"target-evidence-{name}",
                        callsite_id,
                        target.id,
                        (
                            CallTargetCertainty.CERTAIN
                            if name == "alpha"
                            else CallTargetCertainty.POSSIBLE
                        ),
                        1.0 if name == "alpha" else 0.5,
                        "compiler-selected direct target; ranking value is not a probability",
                        "direct" if name == "alpha" else "fixture_possible",
                        SourceSpan(source, 2, 2, 40, 47),
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
                data_flow_analyses=(
                    DataFlowAnalysis(
                        analysis_id,
                        graph_id,
                        True,
                        (),
                        2,
                        64,
                        64,
                        8,
                        4096,
                        unit.id,
                        configuration.id,
                        name,
                    ),
                ),
                memory_locations=(
                    MemoryLocation(
                        location_id,
                        analysis_id,
                        graph_id,
                        MemoryLocationKind.LOCAL,
                        "x",
                        "int",
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
                data_accesses=(
                    DataAccess(
                        definition_id,
                        analysis_id,
                        graph_id,
                        entry_id,
                        location_id,
                        DataAccessKind.INITIALIZATION,
                        0,
                        span=SourceSpan(source, 2, 2, 17, 25),
                        expression="int x = 1",
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                    DataAccess(
                        read_id,
                        analysis_id,
                        graph_id,
                        entry_id,
                        location_id,
                        DataAccessKind.READ,
                        1,
                        span=SourceSpan(source, 2, 2, 50, 50),
                        expression="x",
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
                data_flow_evidence=(
                    DataFlowEvidence(
                        f"flow-evidence-{name}",
                        analysis_id,
                        graph_id,
                        DataFlowRelation.REACHING_DEFINITION,
                        (
                            DataFlowCertainty.CERTAIN
                            if name == "alpha"
                            else DataFlowCertainty.POSSIBLE
                        ),
                        "definition reaches read",
                        source_access_id=definition_id,
                        target_access_id=read_id,
                        evidence_span=SourceSpan(source, 2, 2, 50, 50),
                        translation_unit_id=unit.id,
                        build_configuration_id=configuration.id,
                        build_variant=name,
                    ),
                ),
            )
            store.apply_ingestion(
                config.project_root,
                batch,
                current_translation_unit_ids=frozenset({unit.id}),
                build_variant=variant,
            )


def test_union_and_single_build_queries_are_bounded_and_evidence_ranked(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "index.db")
    _seed(config)

    with build_runtime(config) as runtime:
        union = runtime.analysis_service.data_flow(FlowRequest(function_symbol_id="cxx:analyze"))
        alpha = runtime.analysis_service.data_flow(
            FlowRequest(function_symbol_id="cxx:analyze", builds=["alpha"])
        )
        bounded = runtime.analysis_service.data_flow(
            FlowRequest(
                function_symbol_id="cxx:analyze",
                max_analyses=1,
                max_locations=1,
                max_accesses=1,
                max_evidence=1,
            )
        )
        calls = runtime.analysis_service.calls(
            CallRequest(
                symbol_id="cxx:analyze",
                direction=GraphDirection.OUTGOING,
                max_results=10,
            )
        )

    assert union.scope.kind == "union"
    assert union.scope.label == "union:alpha,beta"
    assert {item.provenance.build_variant for item in union.analyses} == {"alpha", "beta"}
    assert alpha.scope.label == "build:alpha"
    assert {item.provenance.build_variant for item in alpha.analyses} == {"alpha"}
    assert bounded.truncated
    assert len(bounded.analyses) == 1
    assert len(bounded.analyses[0].locations) == 1
    assert len(bounded.analyses[0].accesses) == 1
    assert len(bounded.analyses[0].evidence) == 1
    assert [item.certainty for item in calls.calls] == [
        CallTargetCertainty.CERTAIN,
        CallTargetCertainty.POSSIBLE,
    ]
    with pytest.raises(ValidationError):
        CfgRequest(function_symbol_id="cxx:analyze", max_edges=2_001)


def test_cfg_contract_is_identical_across_service_cli_http_and_mcp(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    database = tmp_path / "index.db"
    config = _config(project, database)
    _seed(config)

    with build_runtime(config) as runtime:
        expected = runtime.analysis_service.control_flow(
            CfgRequest(function_symbol_id="cxx:analyze", builds=["alpha"])
        ).model_dump(mode="json")
        stored = runtime.store.cfg_graphs(
            "cxx:analyze", project, build_scope=BuildScope.single("alpha"), limit=5
        )
        assert [item.id for item in stored.items] == [
            item["graph_id"] for item in expected["graphs"]
        ]
        client = TestClient(
            create_app(
                retrieval_service=runtime.retrieval_service,
                analysis_service=runtime.analysis_service,
            )
        )
        response = client.post(
            "/v1/cfg", json={"function_symbol_id": "cxx:analyze", "builds": ["alpha"]}
        )
        assert response.status_code == 200
        assert response.json() == expected

    assert (
        main(
            [
                "cfg",
                "cxx:analyze",
                "--project",
                str(project),
                "--db",
                str(database),
                "--build",
                "alpha",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == expected

    async def scenario() -> None:
        async with Client(create_mcp_server(config), mode="legacy") as mcp:
            result = await mcp.call_tool(
                "control_flow", {"symbol_id": "cxx:analyze", "builds": ["alpha"]}
            )
            assert not result.is_error
            assert result.structured_content == expected

    anyio.run(scenario)


def test_flow_contract_is_identical_across_service_cli_http_and_mcp(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    database = tmp_path / "index.db"
    config = _config(project, database)
    _seed(config)

    with build_runtime(config) as runtime:
        expected = runtime.analysis_service.data_flow(
            FlowRequest(function_symbol_id="cxx:analyze", builds=["beta"])
        ).model_dump(mode="json")
        stored_graph = runtime.store.cfg_graphs(
            "cxx:analyze", project, build_scope=BuildScope.single("beta"), limit=1
        ).items[0]
        stored_flow = runtime.store.data_flow_analyses(
            stored_graph.id, project, build_scope=BuildScope.single("beta"), limit=1
        ).items[0]
        assert stored_flow.id == expected["analyses"][0]["analysis_id"]
        response = TestClient(
            create_app(
                retrieval_service=runtime.retrieval_service,
                analysis_service=runtime.analysis_service,
            )
        ).post("/v1/flow", json={"function_symbol_id": "cxx:analyze", "builds": ["beta"]})
        assert response.status_code == 200
        assert response.json() == expected

    assert (
        main(
            [
                "flow",
                "cxx:analyze",
                "--project",
                str(project),
                "--db",
                str(database),
                "--build",
                "beta",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == expected

    async def scenario() -> None:
        async with Client(create_mcp_server(config), mode="legacy") as mcp:
            result = await mcp.call_tool(
                "data_flow", {"symbol_id": "cxx:analyze", "builds": ["beta"]}
            )
            assert not result.is_error
            assert result.structured_content == expected

    anyio.run(scenario)


def test_mcp_build_filters_do_not_leak_and_call_evidence_is_ranked(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "index.db")
    _seed(config)

    async def scenario() -> None:
        async with Client(create_mcp_server(config), mode="legacy") as mcp:
            listed = await mcp.call_tool("list_builds", {})
            assert listed.structured_content["active_scope"]["label"] == "union:alpha,beta"
            assert str(project) not in json.dumps(listed.structured_content)

            searched = await mcp.call_tool(
                "search_code",
                {
                    "query": "analyze",
                    "builds": ["beta"],
                    "max_results": 10,
                    "max_context_tokens": 1_000,
                },
            )
            assert not searched.is_error
            assert searched.structured_content["scope_label"] == "build:beta"
            assert {
                item["symbol"]["build_variant"] for item in searched.structured_content["items"]
            } == {"beta"}
            target_item = next(
                item
                for item in searched.structured_content["items"]
                if item["symbol"]["symbol_id"] == "cxx:target"
            )
            assert "call possible" in target_item["reason"]
            assert "confidence 0.50" in target_item["reason"]
            assert "derivation fixture_possible" in target_item["reason"]
            assert "build beta" in target_item["reason"]

            calls = await mcp.call_tool("callees", {"symbol_id": "cxx:analyze"})
            assert not calls.is_error
            assert calls.structured_content["scope_label"] == "union:alpha,beta"
            assert [item["certainty"] for item in calls.structured_content["edges"]] == [
                "certain",
                "possible",
            ]
            assert all(item["callsite_id"] for item in calls.structured_content["edges"])
            beta_calls = await mcp.call_tool(
                "callees", {"symbol_id": "cxx:analyze", "builds": ["beta"]}
            )
            assert {item["build_variant"] for item in beta_calls.structured_content["edges"]} == {
                "beta"
            }
            neighbors = await mcp.call_tool(
                "neighbors",
                {
                    "symbol_id": "cxx:analyze",
                    "relations": ["calls"],
                    "direction": "both",
                },
            )
            assert [item["certainty"] for item in neighbors.structured_content["edges"]] == [
                "certain",
                "possible",
            ]
            assert all(item["callsite_id"] for item in neighbors.structured_content["edges"])

    anyio.run(scenario)


def test_http_search_uses_request_scope_and_sanitizes_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "index.db")
    _seed(config)

    with build_runtime(config) as runtime:
        client = TestClient(
            create_app(
                retrieval_service=runtime.retrieval_service,
                analysis_service=runtime.analysis_service,
                scoped_query=runtime.query_context,
            )
        )
        response = client.post(
            "/v1/context",
            json={"query": "analyze", "builds": ["beta"], "max_context_tokens": 1_000},
        )

    assert response.status_code == 200
    document = response.json()
    assert document["scope"] == {
        "kind": "single",
        "label": "build:beta",
        "variants": ["beta"],
    }
    assert {item["build_variant"] for item in document["items"]} == {"beta"}
    assert all(not item["path"].startswith("/") for item in document["items"])
    assert str(project) not in json.dumps(document)
