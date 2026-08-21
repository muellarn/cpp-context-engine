from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.cli import main
from cpp_context_engine.ingestion import (
    AnalyzerLimitError,
    AnalyzerProtocolError,
    NativeAnalyzerClient,
    NativeClangIngestor,
)
from cpp_context_engine.ingestion.clang import ClangIngestor, ClangUnavailableError
from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.models import (
    BuildScope,
    BuildVariant,
    CfgEdgeKind,
    GraphRelation,
    OccurrenceKind,
)
from cpp_context_engine.storage import SQLiteStore

FIXTURE = Path(__file__).parent / "fixtures" / "analyzer_project"
PARITY_FIXTURE = Path(__file__).parent / "fixtures" / "cpp_project"
CFG_FIXTURE = Path(__file__).parent / "fixtures" / "cfg_project"


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


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake-analyzer"
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_native_handshake_matches_protocol_golden() -> None:
    request = {
        "type": "hello",
        "protocol": "cpp-context-clang-facts",
        "protocol_version": 2,
        "required_clang_major": 18,
    }
    completed = subprocess.run(  # noqa: S603 - repository-built test binary
        [_binary()],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "analyzer_protocol" / "hello.json").read_text()
    )
    assert json.loads(completed.stdout) == expected
    assert completed.stderr == ""


def test_real_ast_macro_template_lambda_and_relationship_facts() -> None:
    batch = NativeClangIngestor(NativeAnalyzerClient(_binary(), timeout_seconds=15)).ingest(
        FIXTURE, FIXTURE / "compile_commands.json"
    )
    relations = {edge.relation for edge in batch.edges}

    assert {
        GraphRelation.CALLS,
        GraphRelation.INCLUDES,
        GraphRelation.INHERITS,
        GraphRelation.OVERRIDES,
        GraphRelation.REFERENCES,
        GraphRelation.USES_TYPE,
    } <= relations
    macro = next(symbol for symbol in batch.symbols if symbol.qualified_name == "APPLY_TWICE")
    expansions = [
        occurrence
        for occurrence in batch.occurrences
        if occurrence.symbol_id == macro.id and occurrence.kind == OccurrenceKind.MACRO_EXPANSION
    ]
    assert len(expansions) == 2
    assert any(
        occurrence.metadata["spelling_span"] != occurrence.metadata["expansion_span"]
        for occurrence in expansions
    )
    templates = [symbol for symbol in batch.symbols if "template_kind" in symbol.metadata]
    assert templates
    assert any(symbol.metadata["template_arguments"] for symbol in templates)
    lambdas = [
        symbol for symbol in batch.symbols if symbol.metadata.get("is_lambda_call_operator") is True
    ]
    assert len(lambdas) == 1
    assert lambdas[0].metadata["stable_lambda_key"].startswith("lambda:")
    assert all(symbol.metadata["advanced_facts_complete"] for symbol in batch.symbols)

    definition = next(
        symbol
        for symbol in batch.symbols
        if symbol.qualified_name == "analyzer_fixture::Derived::evaluate"
        and symbol.metadata["is_definition"]
    )
    source = definition.span.path.read_bytes()
    assert (
        source[
            definition.metadata["start_offset"] : definition.metadata["end_offset_exclusive"]
        ].decode()
        == definition.source_text
    )

    constructor = next(
        symbol
        for symbol in batch.symbols
        if symbol.qualified_name == "analyzer_fixture::Derived::Derived"
    )
    assert any(
        occurrence.symbol_id == constructor.id and occurrence.kind == OccurrenceKind.CALL
        for occurrence in batch.occurrences
    )


def _cfg_batch(*, database: str = "compile_commands.json", variant: str = "default"):
    return NativeClangIngestor(NativeAnalyzerClient(_binary(), timeout_seconds=30)).ingest(
        CFG_FIXTURE, CFG_FIXTURE / database, build_variant=variant
    )


def _cfg_for(batch, qualified_name: str):
    symbol = next(symbol for symbol in batch.symbols if symbol.qualified_name == qualified_name)
    return next(graph for graph in batch.cfg_graphs if graph.function_symbol_id == symbol.id)


def test_real_cfg_snapshot_covers_control_flow_macro_and_lifetime_facts() -> None:
    batch = _cfg_batch()
    snapshot = {}
    for graph in batch.cfg_graphs:
        symbol = next(symbol for symbol in batch.symbols if symbol.id == graph.function_symbol_id)
        blocks = [block for block in batch.cfg_blocks if block.graph_id == graph.id]
        edges = [edge for edge in batch.cfg_edges if edge.graph_id == graph.id]
        snapshot[symbol.qualified_name] = (
            len(blocks),
            sum(not block.reachable for block in blocks),
            Counter(edge.kind.value for edge in edges),
        )

    assert snapshot["cfg_fixture::branching"] == (
        6,
        0,
        Counter({"fallthrough": 3, "true": 1, "false": 1, "return": 1}),
    )
    assert snapshot["cfg_fixture::choose"] == (
        6,
        0,
        Counter({"return": 3, "case": 2, "default": 1, "fallthrough": 1}),
    )
    assert snapshot["cfg_fixture::loop"] == (
        11,
        0,
        Counter(
            {
                "true": 3,
                "false": 3,
                "fallthrough": 3,
                "break": 1,
                "continue": 1,
                "loop_back": 1,
                "return": 1,
            }
        ),
    )
    assert snapshot["cfg_fixture::early_return"][2]["return"] == 2
    assert snapshot["cfg_fixture::jump"][2]["goto"] == 1
    assert snapshot["cfg_fixture::exception_flow"][2]["exception"] == 3
    assert snapshot["cfg_fixture::unreachable_after_return"][1] == 1

    all_edge_kinds = {edge.kind for edge in batch.cfg_edges}
    assert set(CfgEdgeKind) <= all_edge_kinds
    graph = _cfg_for(batch, "cfg_fixture::branching")
    blocks = [block for block in batch.cfg_blocks if block.graph_id == graph.id]
    assert {block.id for block in blocks if block.role.value == "entry"} == {graph.entry_block_id}
    assert {block.id for block in blocks if block.role.value == "normal_exit"} == {
        graph.normal_exit_block_id
    }
    assert graph.exceptional_exit_block_id is None
    assert graph.build_options == {
        "add_cxx_default_init_expr_in_aggregates": True,
        "add_cxx_default_init_expr_in_ctors": True,
        "add_cxx_new_allocator": True,
        "add_eh_edges": True,
        "add_implicit_dtors": True,
        "add_initializers": True,
        "add_lifetime": True,
        "add_loop_exit": True,
        "add_rich_cxx_constructors": True,
        "add_scopes": True,
        "add_static_init_branches": True,
        "add_temporary_dtors": True,
        "add_virtual_base_branches": True,
        "always_add_all_statements": True,
        "mark_elided_cxx_constructors": True,
        "omit_implicit_value_initializers": False,
        "prune_trivially_false_edges": False,
    }
    branch_elements = [element for element in batch.cfg_elements if element.graph_id == graph.id]
    assert any(
        element.spelling_span != element.expansion_span
        for element in branch_elements
        if element.spelling_span and element.expansion_span
    )
    lifetime = _cfg_for(batch, "cfg_fixture::lifetime")
    lifetime_kinds = {
        element.kind for element in batch.cfg_elements if element.graph_id == lifetime.id
    }
    assert {"constructor", "automatic_object_destructor", "lifetime_end"} <= lifetime_kinds
    assert all(
        item.translation_unit_id == graph.translation_unit_id
        and item.build_configuration_id == graph.build_configuration_id
        and item.build_variant == graph.build_variant
        for item in (
            *blocks,
            *branch_elements,
            *(edge for edge in batch.cfg_edges if edge.graph_id == graph.id),
        )
    )


def test_cfg_ids_are_deterministic_and_sqlite_reads_are_bounded(tmp_path: Path) -> None:
    first = _cfg_batch()
    second = _cfg_batch()
    for attribute in ("cfg_graphs", "cfg_blocks", "cfg_elements", "cfg_edges"):
        assert [item.id for item in getattr(first, attribute)] == [
            item.id for item in getattr(second, attribute)
        ]

    with SQLiteStore(tmp_path / "index.db", project_root=CFG_FIXTURE) as store:
        store.apply_ingestion(CFG_FIXTURE, first)
        graphs = store.cfg_graphs(limit=3)
        assert len(graphs.items) == 3
        assert graphs.truncated
        graph = _cfg_for(first, "cfg_fixture::branching")
        assert store.get_cfg_graph(graph.id) == graph
        assert store.cfg_blocks(graph.id).items == tuple(
            sorted(
                (block for block in first.cfg_blocks if block.graph_id == graph.id),
                key=lambda block: (block.index, block.id),
            )
        )
        assert store.cfg_elements(graph.id, limit=2).truncated
        assert not store.cfg_edges(graph.id).truncated
        with pytest.raises(ValueError, match="CFG page limit"):
            store.cfg_blocks(graph.id, limit=0)


def test_cfg_exception_edges_follow_build_configuration_and_build_scope(tmp_path: Path) -> None:
    enabled = _cfg_batch(variant="exceptions")
    disabled = _cfg_batch(database="compile_commands_no_eh.json", variant="no-exceptions")
    enabled_graph = _cfg_for(enabled, "cfg_fixture::exception_flow")
    disabled_graph = _cfg_for(disabled, "cfg_fixture::exception_flow")
    assert enabled_graph.build_options["add_eh_edges"] is True
    assert disabled_graph.build_options["add_eh_edges"] is False
    assert any(
        edge.kind == CfgEdgeKind.EXCEPTION
        for edge in enabled.cfg_edges
        if edge.graph_id == enabled_graph.id
    )
    assert not any(
        edge.kind == CfgEdgeKind.EXCEPTION
        for edge in disabled.cfg_edges
        if edge.graph_id == disabled_graph.id
    )

    with SQLiteStore(tmp_path / "index.db", project_root=CFG_FIXTURE) as store:
        store.apply_ingestion(
            CFG_FIXTURE,
            enabled,
            build_variant=BuildVariant("exceptions", CFG_FIXTURE / "compile_commands.json"),
        )
        store.apply_ingestion(
            CFG_FIXTURE,
            disabled,
            build_variant=BuildVariant(
                "no-exceptions", CFG_FIXTURE / "compile_commands_no_eh.json"
            ),
        )
        assert store.cfg_graphs(
            enabled_graph.function_symbol_id, build_scope=BuildScope.single("exceptions")
        ).items == (enabled_graph,)
        assert store.cfg_graphs(
            disabled_graph.function_symbol_id,
            build_scope=BuildScope.single("no-exceptions"),
        ).items == (disabled_graph,)
        assert store.remove_build_variant("exceptions")
        assert not store.cfg_graphs(build_scope=BuildScope.single("exceptions")).items
        assert (
            store.get_cfg_graph(disabled_graph.id, build_scope=BuildScope.single("no-exceptions"))
            == disabled_graph
        )


def test_cfg_reindex_replaces_only_changed_tu_and_rolls_back_atomically(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(PARITY_FIXTURE, project)
    database = project / "compile_commands.json"
    ingestor = NativeClangIngestor(NativeAnalyzerClient(_binary(), timeout_seconds=30))
    with SQLiteStore(tmp_path / "index.db", project_root=project) as store:
        indexer = ProjectIndexer(ingestor, store)
        first = indexer.index(project, database)
        assert first.indexed_cfg_graphs > 0
        units = store._connection.execute(  # noqa: SLF001
            "SELECT id, source_path FROM translation_units ORDER BY source_path"
        ).fetchall()
        unchanged_unit = next(
            row["id"] for row in units if row["source_path"].endswith("model.cpp")
        )
        unchanged_before = tuple(
            tuple(row)
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT id, function_symbol_id FROM cfg_graphs "
                "WHERE translation_unit_id = ? ORDER BY id",
                (unchanged_unit,),
            )
        )
        old_graph_count = len(store.cfg_graphs(limit=10_000).items)
        source = project / "src" / "main.cpp"
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        changed = indexer.index(project, database)
        assert changed.indexed_translation_units == 1
        assert (
            tuple(
                tuple(row)
                for row in store._connection.execute(  # noqa: SLF001
                    "SELECT id, function_symbol_id FROM cfg_graphs "
                    "WHERE translation_unit_id = ? ORDER BY id",
                    (unchanged_unit,),
                )
            )
            == unchanged_before
        )
        assert len(store.cfg_graphs(limit=10_000).items) == old_graph_count

        current = ingestor.ingest(project, database)
        broken = replace(
            current.cfg_edges[0], id="broken-cfg-edge", target_block_id="missing-block"
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.apply_ingestion(project, replace(current, cfg_edges=(*current.cfg_edges, broken)))
        assert len(store.cfg_graphs(limit=10_000).items) == old_graph_count


@pytest.mark.clang
def test_companion_preserves_baseline_canonical_ids_and_relation_parity() -> None:
    try:
        baseline = ClangIngestor().ingest(PARITY_FIXTURE, PARITY_FIXTURE / "compile_commands.json")
    except ClangUnavailableError as error:
        pytest.skip(str(error))
    native = NativeClangIngestor(NativeAnalyzerClient(_binary(), timeout_seconds=15)).ingest(
        PARITY_FIXTURE, PARITY_FIXTURE / "compile_commands.json"
    )
    names = {"demo::Base", "demo::Derived", "demo::Derived::compute", "demo::helper", "run"}
    baseline_ids = {
        (symbol.qualified_name, symbol.id)
        for symbol in baseline.symbols
        if symbol.qualified_name in names
    }
    native_ids = {
        (symbol.qualified_name, symbol.id)
        for symbol in native.symbols
        if symbol.qualified_name in names
    }
    assert baseline_ids <= native_ids
    assert {edge.relation for edge in baseline.edges} <= {edge.relation for edge in native.edges}


@pytest.mark.clang
def test_switching_from_baseline_to_companion_forces_reindex(tmp_path: Path) -> None:
    try:
        baseline = ClangIngestor()
    except ClangUnavailableError as error:
        pytest.skip(str(error))
    database = PARITY_FIXTURE / "compile_commands.json"
    with SQLiteStore(tmp_path / "index.db", project_root=PARITY_FIXTURE) as store:
        first = ProjectIndexer(baseline, store).index(PARITY_FIXTURE, database)
        upgraded = ProjectIndexer(
            NativeClangIngestor(NativeAnalyzerClient(_binary(), timeout_seconds=15)), store
        ).index(PARITY_FIXTURE, database)
        states = store.translation_unit_states(PARITY_FIXTURE)

    assert first.indexed_translation_units == 2
    assert upgraded.indexed_translation_units == 2
    assert {state.analysis_backend for state in states.values()} == {"clang-libtooling"}
    assert all(state.advanced_facts_complete for state in states.values())


def test_protocol_mismatch_fails_before_analysis_and_sanitizes_path(tmp_path: Path) -> None:
    marker = tmp_path / "analysis-ran"
    script = _script(
        tmp_path,
        f"""import json, sys
request = json.loads(sys.stdin.readline())
print(json.dumps({{"type":"hello","protocol":"wrong","protocol_version":99,
                   "analyzer_version":"x","clang_major":17,"capabilities":[]}}))
for line in sys.stdin:
    open({str(marker)!r}, "w").write("bad")
""",
    )

    with pytest.raises(AnalyzerProtocolError, match="protocol mismatch") as captured:
        NativeAnalyzerClient(script).probe()

    assert not marker.exists()
    assert str(tmp_path) not in str(captured.value)


def test_analyze_revalidates_the_process_handshake(tmp_path: Path) -> None:
    launches = tmp_path / "launches"
    valid = {
        "type": "hello",
        "protocol": "cpp-context-clang-facts",
        "protocol_version": 2,
        "analyzer_version": "test",
        "clang_major": 18,
        "capabilities": [
            "direct_calls",
            "full_ast",
            "function_cfg_v1",
            "includes",
            "inherits",
            "lambda_metadata",
            "macro_provenance",
            "occurrences",
            "overrides",
            "pp_callbacks",
            "source_manager",
            "symbols",
            "template_metadata",
            "uses_type",
        ],
    }
    script = _script(
        tmp_path,
        f"""import json, pathlib, sys
counter = pathlib.Path({str(launches)!r})
launch = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(launch))
json.loads(sys.stdin.readline())
hello = {valid!r}
if launch == 2:
    hello["protocol_version"] = 99
print(json.dumps(hello), flush=True)
if launch == 2:
    request = json.loads(sys.stdin.readline())
    print(json.dumps({{"type": "begin", "request_id": request["request_id"]}}))
    print(json.dumps({{"type": "complete", "request_id": request["request_id"],
                      "success": True}}))
""",
    )
    client = NativeAnalyzerClient(script)
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]

    client.probe()
    with pytest.raises(AnalyzerProtocolError, match="protocol mismatch"):
        client.analyze(FIXTURE, configuration)


def test_broken_analyzer_stdin_still_obeys_timeout(tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        "import os, time\nos.close(0)\ntime.sleep(2)\n",
    )
    client = NativeAnalyzerClient(script, timeout_seconds=0.05)
    request = {"payload": "x" * 200_000}

    started = time.monotonic()
    with pytest.raises(AnalyzerLimitError, match="timeout"):
        client._invoke((request,), output_limit=1024)

    assert time.monotonic() - started < 1


def test_timeout_and_output_limits_are_hard(tmp_path: Path) -> None:
    timeout = _script(tmp_path, "import time\ntime.sleep(10)\n")
    with pytest.raises(AnalyzerLimitError, match="timeout"):
        NativeAnalyzerClient(timeout, timeout_seconds=0.05).probe()

    output = _script(tmp_path, "print('x' * 10000)\n")
    with pytest.raises(AnalyzerLimitError, match="output limit"):
        NativeAnalyzerClient(output, max_output_bytes=64).probe()


def test_sqlite_round_trips_macro_provenance(tmp_path: Path) -> None:
    batch = NativeClangIngestor(NativeAnalyzerClient(_binary(), timeout_seconds=15)).ingest(
        FIXTURE, FIXTURE / "compile_commands.json"
    )
    macro = next(symbol for symbol in batch.symbols if symbol.qualified_name == "APPLY_TWICE")
    with SQLiteStore(tmp_path / "index.db", project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, batch)
        occurrence = next(
            item
            for item in store.occurrences(macro.id)
            if item.kind == OccurrenceKind.MACRO_EXPANSION
        )
        assert occurrence.metadata["spelling_span"]
        assert occurrence.metadata["expansion_span"]


def test_doctor_checks_real_companion(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "doctor",
                "--project",
                str(FIXTURE),
                "--compile-commands",
                str(FIXTURE / "compile_commands.json"),
                "--clang-analyzer",
                str(_binary()),
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["clang_analyzer_executable"] is True
    assert report["clang_analyzer_protocol"] == 2
    assert report["clang_analyzer_clang_major"] == 18
    assert report["advanced_facts_complete"] is True
    assert report["cfg_facts_available"] is True
    assert "function_cfg_v1" in report["clang_analyzer_capabilities"]
    assert "macro_provenance" in report["clang_analyzer_capabilities"]


def test_cli_index_reports_companion_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "index",
                str(FIXTURE),
                "--compile-commands",
                str(FIXTURE / "compile_commands.json"),
                "--clang-analyzer",
                str(_binary()),
                "--db",
                str(tmp_path / "index.db"),
                "--embedding-dimensions",
                "32",
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["analysis_backend"] == "clang-libtooling"
    assert report["advanced_facts_complete"] is True
    assert "macro_provenance" in report["analyzer_capabilities"]
