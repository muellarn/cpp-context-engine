from __future__ import annotations

import json
import os
import stat
import subprocess
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
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.models import GraphRelation, OccurrenceKind
from cpp_context_engine.storage import SQLiteStore

FIXTURE = Path(__file__).parent / "fixtures" / "analyzer_project"
PARITY_FIXTURE = Path(__file__).parent / "fixtures" / "cpp_project"


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
        "protocol_version": 1,
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
    assert expansions
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
    assert report["clang_analyzer_protocol"] == 1
    assert report["clang_analyzer_clang_major"] == 18
    assert report["advanced_facts_complete"] is True
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
