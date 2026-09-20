from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from cpp_context_engine import summary_input
from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    translation_unit_id,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildVariant,
    CfgBlock,
    CfgBlockRole,
    CfgGraph,
    CodeSymbol,
    DataFlowAnalysis,
    FunctionSummary,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION, SQLiteStore


def test_copy_preserves_hot_journal_and_locks_across_hash_fds(tmp_path, monkeypatch):
    source = tmp_path / "original" / "index.db"
    source.parent.mkdir()
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("CREATE TABLE facts(value BLOB)")
        connection.execute("INSERT INTO facts VALUES (?)", (b"committed" * 10000,))
        connection.commit()
    # A real process exits during an embedding-like transaction, leaving hot journal pages.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
            "c.execute('PRAGMA cache_size=1'); c.execute('BEGIN IMMEDIATE'); "
            "c.execute('UPDATE facts SET value=zeroblob(200000)'); os._exit(0)",
            str(source),
        ],
        check=True,
        timeout=5,
    )
    assert Path(f"{source}-journal").stat().st_size > 0
    before = {path.name: path.read_bytes() for path in source.parent.iterdir()}
    original_hash = summary_input._sha256
    attempted = []

    def hashed(path):
        result = original_hash(path)  # Opens and closes another FD before the writer attempts.
        if path == source:
            with (
                closing(sqlite3.connect(source, timeout=0)) as writer,
                pytest.raises(sqlite3.OperationalError, match="locked"),
            ):
                writer.execute("BEGIN IMMEDIATE")
            attempted.append(True)
        return result

    monkeypatch.setattr(summary_input, "_sha256", hashed)
    destination = tmp_path / "copy"
    destination.mkdir()
    summary_input.copy_file_set(source, destination)
    assert attempted
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
    with closing(sqlite3.connect(destination / "index.db")) as recovered:
        assert recovered.execute("SELECT value FROM facts").fetchone()[0] == b"committed" * 10000
    assert {path.name: path.read_bytes() for path in source.parent.iterdir()} == before


@pytest.mark.parametrize("mode", ["active-writer", "unexpected", "symlink", "mutation"])
def test_copy_rejects_unsafe_sources(tmp_path, monkeypatch, mode):
    source = tmp_path / "index.db"
    with closing(sqlite3.connect(source)) as setup:
        setup.execute("CREATE TABLE facts(value)")
    destination = tmp_path / "copy"
    destination.mkdir()
    with closing(sqlite3.connect(source, timeout=0)) as writer:
        if mode == "active-writer":
            writer.execute("BEGIN IMMEDIATE")
        elif mode == "unexpected":
            Path(f"{source}-surprise").write_bytes(b"unexpected")
        elif mode == "symlink":
            original = source.with_name("backing.db")
            source.rename(original)
            source.symlink_to(original)
        else:
            original_copy = summary_input.shutil.copyfile

            def changed(*args, **kwargs):
                result = original_copy(*args, **kwargs)
                with source.open("ab") as stream:
                    stream.write(b"changed")
                return result

            monkeypatch.setattr(summary_input.shutil, "copyfile", changed)
        with pytest.raises((RuntimeError, OSError), match="writer|sidecar|regular|changed"):
            summary_input.copy_file_set(source, destination)
    assert not (destination / "summary-input.json").exists()


def _phase():
    return {
        "schema": "cpp-context-kicad-phase-timings",
        "schema_version": 1,
        "measurement_provenance": {"profile": "full"},
        "measurements": {
            "counts": {
                "selected_tus": 32,
                "staged_tus": 32,
                "indexing": {"indexed_translation_units": 32, "skipped_translation_units": 0},
            },
            "phases": [
                {"name": "tu_processing", "status": "complete", "end_seconds": 2},
                {"name": "post_tu_finalization", "status": "complete", "end_seconds": 3},
                {"name": "embeddings", "status": "incomplete", "start_seconds": 3},
            ],
        },
    }


def test_eligibility_requires_committed_facts_not_embeddings():
    summary_input.validate_phase(_phase())
    phase = _phase()
    phase["measurements"]["phases"][1]["status"] = "incomplete"
    with pytest.raises(RuntimeError, match="committed"):
        summary_input.validate_phase(phase)
    phase = _phase()
    phase["measurements"]["counts"]["staged_tus"] = 31
    with pytest.raises(RuntimeError, match="32"):
        summary_input.validate_phase(phase)


def test_copy_includes_committed_wal_and_rejects_wal_writer(tmp_path):
    source = tmp_path / "original" / "index.db"
    source.parent.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
            "c.execute('PRAGMA journal_mode=WAL'); c.execute('CREATE TABLE facts(value)'); "
            "c.execute('INSERT INTO facts VALUES (7)'); c.commit(); os._exit(0)",
            str(source),
        ],
        check=True,
        timeout=5,
    )
    before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
    assert set(before) == {"index.db", "index.db-wal", "index.db-shm"}
    copy = tmp_path / "copy"
    copy.mkdir()
    summary_input.copy_file_set(source, copy)
    assert {p.name: p.read_bytes() for p in copy.iterdir()} == before
    with closing(sqlite3.connect(copy / "index.db")) as recovered:
        assert recovered.execute("SELECT value FROM facts").fetchone() == (7,)
    assert {p.name: p.read_bytes() for p in source.parent.iterdir()} == before
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        with pytest.raises(RuntimeError, match="writer"):
            summary_input.copy_file_set(source, blocked)


def test_copy_rejects_wal_without_lockable_sidecars(tmp_path, monkeypatch):
    source = tmp_path / "index.db"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE facts(value)")
    assert not Path(f"{source}-shm").exists()
    original_hash = summary_input._sha256
    attempts = []

    def hashed(path):
        if path == source:
            with closing(sqlite3.connect(source, timeout=0)) as writer:
                try:
                    writer.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError:
                    attempts.append("blocked")
                else:
                    attempts.append("writer started")
        return original_hash(path)

    monkeypatch.setattr(summary_input, "_sha256", hashed)
    output = tmp_path / "copy"
    output.mkdir()
    with pytest.raises(RuntimeError, match="WAL.*sidecars"):
        summary_input.copy_file_set(source, output)
    assert not attempts  # Reject this unprovable lock state before hashing/copying.


def test_guard_counts_anonymous_files_and_cleans_failed_worker(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    children = []

    def launch(*_args, **kwargs):
        child = original_popen(
            [
                sys.executable,
                "-c",
                "import tempfile,time; f=tempfile.TemporaryFile(); "
                "f.write(b'x'*1048576); f.flush(); time.sleep(10)",
            ],
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr(summary_input.subprocess, "Popen", launch)
    output = tmp_path / "guarded"
    with pytest.raises(RuntimeError, match="disk use exceeded"):
        summary_input.run(
            {},
            output,
            summary_input.CanaryLimits(
                wall_seconds=5,
                rss_bytes=256 * 1024**2,
                disk_bytes=65536,
            ),
        )
    report = json.loads((output / "guard.json").read_text())
    assert report["peak_bytes"]["anonymous"] >= 1048576
    assert report["processes_clean"] is True
    assert report["elapsed_seconds"] < 5
    assert all(child.poll() is not None for child in children)
    assert not (output / "summary-input.json").exists()


@pytest.fixture
def full_facts(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    entries = []
    for index in range(32):
        source = root / f"unit{index}.cpp"
        source.write_text("int value() { return 1; }\n")
        entries.append(
            {"file": str(source), "directory": str(root), "arguments": ["c++", "-c", str(source)]}
        )
    cdb = root / "compile_commands.json"
    cdb.write_text(json.dumps(entries))
    header = root / "shared.hpp"
    header.write_text("// dependency\n")
    analyzer = root / "analyzer"
    analyzer.write_text("offline fixture; never executed")
    for command in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *command], check=True, timeout=5)
    configurations = CompilationDatabase.load(cdb, project_root=root).configurations
    units = tuple(
        TranslationUnit(
            translation_unit_id(c),
            c.id,
            c.source_path,
            summary_input._sha256(c.source_path),
            dependencies=((header, summary_input._sha256(header)),),
            analysis_backend="clang-libtooling",
            advanced_facts_complete=True,
            analyzer_identity=summary_input._sha256(analyzer),
        )
        for c in configurations
    )
    c, unit = configurations[0], units[0]
    scope = dict(translation_unit_id=unit.id, build_configuration_id=c.id)
    symbol = CodeSymbol(
        "function",
        "value",
        SymbolKind.FUNCTION,
        SourceSpan(c.source_path, 1, 1),
        source_text="int value() { return 1; }",
        metadata={"is_definition": True},
        **scope,
    )
    graph = CfgGraph("graph", symbol.id, "entry", "exit", **scope)
    blocks = (
        CfgBlock("entry", graph.id, 0, CfgBlockRole.ENTRY, True, **scope),
        CfgBlock("exit", graph.id, 1, CfgBlockRole.NORMAL_EXIT, True, **scope),
    )
    analysis = DataFlowAnalysis("analysis", graph.id, True, (), 1, 32, 16, 8, 100, **scope)
    summary = FunctionSummary(
        "summary",
        symbol.id,
        graph.id,
        analysis.id,
        (),
        (),
        True,
        (),
        True,
        (),
        False,
        1,
        32,
        128,
        1024,
        **scope,
    )
    failed = tmp_path / ".gate-32.failed"
    private = failed / ".index.db.fresh-fixture"
    private.mkdir(parents=True)
    database = private / "index.db"
    batch = IngestionBatch(
        configurations,
        units,
        (symbol,),
        (),
        (),
        cfg_graphs=(graph,),
        cfg_blocks=blocks,
        data_flow_analyses=(analysis,),
        function_summaries=(summary,),
    )
    with SQLiteStore(database) as store:
        store.apply_ingestion(root, batch, build_variant=BuildVariant("default", cdb))
        # A failed fresh generation is still in its private rollback-journal phase.
        store._connection.execute("PRAGMA journal_mode=DELETE")
    subset = failed / "compile_commands.json"
    subset.write_bytes(cdb.read_bytes())
    pins = dict(
        profile="full",
        expected_fact_schema_version=SCHEMA_VERSION,
        project_commit=summary_input._git_revision(root),
        engine_commit=summary_input._git_revision(root),
        analyzer_sha256=summary_input._sha256(analyzer),
        source_cdb_sha256=summary_input._sha256(cdb),
        subset_cdb_sha256=summary_input._sha256(subset),
    )
    spec = dict(
        profile="full",
        translation_units=32,
        project_root=str(root),
        generated_source_roots=[],
        measurement_provenance=pins,
        analyzer=str(analyzer),
    )
    phase = _phase()
    phase["measurement_provenance"] = pins
    (failed / "worker-spec.json").write_text(json.dumps(spec))
    (failed / "phase-timings-index.json").write_text(json.dumps(phase))
    (failed / "FAILURE.json").write_text('{"reason": "embedding timeout"}')
    request = dict(
        failed_directory=str(failed),
        database=str(database),
        source_compilation_database=str(cdb),
        producer_engine=str(root),
    )
    return request, spec, subset, database


def test_full_facts_produce_distinct_input_without_embeddings(full_facts, tmp_path):
    request, _spec, _subset, original = full_facts
    before = summary_input._sha256(original)
    output = tmp_path / "validated"
    result = summary_input.run(
        request,
        output,
        summary_input.CanaryLimits(
            wall_seconds=15,
            rss_bytes=256 * 1024**2,
            database_bytes=8 * 1024**2,
            disk_bytes=16 * 1024**2,
        ),
    )
    assert result["schema"] == "cpp-context-validated-summary-input"
    assert result["whole_index_success"] is False
    assert result["embedding_completeness"] == "not_validated"
    assert result["semantic_snapshot"]["counts"]["embedding_vectors"] == 0
    assert len(result["semantic_snapshot"]["table_digests"]) == 28
    assert result["summary_orderings"]["function"]["available"] is True
    assert result["guard"]["processes_clean"] is True
    assert summary_input._sha256(original) == before
    assert (output / "summary-input.json").is_file()
    assert not (output / "SUCCESS").exists()


@pytest.mark.parametrize(
    "failure",
    [
        "schema",
        "coverage",
        "command",
        "producer",
        "dependency",
        "payload",
        "foreign-key",
        "corruption",
    ],
)
def test_strict_fact_validation_fails_closed(full_facts, tmp_path, failure):
    _request, spec, subset, original = full_facts
    output = tmp_path / "trial"
    output.mkdir()
    summary_input.copy_file_set(original, output)
    trial = output / "index.db"
    if failure == "corruption":
        trial.write_bytes(b"not a SQLite database")
    else:
        with closing(sqlite3.connect(trial)) as connection:
            connection.execute(
                {
                    "schema": "PRAGMA user_version=16",
                    "coverage": "UPDATE translation_units SET summary_facts_complete=0",
                    "command": "UPDATE build_configurations SET command_hash='changed'",
                    "producer": "UPDATE translation_units SET analyzer_identity='changed'",
                    "dependency": "UPDATE dependencies SET content_hash='changed'",
                    "payload": "INSERT INTO summary_solution_payloads VALUES "
                    "(1,'summary','zlib-json-v1',1,0,1,'wrong',X'00')",
                    "foreign-key": "UPDATE cfg_blocks SET graph_id='missing'",
                }[failure]
            )
            connection.commit()
    with pytest.raises((RuntimeError, sqlite3.DatabaseError, ValueError)):
        summary_input._validate_facts(trial, spec, subset)
    assert not (output / "summary-input.json").exists()
