from __future__ import annotations

import importlib.util
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_summary_input import full_facts as full_facts


@pytest.fixture
def driver():
    path = Path(__file__).parents[1] / "tools" / "benchmark_summary_refresh.py"
    spec = importlib.util.spec_from_file_location("summary_benchmark_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_driver_times_committed_transaction_and_rolls_back_timeout(driver) -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

        def solve(project_id, variant, functions):
            assert (project_id, variant, functions) == (1, "default", {"function"})
            connection.execute("INSERT INTO evidence VALUES ('complete')")
            return 1

        store = SimpleNamespace(_connection=connection, _refresh_summary_solutions=solve)
        result = driver.refresh(store, 1, {"function"}, 1.0)
        assert result["summaries_refreshed"] == 1
        assert 0 <= result["seconds"] < 1.0
        assert not connection.in_transaction

        def slow_solve(*_args):
            connection.execute("INSERT INTO evidence VALUES ('must roll back')")
            time.sleep(0.2)

        store._refresh_summary_solutions = slow_solve
        with pytest.raises(TimeoutError, match="exceeded"):
            driver.refresh(store, 1, {"function"}, 0.01)
        assert connection.execute("SELECT * FROM evidence").fetchall() == [("complete",)]
        assert not connection.in_transaction


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_refresh_driver_rejects_disabled_limits(driver, budget) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        driver.refresh(None, 1, {"function"}, budget)


@pytest.mark.parametrize("baseline,budget", [(None, 91), (Path("baseline.json"), 31)])
def test_refresh_driver_keeps_baseline_and_candidate_limits_distinct(driver, baseline, budget):
    with pytest.raises(ValueError, match="at most"):
        driver.run(Path("unused-input"), Path("unused-output"), baseline, budget)


@pytest.fixture
def summary_input(tmp_path):
    from cpp_context_engine.kicad_canary import _SEMANTIC_TABLES, CanaryLimits

    evidence = {
        "schema": "cpp-context-validated-summary-input",
        "schema_version": 1,
        "scope": "full32-facts-and-summaries-only",
        "producer_outcome": "failed",
        "whole_index_success": False,
        "embedding_completeness": "not_validated",
        "database_integrity": "ok",
        "database_artifact_sha256": "a" * 64,
        "producer_pins": {
            "profile": "full",
            "expected_fact_schema_version": 17,
            "engine_commit": "b" * 40,
            "project_commit": "c" * 40,
            "analyzer_sha256": "d" * 64,
            "source_cdb_sha256": "e" * 64,
            "subset_cdb_sha256": "f" * 64,
        },
        "semantic_snapshot": {
            "schema_version": 17,
            "digest": "a" * 64,
            "counts": {
                **dict.fromkeys(_SEMANTIC_TABLES, 0),
                "translation_units": 32,
                "function_summaries": 1,
            },
            "table_digests": dict.fromkeys(_SEMANTIC_TABLES, "b" * 64),
        },
        "database_provenance": {
            "translation_unit_groups": [
                {
                    "analysis_backend": "clang-libtooling",
                    "advanced_facts_complete": 1,
                    "index_profile": "full",
                    "navigation_facts_complete": 1,
                    "cfg_facts_complete": 1,
                    "data_flow_facts_complete": 1,
                    "summary_facts_complete": 1,
                    "translation_units": 32,
                }
            ],
            "build_variants": [{"name": "default", "index_profile": "full"}],
        },
        "summary_orderings": {
            "function": {
                "symbol_id": "function",
                "available": True,
                "analysis_count": 1,
                "digest": "c" * 64,
            }
        },
        "source_database": str(tmp_path / "failed-original" / "index.db"),
        "source_files": {
            "index.db": {"identity": {"device": 0, "inode": 0}, "bytes": 1, "sha256": "d" * 64}
        },
        "producer_evidence_sha256": {
            name: "e" * 64
            for name in ("worker-spec.json", "phase-timings-index.json", "compile_commands.json")
        },
        "guard": {
            "limits": asdict(CanaryLimits(wall_seconds=120)),
            "peak_bytes": {"rss": 1024, "swap": 0, "database": 1024, "disk": 2048, "anonymous": 0},
            "elapsed_seconds": 1.0,
            "processes_clean": True,
            "failure": None,
        },
    }
    # Contract-only fixture: no SQLite is opened, and the failed original need not exist.
    (tmp_path / "index.db").write_bytes(b"validated copy placeholder")
    path = tmp_path / "summary-input.json"
    path.write_text(json.dumps(evidence))
    return path, evidence


def test_refresh_driver_accepts_only_the_published_summary_copy(driver, summary_input):
    path, expected = summary_input
    source, evidence, gate = driver.load_input(path)
    assert source == path.parent / "index.db"
    assert source != Path(expected["source_database"])
    assert evidence == gate == expected
    assert not (path.parent / "SUCCESS").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "candidate-file",
        "schema",
        "scope",
        "whole-index",
        "embedding-claim",
        "producer-pin",
        "partial-tables",
        "partial-coverage",
        "public-empty",
        "public-unavailable",
        "hash",
        "guard-missing",
        "guard-failure",
        "guard-live",
        "guard-deadline",
        "guard-swap",
        "guard-disk",
        "original-path",
        "original-inode",
        "producer-evidence",
    ],
)
def test_refresh_driver_rejects_incomplete_summary_contract_before_sqlite(
    driver,
    summary_input,
    monkeypatch,
    invalid,
):
    path, evidence = summary_input
    if invalid == "candidate-file":
        path = path.with_name("candidate.json")
    elif invalid == "schema":
        evidence["schema_version"] = 2
    elif invalid == "scope":
        evidence["scope"] = "navigation"
    elif invalid == "whole-index":
        evidence["whole_index_success"] = True
    elif invalid == "embedding-claim":
        evidence["embedding_completeness"] = "complete"
    elif invalid == "producer-pin":
        del evidence["producer_pins"]["engine_commit"]
    elif invalid == "partial-tables":
        del evidence["semantic_snapshot"]["table_digests"]["embedding_vectors"]
    elif invalid == "partial-coverage":
        evidence["database_provenance"]["translation_unit_groups"][0]["translation_units"] = 31
    elif invalid == "public-empty":
        evidence["summary_orderings"] = {}
    elif invalid == "public-unavailable":
        evidence["summary_orderings"]["function"]["available"] = False
    elif invalid == "hash":
        evidence["database_artifact_sha256"] = "not a hash"
    elif invalid == "guard-missing":
        del evidence["guard"]
    elif invalid == "guard-failure":
        evidence["guard"]["failure"] = "interrupted"
    elif invalid == "guard-live":
        evidence["guard"]["processes_clean"] = False
    elif invalid == "guard-deadline":
        evidence["guard"]["elapsed_seconds"] = 120
    elif invalid == "guard-swap":
        evidence["guard"]["peak_bytes"]["swap"] = 1
    elif invalid == "guard-disk":
        evidence["guard"]["peak_bytes"]["disk"] = 2**40
    elif invalid == "original-path":
        evidence["source_database"] = str(path.parent / "index.db")
    elif invalid == "producer-evidence":
        del evidence["producer_evidence_sha256"]["worker-spec.json"]
    else:
        state = (path.parent / "index.db").stat()
        evidence["source_files"]["index.db"]["identity"] = {
            "device": state.st_dev,
            "inode": state.st_ino,
        }
    path.write_text(json.dumps(evidence))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached SQLite")

    monkeypatch.setattr(driver.sqlite3, "connect", forbidden)
    output = path.parent / "replay"
    with pytest.raises(ValueError):
        driver.run(path, output, None, 90)
    assert not output.exists()


def test_refresh_driver_keeps_canary_success_requirement(driver, tmp_path):
    path = tmp_path / "report.json"
    gate = {"gate": 32, "translation_units": 32, "selected_sources": [str(i) for i in range(32)]}
    evidence = {
        "schema": "cpp-context-kicad-canary-report",
        "schema_version": 1,
        "profile": "full",
        "gates": [gate],
    }
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="SUCCESS"):
        driver.load_input(path)
    (tmp_path / "gate-32").mkdir()
    (tmp_path / "gate-32" / "SUCCESS").write_text("complete\n")
    assert driver.load_input(path) == (tmp_path / "gate-32" / "index.db", evidence, gate)


@pytest.mark.parametrize("timeout", [False, True])
def test_refresh_driver_marks_real_fixture_phases_without_completing_timeout(
    driver, full_facts, summary_input, tmp_path, monkeypatch, capsys, timeout
):
    from cpp_context_engine import summary_input as validator

    request, _spec, _subset, original = full_facts
    _path, contract = summary_input
    validated = tmp_path / "validated"
    validated.mkdir()
    original_hash = validator._sha256(original)
    evidence = validator.prepare(request, validated)
    # This offline fixture tests replay markers, not the separately tested supervisor.
    evidence["guard"] = contract["guard"]
    manifest = validated / "summary-input.json"
    manifest.write_text(json.dumps(evidence))
    if timeout:

        def expired(*_args):
            raise TimeoutError("fixture refresh deadline")

        monkeypatch.setattr(driver, "refresh", expired)
        with pytest.raises(TimeoutError, match="fixture refresh deadline"):
            driver.run(manifest, tmp_path / "trial", None, 90)
    else:
        report = driver.run(manifest, tmp_path / "trial", None, 90)
        assert report["source_parity"] is True
        assert len(report["after"]["table_digests"]) == 28
    markers = [
        line.split()
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("summary-replay:")
    ]
    expected = ["copy", "validation", "refresh"]
    if not timeout:
        expected.append("aftercheck")
    assert [marker[1] for marker in markers] == expected
    times = [float(marker[2]) for marker in markers]
    assert all(value > 0 for value in times)
    assert times == sorted(times)
    assert validator._sha256(original) == original_hash
