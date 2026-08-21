from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from cpp_context_engine import benchmark as benchmark_module
from cpp_context_engine.benchmark import (
    DEFAULT_BUDGETS,
    REPORT_SCHEMA_VERSION,
    SMOKE_WORKLOAD,
    BenchmarkBudgets,
    WorkloadSpec,
    _measure,
    _process_tree_rss_bytes,
    benchmark_temp_root,
    evaluate_budgets,
    generate_workload,
    percentile,
    validate_report,
)


def test_generator_is_root_independent_and_has_exact_workload_shape(tmp_path: Path) -> None:
    spec = WorkloadSpec(translation_units=3, builds=2, functions_per_translation_unit=4, seed=7)
    first = generate_workload(tmp_path / "first", spec)
    second = generate_workload(tmp_path / "second", spec)

    assert first.source_hash == second.source_hash
    assert len(list((first.project_root / "src").glob("*.cpp"))) == 3
    assert [item.name for item in first.build_variants] == ["build-0", "build-1"]
    assert first.shared_header.is_file()
    for variant in first.build_variants:
        entries = json.loads(variant.compilation_database.read_text(encoding="utf-8"))
        assert len(entries) == 3
        assert all(entry["directory"] == "../.." for entry in entries)
        assert all(not Path(entry["file"]).is_absolute() for entry in entries)
    source = (first.project_root / "src" / "unit_0000.cpp").read_text(encoding="utf-8")
    assert source.count("int function_") == 4
    assert "transform<" in source
    assert "polymorphic.apply" in source
    assert "high_fanout_target" in source
    assert "for (int iteration" in source


def test_generator_rejects_nonempty_or_invalid_destinations(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "owned.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        generate_workload(occupied, WorkloadSpec(1, 1, 1))
    with pytest.raises(ValueError, match="positive"):
        WorkloadSpec(0, 1, 1)
    assert (occupied / "owned.txt").read_text(encoding="utf-8") == "keep"


def test_benchmark_prefers_native_linux_temp_over_inherited_wsl_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    assert benchmark_temp_root() == Path("/tmp")


def test_profiles_and_budgets_match_issue_acceptance() -> None:
    assert WorkloadSpec(100, 2, 40) == SMOKE_WORKLOAD
    assert (
        BenchmarkBudgets(
            cold_index_seconds=180.0,
            peak_rss_bytes=2 * 1024**3,
            warm_noop_seconds=10.0,
            query_p95_seconds=1.0,
            database_bytes=1024**3,
        )
        == DEFAULT_BUDGETS
    )
    measurements = {
        "cold_index_seconds": 179.9,
        "peak_rss_bytes": 2 * 1024**3,
        "warm_noop_seconds": 10.0,
        "database_bytes": 1024**3,
        "queries": {
            "single": {"p95_seconds": 0.5},
            "union": {"p95_seconds": 1.0},
        },
    }
    assert all(evaluate_budgets(measurements).values())
    measurements["queries"]["union"]["p95_seconds"] = 1.01
    assert evaluate_budgets(measurements)["query_p95_seconds"] is False


def test_nearest_rank_percentile_is_stable() -> None:
    values = [float(item) for item in range(1, 51)]
    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.95) == 48.0
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.95)


def test_measurement_captures_process_tree_rss() -> None:
    result, seconds, peak_rss = _measure(lambda: "complete")

    assert result == "complete"
    assert seconds >= 0
    assert peak_rss > 0


def test_measurement_sums_concurrent_child_process_rss() -> None:
    baseline = _process_tree_rss_bytes(os.getpid())

    def allocate_in_children() -> None:
        command = [
            sys.executable,
            "-c",
            "import time; payload = bytearray(24 * 1024 * 1024); time.sleep(0.4)",
        ]
        children = [subprocess.Popen(command) for _ in range(2)]
        for child in children:
            assert child.wait(timeout=2) == 0

    _, _, peak_rss = _measure(allocate_in_children)

    assert peak_rss >= baseline + 32 * 1024**2


def test_report_validation_requires_schema_union_checks_and_no_absolute_paths() -> None:
    report = {
        "schema": "cpp-context-benchmark-report",
        "schema_version": REPORT_SCHEMA_VERSION,
        "generator_version": 1,
        "measured_at_utc": "2026-08-21T00:00:00+00:00",
        "profile": "smoke",
        "workload": {},
        "environment": {},
        "measurements": {},
        "fact_counts": {},
        "budgets": {},
        "budget_results": {},
        "all_budgets_passed": True,
        "bounded_union_checks": {
            "retrieval": True,
            "high_fanout_calls": True,
            "control_flow": True,
            "data_flow": True,
        },
        "indexing": {},
    }
    validate_report(report)

    report["environment"] = {"leak": "/tmp/generated-project"}
    with pytest.raises(ValueError, match="absolute paths"):
        validate_report(report)
    report["environment"] = {}
    report["bounded_union_checks"]["data_flow"] = False
    with pytest.raises(ValueError, match="failed union"):
        validate_report(report)


def test_committed_report_schema_is_valid() -> None:
    schema_path = Path(__file__).parents[1] / "docs" / "benchmarks" / "report-schema-v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_cli_labels_dimension_overrides_as_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def run_benchmark(**arguments: object) -> dict[str, object]:
        captured.update(arguments)
        return {"all_budgets_passed": True}

    monkeypatch.setattr(benchmark_module, "run_benchmark", run_benchmark)
    monkeypatch.setattr(benchmark_module, "validate_report", lambda _report: None)

    assert (
        benchmark_module.main(
            [
                "--clang-analyzer",
                "unused-by-test",
                "--translation-units",
                "2",
            ]
        )
        == 0
    )
    assert captured["profile"] == "custom"
