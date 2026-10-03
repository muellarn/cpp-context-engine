"""Small offline calibration/checkpoint regressions; no real KiCad workload."""

import hashlib
import io
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from cpp_context_engine import kicad_canary
from cpp_context_engine.ingestion.indexer import IndexingResult
from cpp_context_engine.runtime_calibration import (
    PHASES,
    RISK_INDICES,
    _compatible_revision,
    calibration_cohorts,
    fit_phase_estimate,
    load_runtime_calibration,
    projected_total_seconds,
)


@pytest.mark.parametrize(
    "checkpoint,tu_budget,tail_budget,staged,stage,reason",
    [
        (601, 1200, 30, 5, "index", "canary worker failed"),
        (601, 6000, 30, 5, "index", "10-minute projection"),
        (1801, 2200, 500, 5, "index", "30-minute projection"),
        (601, 1200, 30, 0, "index", "projection unknown"),
        (10, 1200, 5, 10, "validation", "validation time exceeded"),
    ],
)
def test_measured_phase_budget_can_pass_unknown_checkpoint(
    tmp_path, monkeypatch, checkpoint, tu_budget, tail_budget, staged, stage, reason
):
    now = 0.0
    events = [{"event": "phase", "name": "index", "monotonic_seconds": 0.0}]
    events.extend(
        {"event": "tu_staged", "completed": index, "monotonic_seconds": 0.0}
        for index in range(1, staged + 1)
    )
    if stage == "validation":
        events = [
            {"event": "phase", "name": "validation", "monotonic_seconds": 0.0},
            {"event": "result", "result": {"validated": True}, "monotonic_seconds": checkpoint},
        ]

    class Process:
        pid = 999999
        stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in events))
        stderr = io.StringIO("")

        def poll(self):
            return 0

        def wait(self, **_kwargs):
            return 0

    samples = 0

    def database_size(_path):
        nonlocal samples, now
        samples += 1
        if samples >= (1 if stage == "validation" else len(events)):
            now = float(checkpoint)
        return 0

    monkeypatch.setattr(kicad_canary.time, "monotonic", lambda: now)
    monkeypatch.setattr(kicad_canary.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(kicad_canary, "_database_size", database_size)
    monkeypatch.setattr(kicad_canary, "_directory_size", lambda _path: 0)
    monkeypatch.setattr(
        kicad_canary, "_process_tree_metrics", lambda _pid: kicad_canary._TreeMetrics(0, 0, 0, ())
    )
    monkeypatch.setattr(kicad_canary, "_terminate_process_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        kicad_canary,
        "_AnalyzerTelemetryMonitor",
        lambda **_kwargs: SimpleNamespace(
            check=lambda: None,
            success_report=lambda: {},
            partial_report=lambda _started: {},
            observe_consumer=lambda *_args: None,
        ),
    )
    spec = {
        "full_project": True,
        "supervision_stage": stage,
        # This is the internal result of validated fit/holdout evidence, not a
        # user-supplied bypass. Loader/provenance tests cover its construction.
        "runtime_calibration": {
            "phase_seconds": {
                "tu_processing": float(tu_budget),
                **{phase: float(tail_budget) for phase in PHASES[1:]},
            },
            "overhead_seconds": 10.0,
            "work_limits": {"facts": 3000, "embeddings": 1000, "database_bytes": 1000000},
        },
    }
    if stage == "validation":
        spec["gate_report"] = {
            "phase_measurements": {
                "phases": [
                    {
                        "name": phase,
                        "status": "complete",
                        "start_seconds": 0,
                        "end_seconds": 0,
                        "duration_seconds": 0,
                    }
                    for phase in PHASES[:4]
                ],
                "counts": {},
            }
        }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    # A calibrated 1,330-second estimate may continue at minute ten. The fake
    # process still has no completion result, so it must not publish SUCCESS.
    with pytest.raises(RuntimeError, match=reason):
        kicad_canary._run_supervised(
            spec_path, tmp_path, kicad_canary.CanaryLimits(wall_seconds=5400), 10
        )


def _gate(tus):
    phases = []
    start = 1.0
    for name, seconds in zip(
        PHASES, (2 * tus, tus, 0.5 * tus, 0.25 * tus, 0.25 * tus), strict=True
    ):
        phases.append(
            {
                "name": name,
                "status": "complete",
                "start_seconds": start,
                "end_seconds": start + seconds,
                "duration_seconds": seconds,
            }
        )
        start += seconds
    return {
        "gate": "all",
        "translation_units": tus,
        "completed_translation_units": tus,
        "raw_cdb_entries": tus,
        "selection": "complete_cdb",
        "selected_raw_indices": list(range(tus)),
        "indexing": asdict(IndexingResult(tus, 0, 0, 4 * tus, 3 * tus, 3 * tus)),
        "embedded_symbols": 4 * tus,
        "phase_measurements": {"phases": phases[:4], "counts": {}},
        "validation_phase_measurements": {"phases": phases[4:], "counts": {}},
        "total_elapsed_seconds": start + 1,
        "peak_database_bytes": 100 * tus,
        "peak_rss_bytes": 100,
        "peak_disk_bytes": 200 * tus,
        "peak_swap_bytes": 0,
        "database_integrity": "ok",
        "process_group_clean": True,
        "limits": {"rss_bytes": 1000, "database_bytes": 10000, "disk_bytes": 10000},
        "total_wall_seconds": 600,
        "analyzer": {"sha256": "a" * 64},
        "semantic_snapshot": {
            "schema_version": kicad_canary.SCHEMA_VERSION,
            "counts": dict.fromkeys(kicad_canary._SEMANTIC_TABLES, 0),
        },
    }


def test_fixed_phase_rates_retain_tail_and_nonphase_overhead():
    fits = [_gate(2), _gate(4)]
    model = fit_phase_estimate(fits, _gate(2), 8, risk=[_gate(1) for _ in range(4)])
    assert model == {
        "phase_seconds": dict(zip(PHASES, (16, 8, 4, 2, 2), strict=True)),
        "overhead_seconds": 10,
        "work_limits": {"facts": 80, "embeddings": 32, "database_bytes": 800},
    }
    assert fit_phase_estimate(fits, _gate(2), 8, risk=[_gate(1) for _ in range(4)]) == model
    # A measured zero-unit phase is still positive work, not a zero-cost tail.
    for gate in [*fits]:
        gate["indexing"].update(indexed_symbols=0, indexed_occurrences=0, indexed_edges=0)
    holdout = _gate(2)
    holdout["indexing"].update(indexed_symbols=0, indexed_occurrences=0, indexed_edges=0)
    assert (
        fit_phase_estimate(fits, holdout, 8, risk=[_gate(1) for _ in range(4)])["phase_seconds"][
            "post_tu_finalization"
        ]
        == 160
    )


def test_risk_pipeline_is_paid_once_not_extrapolated_over_the_population():
    risks = [_gate(4) for _ in range(4)]
    # A fixed heavy cohort is a census of H, not a representative sample of R.
    for risk in risks:
        for snapshot in ("phase_measurements", "validation_phase_measurements"):
            for interval in risk[snapshot]["phases"]:
                for field in ("start_seconds", "end_seconds", "duration_seconds"):
                    interval[field] *= 10
        risk["total_elapsed_seconds"] *= 10
    remainder = [_gate(16), _gate(32)]
    model = fit_phase_estimate(remainder, _gate(15), 2252, risk=risks)
    assert model == {
        "phase_seconds": dict(zip(PHASES, (4792, 22520, 11260, 5630, 5630), strict=True)),
        "overhead_seconds": 82,
        "work_limits": {"facts": 22520, "embeddings": 9008, "database_bytes": 225200},
    }


def test_four_risk_blocks_sum_once_and_retain_the_highest_individual_tail_rate():
    risks = [_gate(4) for _ in range(4)]
    # One expensive H block must not be diluted into the mean H rate.
    for snapshot in ("phase_measurements", "validation_phase_measurements"):
        for interval in risks[0][snapshot]["phases"]:
            for field in ("start_seconds", "end_seconds", "duration_seconds"):
                interval[field] *= 10
    risks[0]["total_elapsed_seconds"] *= 10
    model = fit_phase_estimate([_gate(16), _gate(32)], _gate(15), 2252, risk=risks)
    assert model == {
        "phase_seconds": dict(zip(PHASES, (4576, 22520, 11260, 5630, 5630), strict=True)),
        "overhead_seconds": 28,
        "work_limits": {"facts": 22520, "embeddings": 9008, "database_bytes": 225200},
    }


def test_zero_work_risk_phase_keeps_its_positive_floor():
    risks = [_gate(4) for _ in range(4)]
    fits, holdout = [_gate(16), _gate(32)], _gate(15)
    for gate in [*risks, *fits, holdout]:
        gate["peak_database_bytes"] = 0
    interval = risks[2]["validation_phase_measurements"]["phases"][0]
    interval["end_seconds"] += 50
    interval["duration_seconds"] += 50
    risks[2]["total_elapsed_seconds"] += 50
    model = fit_phase_estimate(fits, holdout, 2252, risk=risks)
    assert model["work_limits"]["database_bytes"] == 0
    assert model["phase_seconds"]["validation"] == 51


def test_remainder_cohorts_are_fixed_nested_uniform_draws_without_replacement():
    cohorts = calibration_cohorts(2252, "a" * 64)
    assert cohorts == calibration_cohorts(2252, "a" * 64)
    assert cohorts["risk"] == [
        [9, 862, 1771, 2093],
        [16, 864, 1998, 2133],
        [110, 1266, 2026, 2155],
        [168, 1366, 2034, 2240],
    ]
    flat_risk = [index for block in cohorts["risk"] for index in block]
    assert sorted(flat_risk) == list(RISK_INDICES)
    risk, small, large, check = map(set, (flat_risk, *cohorts["fits"], cohorts["holdout"]))
    assert (len(risk), len(small), len(large), len(check)) == (16, 16, 32, 15)
    assert small < large
    assert not risk & (large | check)
    assert not large & check
    assert all(indices == sorted(indices) for indices in (*cohorts["fits"], cohorts["holdout"]))
    assert calibration_cohorts(2252, "b" * 64)["fits"] != cohorts["fits"]
    with pytest.raises(ValueError, match="pinned"):
        calibration_cohorts(2251, "a" * 64)


def test_expensive_risk_tail_does_not_rescue_a_failing_remainder_holdout():
    risks, holdout = [_gate(4) for _ in range(4)], _gate(15)
    for gate, extra in ((risks[0], 100), (holdout, 1)):
        interval = gate["validation_phase_measurements"]["phases"][0]
        interval["duration_seconds"] += extra
        interval["end_seconds"] += extra
        gate["total_elapsed_seconds"] += extra
    with pytest.raises(ValueError, match="holdout phase"):
        fit_phase_estimate([_gate(16), _gate(32)], holdout, 2252, risk=risks)


def test_live_mixed_whole_rate_remains_conservative_without_subtracting_risk_time():
    model = fit_phase_estimate([_gate(2), _gate(4)], _gate(2), 8, risk=[_gate(1) for _ in range(4)])
    snapshot = {
        "phases": [{"name": "tu_processing", "status": "incomplete", "start_seconds": 1}],
        "counts": {},
    }
    # A heavy early prefix still dominates the unchanged mixed-Whole safeguard.
    assert (
        projected_total_seconds(
            model, [snapshot], elapsed=11, total_tus=8, staged_tus=1, database_bytes=0
        )
        == 106
    )
    assert (
        projected_total_seconds(
            model, [snapshot], elapsed=11, total_tus=8, staged_tus=0, database_bytes=0
        )
        is None
    )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("work", "workload"),
        ("phase", "phase"),
        ("total", "total"),
        ("missing", "five ordered"),
        ("overlap", "interval"),
        ("incomplete", "incomplete"),
    ],
)
def test_holdout_rejects_underprediction_and_incomplete_intervals(fault, reason):
    holdout = _gate(2)
    if fault == "work":
        holdout["embedded_symbols"] += 1
    elif fault == "phase":
        holdout["validation_phase_measurements"]["phases"][0]["end_seconds"] += 1
        holdout["validation_phase_measurements"]["phases"][0]["duration_seconds"] += 1
        holdout["total_elapsed_seconds"] += 1
    elif fault == "total":
        holdout["total_elapsed_seconds"] += 1
    elif fault == "missing":
        holdout["phase_measurements"]["phases"].pop()
    elif fault == "overlap":
        interval = holdout["phase_measurements"]["phases"][1]
        interval["start_seconds"] -= 1
        interval["end_seconds"] -= 1
    else:
        holdout["phase_measurements"]["phases"][0]["status"] = "incomplete"
    with pytest.raises(ValueError, match=reason):
        fit_phase_estimate([_gate(2), _gate(4)], holdout, 8, risk=[_gate(1) for _ in range(4)])


def test_live_validation_keeps_only_unfinished_cost_and_rejects_excess_work():
    model = fit_phase_estimate([_gate(2), _gate(4)], _gate(2), 8, risk=[_gate(1) for _ in range(4)])
    whole = _gate(8)
    producer = whole["phase_measurements"]
    validation = whole["validation_phase_measurements"]
    validation["phases"][0].update(status="incomplete", end_seconds=None, duration_seconds=None)
    assert (
        projected_total_seconds(
            model, [producer, validation], elapsed=32, total_tus=8, staged_tus=8, database_bytes=800
        )
        == 42
    )
    with pytest.raises(ValueError, match="validation time exceeded"):
        projected_total_seconds(
            model, [producer, validation], elapsed=34, total_tus=8, staged_tus=8, database_bytes=800
        )
    with pytest.raises(ValueError, match="database workload exceeded"):
        projected_total_seconds(
            model, [producer], elapsed=31, total_tus=8, staged_tus=8, database_bytes=801
        )
    producer["counts"]["indexing"] = {
        "indexed_symbols": 81,
        "indexed_occurrences": 0,
        "indexed_edges": 0,
        "indexed_callsites": 0,
        "indexed_call_targets": 0,
    }
    with pytest.raises(ValueError, match="fact workload exceeded"):
        projected_total_seconds(
            model, [producer], elapsed=31, total_tus=8, staged_tus=8, database_bytes=800
        )


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "project"
    rows = [
        {
            "directory": str(root),
            "file": f"common/{index}.cc",
            "arguments": ["clang++", "-c", f"common/{index}.cc"],
        }
        for index in range(2252)
    ]
    expected = {
        "engine_commit": "1" * 40,
        "project_commit": "2" * 40,
        "workers": 2,
        "analyzer_max_decoded_bytes": 268_435_456,
        "analyzer_max_spool_bytes": 1_073_741_824,
        "embedding_dimensions": 32,
        "analyzer_sha256": "a" * 64,
        "schema_version": kicad_canary.SCHEMA_VERSION,
        "semantic_tables": kicad_canary._SEMANTIC_TABLES,
        "queries": ["main"],
        "generated_source_roots": [],
        "source_cdb_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
    }
    cohorts = calibration_cohorts(len(rows), expected["source_cdb_sha256"])
    reports, descriptors = [], []
    for number, indices in enumerate((*cohorts["fits"], cohorts["holdout"], *cohorts["risk"])):
        directory = tmp_path / str(number) / "gate-all"
        directory.mkdir(parents=True)
        (directory / "SUCCESS").write_text("success\n")
        subset = json.dumps([rows[index] for index in indices]).encode()
        (directory / "compile_commands.json").write_bytes(subset)
        worker = {
            field: expected[field]
            for field in (
                "workers",
                "embedding_dimensions",
                "queries",
                "generated_source_roots",
                "analyzer_max_decoded_bytes",
                "analyzer_max_spool_bytes",
            )
        }
        worker["measurement_provenance"] = {
            field: expected[field]
            for field in (
                "engine_commit",
                "analyzer_max_decoded_bytes",
                "analyzer_max_spool_bytes",
            )
        }
        (directory / "worker-spec.json").write_text(json.dumps(worker))
        gate = _gate(len(indices))
        for snapshot in ("phase_measurements", "validation_phase_measurements"):
            for interval in gate[snapshot]["phases"]:
                for field in ("start_seconds", "end_seconds", "duration_seconds"):
                    interval[field] /= 10
        gate["total_elapsed_seconds"] /= 10
        gate["subset_cdb_sha256"] = hashlib.sha256(subset).hexdigest()
        report = {
            "schema": "cpp-context-kicad-canary-report",
            "schema_version": 1,
            **{
                field: expected[field]
                for field in (
                    "engine_commit",
                    "project_commit",
                    "workers",
                    "embedding_dimensions",
                    "analyzer_max_decoded_bytes",
                    "analyzer_max_spool_bytes",
                )
            },
            "profile": "navigation",
            "gates": [gate],
        }
        reports.append(report)
        descriptors.append({"report": f"{number}/report.json", "raw_indices": indices})
    bundle = {
        "schema": "cpp-context-runtime-calibration-v3",
        "source_cdb_sha256": expected["source_cdb_sha256"],
        "risk": descriptors[3:],
        "fits": descriptors[:2],
        "holdout": descriptors[2],
    }

    def load():
        for report, descriptor in zip(reports, descriptors, strict=True):
            data = json.dumps(report).encode()
            (tmp_path / descriptor["report"]).write_bytes(data)
            descriptor["sha256"] = hashlib.sha256(data).hexdigest()
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps(bundle))
        return load_runtime_calibration(
            path,
            repository=tmp_path,
            source_rows=rows,
            source_cdb_sha256=expected["source_cdb_sha256"],
            expected=expected,
        )

    return SimpleNamespace(
        load=load,
        reports=reports,
        bundle=bundle,
        expected=expected,
        root=tmp_path,
        rows=rows,
        project=root,
    )


def test_accept_pinned_nested_fit_and_disjoint_holdout(evidence):
    result = evidence.load()
    assert result["empirical"] is True
    assert len(result["evidence"]) == 7
    assert sum(result["phase_seconds"].values()) + result["overhead_seconds"] == pytest.approx(
        901.8
    )


@pytest.mark.parametrize(
    "block,fault", [(0, "failed"), (1, "validator"), (2, "native"), (3, "cap")]
)
def test_every_risk_block_requires_complete_validated_current_evidence(evidence, block, fault):
    number = block + 3
    gate = evidence.reports[number]["gates"][0]
    if fault == "failed":
        (evidence.root / str(number) / "gate-all" / "SUCCESS").unlink()
        reason = "successfully"
    elif fault == "validator":
        gate["validation_phase_measurements"]["phases"][0]["status"] = "incomplete"
        reason = "incomplete"
    elif fault == "native":
        gate["analyzer"]["sha256"] = "b" * 64
        reason = "analyzer"
    else:
        gate["peak_rss_bytes"] = 1001
        reason = "resource cap"
    with pytest.raises(ValueError, match=reason):
        evidence.load()


@pytest.mark.parametrize("fault", ["single", "missing", "extra", "duplicate", "reordered"])
def test_risk_block_contract_rejects_old_shape_missing_or_duplicate_partition(evidence, fault):
    risk = evidence.bundle["risk"]
    if fault == "single":
        evidence.bundle["risk"] = risk[0]
    elif fault == "missing":
        evidence.bundle["risk"] = risk[:3]
    elif fault == "extra":
        evidence.bundle["risk"] = [*risk, risk[0]]
    elif fault == "duplicate":
        evidence.bundle["risk"] = [risk[0], risk[0], *risk[2:]]
    else:
        evidence.bundle["risk"] = list(reversed(risk))
    with pytest.raises(ValueError, match="four risk reports|fixed risk and uniform"):
        evidence.load()


@pytest.mark.parametrize(
    "fault",
    [
        "failed",
        "incomplete",
        "old_contract",
        "v2_contract",
        "risk_selection",
        "remainder_selection",
    ],
)
def test_risk_and_predetermined_remainder_cannot_be_replaced_with_convenient_evidence(
    evidence, fault
):
    if fault == "failed":
        (evidence.root / "3" / "gate-all" / "SUCCESS").unlink()
        reason = "successfully"
    elif fault == "incomplete":
        evidence.reports[3]["gates"][0]["phase_measurements"]["phases"][1]["status"] = "incomplete"
        reason = "incomplete"
    elif fault in {"old_contract", "v2_contract"}:
        evidence.bundle["schema"] = "cpp-context-runtime-calibration-" + (
            "v1" if fault == "old_contract" else "v2"
        )
        reason = "schema"
    else:
        number = 3 if fault == "risk_selection" else 0
        descriptor = evidence.bundle["risk"][0] if number == 3 else evidence.bundle["fits"][0]
        selected = descriptor["raw_indices"]
        replacement = next(index for index in range(2252) if index not in selected)
        descriptor["raw_indices"] = sorted([replacement, *selected[1:]])
        data = json.dumps([evidence.rows[index] for index in descriptor["raw_indices"]]).encode()
        (evidence.root / str(number) / "gate-all" / "compile_commands.json").write_bytes(data)
        evidence.reports[number]["gates"][0]["subset_cdb_sha256"] = hashlib.sha256(data).hexdigest()
        reason = "fixed risk and uniform"
    with pytest.raises(ValueError, match=reason):
        evidence.load()


def test_r16_mathematical_nogo_precedes_reading_later_reports(evidence, monkeypatch):
    gate = evidence.reports[0]["gates"][0]
    for snapshot in ("phase_measurements", "validation_phase_measurements"):
        for interval in gate[snapshot]["phases"]:
            for field in ("start_seconds", "end_seconds", "duration_seconds"):
                interval[field] *= 20
    gate["total_elapsed_seconds"] *= 20
    original_read = Path.read_bytes

    def read(path):
        assert path != evidence.root / "1" / "report.json", "R32 must not be needed after R16 no-go"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="R16 prestart TU"):
        evidence.load()


@pytest.mark.parametrize("field", ["analyzer_max_decoded_bytes", "analyzer_max_spool_bytes"])
@pytest.mark.parametrize("source", ["report", "worker", "measurement_provenance"])
@pytest.mark.parametrize("missing", [False, True])
def test_reject_mismatched_or_missing_analyzer_budgets(evidence, field, source, missing):
    # The holdout must use the same effective limits as both fits and the target run.
    worker_path = evidence.root / "2" / "gate-all" / "worker-spec.json"
    worker = json.loads(worker_path.read_text())
    values = evidence.reports[2] if source == "report" else worker
    if source == "measurement_provenance":
        values = worker["measurement_provenance"]
    if missing:
        del values[field]
    else:
        values[field] *= 2
    if source != "report":
        worker_path.write_text(json.dumps(worker))
    with pytest.raises(ValueError, match=field):
        evidence.load()


def test_navigation_calls_are_valid_and_part_of_finalization_work(evidence):
    for report in evidence.reports:
        gate = report["gates"][0]
        tus = gate["translation_units"]
        gate["indexing"].update(indexed_callsites=2 * tus, indexed_call_targets=3 * tus)
        gate["semantic_snapshot"]["counts"].update(callsites=2 * tus, call_targets=3 * tus)
    result = evidence.load()
    assert result["work_limits"]["facts"] == 15 * 2252
    assert result["phase_seconds"]["post_tu_finalization"] == pytest.approx(225.2)


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("engine", "Git revisions"),
        ("project", "project revision"),
        ("workers", "workers"),
        ("analyzer", "analyzer"),
        ("schema", "report schema"),
        ("factschema", "fact schema"),
        ("source", "source CDB"),
        ("commands", "compile commands"),
        ("holdout", "disjoint"),
        ("deep", "deep facts"),
        ("skipped", "fresh NAV"),
        ("missing", "successfully"),
        ("incomplete", "every selected"),
        ("local_selection", "selected indices"),
        ("caps", "resource cap"),
        ("snapshot", "snapshot"),
        ("queries", "queries"),
        ("roots", "generated_source_roots"),
        ("prestart", "prestart"),
    ],
)
def test_reject_invalid_calibration_evidence(evidence, fault, reason):
    report = evidence.reports[0]
    gate = report["gates"][0]
    if fault == "engine":
        report["engine_commit"] = evidence.expected["engine_commit"] = None
    elif fault == "project":
        report["project_commit"] = None
    elif fault == "workers":
        report["workers"] += 1
    elif fault == "analyzer":
        gate["analyzer"]["sha256"] = "c" * 64
    elif fault == "schema":
        report["schema_version"] = 2
    elif fault == "factschema":
        gate["semantic_snapshot"]["schema_version"] += 1
    elif fault == "source":
        evidence.bundle["source_cdb_sha256"] = "c" * 64
    elif fault == "commands":
        evidence.bundle["fits"][0]["raw_indices"] = [0, 5]
    elif fault == "holdout":
        evidence.bundle["holdout"]["raw_indices"] = evidence.bundle["fits"][0]["raw_indices"][:-1]
        directory = evidence.root / "2" / "gate-all"
        data = json.dumps(
            [evidence.rows[index] for index in evidence.bundle["holdout"]["raw_indices"]]
        ).encode()
        (directory / "compile_commands.json").write_bytes(data)
        evidence.reports[2]["gates"][0]["subset_cdb_sha256"] = hashlib.sha256(data).hexdigest()
    elif fault == "deep":
        gate["semantic_snapshot"]["counts"]["cfg_graphs"] = 1
    elif fault == "skipped":
        gate["indexing"]["skipped_translation_units"] = 1
    elif fault == "missing":
        (evidence.root / "0" / "gate-all" / "SUCCESS").unlink()
    elif fault == "incomplete":
        gate["completed_translation_units"] -= 1
    elif fault == "local_selection":
        gate["selected_raw_indices"][0] = 99
    elif fault == "caps":
        gate["peak_rss_bytes"] = 1001
    elif fault == "snapshot":
        gate["semantic_snapshot"]["counts"].pop("summary_effects")
    elif fault in {"queries", "roots"}:
        evidence.expected["queries" if fault == "queries" else "generated_source_roots"] = [
            "different"
        ]
    else:
        for report in evidence.reports:
            gate = report["gates"][0]
            for snapshot in ("phase_measurements", "validation_phase_measurements"):
                for interval in gate[snapshot]["phases"]:
                    for field in ("start_seconds", "end_seconds", "duration_seconds"):
                        interval[field] *= 200
            gate["total_elapsed_seconds"] *= 200
            gate["total_wall_seconds"] *= 200
    with pytest.raises(ValueError, match=reason):
        evidence.load()


def test_revision_compatibility_allows_only_documentation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_args, **_kwargs: "docs/result.md\0README.md\0"
    )
    assert _compatible_revision(tmp_path, "1" * 40, "2" * 40)
    for path in (
        "src/cpp_context_engine/runtime_calibration.py",
        "src/cpp_context_engine/kicad_canary.py",
        "native/clang-analyzer/main.cpp",
    ):
        monkeypatch.setattr(
            subprocess, "check_output", lambda *_args, path=path, **_kwargs: path + "\0"
        )
        assert not _compatible_revision(tmp_path, "1" * 40, "2" * 40)


@pytest.mark.parametrize("valid", [True, False])
def test_cli_loads_real_evidence_before_starting_supervisor(evidence, monkeypatch, capsys, valid):
    for row in evidence.rows:
        source = evidence.project / row["file"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("int value;\n")
    cdb = evidence.root / "compile_commands.json"
    cdb.write_text(json.dumps(evidence.rows))
    digest = hashlib.sha256(cdb.read_bytes()).hexdigest()
    evidence.bundle["source_cdb_sha256"] = evidence.expected["source_cdb_sha256"] = digest
    analyzer = evidence.root / "analyzer"
    analyzer.write_text("#!/bin/sh\nexit 99\n")
    analyzer.chmod(0o755)
    digest = hashlib.sha256(analyzer.read_bytes()).hexdigest()
    evidence.expected["analyzer_sha256"] = digest
    for report in evidence.reports:
        report["gates"][0]["analyzer"]["sha256"] = digest
    evidence.load()  # Publish the complete tiny files through the fixture writer.
    if not valid:
        path = evidence.root / "0" / "gate-all" / "worker-spec.json"
        worker = json.loads(path.read_text())
        worker["queries"] = ["other"]
        path.write_text(json.dumps(worker))
    monkeypatch.setattr(
        kicad_canary,
        "_git_revision",
        lambda path: evidence.expected[
            "project_commit" if path == evidence.project else "engine_commit"
        ],
    )
    observed = []

    def supervise(spec_path, _directory, _limits, total):
        spec = json.loads(spec_path.read_text())
        observed.append(spec)
        assert total == 2252
        assert spec["runtime_calibration"]["empirical"] is True
        assert len(spec["runtime_calibration"]["evidence"]) == 7
        assert (
            spec["runtime_calibration"]["source_cdb_sha256"] == evidence.bundle["source_cdb_sha256"]
        )
        raise RuntimeError("supervisor boundary reached; no child started")

    monkeypatch.setattr(kicad_canary, "_run_supervised", supervise)
    assert (
        kicad_canary.main(
            [
                "--project-root",
                str(evidence.project),
                "--compile-commands",
                str(cdb),
                "--clang-analyzer",
                str(analyzer),
                "--output-directory",
                str(evidence.root / "run"),
                "--runtime-calibration",
                str(evidence.root / "calibration.json"),
                "--gates",
                "all",
                "--workers",
                "2",
                "--query",
                "main",
                "--database-limit-mib",
                "10",
                "--disk-limit-mib",
                "20",
            ]
        )
        == 2
    )
    assert len(observed) == int(valid)
    assert (
        "supervisor boundary reached" if valid else "worker queries differs"
    ) in capsys.readouterr().err
    assert not list((evidence.root / "run").rglob("SUCCESS"))
