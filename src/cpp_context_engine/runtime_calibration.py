"""Fixed, empirical NAV phase estimates for the local KiCad canary only."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PHASES = (
    "tu_processing",
    "post_tu_finalization",
    "embeddings",
    "producer_checks",
    "validation",
)
WORK = {
    "tu_processing": "tus",
    "post_tu_finalization": "facts",
    "embeddings": "embeddings",
    "producer_checks": "database_bytes",
    "validation": "database_bytes",
}


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid calibration {name}")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid calibration {name}")
    return float(value)


def _facts(indexing: Mapping[str, Any]) -> float:
    return sum(
        _number(indexing[name], name)
        for name in (
            "indexed_symbols",
            "indexed_occurrences",
            "indexed_edges",
            "indexed_callsites",
            "indexed_call_targets",
        )
    )


def _sample(gate: Mapping[str, Any]) -> dict[str, Any]:
    intervals = (
        *gate["phase_measurements"]["phases"],
        *gate["validation_phase_measurements"]["phases"],
    )
    if tuple(item["name"] for item in intervals) != PHASES:
        raise ValueError("calibration needs exactly five ordered phases")
    times = {}
    previous_end = 0.0
    for interval in intervals:
        if interval["status"] != "complete":
            raise ValueError("calibration phase is incomplete")
        seconds = _number(interval["duration_seconds"], "phase duration")
        start = _number(interval["start_seconds"], "phase start")
        end = _number(interval["end_seconds"], "phase end")
        if start < previous_end or not math.isclose(end - start, seconds, abs_tol=1e-6):
            raise ValueError("calibration phase interval differs")
        previous_end = end
        times[interval["name"]] = seconds
    total = _number(gate["total_elapsed_seconds"], "total seconds", positive=True)
    overhead = total - sum(times.values())
    if overhead < -1e-6 or previous_end > total:
        raise ValueError("calibration phases exceed total time")
    return {
        "times": times,
        "overhead": max(0.0, overhead),
        "total": total,
        "work": {
            "tus": _number(gate["translation_units"], "TU count", positive=True),
            "facts": _facts(gate["indexing"]),
            "embeddings": _number(gate["embedded_symbols"], "embedding records"),
            "database_bytes": _number(gate["peak_database_bytes"], "database bytes"),
        },
    }


def fit_phase_estimate(
    fits: Sequence[Mapping[str, Any]], holdout: Mapping[str, Any], total_tus: int
) -> dict[str, Any]:
    """Use measured upper rates, never a guessed safety factor or a TU-linear tail."""
    if len(fits) != 2:
        raise ValueError("calibration requires two fit sizes")
    samples = [_sample(gate) for gate in fits]
    check = _sample(holdout)
    if not samples[0]["work"]["tus"] < samples[1]["work"]["tus"]:
        raise ValueError("calibration fit sizes must increase")
    _number(total_tus, "whole TU count", positive=True)
    amounts_per_tu = {
        name: max(sample["work"][name] / sample["work"]["tus"] for sample in samples)
        for name in ("facts", "embeddings", "database_bytes")
    }
    rates = {
        phase: max(
            sample["times"][phase] / max(1.0, sample["work"][WORK[phase]]) for sample in samples
        )
        for phase in PHASES
    }
    floors = {
        phase: max(
            (sample["times"][phase] for sample in samples if sample["work"][WORK[phase]] == 0),
            default=0.0,
        )
        for phase in PHASES
    }
    overhead = max(sample["overhead"] for sample in samples)

    def estimate(count: float) -> dict[str, Any]:
        work = {name: math.ceil(rate * count) for name, rate in amounts_per_tu.items()}
        return {
            "phase_seconds": {
                phase: max(
                    floors[phase],
                    rates[phase] * (count if WORK[phase] == "tus" else work[WORK[phase]]),
                )
                for phase in PHASES
            },
            "work_limits": work,
            "overhead_seconds": overhead,
        }

    predicted = estimate(check["work"]["tus"])
    if any(check["work"][name] > limit for name, limit in predicted["work_limits"].items()):
        raise ValueError("calibration holdout workload exceeds prediction")
    if any(check["times"][phase] > predicted["phase_seconds"][phase] + 1e-6 for phase in PHASES):
        raise ValueError("calibration holdout phase exceeds prediction")
    if check["total"] > sum(predicted["phase_seconds"].values()) + overhead + 1e-6:
        raise ValueError("calibration holdout total exceeds prediction")
    return estimate(total_tus)


def projected_total_seconds(
    calibration: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    elapsed: float,
    total_tus: int,
    staged_tus: int,
    database_bytes: int,
) -> float | None:
    """Subtract only observed phase work; reject an exceeded empirical envelope."""
    budgets = {name: _number(calibration["phase_seconds"][name], name) for name in PHASES}
    limits = {
        name: _number(calibration["work_limits"][name], name)
        for name in ("facts", "embeddings", "database_bytes")
    }
    overhead = _number(calibration["overhead_seconds"], "overhead")
    if database_bytes > limits["database_bytes"]:
        raise ValueError("calibration database workload exceeded")
    intervals = {item["name"]: item for snapshot in snapshots for item in snapshot["phases"]}
    remaining = observed = 0.0
    for snapshot in snapshots:
        counts = snapshot["counts"]
        if counts.get("indexing") is not None and _facts(counts["indexing"]) > limits["facts"]:
            raise ValueError("calibration fact workload exceeded")
        if (
            counts.get("embedded_symbols") is not None
            and counts["embedded_symbols"] > limits["embeddings"]
        ):
            raise ValueError("calibration embedding workload exceeded")
    for phase in PHASES:
        interval = intervals.get(phase)
        if interval is None or interval["status"] == "not_started":
            remaining += budgets[phase]
            continue
        complete = interval["status"] == "complete"
        spent = interval["duration_seconds"] if complete else elapsed - interval["start_seconds"]
        spent = _number(spent, "observed phase time")
        if spent > budgets[phase] + 1e-6:
            raise ValueError(f"calibration {phase} time exceeded")
        observed += spent
        if not complete:
            if phase == "tu_processing" and staged_tus == 0:
                return None
            rest = budgets[phase] - spent
            if phase == "tu_processing":
                rest = max(rest, spent * (total_tus - staged_tus) / staged_tus)
            remaining += rest
    # Startup, gaps and cleanup are not zero and are not attributed to a phase.
    observed_overhead = max(0.0, elapsed - observed)
    return elapsed + remaining + max(0.0, overhead - observed_overhead)


def _compatible_revision(repository: Path, producer: str, current: str) -> bool:
    if any(
        not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
        for revision in (producer, current)
    ):
        raise ValueError("calibration requires exact Git revisions")
    if producer == current:
        return True
    changed = subprocess.check_output(
        ["git", "-C", str(repository), "diff", "--name-only", "-z", producer, current],
        text=True,
    ).split("\0")
    # Only documentation may differ. Canary/model/producer changes require new evidence.
    return all(not path or path.startswith("docs/") or path == "README.md" for path in changed)


def load_runtime_calibration(
    path: Path,
    *,
    repository: Path,
    source_rows: Sequence[Mapping[str, Any]],
    source_cdb_sha256: str,
    project_root: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept three successful pinned original-CDB samples, without opening their DBs."""
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema") != "cpp-context-runtime-calibration-v1":
        raise ValueError("unknown runtime calibration schema")
    if bundle.get("source_cdb_sha256") != source_cdb_sha256:
        raise ValueError("calibration source CDB differs")
    descriptors = [*bundle["fits"], bundle["holdout"]]
    if len(bundle["fits"]) != 2:
        raise ValueError("calibration requires two fit reports")

    def cohort(row: Mapping[str, Any]) -> str:
        source = Path(row["file"])
        if not source.is_absolute():
            source = Path(row["directory"]) / source
        try:
            return source.resolve().relative_to(project_root).parts[0]
        except ValueError:
            return "generated"

    groups = [cohort(row) for row in source_rows]
    population = Counter(groups)
    indices: list[set[int]] = []
    gates = []
    evidence = []
    for descriptor in descriptors:
        report_path = (path.parent / descriptor["report"]).resolve()
        report_bytes = report_path.read_bytes()
        if hashlib.sha256(report_bytes).hexdigest() != descriptor["sha256"]:
            raise ValueError("calibration report digest differs")
        report = json.loads(report_bytes)
        if (
            report.get("schema") != "cpp-context-kicad-canary-report"
            or report.get("schema_version") != 1
        ):
            raise ValueError("invalid calibration report schema")
        if not _compatible_revision(repository, report["engine_commit"], expected["engine_commit"]):
            raise ValueError("calibration producer code differs")
        if not isinstance(report.get("project_commit"), str) or not re.fullmatch(
            r"[0-9a-f]{40}", report["project_commit"]
        ):
            raise ValueError("calibration requires an exact project revision")
        for field in ("project_commit", "workers", "embedding_dimensions"):
            if report.get(field) != expected[field]:
                raise ValueError(f"calibration {field} differs")
        if report.get("profile") != "navigation" or len(report["gates"]) != 1:
            raise ValueError("calibration requires a single successful NAV gate")
        gate = report["gates"][0]
        directory = report_path.parent / f"gate-{gate['gate']}"
        worker = json.loads((directory / "worker-spec.json").read_text(encoding="utf-8"))
        for field in ("queries", "generated_source_roots", "workers", "embedding_dimensions"):
            if worker.get(field) != expected[field]:
                raise ValueError(f"calibration worker {field} differs")
        if worker["measurement_provenance"]["engine_commit"] != report["engine_commit"]:
            raise ValueError("calibration worker producer differs")
        if not (directory / "SUCCESS").is_file() or gate.get("process_group_clean") is not True:
            raise ValueError("calibration gate was not successfully published")
        if gate.get("peak_swap_bytes") != 0 or gate.get("database_integrity") != "ok":
            raise ValueError("calibration integrity/resource failure")
        for peak, limit in (
            ("peak_rss_bytes", "rss_bytes"),
            ("peak_database_bytes", "database_bytes"),
            ("peak_disk_bytes", "disk_bytes"),
        ):
            if _number(gate[peak], peak) > _number(gate["limits"][limit], limit, positive=True):
                raise ValueError("calibration sample exceeded a resource cap")
        if _number(gate["total_elapsed_seconds"], "total seconds") > _number(
            gate["total_wall_seconds"], "total deadline", positive=True
        ):
            raise ValueError("calibration sample exceeded its deadline")
        if gate["analyzer"]["sha256"] != expected["analyzer_sha256"]:
            raise ValueError("calibration analyzer differs")
        if gate["semantic_snapshot"]["schema_version"] != expected["schema_version"]:
            raise ValueError("calibration fact schema differs")
        if set(gate["semantic_snapshot"]["counts"]) != set(expected["semantic_tables"]):
            raise ValueError("calibration semantic snapshot is incomplete")
        # NAV includes callsites/targets too; only CFG/DFG/summary tables are deep.
        for table in expected["semantic_tables"][9:26]:
            if (
                table not in {"callsites", "call_targets"}
                and gate["semantic_snapshot"]["counts"][table] != 0
            ):
                raise ValueError("calibration NAV sample contains deep facts")
        for field, value in gate["indexing"].items():
            if (
                field
                not in {
                    "indexed_translation_units",
                    "indexed_symbols",
                    "indexed_occurrences",
                    "indexed_edges",
                    "indexed_callsites",
                    "indexed_call_targets",
                }
                and value != 0
            ):
                raise ValueError("calibration needs fresh NAV-only indexing")
        selected = descriptor["raw_indices"]
        if (
            not selected
            or any(
                type(index) is not int or not 0 <= index < len(source_rows) for index in selected
            )
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("invalid calibration original CDB indices")
        subset_bytes = (directory / "compile_commands.json").read_bytes()
        if hashlib.sha256(subset_bytes).hexdigest() != gate["subset_cdb_sha256"]:
            raise ValueError("calibration subset digest differs")
        if json.loads(subset_bytes) != [source_rows[index] for index in selected]:
            raise ValueError("calibration changed original compile commands")
        # Each measured sample uses all of its own CDB: these report indices
        # are local, unlike the bundle's provenance indices in the whole CDB.
        if (
            gate["gate"] != "all"
            or gate["selection"] != "complete_cdb"
            or gate["raw_cdb_entries"] != len(selected)
            or gate["selected_raw_indices"] != list(range(len(selected)))
        ):
            raise ValueError("calibration selected indices differ from the complete sample")
        if (
            gate["translation_units"] != len(selected)
            or gate["completed_translation_units"] != len(selected)
            or gate["indexing"]["indexed_translation_units"] != len(selected)
        ):
            raise ValueError("calibration did not finish every selected configuration")
        indices.append(set(selected))
        gates.append(gate)
        evidence.append(
            {"report_sha256": descriptor["sha256"], "engine_commit": report["engine_commit"]}
        )
    if not indices[0] < indices[1] or indices[1] & indices[2]:
        raise ValueError("calibration needs nested fit sizes and disjoint holdout")
    for selected in indices[:2]:
        if {groups[index] for index in selected} != population.keys():
            raise ValueError("calibration fit misses a source cohort")
    holdout_groups = {groups[index] for index in indices[2]}
    fitted = Counter(groups[index] for index in indices[1])
    if any(
        group not in holdout_groups and fitted[group] != size for group, size in population.items()
    ):
        raise ValueError("calibration holdout misses an unsampled source cohort")
    estimate = fit_phase_estimate(gates[:2], gates[2], len(source_rows))
    if sum(estimate["phase_seconds"].values()) + estimate["overhead_seconds"] > 5_400:
        raise ValueError("calibration prestart projection exceeds the 90-minute hard limit")
    return {
        **estimate,
        "evidence": evidence,
        "source_cdb_sha256": source_cdb_sha256,
        "empirical": True,
    }
