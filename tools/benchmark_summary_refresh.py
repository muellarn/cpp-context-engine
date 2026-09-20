"""Replay a retained real KiCad bench32 summary refresh (local only, issue #47).

Run this exact script with baseline or candidate src/ on PYTHONPATH. Requires
the issue #29 canary module in both revisions; never generates synthetic data.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import resource
import signal
import sqlite3
import time
import zlib
from contextlib import closing
from pathlib import Path
from typing import Any

import cpp_context_engine
from cpp_context_engine.api import FlowRequest
from cpp_context_engine.api.analysis import AnalysisQueryService
from cpp_context_engine.models import BuildScope
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION, SQLiteStore


def refresh(store: SQLiteStore, project_id: int, functions: set[str], seconds: float) -> dict:
    """Time solve, persistence and commit; roll back when the hard budget expires."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("refresh budget must be finite and positive")

    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"summary refresh exceeded {seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    started = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        with store._connection:
            store._connection.execute("BEGIN IMMEDIATE")
            count = store._refresh_summary_solutions(project_id, "default", functions)
        elapsed = time.perf_counter() - started
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return {"seconds": elapsed, "summaries_refreshed": count, "budget_seconds": seconds}


def load_input(input_report: Path) -> tuple[Path, dict, dict]:
    """Accept a successful canary or the distinct, guarded #82 input contract."""
    from cpp_context_engine.kicad_canary import (
        _SEMANTIC_TABLES,
        CanaryLimits,
        _validate_profile_provenance,
    )
    from cpp_context_engine.models import IndexProfile

    evidence = json.loads(input_report.read_text())
    if not isinstance(evidence, dict):
        raise ValueError("summary input report must be an object")
    if evidence.get("schema") == "cpp-context-validated-summary-input":
        try:
            if (
                input_report.name != "summary-input.json"
                or evidence["schema_version"] != 1
                or evidence["scope"] != "full32-facts-and-summaries-only"
                or evidence["producer_outcome"] != "failed"
                or evidence["whole_index_success"] is not False
                or evidence["embedding_completeness"] != "not_validated"
                or evidence["database_integrity"] != "ok"
            ):
                raise ValueError("a published validated summary input is required")
            pins = evidence["producer_pins"]
            if pins["profile"] != "full" or pins["expected_fact_schema_version"] != SCHEMA_VERSION:
                raise ValueError("summary input producer profile/schema mismatch")
            for name in (
                "engine_commit",
                "project_commit",
                "analyzer_sha256",
                "source_cdb_sha256",
                "subset_cdb_sha256",
            ):
                size = 40 if name.endswith("commit") else 64
                if not re.fullmatch(rf"[0-9a-f]{{{size}}}", pins[name]):
                    raise ValueError("summary input producer pin is malformed")
            snapshot = evidence["semantic_snapshot"]
            if (
                snapshot["schema_version"] != SCHEMA_VERSION
                or set(snapshot["table_digests"]) != set(_SEMANTIC_TABLES)
                or set(snapshot["counts"]) != set(_SEMANTIC_TABLES)
                or any(type(count) is not int or count < 0 for count in snapshot["counts"].values())
                or snapshot["counts"]["translation_units"] != 32
                or snapshot["counts"]["function_summaries"] < 1
            ):
                raise ValueError("summary input must retain all full32 semantic tables")
            hashes = [
                evidence["database_artifact_sha256"],
                snapshot["digest"],
                *snapshot["table_digests"].values(),
            ]
            evidence_hashes = evidence["producer_evidence_sha256"]
            if len(evidence_hashes) != 3 or {Path(p).name for p in evidence_hashes} != {
                "worker-spec.json",
                "phase-timings-index.json",
                "compile_commands.json",
            }:
                raise ValueError("summary input producer evidence is incomplete")
            hashes.extend(evidence_hashes.values())
            original = evidence["source_files"]["index.db"]
            hashes.append(original["sha256"])
            if type(original["bytes"]) is not int or original["bytes"] < 1:
                raise ValueError("summary input original-file provenance is incomplete")
            orderings = evidence["summary_orderings"]
            if not 1 <= len(orderings) <= 8:
                raise ValueError("summary input public selections are missing")
            for symbol_id, item in orderings.items():
                if (
                    not symbol_id
                    or item["symbol_id"] != symbol_id
                    or item["available"] is not True
                    or type(item["analysis_count"]) is not int
                    or item["analysis_count"] < 1
                ):
                    raise ValueError("summary input public selection is incomplete")
                hashes.append(item["digest"])
            if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes):
                raise ValueError("summary input semantic hash is malformed")
            _validate_profile_provenance(evidence["database_provenance"], IndexProfile.FULL, 32)
            guard = evidence["guard"]
            limits = CanaryLimits(**guard["limits"])
            peaks = guard["peak_bytes"]
            if (
                guard["processes_clean"] is not True
                or guard["failure"] is not None
                or not math.isfinite(guard["elapsed_seconds"])
                or not 0 < guard["elapsed_seconds"] < limits.wall_seconds
                or any(
                    type(peaks[name]) is not int or peaks[name] < 0
                    for name in ("rss", "swap", "database", "disk", "anonymous")
                )
                or peaks["anonymous"] > peaks["disk"]
                or limits.violation(
                    elapsed=guard["elapsed_seconds"],
                    **{key: peaks[key] for key in ("rss", "swap", "database", "disk")},
                )
            ):
                raise ValueError("summary input resource guard did not complete safely")
            source = input_report.parent / "index.db"
            identity = original["identity"]
            if any(type(identity[k]) is not int or identity[k] < 0 for k in ("device", "inode")):
                raise ValueError("summary input original identity is malformed")
            state = source.stat()
            # Never follow the failed producer path or replay its original inode.
            if (
                source.is_symlink()
                or source.resolve() == Path(evidence["source_database"]).resolve()
                or (state.st_dev, state.st_ino) == (identity["device"], identity["inode"])
            ):
                raise ValueError("summary replay requires the independently validated copy")
        except (KeyError, TypeError, AttributeError, RuntimeError) as error:
            raise ValueError("incomplete or malformed validated summary input") from error
        return source, evidence, evidence

    gates = [gate for gate in evidence.get("gates", ()) if gate["gate"] == 32]
    if (
        evidence.get("schema") != "cpp-context-kicad-canary-report"
        or evidence.get("schema_version") != 1
        or evidence.get("profile") != "full"
        or len(gates) != 1
        or gates[0].get("translation_units") != 32
        or len(set(gates[0].get("selected_sources", ()))) != 32
    ):
        raise ValueError("a successful real 32-source-TU full-profile canary report is required")
    source = input_report.parent / "gate-32" / "index.db"
    if not (source.parent / "SUCCESS").is_file():
        raise ValueError("retained bench32 gate lacks its SUCCESS marker")
    return source, evidence, gates[0]


def run(input_report: Path, output: Path, baseline: Path | None, seconds: float) -> dict:
    role = "candidate" if baseline is not None else "baseline"
    limit = 30 if baseline is not None else 90
    if not math.isfinite(seconds) or not 0 < seconds <= limit:
        raise ValueError(f"{role} refresh budget must be positive and at most {limit} seconds")
    # Reuse the canary's reviewed artifact, coverage and exact SQL/BLOB hashing.
    from cpp_context_engine.kicad_canary import (
        _database_artifact_digest,
        _database_size,
        _git_revision,
        _public_summary_ordering,
        _validate_database_integrity,
        _validate_profile_provenance,
        database_provenance,
        semantic_snapshot,
    )
    from cpp_context_engine.models import IndexProfile

    source, evidence, gate = load_input(input_report)
    expected_artifact = gate["database_artifact_sha256"]
    if _database_artifact_digest(source).sha256 != expected_artifact:
        raise ValueError("source database does not match the pinned input artifact")
    output.mkdir(parents=True, exist_ok=False)
    trial = output / "index.db"
    with (
        closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as origin,
        closing(sqlite3.connect(trial)) as destination,
    ):
        origin.backup(destination)
    if _database_artifact_digest(source).sha256 != expected_artifact:
        raise ValueError("source database changed during the backup")
    before = semantic_snapshot(trial)
    if before != gate["semantic_snapshot"] or before["schema_version"] != SCHEMA_VERSION:
        raise ValueError("trial facts or schema do not match the retained input")
    provenance = database_provenance(trial)
    _validate_profile_provenance(provenance, IndexProfile.FULL, 32)
    expected = json.loads(baseline.read_text()) if baseline else None
    if expected is not None and (
        expected.get("schema") != "cpp-context-summary-refresh-v1"
        or expected.get("role") != "baseline"
        or expected.get("source_parity") is not True
        or expected.get("source_artifact_sha256") != expected_artifact
        or expected.get("before") != before
    ):
        raise ValueError("a successful baseline measured on the same pinned input is required")

    with SQLiteStore(trial) as store:
        projects = store._connection.execute("SELECT id, root FROM projects").fetchall()
        if len(projects) != 1:
            raise ValueError("bench32 must contain exactly one project")
        project_id = projects[0][0]
        functions = {
            row[0]
            for row in store._connection.execute(
                "SELECT DISTINCT function_symbol_id FROM function_summaries "
                "WHERE project_id = ? AND build_variant = 'default'",
                (project_id,),
            )
        }
        if not functions:
            raise ValueError("bench32 has no native function summaries")
        service = AnalysisQueryService(store, Path(projects[0][1]), BuildScope.single())

        def public_ordering() -> dict:
            return {
                query: _public_summary_ordering(
                    service.data_flow(
                        FlowRequest(function_symbol_id=item["symbol_id"], builds=["default"])
                    ),
                    required=True,
                )
                for query, item in gate["summary_orderings"].items()
            }

        if not gate["summary_orderings"] or public_ordering() != gate["summary_orderings"]:
            raise ValueError("input public summary ordering differs from the retained input")
        print("Input verified; starting isolated solve + persist + commit", flush=True)
        measurement = refresh(store, project_id, functions, seconds)
        if measurement["seconds"] >= seconds:
            raise TimeoutError(f"summary refresh did not finish below {seconds:g} seconds")
        integrity = _validate_database_integrity(store._connection)
        summary_orderings = public_ordering()
    after = semantic_snapshot(trial)
    report: dict[str, Any] = {
        "schema": "cpp-context-summary-refresh-v1",
        "role": role,
        "engine_commit": _git_revision(Path(cpp_context_engine.__file__).resolve().parents[2]),
        "python": platform.python_version(),
        "zlib": zlib.ZLIB_RUNTIME_VERSION,
        "input_report": str(input_report.resolve()),
        "input_schema": evidence["schema"],
        **(
            {
                "project_commit": evidence["producer_pins"]["project_commit"],
                "analyzer": {"sha256": evidence["producer_pins"]["analyzer_sha256"]},
                "producer_pins": evidence["producer_pins"],
                "producer_evidence_sha256": evidence["producer_evidence_sha256"],
                "whole_index_success": False,
            }
            if evidence["schema"] == "cpp-context-validated-summary-input"
            else {
                "canary_report": str(input_report.resolve()),
                "project_commit": evidence["project_commit"],
                "analyzer": gate["analyzer"],
            }
        ),
        "source_artifact_sha256": expected_artifact,
        "provenance": provenance,
        "measurement": measurement,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "database_bytes": _database_size(trial),
        "database_integrity": integrity,
        "before": before,
        "after": after,
        "summary_orderings": summary_orderings,
        "source_parity": after == before and summary_orderings == gate["summary_orderings"],
        "baseline_parity": (
            None
            if expected is None
            else after == expected["after"] and summary_orderings == expected["summary_orderings"]
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["source_parity"] or report["baseline_parity"] is False:
        raise ValueError("summary refresh changed exact fact/payload/order/solution hashes")
    print(
        json.dumps(
            {
                **measurement,
                "role": role,
                "source_parity": True,
                "baseline_parity": baseline is not None,
            }
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--max-refresh-seconds", type=float, default=30.0)
    args = parser.parse_args()
    run(args.input_report, args.output_directory, args.baseline_report, args.max_refresh_seconds)


if __name__ == "__main__":
    main()
