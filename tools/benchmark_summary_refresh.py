"""Replay a retained real KiCad bench32 summary refresh (local only, issue #47).

Run this exact script with baseline or candidate src/ on PYTHONPATH. Requires
the issue #29 canary module in both revisions; never generates synthetic data.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
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

    evidence = json.loads(input_report.read_text())
    gates = [gate for gate in evidence["gates"] if gate["gate"] == 32]
    if (
        evidence.get("schema") != "cpp-context-kicad-canary-report"
        or evidence.get("schema_version") != 1
        or evidence.get("profile") != "full"
        or len(gates) != 1
        or gates[0].get("translation_units") != 32
        or len(set(gates[0].get("selected_sources", ()))) != 32
    ):
        raise ValueError("a successful real 32-source-TU full-profile canary report is required")
    gate = gates[0]
    source = input_report.parent / "gate-32" / "index.db"
    if not (source.parent / "SUCCESS").is_file():
        raise ValueError("retained bench32 gate lacks its SUCCESS marker")
    expected_artifact = gate["database_artifact_sha256"]
    if _database_artifact_digest(source).sha256 != expected_artifact:
        raise ValueError("source database does not match the pinned canary artifact")
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
        raise ValueError("trial facts or schema do not match the retained canary")
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
            raise ValueError("input public summary ordering differs from the retained canary")
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
        "canary_report": str(input_report.resolve()),
        "project_commit": evidence["project_commit"],
        "analyzer": gate["analyzer"],
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
