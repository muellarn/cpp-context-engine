"""Local benchmark-only validation of embedding-independent full32 summary inputs.

This does not publish an index or change the failed producer's outcome. SQLite
may recover only a newly owned byte copy, never the original database.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import sqlite3
import stat
import struct
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from cpp_context_engine.analysis.interprocedural import _solution_hash
from cpp_context_engine.api import FlowRequest
from cpp_context_engine.api.analysis import AnalysisQueryService
from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    translation_unit_id,
)
from cpp_context_engine.kicad_canary import (
    _SEMANTIC_TABLES,
    CanaryLimits,
    _database_artifact_digest,
    _database_size,
    _directory_size,
    _file_identity,
    _git_revision,
    _load_raw_cdb,
    _process_identity_live,
    _process_tree_metrics,
    _public_summary_ordering,
    _remember_process_tree,
    _sha256,
    _terminate_process_group,
    _validate_database_integrity,
    _validate_profile_provenance,
    database_provenance,
    inspect_compilation_database,
    select_gate_entries,
    semantic_snapshot,
)
from cpp_context_engine.models import BuildScope, IndexProfile
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION, SQLiteStore


def _file_set(database: Path) -> tuple[Path, ...]:
    paths = (database, *sorted(database.parent.glob(database.name + "-*")))
    allowed = {database, *(Path(f"{database}{s}") for s in ("-journal", "-wal", "-shm"))}
    if set(paths) - allowed:
        raise RuntimeError("unexpected SQLite sidecar")
    for path in paths:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError("source must contain only regular files, not symlinks")
    if Path(f"{database}-shm") in paths and Path(f"{database}-wal") not in paths:
        raise RuntimeError("SQLite shared memory has no WAL sidecar")
    return paths


def copy_file_set(database: Path, destination: Path) -> dict:
    """Hold SQLite-compatible locks throughout hashing and copying, without SQLite."""
    if sys.platform != "linux" or not hasattr(fcntl, "F_OFD_SETLK"):
        raise RuntimeError("Linux OFD locks are required for source writer exclusion")
    paths = _file_set(database)
    with contextlib.ExitStack() as stack:
        for path in paths:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            try:
                # Unlike process-owned locks, OFD locks survive other hashing FDs closing.
                # A whole-file read lock conflicts with SQLite's POSIX writer byte locks.
                fcntl.fcntl(fd, fcntl.F_OFD_SETLK, struct.pack("hhqqi", fcntl.F_RDLCK, 0, 0, 0, 0))
            except OSError as error:
                raise RuntimeError("source database or sidecar has an active writer") from error
            if os.fstat(fd).st_ino != path.stat().st_ino:
                raise RuntimeError("source file changed while acquiring writer exclusion")
            # WAL writers lock SHM, not the database's rollback-journal byte range.
            # Missing sidecars could be created after this lock set was captured.
            if (
                path == database
                and 2 in os.pread(fd, 2, 18)
                and any(Path(f"{database}{s}") not in paths for s in ("-wal", "-shm"))
            ):
                raise RuntimeError("WAL source requires existing lockable WAL/SHM sidecars")
        before = {path.name: (_file_identity(path), _sha256(path)) for path in paths}
        if _file_set(database) != paths:
            raise RuntimeError("source file set changed before copy")
        for path in paths:
            target = destination / path.name
            with target.open("xb"):
                pass
            shutil.copyfile(path, target)
            if _sha256(target) != before[path.name][1]:
                raise RuntimeError("copied database file differs from source")
        after = {path.name: (_file_identity(path), _sha256(path)) for path in paths}
        if before != after or _file_set(database) != paths:
            raise RuntimeError("source files changed during copy")
    return {
        name: {"identity": asdict(identity), "bytes": identity.size, "sha256": digest}
        for name, (identity, digest) in before.items()
    }


def validate_phase(evidence: dict) -> None:
    if (evidence.get("schema"), evidence.get("schema_version")) != (
        "cpp-context-kicad-phase-timings",
        1,
    ) or evidence.get("measurement_provenance", {}).get("profile") != "full":
        raise RuntimeError("full-profile producer phase evidence is required")
    measurements = evidence["measurements"]
    counts = measurements["counts"]
    if (
        counts.get("selected_tus"),
        counts.get("staged_tus"),
        counts.get("indexing", {}).get("indexed_translation_units"),
        counts.get("indexing", {}).get("skipped_translation_units"),
    ) != (32, 32, 32, 0):
        raise RuntimeError("all 32 full-profile translation units must have completed")
    phases = {phase["name"]: phase for phase in measurements["phases"]}
    if any(
        phases.get(name, {}).get("status") != "complete"
        for name in ("tu_processing", "post_tu_finalization")
    ):
        raise RuntimeError("producer facts were not demonstrably committed")
    end = phases["post_tu_finalization"].get("end_seconds")
    if end is None or phases.get("embeddings", {}).get("start_seconds") != end:
        raise RuntimeError("missing post-commit embedding transition")


def _pins(spec: dict, source_cdb: Path, subset: Path, producer: Path) -> dict:
    pins = spec["measurement_provenance"]
    root = Path(spec["project_root"])
    for field in (
        "project_commit",
        "engine_commit",
        "analyzer_sha256",
        "source_cdb_sha256",
        "subset_cdb_sha256",
    ):
        length = 40 if field.endswith("commit") else 64
        if not re.fullmatch(f"[0-9a-f]{{{length}}}", str(pins.get(field))):
            raise RuntimeError(f"missing or malformed producer pin: {field}")
    if (spec["profile"], spec["translation_units"], pins["expected_fact_schema_version"]) != (
        "full",
        32,
        SCHEMA_VERSION,
    ):
        raise RuntimeError("producer full32 profile/schema does not match")
    if (
        _git_revision(root) != pins["project_commit"]
        or _git_revision(producer) != pins["engine_commit"]
        or _sha256(Path(spec["analyzer"])) != pins["analyzer_sha256"]
        or _sha256(source_cdb) != pins["source_cdb_sha256"]
        or _sha256(subset) != pins["subset_cdb_sha256"]
    ):
        raise RuntimeError("producer/source/analyzer/compilation database pin mismatch")
    if subprocess.check_output(
        ["git", "-C", str(producer), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise RuntimeError("producer engine checkout is not clean")
    inspection = inspect_compilation_database(root, source_cdb)
    raw = _load_raw_cdb(source_cdb)
    selected = select_gate_entries(inspection, 32)
    if _load_raw_cdb(subset) != [raw[entry.raw_index] for entry in selected]:
        raise RuntimeError("retained commands differ from original full32 workload")
    return pins


def _validate_facts(database: Path, spec: dict, subset: Path) -> dict:
    root = Path(spec["project_root"])
    normalized = CompilationDatabase.load(
        subset,
        project_root=root,
        generated_source_roots=tuple(Path(p) for p in spec["generated_source_roots"]),
    )
    configurations = {
        configuration.id: configuration for configuration in normalized.configurations
    }
    if len(configurations) != 32 or len({c.source_path for c in configurations.values()}) != 32:
        raise RuntimeError("summary input requires exactly 32 unique full-profile source TUs")
    # Only this private copy is opened. The first read permits ordinary SQLite rollback.
    with contextlib.closing(sqlite3.connect(database, timeout=0)) as connection:
        if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise RuntimeError("summary input schema mismatch; migration is forbidden")
        integrity = _validate_database_integrity(connection)
        snapshot = semantic_snapshot(database, _connection=connection)
        if set(snapshot["table_digests"]) != set(_SEMANTIC_TABLES):
            raise RuntimeError("summary input is missing semantic tables")
        provenance = database_provenance(database, _connection=connection)
        _validate_profile_provenance(provenance, IndexProfile.FULL, 32)
        projects = connection.execute("SELECT id, root FROM projects").fetchall()
        if len(projects) != 1 or projects[0][1] != str(root.resolve()):
            raise RuntimeError("summary input project provenance mismatch")
        project_id = projects[0][0]
        actual = connection.execute(
            "SELECT id, source_path, directory, arguments_json, command_hash, output, "
            "build_variant "
            "FROM build_configurations ORDER BY id"
        ).fetchall()
        expected = [
            (
                c.id,
                str(c.source_path),
                str(c.directory),
                list(c.arguments),
                c.command_hash,
                str(c.output) if c.output else None,
                c.build_variant,
            )
            for c in sorted(configurations.values(), key=lambda c: c.id)
        ]
        if [(r[0], r[1], r[2], json.loads(r[3]), *r[4:]) for r in actual] != expected:
            raise RuntimeError("summary input command/configuration mismatch")
        units = connection.execute(
            "SELECT id, build_configuration_id, source_path, content_hash, analyzer_identity "
            "FROM translation_units ORDER BY id"
        ).fetchall()
        expected_units = sorted(
            (
                translation_unit_id(c),
                c.id,
                str(c.source_path),
                _sha256(c.source_path),
                spec["measurement_provenance"]["analyzer_sha256"],
            )
            for c in configurations.values()
        )
        if units != expected_units:
            raise RuntimeError("summary input TU/source/producer identity mismatch")
        dependencies = {}
        for path, digest in connection.execute(
            "SELECT DISTINCT path, content_hash FROM dependencies"
        ):
            if dependencies.setdefault(path, digest) != digest or _sha256(Path(path)) != digest:
                raise RuntimeError("summary input dependency pin mismatch")
        if connection.execute(
            "SELECT count(*) FROM build_variants WHERE reindex_required != 0"
        ).fetchone()[0]:
            raise RuntimeError("summary input still requires reindexing")
    # Exact current schema was checked first: this constructor cannot migrate the copy.
    with SQLiteStore(database) as store:
        for row in store._connection.execute("SELECT * FROM function_summaries"):
            summary = store._row_to_function_summary(row)
            propagated_effects, propagated_origins = store._summary_solution_payload(
                project_id, summary.id, ("default",)
            )
            effects = {
                item.id: item
                for item in (
                    store._row_to_summary_effect(local)
                    for local in store._connection.execute(
                        "SELECT * FROM summary_effects WHERE summary_id=? AND is_local=1",
                        (summary.id,),
                    )
                )
            }
            origins = {
                item.id: item
                for item in (
                    store._row_to_summary_return_origin(local)
                    for local in store._connection.execute(
                        "SELECT * FROM summary_return_origins WHERE summary_id=? AND is_local=1",
                        (summary.id,),
                    )
                )
            }
            effects.update((item.id, item) for item in propagated_effects)
            origins.update((item.id, item) for item in propagated_origins)
            if (
                _solution_hash(
                    tuple(sorted(effects.values(), key=lambda item: item.id))[
                        : summary.max_summary_effects
                    ],
                    tuple(sorted(origins.values(), key=lambda item: item.id)),
                    summary.incomplete_reasons,
                )
                != summary.solution_hash
            ):
                raise RuntimeError("summary solution hash does not match persisted facts/payload")
        selected = store._connection.execute(
            "SELECT DISTINCT s.function_symbol_id FROM function_summaries s JOIN symbols y "
            "ON y.project_id=s.project_id AND y.id=s.function_symbol_id "
            "ORDER BY y.qualified_name, s.function_symbol_id LIMIT 8"
        ).fetchall()
        if not selected:
            raise RuntimeError("summary input has no native function summaries")
        service = AnalysisQueryService(store, root, BuildScope.single())
        orderings = {
            symbol_id: _public_summary_ordering(
                service.data_flow(FlowRequest(function_symbol_id=symbol_id, builds=["default"])),
                required=True,
            )
            for (symbol_id,) in selected
        }
        if semantic_snapshot(database, _connection=store._connection) != snapshot:
            raise RuntimeError("summary validation changed the copied facts")
    input_files = {**dependencies, **{row[2]: row[3] for row in units}}
    if any(_sha256(Path(path)) != digest for path, digest in input_files.items()):
        raise RuntimeError("source/dependency files changed during validation")
    return {
        "semantic_snapshot": snapshot,
        "database_provenance": provenance,
        "database_integrity": integrity,
        "summary_orderings": orderings,
        "dependency_files": len(dependencies),
    }


def prepare(request: dict, output: Path) -> dict:
    """Internal worker operation. The CLI supervises the complete operation, not just SQL."""
    failed = Path(request["failed_directory"])
    original = Path(request["database"])
    spec_path = failed / "worker-spec.json"
    phase_path = failed / "phase-timings-index.json"
    spec = json.loads(spec_path.read_text())
    phase = json.loads(phase_path.read_text())
    validate_phase(phase)
    if spec["measurement_provenance"] != phase["measurement_provenance"]:
        raise RuntimeError("producer phase/spec provenance mismatch")
    if not (failed / "FAILURE.json").is_file() or (failed / "SUCCESS").exists():
        raise RuntimeError("expected a retained failed producer, never a relabeled success")
    if (
        original.parent.parent != failed
        or not original.parent.name.startswith(".index.db.fresh-")
        or original.name != "index.db"
    ):
        raise RuntimeError("expected the failed producer's private fresh generation")
    source_cdb = Path(request["source_compilation_database"])
    subset = failed / "compile_commands.json"
    producer = Path(request["producer_engine"])
    evidence_hashes = {str(p): _sha256(p) for p in (spec_path, phase_path, subset)}
    pins = _pins(spec, source_cdb, subset, producer)
    source_files = copy_file_set(original, output)
    database = output / original.name
    validation = _validate_facts(database, spec, subset)
    if _pins(spec, source_cdb, subset, producer) != pins or any(
        _sha256(Path(path)) != digest for path, digest in evidence_hashes.items()
    ):
        raise RuntimeError("summary input provenance changed during validation")
    if {
        p.name: {
            "identity": asdict(_file_identity(p)),
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
        }
        for p in _file_set(original)
    } != source_files:
        raise RuntimeError("original files changed during validation")
    return {
        "schema": "cpp-context-validated-summary-input",
        "schema_version": 1,
        "scope": "full32-facts-and-summaries-only",
        "producer_outcome": "failed",
        "whole_index_success": False,
        "embedding_completeness": "not_validated",
        "producer_pins": pins,
        "source_files": source_files,
        "source_database": str(original),
        "producer_evidence_sha256": evidence_hashes,
        "database_artifact_sha256": _database_artifact_digest(database).sha256,
        **validation,
    }


def _anonymous_bytes(pids: tuple[int, ...]) -> int:
    files = {}
    for pid in pids:
        for fd in Path(f"/proc/{pid}/fd").glob("*"):
            try:
                metadata = fd.stat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 0:
                files[metadata.st_dev, metadata.st_ino] = metadata.st_size
    return sum(files.values())


def run(request: dict, output: Path, limits: CanaryLimits) -> dict:
    """Use the canary's process identities, budget policy and cleanup for one local worker."""
    if sys.platform != "linux" or limits.wall_seconds <= 3:
        raise ValueError("Linux supervision and more than three seconds are required")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request))
    started = time.monotonic()
    groups, processes = set(), set()
    peaks = dict(rss=0, swap=0, database=0, disk=0, anonymous=0)
    process = None
    failure = None
    try:
        with (output / "worker.stderr").open("x") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", __spec__.name, "--_worker", str(request_path)],
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                start_new_session=True,
            )
            while True:
                tree = _process_tree_metrics(process.pid)
                _remember_process_tree(groups, processes, tree)
                # Include the supervisor and unlinked SQLite sort/temp files in the budget.
                budget_tree = _process_tree_metrics(os.getpid())
                anonymous = _anonymous_bytes(budget_tree.live_pids)
                values = dict(
                    rss=budget_tree.rss,
                    swap=budget_tree.swap,
                    database=_database_size(output / "index.db"),
                    disk=_directory_size(output) + anonymous,
                )
                for name, value in {**values, "anonymous": anonymous}.items():
                    peaks[name] = max(peaks[name], value)
                violation = limits.violation(elapsed=time.monotonic() - started + 3, **values)
                if violation:
                    raise RuntimeError(violation)
                if process.poll() is not None:
                    if process.returncode:
                        raise RuntimeError("summary input validation failed; see worker.stderr")
                    break
                time.sleep(0.1)
    except BaseException as error:
        failure = str(error) or type(error).__name__
        raise
    finally:
        if process is not None:
            _terminate_process_group(process, known_groups=groups, known_processes=processes)
        clean = not any(_process_identity_live(p) for p in processes)
        guard = {
            "limits": asdict(limits),
            "peak_bytes": peaks,
            "elapsed_seconds": time.monotonic() - started,
            "processes_clean": clean,
            "failure": failure,
        }
        (output / "guard.json").write_text(json.dumps(guard, indent=2) + "\n")
    if not clean or time.monotonic() - started >= limits.wall_seconds:
        raise RuntimeError("summary input cleanup/deadline failed")
    report = json.loads((output / "candidate.json").read_text())
    report["guard"] = guard
    # Only the supervising parent publishes the distinct input identity after a clean exit.
    temporary = output / ".summary-input.json"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    violation = limits.violation(
        elapsed=time.monotonic() - started,
        rss=peaks["rss"],
        swap=peaks["swap"],
        database=peaks["database"],
        disk=_directory_size(output),
    )
    if violation:
        raise RuntimeError(violation)
    temporary.rename(output / "summary-input.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--rss-limit-mib", type=int, default=2560)
    parser.add_argument("--disk-limit-mib", type=int, default=2048)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        report = prepare(json.loads(args._worker.read_text()), args._worker.parent)
        (args._worker.parent / "candidate.json").write_text(json.dumps(report) + "\n")
    else:
        if args.request is None or args.output_directory is None:
            parser.error("--request and --output-directory are required")
        run(
            json.loads(args.request.read_text()),
            args.output_directory,
            CanaryLimits(
                wall_seconds=args.seconds,
                rss_bytes=args.rss_limit_mib * 1024**2,
                database_bytes=args.disk_limit_mib * 1024**2,
                disk_bytes=args.disk_limit_mib * 1024**2,
            ),
        )


if __name__ == "__main__":
    main()
