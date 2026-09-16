"""Fail-fast local acceptance harness for the real KiCad compilation database."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from cpp_context_engine.api import CallRequest, CfgRequest, FlowRequest, QueryRequest
from cpp_context_engine.benchmark import _write_report_atomic
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion import (
    AnalyzerPipelineEvent,
    AnalyzerSlotIdleGate,
    NativeAnalyzerClient,
    NativeClangIngestor,
    ProjectIndexer,
)
from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.models import BuildScope, BuildVariant, GraphDirection, IndexProfile
from cpp_context_engine.runtime import build_runtime
from cpp_context_engine.search import DeterministicLocalEmbeddingProvider, SQLiteVectorSearch
from cpp_context_engine.storage import SQLiteStore
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION

REPORT_SCHEMA_VERSION = 1
DATABASE_ARTIFACT_POLICY = (
    "framed-sha256-v1:main+nonempty-wal;shm-excluded;journal+other-sidecars-forbidden"
)
DEFAULT_GATE_TIMEOUTS: Mapping[str, float] = {
    "1": 60.0,
    "4": 90.0,
    "16": 120.0,
    "32": 150.0,
    "all": 5_400.0,
}
DEFAULT_TOTAL_GATE_TIMEOUTS: Mapping[str, float] = {
    "1": 90.0,
    "4": 120.0,
    "16": 150.0,
    "32": 180.0,
    "all": 5_400.0,
}
DEFAULT_QUERIES = ("compareVersionStrings",)
_SEMANTIC_TABLES = (
    "build_configurations",
    "translation_units",
    "dependencies",
    "symbols",
    "translation_unit_symbols",
    "build_variants",
    "symbol_variants",
    "occurrences",
    "edges",
    "cfg_graphs",
    "cfg_blocks",
    "cfg_elements",
    "cfg_edges",
    "callsites",
    "call_targets",
    "data_flow_analyses",
    "memory_locations",
    "data_accesses",
    "data_flow_evidence",
    "function_summaries",
    "summary_effects",
    "summary_return_origins",
    "call_argument_bindings",
    "call_result_bindings",
    "interprocedural_flows",
    "summary_solution_payloads",
    "embedding_vectors",
    "variant_embeddings",
)
_VOLATILE_COLUMNS = frozenset({"indexed_at", "compilation_database", "root"})


@dataclass(frozen=True, slots=True)
class CdbEntry:
    raw_index: int
    source_path: Path
    directory: Path
    classification: str
    display_path: str


@dataclass(frozen=True, slots=True)
class CdbInspection:
    project_root: Path
    compilation_database: Path
    sha256: str
    entry_count: int
    normalized_configuration_count: int
    entries: tuple[CdbEntry, ...]
    classification_counts: dict[str, int]

    @property
    def numeric_gate_eligible_count(self) -> int:
        return len(
            {
                entry.source_path
                for entry in self.entries
                if entry.classification == "project_source"
            }
        )

    @property
    def canonical_translation_unit_count(self) -> int:
        return len({entry.source_path for entry in self.entries})

    def public_report(self) -> dict[str, Any]:
        return {
            "schema": "cpp-context-kicad-cdb-preflight",
            "schema_version": REPORT_SCHEMA_VERSION,
            "cdb_sha256": self.sha256,
            "entry_count": self.entry_count,
            "normalized_configuration_count": self.normalized_configuration_count,
            "canonical_translation_unit_count": self.canonical_translation_unit_count,
            "classification_counts": self.classification_counts,
            "numeric_gate_eligible_count": self.numeric_gate_eligible_count,
            "external_generated_note": (
                "Out-of-tree entries are valid CMake inputs and are retained by the all gate; "
                "numeric canaries deliberately select project-root sources only."
            ),
        }


@dataclass(frozen=True, slots=True)
class SubsetDatabase:
    sha256: str
    raw_entry_count: int
    normalized_configuration_count: int


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _ValidatedArtifacts:
    subset_identity: _FileIdentity
    database_artifact: _DatabaseArtifact


@dataclass(frozen=True, slots=True)
class _DatabaseArtifact:
    sha256: str
    state: tuple[_FileIdentity, _FileIdentity | None]


def _require_finite_positive(name: str, value: int | float) -> None:
    # NaN bypasses ordinary <= 0 checks and would silently disable a hard gate.
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class CanaryLimits:
    wall_seconds: float
    rss_bytes: int = int(2.5 * 1024**3)
    database_bytes: int = 550 * 1024**2
    disk_bytes: int = 1024**3
    no_progress_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "wall_seconds",
            "rss_bytes",
            "database_bytes",
            "disk_bytes",
            "no_progress_seconds",
        ):
            _require_finite_positive(name, getattr(self, name))

    def violation(
        self,
        *,
        elapsed: float,
        rss: int,
        swap: int,
        database: int,
        disk: int,
    ) -> str | None:
        if swap > 0:
            return "swap used"
        if rss > self.rss_bytes:
            return f"process-tree RSS exceeded {self.rss_bytes} bytes"
        if database > self.database_bytes:
            return f"active database exceeded {self.database_bytes} bytes"
        if disk > self.disk_bytes:
            return f"gate disk use exceeded {self.disk_bytes} bytes"
        if elapsed > self.wall_seconds:
            return f"wall time exceeded {self.wall_seconds:g} seconds"
        return None


class _AnalyzerTelemetryMonitor:
    """Validate child telemetry and enforce the analyzer-slot acceptance gate."""

    def __init__(
        self,
        *,
        max_idle_seconds: float,
        expected_configurations: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if expected_configurations < 1:
            raise ValueError("expected analyzer configuration count must be positive")
        self._max_idle_seconds = max_idle_seconds
        self._expected_configurations = expected_configurations
        self._gate = AnalyzerSlotIdleGate(
            max_idle_seconds=max_idle_seconds,
            clock=clock,
        )
        self._event_count = 0
        self._started: set[int] = set()
        self._finished: set[int] = set()
        self._outcomes: dict[int, str] = {}

    def observe_payload(self, payload: Mapping[str, Any]) -> None:
        event = AnalyzerPipelineEvent.from_protocol_payload(payload)
        if (
            event.configuration_index is not None
            and event.configuration_index >= self._expected_configurations
        ):
            raise ValueError("analyzer telemetry configuration index exceeds the workload")
        if event.kind == "analyzer_started" and event.configuration_index in self._started:
            raise ValueError("analyzer telemetry started one configuration more than once")
        self._gate.observe(event)
        if event.kind == "analyzer_started":
            assert event.configuration_index is not None
            self._started.add(event.configuration_index)
        elif event.kind == "analyzer_finished":
            assert event.configuration_index is not None
            self._finished.add(event.configuration_index)
            assert event.outcome is not None
            self._outcomes[event.configuration_index] = event.outcome
        self._event_count += 1

    def check(self) -> None:
        self._gate.check()

    def success_report(self) -> dict[str, Any]:
        if self._event_count == 0:
            raise RuntimeError("analyzer pipeline telemetry is missing")
        slots = self._gate.report()
        if any(slot["state"] == "active" for slot in slots):
            raise RuntimeError("analyzer pipeline telemetry ended with an active slot")
        expected = set(range(self._expected_configurations))
        if self._started != expected or self._finished != expected:
            raise RuntimeError("analyzer pipeline telemetry is missing lifecycle events")
        if any(self._outcomes[index] != "succeeded" for index in sorted(expected)):
            # A later success on the same logical slot must never erase an
            # earlier configuration failure or cancellation from acceptance.
            raise RuntimeError("analyzer pipeline telemetry contains a non-successful lifecycle")
        if not slots or any(slot["outcome"] != "succeeded" for slot in slots):
            raise RuntimeError("analyzer pipeline telemetry has no successful terminal slot state")
        return {
            "protocol": "analyzer_pipeline",
            "clock": "monotonic",
            "event_count": self._event_count,
            "configuration_count": self._expected_configurations,
            "configuration_outcomes": [
                {"configuration_index": index, "outcome": self._outcomes[index]}
                for index in sorted(expected)
            ],
            "max_idle_seconds": self._max_idle_seconds,
            "slots": slots,
        }


def _validate_analyzer_pipeline_report(
    report: object,
    *,
    expected_configurations: int,
    expected_slots: int,
    max_idle_seconds: float,
) -> None:
    if not isinstance(report, Mapping):
        raise RuntimeError("analyzer pipeline telemetry report is missing")
    if (
        report.get("protocol") != "analyzer_pipeline"
        or report.get("clock") != "monotonic"
        or report.get("configuration_count") != expected_configurations
        or report.get("max_idle_seconds") != max_idle_seconds
        or type(report.get("event_count")) is not int
        or report["event_count"] < expected_configurations * 2 + 1
    ):
        raise RuntimeError("analyzer pipeline telemetry provenance is invalid")
    if report.get("configuration_outcomes") != [
        {"configuration_index": index, "outcome": "succeeded"}
        for index in range(expected_configurations)
    ]:
        raise RuntimeError("analyzer pipeline telemetry lifecycle history is invalid")
    slots = report.get("slots")
    if not isinstance(slots, list) or len(slots) != expected_slots:
        raise RuntimeError("analyzer pipeline telemetry slot report is incomplete")
    if [slot.get("slot_id") for slot in slots if isinstance(slot, Mapping)] != list(
        range(expected_slots)
    ):
        raise RuntimeError("analyzer pipeline telemetry slot order is invalid")
    for slot in slots:
        if (
            not isinstance(slot, Mapping)
            or slot.get("state") != "idle"
            or slot.get("outcome") != "succeeded"
            or slot.get("idle_cause") is not None
        ):
            raise RuntimeError("analyzer pipeline telemetry has no clean terminal slot state")
        maximum_idle = slot.get("maximum_idle_seconds")
        if (
            isinstance(maximum_idle, bool)
            or not isinstance(maximum_idle, (int, float))
            or not 0 <= maximum_idle <= max_idle_seconds
        ):
            raise RuntimeError("analyzer pipeline telemetry idle duration is invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> _FileIdentity:
    metadata = path.stat()
    return _FileIdentity(
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sqlite_artifact_state(
    database: Path,
) -> tuple[_FileIdentity, _FileIdentity | None]:
    wal = Path(f"{database}-wal")
    wal_identity = _file_identity(wal) if wal.is_file() else None
    if wal_identity is not None and wal_identity.size == 0:
        wal_identity = None
    return _file_identity(database), wal_identity


def _validate_sqlite_sidecars(database: Path) -> None:
    wal = Path(f"{database}-wal")
    shm = Path(f"{database}-shm")
    journal = Path(f"{database}-journal")
    if journal.exists():
        raise RuntimeError("retained database has a rollback journal")
    allowed = {wal, shm}
    unexpected = [
        path
        for path in database.parent.glob(database.name + "-*")
        if path.exists() and path not in allowed
    ]
    if unexpected:
        raise RuntimeError("retained database has an unexpected SQLite sidecar")
    if shm.exists() and not wal.exists():
        raise RuntimeError("retained database has shared memory without a WAL")


def _database_artifact_digest(database: Path) -> _DatabaseArtifact:
    _validate_sqlite_sidecars(database)
    before = _sqlite_artifact_state(database)
    # WAL bytes are durable database state; SHM is volatile coordination state.
    digest = hashlib.sha256()
    digest.update(b"cpp-context-sqlite-artifact\0v1\0")
    for role, path, identity in (
        (b"main", database, before[0]),
        (b"wal", Path(f"{database}-wal"), before[1]),
    ):
        digest.update(len(role).to_bytes(8, "big"))
        digest.update(role)
        if identity is None:
            digest.update(b"\0")
            continue
        digest.update(b"\1")
        digest.update(identity.size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    _validate_sqlite_sidecars(database)
    after = _sqlite_artifact_state(database)
    if after != before:
        raise RuntimeError("retained database changed while its artifact digest was computed")
    return _DatabaseArtifact(digest.hexdigest(), after)


@contextlib.contextmanager
def _database_writer_exclusion(database: Path) -> Iterator[sqlite3.Connection]:
    connection: sqlite3.Connection | None = None
    try:
        uri = database.resolve(strict=True).as_uri() + "?mode=rw"
        connection = sqlite3.connect(uri, uri=True, timeout=0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 0")
        connection.execute("BEGIN IMMEDIATE")
    except (OSError, sqlite3.Error) as error:
        if connection is not None:
            connection.close()
        raise RuntimeError(f"retained database has an active writer: {error}") from None
    try:
        yield connection
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def _validate_database_integrity(connection: sqlite3.Connection) -> str:
    # quick_check omits index consistency, so it cannot support release evidence.
    integrity_rows = [tuple(row) for row in connection.execute("PRAGMA integrity_check")]
    foreign_key_rows = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]
    if integrity_rows != [("ok",)] or foreign_key_rows:
        raise RuntimeError("canary database failed integrity or foreign-key checks")
    return "ok"


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_raw_cdb(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("compilation database cannot be read") from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("compilation database must be an array of objects")
    return payload


def inspect_compilation_database(project_root: Path, compilation_database: Path) -> CdbInspection:
    """Validate and classify every CDB entry without assuming all sources share one root."""

    root = project_root.expanduser().resolve(strict=True)
    cdb = compilation_database.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project root must be a directory")
    raw_entries = _load_raw_cdb(cdb)
    normalized = CompilationDatabase.load(cdb)
    entries: list[CdbEntry] = []
    counts = {"project_source": 0, "generated_build_source": 0, "external_source": 0}
    build_root = cdb.parent
    for raw_index, raw in enumerate(raw_entries):
        directory_raw = raw.get("directory")
        source_raw = raw.get("file")
        if not isinstance(directory_raw, str) or not isinstance(source_raw, str):
            # The authoritative loader above normally reports this with the entry number.
            raise ValueError(f"compilation database entry {raw_index} has invalid paths")
        directory = Path(directory_raw)
        if not directory.is_absolute():
            directory = cdb.parent / directory
        directory = directory.resolve(strict=False)
        source = Path(source_raw)
        if not source.is_absolute():
            source = directory / source
        source = source.resolve(strict=False)
        if not source.is_file():
            raise ValueError(f"compilation database entry {raw_index} source file does not exist")
        if _within(source, root):
            classification = "project_source"
            display = source.relative_to(root).as_posix()
        elif _within(source, build_root):
            classification = "generated_build_source"
            display = "generated:" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
        else:
            classification = "external_source"
            display = "external:" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
        counts[classification] += 1
        entries.append(CdbEntry(raw_index, source, directory, classification, display))
    return CdbInspection(
        root,
        cdb,
        _sha256(cdb),
        len(raw_entries),
        len(normalized.configurations),
        tuple(entries),
        counts,
    )


def select_gate_entries(inspection: CdbInspection, gate: int | str) -> tuple[CdbEntry, ...]:
    """Select numeric source canaries or the exact complete CDB in stable input order."""

    if gate == "all":
        return inspection.entries
    count = int(gate)
    if count <= 0:
        raise ValueError("gate sizes must be positive")
    eligible: list[CdbEntry] = []
    seen_sources: set[Path] = set()
    for entry in inspection.entries:
        if entry.classification != "project_source" or entry.source_path in seen_sources:
            continue
        seen_sources.add(entry.source_path)
        eligible.append(entry)
    if len(eligible) < count:
        raise ValueError(f"gate {count} needs {count} project-root TUs, found {len(eligible)}")
    return tuple(eligible[:count])


def _canonical_generated_roots_for_gates(
    inspection: CdbInspection,
    gates: Sequence[int | str],
    configured_roots: Sequence[Path],
) -> tuple[Path, ...]:
    roots = BuildVariant(
        "default",
        inspection.compilation_database,
        generated_source_roots=tuple(configured_roots),
    ).generated_source_roots
    if "all" not in gates:
        return roots
    unauthorized = [
        entry
        for entry in inspection.entries
        if entry.classification != "project_source"
        and not any(_within(entry.source_path, root) for root in roots)
    ]
    if unauthorized:
        # Classification discovers candidates but never grants the analyzer a source boundary.
        raise ValueError(
            "the all gate requires an explicit generated source root covering every "
            "out-of-tree source"
        )
    return roots


def write_subset_database(
    source_database: Path, entries: Sequence[CdbEntry], destination: Path
) -> SubsetDatabase:
    """Write a gate CDB while preserving command bytes and working-directory meaning."""

    raw = _load_raw_cdb(source_database.resolve(strict=True))
    selected: list[dict[str, Any]] = []
    for entry in entries:
        copied = dict(raw[entry.raw_index])
        copied["directory"] = str(entry.directory)
        selected.append(copied)
    document = json.dumps(selected, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_report_atomic(destination, document)
    normalized = CompilationDatabase.load(destination)
    # Real CDBs may repeat raw commands; preserve every row while indexing each identity once.
    return SubsetDatabase(
        sha256=hashlib.sha256(document.encode()).hexdigest(),
        raw_entry_count=len(selected),
        normalized_configuration_count=len(normalized.configurations),
    )


def _encode_digest_value(value: Any) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bytes):
        return b"b" + len(value).to_bytes(8, "big") + value
    encoded = str(value).encode("utf-8", errors="surrogateescape")
    return b"s" + len(encoded).to_bytes(8, "big") + encoded


def semantic_snapshot(
    database: Path, *, _connection: sqlite3.Connection | None = None
) -> dict[str, Any]:
    """Hash stable semantic rows, excluding timestamps and artifact-location columns."""

    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    table_digests: dict[str, str] = {}
    connection_context = (
        sqlite3.connect(database) if _connection is None else contextlib.nullcontext(_connection)
    )
    with connection_context as connection:
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        available = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        }
        for table in _SEMANTIC_TABLES:
            if table not in available:
                continue
            table_digest = hashlib.sha256()
            table_info = tuple(connection.execute(f'PRAGMA table_info("{table}")'))
            columns = tuple(row[1] for row in table_info if row[1] not in _VOLATILE_COLUMNS)
            counts[table] = int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            table_digest.update(table.encode())
            table_digest.update(b"\0")
            table_digest.update(str(counts[table]).encode())
            table_digest.update(b"\0")
            if not columns:
                table_digests[table] = table_digest.hexdigest()
                continue
            projection = ", ".join(f'"{column}"' for column in columns)
            ordering = ", ".join(str(position) for position in range(1, len(columns) + 1))
            for row in connection.execute(
                f'SELECT {projection} FROM "{table}" ORDER BY {ordering}'
            ):
                for value in row:
                    table_digest.update(_encode_digest_value(value))
                table_digest.update(b"\xff")
            table_digests[table] = table_digest.hexdigest()
            digest.update(table.encode())
            digest.update(bytes.fromhex(table_digests[table]))
    return {
        "digest": digest.hexdigest(),
        "counts": counts,
        "schema_version": schema_version,
        "table_digests": table_digests,
    }


def database_provenance(
    database: Path, *, _connection: sqlite3.Connection | None = None
) -> dict[str, Any]:
    """Summarize backend and independent coverage claims for every indexed TU."""

    fields = (
        "analysis_backend",
        "advanced_facts_complete",
        "index_profile",
        "navigation_facts_complete",
        "cfg_facts_complete",
        "data_flow_facts_complete",
        "summary_facts_complete",
    )
    connection_context = (
        sqlite3.connect(database) if _connection is None else contextlib.nullcontext(_connection)
    )
    with connection_context as connection:
        available = {row[1] for row in connection.execute('PRAGMA table_info("translation_units")')}
        if not set(fields).issubset(available):
            raise RuntimeError("translation-unit coverage provenance is incomplete")
        projection = ", ".join(fields)
        groups = [
            {**dict(zip(fields, row[:-1], strict=True)), "translation_units": int(row[-1])}
            for row in connection.execute(
                f"SELECT {projection}, count(*) FROM translation_units "
                f"GROUP BY {projection} ORDER BY {projection}"
            )
        ]
        builds = [
            {"name": str(row[0]), "index_profile": str(row[1])}
            for row in connection.execute(
                "SELECT name, index_profile FROM build_variants ORDER BY name"
            )
        ]
    return {"translation_unit_groups": groups, "build_variants": builds}


def _validate_profile_provenance(
    provenance: Mapping[str, Any], profile: IndexProfile, expected_units: int
) -> None:
    groups = provenance["translation_unit_groups"]
    expected_complete = int(profile is IndexProfile.FULL)
    expected = {
        "analysis_backend": "clang-libtooling",
        "advanced_facts_complete": expected_complete,
        "index_profile": profile.value,
        "navigation_facts_complete": 1,
        "cfg_facts_complete": expected_complete,
        "data_flow_facts_complete": expected_complete,
        "summary_facts_complete": expected_complete,
        "translation_units": expected_units,
    }
    if groups != [expected]:
        raise RuntimeError("database backend/profile coverage provenance does not match the gate")
    if provenance["build_variants"] != [{"name": "default", "index_profile": profile.value}]:
        raise RuntimeError("database build profile provenance does not match the gate")


def _database_size(database: Path) -> int:
    databases = [database]
    if database.parent.is_dir():
        databases.extend(
            private / database.name
            for private in database.parent.iterdir()
            if private.name.startswith(f".{database.name}.fresh-") and private.is_dir()
        )
    # Include unpublished generations and rollback journals, not just the final WAL path.
    return _unique_file_bytes(
        Path(f"{item}{suffix}") for item in databases for suffix in ("", "-wal", "-shm", "-journal")
    )


def _unique_file_bytes(paths: Iterable[Path]) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for path in paths:
        try:
            if not path.is_file():
                continue
            metadata = path.stat()
        except FileNotFoundError:
            continue  # Publication can unlink the private hardlink during a sample.
        identity = (metadata.st_dev, metadata.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += metadata.st_size
    return total


def _directory_size(directory: Path) -> int:
    try:
        return _unique_file_bytes(directory.rglob("*"))
    except FileNotFoundError:
        return 0


@dataclass(frozen=True, slots=True)
class _ProcessIdentity:
    pid: int
    start_ticks: int


@dataclass(frozen=True, slots=True)
class _ProcessGroupIdentity:
    group_id: int
    leader_start_ticks: int


@dataclass(frozen=True, slots=True)
class _TreeMetrics:
    rss: int
    swap: int
    cpu_ticks: int
    live_pids: tuple[int, ...]
    processes: tuple[_ProcessIdentity, ...] = ()
    groups: tuple[_ProcessGroupIdentity, ...] = ()


def _process_tree_metrics(root_pid: int) -> _TreeMetrics:
    processes: dict[int, tuple[int, int, int, int, str, int]] = {}
    try:
        proc_entries = tuple(Path("/proc").iterdir())
    except OSError:
        return _TreeMetrics(0, 0, 0, ())
    for entry in proc_entries:
        if not entry.name.isdecimal():
            continue
        try:
            status = {
                key: value.strip()
                for key, _, value in (
                    line.partition(":")
                    for line in (entry / "status").read_text(encoding="utf-8").splitlines()
                )
                if key in {"PPid", "VmRSS", "VmSwap", "State"}
            }
            stat = (entry / "stat").read_text(encoding="utf-8").split()
            processes[int(entry.name)] = (
                int(status["PPid"]),
                int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
                int(status.get("VmSwap", "0 kB").split()[0]) * 1024,
                int(stat[13]) + int(stat[14]),
                status.get("State", "?"),
                int(stat[21]),
            )
        except (OSError, KeyError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, *_rest) in processes.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    live = tuple(
        sorted(
            pid for pid in descendants if pid in processes and not processes[pid][4].startswith("Z")
        )
    )
    identities = tuple(_ProcessIdentity(pid, processes[pid][5]) for pid in live)
    group_identities: set[_ProcessGroupIdentity] = set()
    for pid in live:
        with contextlib.suppress(ProcessLookupError):
            group_id = os.getpgid(pid)
            leader = processes.get(group_id)
            if leader is not None:
                group_identities.add(_ProcessGroupIdentity(group_id, leader[5]))
    return _TreeMetrics(
        sum(processes[pid][1] for pid in live),
        sum(processes[pid][2] for pid in live),
        sum(processes[pid][3] for pid in live),
        live,
        identities,
        tuple(sorted(group_identities, key=lambda item: item.group_id)),
    )


def _remember_process_tree(
    groups: set[_ProcessGroupIdentity],
    processes: set[_ProcessIdentity],
    tree: _TreeMetrics,
) -> None:
    """Retain every verified descendant identity seen during a gate."""

    groups.update(tree.groups)
    processes.update(tree.processes)


def _read_process_identity(pid: int) -> _ProcessIdentity | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").split()
        if fields[2] == "Z":
            return None
        return _ProcessIdentity(pid, int(fields[21]))
    except (OSError, ValueError, IndexError):
        return None


def _process_identity_live(identity: _ProcessIdentity) -> bool:
    return _read_process_identity(identity.pid) == identity


def _process_group_identity_live(identity: _ProcessGroupIdentity) -> bool:
    leader = _read_process_identity(identity.group_id)
    if leader != _ProcessIdentity(identity.group_id, identity.leader_start_ticks):
        return False
    try:
        return os.getpgid(identity.group_id) == identity.group_id
    except ProcessLookupError:
        return False


def _read_lines(stream: TextIO, destination: queue.Queue[tuple[str, str]], kind: str) -> None:
    try:
        for line in stream:
            destination.put((kind, line.rstrip("\n")))
    finally:
        destination.put((kind + "_eof", ""))


def _terminate_process_group(
    process: subprocess.Popen[str],
    grace_seconds: float = 2.0,
    *,
    known_groups: Iterable[_ProcessGroupIdentity] = (),
    known_processes: Iterable[_ProcessIdentity] = (),
) -> None:
    tree = _process_tree_metrics(process.pid)
    groups = {*known_groups, *tree.groups}
    processes = {*known_processes, *tree.processes}
    for group in groups:
        if _process_group_identity_live(group):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group.group_id, signal.SIGTERM)
    for identity in processes:
        if _process_identity_live(identity):
            with contextlib.suppress(ProcessLookupError):
                os.kill(identity.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and (
        any(_process_group_identity_live(group) for group in groups)
        or any(_process_identity_live(identity) for identity in processes)
    ):
        time.sleep(0.05)
    for group in groups:
        if _process_group_identity_live(group):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group.group_id, signal.SIGKILL)
    for identity in processes:
        if _process_identity_live(identity):
            with contextlib.suppress(ProcessLookupError):
                os.kill(identity.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace_seconds)


def _process_group_live(group_id: int) -> bool:
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            if int(fields[4]) == group_id and fields[2] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def _linux_supervisor_available() -> bool:
    return os.name == "posix" and Path("/proc/self/stat").is_file()


def _require_supervisor_platform() -> None:
    if not _linux_supervisor_available():
        raise RuntimeError(
            "the KiCad canary requires Linux /proc process metrics; no analyzer was started"
        )


class _PhaseMeasurements:
    """The fixed canary phases, measured at worker boundaries, not pipe delivery."""

    def __init__(self, stage: str, started: float, stage_started: float, total_tus: int) -> None:
        self.names = (
            ("validation",)
            if stage == "validation"
            else ("tu_processing", "post_tu_finalization", "embeddings", "producer_checks")
        )
        self.started = started
        self.last_timestamp = stage_started
        self.starts: list[float | None] = [None] * len(self.names)
        self.ends: list[float | None] = [None] * len(self.names)
        self.current = -1
        self.operation_names = ("restore_deferred_indexes", "refresh_summaries")
        self.operation_starts: list[float | None] = [None, None]
        self.operation_ends: list[float | None] = [None, None]
        self.counts: dict[str, Any] = {
            "selected_tus": total_tus,
            "staged_tus": total_tus if stage == "validation" else 0,
            "indexing": None,
            "embedded_symbols": None,
        }

    def _advance(self, name: str, timestamp: float) -> None:
        next_index = self.current + 1
        if next_index >= len(self.names) or self.names[next_index] != name:
            raise ValueError("out-of-order worker phase")
        if self.current == 1 and any(
            start is not None and end is None
            for start, end in zip(self.operation_starts, self.operation_ends, strict=True)
        ):
            raise ValueError("worker phase advanced before operation completion")
        if self.current >= 0:
            self.ends[self.current] = timestamp
        self.current = next_index
        self.starts[self.current] = timestamp

    def observe(self, event: Mapping[str, Any], received_at: float) -> bool:
        timestamp = event.get("monotonic_seconds")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or not self.last_timestamp <= timestamp <= received_at
        ):
            raise ValueError("invalid worker measurement timestamp")
        kind = event["event"]
        changed = True
        if kind == "phase":
            self._advance(
                "tu_processing" if event.get("name") == "index" else event.get("name"),
                timestamp,
            )
            for field in ("indexing", "embedded_symbols"):
                if field in event:
                    self.counts[field] = event[field]
        elif kind == "tu_staged":
            completed = event.get("completed")
            if (
                self.names[0] != "tu_processing"
                or self.current != 0
                or type(completed) is not int
                or completed != self.counts["staged_tus"] + 1
                or completed > self.counts["selected_tus"]
            ):
                raise ValueError("invalid staged TU measurement")
            self.counts["staged_tus"] = completed
            changed = completed == self.counts["selected_tus"]
            # A partially staged stream has not reached the global finalization tail.
            if changed:
                self._advance("post_tu_finalization", timestamp)
        elif kind == "post_tu_operation":
            if self.current != 1 or event.get("name") not in self.operation_names:
                raise ValueError("operation outside post-TU finalization")
            index = self.operation_names.index(event["name"])
            if event.get("status") == "started":
                if any(start is not None for start in self.operation_starts[index:]) or any(
                    start is not None and end is None
                    for start, end in zip(self.operation_starts, self.operation_ends, strict=True)
                ):
                    raise ValueError("out-of-order post-TU operation")
                self.operation_starts[index] = timestamp
            elif event.get("status") == "completed":
                if self.operation_starts[index] is None or self.operation_ends[index] is not None:
                    raise ValueError("post-TU completion without active operation")
                self.operation_ends[index] = timestamp
            else:
                raise ValueError("unknown post-TU operation status")
        elif kind == "result":
            if self.current != len(self.names) - 1 or self.ends[self.current] is not None:
                raise ValueError("worker result before all measurement phases")
            result = event.get("result")
            if not isinstance(result, Mapping):
                raise ValueError("worker measurement result must be an object")
            snapshot = result.get("semantic_snapshot", {})
            if not isinstance(snapshot, Mapping):
                raise ValueError("worker measurement snapshot must be an object")
            self.ends[self.current] = timestamp
            if "counts" in snapshot:
                self.counts["table_counts"] = snapshot["counts"]
        elif kind != "error":
            raise ValueError("unknown worker measurement event")
        self.last_timestamp = timestamp
        return changed

    def snapshot(self) -> dict[str, Any]:
        def interval(name: str, start: float | None, end: float | None) -> dict[str, Any]:
            return {
                "name": name,
                "status": "not_started"
                if start is None
                else "incomplete"
                if end is None
                else "complete",
                "start_seconds": None if start is None else start - self.started,
                "end_seconds": None if end is None else end - self.started,
                "duration_seconds": None if end is None else end - start,
                "observed_seconds": None
                if start is None
                else (end if end is not None else self.last_timestamp) - start,
            }

        phases = [
            interval(name, start, end)
            for name, start, end in zip(self.names, self.starts, self.ends, strict=True)
        ]
        result = {"phases": phases, "counts": dict(self.counts)}
        if self.names[0] == "tu_processing":
            operations = [
                interval(name, start, end)
                for name, start, end in zip(
                    self.operation_names, self.operation_starts, self.operation_ends, strict=True
                )
            ]
            result["post_tu_operations"] = operations
            duration = phases[1]["duration_seconds"]
            # These intervals are nested within the phase, not additional time.
            result["post_tu_unattributed_seconds"] = (
                None
                if duration is None
                else duration
                - sum(
                    operation["duration_seconds"]
                    for operation in operations
                    if operation["duration_seconds"] is not None
                )
            )
        return result


def _run_supervised(
    spec_path: Path,
    gate_directory: Path,
    limits: CanaryLimits,
    total_tus: int,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    stage = spec.get("supervision_stage", "index")
    if stage not in {"index", "validation"}:
        raise ValueError("unknown canary supervision stage")
    stage_started = time.monotonic()
    started = float(spec.get("gate_started_monotonic", stage_started))
    total_wall_seconds = float(spec.get("total_wall_seconds", limits.wall_seconds))
    full_project = bool(spec.get("full_project", False))
    phase = stage
    measurements = _PhaseMeasurements(stage, started, stage_started, total_tus)
    peak_rss = peak_swap = peak_database = peak_disk = 0

    def persist_measurements() -> None:
        _write_report_atomic(
            gate_directory / f"phase-timings-{stage}.json",
            json.dumps(
                {
                    "schema": "cpp-context-kicad-phase-timings",
                    "schema_version": 1,
                    "measurement_provenance": spec.get("measurement_provenance", {}),
                    "limits": asdict(limits),
                    "total_wall_seconds": total_wall_seconds,
                    "captured_at_seconds": time.monotonic() - started,
                    "peak_rss_bytes": peak_rss,
                    "peak_swap_bytes": peak_swap,
                    "peak_database_bytes": peak_database,
                    "peak_disk_bytes": peak_disk,
                    "measurements": measurements.snapshot(),
                },
                sort_keys=True,
            )
            + "\n",
        )

    persist_measurements()
    command = [
        sys.executable,
        "-m",
        "cpp_context_engine.kicad_canary",
        "--_worker-spec",
        str(spec_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        bufsize=1,
    )
    assert process.stdout is not None and process.stderr is not None
    messages: queue.Queue[tuple[str, str]] = queue.Queue()
    readers = (
        threading.Thread(
            target=_read_lines, args=(process.stdout, messages, "stdout"), daemon=True
        ),
        threading.Thread(
            target=_read_lines, args=(process.stderr, messages, "stderr"), daemon=True
        ),
    )
    for reader in readers:
        reader.start()
    last_report = stage_started
    last_useful = stage_started
    last_signature: tuple[int, int, int] | None = None
    completed_tus = total_tus if stage == "validation" else 0
    stderr_tail: list[str] = []
    worker_result: dict[str, Any] | None = None
    violation: str | None = None
    analyzer_telemetry = (
        _AnalyzerTelemetryMonitor(
            max_idle_seconds=limits.no_progress_seconds,
            expected_configurations=total_tus,
        )
        if stage == "index"
        else None
    )
    stdout_eof = stderr_eof = False
    observed_groups: set[_ProcessGroupIdentity] = set()
    observed_processes: set[_ProcessIdentity] = set()
    database = gate_directory / "index.db"
    last_measurement_write = stage_started
    try:
        while process.poll() is None or not (stdout_eof and stderr_eof):
            measurement_transition = False
            try:
                kind, line = messages.get(timeout=0.1)
            except queue.Empty:
                kind = line = ""
            if kind == "stdout_eof":
                stdout_eof = True
            elif kind == "stderr_eof":
                stderr_eof = True
            elif kind == "stderr":
                stderr_tail.append(line)
                stderr_tail = stderr_tail[-100:]
            elif kind == "stdout" and line:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    violation = "worker emitted non-JSON protocol output"
                else:
                    if isinstance(event, Mapping) and event.get("event") in {
                        "phase",
                        "tu_staged",
                        "post_tu_operation",
                        "result",
                        "error",
                    }:
                        try:
                            measurement_transition = measurements.observe(event, time.monotonic())
                        except (KeyError, TypeError, ValueError) as error:
                            violation = violation or f"invalid phase measurement: {error}"
                    if not isinstance(event, Mapping):
                        violation = "worker emitted a non-object protocol payload"
                    elif event.get("event") == "analyzer_pipeline":
                        try:
                            if analyzer_telemetry is None:
                                raise ValueError("validator emitted analyzer telemetry")
                            analyzer_telemetry.observe_payload(event)
                        except (RuntimeError, ValueError) as error:
                            violation = f"invalid analyzer pipeline telemetry: {error}"
                    elif event.get("event") == "tu_staged":
                        completed_tus = int(event["completed"])
                    elif event.get("event") == "result":
                        worker_result = event["result"]
                        if stage == "validation":
                            phase = "complete"
                    elif event.get("event") == "error":
                        violation = str(event.get("message", "worker failed"))
                    elif event.get("event") == "post_tu_operation":
                        pass  # Validated and persisted by the existing phase measurements above.
                    elif event.get("event") == "phase":
                        if event.get("name") not in {
                            "index",
                            "embeddings",
                            "producer_checks",
                            "validation",
                        }:
                            violation = "worker emitted an unknown phase"
                        else:
                            phase = str(event["name"])
                    else:
                        violation = "worker emitted an unknown protocol event"
            elapsed = time.monotonic() - stage_started
            total_elapsed = time.monotonic() - started
            tree = _process_tree_metrics(process.pid)
            # Descendants can reparent or create their own sessions before failure cleanup.
            _remember_process_tree(observed_groups, observed_processes, tree)
            # Count the coordinator too, but never include it in owned-child cleanup.
            budget_tree = _process_tree_metrics(os.getpid())
            database_bytes = _database_size(database)
            disk_bytes = _directory_size(gate_directory)
            peak_rss = max(peak_rss, budget_tree.rss)
            peak_swap = max(peak_swap, budget_tree.swap)
            peak_database = max(peak_database, database_bytes)
            peak_disk = max(peak_disk, disk_bytes)
            if measurement_transition or time.monotonic() - last_measurement_write >= 5:
                persist_measurements()
                last_measurement_write = time.monotonic()
            current = limits.violation(
                elapsed=elapsed,
                rss=budget_tree.rss,
                swap=budget_tree.swap,
                database=database_bytes,
                disk=disk_bytes,
            )
            if violation is None:
                violation = current
            if violation is None and total_elapsed > total_wall_seconds:
                violation = "end-to-end gate deadline exceeded"
            if violation is None and analyzer_telemetry is not None:
                try:
                    # Analyzer events are transition-based, so enforce open idle
                    # intervals even while the child emits no protocol records.
                    analyzer_telemetry.check()
                except (RuntimeError, ValueError) as error:
                    violation = f"analyzer slot idle gate failed: {error}"
            signature = (completed_tus, database_bytes, tree.cpu_ticks)
            if signature != last_signature:
                last_signature = signature
                last_useful = time.monotonic()
            elif (
                process.poll() is None
                and time.monotonic() - last_useful > limits.no_progress_seconds
            ):
                violation = f"no observable progress for {limits.no_progress_seconds:g} seconds"
            if violation is None and full_project and completed_tus and phase != "complete":
                projected = total_elapsed * total_tus / completed_tus
                if total_elapsed >= 1_800 and projected > 3_600:
                    violation = "30-minute projection exceeds the 60-minute target"
                elif total_elapsed >= 600 and projected > 5_400:
                    violation = "10-minute projection exceeds the 90-minute hard limit"
            if violation is None and full_project and total_elapsed >= 600 and phase != "complete":
                # TU throughput cannot estimate still-unmeasured embeddings/verification.
                violation = f"total projection unknown at decision checkpoint (phase {phase})"
            if time.monotonic() - last_report >= 5 or kind == "stdout" and completed_tus:
                rate = completed_tus / elapsed if elapsed > 0 else 0.0
                eta = (
                    (total_tus - completed_tus) / rate
                    if phase == "index" and 0 < completed_tus < total_tus and rate > 0
                    else None
                )
                eta_text = f"{eta:.1f}s" if eta is not None else "unknown"
                print(
                    f"canary: {completed_tus}/{total_tus} TUs, {total_elapsed:.1f}s, "
                    f"phase {phase}, "
                    f"TU-only ETA {eta_text}, total ETA unknown, "
                    f"RSS {budget_tree.rss / 1024**2:.1f} MiB, "
                    f"DB {database_bytes / 1024**2:.1f} MiB",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = time.monotonic()
            if violation is not None and process.poll() is None:
                _terminate_process_group(
                    process,
                    known_groups=observed_groups,
                    known_processes=observed_processes,
                )
        return_code = process.wait(timeout=2)
    finally:
        _terminate_process_group(
            process,
            known_groups=observed_groups,
            known_processes=observed_processes,
        )
        for reader in readers:
            reader.join(timeout=1)
        persist_measurements()
    if violation is not None:
        detail = stderr_tail[-1] if stderr_tail else ""
        raise RuntimeError(f"{violation}" + (f": {detail}" if detail else ""))
    if return_code != 0 or worker_result is None:
        detail = stderr_tail[-1] if stderr_tail else f"exit {return_code}"
        raise RuntimeError(f"canary worker failed: {detail}")
    try:
        analyzer_report = analyzer_telemetry.success_report() if analyzer_telemetry else None
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(f"analyzer pipeline telemetry did not finish safely: {error}") from None
    return {
        **worker_result,
        "phase_measurements": measurements.snapshot(),
        **({"analyzer_pipeline": analyzer_report} if analyzer_report is not None else {}),
        "elapsed_seconds": time.monotonic() - stage_started,
        "total_elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "peak_swap_bytes": peak_swap,
        "peak_database_bytes": peak_database,
        "peak_disk_bytes": peak_disk,
        "completed_translation_units": completed_tus,
        "process_group_clean": not any(
            _process_group_identity_live(group) for group in observed_groups
        )
        and not any(_process_identity_live(identity) for identity in observed_processes),
    }


class _ObservedIngestor:
    def __init__(self, delegate: NativeClangIngestor, total: int) -> None:
        self.delegate = delegate
        self.total = total
        self.analysis_backend = delegate.analysis_backend
        self.advanced_facts_complete = delegate.advanced_facts_complete

    @property
    def analyzer_identity(self) -> str:
        return self.delegate.analyzer_identity

    def iter_configuration_batches(
        self, project_root: Path, configurations: Iterable[Any]
    ) -> Iterable[Any]:
        batches = self.delegate.iter_configuration_batches(project_root, configurations)
        for completed, batch in enumerate(batches, start=1):
            yield batch
            _worker_event("tu_staged", completed=completed, total=self.total)


_WORKER_EVENT_LOCK = threading.Lock()


def _write_worker_payload(payload: Mapping[str, Any]) -> None:
    with _WORKER_EVENT_LOCK:
        print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _worker_event(event: str, **fields: Any) -> None:
    _write_worker_payload({"event": event, "monotonic_seconds": time.monotonic(), **fields})


def _worker_finalization_event(name: str, status: str) -> None:
    _worker_event("post_tu_operation", name=name, status=status)


def _worker_analyzer_event(event: AnalyzerPipelineEvent) -> None:
    _write_worker_payload(event.to_protocol_payload())


def _ordered_public_result_digest(result: Any) -> str:
    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    document = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _public_summary_ordering(data_flow: Any, *, required: bool) -> dict[str, Any]:
    # Row digests cannot detect changes in the public service's summary ordering.
    payload = data_flow.model_dump(mode="json")
    analyses = [
        {
            field: analysis.get(field)
            for field in (
                "analysis_id",
                "summary_complete",
                "summary_incomplete_reasons",
                "effects",
                "return_origins",
                "interprocedural",
            )
        }
        for analysis in payload.get("analyses", ())
        if analysis.get("summary_complete") is not None
    ]
    available = bool(getattr(data_flow, "available", True)) and bool(analyses)
    if not available:
        if required:
            raise RuntimeError("full-profile public summary response is unavailable")
        return {
            "symbol_id": data_flow.function_symbol_id,
            "available": False,
            "unavailable_reason": (
                getattr(data_flow, "unavailable_reason", None)
                or "public summary data is not materialized"
            ),
            "required_action": getattr(data_flow, "required_action", None),
        }
    summary_payload = {
        "function_symbol_id": payload["function_symbol_id"],
        "scope": payload["scope"],
        "truncated": payload["truncated"],
        "analyses": analyses,
    }
    return {
        "symbol_id": data_flow.function_symbol_id,
        "available": True,
        "analysis_count": len(analyses),
        "digest": _ordered_public_result_digest(summary_payload),
    }


def _ranking_canaries(
    config: AppConfig, queries: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rankings: dict[str, Any] = {}
    public_orderings: dict[str, Any] = {}
    summary_orderings: dict[str, Any] = {}
    with build_runtime(config) as runtime:
        for query in queries:
            response = runtime.query_context(
                QueryRequest(
                    query,
                    max_context_tokens=4_000,
                    max_results=10,
                    builds=config.build_scope.variants,
                )
            ).context
            rankings[query] = [
                {
                    "symbol_id": item.hit.symbol.id,
                    "score_hex": item.hit.score.hex(),
                    "source": item.hit.source,
                    "reason": item.reason,
                }
                for item in response.items
            ]
            analyses: list[dict[str, Any]] = []
            seen_symbols: set[str] = set()
            pinned_symbol_id = response.items[0].hit.symbol.id if response.items else None
            if pinned_symbol_id is None:
                if config.index_profile == IndexProfile.FULL:
                    raise RuntimeError("full-profile canary query returned no pinned symbol")
                summary_orderings[query] = {
                    "symbol_id": None,
                    "available": False,
                    "unavailable_reason": "canary query returned no pinned symbol",
                    "required_action": None,
                }
            for item in response.items:
                symbol_id = item.hit.symbol.id
                if symbol_id in seen_symbols:
                    continue
                seen_symbols.add(symbol_id)
                control_flow = runtime.analysis_service.control_flow(
                    CfgRequest(
                        function_symbol_id=symbol_id,
                        builds=list(config.build_scope.variants),
                    )
                )
                data_flow = runtime.analysis_service.data_flow(
                    FlowRequest(
                        function_symbol_id=symbol_id,
                        builds=list(config.build_scope.variants),
                    )
                )
                if symbol_id == pinned_symbol_id:
                    summary_orderings[query] = _public_summary_ordering(
                        data_flow,
                        required=config.index_profile == IndexProfile.FULL,
                    )
                incoming = runtime.analysis_service.calls(
                    CallRequest(
                        symbol_id=symbol_id,
                        direction=GraphDirection.INCOMING,
                        builds=list(config.build_scope.variants),
                    )
                )
                outgoing = runtime.analysis_service.calls(
                    CallRequest(
                        symbol_id=symbol_id,
                        direction=GraphDirection.OUTGOING,
                        builds=list(config.build_scope.variants),
                    )
                )
                analyses.append(
                    {
                        "symbol_id": symbol_id,
                        "control_flow": _ordered_public_result_digest(control_flow),
                        "data_flow": _ordered_public_result_digest(data_flow),
                        "incoming_calls": _ordered_public_result_digest(incoming),
                        "outgoing_calls": _ordered_public_result_digest(outgoing),
                    }
                )
            public_orderings[query] = {
                "retrieval_symbol_ids": [item.hit.symbol.id for item in response.items],
                "analyses": analyses,
            }
    return rankings, public_orderings, summary_orderings


def _revalidate_final_artifacts(
    *,
    project_root: Path,
    subset: Path,
    expected_subset: SubsetDatabase,
    database: Path,
    analyzer: Path,
    profile: IndexProfile,
    workers: int,
    embedding_dimensions: int,
    queries: Sequence[str],
    generated_source_roots: tuple[Path, ...],
    child_result: Mapping[str, Any],
) -> _ValidatedArtifacts:
    try:
        subset_identity = _file_identity(subset)
        normalized = CompilationDatabase.load(
            subset,
            project_root=project_root,
            generated_source_roots=generated_source_roots,
        )
        actual_subset = SubsetDatabase(
            sha256=_sha256(subset),
            raw_entry_count=len(_load_raw_cdb(subset)),
            normalized_configuration_count=len(normalized.configurations),
        )
        if actual_subset != expected_subset:
            raise RuntimeError("retained compilation database differs from the selected subset")

        with _database_writer_exclusion(database) as connection:
            database_artifact = _database_artifact_digest(database)
            integrity = _validate_database_integrity(connection)
            snapshot = semantic_snapshot(database, _connection=connection)
            provenance = database_provenance(database, _connection=connection)

        variant = BuildVariant("default", subset, generated_source_roots=generated_source_roots)
        scope = BuildScope((variant.name,))
        config = AppConfig(
            project_root=project_root,
            index_directory=database.parent,
            database_path=database,
            compilation_database=subset,
            build_variants=(variant,),
            build_scope=scope,
            index_profile=profile,
            clang_analyzer_path=analyzer,
            analyzer_max_workers=workers,
            embedding_dimensions=embedding_dimensions,
        )
        rankings, public_orderings, summary_orderings = _ranking_canaries(config, queries)
        parent_evidence = {
            "rankings": rankings,
            "public_orderings": public_orderings,
            "summary_orderings": summary_orderings,
            "semantic_snapshot": snapshot,
            "database_provenance": provenance,
            "database_artifact_sha256": database_artifact.sha256,
            "database_sidecar_policy": DATABASE_ARTIFACT_POLICY,
            "database_integrity": integrity,
        }
        for field, value in parent_evidence.items():
            if child_result.get(field) != value:
                raise RuntimeError(f"child result differs for {field}")
        child_analyzer = child_result.get("analyzer")
        if not isinstance(child_analyzer, Mapping) or child_analyzer.get("sha256") != _sha256(
            analyzer
        ):
            raise RuntimeError("child result differs for analyzer sha256")

        # A child or external writer changing either artifact during validation
        # invalidates every digest derived from that read window.
        if _file_identity(subset) != subset_identity:
            raise RuntimeError("retained compilation database changed during validation")
        with _database_writer_exclusion(database):
            final_database_artifact = _database_artifact_digest(database)
        if final_database_artifact != database_artifact:
            raise RuntimeError("retained database changed during validation")
        return _ValidatedArtifacts(subset_identity, final_database_artifact)
    except Exception as error:
        raise RuntimeError(f"parent artifact revalidation failed: {error}") from None


@contextlib.contextmanager
def _validated_artifact_publication(
    subset: Path, database: Path, validated: _ValidatedArtifacts
) -> Iterator[None]:
    try:
        if _file_identity(subset) != validated.subset_identity:
            raise RuntimeError("retained compilation database changed")
        # Hold SQLite's writer reservation through marker publication; a momentary
        # preflight lock would still allow a late commit behind SUCCESS.
        with _database_writer_exclusion(database):
            artifact = _database_artifact_digest(database)
            if artifact != validated.database_artifact:
                raise RuntimeError("retained database changed")
            yield
    except Exception as error:
        raise RuntimeError(f"parent artifact revalidation failed: {error}") from None


def _run_worker(spec_path: Path) -> int:
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("supervision_stage") == "validation":
            _worker_event("phase", name="validation")
            _validate_gate_spec(spec)
            _worker_event("result", result={"validated": True})
            return 0
        project = Path(spec["project_root"])
        cdb = Path(spec["compilation_database"])
        database = Path(spec["database"])
        analyzer = Path(spec["analyzer"])
        total = int(spec["translation_units"])
        profile = IndexProfile(spec["profile"])
        client = NativeAnalyzerClient(
            analyzer,
            timeout_seconds=float(spec["analyzer_timeout_seconds"]),
            profile=profile,
        )
        info = client.probe()
        ingestor = NativeClangIngestor(
            client,
            max_workers=int(spec["workers"]),
            profile=profile,
            observer=_worker_analyzer_event,
        )
        generated_source_roots = tuple(
            Path(path) for path in spec.get("generated_source_roots", ())
        )
        variant = BuildVariant("default", cdb, generated_source_roots=generated_source_roots)
        scope = BuildScope((variant.name,))
        _worker_event("phase", name="index")
        with SQLiteStore.indexing_generation(
            database, project_root=project, build_scope=scope
        ) as store:
            indexing = ProjectIndexer(
                _ObservedIngestor(ingestor, total), store, profile=profile
            ).index(
                project,
                cdb,
                build_variant=variant,
                finalization_observer=_worker_finalization_event,
            )
            _worker_event("phase", name="embeddings", indexing=asdict(indexing))
            embedded = SQLiteVectorSearch(
                store,
                DeterministicLocalEmbeddingProvider(int(spec["embedding_dimensions"])),
                project_root=project,
                build_scope=scope,
            ).index_missing()
        _worker_event("phase", name="producer_checks", embedded_symbols=embedded)
        config = AppConfig(
            project_root=project,
            index_directory=database.parent,
            database_path=database,
            compilation_database=cdb,
            build_variants=(variant,),
            build_scope=scope,
            index_profile=profile,
            clang_analyzer_path=analyzer,
            analyzer_max_workers=int(spec["workers"]),
            embedding_dimensions=int(spec["embedding_dimensions"]),
        )
        rankings, public_orderings, summary_orderings = _ranking_canaries(
            config, tuple(spec["queries"])
        )
        snapshot = semantic_snapshot(database)
        provenance = database_provenance(database)
        with _database_writer_exclusion(database) as connection:
            integrity = _validate_database_integrity(connection)
            database_artifact = _database_artifact_digest(database)
        analyzer_sha256 = _sha256(analyzer)
        expected_analyzer = spec.get("measurement_provenance", {}).get("analyzer_sha256")
        if expected_analyzer is not None and analyzer_sha256 != expected_analyzer:
            raise RuntimeError("analyzer changed from the phase measurement input pin")
        _worker_event(
            "result",
            result={
                "indexing": asdict(indexing),
                "embedded_symbols": embedded,
                "rankings": rankings,
                "public_orderings": public_orderings,
                "summary_orderings": summary_orderings,
                "semantic_snapshot": snapshot,
                "database_provenance": provenance,
                "database_artifact_sha256": database_artifact.sha256,
                "database_sidecar_policy": DATABASE_ARTIFACT_POLICY,
                "database_integrity": integrity,
                "analyzer": {
                    "version": info.analyzer_version,
                    "protocol": info.protocol,
                    "protocol_version": info.protocol_version,
                    "clang_major": info.clang_major,
                    "capabilities": sorted(info.capabilities),
                    "sha256": analyzer_sha256,
                },
                "native_spool_budget_bytes": ingestor.max_spool_bytes,
            },
        )
        return 0
    except BaseException as error:
        _worker_event("error", message=f"{type(error).__name__}: {error}")
        return 2


def _validate_gate_spec(spec: Mapping[str, Any]) -> None:
    """Independent validator process; never run these checks in the index producer."""

    database = Path(spec["database"])
    subset = Path(spec["compilation_database"])
    source_cdb = Path(spec["source_compilation_database"])
    gate_report = spec["gate_report"]
    validated = _revalidate_final_artifacts(
        project_root=Path(spec["project_root"]),
        subset=subset,
        expected_subset=SubsetDatabase(**spec["subset_metadata"]),
        database=database,
        analyzer=Path(spec["analyzer"]),
        profile=IndexProfile(spec["profile"]),
        workers=int(spec["workers"]),
        embedding_dimensions=int(spec["embedding_dimensions"]),
        queries=tuple(spec["queries"]),
        generated_source_roots=tuple(Path(root) for root in spec["generated_source_roots"]),
        child_result=gate_report,
    )
    _validate_profile_provenance(
        gate_report["database_provenance"], IndexProfile(spec["profile"]), spec["translation_units"]
    )
    if spec["baseline_gate"] is not None:
        _compare_baseline_gate(gate_report, spec["baseline_gate"])
    if _sha256(source_cdb) != spec["source_cdb_sha256"]:
        raise RuntimeError("source compilation database changed during the canary")
    with _validated_artifact_publication(subset, database, validated):
        # The supervisor can interrupt a blocked validator. Recheck the shared
        # deadline under the publication lock too, before any SUCCESS appears.
        if time.monotonic() - spec["gate_started_monotonic"] > spec["total_wall_seconds"]:
            raise RuntimeError("end-to-end gate deadline exceeded before publication")
        (database.parent / ".running").unlink()
        _write_report_atomic(database.parent / "SUCCESS", "complete\n")


def _run_validation_supervised(
    spec_path: Path, directory: Path, limits: CanaryLimits, total_tus: int
) -> dict[str, Any]:
    return _run_supervised(spec_path, directory, limits, total_tus)


def _parse_gates(raw: str) -> tuple[int | str, ...]:
    gates: list[int | str] = []
    for item in raw.split(","):
        value = item.strip()
        if value == "all":
            gate: int | str = "all"
        else:
            try:
                gate = int(value)
            except ValueError as error:
                raise ValueError("gates must be positive integers or 'all'") from error
            if gate <= 0:
                raise ValueError("gates must be positive integers or 'all'")
        if gate in gates:
            raise ValueError("gates must not repeat")
        gates.append(gate)
    if not gates:
        raise ValueError("at least one gate is required")
    if "all" in gates and gates[-1] != "all":
        raise ValueError("the all gate must be last")
    numeric = [gate for gate in gates if isinstance(gate, int)]
    if numeric != sorted(numeric):
        raise ValueError("numeric gates must increase")
    return tuple(gates)


def _parse_timeouts(raw: str) -> dict[str, float]:
    values = dict(DEFAULT_GATE_TIMEOUTS)
    for item in raw.split(","):
        name, separator, seconds_raw = item.partition(":")
        if not separator:
            raise ValueError("gate timeouts must use GATE:SECONDS")
        seconds = float(seconds_raw)
        _require_finite_positive("gate timeout", seconds)
        values[name] = seconds
    return values


def _git_revision(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(project_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip()


def _comparable_gates(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    gates = report.get("gates")
    if not isinstance(gates, list) or not all(isinstance(gate, Mapping) for gate in gates):
        raise ValueError("baseline report has no comparable gates")
    return gates


def _compare_baseline_header(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    for field in (
        "engine_commit",
        "project_commit",
        "profile",
        "workers",
        "embedding_dimensions",
    ):
        if report.get(field) != baseline.get(field):
            raise RuntimeError(f"baseline provenance differs for {field}")
    current_input = report.get("input")
    baseline_input = baseline.get("input")
    if not isinstance(current_input, Mapping) or not isinstance(baseline_input, Mapping):
        raise ValueError("baseline report has no input provenance")
    if current_input.get("cdb_sha256") != baseline_input.get("cdb_sha256"):
        raise RuntimeError("baseline compilation database digest differs")
    current_gate_names = [str(gate.get("gate")) for gate in _comparable_gates(report)]
    baseline_gate_names = [str(gate.get("gate")) for gate in _comparable_gates(baseline)]
    if current_gate_names != baseline_gate_names:
        raise RuntimeError("baseline gate set or order differs")


def _compare_baseline_gate(gate: Mapping[str, Any], baseline_gate: Mapping[str, Any]) -> None:
    key = str(gate.get("gate"))
    # Independent SQLite files contain different timestamps and output paths;
    # physical hashes protect each artifact's validation/publication, not parity.
    for field in (
        "selection",
        "raw_cdb_entries",
        "translation_units",
        "selected_raw_indices",
        "subset_cdb_sha256",
        "semantic_snapshot",
        "rankings",
        "public_orderings",
        "summary_orderings",
        "database_provenance",
        "database_sidecar_policy",
        "database_integrity",
        "analyzer",
    ):
        if gate.get(field) != baseline_gate.get(field):
            raise RuntimeError(f"baseline gate {key} differs for {field}")


def _compare_baseline(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    _compare_baseline_header(report, baseline)
    for gate, baseline_gate in zip(
        _comparable_gates(report), _comparable_gates(baseline), strict=True
    ):
        _compare_baseline_gate(gate, baseline_gate)


@contextlib.contextmanager
def _gate_failure_publication(running: Path, failed: Path) -> Iterator[None]:
    try:
        yield
    except BaseException as error:
        if running.exists():
            (running / "SUCCESS").unlink(missing_ok=True)
            error_type = "".join(
                character
                for character in type(error).__name__
                if character.isalnum() or character == "_"
            )[:128]
            _write_report_atomic(
                running / "FAILURE.json",
                json.dumps(
                    {
                        "schema": "cpp-context-kicad-canary-failure",
                        "schema_version": 1,
                        "cause": "gate_setup_or_execution_failed",
                        "error_type": error_type or "Exception",
                    },
                    sort_keys=True,
                )
                + "\n",
            )
            running.rename(failed)
        raise


def run_canary(
    *,
    project_root: Path,
    compilation_database: Path,
    analyzer: Path,
    output_directory: Path,
    gates: Sequence[int | str],
    gate_timeouts: Mapping[str, float],
    workers: int,
    analyzer_timeout_seconds: float,
    embedding_dimensions: int,
    queries: Sequence[str],
    rss_bytes: int,
    database_bytes: int,
    disk_bytes: int,
    no_progress_seconds: float,
    profile: IndexProfile = IndexProfile.NAVIGATION,
    baseline_report: Path | None = None,
    generated_source_roots: Sequence[Path] = (),
    total_gate_timeouts: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    total_timeouts = gate_timeouts if total_gate_timeouts is None else total_gate_timeouts
    for name, value in (
        ("workers", workers),
        ("analyzer timeout", analyzer_timeout_seconds),
        ("embedding dimensions", embedding_dimensions),
        ("RSS limit", rss_bytes),
        ("database limit", database_bytes),
        ("disk limit", disk_bytes),
        ("no-progress timeout", no_progress_seconds),
    ):
        _require_finite_positive(name, value)
    for name, seconds in gate_timeouts.items():
        _require_finite_positive(f"gate {name} timeout", seconds)
    missing_timeouts = [str(gate) for gate in gates if str(gate) not in gate_timeouts]
    if missing_timeouts:
        raise ValueError("no timeout configured for gates: " + ", ".join(missing_timeouts))
    for gate in gates:
        if str(gate) not in total_timeouts:
            raise ValueError(f"no end-to-end timeout configured for gate: {gate}")
        _require_finite_positive(f"gate {gate} end-to-end timeout", total_timeouts[str(gate)])
    profile = IndexProfile(profile)
    inspection = inspect_compilation_database(project_root, compilation_database)
    canonical_generated_roots = _canonical_generated_roots_for_gates(
        inspection, gates, generated_source_roots
    )
    _require_supervisor_platform()
    analyzer = analyzer.expanduser().resolve(strict=True)
    if not analyzer.is_file() or not os.access(analyzer, os.X_OK):
        raise ValueError("Clang analyzer must be an executable file")
    output = output_directory.expanduser().resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError("canary output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    gate_reports: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema": "cpp-context-kicad-canary-report",
        "schema_version": REPORT_SCHEMA_VERSION,
        "measured_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "engine_commit": _git_revision(Path(__file__).resolve().parents[2]),
        "project_commit": _git_revision(inspection.project_root),
        "profile": profile.value,
        "workers": workers,
        "embedding_dimensions": embedding_dimensions,
        "input": inspection.public_report(),
        "gates": gate_reports,
        "baseline_parity": baseline_report is not None,
    }
    analyzer_sha256 = _sha256(analyzer)
    baseline: Mapping[str, Any] | None = None
    baseline_gates: list[Mapping[str, Any]] = []
    if baseline_report is not None:
        loaded_baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
        if not isinstance(loaded_baseline, Mapping):
            raise ValueError("baseline report must be an object")
        baseline = loaded_baseline
        # Validate immutable provenance and the exact requested gate set before indexing.
        requested_report = {**report, "gates": [{"gate": gate} for gate in gates]}
        _compare_baseline_header(requested_report, baseline)
        baseline_gates = _comparable_gates(baseline)
    for gate in gates:
        gate_started = time.monotonic()
        name = str(gate)
        if _sha256(inspection.compilation_database) != inspection.sha256:
            raise RuntimeError("source compilation database changed during the canary")
        selected = select_gate_entries(inspection, gate)
        running = output / f"gate-{name}"
        failed = output / f".gate-{name}.failed"
        running.mkdir()
        running_marker = running / ".running"
        _write_report_atomic(running_marker, "incomplete\n")
        # Setup used to escape the failure boundary and leave an ambiguous live gate.
        with _gate_failure_publication(running, failed):
            subset = running / "compile_commands.json"
            subset_metadata = write_subset_database(
                inspection.compilation_database, selected, subset
            )
            if gate != "all" and (
                subset_metadata.normalized_configuration_count != subset_metadata.raw_entry_count
            ):
                raise RuntimeError("numeric gate did not select unique compiler configurations")
            expected_translation_units = subset_metadata.normalized_configuration_count
            gate_generated_roots = canonical_generated_roots if gate == "all" else ()
            spec = {
                "measurement_provenance": {
                    "engine_commit": report["engine_commit"],
                    "project_commit": report["project_commit"],
                    "source_cdb_sha256": inspection.sha256,
                    "subset_cdb_sha256": subset_metadata.sha256,
                    "analyzer_sha256": analyzer_sha256,
                    "expected_fact_schema_version": SCHEMA_VERSION,
                    "profile": profile.value,
                    "workers": workers,
                    "embedding_dimensions": embedding_dimensions,
                    "generated_source_roots": [str(root) for root in gate_generated_roots],
                },
                "gate_started_monotonic": gate_started,
                "total_wall_seconds": total_timeouts[name],
                "full_project": gate == "all",
                "project_root": str(inspection.project_root),
                "compilation_database": str(subset),
                "database": str(running / "index.db"),
                "analyzer": str(analyzer),
                "translation_units": expected_translation_units,
                "workers": workers,
                "analyzer_timeout_seconds": analyzer_timeout_seconds,
                "embedding_dimensions": embedding_dimensions,
                "queries": list(queries),
                "profile": profile.value,
                "generated_source_roots": [str(root) for root in gate_generated_roots],
            }
            spec_path = running / "worker-spec.json"
            _write_report_atomic(spec_path, json.dumps(spec, sort_keys=True) + "\n")
            limits = CanaryLimits(
                wall_seconds=gate_timeouts[name],
                rss_bytes=rss_bytes,
                database_bytes=database_bytes,
                disk_bytes=disk_bytes,
                no_progress_seconds=no_progress_seconds,
            )
            measured = _run_supervised(spec_path, running, limits, expected_translation_units)
            # Never publish success while an analyzer descendant observed by the supervisor lives.
            if measured.get("process_group_clean") is not True:
                raise RuntimeError("canary worker process tree was not cleaned up")
            _validate_analyzer_pipeline_report(
                measured.get("analyzer_pipeline"),
                expected_configurations=expected_translation_units,
                expected_slots=min(workers, expected_translation_units),
                max_idle_seconds=limits.no_progress_seconds,
            )
            if measured["completed_translation_units"] != expected_translation_units:
                raise RuntimeError("worker did not stage every selected translation unit")
            gate_report = {
                "gate": gate,
                "selection": "complete_cdb" if gate == "all" else "project_source_prefix",
                "raw_cdb_entries": subset_metadata.raw_entry_count,
                "translation_units": expected_translation_units,
                "selected_raw_indices": [entry.raw_index for entry in selected],
                "selected_sources": [entry.display_path for entry in selected],
                "subset_cdb_sha256": subset_metadata.sha256,
                "limits": asdict(limits),
                "total_wall_seconds": total_timeouts[name],
                **measured,
            }
            validation_spec = {
                **spec,
                "supervision_stage": "validation",
                "gate_report": gate_report,
                "subset_metadata": asdict(subset_metadata),
                "source_compilation_database": str(inspection.compilation_database),
                "source_cdb_sha256": inspection.sha256,
                "baseline_gate": baseline_gates[len(gate_reports)]
                if baseline is not None
                else None,
            }
            validation_path = running / "validation-spec.json"
            _write_report_atomic(
                validation_path, json.dumps(validation_spec, sort_keys=True) + "\n"
            )
            validation = _run_validation_supervised(
                validation_path,
                running,
                replace(limits, wall_seconds=total_timeouts[name]),
                expected_translation_units,
            )
            if not validation.get("validated") or not validation.get("process_group_clean"):
                raise RuntimeError("independent validator did not finish cleanly")
            gate_report["validation_elapsed_seconds"] = validation["elapsed_seconds"]
            gate_report["validation_phase_measurements"] = validation.get("phase_measurements")
            gate_report["total_elapsed_seconds"] = validation["total_elapsed_seconds"]
            for field in (
                "peak_rss_bytes",
                "peak_swap_bytes",
                "peak_database_bytes",
                "peak_disk_bytes",
            ):
                gate_report[field] = max(measured.get(field, 0), validation[field])
            gate_report["artifact_directory_peak_bytes"] = gate_report["peak_disk_bytes"]
            gate_reports.append(gate_report)
    if baseline is not None:
        _compare_baseline(report, baseline)
    _write_report_atomic(
        output / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpp-context-kicad-canary",
        description="Run fail-fast progressive navigation gates against a real CDB.",
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--compile-commands", type=Path)
    parser.add_argument("--clang-analyzer", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--profile", choices=("navigation", "full"), default="navigation")
    parser.add_argument("--gates", default="1,4,16,32")
    parser.add_argument("--gate-timeouts", default="")
    parser.add_argument("--total-gate-timeouts", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analyzer-timeout-seconds", type=float, default=75.0)
    parser.add_argument("--embedding-dimensions", type=int, default=32)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--rss-limit-mib", type=float, default=2.5 * 1024)
    parser.add_argument("--database-limit-mib", type=float)
    parser.add_argument("--disk-limit-mib", type=float)
    parser.add_argument("--no-progress-seconds", type=float, default=10)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument(
        "--generated-source-root",
        action="append",
        default=[],
        type=Path,
        help="explicit generated-source directory for the all gate; repeat as needed",
    )
    parser.add_argument("--_worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker_spec is not None:
        return _run_worker(args._worker_spec)
    try:
        if args.project_root is None or args.compile_commands is None:
            raise ValueError("--project-root and --compile-commands are required")
        if args.preflight_only:
            inspection = inspect_compilation_database(args.project_root, args.compile_commands)
            _canonical_generated_roots_for_gates(
                inspection, _parse_gates(args.gates), args.generated_source_root
            )
            print(json.dumps(inspection.public_report(), indent=2, sort_keys=True))
            return 0
        if args.clang_analyzer is None or args.output_directory is None:
            raise ValueError("--clang-analyzer and --output-directory are required for a run")
        profile = IndexProfile(args.profile)
        gates = _parse_gates(args.gates)
        if "all" in gates and (args.database_limit_mib is None or args.disk_limit_mib is None):
            raise ValueError("all gate requires explicit --database-limit-mib and --disk-limit-mib")
        database_limit_mib = args.database_limit_mib
        if database_limit_mib is None:
            database_limit_mib = 550 if profile is IndexProfile.NAVIGATION else 1_350
        disk_limit_mib = args.disk_limit_mib
        if disk_limit_mib is None:
            disk_limit_mib = 1_024 if profile is IndexProfile.NAVIGATION else 2_048
        for name, value in (
            ("workers", args.workers),
            ("analyzer timeout", args.analyzer_timeout_seconds),
            ("embedding dimensions", args.embedding_dimensions),
            ("RSS limit", args.rss_limit_mib),
            ("database limit", database_limit_mib),
            ("disk limit", disk_limit_mib),
            ("no-progress timeout", args.no_progress_seconds),
        ):
            _require_finite_positive(name, value)
        timeouts = (
            _parse_timeouts(args.gate_timeouts)
            if args.gate_timeouts
            else dict(DEFAULT_GATE_TIMEOUTS)
        )
        total_timeouts = (
            _parse_timeouts(args.total_gate_timeouts)
            if args.total_gate_timeouts
            else dict(DEFAULT_TOTAL_GATE_TIMEOUTS)
        )
        missing = [str(gate) for gate in gates if str(gate) not in timeouts]
        if missing:
            raise ValueError("no timeout configured for gates: " + ", ".join(missing))
        report = run_canary(
            project_root=args.project_root,
            compilation_database=args.compile_commands,
            analyzer=args.clang_analyzer,
            output_directory=args.output_directory,
            gates=gates,
            gate_timeouts=timeouts,
            workers=args.workers,
            analyzer_timeout_seconds=args.analyzer_timeout_seconds,
            embedding_dimensions=args.embedding_dimensions,
            queries=tuple(args.queries or DEFAULT_QUERIES),
            rss_bytes=int(args.rss_limit_mib * 1024**2),
            database_bytes=int(database_limit_mib * 1024**2),
            disk_bytes=int(disk_limit_mib * 1024**2),
            no_progress_seconds=args.no_progress_seconds,
            profile=profile,
            baseline_report=args.baseline_report,
            generated_source_roots=args.generated_source_root,
            total_gate_timeouts=total_timeouts,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
