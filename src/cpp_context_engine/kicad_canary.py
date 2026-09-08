"""Fail-fast local acceptance harness for the real KiCad compilation database."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import queue
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from cpp_context_engine.api import QueryRequest
from cpp_context_engine.benchmark import _write_report_atomic
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion import NativeAnalyzerClient, NativeClangIngestor, ProjectIndexer
from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.models import BuildScope, BuildVariant, IndexProfile
from cpp_context_engine.runtime import build_runtime
from cpp_context_engine.search import DeterministicLocalEmbeddingProvider, SQLiteVectorSearch
from cpp_context_engine.storage import SQLiteStore

REPORT_SCHEMA_VERSION = 1
DEFAULT_GATE_TIMEOUTS: Mapping[str, float] = {
    "1": 60.0,
    "4": 90.0,
    "16": 120.0,
    "32": 150.0,
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

    def public_report(self) -> dict[str, Any]:
        return {
            "schema": "cpp-context-kicad-cdb-preflight",
            "schema_version": REPORT_SCHEMA_VERSION,
            "cdb_sha256": self.sha256,
            "entry_count": self.entry_count,
            "normalized_configuration_count": self.normalized_configuration_count,
            "classification_counts": self.classification_counts,
            "numeric_gate_eligible_count": self.classification_counts["project_source"],
            "external_generated_note": (
                "Out-of-tree entries are valid CMake inputs and are retained by the all gate; "
                "numeric canaries deliberately select project-root sources only."
            ),
        }


@dataclass(frozen=True, slots=True)
class CanaryLimits:
    wall_seconds: float
    rss_bytes: int = int(2.5 * 1024**3)
    database_bytes: int = 550 * 1024**2
    disk_bytes: int = 1024**3
    no_progress_seconds: float = 10.0

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    eligible = tuple(
        entry for entry in inspection.entries if entry.classification == "project_source"
    )
    if len(eligible) < count:
        raise ValueError(f"gate {count} needs {count} project-root TUs, found {len(eligible)}")
    return eligible[:count]


def write_subset_database(
    source_database: Path, entries: Sequence[CdbEntry], destination: Path
) -> str:
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
    if len(normalized.configurations) != len(entries):
        raise ValueError("selected CDB contains duplicate compiler configurations")
    return hashlib.sha256(document.encode()).hexdigest()


def _encode_digest_value(value: Any) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bytes):
        return b"b" + len(value).to_bytes(8, "big") + value
    encoded = str(value).encode("utf-8", errors="surrogateescape")
    return b"s" + len(encoded).to_bytes(8, "big") + encoded


def semantic_snapshot(database: Path) -> dict[str, Any]:
    """Hash stable semantic rows, excluding timestamps and artifact-location columns."""

    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    table_digests: dict[str, str] = {}
    with sqlite3.connect(database) as connection:
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


def database_provenance(database: Path) -> dict[str, Any]:
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
    with sqlite3.connect(database) as connection:
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
    return sum(
        path.stat().st_size
        for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm"))
        if path.is_file()
    )


def _directory_size(directory: Path) -> int:
    total = 0
    try:
        paths = directory.rglob("*")
        for path in paths:
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


@dataclass(frozen=True, slots=True)
class _TreeMetrics:
    rss: int
    swap: int
    cpu_ticks: int
    live_pids: tuple[int, ...]


def _process_tree_metrics(root_pid: int) -> _TreeMetrics:
    processes: dict[int, tuple[int, int, int, int, str]] = {}
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
    return _TreeMetrics(
        sum(processes[pid][1] for pid in live),
        sum(processes[pid][2] for pid in live),
        sum(processes[pid][3] for pid in live),
        live,
    )


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
    known_groups: Iterable[int] = (),
) -> None:
    groups = {
        process.pid,
        *known_groups,
        *_process_groups(_process_tree_metrics(process.pid).live_pids),
    }
    for group in groups:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(_process_group_live(group) for group in groups):
        time.sleep(0.05)
    for group in groups:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
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


def _process_groups(pids: Iterable[int]) -> set[int]:
    groups: set[int] = set()
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            groups.add(os.getpgid(pid))
    return groups


def _run_supervised(
    spec_path: Path,
    gate_directory: Path,
    limits: CanaryLimits,
    total_tus: int,
) -> dict[str, Any]:
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
    started = time.monotonic()
    last_report = started
    last_useful = started
    last_signature: tuple[int, int, int] | None = None
    completed_tus = 0
    peak_rss = peak_swap = peak_database = peak_disk = 0
    stderr_tail: list[str] = []
    worker_result: dict[str, Any] | None = None
    violation: str | None = None
    stdout_eof = stderr_eof = False
    observed_groups = {process.pid}
    database = gate_directory / "index.db"
    try:
        while process.poll() is None or not (stdout_eof and stderr_eof):
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
                    if event.get("event") == "tu_staged":
                        completed_tus = int(event["completed"])
                    elif event.get("event") == "result":
                        worker_result = event["result"]
                    elif event.get("event") == "error":
                        violation = str(event.get("message", "worker failed"))
            elapsed = time.monotonic() - started
            tree = _process_tree_metrics(process.pid)
            # Keep only the immediately observed live tree. Once the worker exits,
            # retain the final sample long enough to kill a just-reparented analyzer.
            if process.poll() is None:
                observed_groups = {process.pid, *_process_groups(tree.live_pids)}
            database_bytes = _database_size(database)
            disk_bytes = _directory_size(gate_directory)
            peak_rss = max(peak_rss, tree.rss)
            peak_swap = max(peak_swap, tree.swap)
            peak_database = max(peak_database, database_bytes)
            peak_disk = max(peak_disk, disk_bytes)
            current = limits.violation(
                elapsed=elapsed,
                rss=tree.rss,
                swap=tree.swap,
                database=database_bytes,
                disk=disk_bytes,
            )
            if violation is None:
                violation = current
            signature = (completed_tus, database_bytes, tree.cpu_ticks)
            if signature != last_signature:
                last_signature = signature
                last_useful = time.monotonic()
            elif (
                process.poll() is None
                and time.monotonic() - last_useful > limits.no_progress_seconds
            ):
                violation = f"no observable progress for {limits.no_progress_seconds:g} seconds"
            if completed_tus:
                projected = elapsed * total_tus / completed_tus
                if elapsed >= 1_800 and projected > 3_600:
                    violation = "30-minute projection exceeds the 60-minute target"
                elif elapsed >= 600 and projected > 5_400:
                    violation = "10-minute projection exceeds the 90-minute hard limit"
            if time.monotonic() - last_report >= 5 or kind == "stdout" and completed_tus:
                rate = completed_tus / elapsed if elapsed > 0 else 0.0
                eta = (total_tus - completed_tus) / rate if rate > 0 else None
                eta_text = f"{eta:.1f}s" if eta is not None else "unknown"
                print(
                    f"canary: {completed_tus}/{total_tus} TUs, {elapsed:.1f}s, ETA {eta_text}, "
                    f"RSS {tree.rss / 1024**2:.1f} MiB, DB {database_bytes / 1024**2:.1f} MiB",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = time.monotonic()
            if violation is not None and process.poll() is None:
                _terminate_process_group(process, known_groups=observed_groups)
        return_code = process.wait(timeout=2)
    finally:
        _terminate_process_group(process, known_groups=observed_groups)
        for reader in readers:
            reader.join(timeout=1)
    if violation is not None:
        detail = stderr_tail[-1] if stderr_tail else ""
        raise RuntimeError(f"{violation}" + (f": {detail}" if detail else ""))
    if return_code != 0 or worker_result is None:
        detail = stderr_tail[-1] if stderr_tail else f"exit {return_code}"
        raise RuntimeError(f"canary worker failed: {detail}")
    return {
        **worker_result,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "peak_swap_bytes": peak_swap,
        "peak_database_bytes": peak_database,
        "peak_disk_bytes": peak_disk,
        "completed_translation_units": completed_tus,
        "process_group_clean": not any(_process_group_live(group) for group in observed_groups),
    }


class _ObservedIngestor:
    def __init__(self, delegate: NativeClangIngestor, total: int) -> None:
        self.delegate = delegate
        self.total = total
        self.analysis_backend = delegate.analysis_backend
        self.advanced_facts_complete = delegate.advanced_facts_complete

    def iter_configuration_batches(
        self, project_root: Path, configurations: Iterable[Any]
    ) -> Iterable[Any]:
        batches = self.delegate.iter_configuration_batches(project_root, configurations)
        for completed, batch in enumerate(batches, start=1):
            yield batch
            _worker_event("tu_staged", completed=completed, total=self.total)


def _worker_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _ranking_canaries(config: AppConfig, queries: Sequence[str]) -> dict[str, Any]:
    rankings: dict[str, Any] = {}
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
    return rankings


def _run_worker(spec_path: Path) -> int:
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
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
        )
        variant = BuildVariant("default", cdb)
        scope = BuildScope((variant.name,))
        _worker_event("phase", name="index")
        with SQLiteStore(database, project_root=project, build_scope=scope) as store:
            indexing = ProjectIndexer(
                _ObservedIngestor(ingestor, total), store, profile=profile
            ).index(project, cdb, build_variant=variant)
            _worker_event("phase", name="embeddings")
            embedded = SQLiteVectorSearch(
                store,
                DeterministicLocalEmbeddingProvider(int(spec["embedding_dimensions"])),
                project_root=project,
                build_scope=scope,
            ).index_missing()
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
        rankings = _ranking_canaries(config, tuple(spec["queries"]))
        snapshot = semantic_snapshot(database)
        provenance = database_provenance(database)
        with sqlite3.connect(database) as connection:
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchone()
        if integrity != "ok" or foreign_keys is not None:
            raise RuntimeError("canary database failed post-run integrity checks")
        _worker_event(
            "result",
            result={
                "indexing": asdict(indexing),
                "embedded_symbols": embedded,
                "rankings": rankings,
                "semantic_snapshot": snapshot,
                "database_provenance": provenance,
                "database_file_sha256": _sha256(database),
                "database_integrity": integrity,
                "analyzer": {
                    "version": info.analyzer_version,
                    "protocol": info.protocol,
                    "protocol_version": info.protocol_version,
                    "clang_major": info.clang_major,
                    "capabilities": sorted(info.capabilities),
                    "sha256": _sha256(analyzer),
                },
            },
        )
        return 0
    except BaseException as error:
        _worker_event("error", message=f"{type(error).__name__}: {error}")
        return 2


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
        if seconds <= 0:
            raise ValueError("gate timeouts must be positive")
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


def _compare_baseline(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    for field in ("engine_commit", "project_commit", "profile", "embedding_dimensions"):
        if report.get(field) != baseline.get(field):
            raise RuntimeError(f"baseline provenance differs for {field}")
    current_input = report.get("input")
    baseline_input = baseline.get("input")
    if not isinstance(current_input, Mapping) or not isinstance(baseline_input, Mapping):
        raise ValueError("baseline report has no input provenance")
    if current_input.get("cdb_sha256") != baseline_input.get("cdb_sha256"):
        raise RuntimeError("baseline compilation database digest differs")
    current_gates = report.get("gates")
    baseline_gates = baseline.get("gates")
    if not isinstance(current_gates, list) or not isinstance(baseline_gates, list):
        raise ValueError("baseline report has no comparable gates")
    expected = {
        str(gate["gate"]): (gate["semantic_snapshot"], gate["rankings"]) for gate in baseline_gates
    }
    for gate in current_gates:
        key = str(gate["gate"])
        if key not in expected:
            raise RuntimeError(f"baseline has no gate {key}")
        baseline_gate = next(item for item in baseline_gates if str(item["gate"]) == key)
        for field in ("selection", "selected_raw_indices", "subset_cdb_sha256"):
            if gate[field] != baseline_gate[field]:
                raise RuntimeError(f"baseline gate {key} differs for {field}")
        if (gate["semantic_snapshot"], gate["rankings"]) != expected[key]:
            raise RuntimeError(f"semantic or ranking parity failed for gate {key}")


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
) -> dict[str, Any]:
    profile = IndexProfile(profile)
    inspection = inspect_compilation_database(project_root, compilation_database)
    analyzer = analyzer.expanduser().resolve(strict=True)
    if not analyzer.is_file() or not os.access(analyzer, os.X_OK):
        raise ValueError("Clang analyzer must be an executable file")
    output = output_directory.expanduser().resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError("canary output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    gate_reports: list[dict[str, Any]] = []
    for gate in gates:
        name = str(gate)
        if _sha256(inspection.compilation_database) != inspection.sha256:
            raise RuntimeError("source compilation database changed during the canary")
        selected = select_gate_entries(inspection, gate)
        running = output / f"gate-{name}"
        failed = output / f".gate-{name}.failed"
        running.mkdir()
        running_marker = running / ".running"
        _write_report_atomic(running_marker, "incomplete\n")
        subset = running / "compile_commands.json"
        subset_hash = write_subset_database(inspection.compilation_database, selected, subset)
        spec = {
            "project_root": str(inspection.project_root),
            "compilation_database": str(subset),
            "database": str(running / "index.db"),
            "analyzer": str(analyzer),
            "translation_units": len(selected),
            "workers": workers,
            "analyzer_timeout_seconds": analyzer_timeout_seconds,
            "embedding_dimensions": embedding_dimensions,
            "queries": list(queries),
            "profile": profile.value,
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
        try:
            measured = _run_supervised(spec_path, running, limits, len(selected))
            if measured["completed_translation_units"] != len(selected):
                raise RuntimeError("worker did not stage every selected translation unit")
            _validate_profile_provenance(measured["database_provenance"], profile, len(selected))
            running_marker.unlink()
            _write_report_atomic(running / "SUCCESS", "complete\n")
        except BaseException:
            if running.exists():
                running.rename(failed)
            raise
        gate_reports.append(
            {
                "gate": gate,
                "selection": "complete_cdb" if gate == "all" else "project_source_prefix",
                "translation_units": len(selected),
                "selected_raw_indices": [entry.raw_index for entry in selected],
                "selected_sources": [entry.display_path for entry in selected],
                "subset_cdb_sha256": subset_hash,
                "limits": asdict(limits),
                **measured,
            }
        )
    if _sha256(inspection.compilation_database) != inspection.sha256:
        raise RuntimeError("source compilation database changed during the canary")
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
    if baseline_report is not None:
        baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analyzer-timeout-seconds", type=float, default=75.0)
    parser.add_argument("--embedding-dimensions", type=int, default=32)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--rss-limit-mib", type=float, default=2.5 * 1024)
    parser.add_argument("--database-limit-mib", type=float)
    parser.add_argument("--disk-limit-mib", type=float)
    parser.add_argument("--no-progress-seconds", type=float, default=10)
    parser.add_argument("--baseline-report", type=Path)
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
            print(json.dumps(inspection.public_report(), indent=2, sort_keys=True))
            return 0
        if args.clang_analyzer is None or args.output_directory is None:
            raise ValueError("--clang-analyzer and --output-directory are required for a run")
        profile = IndexProfile(args.profile)
        database_limit_mib = args.database_limit_mib
        if database_limit_mib is None:
            database_limit_mib = 550 if profile is IndexProfile.NAVIGATION else 1_350
        disk_limit_mib = args.disk_limit_mib
        if disk_limit_mib is None:
            disk_limit_mib = 1_024 if profile is IndexProfile.NAVIGATION else 2_048
        if (
            min(
                args.workers,
                args.analyzer_timeout_seconds,
                args.embedding_dimensions,
                args.rss_limit_mib,
                database_limit_mib,
                disk_limit_mib,
                args.no_progress_seconds,
            )
            <= 0
        ):
            raise ValueError("canary limits and worker count must be positive")
        gates = _parse_gates(args.gates)
        timeouts = (
            _parse_timeouts(args.gate_timeouts)
            if args.gate_timeouts
            else dict(DEFAULT_GATE_TIMEOUTS)
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
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
