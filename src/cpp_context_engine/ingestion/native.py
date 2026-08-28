"""Bounded adapter for the versioned Clang LibTooling companion protocol."""

from __future__ import annotations

import hashlib
import json
import marshal
import os
import signal
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from io import BufferedRandom
from pathlib import Path
from typing import Any

from cpp_context_engine.analysis.interprocedural import InterproceduralLimits, solve_interprocedural
from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    libclang_arguments,
    translation_unit_id,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    CallArgumentBinding,
    CallDispatchKind,
    CallResultBinding,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    CfgBlock,
    CfgBlockRole,
    CfgEdge,
    CfgEdgeKind,
    CfgElement,
    CfgGraph,
    CodeSymbol,
    DataAccess,
    DataAccessKind,
    DataFlowAnalysis,
    DataFlowCertainty,
    DataFlowEvidence,
    DataFlowRelation,
    FunctionSummary,
    GraphEdge,
    GraphRelation,
    IndexProfile,
    MacroExpansionFrame,
    MemoryLocation,
    MemoryLocationKind,
    OccurrenceKind,
    SourceSpan,
    SummaryEffect,
    SummaryEffectKind,
    SummaryReturnOrigin,
    SummaryReturnOriginKind,
    SymbolKind,
    SymbolOccurrence,
    TranslationUnit,
)

PROTOCOL = "cpp-context-clang-facts"
PROTOCOL_VERSION = 5
REQUIRED_CLANG_MAJOR = 18
REQUIRED_CAPABILITIES = frozenset(
    {
        "direct_calls",
        "full_ast",
        "includes",
        "inherits",
        "lambda_metadata",
        "macro_provenance",
        "occurrences",
        "overrides",
        "pp_callbacks",
        "source_manager",
        "symbols",
        "template_metadata",
        "uses_type",
        "function_cfg_v1",
        "callsites_v1",
        "dispatch_targets_v1",
        "macro_expansion_stack",
        "template_relationships_v1",
        "intraprocedural_dataflow_v1",
        "points_to_v1",
        "function_summaries_v1",
        "interprocedural_bindings_v1",
    }
)
DEFAULT_TIMEOUT_SECONDS = 75.0
DEFAULT_MAX_INPUT_BYTES = 1_048_576
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1_048_576
DEFAULT_MAX_DECODED_BYTES = 256 * 1_048_576
DEFAULT_MAX_RECORD_BYTES = 16 * 1_048_576
DEFAULT_MAX_STDERR_BYTES = 256 * 1024
GZIP_TRANSPORT = "gzip_jsonl_v1"
PROFILE_CAPABILITY = "analysis_profiles_v1"
MAX_FACT_KINDS = 64
_FRAME_HEADER = struct.Struct(">I")


class AnalyzerUnavailableError(RuntimeError):
    """Raised when a configured companion cannot be safely executed."""


class AnalyzerProtocolError(RuntimeError):
    """Raised for a mismatched, malformed, or incomplete companion response."""


class AnalyzerLimitError(RuntimeError):
    """Raised when the companion exceeds an operator-owned resource bound."""


@dataclass(frozen=True, slots=True)
class AnalyzerInfo:
    protocol: str
    protocol_version: int
    analyzer_version: str
    clang_major: int
    capabilities: frozenset[str]


class _JsonlStreamDecoder:
    """Incrementally decompress and parse one bounded JSONL response stream."""

    def __init__(
        self,
        *,
        gzip_transport: bool,
        max_wire_bytes: int,
        max_decoded_bytes: int,
        max_record_bytes: int,
        on_record: Callable[[dict[str, Any]], None],
    ) -> None:
        self._decompressor = zlib.decompressobj(wbits=31) if gzip_transport else None
        self._max_wire_bytes = max_wire_bytes
        self._max_decoded_bytes = max_decoded_bytes
        self._max_record_bytes = max_record_bytes
        self._on_record = on_record
        self._wire_bytes = 0
        self._decoded_bytes = 0
        self._record = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._wire_bytes += len(chunk)
        if self._wire_bytes > self._max_wire_bytes:
            raise AnalyzerLimitError("analyzer exceeded the compressed output limit")
        if self._decompressor is None:
            self._feed_decoded(chunk)
            return
        pending = chunk
        try:
            while pending:
                remaining = self._max_decoded_bytes - self._decoded_bytes
                decoded = self._decompressor.decompress(pending, remaining + 1)
                self._feed_decoded(decoded)
                pending = self._decompressor.unconsumed_tail
                if pending and self._decoded_bytes >= self._max_decoded_bytes:
                    raise AnalyzerLimitError("analyzer exceeded the decoded output limit")
        except zlib.error as error:
            raise AnalyzerProtocolError("analyzer returned malformed gzip data") from error

    def finish(self) -> None:
        if self._decompressor is not None:
            try:
                remaining = self._max_decoded_bytes - self._decoded_bytes
                self._feed_decoded(self._decompressor.flush(remaining + 1))
            except zlib.error as error:
                raise AnalyzerProtocolError("analyzer returned malformed gzip data") from error
            if not self._decompressor.eof or self._decompressor.unused_data:
                raise AnalyzerProtocolError("analyzer returned malformed gzip data")
        if self._record:
            raise AnalyzerProtocolError("analyzer returned unterminated JSONL")

    def _feed_decoded(self, chunk: bytes) -> None:
        self._decoded_bytes += len(chunk)
        if self._decoded_bytes > self._max_decoded_bytes:
            raise AnalyzerLimitError("analyzer exceeded the decoded output limit")
        parts = chunk.split(b"\n")
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                self._record.extend(part)
                if len(self._record) > self._max_record_bytes:
                    raise AnalyzerLimitError("analyzer exceeded the record limit")
                continue
            self._record.extend(part)
            if not self._record:
                raise AnalyzerProtocolError("analyzer returned malformed JSONL")
            if len(self._record) > self._max_record_bytes:
                raise AnalyzerLimitError("analyzer exceeded the record limit")
            try:
                record = json.loads(self._record)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AnalyzerProtocolError("analyzer returned malformed JSONL") from error
            if not isinstance(record, dict):
                raise AnalyzerProtocolError("analyzer JSONL records must be objects")
            self._record.clear()
            self._on_record(record)


class _ResourceBudget:
    """Thread-safe hard bound shared by all registries in one ingestion pipeline."""

    def __init__(self, limit: int, message: str) -> None:
        if limit <= 0:
            raise ValueError("registry resource limits must be positive")
        self.limit = limit
        self._message = message
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def acquire(self, amount: int = 1) -> None:
        with self._lock:
            if self._used + amount > self.limit:
                raise AnalyzerLimitError(self._message)
            self._used += amount

    def release(self, amount: int = 1) -> None:
        with self._lock:
            self._used -= amount
            if self._used < 0:  # pragma: no cover - internal invariant
                raise RuntimeError("registry resource budget underflow")


class _FactRegistry:
    """Bounded compact fact-kind spools exposed only after successful analysis."""

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_DECODED_BYTES,
        max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
        byte_budget: _ResourceBudget | None = None,
        fd_budget: _ResourceBudget | None = None,
        cancelled: threading.Event | None = None,
    ) -> None:
        if min(max_bytes, max_record_bytes) <= 0:
            raise ValueError("registry byte limits must be positive")
        self._directory = "/tmp" if os.name != "nt" and Path("/tmp").is_dir() else None
        self._files: dict[str, BufferedRandom] = {}
        self._max_bytes = max_bytes
        self._max_record_bytes = max_record_bytes
        self._byte_budget = byte_budget or _ResourceBudget(
            max_bytes, "analyzer pipeline exceeded the spool byte limit"
        )
        self._fd_budget = fd_budget or _ResourceBudget(
            MAX_FACT_KINDS, "analyzer pipeline exceeded the spool file limit"
        )
        self._cancelled = cancelled
        self._bytes = 0
        self._closed = False

    @property
    def byte_count(self) -> int:
        return self._bytes

    def add(self, fact: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("fact registry is closed")
        fact_kind = fact.get("fact")
        if not isinstance(fact_kind, str) or not fact_kind:
            raise AnalyzerProtocolError("analyzer fact has no valid kind")
        destination = self._files.get(fact_kind)
        if destination is None:
            # One file per arbitrary kind would let tiny malformed records exhaust
            # descriptors long before the decoded-byte limit is reached.
            if len(self._files) >= MAX_FACT_KINDS:
                raise AnalyzerLimitError("analyzer exceeded the fact-kind registry limit")
            self._fd_budget.acquire()
            try:
                destination = tempfile.TemporaryFile(  # noqa: SIM115 - registry owns lifetime
                    mode="w+b", dir=self._directory
                )
            except BaseException:
                self._fd_budget.release()
                raise
            self._files[fact_kind] = destination
        try:
            payload = marshal.dumps(dict(fact), 4)
        except (TypeError, ValueError) as error:
            raise AnalyzerProtocolError("analyzer fact contains unsupported values") from error
        if len(payload) > self._max_record_bytes:
            raise AnalyzerLimitError("analyzer exceeded the record limit")
        frame_bytes = _FRAME_HEADER.size + len(payload)
        if self._bytes + frame_bytes > self._max_bytes:
            raise AnalyzerLimitError("analyzer exceeded the registry spool limit")
        self._byte_budget.acquire(frame_bytes)
        try:
            destination.write(_FRAME_HEADER.pack(len(payload)))
            destination.write(payload)
        except BaseException:
            self._byte_budget.release(frame_bytes)
            raise
        self._bytes += frame_bytes

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for fact_kind in sorted(self._files):
            yield from self.records(fact_kind)

    def records(self, fact_kind: str) -> Iterator[Mapping[str, Any]]:
        source = self._files.get(fact_kind)
        if source is None:
            return
        source.seek(0)
        while header := source.read(_FRAME_HEADER.size):
            if self._cancelled is not None and self._cancelled.is_set():
                raise AnalyzerLimitError("analyzer analysis was cancelled")
            if len(header) != _FRAME_HEADER.size:
                raise AnalyzerProtocolError("fact registry has a truncated frame header")
            (size,) = _FRAME_HEADER.unpack(header)
            if size == 0 or size > self._max_record_bytes:
                raise AnalyzerProtocolError("fact registry has an invalid frame length")
            payload = source.read(size)
            if len(payload) != size:
                raise AnalyzerProtocolError("fact registry has a truncated frame")
            try:
                record = marshal.loads(payload)
            except (EOFError, TypeError, ValueError) as error:
                raise AnalyzerProtocolError("fact registry contains an invalid frame") from error
            if not isinstance(record, dict) or record.get("fact") != fact_kind:
                raise AnalyzerProtocolError("fact registry contains an invalid fact record")
            yield record

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for source in self._files.values():
            source.close()
            self._fd_budget.release()
        self._files.clear()
        self._byte_budget.release(self._bytes)
        self._bytes = 0

    def __enter__(self) -> _FactRegistry:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _fact_records(
    facts: Iterable[Mapping[str, Any]], fact_kind: str
) -> Iterable[Mapping[str, Any]]:
    if isinstance(facts, _FactRegistry):
        return facts.records(fact_kind)
    return (fact for fact in facts if fact.get("fact") == fact_kind)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_text(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class NativeAnalyzerClient:
    """Execute one explicitly configured binary without a shell or unbounded pipes."""

    def __init__(
        self,
        binary: Path,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES,
        max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
        max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
        prefer_compression: bool = True,
        profile: IndexProfile = IndexProfile.FULL,
    ) -> None:
        self.binary = binary.expanduser().resolve(strict=False)
        if (
            timeout_seconds <= 0
            or min(
                max_input_bytes,
                max_output_bytes,
                max_decoded_bytes,
                max_record_bytes,
                max_stderr_bytes,
            )
            <= 0
        ):
            raise ValueError("analyzer timeout and byte limits must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_decoded_bytes = max_decoded_bytes
        self.max_record_bytes = max_record_bytes
        self.max_stderr_bytes = max_stderr_bytes
        self.prefer_compression = prefer_compression
        self.profile = IndexProfile(profile)
        self._info: AnalyzerInfo | None = None

    def probe(self, *, refresh: bool = False) -> AnalyzerInfo:
        if self._info is not None and not refresh:
            return self._info
        records = self._invoke(
            (self._hello(),), output_limit=min(256 * 1024, self.max_output_bytes)
        )
        if len(records) != 1 or records[0].get("type") != "hello":
            raise AnalyzerProtocolError("analyzer handshake returned an invalid response")
        info = self._validate_handshake(records[0])
        self._info = info
        return info

    @staticmethod
    def _validate_handshake(record: Mapping[str, Any]) -> AnalyzerInfo:
        capabilities_raw = record.get("capabilities")
        if not isinstance(capabilities_raw, list) or not all(
            isinstance(item, str) for item in capabilities_raw
        ):
            raise AnalyzerProtocolError("analyzer handshake has invalid capabilities")
        info = AnalyzerInfo(
            protocol=_string(record, "protocol"),
            protocol_version=_integer(record, "protocol_version"),
            analyzer_version=_string(record, "analyzer_version"),
            clang_major=_integer(record, "clang_major"),
            capabilities=frozenset(capabilities_raw),
        )
        if info.protocol != PROTOCOL or info.protocol_version != PROTOCOL_VERSION:
            raise AnalyzerProtocolError(
                f"analyzer protocol mismatch; expected {PROTOCOL} version {PROTOCOL_VERSION}"
            )
        if info.clang_major != REQUIRED_CLANG_MAJOR:
            raise AnalyzerProtocolError(
                f"analyzer Clang major mismatch; expected {REQUIRED_CLANG_MAJOR}"
            )
        missing = REQUIRED_CAPABILITIES - info.capabilities
        if missing:
            raise AnalyzerProtocolError(
                "analyzer is missing required capabilities: " + ", ".join(sorted(missing))
            )
        return info

    def analyze(
        self, project_root: Path, configuration: BuildConfiguration
    ) -> tuple[Mapping[str, Any], ...]:
        facts: list[Mapping[str, Any]] = []
        self.analyze_stream(project_root, configuration, facts.append)
        return tuple(facts)

    def analyze_stream(
        self,
        project_root: Path,
        configuration: BuildConfiguration,
        on_fact: Callable[[Mapping[str, Any]], None],
        *,
        cancelled: threading.Event | None = None,
    ) -> None:
        """Validate one response while forwarding facts without retaining raw output."""

        info = self.probe()
        if self.profile is IndexProfile.NAVIGATION and PROFILE_CAPABILITY not in info.capabilities:
            raise AnalyzerProtocolError("analyzer does not support the navigation profile")
        unit_id = translation_unit_id(configuration)
        transport = (
            GZIP_TRANSPORT
            if self.prefer_compression and GZIP_TRANSPORT in info.capabilities
            else "plain_jsonl_v1"
        )
        request = {
            "type": "analyze",
            "request_id": unit_id,
            "project_root": str(project_root.resolve(strict=False)),
            "source_path": str(configuration.source_path),
            "directory": str(configuration.directory),
            "arguments": list(libclang_arguments(configuration)),
        }
        # Omit the default so existing protocol-v5 companions keep accepting full requests.
        if self.profile is IndexProfile.NAVIGATION:
            request["profile"] = self.profile.value
        state = "hello"

        def accept(record: dict[str, Any]) -> None:
            nonlocal state
            record_type = record.get("type")
            if state == "hello":
                if record_type != "hello":
                    raise AnalyzerProtocolError("analyzer did not repeat its validated handshake")
                if self._validate_handshake(record) != info:
                    raise AnalyzerProtocolError("analyzer handshake changed between invocations")
                state = "begin"
                return
            if state == "begin":
                if record != {"request_id": unit_id, "type": "begin"}:
                    raise AnalyzerProtocolError("analyzer response has no matching begin record")
                state = "facts"
                return
            if state != "facts":
                raise AnalyzerProtocolError("analyzer emitted records after completion")
            if record_type == "fact":
                on_fact(record)
                return
            if record_type != "complete" or record.get("request_id") != unit_id:
                raise AnalyzerProtocolError("analyzer emitted a non-fact record during analysis")
            if record.get("success") is not True:
                raise AnalyzerProtocolError("analyzer did not complete successfully")
            state = "complete"

        self._invoke(
            (self._hello(response_transport=transport), request),
            output_limit=self.max_output_bytes,
            transport=transport,
            on_record=accept,
            cancelled=cancelled,
        )
        if state != "complete":
            raise AnalyzerProtocolError("analyzer response is incomplete")

    @staticmethod
    def _hello(*, response_transport: str | None = None) -> dict[str, Any]:
        hello: dict[str, Any] = {
            "type": "hello",
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "required_clang_major": REQUIRED_CLANG_MAJOR,
            "required_capabilities": sorted(REQUIRED_CAPABILITIES),
        }
        if response_transport == GZIP_TRANSPORT:
            hello["response_transport"] = GZIP_TRANSPORT
        return hello

    def _invoke(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        output_limit: int,
        transport: str = "plain_jsonl_v1",
        on_record: Callable[[dict[str, Any]], None] | None = None,
        cancelled: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise AnalyzerUnavailableError(
                "configured analyzer is missing or not executable; build it with CMake"
            )
        payload = b"".join(
            json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
            for request in requests
        )
        if len(payload) > self.max_input_bytes:
            raise AnalyzerLimitError("analyzer request exceeds the configured input limit")
        try:
            process = subprocess.Popen(  # noqa: S603 - explicit operator-owned executable
                [str(self.binary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name != "nt",
            )
        except OSError as error:
            raise AnalyzerUnavailableError("configured analyzer could not be started") from error
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        stderr = bytearray()
        stopped = threading.Event()
        failures: list[BaseException] = []
        records: list[dict[str, Any]] = []

        def stop_process() -> None:
            with suppress(ProcessLookupError, PermissionError):
                if os.name != "nt":
                    # The leader may exit while descendants still hold protocol
                    # pipes open, so terminate its process group even after poll().
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()

        def fail(error: BaseException) -> None:
            failures.append(error)
            stopped.set()
            stop_process()

        def accept(record: dict[str, Any]) -> None:
            error_record = record if record.get("type") == "error" else None
            if error_record is not None:
                code = error_record.get("code", "unknown")
                if not isinstance(code, str) or not code.replace("_", "").isalnum():
                    code = "unknown"
                raise AnalyzerProtocolError(f"analyzer rejected the request ({code})")
            if on_record is None:
                records.append(record)
            else:
                on_record(record)

        decoder = _JsonlStreamDecoder(
            gzip_transport=transport == GZIP_TRANSPORT,
            max_wire_bytes=output_limit,
            max_decoded_bytes=self.max_decoded_bytes,
            max_record_bytes=self.max_record_bytes,
            on_record=accept,
        )

        def read_stdout() -> None:
            try:
                while chunk := process.stdout.read1(64 * 1024):
                    decoder.feed(chunk)
                decoder.finish()
            except BaseException as error:
                fail(error)

        def read_stderr() -> None:
            try:
                while chunk := process.stderr.read1(64 * 1024):
                    remaining = self.max_stderr_bytes + 1 - len(stderr)
                    if remaining > 0:
                        stderr.extend(chunk[:remaining])
                    if len(stderr) > self.max_stderr_bytes or len(chunk) > remaining:
                        raise AnalyzerLimitError("analyzer exceeded the stderr output limit")
            except BaseException as error:
                fail(error)

        readers = (
            threading.Thread(target=read_stdout, daemon=True),
            threading.Thread(target=read_stderr, daemon=True),
        )
        for reader in readers:
            reader.start()

        def write_input() -> None:
            # Keep a blocked or broken stdin write under the same deadline as the process.
            try:
                process.stdin.write(payload)
            except BrokenPipeError:
                pass
            finally:
                with suppress(BrokenPipeError):
                    process.stdin.close()

        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()
        timed_out = False
        was_cancelled = False
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while process.poll() is None:
                if cancelled is not None and cancelled.is_set():
                    was_cancelled = True
                    stop_process()
                    break
                if stopped.is_set():
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    stop_process()
                    break
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            process.wait(timeout=2)
            # A successful leader exit does not imply that descendants released
            # inherited protocol pipes; close the whole session before joining.
            stop_process()
        except subprocess.TimeoutExpired:
            stop_process()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
        except BaseException:
            stop_process()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
            raise
        finally:
            writer.join(timeout=2)
            for reader in readers:
                reader.join(timeout=2)
        if writer.is_alive() or any(reader.is_alive() for reader in readers):
            stop_process()
            raise AnalyzerLimitError("analyzer cleanup exceeded two seconds")
        if was_cancelled:
            raise AnalyzerLimitError("analyzer analysis was cancelled")
        if timed_out:
            raise AnalyzerLimitError("analyzer exceeded the configured timeout")
        if failures:
            limit_failure = next(
                (failure for failure in failures if isinstance(failure, AnalyzerLimitError)), None
            )
            if limit_failure is not None:
                raise limit_failure
            raise failures[0]
        if process.returncode != 0:
            raise AnalyzerProtocolError("analyzer process failed; inspect compiler diagnostics")
        return records


class NativeClangIngestor:
    """Convert complete companion facts into the existing durable domain model."""

    def __init__(
        self,
        client: NativeAnalyzerClient,
        *,
        max_workers: int = 1,
        max_spool_registries: int | None = None,
        max_spool_bytes: int | None = None,
        max_spool_fds: int | None = None,
        max_domain_batches: int = 2,
        profile: IndexProfile = IndexProfile.FULL,
    ) -> None:
        registry_limit = max_workers * 2 if max_spool_registries is None else max_spool_registries
        decoded_limit = int(getattr(client, "max_decoded_bytes", DEFAULT_MAX_DECODED_BYTES))
        spool_byte_limit = (
            registry_limit * decoded_limit if max_spool_bytes is None else max_spool_bytes
        )
        spool_fd_limit = registry_limit * MAX_FACT_KINDS if max_spool_fds is None else max_spool_fds
        if (
            min(
                max_workers,
                registry_limit,
                spool_byte_limit,
                spool_fd_limit,
                max_domain_batches,
            )
            <= 0
        ):
            raise ValueError("native analyzer pipeline limits must be positive")
        if registry_limit < max_workers:
            raise ValueError("registry bound must cover every analyzer worker")
        selected_profile = IndexProfile(profile)
        client_profile = IndexProfile(getattr(client, "profile", selected_profile))
        if client_profile is not selected_profile:
            raise ValueError("native analyzer client and ingestor profiles must match")
        self.client = client
        self.max_workers = max_workers
        self.max_spool_registries = registry_limit
        self.max_spool_bytes = spool_byte_limit
        self.max_spool_fds = spool_fd_limit
        self.max_domain_batches = min(max_domain_batches, registry_limit)
        self.profile = selected_profile
        self.advanced_facts_complete = self.profile is IndexProfile.FULL

    analysis_backend = "clang-libtooling"

    @property
    def analyzer_info(self) -> AnalyzerInfo:
        return self.client.probe()

    def ingest(
        self,
        project_root: Path,
        compilation_database: Path,
        *,
        build_variant: str = "default",
    ) -> IngestionBatch:
        database = CompilationDatabase.load(compilation_database, build_variant=build_variant)
        return self.ingest_configurations(project_root, database.configurations)

    def ingest_configurations(
        self, project_root: Path, configurations: Iterable[BuildConfiguration]
    ) -> IngestionBatch:
        batches = list(self.iter_configuration_batches(project_root, configurations))
        return self._merge_batches(batches, profile=self.profile)

    def iter_configuration_batches(
        self, project_root: Path, configurations: Iterable[BuildConfiguration]
    ) -> Iterator[IngestionBatch]:
        """Pipeline analysis and conversion while publishing TU batches in input order."""

        root = project_root.resolve(strict=False)
        self.client.probe()
        selected = tuple(configurations)
        if not selected:
            return
        cancelled = threading.Event()
        spool_bytes = _ResourceBudget(
            self.max_spool_bytes, "analyzer pipeline exceeded the spool byte limit"
        )
        spool_fds = _ResourceBudget(
            self.max_spool_fds, "analyzer pipeline exceeded the spool file limit"
        )
        record_limit = int(getattr(self.client, "max_record_bytes", DEFAULT_MAX_RECORD_BYTES))
        # Protocol decoded bytes are already bounded before facts reach this
        # adapter. The compact framing adds four bytes per fact, so only the
        # independent global spool budget may bound its ephemeral representation.
        registry_byte_limit = self.max_spool_bytes

        def analyze(configuration: BuildConfiguration) -> _FactRegistry:
            facts = _FactRegistry(
                max_bytes=registry_byte_limit,
                max_record_bytes=record_limit,
                byte_budget=spool_bytes,
                fd_budget=spool_fds,
                cancelled=cancelled,
            )
            analyze_stream = getattr(self.client, "analyze_stream", None)
            try:
                if analyze_stream is None:
                    for fact in self.client.analyze(root, configuration):
                        facts.add(fact)
                else:
                    analyze_stream(root, configuration, facts.add, cancelled=cancelled)
                return facts
            except BaseException:
                facts.close()
                raise

        def convert(index: int, facts: _FactRegistry) -> IngestionBatch:
            try:
                return _FactBatchBuilder(root, selected[index], self.profile).build(facts)
            finally:
                facts.close()

        worker_count = min(self.max_workers, max(1, len(selected)))
        converter_count = min(self.max_domain_batches, len(selected))
        analyzer_executor = ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="cpp-context-analyzer"
        )
        converter_executor = ThreadPoolExecutor(
            max_workers=converter_count, thread_name_prefix="cpp-context-converter"
        )
        condition = threading.Condition()
        analysis_futures: dict[Future[_FactRegistry], int] = {}
        pending_registries: dict[int, _FactRegistry] = {}
        conversion_futures: dict[int, Future[IngestionBatch]] = {}
        conversion_registries: dict[int, _FactRegistry] = {}
        next_configuration = 0
        next_conversion = 0
        held_registries = 0
        stopped = False
        failure: BaseException | None = None
        completion_revision = 0

        def wake(_future: Future[object]) -> None:
            nonlocal completion_revision
            with condition:
                completion_revision += 1
                condition.notify_all()

        def coordinate() -> None:
            nonlocal failure, held_registries, next_configuration, next_conversion, stopped
            try:
                with condition:
                    while True:
                        observed_revision = completion_revision
                        if stopped:
                            return

                        completed_analysis = sorted(
                            (
                                (index, future)
                                for future, index in analysis_futures.items()
                                if future.done()
                            ),
                            key=lambda item: item[0],
                        )
                        completed_failures: list[tuple[int, BaseException]] = []
                        for index, future in completed_analysis:
                            analysis_futures.pop(future)
                            if future.cancelled():
                                continue
                            error = future.exception()
                            if error is not None:
                                completed_failures.append((index, error))
                            else:
                                pending_registries[index] = future.result()

                        for index, future in sorted(conversion_futures.items()):
                            if future.done() and not future.cancelled():
                                error = future.exception()
                                if error is not None:
                                    completed_failures.append((index, error))

                        if completed_failures:
                            # Select the earliest CDB item among failures observed in
                            # the same completion interval, then preserve that cause.
                            failure = min(completed_failures, key=lambda item: item[0])[1]
                            cancelled.set()
                            for future in analysis_futures:
                                future.cancel()
                            for future in conversion_futures.values():
                                future.cancel()
                            condition.notify_all()
                            return

                        while (
                            next_conversion in pending_registries
                            and len(conversion_futures) < self.max_domain_batches
                        ):
                            index = next_conversion
                            next_conversion += 1
                            registry = pending_registries.pop(index)
                            conversion_registries[index] = registry
                            converted = converter_executor.submit(convert, index, registry)
                            conversion_futures[index] = converted
                            converted.add_done_callback(wake)

                        while (
                            next_configuration < len(selected)
                            and len(analysis_futures) < worker_count
                            and held_registries < self.max_spool_registries
                        ):
                            index = next_configuration
                            next_configuration += 1
                            analyzed = analyzer_executor.submit(analyze, selected[index])
                            analysis_futures[analyzed] = index
                            held_registries += 1
                            analyzed.add_done_callback(wake)

                        analyses_finished = (
                            next_configuration == len(selected) and not analysis_futures
                        )
                        conversions_finished = not pending_registries and all(
                            future.done() for future in conversion_futures.values()
                        )
                        if analyses_finished and conversions_finished:
                            return
                        # A tiny worker can complete while it is being registered
                        # above. Its synchronous callback cannot wake a wait that
                        # has not started yet, so re-scan instead of losing it.
                        if completion_revision == observed_revision:
                            condition.wait()
            except BaseException as error:  # pragma: no cover - defensive scheduler boundary
                with condition:
                    if failure is None:
                        failure = error
                    cancelled.set()
                    condition.notify_all()

        coordinator = threading.Thread(target=coordinate, name="cpp-context-pipeline", daemon=True)
        coordinator.start()
        try:
            for index in range(len(selected)):
                with condition:
                    while True:
                        if failure is not None:
                            raise failure
                        current = conversion_futures.get(index)
                        if current is not None and current.done():
                            batch = current.result()
                            break
                        condition.wait()
                try:
                    yield batch
                finally:
                    with condition:
                        conversion_futures.pop(index, None)
                        conversion_registries.pop(index, None)
                        held_registries -= 1
                        condition.notify_all()
                    del batch
        finally:
            with condition:
                stopped = True
                cancelled.set()
                for future in analysis_futures:
                    future.cancel()
                for future in conversion_futures.values():
                    future.cancel()
                condition.notify_all()
            coordinator.join()
            analyzer_executor.shutdown(wait=True, cancel_futures=True)
            converter_executor.shutdown(wait=True, cancel_futures=True)
            # A future can complete between the coordinator stopping and executor
            # shutdown. Close every registry idempotently so all byte/FD budgets
            # and unnamed temporary files are released on every exit path.
            for registry in pending_registries.values():
                registry.close()
            for registry in conversion_registries.values():
                registry.close()
            for future in analysis_futures:
                if future.done() and not future.cancelled() and future.exception() is None:
                    future.result().close()

    @staticmethod
    def _merge_batches(
        batches: Sequence[IngestionBatch], *, profile: IndexProfile = IndexProfile.FULL
    ) -> IngestionBatch:
        callsites = tuple(site for batch in batches for site in batch.callsites)
        call_targets = tuple(target for batch in batches for target in batch.call_targets)
        all_edges = tuple(edge for batch in batches for edge in batch.edges)
        call_targets = _add_indexed_override_candidates(callsites, call_targets, all_edges)
        inputs = IngestionBatch(
            build_configurations=tuple(
                configuration for batch in batches for configuration in batch.build_configurations
            ),
            translation_units=tuple(unit for batch in batches for unit in batch.translation_units),
            symbols=tuple(symbol for batch in batches for symbol in batch.symbols),
            occurrences=tuple(occurrence for batch in batches for occurrence in batch.occurrences),
            edges=all_edges,
            cfg_graphs=tuple(graph for batch in batches for graph in batch.cfg_graphs),
            cfg_blocks=tuple(block for batch in batches for block in batch.cfg_blocks),
            cfg_elements=tuple(element for batch in batches for element in batch.cfg_elements),
            cfg_edges=tuple(edge for batch in batches for edge in batch.cfg_edges),
            callsites=callsites,
            call_targets=call_targets,
            data_flow_analyses=tuple(
                analysis for batch in batches for analysis in batch.data_flow_analyses
            ),
            memory_locations=tuple(
                location for batch in batches for location in batch.memory_locations
            ),
            data_accesses=tuple(access for batch in batches for access in batch.data_accesses),
            data_flow_evidence=tuple(
                evidence for batch in batches for evidence in batch.data_flow_evidence
            ),
            function_summaries=tuple(
                summary for batch in batches for summary in batch.function_summaries
            ),
            summary_effects=tuple(effect for batch in batches for effect in batch.summary_effects),
            summary_return_origins=tuple(
                origin for batch in batches for origin in batch.summary_return_origins
            ),
            call_argument_bindings=tuple(
                binding for batch in batches for binding in batch.call_argument_bindings
            ),
            call_result_bindings=tuple(
                binding for batch in batches for binding in batch.call_result_bindings
            ),
        )
        if profile is IndexProfile.NAVIGATION:
            return inputs
        solution = solve_interprocedural(
            inputs.function_summaries,
            inputs.summary_effects,
            inputs.summary_return_origins,
            inputs.call_argument_bindings,
            inputs.call_result_bindings,
            inputs.callsites,
            inputs.call_targets,
        )
        return replace(
            inputs,
            function_summaries=solution.summaries,
            summary_effects=solution.effects,
            summary_return_origins=solution.return_origins,
            interprocedural_flows=solution.flows,
        )


class _FactBatchBuilder:
    def __init__(
        self,
        root: Path,
        configuration: BuildConfiguration,
        profile: IndexProfile = IndexProfile.FULL,
    ) -> None:
        self.root = root
        self.configuration = configuration
        self.profile = IndexProfile(profile)
        self.unit_id = translation_unit_id(configuration)
        self.symbols: dict[str, CodeSymbol] = {}
        self.keys: dict[str, str] = {}
        self.files: dict[str, Path] = {}
        self.cfg_graph_ids: dict[str, str] = {}
        self.cfg_graph_function_ids: dict[str, str] = {}
        self.cfg_block_ids: dict[str, str] = {}
        self.cfg_element_ids: dict[str, str] = {}
        self.cfg_block_graph_ids: dict[str, str] = {}
        self.cfg_element_graph_ids: dict[str, str] = {}
        self.callsite_ids: dict[str, str] = {}
        self.data_flow_analysis_ids: dict[str, str] = {}
        self.data_flow_analysis_graph_ids: dict[str, str] = {}
        self.memory_location_ids: dict[str, str] = {}
        self.memory_location_analysis_ids: dict[str, str] = {}
        self.data_access_ids: dict[str, str] = {}
        self.data_access_analysis_ids: dict[str, str] = {}
        self.function_summary_ids: dict[str, str] = {}
        self.function_summary_analysis_ids: dict[str, str] = {}
        self.function_summary_parameter_counts: dict[str, int] = {}
        self.path_cache: dict[str, Path] = {}

    def build(self, facts: Iterable[Mapping[str, Any]]) -> IngestionBatch:
        for fact in _fact_records(facts, "file"):
            self._file_fact(fact)
        for fact in _fact_records(facts, "symbol"):
            self._symbol_fact(fact)
        for fact in _fact_records(facts, "include"):
            if isinstance(fact.get("resolved_path"), str):
                path = self._path(fact["resolved_path"])
                self._file_symbol(path)
        occurrences: dict[str, SymbolOccurrence] = {}
        edges: dict[str, GraphEdge] = {}
        for fact in _fact_records(facts, "occurrence"):
            occurrence = self._occurrence_fact(fact)
            occurrences[occurrence.id] = occurrence
        for fact_kind in ("edge", "include"):
            for fact in _fact_records(facts, fact_kind):
                edge = self._edge_fact(fact)
                if edge is not None:
                    edges[edge.id] = edge
        callsites, call_targets = self._call_facts(facts)
        if self.profile is IndexProfile.FULL:
            cfg_graphs, cfg_blocks, cfg_elements, cfg_edges = self._cfg_facts(facts)
            analyses, locations, accesses, evidence = self._data_flow_facts(facts)
            summaries, effects, origins, argument_bindings, result_bindings = (
                self._interprocedural_facts(facts)
            )
        else:
            cfg_graphs = cfg_blocks = cfg_elements = cfg_edges = ()
            analyses = locations = accesses = evidence = ()
            summaries = effects = origins = argument_bindings = result_bindings = ()
        dependencies = tuple(
            (path, _hash_bytes(path.read_bytes())) for path in sorted(set(self.files.values()))
        )
        unit = TranslationUnit(
            id=self.unit_id,
            build_configuration_id=self.configuration.id,
            source_path=self.configuration.source_path,
            content_hash=_hash_bytes(self.configuration.source_path.read_bytes()),
            dependencies=dependencies,
            build_variant=self.configuration.build_variant,
            analysis_backend=NativeClangIngestor.analysis_backend,
            advanced_facts_complete=self.profile is IndexProfile.FULL,
            index_profile=self.profile,
            navigation_facts_complete=True,
            cfg_facts_complete=self.profile is IndexProfile.FULL,
            data_flow_facts_complete=self.profile is IndexProfile.FULL,
            summary_facts_complete=self.profile is IndexProfile.FULL,
        )
        return IngestionBatch(
            build_configurations=(self.configuration,),
            translation_units=(unit,),
            symbols=tuple(sorted(self.symbols.values(), key=lambda item: item.id)),
            occurrences=tuple(sorted(occurrences.values(), key=lambda item: item.id)),
            edges=tuple(sorted(edges.values(), key=lambda item: item.id)),
            cfg_graphs=cfg_graphs,
            cfg_blocks=cfg_blocks,
            cfg_elements=cfg_elements,
            cfg_edges=cfg_edges,
            callsites=callsites,
            call_targets=call_targets,
            data_flow_analyses=analyses,
            memory_locations=locations,
            data_accesses=accesses,
            data_flow_evidence=evidence,
            function_summaries=summaries,
            summary_effects=effects,
            summary_return_origins=origins,
            call_argument_bindings=argument_bindings,
            call_result_bindings=result_bindings,
        )

    def _data_flow_facts(
        self, facts: Iterable[Mapping[str, Any]]
    ) -> tuple[
        tuple[DataFlowAnalysis, ...],
        tuple[MemoryLocation, ...],
        tuple[DataAccess, ...],
        tuple[DataFlowEvidence, ...],
    ]:
        analysis_facts = list(_fact_records(facts, "data_flow_analysis_v1"))
        location_facts = list(_fact_records(facts, "memory_location_v1"))
        access_facts = list(_fact_records(facts, "data_access_v1"))
        for fact in analysis_facts:
            key = _string(fact, "key")
            graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
            self.data_flow_analysis_ids[key] = "data_flow_" + _hash_text(graph_id)[:32]
            self.data_flow_analysis_graph_ids[key] = graph_id
        for fact in location_facts:
            key = _string(fact, "key")
            analysis_id = self._known_data_flow_analysis(_string(fact, "analysis_key"))
            self.memory_location_ids[key] = "memory_" + _hash_text(analysis_id, key)[:32]
            self.memory_location_analysis_ids[key] = analysis_id
        for fact in access_facts:
            key = _string(fact, "key")
            analysis_id = self._known_data_flow_analysis(_string(fact, "analysis_key"))
            self.data_access_ids[key] = "access_" + _hash_text(analysis_id, key)[:32]
            self.data_access_analysis_ids[key] = analysis_id

        for fact in (*location_facts, *access_facts):
            analysis_key = _string(fact, "analysis_key")
            graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
            if self.data_flow_analysis_graph_ids.get(analysis_key) != graph_id:
                raise AnalyzerProtocolError(
                    "analyzer data-flow facts have inconsistent graph references"
                )
        # Individual SQLite FKs cannot prove that referenced rows belong to this analysis.
        for fact in location_facts:
            analysis_id = self._known_data_flow_analysis(_string(fact, "analysis_key"))
            base_key = fact.get("base_key")
            if base_key and self.memory_location_analysis_ids.get(str(base_key)) != analysis_id:
                raise AnalyzerProtocolError(
                    "analyzer data-flow facts have inconsistent analysis references"
                )
        for fact in access_facts:
            analysis_id = self._known_data_flow_analysis(_string(fact, "analysis_key"))
            if self.memory_location_analysis_ids.get(_string(fact, "location_key")) != analysis_id:
                raise AnalyzerProtocolError(
                    "analyzer data-flow facts have inconsistent analysis references"
                )
            graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
            if self.cfg_block_graph_ids.get(_string(fact, "block_key")) != graph_id:
                raise AnalyzerProtocolError(
                    "analyzer data-flow facts have inconsistent graph references"
                )
            element_key = fact.get("cfg_element_key")
            if element_key and self.cfg_element_graph_ids.get(str(element_key)) != graph_id:
                raise AnalyzerProtocolError(
                    "analyzer data-flow facts have inconsistent graph references"
                )

        try:
            analyses = tuple(
                sorted(
                    (self._data_flow_analysis_fact(fact) for fact in analysis_facts),
                    key=lambda item: item.id,
                )
            )
            locations = tuple(
                sorted(
                    (self._memory_location_fact(fact) for fact in location_facts),
                    key=lambda item: (item.analysis_id, item.kind.value, item.name, item.id),
                )
            )
            accesses = tuple(
                sorted(
                    (self._data_access_fact(fact) for fact in access_facts),
                    key=lambda item: (item.analysis_id, item.block_id, item.sequence, item.id),
                )
            )
            evidence = tuple(
                sorted(
                    (
                        self._data_flow_evidence_fact(fact)
                        for fact in _fact_records(facts, "data_flow_evidence_v1")
                    ),
                    key=lambda item: (item.analysis_id, item.relation.value, item.id),
                )
            )
        except ValueError as error:
            # Invalid enums and model invariants are malformed protocol-v5 facts.
            raise AnalyzerProtocolError("analyzer returned an invalid data-flow fact") from error
        return analyses, locations, accesses, evidence

    def _data_flow_analysis_fact(self, fact: Mapping[str, Any]) -> DataFlowAnalysis:
        raw_reasons = fact.get("incomplete_reasons")
        if not isinstance(raw_reasons, list) or not all(
            isinstance(reason, str) for reason in raw_reasons
        ):
            raise AnalyzerProtocolError("analyzer data-flow reasons must be strings")
        limits = _mapping(fact, "limits")
        return DataFlowAnalysis(
            id=self._known_data_flow_analysis(_string(fact, "key")),
            graph_id=self._known_cfg_graph(_string(fact, "graph_key")),
            complete=_boolean(fact, "complete"),
            incomplete_reasons=tuple(raw_reasons),
            iteration_count=_non_negative_integer(fact, "iteration_count"),
            max_iterations=_positive_mapping_integer(limits, "max_iterations"),
            max_alias_targets=_positive_mapping_integer(limits, "max_alias_targets"),
            max_access_path_depth=_positive_mapping_integer(limits, "max_access_path_depth"),
            max_locations=_positive_mapping_integer(limits, "max_locations"),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _memory_location_fact(self, fact: Mapping[str, Any]) -> MemoryLocation:
        raw_path = fact.get("access_path", [])
        if not isinstance(raw_path, list) or not all(
            isinstance(component, str) for component in raw_path
        ):
            raise AnalyzerProtocolError("analyzer memory access path is invalid")
        declaration_key = fact.get("declaration_key")
        base_key = fact.get("base_key")
        if declaration_key is not None and not isinstance(declaration_key, str):
            raise AnalyzerProtocolError("analyzer memory declaration key is invalid")
        if base_key is not None and not isinstance(base_key, str):
            raise AnalyzerProtocolError("analyzer memory base key is invalid")
        analysis_id = self._known_data_flow_analysis(_string(fact, "analysis_key"))
        graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
        return MemoryLocation(
            id=self._known_memory_location(_string(fact, "key")),
            analysis_id=analysis_id,
            graph_id=graph_id,
            kind=MemoryLocationKind(_string(fact, "kind")),
            name=_string(fact, "name"),
            type_name=_optional_string(fact, "type_name"),
            declaration_symbol_id=(self._known_id(declaration_key) if declaration_key else None),
            base_location_id=(self._known_memory_location(base_key) if base_key else None),
            access_path=tuple(raw_path),
            is_volatile=_boolean(fact, "is_volatile"),
            is_atomic=_boolean(fact, "is_atomic"),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _data_access_fact(self, fact: Mapping[str, Any]) -> DataAccess:
        raw_pointees = fact.get("pointee_keys", [])
        if not isinstance(raw_pointees, list) or not all(
            isinstance(key, str) for key in raw_pointees
        ):
            raise AnalyzerProtocolError("analyzer points-to targets are invalid")
        element_key = fact.get("cfg_element_key")
        if element_key is not None and not isinstance(element_key, str):
            raise AnalyzerProtocolError("analyzer data access CFG element is invalid")
        return DataAccess(
            id=self._known_data_access(_string(fact, "key")),
            analysis_id=self._known_data_flow_analysis(_string(fact, "analysis_key")),
            graph_id=self._known_cfg_graph(_string(fact, "graph_key")),
            block_id=self._known_cfg_block(_string(fact, "block_key")),
            cfg_element_id=(self._known_cfg_element(element_key) if element_key else None),
            location_id=self._known_memory_location(_string(fact, "location_key")),
            kind=DataAccessKind(_string(fact, "kind")),
            sequence=_non_negative_integer(fact, "sequence"),
            span=self._optional_span(fact, "span"),
            expression=_optional_string(fact, "expression"),
            pointee_symbol_ids=tuple(self._known_id(key) for key in raw_pointees),
            points_to_complete=_boolean(fact, "points_to_complete"),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _data_flow_evidence_fact(self, fact: Mapping[str, Any]) -> DataFlowEvidence:
        source_access_key = fact.get("source_access_key")
        target_access_key = fact.get("target_access_key")
        source_location_key = fact.get("source_location_key")
        target_location_key = fact.get("target_location_key")
        for value in (
            source_access_key,
            target_access_key,
            source_location_key,
            target_location_key,
        ):
            if value is not None and not isinstance(value, str):
                raise AnalyzerProtocolError("analyzer data-flow evidence key is invalid")
        key = _string(fact, "key")
        analysis_key = _string(fact, "analysis_key")
        analysis_id = self._known_data_flow_analysis(analysis_key)
        graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
        if self.data_flow_analysis_graph_ids.get(analysis_key) != graph_id:
            raise AnalyzerProtocolError(
                "analyzer data-flow facts have inconsistent graph references"
            )
        relation = DataFlowRelation(_string(fact, "relation"))
        access_keys = (source_access_key, target_access_key)
        location_keys = (source_location_key, target_location_key)
        if relation in {
            DataFlowRelation.REACHING_DEFINITION,
            DataFlowRelation.OVERWRITES,
        }:
            valid_pair = all(access_keys) and not any(location_keys)
        else:
            valid_pair = all(location_keys) and not any(access_keys)
        if not valid_pair:
            raise AnalyzerProtocolError("analyzer data-flow evidence relation is invalid")
        if any(
            self.data_access_analysis_ids.get(str(value)) != analysis_id
            for value in access_keys
            if value
        ) or any(
            self.memory_location_analysis_ids.get(str(value)) != analysis_id
            for value in location_keys
            if value
        ):
            raise AnalyzerProtocolError(
                "analyzer data-flow facts have inconsistent analysis references"
            )
        return DataFlowEvidence(
            id="evidence_" + _hash_text(analysis_id, key)[:32],
            analysis_id=analysis_id,
            graph_id=graph_id,
            relation=relation,
            certainty=DataFlowCertainty(_string(fact, "certainty")),
            reason=_string(fact, "reason"),
            source_access_id=(
                self._known_data_access(source_access_key) if source_access_key else None
            ),
            target_access_id=(
                self._known_data_access(target_access_key) if target_access_key else None
            ),
            source_location_id=(
                self._known_memory_location(source_location_key) if source_location_key else None
            ),
            target_location_id=(
                self._known_memory_location(target_location_key) if target_location_key else None
            ),
            evidence_span=self._optional_span(fact, "evidence_span"),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _known_data_flow_analysis(self, key: str) -> str:
        try:
            return self.data_flow_analysis_ids[key]
        except KeyError as error:
            raise AnalyzerProtocolError(
                "analyzer data-flow fact references an unknown analysis"
            ) from error

    def _known_memory_location(self, key: str) -> str:
        try:
            return self.memory_location_ids[key]
        except KeyError as error:
            raise AnalyzerProtocolError(
                "analyzer data-flow fact references an unknown memory location"
            ) from error

    def _known_data_access(self, key: str) -> str:
        try:
            return self.data_access_ids[key]
        except KeyError as error:
            raise AnalyzerProtocolError(
                "analyzer data-flow evidence references an unknown access"
            ) from error

    def _interprocedural_facts(
        self, facts: Iterable[Mapping[str, Any]]
    ) -> tuple[
        tuple[FunctionSummary, ...],
        tuple[SummaryEffect, ...],
        tuple[SummaryReturnOrigin, ...],
        tuple[CallArgumentBinding, ...],
        tuple[CallResultBinding, ...],
    ]:
        summary_facts = list(_fact_records(facts, "function_summary_v1"))
        for fact in summary_facts:
            key = _string(fact, "key")
            analysis_id = self._known_data_flow_analysis(_string(fact, "analysis_key"))
            graph_key = _string(fact, "graph_key")
            graph_id = self._known_cfg_graph(graph_key)
            if self.data_flow_analysis_graph_ids.get(_string(fact, "analysis_key")) != graph_id:
                raise AnalyzerProtocolError(
                    "analyzer summary facts have inconsistent graph references"
                )
            function_id = self._known_id(_string(fact, "function_key"))
            if self.cfg_graph_function_ids.get(graph_key) != function_id:
                raise AnalyzerProtocolError(
                    "analyzer summary facts have inconsistent graph references"
                )
            self.function_summary_ids[key] = (
                "summary_"
                + _hash_text(
                    self.configuration.build_variant,
                    self.configuration.id,
                    self.unit_id,
                    graph_id,
                )[:32]
            )
            self.function_summary_analysis_ids[key] = analysis_id
            self.function_summary_parameter_counts[key] = len(_string_list(fact, "parameter_modes"))
        try:
            summaries = tuple(
                sorted(
                    (self._function_summary_fact(fact) for fact in summary_facts),
                    key=lambda item: item.id,
                )
            )
            effects = tuple(
                sorted(
                    (
                        self._summary_effect_fact(fact)
                        for fact in _fact_records(facts, "summary_effect_v1")
                    ),
                    key=lambda item: item.id,
                )
            )
            origins = tuple(
                sorted(
                    (
                        self._summary_return_origin_fact(fact)
                        for fact in _fact_records(facts, "summary_return_origin_v1")
                    ),
                    key=lambda item: item.id,
                )
            )
            arguments = tuple(
                sorted(
                    (
                        self._call_argument_binding_fact(fact)
                        for fact in _fact_records(facts, "call_argument_binding_v1")
                    ),
                    key=lambda item: item.id,
                )
            )
            results = tuple(
                sorted(
                    (
                        self._call_result_binding_fact(fact)
                        for fact in _fact_records(facts, "call_result_binding_v1")
                    ),
                    key=lambda item: item.id,
                )
            )
        except ValueError as error:
            raise AnalyzerProtocolError("analyzer returned an invalid summary fact") from error
        return summaries, effects, origins, arguments, results

    def _function_summary_fact(self, fact: Mapping[str, Any]) -> FunctionSummary:
        key = _string(fact, "key")
        modes = _string_list(fact, "parameter_modes")
        location_keys = _string_list(fact, "parameter_location_keys")
        reasons = _string_list(fact, "local_incomplete_reasons")
        complete = _boolean(fact, "local_complete")
        if len(modes) != len(location_keys) or complete == bool(reasons):
            raise AnalyzerProtocolError("analyzer function summary shape is invalid")
        analysis_id = self._known_data_flow_analysis(_string(fact, "analysis_key"))
        if any(
            self.memory_location_analysis_ids.get(location_key) != analysis_id
            for location_key in location_keys
        ):
            raise AnalyzerProtocolError(
                "analyzer summary facts have inconsistent analysis references"
            )
        limits = InterproceduralLimits()
        return FunctionSummary(
            id=self._known_function_summary(key),
            function_symbol_id=self._known_id(_string(fact, "function_key")),
            graph_id=self._known_cfg_graph(_string(fact, "graph_key")),
            analysis_id=analysis_id,
            parameter_modes=modes,
            parameter_location_ids=tuple(
                self._known_memory_location(location_key) for location_key in location_keys
            ),
            local_complete=complete,
            local_incomplete_reasons=reasons,
            complete=complete,
            incomplete_reasons=reasons,
            recursive=False,
            iteration_count=0,
            max_scc_iterations=limits.max_scc_iterations,
            max_scc_size=limits.max_scc_size,
            max_summary_effects=limits.max_summary_effects,
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _summary_effect_fact(self, fact: Mapping[str, Any]) -> SummaryEffect:
        summary_key = _string(fact, "summary_key")
        summary_id = self._known_function_summary(summary_key)
        location_key = _optional_key(fact, "location_key")
        access_key = _optional_key(fact, "source_access_key")
        self._validate_summary_analysis(summary_key, location_key, access_key)
        key = _string(fact, "key")
        parameter_index = _optional_non_negative_integer(fact, "parameter_index")
        if (
            parameter_index is not None
            and parameter_index >= self.function_summary_parameter_counts[summary_key]
        ):
            raise AnalyzerProtocolError("analyzer summary effect has an invalid parameter index")
        return SummaryEffect(
            id="summary_effect_" + _hash_text(summary_id, key)[:32],
            summary_id=summary_id,
            kind=SummaryEffectKind(_string(fact, "kind")),
            location_kind=MemoryLocationKind(_string(fact, "location_kind")),
            certainty=DataFlowCertainty(_string(fact, "certainty")),
            reason=_string(fact, "reason"),
            parameter_index=parameter_index,
            access_path=_string_list(fact, "access_path"),
            location_id=self._known_memory_location(location_key) if location_key else None,
            source_access_id=self._known_data_access(access_key) if access_key else None,
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _summary_return_origin_fact(self, fact: Mapping[str, Any]) -> SummaryReturnOrigin:
        summary_key = _string(fact, "summary_key")
        summary_id = self._known_function_summary(summary_key)
        location_key = _optional_key(fact, "location_key")
        self._validate_summary_analysis(summary_key, location_key, None)
        callsite_key = _optional_key(fact, "callsite_key")
        location_kind = _optional_string(fact, "location_kind")
        key = _string(fact, "key")
        parameter_index = _optional_non_negative_integer(fact, "parameter_index")
        if (
            parameter_index is not None
            and parameter_index >= self.function_summary_parameter_counts[summary_key]
        ):
            raise AnalyzerProtocolError("analyzer summary origin has an invalid parameter index")
        return SummaryReturnOrigin(
            id="summary_return_" + _hash_text(summary_id, key)[:32],
            summary_id=summary_id,
            kind=SummaryReturnOriginKind(_string(fact, "kind")),
            certainty=DataFlowCertainty(_string(fact, "certainty")),
            reason=_string(fact, "reason"),
            location_kind=MemoryLocationKind(location_kind) if location_kind else None,
            parameter_index=parameter_index,
            access_path=_string_list(fact, "access_path"),
            location_id=self._known_memory_location(location_key) if location_key else None,
            callsite_id=self._known_callsite(callsite_key) if callsite_key else None,
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _call_argument_binding_fact(self, fact: Mapping[str, Any]) -> CallArgumentBinding:
        summary_key = _string(fact, "summary_key")
        summary_id = self._known_function_summary(summary_key)
        location_key = _optional_key(fact, "location_key")
        self._validate_summary_analysis(summary_key, location_key, None)
        complete = _boolean(fact, "complete")
        reason = _optional_string(fact, "incomplete_reason")
        if complete == bool(reason):
            raise AnalyzerProtocolError("analyzer call argument completeness is invalid")
        callsite_id = self._known_callsite(_string(fact, "callsite_key"))
        index = _non_negative_integer(fact, "argument_index")
        parameter_index = _optional_non_negative_integer(fact, "parameter_index")
        if (
            parameter_index is not None
            and parameter_index >= self.function_summary_parameter_counts[summary_key]
        ):
            raise AnalyzerProtocolError("analyzer call binding has an invalid parameter index")
        return CallArgumentBinding(
            id="call_argument_" + _hash_text(summary_id, callsite_id, str(index))[:32],
            caller_summary_id=summary_id,
            callsite_id=callsite_id,
            argument_index=index,
            location_id=self._known_memory_location(location_key) if location_key else None,
            location_kind=MemoryLocationKind(_string(fact, "location_kind")),
            parameter_index=parameter_index,
            access_path=_string_list(fact, "access_path"),
            writeback_candidate=_boolean(fact, "writeback_candidate"),
            complete=complete,
            incomplete_reason=reason,
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _call_result_binding_fact(self, fact: Mapping[str, Any]) -> CallResultBinding:
        summary_key = _string(fact, "summary_key")
        summary_id = self._known_function_summary(summary_key)
        location_key = _string(fact, "location_key")
        access_key = _string(fact, "definition_access_key")
        self._validate_summary_analysis(summary_key, location_key, access_key)
        callsite_id = self._known_callsite(_string(fact, "callsite_key"))
        return CallResultBinding(
            id="call_result_" + _hash_text(summary_id, callsite_id)[:32],
            caller_summary_id=summary_id,
            callsite_id=callsite_id,
            location_id=self._known_memory_location(location_key),
            definition_access_id=self._known_data_access(access_key),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _validate_summary_analysis(
        self, summary_key: str, location_key: str | None, access_key: str | None
    ) -> None:
        analysis_id = self.function_summary_analysis_ids.get(summary_key)
        if analysis_id is None:
            raise AnalyzerProtocolError("analyzer fact references an unknown function summary")
        if location_key and self.memory_location_analysis_ids.get(location_key) != analysis_id:
            raise AnalyzerProtocolError(
                "analyzer summary facts have inconsistent analysis references"
            )
        if access_key and self.data_access_analysis_ids.get(access_key) != analysis_id:
            raise AnalyzerProtocolError(
                "analyzer summary facts have inconsistent analysis references"
            )

    def _known_function_summary(self, key: str) -> str:
        try:
            return self.function_summary_ids[key]
        except KeyError as error:
            raise AnalyzerProtocolError(
                "analyzer fact references an unknown function summary"
            ) from error

    def _call_facts(
        self, facts: Iterable[Mapping[str, Any]]
    ) -> tuple[tuple[CallSite, ...], tuple[CallTarget, ...]]:
        site_facts = list(_fact_records(facts, "callsite_v1"))
        for fact in site_facts:
            key = _string(fact, "key")
            owner_id = self._known_id(_string(fact, "owner_key"))
            spelling = self._optional_span(fact, "spelling_span")
            expansion = self._span(_mapping(fact, "expansion_span"))
            spelling_identity = spelling or expansion
            self.callsite_ids[key] = (
                "callsite_"
                + _hash_text(
                    self.configuration.build_variant,
                    self.configuration.id,
                    self.unit_id,
                    owner_id,
                    str(spelling_identity.path),
                    str(spelling_identity.start_line),
                    str(spelling_identity.start_column),
                    str(spelling_identity.end_line),
                    str(spelling_identity.end_column),
                    str(expansion.path),
                    str(expansion.start_line),
                    str(expansion.start_column),
                    str(expansion.end_line),
                    str(expansion.end_column),
                    key,
                )[:32]
            )
        sites_by_key = {_string(fact, "key"): self._callsite_fact(fact) for fact in site_facts}
        for fact in _fact_records(facts, "callsite_resolution_v1"):
            key = _string(fact, "callsite_key")
            try:
                site = sites_by_key[key]
            except KeyError as error:
                raise AnalyzerProtocolError(
                    "analyzer resolution references an unknown callsite"
                ) from error
            complete = _boolean(fact, "target_set_complete")
            reason = _optional_string(fact, "unresolved_reason")
            if complete == bool(reason):
                raise AnalyzerProtocolError("analyzer callsite resolution completeness is invalid")
            sites_by_key[key] = replace(
                site,
                target_set_complete=complete,
                unresolved_reason=reason,
            )
        sites = tuple(sorted(sites_by_key.values(), key=lambda x: x.id))
        targets = tuple(
            sorted(
                (self._call_target_fact(fact) for fact in _fact_records(facts, "call_target_v1")),
                key=lambda item: (item.callsite_id, item.target_symbol_id, item.id),
            )
        )
        return sites, targets

    def _callsite_fact(self, fact: Mapping[str, Any]) -> CallSite:
        key = _string(fact, "key")
        static_key = fact.get("static_target_key")
        if static_key is not None and not isinstance(static_key, str):
            raise AnalyzerProtocolError("analyzer callsite has invalid static target")
        raw_stack = fact.get("expansion_stack", [])
        if not isinstance(raw_stack, list):
            raise AnalyzerProtocolError("analyzer callsite has invalid macro expansion stack")
        stack: list[MacroExpansionFrame] = []
        for raw_frame in raw_stack:
            if not isinstance(raw_frame, dict):
                raise AnalyzerProtocolError("analyzer callsite has invalid macro expansion frame")
            stack.append(
                MacroExpansionFrame(
                    macro_symbol_id=self._known_id(_string(raw_frame, "macro_key")),
                    name=_string(raw_frame, "name"),
                    spelling_span=self._span(_mapping(raw_frame, "spelling_span")),
                    expansion_span=self._span(_mapping(raw_frame, "expansion_span")),
                )
            )
        return CallSite(
            id=self._known_callsite(key),
            owner_symbol_id=self._known_id(_string(fact, "owner_key")),
            dispatch_kind=CallDispatchKind(_string(fact, "dispatch_kind")),
            spelling_span=self._optional_span(fact, "spelling_span"),
            expansion_span=self._span(_mapping(fact, "expansion_span")),
            target_set_complete=_boolean(fact, "target_set_complete"),
            static_target_symbol_id=(self._known_id(static_key) if static_key else None),
            unresolved_reason=_optional_string(fact, "unresolved_reason"),
            callee_text=_optional_string(fact, "callee_text"),
            expansion_stack=tuple(stack),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _call_target_fact(self, fact: Mapping[str, Any]) -> CallTarget:
        callsite_id = self._known_callsite(_string(fact, "callsite_key"))
        target_id = self._known_id(_string(fact, "target_key"))
        certainty = CallTargetCertainty(_string(fact, "certainty"))
        confidence = _number(fact, "confidence")
        evidence = self._span(_mapping(fact, "evidence_span"))
        derivation = _string(fact, "derivation")
        return CallTarget(
            id="call_target_"
            + _hash_text(
                self.configuration.build_variant,
                self.configuration.id,
                self.unit_id,
                callsite_id,
                target_id,
                certainty.value,
                derivation,
            )[:32],
            callsite_id=callsite_id,
            target_symbol_id=target_id,
            certainty=certainty,
            confidence=confidence,
            confidence_reason=_string(fact, "confidence_reason"),
            derivation=derivation,
            evidence_span=evidence,
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _known_callsite(self, key: str) -> str:
        try:
            return self.callsite_ids[key]
        except KeyError as error:
            raise AnalyzerProtocolError("analyzer target references an unknown callsite") from error

    def _cfg_facts(
        self, facts: Iterable[Mapping[str, Any]]
    ) -> tuple[
        tuple[CfgGraph, ...],
        tuple[CfgBlock, ...],
        tuple[CfgElement, ...],
        tuple[CfgEdge, ...],
    ]:
        graph_facts = list(_fact_records(facts, "cfg_graph_v1"))
        block_facts = list(_fact_records(facts, "cfg_block_v1"))
        for fact in graph_facts:
            graph_key = _string(fact, "key")
            function_id = self._known_id(_string(fact, "function_key"))
            self.cfg_graph_function_ids[graph_key] = function_id
            self.cfg_graph_ids[graph_key] = (
                "cfg_"
                + _hash_text(
                    self.configuration.build_variant,
                    self.configuration.id,
                    self.unit_id,
                    function_id,
                )[:32]
            )
        block_graph_ids: dict[str, str] = {}
        for fact in block_facts:
            graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
            index = _non_negative_integer(fact, "index")
            block_key = _string(fact, "key")
            block_graph_ids[block_key] = graph_id
            self.cfg_block_graph_ids[block_key] = graph_id
            self.cfg_block_ids[block_key] = "cfg_block_" + _hash_text(graph_id, str(index))[:32]
        for fact in _fact_records(facts, "cfg_element_v1"):
            graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
            block_id = self._known_cfg_block(_string(fact, "block_key"))
            index = _non_negative_integer(fact, "index")
            self.cfg_element_ids[_string(fact, "key")] = (
                "cfg_element_" + _hash_text(graph_id, block_id, str(index))[:32]
            )
            self.cfg_element_graph_ids[_string(fact, "key")] = graph_id

        # A compromised or mismatched companion must not be able to persist a CFG
        # relation that crosses graph boundaries while still satisfying SQLite FKs.
        for fact in graph_facts:
            graph_id = self._known_cfg_graph(_string(fact, "key"))
            endpoint_keys = [
                _string(fact, "entry_block_key"),
                _string(fact, "normal_exit_block_key"),
            ]
            exceptional_key = fact.get("exceptional_exit_block_key")
            if exceptional_key is not None:
                if not isinstance(exceptional_key, str) or not exceptional_key:
                    raise AnalyzerProtocolError("analyzer CFG exceptional exit key is invalid")
                endpoint_keys.append(exceptional_key)
            if any(block_graph_ids.get(key) != graph_id for key in endpoint_keys):
                raise AnalyzerProtocolError("analyzer CFG facts have inconsistent graph references")
        for fact_kind in ("cfg_element_v1", "cfg_edge_v1"):
            for fact in _fact_records(facts, fact_kind):
                if fact_kind == "cfg_element_v1":
                    graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
                    if block_graph_ids.get(_string(fact, "block_key")) != graph_id:
                        raise AnalyzerProtocolError(
                            "analyzer CFG facts have inconsistent graph references"
                        )
                    continue
                graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
                if any(
                    block_graph_ids.get(_string(fact, key)) != graph_id
                    for key in ("source_block_key", "target_block_key")
                ):
                    raise AnalyzerProtocolError(
                        "analyzer CFG facts have inconsistent graph references"
                    )

        graphs = tuple(
            sorted((self._cfg_graph_fact(fact) for fact in graph_facts), key=lambda item: item.id)
        )
        blocks = tuple(
            sorted(
                (self._cfg_block_fact(fact) for fact in block_facts),
                key=lambda item: (item.graph_id, item.index, item.id),
            )
        )
        elements = tuple(
            sorted(
                (self._cfg_element_fact(fact) for fact in _fact_records(facts, "cfg_element_v1")),
                key=lambda item: (item.graph_id, item.block_id, item.index, item.id),
            )
        )
        edges = tuple(
            sorted(
                (self._cfg_edge_fact(fact) for fact in _fact_records(facts, "cfg_edge_v1")),
                key=lambda item: (
                    item.graph_id,
                    item.source_block_id,
                    item.successor_index,
                    item.target_block_id,
                    item.kind.value,
                    item.id,
                ),
            )
        )
        return graphs, blocks, elements, edges

    def _cfg_graph_fact(self, fact: Mapping[str, Any]) -> CfgGraph:
        graph_id = self._known_cfg_graph(_string(fact, "key"))
        schema_version = _integer(fact, "fact_schema_version")
        clang_major = _integer(fact, "clang_major")
        if schema_version != 1 or clang_major != REQUIRED_CLANG_MAJOR:
            raise AnalyzerProtocolError("analyzer returned an unsupported CFG fact schema")
        options = _mapping(fact, "build_options")
        exceptional_key = fact.get("exceptional_exit_block_key")
        if exceptional_key is not None and not isinstance(exceptional_key, str):
            raise AnalyzerProtocolError("analyzer CFG exceptional exit key is invalid")
        return CfgGraph(
            id=graph_id,
            function_symbol_id=self._known_id(_string(fact, "function_key")),
            entry_block_id=self._known_cfg_block(_string(fact, "entry_block_key")),
            normal_exit_block_id=self._known_cfg_block(_string(fact, "normal_exit_block_key")),
            exceptional_exit_block_id=(
                self._known_cfg_block(exceptional_key) if exceptional_key else None
            ),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
            clang_major=clang_major,
            fact_schema_version=schema_version,
            build_options=dict(options),
        )

    def _cfg_block_fact(self, fact: Mapping[str, Any]) -> CfgBlock:
        return CfgBlock(
            id=self._known_cfg_block(_string(fact, "key")),
            graph_id=self._known_cfg_graph(_string(fact, "graph_key")),
            index=_non_negative_integer(fact, "index"),
            role=CfgBlockRole(_string(fact, "role")),
            reachable=_boolean(fact, "reachable"),
            terminator_kind=_optional_string(fact, "terminator_kind"),
            terminator_text=_optional_string(fact, "terminator_text"),
            terminator_spelling_span=self._optional_span(fact, "terminator_spelling_span"),
            terminator_expansion_span=self._optional_span(fact, "terminator_expansion_span"),
            label_kind=_optional_string(fact, "label_kind"),
            label_text=_optional_string(fact, "label_text"),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _cfg_element_fact(self, fact: Mapping[str, Any]) -> CfgElement:
        graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
        block_id = self._known_cfg_block(_string(fact, "block_key"))
        index = _non_negative_integer(fact, "index")
        metadata = _mapping(fact, "metadata")
        return CfgElement(
            id=self._known_cfg_element(_string(fact, "key")),
            graph_id=graph_id,
            block_id=block_id,
            index=index,
            kind=_string(fact, "kind"),
            statement_class=_optional_string(fact, "statement_class"),
            text=_optional_string(fact, "text"),
            spelling_span=self._optional_span(fact, "spelling_span"),
            expansion_span=self._optional_span(fact, "expansion_span"),
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
            metadata=dict(metadata),
        )

    def _cfg_edge_fact(self, fact: Mapping[str, Any]) -> CfgEdge:
        graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
        source_id = self._known_cfg_block(_string(fact, "source_block_key"))
        target_id = self._known_cfg_block(_string(fact, "target_block_key"))
        successor_index = _non_negative_integer(fact, "successor_index")
        kind = CfgEdgeKind(_string(fact, "kind"))
        feasible = _boolean(fact, "feasible")
        return CfgEdge(
            id="cfg_edge_"
            + _hash_text(
                graph_id,
                source_id,
                target_id,
                str(successor_index),
                kind.value,
                str(feasible),
            )[:32],
            graph_id=graph_id,
            source_block_id=source_id,
            target_block_id=target_id,
            kind=kind,
            successor_index=successor_index,
            feasible=feasible,
            translation_unit_id=self.unit_id,
            build_configuration_id=self.configuration.id,
            build_variant=self.configuration.build_variant,
        )

    def _optional_span(self, fact: Mapping[str, Any], name: str) -> SourceSpan | None:
        value = fact.get(name)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
        return self._span(value)

    def _known_cfg_graph(self, key: str) -> str:
        try:
            return self.cfg_graph_ids[key]
        except KeyError as error:
            raise AnalyzerProtocolError("analyzer CFG fact references an unknown graph") from error

    def _known_cfg_block(self, key: str) -> str:
        try:
            return self.cfg_block_ids[key]
        except KeyError as error:
            raise AnalyzerProtocolError("analyzer CFG fact references an unknown block") from error

    def _known_cfg_element(self, key: str) -> str:
        try:
            return self.cfg_element_ids[key]
        except KeyError as error:
            raise AnalyzerProtocolError(
                "analyzer data-flow fact references an unknown CFG element"
            ) from error

    def _file_fact(self, fact: Mapping[str, Any]) -> None:
        key = _string(fact, "key")
        path = self._path(_string(fact, "path"))
        self.files[key] = path
        self._file_symbol(path, key=key)

    def _file_symbol(self, path: Path, *, key: str | None = None) -> CodeSymbol:
        relative = path.relative_to(self.root).as_posix()
        symbol_id = "file_" + _hash_text(relative)[:32]
        key = key or "file:" + relative
        self.keys[key] = symbol_id
        self.files[key] = path
        existing = self.symbols.get(symbol_id)
        if existing is not None:
            return existing
        content = path.read_bytes()
        lines = content.decode("utf-8", errors="replace").splitlines()
        symbol = CodeSymbol(
            id=symbol_id,
            qualified_name=relative,
            kind=SymbolKind.FILE,
            span=SourceSpan(
                path,
                1,
                max(1, len(lines)),
                1,
                len(lines[-1]) + 1 if lines else 1,
            ),
            signature=relative,
            source_hash=_hash_bytes(content),
            build_configuration_id=self.configuration.id,
            translation_unit_id=self.unit_id,
            build_variant=self.configuration.build_variant,
            variant_id=self._variant_id(symbol_id),
            metadata={
                "relative_path": relative,
                "analysis_backend": "clang-libtooling",
                "advanced_facts_complete": True,
            },
        )
        self.symbols[symbol_id] = symbol
        return symbol

    def _symbol_fact(self, fact: Mapping[str, Any]) -> None:
        key = _string(fact, "key")
        symbol_id = "sym_" + _hash_text(key)[:32]
        self.keys[key] = symbol_id
        span = self._span(_mapping(fact, "span"))
        metadata_raw = fact.get("metadata", {})
        if not isinstance(metadata_raw, dict):
            raise AnalyzerProtocolError("symbol metadata must be an object")
        metadata = dict(metadata_raw)
        metadata.pop("advanced_facts_complete", None)
        metadata["analyzer_protocol"] = PROTOCOL_VERSION
        metadata["analyzer_clang_major"] = REQUIRED_CLANG_MAJOR
        if key.startswith("usr:"):
            metadata["usr"] = key.removeprefix("usr:")
        symbol = CodeSymbol(
            id=symbol_id,
            qualified_name=_string(fact, "qualified_name"),
            kind=SymbolKind(_string(fact, "kind")),
            span=span,
            signature=_optional_string(fact, "signature"),
            documentation=_optional_string(fact, "documentation"),
            source_hash=_hash_text(_optional_string(fact, "source_text")),
            source_text=_optional_string(fact, "source_text"),
            build_configuration_id=self.configuration.id,
            translation_unit_id=self.unit_id,
            build_variant=self.configuration.build_variant,
            variant_id=self._variant_id(symbol_id),
            metadata=metadata,
        )
        previous = self.symbols.get(symbol_id)
        if previous is None or (
            bool(symbol.metadata.get("is_definition"))
            and not bool(previous.metadata.get("is_definition"))
        ):
            self.symbols[symbol_id] = symbol

    def _occurrence_fact(self, fact: Mapping[str, Any]) -> SymbolOccurrence:
        key = _string(fact, "symbol_key")
        symbol_id = self._known_id(key)
        span = self._span(_mapping(fact, "span"))
        kind = OccurrenceKind(_string(fact, "kind"))
        enclosing_key = fact.get("enclosing_key")
        enclosing_id = (
            self._known_id(enclosing_key)
            if isinstance(enclosing_key, str) and enclosing_key
            else None
        )
        metadata: dict[str, Any] = {}
        if kind == OccurrenceKind.MACRO_EXPANSION:
            metadata = {
                "spelling_span": _span_payload(self._span(_mapping(fact, "spelling_span"))),
                "expansion_span": _span_payload(self._span(_mapping(fact, "expansion_span"))),
            }
        occurrence_id = (
            "occ_"
            + _hash_text(
                symbol_id,
                self.configuration.build_variant,
                self.unit_id,
                kind.value,
                str(span.path),
                str(span.start_line),
                str(span.start_column),
                str(span.end_line),
                str(span.end_column),
            )[:32]
        )
        return SymbolOccurrence(
            occurrence_id,
            symbol_id,
            span,
            kind,
            enclosing_id,
            self.unit_id,
            self.configuration.id,
            self.configuration.build_variant,
            metadata,
        )

    def _edge_fact(self, fact: Mapping[str, Any]) -> GraphEdge | None:
        if fact.get("fact") == "include":
            relation = GraphRelation.INCLUDES
        else:
            relation = GraphRelation(_string(fact, "relation"))
        source_key = _string(fact, "source_key")
        target_raw = fact.get("target_key")
        if not isinstance(target_raw, str):
            return None
        source_id = self._known_id(source_key)
        target_id = self._known_id(target_raw)
        location = ("", "", "", "", "")
        if isinstance(fact.get("span"), dict):
            span = self._span(fact["span"])
            location = (
                str(span.path),
                str(span.start_line),
                str(span.start_column),
                str(span.end_line),
                str(span.end_column),
            )
        edge_id = (
            "edge_"
            + _hash_text(
                self.configuration.build_variant,
                self.unit_id,
                source_id,
                target_id,
                relation.value,
                *location,
            )[:32]
        )
        return GraphEdge(
            source_id,
            target_id,
            relation,
            self.unit_id,
            edge_id,
            self.configuration.id,
            self.configuration.build_variant,
        )

    def _known_id(self, key: str) -> str:
        try:
            return self.keys[key]
        except KeyError as error:
            raise AnalyzerProtocolError("analyzer fact references an unknown symbol") from error

    def _path(self, raw: str) -> Path:
        cached = self.path_cache.get(raw)
        if cached is not None:
            return cached
        path = Path(raw).resolve(strict=False)
        if not _within(path, self.root) or not path.is_file():
            raise AnalyzerProtocolError("analyzer returned a path outside the project")
        self.path_cache[raw] = path
        return path

    def _span(self, raw: Mapping[str, Any]) -> SourceSpan:
        return SourceSpan(
            self._path(_string(raw, "path")),
            _integer(raw, "start_line"),
            _integer(raw, "end_line"),
            _integer(raw, "start_column"),
            _integer(raw, "end_column"),
        )

    def _variant_id(self, symbol_id: str) -> str:
        return (
            "variant_" + _hash_text(self.configuration.build_variant, self.unit_id, symbol_id)[:32]
        )


def _mapping(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = record.get(name)
    if not isinstance(value, dict):
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _string(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _optional_string(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name, "")
    if not isinstance(value, str):
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _integer(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _non_negative_integer(record: Mapping[str, Any], name: str) -> int:
    value = _integer(record, name)
    if value < 0:
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _optional_non_negative_integer(record: Mapping[str, Any], name: str) -> int | None:
    value = record.get(name)
    if value is None:
        return None
    return _non_negative_integer(record, name)


def _optional_key(record: Mapping[str, Any], name: str) -> str | None:
    value = record.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _string_list(record: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = record.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return tuple(value)


def _positive_mapping_integer(record: Mapping[str, Any], name: str) -> int:
    value = _integer(record, name)
    if value <= 0:
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _boolean(record: Mapping[str, Any], name: str) -> bool:
    value = record.get(name)
    if not isinstance(value, bool):
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _number(record: Mapping[str, Any], name: str) -> float:
    value = record.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return result


def _span_payload(span: SourceSpan) -> dict[str, Any]:
    return {
        "path": str(span.path),
        "start_line": span.start_line,
        "end_line": span.end_line,
        "start_column": span.start_column,
        "end_column": span.end_column,
    }


def _add_indexed_override_candidates(
    callsites: Sequence[CallSite],
    targets: Sequence[CallTarget],
    edges: Sequence[GraphEdge],
) -> tuple[CallTarget, ...]:
    """Add build-local transitive overrides without claiming an open-world complete set."""

    direct_overrides: dict[tuple[str, str], set[str]] = {}
    for edge in edges:
        if edge.relation == GraphRelation.OVERRIDES:
            direct_overrides.setdefault((edge.build_variant, edge.target_id), set()).add(
                edge.source_id
            )
    result = {(target.callsite_id, target.target_symbol_id): target for target in targets}
    for site in callsites:
        if site.dispatch_kind != CallDispatchKind.VIRTUAL or site.static_target_symbol_id is None:
            continue
        pending = sorted(
            direct_overrides.get((site.build_variant, site.static_target_symbol_id), ())
        )
        visited: set[str] = set()
        while pending:
            target_id = pending.pop(0)
            if target_id in visited:
                continue
            visited.add(target_id)
            pending.extend(sorted(direct_overrides.get((site.build_variant, target_id), ())))
            key = (site.id, target_id)
            if key in result:
                continue
            derivation = "indexed_override_candidate"
            result[key] = CallTarget(
                id="call_target_"
                + _hash_text(
                    site.build_variant,
                    site.build_configuration_id,
                    site.translation_unit_id,
                    site.id,
                    target_id,
                    CallTargetCertainty.POSSIBLE.value,
                    derivation,
                )[:32],
                callsite_id=site.id,
                target_symbol_id=target_id,
                certainty=CallTargetCertainty.POSSIBLE,
                confidence=0.5,
                confidence_reason=(
                    "target overrides the statically selected virtual method in this build; "
                    "the value is deterministic ranking evidence, not a probability"
                ),
                derivation=derivation,
                evidence_span=site.expansion_span,
                translation_unit_id=site.translation_unit_id,
                build_configuration_id=site.build_configuration_id,
                build_variant=site.build_variant,
            )
    return tuple(
        sorted(result.values(), key=lambda item: (item.callsite_id, item.target_symbol_id, item.id))
    )
