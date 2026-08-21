"""Bounded adapter for the versioned Clang LibTooling companion protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
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
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_INPUT_BYTES = 1_048_576
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1_048_576
DEFAULT_MAX_STDERR_BYTES = 256 * 1024


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
        max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
    ) -> None:
        self.binary = binary.expanduser().resolve(strict=False)
        if timeout_seconds <= 0 or min(max_input_bytes, max_output_bytes, max_stderr_bytes) <= 0:
            raise ValueError("analyzer timeout and byte limits must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_stderr_bytes = max_stderr_bytes
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
        self.probe()
        unit_id = translation_unit_id(configuration)
        request = {
            "type": "analyze",
            "request_id": unit_id,
            "project_root": str(project_root.resolve(strict=False)),
            "source_path": str(configuration.source_path),
            "directory": str(configuration.directory),
            "arguments": list(libclang_arguments(configuration)),
        }
        records = self._invoke((self._hello(), request), output_limit=self.max_output_bytes)
        if not records or records[0].get("type") != "hello":
            raise AnalyzerProtocolError("analyzer did not repeat its validated handshake")
        if self._validate_handshake(records[0]) != self._info:
            raise AnalyzerProtocolError("analyzer handshake changed between invocations")
        if len(records) < 3 or records[1] != {"request_id": unit_id, "type": "begin"}:
            raise AnalyzerProtocolError("analyzer response has no matching begin record")
        complete = records[-1]
        if complete.get("type") != "complete" or complete.get("request_id") != unit_id:
            raise AnalyzerProtocolError("analyzer response is incomplete")
        if complete.get("success") is not True:
            raise AnalyzerProtocolError("analyzer did not complete successfully")
        facts = records[2:-1]
        if not all(record.get("type") == "fact" for record in facts):
            raise AnalyzerProtocolError("analyzer emitted a non-fact record during analysis")
        return tuple(facts)

    @staticmethod
    def _hello() -> dict[str, Any]:
        return {
            "type": "hello",
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "required_clang_major": REQUIRED_CLANG_MAJOR,
            "required_capabilities": sorted(REQUIRED_CAPABILITIES),
        }

    def _invoke(
        self, requests: Sequence[Mapping[str, Any]], *, output_limit: int
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
            )
        except OSError as error:
            raise AnalyzerUnavailableError("configured analyzer could not be started") from error
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        stdout = bytearray()
        stderr = bytearray()
        exceeded = threading.Event()

        def read_bounded(stream: Any, destination: bytearray, limit: int) -> None:
            while chunk := stream.read(64 * 1024):
                remaining = limit + 1 - len(destination)
                if remaining > 0:
                    destination.extend(chunk[:remaining])
                if len(destination) > limit or len(chunk) > remaining:
                    exceeded.set()
                    process.kill()
                    return

        readers = (
            threading.Thread(
                target=read_bounded,
                args=(process.stdout, stdout, output_limit),
                daemon=True,
            ),
            threading.Thread(
                target=read_bounded,
                args=(process.stderr, stderr, self.max_stderr_bytes),
                daemon=True,
            ),
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
        try:
            process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        finally:
            writer.join(timeout=2)
            for reader in readers:
                reader.join(timeout=2)
        if timed_out:
            raise AnalyzerLimitError("analyzer exceeded the configured timeout")
        if exceeded.is_set():
            raise AnalyzerLimitError("analyzer exceeded a configured output limit")
        try:
            records = [json.loads(line) for line in stdout.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AnalyzerProtocolError("analyzer returned malformed JSONL") from error
        if not all(isinstance(record, dict) for record in records):
            raise AnalyzerProtocolError("analyzer JSONL records must be objects")
        error_record = next((record for record in records if record.get("type") == "error"), None)
        if error_record is not None:
            code = error_record.get("code", "unknown")
            if not isinstance(code, str) or not code.replace("_", "").isalnum():
                code = "unknown"
            raise AnalyzerProtocolError(f"analyzer rejected the request ({code})")
        if process.returncode != 0:
            raise AnalyzerProtocolError("analyzer process failed; inspect compiler diagnostics")
        return records


class NativeClangIngestor:
    """Convert complete companion facts into the existing durable domain model."""

    def __init__(self, client: NativeAnalyzerClient) -> None:
        self.client = client

    analysis_backend = "clang-libtooling"
    advanced_facts_complete = True

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
        root = project_root.resolve(strict=False)
        self.client.probe()
        selected = tuple(configurations)

        def analyze(configuration: BuildConfiguration) -> IngestionBatch:
            return _FactBatchBuilder(root, configuration).build(
                self.client.analyze(root, configuration)
            )

        # Each configuration is a separate companion process. Consume futures in
        # submission order so durable IDs and fact aggregation remain deterministic.
        # Seven companions keep the measured aggregate process tree below the
        # 2-GiB benchmark budget while retaining bounded TU parallelism.
        worker_count = min(7, max(1, len(selected)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(analyze, configuration) for configuration in selected[:worker_count]
            ]
            batches = []
            next_configuration = worker_count
            try:
                for future in futures:
                    batches.append(future.result())
                    if next_configuration < len(selected):
                        futures.append(executor.submit(analyze, selected[next_configuration]))
                        next_configuration += 1
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
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
    def __init__(self, root: Path, configuration: BuildConfiguration) -> None:
        self.root = root
        self.configuration = configuration
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

    def build(self, facts: Sequence[Mapping[str, Any]]) -> IngestionBatch:
        for fact in facts:
            if fact.get("fact") == "file":
                self._file_fact(fact)
            elif fact.get("fact") == "symbol":
                self._symbol_fact(fact)
            elif fact.get("fact") == "include" and isinstance(fact.get("resolved_path"), str):
                path = self._path(fact["resolved_path"])
                self._file_symbol(path)
        occurrences: dict[str, SymbolOccurrence] = {}
        edges: dict[str, GraphEdge] = {}
        for fact in facts:
            if fact.get("fact") == "occurrence":
                occurrence = self._occurrence_fact(fact)
                occurrences[occurrence.id] = occurrence
            elif fact.get("fact") in {"edge", "include"}:
                edge = self._edge_fact(fact)
                if edge is not None:
                    edges[edge.id] = edge
        cfg_graphs, cfg_blocks, cfg_elements, cfg_edges = self._cfg_facts(facts)
        callsites, call_targets = self._call_facts(facts)
        analyses, locations, accesses, evidence = self._data_flow_facts(facts)
        summaries, effects, origins, argument_bindings, result_bindings = (
            self._interprocedural_facts(facts)
        )
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
            advanced_facts_complete=NativeClangIngestor.advanced_facts_complete,
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
        self, facts: Sequence[Mapping[str, Any]]
    ) -> tuple[
        tuple[DataFlowAnalysis, ...],
        tuple[MemoryLocation, ...],
        tuple[DataAccess, ...],
        tuple[DataFlowEvidence, ...],
    ]:
        analysis_facts = [fact for fact in facts if fact.get("fact") == "data_flow_analysis_v1"]
        location_facts = [fact for fact in facts if fact.get("fact") == "memory_location_v1"]
        access_facts = [fact for fact in facts if fact.get("fact") == "data_access_v1"]
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
                        for fact in facts
                        if fact.get("fact") == "data_flow_evidence_v1"
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
        self, facts: Sequence[Mapping[str, Any]]
    ) -> tuple[
        tuple[FunctionSummary, ...],
        tuple[SummaryEffect, ...],
        tuple[SummaryReturnOrigin, ...],
        tuple[CallArgumentBinding, ...],
        tuple[CallResultBinding, ...],
    ]:
        summary_facts = [fact for fact in facts if fact.get("fact") == "function_summary_v1"]
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
                        for fact in facts
                        if fact.get("fact") == "summary_effect_v1"
                    ),
                    key=lambda item: item.id,
                )
            )
            origins = tuple(
                sorted(
                    (
                        self._summary_return_origin_fact(fact)
                        for fact in facts
                        if fact.get("fact") == "summary_return_origin_v1"
                    ),
                    key=lambda item: item.id,
                )
            )
            arguments = tuple(
                sorted(
                    (
                        self._call_argument_binding_fact(fact)
                        for fact in facts
                        if fact.get("fact") == "call_argument_binding_v1"
                    ),
                    key=lambda item: item.id,
                )
            )
            results = tuple(
                sorted(
                    (
                        self._call_result_binding_fact(fact)
                        for fact in facts
                        if fact.get("fact") == "call_result_binding_v1"
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
        self, facts: Sequence[Mapping[str, Any]]
    ) -> tuple[tuple[CallSite, ...], tuple[CallTarget, ...]]:
        site_facts = [fact for fact in facts if fact.get("fact") == "callsite_v1"]
        for fact in site_facts:
            key = _string(fact, "key")
            owner_id = self._known_id(_string(fact, "owner_key"))
            spelling = self._span(_mapping(fact, "spelling_span"))
            expansion = self._span(_mapping(fact, "expansion_span"))
            self.callsite_ids[key] = (
                "callsite_"
                + _hash_text(
                    self.configuration.build_variant,
                    self.configuration.id,
                    self.unit_id,
                    owner_id,
                    str(spelling.path),
                    str(spelling.start_line),
                    str(spelling.start_column),
                    str(spelling.end_line),
                    str(spelling.end_column),
                    str(expansion.path),
                    str(expansion.start_line),
                    str(expansion.start_column),
                    str(expansion.end_line),
                    str(expansion.end_column),
                    key,
                )[:32]
            )
        sites_by_key = {_string(fact, "key"): self._callsite_fact(fact) for fact in site_facts}
        for fact in facts:
            if fact.get("fact") != "callsite_resolution_v1":
                continue
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
                (
                    self._call_target_fact(fact)
                    for fact in facts
                    if fact.get("fact") == "call_target_v1"
                ),
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
            spelling_span=self._span(_mapping(fact, "spelling_span")),
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
        self, facts: Sequence[Mapping[str, Any]]
    ) -> tuple[
        tuple[CfgGraph, ...],
        tuple[CfgBlock, ...],
        tuple[CfgElement, ...],
        tuple[CfgEdge, ...],
    ]:
        graph_facts = [fact for fact in facts if fact.get("fact") == "cfg_graph_v1"]
        block_facts = [fact for fact in facts if fact.get("fact") == "cfg_block_v1"]
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
        for fact in facts:
            if fact.get("fact") != "cfg_element_v1":
                continue
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
        for fact in facts:
            fact_kind = fact.get("fact")
            if fact_kind == "cfg_element_v1":
                graph_id = self._known_cfg_graph(_string(fact, "graph_key"))
                if block_graph_ids.get(_string(fact, "block_key")) != graph_id:
                    raise AnalyzerProtocolError(
                        "analyzer CFG facts have inconsistent graph references"
                    )
            elif fact_kind == "cfg_edge_v1":
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
                (
                    self._cfg_element_fact(fact)
                    for fact in facts
                    if fact.get("fact") == "cfg_element_v1"
                ),
                key=lambda item: (item.graph_id, item.block_id, item.index, item.id),
            )
        )
        edges = tuple(
            sorted(
                (self._cfg_edge_fact(fact) for fact in facts if fact.get("fact") == "cfg_edge_v1"),
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
