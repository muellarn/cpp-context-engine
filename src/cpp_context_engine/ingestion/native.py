"""Bounded adapter for the versioned Clang LibTooling companion protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    libclang_arguments,
    translation_unit_id,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    CfgBlock,
    CfgBlockRole,
    CfgEdge,
    CfgEdgeKind,
    CfgElement,
    CfgGraph,
    CodeSymbol,
    GraphEdge,
    GraphRelation,
    OccurrenceKind,
    SourceSpan,
    SymbolKind,
    SymbolOccurrence,
    TranslationUnit,
)

PROTOCOL = "cpp-context-clang-facts"
PROTOCOL_VERSION = 2
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
        batches = [
            _FactBatchBuilder(root, configuration).build(self.client.analyze(root, configuration))
            for configuration in configurations
        ]
        return IngestionBatch(
            build_configurations=tuple(
                configuration for batch in batches for configuration in batch.build_configurations
            ),
            translation_units=tuple(unit for batch in batches for unit in batch.translation_units),
            symbols=tuple(symbol for batch in batches for symbol in batch.symbols),
            occurrences=tuple(occurrence for batch in batches for occurrence in batch.occurrences),
            edges=tuple(edge for batch in batches for edge in batch.edges),
            cfg_graphs=tuple(graph for batch in batches for graph in batch.cfg_graphs),
            cfg_blocks=tuple(block for batch in batches for block in batch.cfg_blocks),
            cfg_elements=tuple(element for batch in batches for element in batch.cfg_elements),
            cfg_edges=tuple(edge for batch in batches for edge in batch.cfg_edges),
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
        self.cfg_block_ids: dict[str, str] = {}

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
            (self.configuration,),
            (unit,),
            tuple(sorted(self.symbols.values(), key=lambda item: item.id)),
            tuple(sorted(occurrences.values(), key=lambda item: item.id)),
            tuple(sorted(edges.values(), key=lambda item: item.id)),
            (),
            cfg_graphs,
            cfg_blocks,
            cfg_elements,
            cfg_edges,
        )

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
            self.cfg_block_ids[block_key] = "cfg_block_" + _hash_text(graph_id, str(index))[:32]

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
            id="cfg_element_" + _hash_text(graph_id, block_id, str(index))[:32],
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
        path = Path(raw).resolve(strict=False)
        if not _within(path, self.root) or not path.is_file():
            raise AnalyzerProtocolError("analyzer returned a path outside the project")
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


def _boolean(record: Mapping[str, Any], name: str) -> bool:
    value = record.get(name)
    if not isinstance(value, bool):
        raise AnalyzerProtocolError(f"analyzer record has invalid {name}")
    return value


def _span_payload(span: SourceSpan) -> dict[str, Any]:
    return {
        "path": str(span.path),
        "start_line": span.start_line,
        "end_line": span.end_line,
        "start_column": span.start_column,
        "end_column": span.end_column,
    }
