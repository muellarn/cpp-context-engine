"""Incremental orchestration between a compilation database and durable storage."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    translation_unit_id,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    DEFAULT_BUILD_VARIANT,
    BuildConfiguration,
    BuildVariant,
)
from cpp_context_engine.storage.sqlite import SQLiteStore, TranslationUnitState


@dataclass(frozen=True, slots=True)
class IndexingResult:
    indexed_translation_units: int
    skipped_translation_units: int
    removed_translation_units: int
    indexed_symbols: int
    indexed_occurrences: int
    indexed_edges: int
    indexed_cfg_graphs: int = 0
    indexed_cfg_blocks: int = 0
    indexed_cfg_elements: int = 0
    indexed_cfg_edges: int = 0
    indexed_callsites: int = 0
    indexed_call_targets: int = 0
    indexed_data_flow_analyses: int = 0
    indexed_memory_locations: int = 0
    indexed_data_accesses: int = 0
    indexed_data_flow_evidence: int = 0
    indexed_function_summaries: int = 0
    indexed_summary_effects: int = 0
    indexed_summary_return_origins: int = 0
    indexed_call_argument_bindings: int = 0
    indexed_call_result_bindings: int = 0
    indexed_interprocedural_flows: int = 0
    invalidated_function_summaries: int = 0


class ProjectIndexer:
    """Reparse only units whose command, source, or project dependency changed."""

    def __init__(self, ingestor: Any, store: SQLiteStore) -> None:
        self._ingestor = ingestor
        self._store = store

    def index(
        self,
        project_root: Path,
        compilation_database: Path,
        *,
        build_variant: BuildVariant | None = None,
    ) -> IndexingResult:
        project_root = project_root.resolve(strict=False)
        variant = build_variant or BuildVariant(DEFAULT_BUILD_VARIANT, compilation_database)
        if variant.compilation_database != compilation_database.resolve(strict=False):
            raise ValueError("build variant compilation database does not match index request")
        database = CompilationDatabase.load(compilation_database, build_variant=variant.name)
        previous = self._store.translation_unit_states(project_root, build_scope=(variant.name,))
        current_ids = frozenset(translation_unit_id(config) for config in database.configurations)
        changed = [
            configuration
            for configuration in database.configurations
            if self._needs_reindex(
                configuration,
                previous.get(translation_unit_id(configuration)),
                analysis_backend=getattr(self._ingestor, "analysis_backend", "unknown"),
                advanced_facts_complete=bool(
                    getattr(self._ingestor, "advanced_facts_complete", False)
                ),
            )
        ]
        counts = {result_name: 0 for result_name, _batch_name in _BATCH_COUNT_FIELDS}
        if changed:
            iter_batches = getattr(self._ingestor, "iter_configuration_batches", None)
            if iter_batches is None:
                batch_stream: Iterator[IngestionBatch] = iter(
                    (self._ingestor.ingest_configurations(project_root, changed),)
                )
            else:
                batch_stream = iter(iter_batches(project_root, changed))
        else:
            batch_stream = iter(())

        def counted_batches() -> Iterator[IngestionBatch]:
            for batch in batch_stream:
                for result_name, batch_name in _BATCH_COUNT_FIELDS:
                    counts[result_name] += len(getattr(batch, batch_name))
                try:
                    yield batch
                finally:
                    # Do not retain an already staged TU while the producer is
                    # blocked building the next potentially large batch.
                    del batch

        removed = len(set(previous) - current_ids)
        try:
            invalidated_summaries = self._store.apply_ingestion_batches(
                project_root,
                counted_batches(),
                current_translation_unit_ids=current_ids,
                changed_translation_unit_ids=frozenset(
                    translation_unit_id(configuration) for configuration in changed
                ),
                build_variant=variant,
            )
        finally:
            close = getattr(batch_stream, "close", None)
            if close is not None:
                close()
        return IndexingResult(
            indexed_translation_units=len(changed),
            skipped_translation_units=len(database.configurations) - len(changed),
            removed_translation_units=removed,
            invalidated_function_summaries=invalidated_summaries,
            **counts,
        )

    @staticmethod
    def _needs_reindex(
        configuration: BuildConfiguration,
        state: TranslationUnitState | None,
        *,
        analysis_backend: str,
        advanced_facts_complete: bool,
    ) -> bool:
        if state is None:
            return True
        if state.command_hash != configuration.command_hash:
            return True
        if (
            state.analysis_backend != analysis_backend
            or state.advanced_facts_complete != advanced_facts_complete
        ):
            return True
        source_path = configuration.source_path
        if not source_path.is_file() or _file_hash(source_path) != state.content_hash:
            return True
        for path, expected_hash in state.dependencies:
            if not path.is_file() or _file_hash(path) != expected_hash:
                return True
        return False


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_BATCH_COUNT_FIELDS = (
    ("indexed_symbols", "symbols"),
    ("indexed_occurrences", "occurrences"),
    ("indexed_edges", "edges"),
    ("indexed_cfg_graphs", "cfg_graphs"),
    ("indexed_cfg_blocks", "cfg_blocks"),
    ("indexed_cfg_elements", "cfg_elements"),
    ("indexed_cfg_edges", "cfg_edges"),
    ("indexed_callsites", "callsites"),
    ("indexed_call_targets", "call_targets"),
    ("indexed_data_flow_analyses", "data_flow_analyses"),
    ("indexed_memory_locations", "memory_locations"),
    ("indexed_data_accesses", "data_accesses"),
    ("indexed_data_flow_evidence", "data_flow_evidence"),
    ("indexed_function_summaries", "function_summaries"),
    ("indexed_summary_effects", "summary_effects"),
    ("indexed_summary_return_origins", "summary_return_origins"),
    ("indexed_call_argument_bindings", "call_argument_bindings"),
    ("indexed_call_result_bindings", "call_result_bindings"),
    ("indexed_interprocedural_flows", "interprocedural_flows"),
)
