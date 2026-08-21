"""Incremental orchestration between a compilation database and durable storage."""

from __future__ import annotations

import hashlib
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
        if changed:
            batch = self._ingestor.ingest_configurations(project_root, changed)
        else:
            batch = IngestionBatch((), (), (), (), (), (variant,))
        if changed:
            batch = IngestionBatch(
                build_configurations=batch.build_configurations,
                translation_units=batch.translation_units,
                symbols=batch.symbols,
                occurrences=batch.occurrences,
                edges=batch.edges,
                build_variants=(variant,),
                cfg_graphs=batch.cfg_graphs,
                cfg_blocks=batch.cfg_blocks,
                cfg_elements=batch.cfg_elements,
                cfg_edges=batch.cfg_edges,
                callsites=batch.callsites,
                call_targets=batch.call_targets,
                data_flow_analyses=batch.data_flow_analyses,
                memory_locations=batch.memory_locations,
                data_accesses=batch.data_accesses,
                data_flow_evidence=batch.data_flow_evidence,
            )
        removed = len(set(previous) - current_ids)
        self._store.apply_ingestion(
            project_root,
            batch,
            current_translation_unit_ids=current_ids,
            build_variant=variant,
        )
        return IndexingResult(
            indexed_translation_units=len(changed),
            skipped_translation_units=len(database.configurations) - len(changed),
            removed_translation_units=removed,
            indexed_symbols=len(batch.symbols),
            indexed_occurrences=len(batch.occurrences),
            indexed_edges=len(batch.edges),
            indexed_cfg_graphs=len(batch.cfg_graphs),
            indexed_cfg_blocks=len(batch.cfg_blocks),
            indexed_cfg_elements=len(batch.cfg_elements),
            indexed_cfg_edges=len(batch.cfg_edges),
            indexed_callsites=len(batch.callsites),
            indexed_call_targets=len(batch.call_targets),
            indexed_data_flow_analyses=len(batch.data_flow_analyses),
            indexed_memory_locations=len(batch.memory_locations),
            indexed_data_accesses=len(batch.data_accesses),
            indexed_data_flow_evidence=len(batch.data_flow_evidence),
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
