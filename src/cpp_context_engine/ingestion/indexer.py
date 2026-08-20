"""Incremental orchestration between a compilation database and durable storage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from cpp_context_engine.ingestion.clang import ClangIngestor
from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    translation_unit_id,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import BuildConfiguration
from cpp_context_engine.storage.sqlite import SQLiteStore, TranslationUnitState


@dataclass(frozen=True, slots=True)
class IndexingResult:
    indexed_translation_units: int
    skipped_translation_units: int
    removed_translation_units: int
    indexed_symbols: int
    indexed_occurrences: int
    indexed_edges: int


class ProjectIndexer:
    """Reparse only units whose command, source, or project dependency changed."""

    def __init__(self, ingestor: ClangIngestor, store: SQLiteStore) -> None:
        self._ingestor = ingestor
        self._store = store

    def index(self, project_root: Path, compilation_database: Path) -> IndexingResult:
        project_root = project_root.resolve(strict=False)
        database = CompilationDatabase.load(compilation_database)
        previous = self._store.translation_unit_states(project_root)
        current_ids = frozenset(translation_unit_id(config) for config in database.configurations)
        changed = [
            configuration
            for configuration in database.configurations
            if self._needs_reindex(configuration, previous.get(translation_unit_id(configuration)))
        ]
        if changed:
            batch = self._ingestor.ingest_configurations(project_root, changed)
        else:
            batch = IngestionBatch((), (), (), (), ())
        removed = len(set(previous) - current_ids)
        self._store.apply_ingestion(
            project_root,
            batch,
            current_translation_unit_ids=current_ids,
        )
        return IndexingResult(
            indexed_translation_units=len(changed),
            skipped_translation_units=len(database.configurations) - len(changed),
            removed_translation_units=removed,
            indexed_symbols=len(batch.symbols),
            indexed_occurrences=len(batch.occurrences),
            indexed_edges=len(batch.edges),
        )

    @staticmethod
    def _needs_reindex(
        configuration: BuildConfiguration, state: TranslationUnitState | None
    ) -> bool:
        if state is None:
            return True
        if state.command_hash != configuration.command_hash:
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
