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


class ProjectIndexer:
    """Reparse only units whose command, source, or project dependency changed."""

    def __init__(self, ingestor: ClangIngestor, store: SQLiteStore) -> None:
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
            if self._needs_reindex(configuration, previous.get(translation_unit_id(configuration)))
        ]
        if changed:
            batch = self._ingestor.ingest_configurations(project_root, changed)
        else:
            batch = IngestionBatch((), (), (), (), (), (variant,))
        if changed:
            batch = IngestionBatch(
                batch.build_configurations,
                batch.translation_units,
                batch.symbols,
                batch.occurrences,
                batch.edges,
                (variant,),
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
