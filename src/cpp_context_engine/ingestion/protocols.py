"""Interfaces implemented by Clang, SCIP, or other ingestion adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cpp_context_engine.models import (
    BuildConfiguration,
    BuildVariant,
    CodeSymbol,
    GraphEdge,
    SymbolOccurrence,
    TranslationUnit,
)


@dataclass(frozen=True, slots=True)
class IngestionBatch:
    build_configurations: tuple[BuildConfiguration, ...]
    translation_units: tuple[TranslationUnit, ...]
    symbols: tuple[CodeSymbol, ...]
    occurrences: tuple[SymbolOccurrence, ...]
    edges: tuple[GraphEdge, ...]
    build_variants: tuple[BuildVariant, ...] = ()


class Ingestor(Protocol):
    def ingest(self, project_root: Path, compilation_database: Path) -> IngestionBatch:
        """Normalize a configured C++ project into symbols and relationships."""
        ...
