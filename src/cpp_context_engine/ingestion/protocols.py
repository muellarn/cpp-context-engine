"""Interfaces implemented by Clang, SCIP, or other ingestion adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cpp_context_engine.models import CodeSymbol, GraphEdge


@dataclass(frozen=True, slots=True)
class IngestionBatch:
    symbols: tuple[CodeSymbol, ...]
    edges: tuple[GraphEdge, ...]


class Ingestor(Protocol):
    def ingest(self, project_root: Path, compilation_database: Path) -> IngestionBatch:
        """Normalize a configured C++ project into symbols and relationships."""
        ...
