"""C++ source and compiler-index ingestion."""

from cpp_context_engine.ingestion.clang import (
    ClangIngestor,
    ClangUnavailableError,
    TranslationUnitError,
)
from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    CompilationDatabaseError,
)
from cpp_context_engine.ingestion.indexer import IndexingResult, ProjectIndexer
from cpp_context_engine.ingestion.protocols import IngestionBatch, Ingestor

__all__ = [
    "ClangIngestor",
    "ClangUnavailableError",
    "CompilationDatabase",
    "CompilationDatabaseError",
    "IndexingResult",
    "IngestionBatch",
    "Ingestor",
    "ProjectIndexer",
    "TranslationUnitError",
]
