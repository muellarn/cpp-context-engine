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
from cpp_context_engine.ingestion.deep import (
    DeepCancellation,
    DeepMaterializer,
    MaterializeDeepRequest,
    MaterializeDeepResult,
)
from cpp_context_engine.ingestion.indexer import IndexingResult, ProjectIndexer
from cpp_context_engine.ingestion.native import (
    AnalyzerInfo,
    AnalyzerLimitError,
    AnalyzerProtocolError,
    AnalyzerUnavailableError,
    NativeAnalyzerClient,
    NativeClangIngestor,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch, Ingestor
from cpp_context_engine.ingestion.telemetry import (
    AnalyzerPipelineEvent,
    AnalyzerPipelineObserver,
    AnalyzerSlotIdleError,
    AnalyzerSlotIdleGate,
    AnalyzerTelemetryError,
)

__all__ = [
    "ClangIngestor",
    "ClangUnavailableError",
    "CompilationDatabase",
    "CompilationDatabaseError",
    "DeepMaterializer",
    "DeepCancellation",
    "IndexingResult",
    "IngestionBatch",
    "Ingestor",
    "AnalyzerInfo",
    "AnalyzerLimitError",
    "AnalyzerPipelineEvent",
    "AnalyzerPipelineObserver",
    "AnalyzerProtocolError",
    "AnalyzerSlotIdleError",
    "AnalyzerSlotIdleGate",
    "AnalyzerTelemetryError",
    "AnalyzerUnavailableError",
    "NativeAnalyzerClient",
    "NativeClangIngestor",
    "MaterializeDeepRequest",
    "MaterializeDeepResult",
    "ProjectIndexer",
    "TranslationUnitError",
]
