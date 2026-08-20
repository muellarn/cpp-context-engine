"""Lexical, symbol, and vector candidate-search adapters."""

from cpp_context_engine.search.embeddings import (
    DeterministicLocalEmbeddingProvider,
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingProvider,
)
from cpp_context_engine.search.protocols import LexicalSearch, SymbolSearch, VectorSearch
from cpp_context_engine.search.sqlite import SQLiteLexicalSearch, SQLiteSymbolSearch
from cpp_context_engine.search.vector import EmbeddingProvider, SQLiteVectorSearch

__all__ = [
    "DeterministicLocalEmbeddingProvider",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "LexicalSearch",
    "OpenAICompatibleEmbeddingProvider",
    "SQLiteLexicalSearch",
    "SQLiteSymbolSearch",
    "SQLiteVectorSearch",
    "SymbolSearch",
    "VectorSearch",
]
