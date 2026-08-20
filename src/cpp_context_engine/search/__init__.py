"""Lexical, symbol, and vector candidate-search adapters."""

from cpp_context_engine.search.protocols import LexicalSearch, SymbolSearch, VectorSearch
from cpp_context_engine.search.vector import EmbeddingProvider, SQLiteVectorSearch

__all__ = [
    "EmbeddingProvider",
    "LexicalSearch",
    "SQLiteVectorSearch",
    "SymbolSearch",
    "VectorSearch",
]
