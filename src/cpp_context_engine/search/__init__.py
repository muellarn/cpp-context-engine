"""Lexical and vector candidate-search contracts."""

from cpp_context_engine.search.protocols import LexicalSearch, VectorSearch
from cpp_context_engine.search.vector import EmbeddingProvider, SQLiteVectorSearch

__all__ = ["EmbeddingProvider", "LexicalSearch", "SQLiteVectorSearch", "VectorSearch"]
