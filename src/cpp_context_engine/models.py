"""Transport-neutral domain types shared by all adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any


class SymbolKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    STRUCT = "struct"
    ENUM = "enum"
    NAMESPACE = "namespace"
    VARIABLE = "variable"
    TYPE_ALIAS = "type_alias"
    MACRO = "macro"
    UNKNOWN = "unknown"


class GraphRelation(StrEnum):
    CONTAINS = "contains"
    REFERENCES = "references"
    CALLS = "calls"
    INHERITS = "inherits"
    OVERRIDES = "overrides"
    USES_TYPE = "uses_type"
    INCLUDES = "includes"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    path: Path
    start_line: int
    end_line: int
    start_column: int = 1
    end_column: int = 1

    def __post_init__(self) -> None:
        if min(self.start_line, self.end_line, self.start_column, self.end_column) < 1:
            raise ValueError("source coordinates are one-based and must be positive")
        if (self.end_line, self.end_column) < (self.start_line, self.start_column):
            raise ValueError("source span end must not precede its start")


@dataclass(frozen=True, slots=True)
class CodeSymbol:
    id: str
    qualified_name: str
    kind: SymbolKind
    span: SourceSpan
    signature: str = ""
    documentation: str = ""
    source_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("symbol id must not be empty")
        if not self.qualified_name.strip():
            raise ValueError("qualified name must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: GraphRelation


@dataclass(frozen=True, slots=True)
class SearchQuery:
    text: str
    limit: int = 20

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("search text must not be empty")
        if self.limit <= 0:
            raise ValueError("search limit must be greater than zero")


@dataclass(frozen=True, slots=True)
class SearchHit:
    symbol: CodeSymbol
    score: float
    source: str
