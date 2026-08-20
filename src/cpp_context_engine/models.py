"""Transport-neutral domain types shared by all adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any


class SymbolKind(StrEnum):
    FILE = "file"
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


class GraphDirection(StrEnum):
    """Traversal direction relative to the symbol at each graph hop."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class OccurrenceKind(StrEnum):
    DECLARATION = "declaration"
    DEFINITION = "definition"
    REFERENCE = "reference"
    CALL = "call"
    TYPE = "type"
    MACRO_EXPANSION = "macro_expansion"


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
class BuildConfiguration:
    """One normalized entry from a JSON compilation database."""

    id: str
    source_path: Path
    directory: Path
    arguments: tuple[str, ...]
    command_hash: str
    output: Path | None = None


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    """The durable fingerprint and diagnostics for one compiler invocation."""

    id: str
    build_configuration_id: str
    source_path: Path
    content_hash: str
    dependencies: tuple[tuple[Path, str], ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CodeSymbol:
    id: str
    qualified_name: str
    kind: SymbolKind
    span: SourceSpan
    signature: str = ""
    documentation: str = ""
    source_hash: str = ""
    source_text: str = ""
    build_configuration_id: str = ""
    translation_unit_id: str = ""
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
    translation_unit_id: str = ""


@dataclass(frozen=True, slots=True)
class SymbolOccurrence:
    id: str
    symbol_id: str
    span: SourceSpan
    kind: OccurrenceKind
    enclosing_symbol_id: str | None = None
    translation_unit_id: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("occurrence id must not be empty")
        if not self.symbol_id.strip():
            raise ValueError("occurrence symbol id must not be empty")


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
