"""Transport-neutral domain types shared by all adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, TypeVar

DEFAULT_BUILD_VARIANT = "default"
MAX_BUILD_VARIANTS = 16
MAX_BUILD_VARIANT_NAME_CHARS = 128


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
    INSTANTIATES = "instantiates"
    SPECIALIZES = "specializes"
    GENERATED_BY_MACRO = "generated_by_macro"


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


class CfgBlockRole(StrEnum):
    NORMAL = "normal"
    ENTRY = "entry"
    NORMAL_EXIT = "normal_exit"
    EXCEPTIONAL_EXIT = "exceptional_exit"


class CfgEdgeKind(StrEnum):
    FALLTHROUGH = "fallthrough"
    TRUE = "true"
    FALSE = "false"
    CASE = "case"
    DEFAULT = "default"
    LOOP_BACK = "loop_back"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"
    GOTO = "goto"
    EXCEPTION = "exception"


class CallDispatchKind(StrEnum):
    """Compiler-observed dispatch form at one syntactic callsite."""

    DIRECT = "direct"
    CONSTRUCTOR = "constructor"
    VIRTUAL = "virtual"
    DEVIRTUALIZED = "devirtualized"
    LAMBDA = "lambda"
    GENERIC_LAMBDA = "generic_lambda"
    FUNCTOR = "functor"
    DEPENDENT_TEMPLATE = "dependent_template"
    UNRESOLVED_INDIRECT = "unresolved_indirect"


class CallTargetCertainty(StrEnum):
    CERTAIN = "certain"
    POSSIBLE = "possible"


class MemoryLocationKind(StrEnum):
    """Compiler-normalized storage modeled by intraprocedural data flow."""

    LOCAL = "local"
    PARAMETER = "parameter"
    GLOBAL = "global"
    RETURN = "return"
    DEREFERENCE = "dereference"
    FIELD = "field"
    UNKNOWN = "unknown"


class DataAccessKind(StrEnum):
    PARAMETER_DEFINITION = "parameter_definition"
    INITIALIZATION = "initialization"
    ASSIGNMENT = "assignment"
    COMPOUND_ASSIGNMENT = "compound_assignment"
    INCREMENT = "increment"
    DECREMENT = "decrement"
    CALL_RETURN = "call_return"
    UNKNOWN_CLOBBER = "unknown_clobber"
    READ = "read"
    CALL_ARGUMENT = "call_argument"
    RETURN_VALUE = "return_value"
    CONDITION = "condition"


class DataFlowRelation(StrEnum):
    REACHING_DEFINITION = "reaching_definition"
    OVERWRITES = "overwrites"
    MUST_ALIAS = "must_alias"
    MAY_ALIAS = "may_alias"


class DataFlowCertainty(StrEnum):
    CERTAIN = "certain"
    POSSIBLE = "possible"


class SummaryEffectKind(StrEnum):
    READ = "read"
    WRITE = "write"
    ESCAPE = "escape"


class SummaryReturnOriginKind(StrEnum):
    LOCATION = "location"
    CALL_RESULT = "call_result"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


class InterproceduralFlowKind(StrEnum):
    ARGUMENT_TO_PARAMETER = "argument_to_parameter"
    RETURN_TO_CALLER = "return_to_caller"
    WRITEBACK = "writeback"


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
class BuildVariant:
    """A named, operator-owned compilation-database view of one project."""

    name: str
    compilation_database: Path
    target: str = ""
    platform: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name or any(character.isspace() for character in name):
            raise ValueError("build variant name must be non-empty and contain no whitespace")
        if len(name) > MAX_BUILD_VARIANT_NAME_CHARS:
            raise ValueError(
                f"build variant name must not exceed {MAX_BUILD_VARIANT_NAME_CHARS} characters"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "compilation_database",
            self.compilation_database.expanduser().resolve(strict=False),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class BuildScope:
    """One or more named variants included in a read operation."""

    variants: tuple[str, ...] = (DEFAULT_BUILD_VARIANT,)

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(name.strip() for name in self.variants if name.strip()))
        if not normalized:
            raise ValueError("build scope must contain at least one variant")
        if any(any(character.isspace() for character in name) for name in normalized):
            raise ValueError("build scope names must not contain whitespace")
        if len(normalized) > MAX_BUILD_VARIANTS:
            raise ValueError(f"build scope must not exceed {MAX_BUILD_VARIANTS} variants")
        if any(len(name) > MAX_BUILD_VARIANT_NAME_CHARS for name in normalized):
            raise ValueError(
                f"build variant names must not exceed {MAX_BUILD_VARIANT_NAME_CHARS} characters"
            )
        object.__setattr__(self, "variants", normalized)

    @classmethod
    def single(cls, name: str = DEFAULT_BUILD_VARIANT) -> BuildScope:
        return cls((name,))

    @property
    def is_union(self) -> bool:
        return len(self.variants) > 1


@dataclass(frozen=True, slots=True)
class BuildConfiguration:
    """One normalized entry from a JSON compilation database."""

    id: str
    source_path: Path
    directory: Path
    arguments: tuple[str, ...]
    command_hash: str
    output: Path | None = None
    build_variant: str = DEFAULT_BUILD_VARIANT


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    """The durable fingerprint and diagnostics for one compiler invocation."""

    id: str
    build_configuration_id: str
    source_path: Path
    content_hash: str
    dependencies: tuple[tuple[Path, str], ...] = ()
    diagnostics: tuple[str, ...] = ()
    build_variant: str = DEFAULT_BUILD_VARIANT
    analysis_backend: str = "unknown"
    advanced_facts_complete: bool = False


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
    build_variant: str = DEFAULT_BUILD_VARIANT
    variant_id: str = ""
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
    translation_unit_id: str = field(default="", compare=False)
    id: str = field(default="", compare=False)
    build_configuration_id: str = field(default="", compare=False)
    build_variant: str = field(default=DEFAULT_BUILD_VARIANT, compare=False)


@dataclass(frozen=True, slots=True)
class SymbolOccurrence:
    id: str
    symbol_id: str
    span: SourceSpan
    kind: OccurrenceKind
    enclosing_symbol_id: str | None = None
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("occurrence id must not be empty")
        if not self.symbol_id.strip():
            raise ValueError("occurrence symbol id must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class CfgGraph:
    """One Clang CFG for one function definition in one concrete build/TU."""

    id: str
    function_symbol_id: str
    entry_block_id: str
    normal_exit_block_id: str
    exceptional_exit_block_id: str | None = None
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT
    clang_major: int = 18
    fact_schema_version: int = 1
    build_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = (
            self.id,
            self.function_symbol_id,
            self.entry_block_id,
            self.normal_exit_block_id,
        )
        if not all(value.strip() for value in required):
            raise ValueError("CFG graph identifiers must not be empty")
        object.__setattr__(self, "build_options", MappingProxyType(dict(self.build_options)))


@dataclass(frozen=True, slots=True)
class CfgBlock:
    id: str
    graph_id: str
    index: int
    role: CfgBlockRole
    reachable: bool
    terminator_kind: str = ""
    terminator_text: str = ""
    terminator_spelling_span: SourceSpan | None = None
    terminator_expansion_span: SourceSpan | None = None
    label_kind: str = ""
    label_text: str = ""
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.graph_id.strip():
            raise ValueError("CFG block identifiers must not be empty")
        if self.index < 0:
            raise ValueError("CFG block index must be non-negative")


@dataclass(frozen=True, slots=True)
class CfgElement:
    id: str
    graph_id: str
    block_id: str
    index: int
    kind: str
    statement_class: str = ""
    text: str = ""
    spelling_span: SourceSpan | None = None
    expansion_span: SourceSpan | None = None
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.graph_id.strip() or not self.block_id.strip():
            raise ValueError("CFG element identifiers must not be empty")
        if self.index < 0 or not self.kind.strip():
            raise ValueError("CFG elements require a non-negative index and kind")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class CfgEdge:
    id: str
    graph_id: str
    source_block_id: str
    target_block_id: str
    kind: CfgEdgeKind
    successor_index: int
    feasible: bool = True
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        required = (self.id, self.graph_id, self.source_block_id, self.target_block_id)
        if not all(value.strip() for value in required):
            raise ValueError("CFG edge identifiers must not be empty")
        if self.successor_index < 0:
            raise ValueError("CFG successor index must be non-negative")


@dataclass(frozen=True, slots=True)
class MacroExpansionFrame:
    """One innermost-to-outermost preprocessor expansion frame."""

    macro_symbol_id: str
    name: str
    spelling_span: SourceSpan
    expansion_span: SourceSpan

    def __post_init__(self) -> None:
        if not self.macro_symbol_id.strip() or not self.name.strip():
            raise ValueError("macro expansion frames require a symbol and name")


@dataclass(frozen=True, slots=True)
class CallSite:
    """One syntactic call expression in one concrete compiler invocation."""

    id: str
    owner_symbol_id: str
    dispatch_kind: CallDispatchKind
    spelling_span: SourceSpan | None
    expansion_span: SourceSpan
    target_set_complete: bool
    static_target_symbol_id: str | None = None
    unresolved_reason: str = ""
    callee_text: str = ""
    expansion_stack: tuple[MacroExpansionFrame, ...] = ()
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.owner_symbol_id.strip():
            raise ValueError("callsite identifiers must not be empty")
        if not self.target_set_complete and not self.unresolved_reason.strip():
            raise ValueError("incomplete callsites require an unresolved reason")
        if self.target_set_complete and self.unresolved_reason:
            raise ValueError("complete callsites must not have an unresolved reason")


@dataclass(frozen=True, slots=True)
class CallTarget:
    """One compiler-derived target edge for a CallSite."""

    id: str
    callsite_id: str
    target_symbol_id: str
    certainty: CallTargetCertainty
    confidence: float
    confidence_reason: str
    derivation: str
    evidence_span: SourceSpan
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.callsite_id.strip() or not self.target_symbol_id.strip():
            raise ValueError("call target identifiers must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("call target confidence must be between zero and one")
        if not self.confidence_reason.strip() or not self.derivation.strip():
            raise ValueError("call targets require confidence and derivation evidence")


@dataclass(frozen=True, slots=True)
class DataFlowAnalysis:
    """One bounded fixed-point result for a concrete CFG graph."""

    id: str
    graph_id: str
    complete: bool
    incomplete_reasons: tuple[str, ...]
    iteration_count: int
    max_iterations: int
    max_alias_targets: int
    max_access_path_depth: int
    max_locations: int
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.graph_id.strip():
            raise ValueError("data-flow analysis identifiers must not be empty")
        if self.iteration_count < 0:
            raise ValueError("data-flow iteration count must be non-negative")
        if (
            min(
                self.max_iterations,
                self.max_alias_targets,
                self.max_access_path_depth,
                self.max_locations,
            )
            <= 0
        ):
            raise ValueError("data-flow limits must be positive")
        reasons = tuple(
            dict.fromkeys(reason.strip() for reason in self.incomplete_reasons if reason.strip())
        )
        if self.complete and reasons:
            raise ValueError("complete data-flow analyses must not contain incomplete reasons")
        if not self.complete and not reasons:
            raise ValueError("incomplete data-flow analyses require an explicit reason")
        object.__setattr__(self, "incomplete_reasons", reasons)


@dataclass(frozen=True, slots=True)
class MemoryLocation:
    """A stable local storage/access-path identity within one analysis."""

    id: str
    analysis_id: str
    graph_id: str
    kind: MemoryLocationKind
    name: str
    type_name: str = ""
    declaration_symbol_id: str | None = None
    base_location_id: str | None = None
    access_path: tuple[str, ...] = ()
    is_volatile: bool = False
    is_atomic: bool = False
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.analysis_id.strip() or not self.graph_id.strip():
            raise ValueError("memory location identifiers must not be empty")
        if not self.name.strip():
            raise ValueError("memory locations require a display name")
        if (
            self.kind in {MemoryLocationKind.FIELD, MemoryLocationKind.DEREFERENCE}
            and not self.base_location_id
        ):
            raise ValueError("field and dereference locations require a base location")


@dataclass(frozen=True, slots=True)
class DataAccess:
    """One compiler-observed definition, use, or conservative clobber."""

    id: str
    analysis_id: str
    graph_id: str
    block_id: str
    location_id: str
    kind: DataAccessKind
    sequence: int
    cfg_element_id: str | None = None
    span: SourceSpan | None = None
    expression: str = ""
    pointee_symbol_ids: tuple[str, ...] = ()
    points_to_complete: bool = True
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        required = (self.id, self.analysis_id, self.graph_id, self.block_id, self.location_id)
        if not all(value.strip() for value in required):
            raise ValueError("data access identifiers must not be empty")
        if self.sequence < 0:
            raise ValueError("data access sequence must be non-negative")
        pointees = tuple(
            dict.fromkeys(item.strip() for item in self.pointee_symbol_ids if item.strip())
        )
        object.__setattr__(self, "pointee_symbol_ids", pointees)


@dataclass(frozen=True, slots=True)
class DataFlowEvidence:
    """Evidence connecting accesses or aliasing storage locations."""

    id: str
    analysis_id: str
    graph_id: str
    relation: DataFlowRelation
    certainty: DataFlowCertainty
    reason: str
    source_access_id: str | None = None
    target_access_id: str | None = None
    source_location_id: str | None = None
    target_location_id: str | None = None
    evidence_span: SourceSpan | None = None
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.analysis_id.strip() or not self.graph_id.strip():
            raise ValueError("data-flow evidence identifiers must not be empty")
        if not self.reason.strip():
            raise ValueError("data-flow evidence requires a reason")
        access_pair = self.source_access_id is not None and self.target_access_id is not None
        location_pair = self.source_location_id is not None and self.target_location_id is not None
        if access_pair == location_pair:
            raise ValueError("data-flow evidence requires exactly one access or location pair")


@dataclass(frozen=True, slots=True)
class FunctionSummary:
    """Bounded local and transitive effects for one concrete function body variant."""

    id: str
    function_symbol_id: str
    graph_id: str
    analysis_id: str
    parameter_modes: tuple[str, ...]
    parameter_location_ids: tuple[str, ...]
    local_complete: bool
    local_incomplete_reasons: tuple[str, ...]
    complete: bool
    incomplete_reasons: tuple[str, ...]
    recursive: bool
    iteration_count: int
    max_scc_iterations: int
    max_scc_size: int
    max_summary_effects: int
    solution_hash: str = ""
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not all((self.id.strip(), self.function_symbol_id.strip(), self.graph_id.strip())):
            raise ValueError("function summary identifiers must not be empty")
        if (
            self.iteration_count < 0
            or min(self.max_scc_iterations, self.max_scc_size, self.max_summary_effects) <= 0
        ):
            raise ValueError("function summary limits must be positive")
        local_reasons = tuple(dict.fromkeys(x for x in self.local_incomplete_reasons if x))
        reasons = tuple(dict.fromkeys(x for x in self.incomplete_reasons if x))
        if self.local_complete == bool(local_reasons):
            raise ValueError("local summary completeness and reasons disagree")
        if self.complete == bool(reasons):
            raise ValueError("summary completeness and reasons disagree")
        object.__setattr__(self, "local_incomplete_reasons", local_reasons)
        object.__setattr__(self, "incomplete_reasons", reasons)


@dataclass(frozen=True, slots=True)
class SummaryEffect:
    """One read, write, or escape retained as evidence rather than a verdict."""

    id: str
    summary_id: str
    kind: SummaryEffectKind
    location_kind: MemoryLocationKind
    certainty: DataFlowCertainty
    reason: str
    parameter_index: int | None = None
    access_path: tuple[str, ...] = ()
    location_id: str | None = None
    source_access_id: str | None = None
    is_local: bool = True
    via_callsite_id: str | None = None
    target_symbol_id: str | None = None
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.summary_id.strip() or not self.reason.strip():
            raise ValueError("summary effects require identifiers and a reason")
        if self.parameter_index is not None and self.parameter_index < 0:
            raise ValueError("summary parameter indexes must be non-negative")
        if not self.is_local and (not self.via_callsite_id or not self.target_symbol_id):
            raise ValueError("propagated summary effects require call-target provenance")


@dataclass(frozen=True, slots=True)
class SummaryReturnOrigin:
    id: str
    summary_id: str
    kind: SummaryReturnOriginKind
    certainty: DataFlowCertainty
    reason: str
    location_kind: MemoryLocationKind | None = None
    parameter_index: int | None = None
    access_path: tuple[str, ...] = ()
    location_id: str | None = None
    callsite_id: str | None = None
    is_local: bool = True
    via_callsite_id: str | None = None
    target_symbol_id: str | None = None
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.summary_id.strip() or not self.reason.strip():
            raise ValueError("summary return origins require identifiers and a reason")
        if self.parameter_index is not None and self.parameter_index < 0:
            raise ValueError("summary parameter indexes must be non-negative")
        if self.kind == SummaryReturnOriginKind.CALL_RESULT and not self.callsite_id:
            raise ValueError("call-result return origins require a callsite")
        if not self.is_local and (not self.via_callsite_id or not self.target_symbol_id):
            raise ValueError("propagated return origins require call-target provenance")


@dataclass(frozen=True, slots=True)
class CallArgumentBinding:
    id: str
    caller_summary_id: str
    callsite_id: str
    argument_index: int
    location_id: str | None
    location_kind: MemoryLocationKind
    parameter_index: int | None
    access_path: tuple[str, ...]
    writeback_candidate: bool
    complete: bool
    incomplete_reason: str = ""
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not all((self.id.strip(), self.caller_summary_id.strip(), self.callsite_id.strip())):
            raise ValueError("call argument binding identifiers must not be empty")
        if self.argument_index < 0:
            raise ValueError("call argument indexes must be non-negative")
        if self.parameter_index is not None and self.parameter_index < 0:
            raise ValueError("caller parameter indexes must be non-negative")
        if self.complete == bool(self.incomplete_reason):
            raise ValueError("call argument binding completeness and reason disagree")


@dataclass(frozen=True, slots=True)
class CallResultBinding:
    id: str
    caller_summary_id: str
    callsite_id: str
    location_id: str
    definition_access_id: str
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        if not all(
            (
                self.id.strip(),
                self.caller_summary_id.strip(),
                self.callsite_id.strip(),
                self.location_id.strip(),
                self.definition_access_id.strip(),
            )
        ):
            raise ValueError("call result binding identifiers must not be empty")


@dataclass(frozen=True, slots=True)
class InterproceduralFlow:
    """One cross-call evidence edge with concrete target and compiler provenance."""

    id: str
    kind: InterproceduralFlowKind
    caller_summary_id: str
    callee_summary_id: str
    callsite_id: str
    target_symbol_id: str
    target_certainty: CallTargetCertainty
    certainty: DataFlowCertainty
    reason: str
    argument_index: int | None = None
    caller_location_id: str | None = None
    callee_location_id: str | None = None
    caller_access_id: str | None = None
    translation_unit_id: str = ""
    build_configuration_id: str = ""
    build_variant: str = DEFAULT_BUILD_VARIANT

    def __post_init__(self) -> None:
        required = (
            self.id,
            self.caller_summary_id,
            self.callee_summary_id,
            self.callsite_id,
            self.target_symbol_id,
            self.reason,
            self.translation_unit_id,
            self.build_configuration_id,
            self.build_variant,
        )
        if not all(value.strip() for value in required):
            raise ValueError("interprocedural flows require identifiers and provenance")


CfgFact = TypeVar("CfgFact")


@dataclass(frozen=True, slots=True)
class BoundedCfgResult(Generic[CfgFact]):
    """A deterministic page whose truncation is explicit to callers."""

    items: tuple[CfgFact, ...]
    truncated: bool


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
