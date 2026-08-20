"""Compiler-aware C++ ingestion backed by libclang."""

from __future__ import annotations

import ctypes
import hashlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    libclang_arguments,
    translation_unit_id,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    CodeSymbol,
    GraphEdge,
    GraphRelation,
    OccurrenceKind,
    SourceSpan,
    SymbolKind,
    SymbolOccurrence,
    TranslationUnit,
)


class ClangUnavailableError(RuntimeError):
    """Raised when the optional bindings or native libclang cannot be loaded."""


class TranslationUnitError(RuntimeError):
    """Raised when libclang reports compiler errors for a translation unit."""

    def __init__(self, source_path: Path, command: tuple[str, ...], diagnostics: tuple[str, ...]):
        rendered = "\n".join(f"  {diagnostic}" for diagnostic in diagnostics)
        super().__init__(
            f"cannot index translation unit {source_path}\n"
            f"compiler arguments: {' '.join(command)}\n"
            f"diagnostics:\n{rendered}"
        )
        self.source_path = source_path
        self.command = command
        self.diagnostics = diagnostics


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_text(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _discover_libclang() -> Path | None:
    explicit_file = os.environ.get("LIBCLANG_LIBRARY_FILE")
    if explicit_file:
        return Path(explicit_file)
    explicit_path = os.environ.get("LIBCLANG_LIBRARY_PATH")
    if explicit_path:
        directory = Path(explicit_path)
        for name in ("libclang.so", "libclang.dylib", "libclang.dll"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    candidates: list[Path] = []
    for root in (Path("/usr/lib"), Path("/usr/local/lib"), Path("/opt/homebrew/opt/llvm/lib")):
        candidates.extend(root.glob("llvm-*/lib/libclang.so"))
        candidates.extend(root.glob("libclang.so"))
        candidates.extend(root.glob("libclang.dylib"))
    return sorted(candidates, reverse=True)[0] if candidates else None


def _load_cindex(library_file: Path | None) -> Any:
    try:
        from clang import cindex
    except ImportError as error:
        raise ClangUnavailableError(
            "clang Python bindings are not installed; install cpp-context-engine[clang]"
        ) from error

    if not cindex.Config.loaded:
        selected = library_file or _discover_libclang()
        if selected is not None:
            cindex.Config.set_library_file(str(selected))
    try:
        cindex.Config().get_cindex_library()
    except Exception as error:
        raise ClangUnavailableError(
            f"could not load libclang; set LIBCLANG_LIBRARY_FILE to a compatible library: {error}"
        ) from error
    return cindex


class _TranslationUnitCollector:
    def __init__(
        self,
        cindex: Any,
        project_root: Path,
        configuration: BuildConfiguration,
        translation_unit_id: str,
    ) -> None:
        self.cindex = cindex
        self.project_root = project_root
        self.configuration = configuration
        self.translation_unit_id = translation_unit_id
        self.symbols: dict[str, CodeSymbol] = {}
        self.occurrences: dict[str, SymbolOccurrence] = {}
        self.edges: set[GraphEdge] = set()
        self.dependencies: set[Path] = {configuration.source_path}
        self._contents: dict[Path, bytes] = {}

    def collect(self, cursor: Any) -> None:
        self._visit(cursor, None)

    def _visit(self, cursor: Any, enclosing_symbol_id: str | None) -> None:
        if cursor.kind == self.cindex.CursorKind.TRANSLATION_UNIT:
            for child in cursor.get_children():
                self._visit(child, None)
            return

        path = self._cursor_path(cursor)
        if path is None or not _is_within(path, self.project_root):
            return
        self.dependencies.add(path)

        current_owner = enclosing_symbol_id
        symbol_kind = self._symbol_kind(cursor)
        if symbol_kind is not None:
            symbol = self._make_symbol(cursor, symbol_kind, path)
            self._put_symbol(symbol)
            occurrence_kind = (
                OccurrenceKind.DEFINITION
                if self._is_definition(cursor)
                else OccurrenceKind.DECLARATION
            )
            self._put_occurrence(cursor, symbol.id, occurrence_kind, enclosing_symbol_id)
            container_id = enclosing_symbol_id or self._file_symbol(path).id
            if container_id != symbol.id:
                self._put_edge(container_id, symbol.id, GraphRelation.CONTAINS)
            if symbol_kind in {
                SymbolKind.FUNCTION,
                SymbolKind.METHOD,
                SymbolKind.CLASS,
                SymbolKind.STRUCT,
                SymbolKind.ENUM,
                SymbolKind.NAMESPACE,
            }:
                current_owner = symbol.id
            if symbol_kind == SymbolKind.METHOD:
                self._collect_overrides(cursor, symbol.id)

        self._collect_relationship(cursor, current_owner, path)
        for child in cursor.get_children():
            self._visit(child, current_owner)

    def _collect_relationship(
        self, cursor: Any, enclosing_symbol_id: str | None, path: Path
    ) -> None:
        kind = cursor.kind
        if kind == self.cindex.CursorKind.INCLUSION_DIRECTIVE:
            included = cursor.get_included_file()
            if included is None:
                return
            included_path = Path(included.name).resolve(strict=False)
            if _is_within(included_path, self.project_root):
                self.dependencies.add(included_path)
                self._put_edge(
                    self._file_symbol(path).id,
                    self._file_symbol(included_path).id,
                    GraphRelation.INCLUDES,
                )
            return
        if enclosing_symbol_id is None:
            return

        relation: GraphRelation | None = None
        occurrence_kind = OccurrenceKind.REFERENCE
        if kind == self.cindex.CursorKind.CALL_EXPR:
            relation = GraphRelation.CALLS
            occurrence_kind = OccurrenceKind.CALL
        elif kind == self.cindex.CursorKind.CXX_BASE_SPECIFIER:
            relation = GraphRelation.INHERITS
            occurrence_kind = OccurrenceKind.TYPE
        elif kind in {self.cindex.CursorKind.TYPE_REF, self.cindex.CursorKind.TEMPLATE_REF}:
            relation = GraphRelation.USES_TYPE
            occurrence_kind = OccurrenceKind.TYPE
        elif kind == self.cindex.CursorKind.MACRO_INSTANTIATION:
            relation = GraphRelation.REFERENCES
            occurrence_kind = OccurrenceKind.MACRO_EXPANSION
        elif kind in {
            self.cindex.CursorKind.DECL_REF_EXPR,
            self.cindex.CursorKind.MEMBER_REF_EXPR,
            self.cindex.CursorKind.NAMESPACE_REF,
            self.cindex.CursorKind.OVERLOADED_DECL_REF,
        }:
            relation = GraphRelation.REFERENCES
        if relation is None:
            return

        referenced = cursor.referenced
        if referenced is None:
            return
        referenced_path = self._cursor_path(referenced)
        if referenced_path is None or not _is_within(referenced_path, self.project_root):
            return
        target_kind = self._symbol_kind(referenced)
        if target_kind is None:
            return
        target = self._make_symbol(referenced, target_kind, referenced_path)
        self._put_symbol(target)
        self._put_edge(enclosing_symbol_id, target.id, relation)
        self._put_occurrence(cursor, target.id, occurrence_kind, enclosing_symbol_id)
        if relation != GraphRelation.REFERENCES:
            self._put_edge(enclosing_symbol_id, target.id, GraphRelation.REFERENCES)

    def _collect_overrides(self, cursor: Any, source_id: str) -> None:
        """Use libclang's native override API, absent from the Python wrapper."""

        library = self.cindex.conf.lib
        try:
            get_overridden = library.clang_getOverriddenCursors
            dispose = library.clang_disposeOverriddenCursors
        except AttributeError:
            return
        cursor_pointer = ctypes.POINTER(self.cindex.Cursor)()
        count = ctypes.c_uint()
        get_overridden.argtypes = [
            self.cindex.Cursor,
            ctypes.POINTER(ctypes.POINTER(self.cindex.Cursor)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        get_overridden.restype = None
        dispose.argtypes = [ctypes.POINTER(self.cindex.Cursor)]
        dispose.restype = None
        get_overridden(cursor, ctypes.byref(cursor_pointer), ctypes.byref(count))
        try:
            for index in range(count.value):
                overridden = cursor_pointer[index]
                overridden._tu = cursor._tu
                path = self._cursor_path(overridden)
                kind = self._symbol_kind(overridden)
                if path is None or kind is None or not _is_within(path, self.project_root):
                    continue
                target = self._make_symbol(overridden, kind, path)
                self._put_symbol(target)
                self._put_edge(source_id, target.id, GraphRelation.OVERRIDES)
        finally:
            if cursor_pointer:
                dispose(cursor_pointer)

    def _put_symbol(self, symbol: CodeSymbol) -> None:
        previous = self.symbols.get(symbol.id)
        if previous is None or (
            bool(symbol.metadata.get("is_definition"))
            and not bool(previous.metadata.get("is_definition"))
        ):
            self.symbols[symbol.id] = symbol

    def _put_edge(self, source_id: str, target_id: str, relation: GraphRelation) -> None:
        self.edges.add(GraphEdge(source_id, target_id, relation, self.translation_unit_id))

    def _put_occurrence(
        self,
        cursor: Any,
        symbol_id: str,
        kind: OccurrenceKind,
        enclosing_symbol_id: str | None,
    ) -> None:
        span = self._span(cursor)
        if span is None:
            return
        occurrence_id = (
            "occ_"
            + _hash_text(
                symbol_id,
                kind.value,
                str(span.path),
                str(span.start_line),
                str(span.start_column),
                str(span.end_line),
                str(span.end_column),
            )[:32]
        )
        self.occurrences[occurrence_id] = SymbolOccurrence(
            id=occurrence_id,
            symbol_id=symbol_id,
            span=span,
            kind=kind,
            enclosing_symbol_id=enclosing_symbol_id,
            translation_unit_id=self.translation_unit_id,
        )

    def _file_symbol(self, path: Path) -> CodeSymbol:
        relative = path.relative_to(self.project_root).as_posix()
        symbol_id = "file_" + _hash_text(relative)[:32]
        existing = self.symbols.get(symbol_id)
        if existing is not None:
            return existing
        content = self._read(path)
        lines = content.decode("utf-8", errors="replace").splitlines()
        end_line = max(1, len(lines))
        end_column = len(lines[-1]) + 1 if lines else 1
        symbol = CodeSymbol(
            id=symbol_id,
            qualified_name=relative,
            kind=SymbolKind.FILE,
            span=SourceSpan(path, 1, end_line, 1, end_column),
            signature=relative,
            source_hash=_hash_bytes(content),
            build_configuration_id=self.configuration.id,
            translation_unit_id=self.translation_unit_id,
            metadata={"relative_path": relative},
        )
        self.symbols[symbol_id] = symbol
        return symbol

    def _make_symbol(self, cursor: Any, kind: SymbolKind, path: Path) -> CodeSymbol:
        qualified_name = self._qualified_name(cursor, kind)
        usr = cursor.get_usr() or ""
        if usr:
            identity = f"usr:{usr}"
        else:
            relative = path.relative_to(self.project_root).as_posix()
            identity = f"fallback:{kind.value}:{relative}:{qualified_name}"
        symbol_id = "sym_" + _hash_text(identity)[:32]
        span = self._span(cursor) or SourceSpan(path, 1, 1)
        source_text = self._source(cursor, path)
        documentation = cursor.raw_comment or cursor.brief_comment or ""
        return CodeSymbol(
            id=symbol_id,
            qualified_name=qualified_name,
            kind=kind,
            span=span,
            signature=self._signature(cursor, qualified_name),
            documentation=documentation,
            source_hash=_hash_text(source_text),
            source_text=source_text,
            build_configuration_id=self.configuration.id,
            translation_unit_id=self.translation_unit_id,
            metadata={
                "usr": usr,
                "display_name": cursor.displayname or cursor.spelling,
                "is_definition": self._is_definition(cursor),
                "start_offset": cursor.extent.start.offset,
                "end_offset_exclusive": cursor.extent.end.offset,
            },
        )

    def _qualified_name(self, cursor: Any, kind: SymbolKind) -> str:
        names: list[str] = []
        current = cursor
        while current is not None and current.kind != self.cindex.CursorKind.TRANSLATION_UNIT:
            name = current.spelling
            if not name:
                location = current.location
                name = f"<anonymous-{kind.value}@{location.line}:{location.column}>"
            names.append(name)
            current = current.semantic_parent
        return "::".join(reversed(names))

    @staticmethod
    def _signature(cursor: Any, qualified_name: str) -> str:
        type_spelling = cursor.type.spelling if cursor.type is not None else ""
        return f"{qualified_name}: {type_spelling}" if type_spelling else qualified_name

    def _source(self, cursor: Any, path: Path) -> str:
        content = self._read(path)
        start = max(0, cursor.extent.start.offset)
        end = max(start, cursor.extent.end.offset)
        return content[start:end].decode("utf-8", errors="replace")

    def _read(self, path: Path) -> bytes:
        content = self._contents.get(path)
        if content is None:
            content = path.read_bytes()
            self._contents[path] = content
        return content

    @staticmethod
    def _is_definition(cursor: Any) -> bool:
        try:
            return bool(cursor.is_definition())
        except Exception:
            return False

    @staticmethod
    def _cursor_path(cursor: Any) -> Path | None:
        location = cursor.location
        if location is None or location.file is None:
            return None
        return Path(location.file.name).resolve(strict=False)

    @staticmethod
    def _span(cursor: Any) -> SourceSpan | None:
        extent = cursor.extent
        if extent.start.file is None or extent.end.file is None:
            return None
        path = Path(extent.start.file.name).resolve(strict=False)
        return SourceSpan(
            path=path,
            start_line=max(1, extent.start.line),
            end_line=max(1, extent.end.line),
            start_column=max(1, extent.start.column),
            end_column=max(1, extent.end.column),
        )

    def _symbol_kind(self, cursor: Any) -> SymbolKind | None:
        kinds = self.cindex.CursorKind
        mapping = {
            kinds.FUNCTION_DECL: SymbolKind.FUNCTION,
            kinds.FUNCTION_TEMPLATE: SymbolKind.FUNCTION,
            kinds.CXX_METHOD: SymbolKind.METHOD,
            kinds.CONSTRUCTOR: SymbolKind.METHOD,
            kinds.DESTRUCTOR: SymbolKind.METHOD,
            kinds.CONVERSION_FUNCTION: SymbolKind.METHOD,
            kinds.CLASS_DECL: SymbolKind.CLASS,
            kinds.CLASS_TEMPLATE: SymbolKind.CLASS,
            kinds.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION: SymbolKind.CLASS,
            kinds.STRUCT_DECL: SymbolKind.STRUCT,
            kinds.ENUM_DECL: SymbolKind.ENUM,
            kinds.NAMESPACE: SymbolKind.NAMESPACE,
            kinds.VAR_DECL: SymbolKind.VARIABLE,
            kinds.FIELD_DECL: SymbolKind.VARIABLE,
            kinds.ENUM_CONSTANT_DECL: SymbolKind.VARIABLE,
            kinds.PARM_DECL: SymbolKind.VARIABLE,
            kinds.TYPEDEF_DECL: SymbolKind.TYPE_ALIAS,
            kinds.TYPE_ALIAS_DECL: SymbolKind.TYPE_ALIAS,
            kinds.MACRO_DEFINITION: SymbolKind.MACRO,
        }
        return mapping.get(cursor.kind)


class ClangIngestor:
    """Parse configured translation units into normalized compiler facts."""

    def __init__(self, *, library_file: Path | None = None, fail_on_error: bool = True) -> None:
        self._cindex = _load_cindex(library_file)
        self._fail_on_error = fail_on_error

    def ingest(self, project_root: Path, compilation_database: Path) -> IngestionBatch:
        database = CompilationDatabase.load(compilation_database)
        return self.ingest_configurations(project_root, database.configurations)

    def ingest_configurations(
        self, project_root: Path, configurations: Iterable[BuildConfiguration]
    ) -> IngestionBatch:
        project_root = project_root.resolve(strict=False)
        build_configurations: list[BuildConfiguration] = []
        translation_units: list[TranslationUnit] = []
        symbols: list[CodeSymbol] = []
        occurrences: list[SymbolOccurrence] = []
        edges: list[GraphEdge] = []
        index = self._cindex.Index.create()

        for configuration in configurations:
            build_configurations.append(configuration)
            parser_arguments = libclang_arguments(configuration)
            try:
                parsed = index.parse(
                    str(configuration.source_path),
                    args=list(parser_arguments),
                    options=self._cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
                )
            except self._cindex.TranslationUnitLoadError as error:
                raise TranslationUnitError(
                    configuration.source_path,
                    parser_arguments,
                    (f"libclang could not load the translation unit: {error}",),
                ) from error

            diagnostics = tuple(
                self._format_diagnostic(diagnostic) for diagnostic in parsed.diagnostics
            )
            errors = tuple(
                rendered
                for diagnostic, rendered in zip(parsed.diagnostics, diagnostics, strict=True)
                if diagnostic.severity >= self._cindex.Diagnostic.Error
            )
            if errors and self._fail_on_error:
                raise TranslationUnitError(configuration.source_path, parser_arguments, errors)

            unit_id = translation_unit_id(configuration)
            collector = _TranslationUnitCollector(
                self._cindex, project_root, configuration, unit_id
            )
            collector.collect(parsed.cursor)
            dependencies = tuple(
                (path, _hash_bytes(path.read_bytes())) for path in sorted(collector.dependencies)
            )
            translation_units.append(
                TranslationUnit(
                    id=unit_id,
                    build_configuration_id=configuration.id,
                    source_path=configuration.source_path,
                    content_hash=_hash_bytes(configuration.source_path.read_bytes()),
                    dependencies=dependencies,
                    diagnostics=diagnostics,
                )
            )
            symbols.extend(collector.symbols.values())
            occurrences.extend(collector.occurrences.values())
            edges.extend(collector.edges)

        return IngestionBatch(
            build_configurations=tuple(build_configurations),
            translation_units=tuple(translation_units),
            symbols=tuple(symbols),
            occurrences=tuple(occurrences),
            edges=tuple(edges),
        )

    @staticmethod
    def _format_diagnostic(diagnostic: Any) -> str:
        severity_names = {0: "ignored", 1: "note", 2: "warning", 3: "error", 4: "fatal"}
        location = diagnostic.location
        if location.file is None:
            rendered_location = "<unknown>"
        else:
            rendered_location = f"{location.file.name}:{location.line}:{location.column}"
        option = f" [{diagnostic.option}]" if diagnostic.option else ""
        return (
            f"{rendered_location}: {severity_names.get(diagnostic.severity, 'diagnostic')}: "
            f"{diagnostic.spelling}{option}"
        )
