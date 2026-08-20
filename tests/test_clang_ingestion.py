from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cpp_context_engine.ingestion.clang import (
    ClangIngestor,
    ClangUnavailableError,
    TranslationUnitError,
)
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.models import GraphRelation, SearchQuery, SymbolKind
from cpp_context_engine.storage.sqlite import SQLiteStore

FIXTURE = Path(__file__).parent / "fixtures" / "cpp_project"


def _ingestor() -> ClangIngestor:
    try:
        return ClangIngestor()
    except ClangUnavailableError as error:
        pytest.skip(str(error))


@pytest.mark.clang
def test_extracts_semantic_chunks_occurrences_and_relationships() -> None:
    batch = _ingestor().ingest(FIXTURE, FIXTURE / "compile_commands.json")
    symbols = {(symbol.qualified_name, symbol.kind) for symbol in batch.symbols}
    relations = {edge.relation for edge in batch.edges}

    assert ("demo::Derived::compute", SymbolKind.METHOD) in symbols
    assert ("demo::Kind", SymbolKind.ENUM) in symbols
    assert ("SCALE_VALUE", SymbolKind.MACRO) in symbols
    assert {
        GraphRelation.CONTAINS,
        GraphRelation.REFERENCES,
        GraphRelation.CALLS,
        GraphRelation.INHERITS,
        GraphRelation.OVERRIDES,
        GraphRelation.USES_TYPE,
        GraphRelation.INCLUDES,
    } <= relations
    compute = next(
        symbol
        for symbol in batch.symbols
        if symbol.qualified_name == "demo::Derived::compute" and symbol.metadata["is_definition"]
    )
    assert compute.source_text.startswith("int Derived::compute")
    assert compute.span.start_line == 7
    assert compute.source_hash
    assert any(occurrence.symbol_id == compute.id for occurrence in batch.occurrences)
    assert all(unit.dependencies for unit in batch.translation_units)


@pytest.mark.clang
def test_compiler_errors_include_source_command_and_diagnostic(tmp_path: Path) -> None:
    source = tmp_path / "broken.cpp"
    source.write_text("int broken( {\n", encoding="utf-8")
    database = tmp_path / "compile_commands.json"
    database.write_text(
        json.dumps(
            [
                {
                    "directory": ".",
                    "file": "broken.cpp",
                    "arguments": ["clang++", "-std=c++20", "-c", "broken.cpp"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(TranslationUnitError) as captured:
        _ingestor().ingest(tmp_path, database)

    message = str(captured.value)
    assert str(source) in message
    assert "-std=c++20" in message
    assert "broken.cpp:1:" in message
    assert "error" in message or "fatal" in message


@pytest.mark.clang
def test_incremental_reindex_skips_unchanged_and_tracks_header_dependencies(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)

    with SQLiteStore(tmp_path / "index.db", project_root=project) as store:
        indexer = ProjectIndexer(_ingestor(), store)
        first = indexer.index(project, project / "compile_commands.json")
        second = indexer.index(project, project / "compile_commands.json")
        header = project / "include" / "model.hpp"
        header.write_text(header.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        third = indexer.index(project, project / "compile_commands.json")

        assert first.indexed_translation_units == 2
        assert second.indexed_translation_units == 0
        assert second.skipped_translation_units == 2
        assert third.indexed_translation_units == 2
        hit = store.search(SearchQuery("Derived compute"))[0]
        assert hit.symbol.qualified_name.startswith("demo")


@pytest.mark.clang
def test_incremental_reindex_removes_commands_no_longer_present(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    database_path = project / "compile_commands.json"

    with SQLiteStore(tmp_path / "index.db", project_root=project) as store:
        indexer = ProjectIndexer(_ingestor(), store)
        indexer.index(project, database_path)
        entries = json.loads(database_path.read_text(encoding="utf-8"))
        database_path.write_text(json.dumps(entries[1:]), encoding="utf-8")
        result = indexer.index(project, database_path)

        assert result.removed_translation_units == 1
        assert len(store.translation_unit_states()) == 1
        helper = next(
            symbol for symbol in store.symbols() if symbol.qualified_name == "demo::helper"
        )
        assert helper.span.path == project / "include" / "model.hpp"
        assert helper.metadata["is_definition"] is False
        assert any(symbol.qualified_name == "run" for symbol in store.symbols())
