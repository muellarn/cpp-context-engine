from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from cpp_context_engine.ingestion.clang import ClangIngestor, ClangUnavailableError
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.models import BuildScope, BuildVariant, GraphRelation, SearchQuery
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION, SQLiteStore

FIXTURE = Path(__file__).parent / "fixtures" / "multi_build_project"


def _ingestor() -> ClangIngestor:
    try:
        return ClangIngestor()
    except ClangUnavailableError as error:
        pytest.skip(str(error))


def _variant(project: Path, name: str) -> BuildVariant:
    return BuildVariant(
        name,
        project / f"build-{name}" / "compile_commands.json",
        target="features",
        platform="test-host",
        metadata={"define": "FEATURE_ALPHA" if name == "alpha" else "!FEATURE_ALPHA"},
    )


def _snapshot(store: SQLiteStore, name: str) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in store._connection.execute(  # noqa: SLF001 - byte-stability assertion
            """
            SELECT id, symbol_id, build_configuration_id, translation_unit_id,
                   is_definition, snapshot_json
            FROM symbol_variants WHERE build_variant = ? ORDER BY id
            """,
            (name,),
        )
    )


@pytest.mark.clang
def test_two_opposing_builds_coexist_filter_union_reindex_and_remove(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    alpha = _variant(project, "alpha")
    beta = _variant(project, "beta")

    with SQLiteStore(tmp_path / "index.db", project_root=project) as store:
        indexer = ProjectIndexer(_ingestor(), store)
        assert (
            indexer.index(
                project, alpha.compilation_database, build_variant=alpha
            ).indexed_translation_units
            == 1
        )
        assert (
            indexer.index(
                project, beta.compilation_database, build_variant=beta
            ).indexed_translation_units
            == 1
        )

        assert store.search(SearchQuery("alpha_only"), build_scope=BuildScope.single("alpha"))
        assert not store.search(SearchQuery("alpha_only"), build_scope=BuildScope.single("beta"))
        assert store.search(SearchQuery("beta_only"), build_scope=BuildScope.single("beta"))
        union = store.search(
            SearchQuery("selected_feature"), build_scope=BuildScope(("alpha", "beta"))
        )
        assert {hit.symbol.build_variant for hit in union} == {"alpha", "beta"}
        assert all(hit.symbol.variant_id for hit in union)

        repeated = next(
            hit.symbol
            for hit in store.search(
                SearchQuery("repeated_calls"), build_scope=BuildScope.single("alpha")
            )
            if hit.symbol.qualified_name == "repeated_calls"
        )
        calls = store.neighbors(
            repeated.id,
            relations=frozenset({GraphRelation.CALLS}),
            build_scope=BuildScope.single("alpha"),
        )
        assert len(calls) == 2
        assert len({edge.id for edge in calls}) == 2
        assert {edge.build_variant for edge in calls} == {"alpha"}

        beta_before = _snapshot(store, "beta")
        source = project / "src" / "features.cpp"
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        assert (
            indexer.index(
                project, alpha.compilation_database, build_variant=alpha
            ).indexed_translation_units
            == 1
        )
        assert _snapshot(store, "beta") == beta_before

        variants = {variant.name: variant for variant in store.build_variants()}
        assert variants["alpha"].target == "features"
        assert variants["beta"].metadata["define"] == "!FEATURE_ALPHA"
        assert store.remove_build_variant("alpha")
        assert not store.search(SearchQuery("alpha_only"), build_scope=BuildScope.single("alpha"))
        assert store.search(SearchQuery("beta_only"), build_scope=BuildScope.single("beta"))


@pytest.mark.clang
def test_variant_and_fact_ids_are_deterministic(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    alpha = _variant(project, "alpha")

    snapshots: list[tuple[tuple[object, ...], ...]] = []
    for name in ("first.db", "second.db"):
        with SQLiteStore(tmp_path / name, project_root=project) as store:
            ProjectIndexer(_ingestor(), store).index(
                project, alpha.compilation_database, build_variant=alpha
            )
            snapshots.append(
                tuple(
                    tuple(row)
                    for row in store._connection.execute(  # noqa: SLF001
                        """
                        SELECT id, symbol_id, translation_unit_id FROM symbol_variants ORDER BY id
                        """
                    )
                )
                + tuple(
                    tuple(row)
                    for row in store._connection.execute(  # noqa: SLF001
                        """
                        SELECT id, source_id, target_id, translation_unit_id
                        FROM edges ORDER BY id
                        """
                    )
                )
            )
    assert snapshots[0] == snapshots[1]


def test_v3_migration_rolls_back_all_schema_changes_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE projects(id INTEGER PRIMARY KEY, root TEXT NOT NULL UNIQUE);
        CREATE TABLE build_configurations(
          project_id INTEGER NOT NULL, id TEXT NOT NULL, source_path TEXT NOT NULL,
          directory TEXT NOT NULL, arguments_json TEXT NOT NULL, command_hash TEXT NOT NULL,
          output TEXT, PRIMARY KEY(project_id,id));
        CREATE TABLE translation_units(
          project_id INTEGER NOT NULL, id TEXT NOT NULL, build_configuration_id TEXT NOT NULL,
          source_path TEXT NOT NULL, content_hash TEXT NOT NULL, diagnostics_json TEXT NOT NULL,
          indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(project_id,id));
        CREATE TABLE symbols(
          project_id INTEGER NOT NULL, id TEXT NOT NULL, qualified_name TEXT NOT NULL,
          kind TEXT NOT NULL, path TEXT NOT NULL, start_line INTEGER NOT NULL,
          end_line INTEGER NOT NULL, start_column INTEGER NOT NULL, end_column INTEGER NOT NULL,
          signature TEXT NOT NULL, documentation TEXT NOT NULL, source_hash TEXT NOT NULL,
          source_text TEXT NOT NULL, build_configuration_id TEXT NOT NULL,
          metadata_json TEXT NOT NULL, PRIMARY KEY(project_id,id));
        CREATE TABLE translation_unit_symbols(
          project_id INTEGER, translation_unit_id TEXT, symbol_id TEXT, is_definition INTEGER,
          snapshot_json TEXT, PRIMARY KEY(project_id,translation_unit_id,symbol_id));
        CREATE TABLE occurrences(
          project_id INTEGER, translation_unit_id TEXT, id TEXT, symbol_id TEXT,
          enclosing_symbol_id TEXT, kind TEXT, path TEXT, start_line INTEGER, end_line INTEGER,
          start_column INTEGER, end_column INTEGER,
          PRIMARY KEY(project_id,translation_unit_id,id));
        CREATE TABLE edges(
          project_id INTEGER, translation_unit_id TEXT, source_id TEXT, target_id TEXT,
          relation TEXT, PRIMARY KEY(project_id,translation_unit_id,source_id,target_id,relation));
        PRAGMA user_version=2;
        """
    )
    connection.close()

    def fail_rebuild(_store: SQLiteStore) -> None:
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(SQLiteStore, "_rebuild_variant_fts", fail_rebuild)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        SQLiteStore(database)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'build_variants'"
            ).fetchone()
            is None
        )
        assert "id" not in {
            row[1] for row in connection.execute("PRAGMA table_info(edges)").fetchall()
        }
    finally:
        connection.close()


def test_v2_migration_preserves_baseline_search_and_requests_reindex(tmp_path: Path) -> None:
    root = (tmp_path / "project").resolve()
    root.mkdir()
    source = root / "legacy.cpp"
    source.write_text("int legacy() { return 1; }\n", encoding="utf-8")
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE projects(id INTEGER PRIMARY KEY, root TEXT NOT NULL UNIQUE);
        CREATE TABLE build_configurations(
          project_id INTEGER NOT NULL, id TEXT NOT NULL, source_path TEXT NOT NULL,
          directory TEXT NOT NULL, arguments_json TEXT NOT NULL, command_hash TEXT NOT NULL,
          output TEXT, PRIMARY KEY(project_id,id));
        CREATE TABLE translation_units(
          project_id INTEGER NOT NULL, id TEXT NOT NULL, build_configuration_id TEXT NOT NULL,
          source_path TEXT NOT NULL, content_hash TEXT NOT NULL, diagnostics_json TEXT NOT NULL,
          indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(project_id,id));
        CREATE TABLE dependencies(project_id INTEGER, translation_unit_id TEXT, path TEXT,
          content_hash TEXT, PRIMARY KEY(project_id,translation_unit_id,path));
        CREATE TABLE symbols(
          project_id INTEGER NOT NULL, id TEXT NOT NULL, qualified_name TEXT NOT NULL,
          kind TEXT NOT NULL, path TEXT NOT NULL, start_line INTEGER NOT NULL,
          end_line INTEGER NOT NULL, start_column INTEGER NOT NULL, end_column INTEGER NOT NULL,
          signature TEXT NOT NULL, documentation TEXT NOT NULL, source_hash TEXT NOT NULL,
          source_text TEXT NOT NULL, build_configuration_id TEXT NOT NULL,
          metadata_json TEXT NOT NULL, PRIMARY KEY(project_id,id));
        CREATE TABLE translation_unit_symbols(
          project_id INTEGER, translation_unit_id TEXT, symbol_id TEXT, is_definition INTEGER,
          snapshot_json TEXT, PRIMARY KEY(project_id,translation_unit_id,symbol_id));
        CREATE TABLE occurrences(
          project_id INTEGER, translation_unit_id TEXT, id TEXT, symbol_id TEXT,
          enclosing_symbol_id TEXT, kind TEXT, path TEXT, start_line INTEGER, end_line INTEGER,
          start_column INTEGER, end_column INTEGER,
          PRIMARY KEY(project_id,translation_unit_id,id));
        CREATE TABLE edges(
          project_id INTEGER, translation_unit_id TEXT, source_id TEXT, target_id TEXT,
          relation TEXT, PRIMARY KEY(project_id,translation_unit_id,source_id,target_id,relation));
        CREATE TABLE embeddings(project_id INTEGER, symbol_id TEXT, model TEXT,
          dimensions INTEGER, magnitude REAL, vector BLOB,
          PRIMARY KEY(project_id,symbol_id,model));
        CREATE VIRTUAL TABLE symbol_fts USING fts5(
          project_id UNINDEXED, symbol_id UNINDEXED, qualified_name, signature,
          documentation, source_text, tokenize='unicode61');
        PRAGMA user_version=2;
        """
    )
    snapshot = json.dumps(
        {
            "id": "legacy-symbol",
            "qualified_name": "legacy",
            "kind": "function",
            "path": str(source),
            "start_line": 1,
            "end_line": 1,
            "start_column": 1,
            "end_column": 27,
            "signature": "int legacy()",
            "documentation": "",
            "source_hash": "legacy-hash",
            "source_text": "int legacy() { return 1; }",
            "build_configuration_id": "legacy-config",
            "translation_unit_id": "legacy-tu",
            "metadata": {"is_definition": True},
        }
    )
    connection.execute("INSERT INTO projects VALUES(1,?)", (str(root),))
    connection.execute(
        "INSERT INTO build_configurations VALUES(1,'legacy-config',?,?,?,'cmd',NULL)",
        (str(source), str(root), "[]"),
    )
    connection.execute(
        """
        INSERT INTO translation_units(
            project_id, id, build_configuration_id, source_path,
            content_hash, diagnostics_json
        ) VALUES(1, 'legacy-tu', 'legacy-config', ?, 'hash', '[]')
        """,
        (str(source),),
    )
    connection.execute(
        """
        INSERT INTO symbols VALUES(
            1, 'legacy-symbol', 'legacy', 'function', ?, 1, 1, 1, 27,
            'int legacy()', '', 'legacy-hash', 'int legacy() { return 1; }',
            'legacy-config', '{"is_definition":true}'
        )
        """,
        (str(source),),
    )
    connection.execute(
        "INSERT INTO translation_unit_symbols VALUES(1,'legacy-tu','legacy-symbol',1,?)",
        (snapshot,),
    )
    connection.commit()
    connection.close()

    with SQLiteStore(database, project_root=root) as store:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION  # noqa: SLF001
        assert store.search(SearchQuery("legacy"))[0].symbol.id == "legacy-symbol"
        assert store.reindex_required_variants() == ("default",)


def test_v4_database_migrates_cfg_tables_in_order(tmp_path: Path) -> None:
    database = tmp_path / "v4.db"
    with SQLiteStore(database):
        pass
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        DROP TABLE cfg_edges;
        DROP TABLE cfg_elements;
        DROP TABLE cfg_blocks;
        DROP TABLE cfg_graphs;
        PRAGMA user_version = 4;
        """
    )
    connection.close()

    with SQLiteStore(database) as store:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 5  # noqa: SLF001
        tables = {
            row[0]
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"cfg_graphs", "cfg_blocks", "cfg_elements", "cfg_edges"} <= tables
