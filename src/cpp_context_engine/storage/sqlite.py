"""Transactional SQLite persistence, FTS5, graph traversal, and vector search."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cpp_context_engine.models import (
    DEFAULT_BUILD_VARIANT,
    BoundedCfgResult,
    BuildScope,
    BuildVariant,
    CallDispatchKind,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    CfgBlock,
    CfgBlockRole,
    CfgEdge,
    CfgEdgeKind,
    CfgElement,
    CfgGraph,
    CodeSymbol,
    GraphDirection,
    GraphEdge,
    GraphRelation,
    MacroExpansionFrame,
    OccurrenceKind,
    SearchHit,
    SearchQuery,
    SourceSpan,
    SymbolKind,
    SymbolOccurrence,
)

if TYPE_CHECKING:
    from cpp_context_engine.ingestion.protocols import IngestionBatch

SCHEMA_VERSION = 6
MAX_CFG_PAGE_SIZE = 10_000
MAX_CALL_PAGE_SIZE = 10_000


@dataclass(frozen=True, slots=True)
class TranslationUnitState:
    translation_unit_id: str
    build_configuration_id: str
    command_hash: str
    content_hash: str
    dependencies: tuple[tuple[Path, str], ...]
    build_variant: str = DEFAULT_BUILD_VARIANT
    analysis_backend: str = "unknown"
    advanced_facts_complete: bool = False


class SQLiteStore:
    """A replaceable local store with atomic translation-unit updates."""

    def __init__(
        self,
        path: Path,
        *,
        project_root: Path | None = None,
        build_scope: BuildScope | None = None,
    ) -> None:
        self.path = path
        self.project_root = project_root.resolve(strict=False) if project_root else None
        self.build_scope = build_scope or BuildScope.single()
        path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI executes synchronous handlers in worker threads; SQLite's serialized
        # mode safely supports this read-heavy connection when the thread guard is off.
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _migrate(self) -> None:
        current = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
            )
        if current == 0:
            with self._connection:
                self._connection.executescript(
                    """
                    CREATE TABLE projects (
                        id INTEGER PRIMARY KEY,
                        root TEXT NOT NULL UNIQUE
                    );

                    CREATE TABLE build_configurations (
                        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        id TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        directory TEXT NOT NULL,
                        arguments_json TEXT NOT NULL,
                        command_hash TEXT NOT NULL,
                        output TEXT,
                        PRIMARY KEY (project_id, id)
                    );

                    CREATE TABLE translation_units (
                        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        id TEXT NOT NULL,
                        build_configuration_id TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        diagnostics_json TEXT NOT NULL,
                        indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (project_id, id),
                        FOREIGN KEY (project_id, build_configuration_id)
                            REFERENCES build_configurations(project_id, id)
                    );

                    CREATE TABLE dependencies (
                        project_id INTEGER NOT NULL,
                        translation_unit_id TEXT NOT NULL,
                        path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        PRIMARY KEY (project_id, translation_unit_id, path),
                        FOREIGN KEY (project_id, translation_unit_id)
                            REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                    );

                    CREATE TABLE symbols (
                        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                        id TEXT NOT NULL,
                        qualified_name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        path TEXT NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        start_column INTEGER NOT NULL,
                        end_column INTEGER NOT NULL,
                        signature TEXT NOT NULL,
                        documentation TEXT NOT NULL,
                        source_hash TEXT NOT NULL,
                        source_text TEXT NOT NULL,
                        build_configuration_id TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        PRIMARY KEY (project_id, id)
                    );

                    CREATE TABLE translation_unit_symbols (
                        project_id INTEGER NOT NULL,
                        translation_unit_id TEXT NOT NULL,
                        symbol_id TEXT NOT NULL,
                        is_definition INTEGER NOT NULL,
                        snapshot_json TEXT NOT NULL,
                        PRIMARY KEY (project_id, translation_unit_id, symbol_id),
                        FOREIGN KEY (project_id, translation_unit_id)
                            REFERENCES translation_units(project_id, id) ON DELETE CASCADE,
                        FOREIGN KEY (project_id, symbol_id)
                            REFERENCES symbols(project_id, id) ON DELETE CASCADE
                    );

                    CREATE TABLE occurrences (
                        project_id INTEGER NOT NULL,
                        translation_unit_id TEXT NOT NULL,
                        id TEXT NOT NULL,
                        symbol_id TEXT NOT NULL,
                        enclosing_symbol_id TEXT,
                        kind TEXT NOT NULL,
                        path TEXT NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        start_column INTEGER NOT NULL,
                        end_column INTEGER NOT NULL,
                        PRIMARY KEY (project_id, translation_unit_id, id),
                        FOREIGN KEY (project_id, translation_unit_id)
                            REFERENCES translation_units(project_id, id) ON DELETE CASCADE,
                        FOREIGN KEY (project_id, symbol_id)
                            REFERENCES symbols(project_id, id) ON DELETE CASCADE
                    );

                    CREATE TABLE edges (
                        project_id INTEGER NOT NULL,
                        translation_unit_id TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        PRIMARY KEY (
                            project_id, translation_unit_id, source_id, target_id, relation
                        ),
                        FOREIGN KEY (project_id, translation_unit_id)
                            REFERENCES translation_units(project_id, id) ON DELETE CASCADE,
                        FOREIGN KEY (project_id, source_id)
                            REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                        FOREIGN KEY (project_id, target_id)
                            REFERENCES symbols(project_id, id) ON DELETE CASCADE
                    );

                    CREATE TABLE embeddings (
                        project_id INTEGER NOT NULL,
                        symbol_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        magnitude REAL NOT NULL,
                        vector BLOB NOT NULL,
                        PRIMARY KEY (project_id, symbol_id, model),
                        FOREIGN KEY (project_id, symbol_id)
                            REFERENCES symbols(project_id, id) ON DELETE CASCADE
                    );

                    CREATE VIRTUAL TABLE symbol_fts USING fts5(
                        project_id UNINDEXED,
                        symbol_id UNINDEXED,
                        qualified_name,
                        signature,
                        documentation,
                        source_text,
                        tokenize = 'unicode61'
                    );

                    CREATE TRIGGER symbols_ai AFTER INSERT ON symbols BEGIN
                        INSERT INTO symbol_fts(
                            project_id, symbol_id, qualified_name, signature,
                            documentation, source_text
                        ) VALUES (
                            new.project_id, new.id, new.qualified_name, new.signature,
                            new.documentation, new.source_text
                        );
                    END;
                    CREATE TRIGGER symbols_ad AFTER DELETE ON symbols BEGIN
                        DELETE FROM symbol_fts
                        WHERE project_id = old.project_id AND symbol_id = old.id;
                    END;
                    CREATE TRIGGER symbols_au AFTER UPDATE ON symbols BEGIN
                        DELETE FROM embeddings
                        WHERE project_id = old.project_id AND symbol_id = old.id
                          AND (
                            old.source_hash <> new.source_hash OR
                            old.qualified_name <> new.qualified_name OR
                            old.signature <> new.signature OR
                            old.documentation <> new.documentation OR
                            old.source_text <> new.source_text
                          );
                        DELETE FROM symbol_fts
                        WHERE project_id = old.project_id AND symbol_id = old.id;
                        INSERT INTO symbol_fts(
                            project_id, symbol_id, qualified_name, signature,
                            documentation, source_text
                        ) VALUES (
                            new.project_id, new.id, new.qualified_name, new.signature,
                            new.documentation, new.source_text
                        );
                    END;

                    CREATE INDEX edges_source ON edges(project_id, source_id, relation);
                    CREATE INDEX edges_target ON edges(project_id, target_id, relation);
                    CREATE INDEX occurrences_symbol ON occurrences(project_id, symbol_id);
                    PRAGMA user_version = 2;
                    """
                )
        elif current == 1:
            with self._connection:
                self._connection.executescript(
                    """
                    DROP TRIGGER symbols_au;
                    CREATE TRIGGER symbols_au AFTER UPDATE ON symbols BEGIN
                        DELETE FROM embeddings
                        WHERE project_id = old.project_id AND symbol_id = old.id
                          AND (
                            old.source_hash <> new.source_hash OR
                            old.qualified_name <> new.qualified_name OR
                            old.signature <> new.signature OR
                            old.documentation <> new.documentation OR
                            old.source_text <> new.source_text
                          );
                        DELETE FROM symbol_fts
                        WHERE project_id = old.project_id AND symbol_id = old.id;
                        INSERT INTO symbol_fts(
                            project_id, symbol_id, qualified_name, signature,
                            documentation, source_text
                        ) VALUES (
                            new.project_id, new.id, new.qualified_name, new.signature,
                            new.documentation, new.source_text
                        );
                    END;
                    PRAGMA user_version = 2;
                    """
                )
        if current <= 2:
            self._migrate_v3()
        if current <= 3:
            self._migrate_v4()
        if current <= 4:
            self._migrate_v5()
        if current <= 5:
            self._migrate_v6()

    def _migrate_v3(self) -> None:
        """Add build/TU evidence tables without discarding baseline v2 reads."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            _execute_script(
                self._connection,
                """
                CREATE TABLE build_variants (
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    compilation_database TEXT NOT NULL,
                    target TEXT NOT NULL DEFAULT '',
                    platform TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    reindex_required INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (project_id, name)
                );

                ALTER TABLE build_configurations
                    ADD COLUMN build_variant TEXT NOT NULL DEFAULT 'default';
                ALTER TABLE translation_units
                    ADD COLUMN build_variant TEXT NOT NULL DEFAULT 'default';
                ALTER TABLE occurrences
                    ADD COLUMN build_variant TEXT NOT NULL DEFAULT 'default';
                ALTER TABLE occurrences
                    ADD COLUMN build_configuration_id TEXT NOT NULL DEFAULT '';
                ALTER TABLE edges ADD COLUMN id TEXT NOT NULL DEFAULT '';
                ALTER TABLE edges
                    ADD COLUMN build_variant TEXT NOT NULL DEFAULT 'default';
                ALTER TABLE edges
                    ADD COLUMN build_configuration_id TEXT NOT NULL DEFAULT '';

                CREATE TABLE symbol_variants (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    symbol_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    is_definition INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (project_id, build_variant, translation_unit_id, symbol_id),
                    FOREIGN KEY (project_id, symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE variant_embeddings (
                    project_id INTEGER NOT NULL,
                    variant_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    magnitude REAL NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (project_id, variant_id, model),
                    FOREIGN KEY (project_id, variant_id)
                        REFERENCES symbol_variants(project_id, id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE symbol_variant_fts USING fts5(
                    project_id UNINDEXED,
                    variant_id UNINDEXED,
                    symbol_id UNINDEXED,
                    build_variant UNINDEXED,
                    qualified_name,
                    signature,
                    documentation,
                    source_text,
                    tokenize = 'unicode61'
                );

                CREATE INDEX symbol_variants_scope
                    ON symbol_variants(project_id, build_variant, symbol_id);
                CREATE INDEX translation_units_scope
                    ON translation_units(project_id, build_variant, id);
                CREATE INDEX edges_scope_source
                    ON edges(project_id, build_variant, source_id, relation);
                CREATE INDEX edges_scope_target
                    ON edges(project_id, build_variant, target_id, relation);
                """,
            )
            projects = self._connection.execute("SELECT id FROM projects").fetchall()
            for project in projects:
                project_id = int(project[0])
                self._connection.execute(
                    """
                    INSERT INTO build_variants(
                        project_id, name, compilation_database, reindex_required
                    ) VALUES (?, 'default', '', 1)
                    """,
                    (project_id,),
                )
                units = {
                    row["id"]: row["build_configuration_id"]
                    for row in self._connection.execute(
                        """
                        SELECT id, build_configuration_id FROM translation_units
                        WHERE project_id = ?
                        """,
                        (project_id,),
                    )
                }
                origins = self._connection.execute(
                    """
                    SELECT translation_unit_id, symbol_id, is_definition, snapshot_json
                    FROM translation_unit_symbols WHERE project_id = ?
                    """,
                    (project_id,),
                ).fetchall()
                for origin in origins:
                    variant_id = _stable_id(
                        "variant",
                        DEFAULT_BUILD_VARIANT,
                        origin["translation_unit_id"],
                        origin["symbol_id"],
                    )
                    self._connection.execute(
                        """
                        INSERT INTO symbol_variants(
                            project_id, id, symbol_id, build_variant,
                            build_configuration_id, translation_unit_id,
                            is_definition, snapshot_json
                        ) VALUES (?, ?, ?, 'default', ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            variant_id,
                            origin["symbol_id"],
                            units.get(origin["translation_unit_id"], ""),
                            origin["translation_unit_id"],
                            origin["is_definition"],
                            origin["snapshot_json"],
                        ),
                    )
                edge_rows = self._connection.execute(
                    """
                    SELECT rowid, translation_unit_id, source_id, target_id, relation
                    FROM edges WHERE project_id = ? ORDER BY rowid
                    """,
                    (project_id,),
                ).fetchall()
                for edge in edge_rows:
                    edge_id = _stable_id(
                        "edge",
                        DEFAULT_BUILD_VARIANT,
                        edge["translation_unit_id"],
                        edge["source_id"],
                        edge["target_id"],
                        edge["relation"],
                        str(edge["rowid"]),
                    )
                    self._connection.execute(
                        """
                        UPDATE edges SET id = ?, build_configuration_id = ?
                        WHERE rowid = ?
                        """,
                        (edge_id, units.get(edge["translation_unit_id"], ""), edge["rowid"]),
                    )
                self._connection.execute(
                    """
                    UPDATE occurrences
                    SET build_configuration_id = COALESCE((
                        SELECT units.build_configuration_id FROM translation_units units
                        WHERE units.project_id = occurrences.project_id
                          AND units.id = occurrences.translation_unit_id
                    ), '')
                    WHERE project_id = ?
                    """,
                    (project_id,),
                )
            self._rebuild_variant_fts()
            _execute_script(
                self._connection,
                """
                CREATE TABLE edges_v3 (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, source_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, target_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE
                );
                INSERT INTO edges_v3(
                    project_id, id, translation_unit_id, build_configuration_id,
                    build_variant, source_id, target_id, relation
                )
                SELECT project_id, id, translation_unit_id, build_configuration_id,
                       build_variant, source_id, target_id, relation
                FROM edges;
                DROP TABLE edges;
                ALTER TABLE edges_v3 RENAME TO edges;
                CREATE INDEX edges_source ON edges(project_id, source_id, relation);
                CREATE INDEX edges_target ON edges(project_id, target_id, relation);
                CREATE INDEX edges_scope_source
                    ON edges(project_id, build_variant, source_id, relation);
                CREATE INDEX edges_scope_target
                    ON edges(project_id, build_variant, target_id, relation);
                """,
            )
            self._connection.execute("PRAGMA user_version = 3")
        except BaseException:
            # DDL is transactional in SQLite as long as executescript does not commit it early.
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_v4(self) -> None:
        """Persist analyzer-specific occurrence evidence such as macro provenance."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                "ALTER TABLE occurrences ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
            self._connection.execute(
                "ALTER TABLE translation_units "
                "ADD COLUMN analysis_backend TEXT NOT NULL DEFAULT 'unknown'"
            )
            self._connection.execute(
                "ALTER TABLE translation_units "
                "ADD COLUMN advanced_facts_complete INTEGER NOT NULL DEFAULT 0"
            )
            self._connection.execute("PRAGMA user_version = 4")
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_v5(self) -> None:
        """Add versioned, build-scoped CFG facts without overloading symbol tables."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            _execute_script(
                self._connection,
                """
                CREATE TABLE cfg_graphs (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    function_symbol_id TEXT NOT NULL,
                    entry_block_id TEXT NOT NULL,
                    normal_exit_block_id TEXT NOT NULL,
                    exceptional_exit_block_id TEXT,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    clang_major INTEGER NOT NULL,
                    fact_schema_version INTEGER NOT NULL,
                    build_options_json TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (
                        project_id, build_variant, build_configuration_id,
                        translation_unit_id, function_symbol_id
                    ),
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, function_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE cfg_blocks (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    block_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    reachable INTEGER NOT NULL,
                    terminator_kind TEXT NOT NULL,
                    terminator_text TEXT NOT NULL,
                    terminator_spelling_span_json TEXT,
                    terminator_expansion_span_json TEXT,
                    label_kind TEXT NOT NULL,
                    label_text TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (project_id, graph_id, block_index),
                    FOREIGN KEY (project_id, graph_id)
                        REFERENCES cfg_graphs(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE cfg_elements (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    element_index INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    statement_class TEXT NOT NULL,
                    text TEXT NOT NULL,
                    spelling_span_json TEXT,
                    expansion_span_json TEXT,
                    metadata_json TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (project_id, block_id, element_index),
                    FOREIGN KEY (project_id, graph_id)
                        REFERENCES cfg_graphs(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, block_id)
                        REFERENCES cfg_blocks(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE cfg_edges (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    source_block_id TEXT NOT NULL,
                    target_block_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    successor_index INTEGER NOT NULL,
                    feasible INTEGER NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY (project_id, graph_id)
                        REFERENCES cfg_graphs(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, source_block_id)
                        REFERENCES cfg_blocks(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, target_block_id)
                        REFERENCES cfg_blocks(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE INDEX cfg_graphs_function_scope
                    ON cfg_graphs(project_id, function_symbol_id, build_variant, id);
                CREATE INDEX cfg_graphs_tu
                    ON cfg_graphs(project_id, translation_unit_id);
                CREATE INDEX cfg_blocks_graph_order
                    ON cfg_blocks(project_id, graph_id, block_index, id);
                CREATE INDEX cfg_elements_graph_order
                    ON cfg_elements(project_id, graph_id, block_id, element_index, id);
                CREATE INDEX cfg_edges_graph_order
                    ON cfg_edges(
                        project_id, graph_id, source_block_id, successor_index,
                        target_block_id, kind, id
                    );
                """,
            )
            # Native v4 rows predate CFG facts but otherwise looked "advanced complete".
            # Downgrading only those rows forces one normal companion reindex.
            self._connection.execute(
                "UPDATE translation_units SET advanced_facts_complete = 0 "
                "WHERE analysis_backend = 'clang-libtooling'"
            )
            self._connection.execute("PRAGMA user_version = 5")
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_v6(self) -> None:
        """Add build-scoped callsite and dispatch-target evidence atomically."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            _execute_script(
                self._connection,
                """
                CREATE TABLE callsites (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    owner_symbol_id TEXT NOT NULL,
                    dispatch_kind TEXT NOT NULL,
                    spelling_span_json TEXT NOT NULL,
                    expansion_span_json TEXT NOT NULL,
                    expansion_stack_json TEXT NOT NULL,
                    static_target_symbol_id TEXT,
                    target_set_complete INTEGER NOT NULL,
                    unresolved_reason TEXT NOT NULL,
                    callee_text TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY (project_id, owner_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, static_target_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE call_targets (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    callsite_id TEXT NOT NULL,
                    target_symbol_id TEXT NOT NULL,
                    certainty TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
                    confidence_reason TEXT NOT NULL,
                    derivation TEXT NOT NULL,
                    evidence_span_json TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (project_id, callsite_id, target_symbol_id),
                    FOREIGN KEY (project_id, callsite_id)
                        REFERENCES callsites(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, target_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE INDEX callsites_owner_scope
                    ON callsites(project_id, owner_symbol_id, build_variant, id);
                CREATE INDEX callsites_static_target_scope
                    ON callsites(project_id, static_target_symbol_id, build_variant, id);
                CREATE INDEX callsites_tu ON callsites(project_id, translation_unit_id);
                CREATE INDEX call_targets_callsite_order
                    ON call_targets(project_id, callsite_id, certainty, target_symbol_id, id);
                CREATE INDEX call_targets_target_scope
                    ON call_targets(project_id, target_symbol_id, build_variant, id);
                """,
            )
            # Older native rows did not contain explicit unresolved indirect calls or
            # dispatch certainty, so a normal incremental run must replace them.
            self._connection.execute(
                "UPDATE translation_units SET advanced_facts_complete = 0 "
                "WHERE analysis_backend = 'clang-libtooling'"
            )
            self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _rebuild_variant_fts(self) -> None:
        self._connection.execute("DELETE FROM symbol_variant_fts")
        rows = self._connection.execute(
            """
            SELECT project_id, id, symbol_id, build_variant, snapshot_json
            FROM symbol_variants ORDER BY project_id, id
            """
        )
        for row in rows:
            snapshot = json.loads(row["snapshot_json"])
            self._connection.execute(
                """
                INSERT INTO symbol_variant_fts(
                    project_id, variant_id, symbol_id, build_variant,
                    qualified_name, signature, documentation, source_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["project_id"],
                    row["id"],
                    row["symbol_id"],
                    row["build_variant"],
                    snapshot["qualified_name"],
                    snapshot["signature"],
                    snapshot["documentation"],
                    snapshot["source_text"],
                ),
            )

    def apply_ingestion(
        self,
        project_root: Path,
        batch: IngestionBatch,
        *,
        current_translation_unit_ids: frozenset[str] | None = None,
        build_variant: BuildVariant | None = None,
    ) -> None:
        """Atomically replace changed units and optionally remove stale units."""

        root = str(project_root.resolve(strict=False))
        selected_variant = build_variant or (
            batch.build_variants[0]
            if batch.build_variants
            else BuildVariant(DEFAULT_BUILD_VARIANT, Path("."))
        )
        with self._connection:
            project_id = self._ensure_project(root)
            self._put_build_variant(project_id, selected_variant)
            existing: set[str] = set()
            if current_translation_unit_ids is not None:
                existing = {
                    row[0]
                    for row in self._connection.execute(
                        """
                        SELECT id FROM translation_units
                        WHERE project_id = ? AND build_variant = ?
                        """,
                        (project_id, selected_variant.name),
                    )
                }
            changed_ids = {unit.id for unit in batch.translation_units}
            removed_ids = (
                existing - current_translation_unit_ids
                if current_translation_unit_ids is not None
                else set()
            )
            replaced_ids = removed_ids | changed_ids
            affected_symbols = self._symbols_from_units(project_id, replaced_ids)
            self._delete_translation_units(project_id, replaced_ids)

            for configuration in batch.build_configurations:
                self._connection.execute(
                    """
                    INSERT INTO build_configurations(
                        project_id, id, source_path, directory, arguments_json,
                        command_hash, output, build_variant
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, id) DO UPDATE SET
                        source_path = excluded.source_path,
                        directory = excluded.directory,
                        arguments_json = excluded.arguments_json,
                        command_hash = excluded.command_hash,
                        output = excluded.output
                    """,
                    (
                        project_id,
                        configuration.id,
                        str(configuration.source_path),
                        str(configuration.directory),
                        json.dumps(configuration.arguments),
                        configuration.command_hash,
                        str(configuration.output) if configuration.output else None,
                        configuration.build_variant,
                    ),
                )
            for unit in batch.translation_units:
                self._connection.execute(
                    """
                    INSERT INTO translation_units(
                        project_id, id, build_configuration_id, source_path,
                        content_hash, diagnostics_json, build_variant,
                        analysis_backend, advanced_facts_complete
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        unit.id,
                        unit.build_configuration_id,
                        str(unit.source_path),
                        unit.content_hash,
                        json.dumps(unit.diagnostics),
                        unit.build_variant,
                        unit.analysis_backend,
                        int(unit.advanced_facts_complete),
                    ),
                )
                self._connection.executemany(
                    """
                    INSERT INTO dependencies(
                        project_id, translation_unit_id, path, content_hash
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        (project_id, unit.id, str(path), content_hash)
                        for path, content_hash in unit.dependencies
                    ),
                )

            for symbol in batch.symbols:
                self._put_symbol(project_id, symbol)
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO translation_unit_symbols(
                        project_id, translation_unit_id, symbol_id,
                        is_definition, snapshot_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        symbol.translation_unit_id,
                        symbol.id,
                        int(bool(symbol.metadata.get("is_definition"))),
                        self._symbol_snapshot(symbol),
                    ),
                )
                self._put_symbol_variant(project_id, symbol)
            self._put_occurrences(project_id, batch.occurrences)
            self._put_edges(project_id, batch.edges)
            self._put_cfg_facts(
                project_id,
                batch.cfg_graphs,
                batch.cfg_blocks,
                batch.cfg_elements,
                batch.cfg_edges,
            )
            self._put_call_facts(project_id, batch.callsites, batch.call_targets)
            self._refresh_symbols(
                project_id, affected_symbols | {symbol.id for symbol in batch.symbols}
            )
            self._delete_orphans(project_id)
            self._connection.execute(
                """
                UPDATE build_variants SET reindex_required = 0
                WHERE project_id = ? AND name = ?
                """,
                (project_id, selected_variant.name),
            )

    def _put_build_variant(self, project_id: int, variant: BuildVariant) -> None:
        self._connection.execute(
            """
            INSERT INTO build_variants(
                project_id, name, compilation_database, target, platform,
                metadata_json, reindex_required
            ) VALUES (?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(project_id, name) DO UPDATE SET
                compilation_database = excluded.compilation_database,
                target = excluded.target,
                platform = excluded.platform,
                metadata_json = excluded.metadata_json
            """,
            (
                project_id,
                variant.name,
                str(variant.compilation_database),
                variant.target,
                variant.platform,
                json.dumps(dict(variant.metadata), sort_keys=True),
            ),
        )

    def build_variants(self, project_root: Path | None = None) -> tuple[BuildVariant, ...]:
        try:
            project_id = self._project_id(project_root)
        except KeyError:
            return ()
        return tuple(
            BuildVariant(
                row["name"],
                Path(row["compilation_database"] or "."),
                row["target"],
                row["platform"],
                json.loads(row["metadata_json"]),
            )
            for row in self._connection.execute(
                "SELECT * FROM build_variants WHERE project_id = ? ORDER BY name",
                (project_id,),
            )
        )

    def reindex_required_variants(self, project_root: Path | None = None) -> tuple[str, ...]:
        project_id = self._project_id(project_root)
        return tuple(
            row[0]
            for row in self._connection.execute(
                """
                SELECT name FROM build_variants
                WHERE project_id = ? AND reindex_required = 1 ORDER BY name
                """,
                (project_id,),
            )
        )

    def remove_build_variant(self, name: str, project_root: Path | None = None) -> bool:
        """Explicitly delete one build's facts while preserving every other variant."""

        project_id = self._project_id(project_root)
        with self._connection:
            affected = {
                row[0]
                for row in self._connection.execute(
                    """
                    SELECT DISTINCT symbol_id FROM symbol_variants
                    WHERE project_id = ? AND build_variant = ?
                    """,
                    (project_id, name),
                )
            }
            cursor = self._connection.execute(
                "DELETE FROM build_variants WHERE project_id = ? AND name = ?",
                (project_id, name),
            )
            if cursor.rowcount == 0:
                return False
            unit_ids = tuple(
                row[0]
                for row in self._connection.execute(
                    """
                    SELECT id FROM translation_units
                    WHERE project_id = ? AND build_variant = ?
                    """,
                    (project_id, name),
                )
            )
            self._delete_translation_units(project_id, unit_ids)
            self._connection.execute(
                "DELETE FROM build_configurations WHERE project_id = ? AND build_variant = ?",
                (project_id, name),
            )
            self._refresh_symbols(project_id, affected)
            self._delete_orphans(project_id)
            return True

    def _ensure_project(self, root: str) -> int:
        self._connection.execute("INSERT OR IGNORE INTO projects(root) VALUES (?)", (root,))
        return self._connection.execute(
            "SELECT id FROM projects WHERE root = ?", (root,)
        ).fetchone()[0]

    def _project_id(self, project_root: Path | None = None) -> int:
        root = project_root or self.project_root
        if root is None:
            rows = self._connection.execute(
                "SELECT id FROM projects ORDER BY id LIMIT 2"
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(
                    "project_root is required when the database has zero or multiple projects"
                )
            return rows[0][0]
        row = self._connection.execute(
            "SELECT id FROM projects WHERE root = ?", (str(root.resolve(strict=False)),)
        ).fetchone()
        if row is None:
            raise KeyError(f"project is not indexed: {root}")
        return row[0]

    def has_project(self, project_root: Path | None = None) -> bool:
        try:
            self._project_id(project_root)
        except (KeyError, ValueError):
            return False
        return True

    def _delete_translation_units(self, project_id: int, unit_ids: Iterable[str]) -> None:
        ids = tuple(unit_ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._connection.execute(
            f"""
            DELETE FROM symbol_variant_fts
            WHERE project_id = ? AND variant_id IN (
                SELECT id FROM symbol_variants
                WHERE project_id = ? AND translation_unit_id IN ({placeholders})
            )
            """,
            (project_id, project_id, *ids),
        )
        self._connection.executemany(
            "DELETE FROM translation_units WHERE project_id = ? AND id = ?",
            ((project_id, unit_id) for unit_id in ids),
        )
        self._delete_orphans(project_id)

    def _symbols_from_units(self, project_id: int, unit_ids: set[str]) -> set[str]:
        if not unit_ids:
            return set()
        placeholders = ",".join("?" for _ in unit_ids)
        return {
            row[0]
            for row in self._connection.execute(
                f"""
                SELECT DISTINCT symbol_id FROM translation_unit_symbols
                WHERE project_id = ? AND translation_unit_id IN ({placeholders})
                """,
                (project_id, *sorted(unit_ids)),
            )
        }

    def _delete_orphans(self, project_id: int) -> None:
        self._connection.execute(
            """
            DELETE FROM symbols
            WHERE project_id = ? AND NOT EXISTS (
                SELECT 1 FROM symbol_variants variants
                WHERE variants.project_id = symbols.project_id
                  AND variants.symbol_id = symbols.id
            )
            """,
            (project_id,),
        )
        self._connection.execute(
            """
            DELETE FROM build_configurations
            WHERE project_id = ? AND NOT EXISTS (
                SELECT 1 FROM translation_units units
                WHERE units.project_id = build_configurations.project_id
                  AND units.build_configuration_id = build_configurations.id
            )
            """,
            (project_id,),
        )

    def _put_symbol(
        self, project_id: int, symbol: CodeSymbol, *, prefer_definition: bool = True
    ) -> None:
        existing = self._connection.execute(
            "SELECT metadata_json FROM symbols WHERE project_id = ? AND id = ?",
            (project_id, symbol.id),
        ).fetchone()
        if existing is not None and prefer_definition:
            existing_metadata = json.loads(existing["metadata_json"])
            if existing_metadata.get("is_definition") and not symbol.metadata.get("is_definition"):
                return
        self._connection.execute(
            """
            INSERT INTO symbols(
                project_id, id, qualified_name, kind, path, start_line, end_line,
                start_column, end_column, signature, documentation, source_hash,
                source_text, build_configuration_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, id) DO UPDATE SET
                qualified_name = excluded.qualified_name,
                kind = excluded.kind,
                path = excluded.path,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                start_column = excluded.start_column,
                end_column = excluded.end_column,
                signature = excluded.signature,
                documentation = excluded.documentation,
                source_hash = excluded.source_hash,
                source_text = excluded.source_text,
                build_configuration_id = excluded.build_configuration_id,
                metadata_json = excluded.metadata_json
            """,
            (
                project_id,
                symbol.id,
                symbol.qualified_name,
                symbol.kind.value,
                str(symbol.span.path),
                symbol.span.start_line,
                symbol.span.end_line,
                symbol.span.start_column,
                symbol.span.end_column,
                symbol.signature,
                symbol.documentation,
                symbol.source_hash,
                symbol.source_text,
                symbol.build_configuration_id,
                json.dumps(dict(symbol.metadata), sort_keys=True),
            ),
        )

    @staticmethod
    def _symbol_snapshot(symbol: CodeSymbol) -> str:
        return json.dumps(
            {
                "id": symbol.id,
                "qualified_name": symbol.qualified_name,
                "kind": symbol.kind.value,
                "path": str(symbol.span.path),
                "start_line": symbol.span.start_line,
                "end_line": symbol.span.end_line,
                "start_column": symbol.span.start_column,
                "end_column": symbol.span.end_column,
                "signature": symbol.signature,
                "documentation": symbol.documentation,
                "source_hash": symbol.source_hash,
                "source_text": symbol.source_text,
                "build_configuration_id": symbol.build_configuration_id,
                "translation_unit_id": symbol.translation_unit_id,
                "build_variant": symbol.build_variant,
                "variant_id": symbol.variant_id,
                "metadata": dict(symbol.metadata),
            },
            sort_keys=True,
        )

    @staticmethod
    def _snapshot_symbol(snapshot: str) -> CodeSymbol:
        data = json.loads(snapshot)
        return CodeSymbol(
            id=data["id"],
            qualified_name=data["qualified_name"],
            kind=SymbolKind(data["kind"]),
            span=SourceSpan(
                Path(data["path"]),
                data["start_line"],
                data["end_line"],
                data["start_column"],
                data["end_column"],
            ),
            signature=data["signature"],
            documentation=data["documentation"],
            source_hash=data["source_hash"],
            source_text=data["source_text"],
            build_configuration_id=data["build_configuration_id"],
            translation_unit_id=data["translation_unit_id"],
            build_variant=data.get("build_variant", DEFAULT_BUILD_VARIANT),
            variant_id=data.get("variant_id", ""),
            metadata=data["metadata"],
        )

    def _put_symbol_variant(self, project_id: int, symbol: CodeSymbol) -> None:
        variant_id = symbol.variant_id or _stable_id(
            "variant", symbol.build_variant, symbol.translation_unit_id, symbol.id
        )
        snapshot = self._symbol_snapshot(
            CodeSymbol(
                id=symbol.id,
                qualified_name=symbol.qualified_name,
                kind=symbol.kind,
                span=symbol.span,
                signature=symbol.signature,
                documentation=symbol.documentation,
                source_hash=symbol.source_hash,
                source_text=symbol.source_text,
                build_configuration_id=symbol.build_configuration_id,
                translation_unit_id=symbol.translation_unit_id,
                build_variant=symbol.build_variant,
                variant_id=variant_id,
                metadata=symbol.metadata,
            )
        )
        existing = self._connection.execute(
            """
            SELECT snapshot_json FROM symbol_variants
            WHERE project_id = ? AND id = ?
            """,
            (project_id, variant_id),
        ).fetchone()
        if existing is not None and existing["snapshot_json"] != snapshot:
            self._connection.execute(
                "DELETE FROM variant_embeddings WHERE project_id = ? AND variant_id = ?",
                (project_id, variant_id),
            )
        self._connection.execute(
            "DELETE FROM symbol_variant_fts WHERE project_id = ? AND variant_id = ?",
            (project_id, variant_id),
        )
        self._connection.execute(
            """
            INSERT INTO symbol_variants(
                project_id, id, symbol_id, build_variant, build_configuration_id,
                translation_unit_id, is_definition, snapshot_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, id) DO UPDATE SET
                symbol_id = excluded.symbol_id,
                build_variant = excluded.build_variant,
                build_configuration_id = excluded.build_configuration_id,
                translation_unit_id = excluded.translation_unit_id,
                is_definition = excluded.is_definition,
                snapshot_json = excluded.snapshot_json
            """,
            (
                project_id,
                variant_id,
                symbol.id,
                symbol.build_variant,
                symbol.build_configuration_id,
                symbol.translation_unit_id,
                int(bool(symbol.metadata.get("is_definition"))),
                snapshot,
            ),
        )
        self._connection.execute(
            """
            INSERT INTO symbol_variant_fts(
                project_id, variant_id, symbol_id, build_variant,
                qualified_name, signature, documentation, source_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                variant_id,
                symbol.id,
                symbol.build_variant,
                symbol.qualified_name,
                symbol.signature,
                symbol.documentation,
                symbol.source_text,
            ),
        )

    def _refresh_symbols(self, project_id: int, symbol_ids: set[str]) -> None:
        for symbol_id in symbol_ids:
            row = self._connection.execute(
                """
                SELECT snapshot_json FROM symbol_variants
                WHERE project_id = ? AND symbol_id = ?
                ORDER BY is_definition DESC, build_variant, translation_unit_id
                LIMIT 1
                """,
                (project_id, symbol_id),
            ).fetchone()
            if row is not None:
                self._put_symbol(
                    project_id,
                    self._snapshot_symbol(row["snapshot_json"]),
                    prefer_definition=False,
                )

    def _put_occurrences(self, project_id: int, occurrences: Iterable[SymbolOccurrence]) -> None:
        self._connection.executemany(
            """
            INSERT OR REPLACE INTO occurrences(
                project_id, translation_unit_id, id, symbol_id, enclosing_symbol_id,
                kind, path, start_line, end_line, start_column, end_column,
                build_configuration_id, build_variant, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    occurrence.translation_unit_id,
                    occurrence.id,
                    occurrence.symbol_id,
                    occurrence.enclosing_symbol_id,
                    occurrence.kind.value,
                    str(occurrence.span.path),
                    occurrence.span.start_line,
                    occurrence.span.end_line,
                    occurrence.span.start_column,
                    occurrence.span.end_column,
                    occurrence.build_configuration_id,
                    occurrence.build_variant,
                    json.dumps(dict(occurrence.metadata), sort_keys=True),
                )
                for occurrence in occurrences
            ),
        )

    def _put_edges(self, project_id: int, edges: Iterable[GraphEdge]) -> None:
        self._connection.executemany(
            """
            INSERT OR IGNORE INTO edges(
                project_id, id, translation_unit_id, build_configuration_id,
                build_variant, source_id, target_id, relation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    edge.id
                    or _stable_id(
                        "edge",
                        edge.build_variant,
                        edge.translation_unit_id,
                        edge.source_id,
                        edge.target_id,
                        edge.relation.value,
                    ),
                    edge.translation_unit_id,
                    edge.build_configuration_id,
                    edge.build_variant,
                    edge.source_id,
                    edge.target_id,
                    edge.relation.value,
                )
                for edge in edges
            ),
        )

    def _put_cfg_facts(
        self,
        project_id: int,
        graphs: Iterable[CfgGraph],
        blocks: Iterable[CfgBlock],
        elements: Iterable[CfgElement],
        edges: Iterable[CfgEdge],
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO cfg_graphs(
                project_id, id, function_symbol_id, entry_block_id,
                normal_exit_block_id, exceptional_exit_block_id,
                translation_unit_id, build_configuration_id, build_variant,
                clang_major, fact_schema_version, build_options_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    graph.id,
                    graph.function_symbol_id,
                    graph.entry_block_id,
                    graph.normal_exit_block_id,
                    graph.exceptional_exit_block_id,
                    graph.translation_unit_id,
                    graph.build_configuration_id,
                    graph.build_variant,
                    graph.clang_major,
                    graph.fact_schema_version,
                    json.dumps(dict(graph.build_options), sort_keys=True),
                )
                for graph in graphs
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO cfg_blocks(
                project_id, id, graph_id, block_index, role, reachable,
                terminator_kind, terminator_text,
                terminator_spelling_span_json, terminator_expansion_span_json,
                label_kind, label_text, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    block.id,
                    block.graph_id,
                    block.index,
                    block.role.value,
                    int(block.reachable),
                    block.terminator_kind,
                    block.terminator_text,
                    _span_json(block.terminator_spelling_span),
                    _span_json(block.terminator_expansion_span),
                    block.label_kind,
                    block.label_text,
                    block.translation_unit_id,
                    block.build_configuration_id,
                    block.build_variant,
                )
                for block in blocks
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO cfg_elements(
                project_id, id, graph_id, block_id, element_index, kind,
                statement_class, text, spelling_span_json, expansion_span_json,
                metadata_json, translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    element.id,
                    element.graph_id,
                    element.block_id,
                    element.index,
                    element.kind,
                    element.statement_class,
                    element.text,
                    _span_json(element.spelling_span),
                    _span_json(element.expansion_span),
                    json.dumps(dict(element.metadata), sort_keys=True),
                    element.translation_unit_id,
                    element.build_configuration_id,
                    element.build_variant,
                )
                for element in elements
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO cfg_edges(
                project_id, id, graph_id, source_block_id, target_block_id,
                kind, successor_index, feasible, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    edge.id,
                    edge.graph_id,
                    edge.source_block_id,
                    edge.target_block_id,
                    edge.kind.value,
                    edge.successor_index,
                    int(edge.feasible),
                    edge.translation_unit_id,
                    edge.build_configuration_id,
                    edge.build_variant,
                )
                for edge in edges
            ),
        )

    def _put_call_facts(
        self,
        project_id: int,
        callsites: Iterable[CallSite],
        targets: Iterable[CallTarget],
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO callsites(
                project_id, id, owner_symbol_id, dispatch_kind,
                spelling_span_json, expansion_span_json, expansion_stack_json,
                static_target_symbol_id, target_set_complete, unresolved_reason,
                callee_text, translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    site.id,
                    site.owner_symbol_id,
                    site.dispatch_kind.value,
                    _span_json(site.spelling_span),
                    _span_json(site.expansion_span),
                    json.dumps(
                        [
                            {
                                "macro_symbol_id": frame.macro_symbol_id,
                                "name": frame.name,
                                "spelling_span": _span_payload(frame.spelling_span),
                                "expansion_span": _span_payload(frame.expansion_span),
                            }
                            for frame in site.expansion_stack
                        ],
                        sort_keys=True,
                    ),
                    site.static_target_symbol_id,
                    int(site.target_set_complete),
                    site.unresolved_reason,
                    site.callee_text,
                    site.translation_unit_id,
                    site.build_configuration_id,
                    site.build_variant,
                )
                for site in callsites
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO call_targets(
                project_id, id, callsite_id, target_symbol_id, certainty,
                confidence, confidence_reason, derivation, evidence_span_json,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    target.id,
                    target.callsite_id,
                    target.target_symbol_id,
                    target.certainty.value,
                    target.confidence,
                    target.confidence_reason,
                    target.derivation,
                    _span_json(target.evidence_span),
                    target.translation_unit_id,
                    target.build_configuration_id,
                    target.build_variant,
                )
                for target in targets
            ),
        )

    def translation_unit_states(
        self,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> dict[str, TranslationUnitState]:
        try:
            project_id = self._project_id(project_root)
        except KeyError:
            return {}
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT units.id, units.build_configuration_id, configs.command_hash,
                   units.content_hash, units.build_variant, units.analysis_backend,
                   units.advanced_facts_complete
            FROM translation_units units
            JOIN build_configurations configs
              ON configs.project_id = units.project_id
             AND configs.id = units.build_configuration_id
            WHERE units.project_id = ? AND units.build_variant IN ({placeholders})
            """,
            (project_id, *names),
        ).fetchall()
        result: dict[str, TranslationUnitState] = {}
        for row in rows:
            dependencies = tuple(
                (Path(dependency[0]), dependency[1])
                for dependency in self._connection.execute(
                    """
                    SELECT path, content_hash FROM dependencies
                    WHERE project_id = ? AND translation_unit_id = ? ORDER BY path
                    """,
                    (project_id, row["id"]),
                )
            )
            result[row["id"]] = TranslationUnitState(
                translation_unit_id=row["id"],
                build_configuration_id=row["build_configuration_id"],
                command_hash=row["command_hash"],
                content_hash=row["content_hash"],
                dependencies=dependencies,
                build_variant=row["build_variant"],
                analysis_backend=row["analysis_backend"],
                advanced_facts_complete=bool(row["advanced_facts_complete"]),
            )
        return result

    def get_symbol(
        self,
        symbol_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> CodeSymbol | None:
        try:
            project_id = self._project_id(project_root)
        except KeyError:
            return None
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        row = self._connection.execute(
            f"""
            SELECT * FROM symbol_variants
            WHERE project_id = ? AND (symbol_id = ? OR id = ?)
              AND build_variant IN ({placeholders})
            ORDER BY is_definition DESC, build_variant, translation_unit_id LIMIT 1
            """,
            (project_id, symbol_id, symbol_id, *names),
        ).fetchone()
        if row is not None:
            return self._variant_row_to_symbol(row)
        if DEFAULT_BUILD_VARIANT not in names:
            return None
        canonical = self._connection.execute(
            "SELECT * FROM symbols WHERE project_id = ? AND id = ?", (project_id, symbol_id)
        ).fetchone()
        return self._row_to_symbol(canonical) if canonical else None

    def put_symbols(self, symbols: Iterable[CodeSymbol]) -> None:
        project_id = self._project_id()
        with self._connection:
            for symbol in symbols:
                self._put_symbol(project_id, symbol)

    def symbols(
        self,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> tuple[CodeSymbol, ...]:
        try:
            project_id = self._project_id(project_root)
        except KeyError:
            return ()
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM symbol_variants
            WHERE project_id = ? AND build_variant IN ({placeholders})
            ORDER BY json_extract(snapshot_json, '$.qualified_name'), build_variant, id
            """,
            (project_id, *names),
        )
        result = [self._variant_row_to_symbol(row) for row in rows]
        if DEFAULT_BUILD_VARIANT in names:
            result.extend(
                self._row_to_symbol(row)
                for row in self._connection.execute(
                    """
                    SELECT * FROM symbols WHERE project_id = ? AND NOT EXISTS (
                        SELECT 1 FROM symbol_variants variants
                        WHERE variants.project_id = symbols.project_id
                          AND variants.symbol_id = symbols.id
                    ) ORDER BY qualified_name, id
                    """,
                    (project_id,),
                )
            )
        return tuple(result)

    def _scope_names(self, scope: BuildScope | tuple[str, ...] | None) -> tuple[str, ...]:
        if scope is None:
            return self.build_scope.variants
        if isinstance(scope, BuildScope):
            return scope.variants
        return BuildScope(scope).variants

    @staticmethod
    def _row_to_symbol(row: sqlite3.Row) -> CodeSymbol:
        return CodeSymbol(
            id=row["id"],
            qualified_name=row["qualified_name"],
            kind=SymbolKind(row["kind"]),
            span=SourceSpan(
                path=Path(row["path"]),
                start_line=row["start_line"],
                end_line=row["end_line"],
                start_column=row["start_column"],
                end_column=row["end_column"],
            ),
            signature=row["signature"],
            documentation=row["documentation"],
            source_hash=row["source_hash"],
            source_text=row["source_text"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=DEFAULT_BUILD_VARIANT,
            metadata=json.loads(row["metadata_json"]),
        )

    @classmethod
    def _variant_row_to_symbol(cls, row: sqlite3.Row) -> CodeSymbol:
        symbol = cls._snapshot_symbol(row["snapshot_json"])
        return CodeSymbol(
            id=symbol.id,
            qualified_name=symbol.qualified_name,
            kind=symbol.kind,
            span=symbol.span,
            signature=symbol.signature,
            documentation=symbol.documentation,
            source_hash=symbol.source_hash,
            source_text=symbol.source_text,
            build_configuration_id=row["build_configuration_id"],
            translation_unit_id=row["translation_unit_id"],
            build_variant=row["build_variant"],
            variant_id=row["id"],
            metadata=symbol.metadata,
        )

    def cfg_graphs(
        self,
        function_symbol_id: str | None = None,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> BoundedCfgResult[CfgGraph]:
        """Return build-specific function CFGs in stable order with explicit truncation."""

        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        function_sql = ""
        parameters: list[object] = [project_id, *names]
        if function_symbol_id is not None:
            function_sql = " AND function_symbol_id = ?"
            parameters.append(function_symbol_id)
        parameters.append(limit + 1)
        rows = self._connection.execute(
            f"""
            SELECT * FROM cfg_graphs
            WHERE project_id = ? AND build_variant IN ({placeholders}){function_sql}
            ORDER BY build_variant, build_configuration_id, translation_unit_id,
                     function_symbol_id, id
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_cfg_graph(row) for row in rows[:limit]), len(rows) > limit
        )

    def callsites(
        self,
        owner_symbol_id: str | None = None,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> BoundedCfgResult[CallSite]:
        """Return compiler callsites in stable order with explicit truncation."""

        limit = _call_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        owner_sql = ""
        parameters: list[object] = [project_id, *names]
        if owner_symbol_id is not None:
            owner_sql = " AND owner_symbol_id = ?"
            parameters.append(owner_symbol_id)
        parameters.append(limit + 1)
        rows = self._connection.execute(
            f"""
            SELECT * FROM callsites
            WHERE project_id = ? AND build_variant IN ({placeholders}){owner_sql}
            ORDER BY build_variant, build_configuration_id, translation_unit_id,
                     json_extract(expansion_span_json, '$.path'),
                     json_extract(expansion_span_json, '$.start_line'),
                     json_extract(expansion_span_json, '$.start_column'), id
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_callsite(row) for row in rows[:limit]), len(rows) > limit
        )

    def get_callsite(
        self,
        callsite_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> CallSite | None:
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        row = self._connection.execute(
            f"""
            SELECT * FROM callsites
            WHERE project_id = ? AND id = ? AND build_variant IN ({placeholders})
            """,
            (project_id, callsite_id, *names),
        ).fetchone()
        return self._row_to_callsite(row) if row else None

    def call_targets(
        self,
        callsite_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> BoundedCfgResult[CallTarget]:
        limit = _call_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM call_targets
            WHERE project_id = ? AND callsite_id = ?
              AND build_variant IN ({placeholders})
            ORDER BY CASE certainty WHEN 'certain' THEN 0 ELSE 1 END,
                     confidence DESC, target_symbol_id, derivation, id
            LIMIT ?
            """,
            (project_id, callsite_id, *names, limit + 1),
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_call_target(row) for row in rows[:limit]), len(rows) > limit
        )

    @staticmethod
    def _row_to_callsite(row: sqlite3.Row) -> CallSite:
        stack: list[MacroExpansionFrame] = []
        for frame in json.loads(row["expansion_stack_json"]):
            stack.append(
                MacroExpansionFrame(
                    macro_symbol_id=frame["macro_symbol_id"],
                    name=frame["name"],
                    spelling_span=_span_from_payload(frame["spelling_span"]),
                    expansion_span=_span_from_payload(frame["expansion_span"]),
                )
            )
        return CallSite(
            id=row["id"],
            owner_symbol_id=row["owner_symbol_id"],
            dispatch_kind=CallDispatchKind(row["dispatch_kind"]),
            spelling_span=_required_span_from_json(row["spelling_span_json"]),
            expansion_span=_required_span_from_json(row["expansion_span_json"]),
            target_set_complete=bool(row["target_set_complete"]),
            static_target_symbol_id=row["static_target_symbol_id"],
            unresolved_reason=row["unresolved_reason"],
            callee_text=row["callee_text"],
            expansion_stack=tuple(stack),
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_call_target(row: sqlite3.Row) -> CallTarget:
        return CallTarget(
            id=row["id"],
            callsite_id=row["callsite_id"],
            target_symbol_id=row["target_symbol_id"],
            certainty=CallTargetCertainty(row["certainty"]),
            confidence=row["confidence"],
            confidence_reason=row["confidence_reason"],
            derivation=row["derivation"],
            evidence_span=_required_span_from_json(row["evidence_span_json"]),
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    def get_cfg_graph(
        self,
        graph_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> CfgGraph | None:
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        row = self._connection.execute(
            f"""
            SELECT * FROM cfg_graphs
            WHERE project_id = ? AND id = ? AND build_variant IN ({placeholders})
            """,
            (project_id, graph_id, *names),
        ).fetchone()
        return self._row_to_cfg_graph(row) if row else None

    def cfg_blocks(
        self,
        graph_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 1_000,
    ) -> BoundedCfgResult[CfgBlock]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM cfg_blocks
            WHERE project_id = ? AND graph_id = ? AND build_variant IN ({placeholders})
            ORDER BY block_index, id LIMIT ?
            """,
            (project_id, graph_id, *names, limit + 1),
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_cfg_block(row) for row in rows[:limit]), len(rows) > limit
        )

    def cfg_elements(
        self,
        graph_id: str,
        project_root: Path | None = None,
        *,
        block_id: str | None = None,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 2_000,
    ) -> BoundedCfgResult[CfgElement]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        block_sql = ""
        parameters: list[object] = [project_id, graph_id, *names]
        if block_id is not None:
            block_sql = " AND elements.block_id = ?"
            parameters.append(block_id)
        parameters.append(limit + 1)
        rows = self._connection.execute(
            f"""
            SELECT elements.* FROM cfg_elements elements
            JOIN cfg_blocks blocks
              ON blocks.project_id = elements.project_id AND blocks.id = elements.block_id
            WHERE elements.project_id = ? AND elements.graph_id = ?
              AND elements.build_variant IN ({placeholders}){block_sql}
            ORDER BY blocks.block_index, elements.element_index, elements.id LIMIT ?
            """,
            parameters,
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_cfg_element(row) for row in rows[:limit]), len(rows) > limit
        )

    def cfg_edges(
        self,
        graph_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 2_000,
    ) -> BoundedCfgResult[CfgEdge]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT edges.* FROM cfg_edges edges
            JOIN cfg_blocks source
              ON source.project_id = edges.project_id AND source.id = edges.source_block_id
            JOIN cfg_blocks target
              ON target.project_id = edges.project_id AND target.id = edges.target_block_id
            WHERE edges.project_id = ? AND edges.graph_id = ?
              AND edges.build_variant IN ({placeholders})
            ORDER BY source.block_index, edges.successor_index, target.block_index,
                     edges.kind, edges.feasible DESC, edges.id
            LIMIT ?
            """,
            (project_id, graph_id, *names, limit + 1),
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_cfg_edge(row) for row in rows[:limit]), len(rows) > limit
        )

    @staticmethod
    def _row_to_cfg_graph(row: sqlite3.Row) -> CfgGraph:
        return CfgGraph(
            id=row["id"],
            function_symbol_id=row["function_symbol_id"],
            entry_block_id=row["entry_block_id"],
            normal_exit_block_id=row["normal_exit_block_id"],
            exceptional_exit_block_id=row["exceptional_exit_block_id"],
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
            clang_major=row["clang_major"],
            fact_schema_version=row["fact_schema_version"],
            build_options=json.loads(row["build_options_json"]),
        )

    @staticmethod
    def _row_to_cfg_block(row: sqlite3.Row) -> CfgBlock:
        return CfgBlock(
            id=row["id"],
            graph_id=row["graph_id"],
            index=row["block_index"],
            role=CfgBlockRole(row["role"]),
            reachable=bool(row["reachable"]),
            terminator_kind=row["terminator_kind"],
            terminator_text=row["terminator_text"],
            terminator_spelling_span=_span_from_json(row["terminator_spelling_span_json"]),
            terminator_expansion_span=_span_from_json(row["terminator_expansion_span_json"]),
            label_kind=row["label_kind"],
            label_text=row["label_text"],
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_cfg_element(row: sqlite3.Row) -> CfgElement:
        return CfgElement(
            id=row["id"],
            graph_id=row["graph_id"],
            block_id=row["block_id"],
            index=row["element_index"],
            kind=row["kind"],
            statement_class=row["statement_class"],
            text=row["text"],
            spelling_span=_span_from_json(row["spelling_span_json"]),
            expansion_span=_span_from_json(row["expansion_span_json"]),
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _row_to_cfg_edge(row: sqlite3.Row) -> CfgEdge:
        return CfgEdge(
            id=row["id"],
            graph_id=row["graph_id"],
            source_block_id=row["source_block_id"],
            target_block_id=row["target_block_id"],
            kind=CfgEdgeKind(row["kind"]),
            successor_index=row["successor_index"],
            feasible=bool(row["feasible"]),
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    def occurrences(
        self,
        symbol_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> tuple[SymbolOccurrence, ...]:
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        return tuple(
            SymbolOccurrence(
                id=row["id"],
                symbol_id=row["symbol_id"],
                enclosing_symbol_id=row["enclosing_symbol_id"],
                kind=OccurrenceKind(row["kind"]),
                span=SourceSpan(
                    Path(row["path"]),
                    row["start_line"],
                    row["end_line"],
                    row["start_column"],
                    row["end_column"],
                ),
                translation_unit_id=row["translation_unit_id"],
                build_configuration_id=row["build_configuration_id"],
                build_variant=row["build_variant"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in self._connection.execute(
                f"""
                SELECT * FROM occurrences
                WHERE project_id = ? AND symbol_id = ?
                  AND build_variant IN ({placeholders})
                ORDER BY build_variant, path, start_line, start_column
                """,
                (project_id, symbol_id, *names),
            )
        )

    def search(
        self,
        query: SearchQuery,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> Sequence[SearchHit]:
        project_id = self._project_id(project_root)
        terms = re.findall(r"[\w:]+", query.text, flags=re.UNICODE)
        if not terms:
            return ()
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT variants.*,
                   bm25(symbol_variant_fts, 0.0, 0.0, 0.0, 0.0, 8.0, 4.0, 2.0, 1.0)
                       AS rank
            FROM symbol_variant_fts
            JOIN symbol_variants variants
              ON variants.project_id = CAST(symbol_variant_fts.project_id AS INTEGER)
             AND variants.id = symbol_variant_fts.variant_id
            WHERE symbol_variant_fts MATCH ? AND variants.project_id = ?
              AND variants.build_variant IN ({placeholders})
            ORDER BY rank, variants.build_variant, variants.id
            LIMIT ?
            """,
            (expression, project_id, *names, query.limit),
        ).fetchall()
        hits = [
            SearchHit(
                symbol=self._variant_row_to_symbol(row),
                score=-float(row["rank"]),
                source="fts5",
            )
            for row in rows
        ]
        if DEFAULT_BUILD_VARIANT in names and len(hits) < query.limit:
            hits.extend(
                self._standalone_search(
                    expression, project_id, query.limit - len(hits), symbols_only=False
                )
            )
        return tuple(hits[: query.limit])

    def search_symbols(
        self,
        query: SearchQuery,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> Sequence[SearchHit]:
        """Search only compiler-resolved names and signatures through FTS5."""

        project_id = self._project_id(project_root)
        terms = re.findall(r"[\w:]+", query.text, flags=re.UNICODE)
        if not terms:
            return ()
        escaped = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
        expression = (
            f"qualified_name : ({' OR '.join(escaped)}) OR signature : ({' OR '.join(escaped)})"
        )
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT variants.*,
                   bm25(symbol_variant_fts, 0.0, 0.0, 0.0, 0.0, 12.0, 6.0, 0.0, 0.0)
                       AS rank
            FROM symbol_variant_fts
            JOIN symbol_variants variants
              ON variants.project_id = CAST(symbol_variant_fts.project_id AS INTEGER)
             AND variants.id = symbol_variant_fts.variant_id
            WHERE symbol_variant_fts MATCH ? AND variants.project_id = ?
              AND variants.build_variant IN ({placeholders})
            ORDER BY rank, variants.build_variant, variants.id
            LIMIT ?
            """,
            (expression, project_id, *names, query.limit),
        ).fetchall()
        query_text = query.text.casefold().strip()
        hits = [
            SearchHit(
                symbol=self._variant_row_to_symbol(row),
                score=-float(row["rank"])
                + (
                    2.0
                    if self._variant_row_to_symbol(row).qualified_name.casefold() == query_text
                    else 0.0
                )
                + (
                    1.0
                    if self._variant_row_to_symbol(row)
                    .qualified_name.casefold()
                    .endswith(f"::{query_text}")
                    else 0.0
                ),
                source="sqlite-symbol",
            )
            for row in rows
        ]
        if DEFAULT_BUILD_VARIANT in names and len(hits) < query.limit:
            hits.extend(
                self._standalone_search(
                    expression, project_id, query.limit - len(hits), symbols_only=True
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.symbol.qualified_name, hit.symbol.id))
        return tuple(hits[: query.limit])

    def _standalone_search(
        self, expression: str, project_id: int, limit: int, *, symbols_only: bool
    ) -> list[SearchHit]:
        weights = "12.0, 6.0, 0.0, 0.0" if symbols_only else "8.0, 4.0, 2.0, 1.0"
        rows = self._connection.execute(
            f"""
            SELECT symbols.*, bm25(symbol_fts, 0.0, 0.0, {weights}) AS rank
            FROM symbol_fts JOIN symbols
              ON symbols.project_id = CAST(symbol_fts.project_id AS INTEGER)
             AND symbols.id = symbol_fts.symbol_id
            WHERE symbol_fts MATCH ? AND symbols.project_id = ? AND NOT EXISTS (
                SELECT 1 FROM symbol_variants variants
                WHERE variants.project_id = symbols.project_id
                  AND variants.symbol_id = symbols.id
            ) ORDER BY rank, symbols.qualified_name LIMIT ?
            """,
            (expression, project_id, limit),
        )
        return [
            SearchHit(
                self._row_to_symbol(row),
                -float(row["rank"]),
                "sqlite-symbol" if symbols_only else "fts5",
            )
            for row in rows
        ]

    def put_edges(self, edges: Iterable[GraphEdge]) -> None:
        project_id = self._project_id()
        with self._connection:
            self._put_edges(project_id, edges)

    def neighbors(
        self,
        symbol_id: str,
        *,
        relations: frozenset[GraphRelation] | None = None,
        depth: int = 1,
        direction: GraphDirection = GraphDirection.BOTH,
        max_edges: int | None = None,
        per_node_limit: int | None = None,
        project_root: Path | None = None,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> Sequence[GraphEdge]:
        if depth < 1:
            raise ValueError("graph depth must be at least one")
        if max_edges is not None and max_edges < 1:
            raise ValueError("graph edge limit must be at least one")
        if per_node_limit is not None and per_node_limit < 1:
            raise ValueError("per-node graph limit must be at least one")
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        scope_placeholders = ",".join("?" for _ in names)
        frontier = deque([(symbol_id, 0)])
        visited_symbols = {symbol_id}
        found: dict[str, GraphEdge] = {}
        while frontier:
            if max_edges is not None and len(found) >= max_edges:
                break
            current, level = frontier.popleft()
            if level >= depth:
                continue
            if direction == GraphDirection.OUTGOING:
                endpoint_sql = "source_id = ?"
                parameters: list[object] = [project_id, *names, current]
            elif direction == GraphDirection.INCOMING:
                endpoint_sql = "target_id = ?"
                parameters = [project_id, *names, current]
            else:
                endpoint_sql = "(source_id = ? OR target_id = ?)"
                parameters = [project_id, *names, current, current]
            relation_sql = ""
            if relations:
                placeholders = ",".join("?" for _ in relations)
                relation_sql = f" AND relation IN ({placeholders})"
                parameters.extend(relation.value for relation in sorted(relations, key=str))
            limit_sql = ""
            if per_node_limit is not None:
                limit_sql = " LIMIT ?"
                parameters.append(per_node_limit)
            rows = self._connection.execute(
                (
                    "SELECT id, translation_unit_id, build_configuration_id, build_variant, "
                    "source_id, target_id, relation FROM edges "
                    f"WHERE project_id = ? AND build_variant IN ({scope_placeholders}) "
                    f"AND {endpoint_sql}"
                    + relation_sql
                    + " ORDER BY relation, source_id, target_id, build_variant, id"
                    + limit_sql
                ),
                parameters,
            )
            for row in rows:
                relation = GraphRelation(row["relation"])
                edge_id = row["id"]
                found[edge_id] = GraphEdge(
                    row["source_id"],
                    row["target_id"],
                    relation,
                    row["translation_unit_id"],
                    edge_id,
                    row["build_configuration_id"],
                    row["build_variant"],
                )
                if max_edges is not None and len(found) >= max_edges:
                    break
                adjacent = (
                    row["target_id"]
                    if direction == GraphDirection.OUTGOING
                    else row["source_id"]
                    if direction == GraphDirection.INCOMING
                    else (row["target_id"] if row["source_id"] == current else row["source_id"])
                )
                if adjacent not in visited_symbols:
                    visited_symbols.add(adjacent)
                    frontier.append((adjacent, level + 1))
        return tuple(found.values())

    def put_embedding(
        self,
        symbol_id: str,
        model: str,
        vector: Sequence[float],
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> None:
        project_id = self._project_id(project_root)
        symbol = self.get_symbol(symbol_id, project_root, build_scope=build_scope)
        if symbol is None or not symbol.variant_id:
            raise KeyError(f"symbol variant is not indexed: {symbol_id}")
        variant_id = symbol.variant_id
        normalized = _validate_vector(vector)
        dimensions = {
            row[0]
            for row in self._connection.execute(
                """
                SELECT DISTINCT dimensions FROM variant_embeddings
                WHERE project_id = ? AND model = ?
                """,
                (project_id, model),
            )
        }
        if dimensions and dimensions != {len(normalized)}:
            raise ValueError(
                f"embedding model {model!r} already uses dimension {next(iter(dimensions))}, "
                f"not {len(normalized)}"
            )
        magnitude = math.sqrt(sum(value * value for value in normalized))
        encoded = struct.pack(f"<{len(normalized)}d", *normalized)
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO variant_embeddings(
                    project_id, variant_id, model, dimensions, magnitude, vector
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, variant_id, model) DO UPDATE SET
                    dimensions = excluded.dimensions,
                    magnitude = excluded.magnitude,
                    vector = excluded.vector
                """,
                (project_id, variant_id, model, len(normalized), magnitude, encoded),
            )

    def missing_embedding_symbol_ids(
        self,
        model: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """Return stable IDs whose current source snapshot has no vector for ``model``."""

        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        return tuple(
            dict.fromkeys(
                row[0]
                for row in self._connection.execute(
                    f"""
                    SELECT variants.symbol_id FROM symbol_variants variants
                    LEFT JOIN variant_embeddings embeddings
                      ON embeddings.project_id = variants.project_id
                     AND embeddings.variant_id = variants.id
                     AND embeddings.model = ?
                    WHERE variants.project_id = ?
                      AND variants.build_variant IN ({placeholders})
                      AND embeddings.variant_id IS NULL
                    ORDER BY variants.build_variant, variants.symbol_id
                    """,
                    (model, project_id, *names),
                )
            )
        )

    def missing_embedding_variant_ids(
        self,
        model: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        return tuple(
            row[0]
            for row in self._connection.execute(
                f"""
                SELECT variants.id FROM symbol_variants variants
                LEFT JOIN variant_embeddings embeddings
                  ON embeddings.project_id = variants.project_id
                 AND embeddings.variant_id = variants.id
                 AND embeddings.model = ?
                WHERE variants.project_id = ?
                  AND variants.build_variant IN ({placeholders})
                  AND embeddings.variant_id IS NULL
                ORDER BY variants.build_variant, variants.id
                """,
                (model, project_id, *names),
            )
        )

    def embedding_count(self, model: str, project_root: Path | None = None) -> int:
        project_id = self._project_id(project_root)
        return int(
            self._connection.execute(
                "SELECT count(*) FROM variant_embeddings WHERE project_id = ? AND model = ?",
                (project_id, model),
            ).fetchone()[0]
        )

    def search_vector(
        self,
        vector: Sequence[float],
        *,
        model: str,
        limit: int = 20,
        project_root: Path | None = None,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> Sequence[SearchHit]:
        if limit <= 0:
            raise ValueError("vector search limit must be greater than zero")
        query_vector = _validate_vector(vector)
        query_magnitude = math.sqrt(sum(value * value for value in query_vector))
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        dimensions = {
            row[0]
            for row in self._connection.execute(
                f"""
                SELECT DISTINCT dimensions FROM variant_embeddings embeddings
                JOIN symbol_variants variants
                  ON variants.project_id = embeddings.project_id
                 AND variants.id = embeddings.variant_id
                WHERE embeddings.project_id = ? AND embeddings.model = ?
                  AND variants.build_variant IN ({placeholders})
                """,
                (project_id, model, *names),
            )
        }
        if dimensions and dimensions != {len(query_vector)}:
            raise ValueError(
                f"query dimension {len(query_vector)} does not match model {model!r} "
                f"dimension {next(iter(dimensions))}"
            )
        rows = self._connection.execute(
            f"""
            SELECT embeddings.*, variants.*
            FROM variant_embeddings embeddings
            JOIN symbol_variants variants
              ON variants.project_id = embeddings.project_id
             AND variants.id = embeddings.variant_id
            WHERE embeddings.project_id = ? AND embeddings.model = ?
              AND embeddings.dimensions = ?
              AND variants.build_variant IN ({placeholders})
            """,
            (project_id, model, len(query_vector), *names),
        )
        scored: list[SearchHit] = []
        for row in rows:
            candidate = struct.unpack(f"<{row['dimensions']}d", row["vector"])
            dot_product = sum(
                left * right for left, right in zip(query_vector, candidate, strict=True)
            )
            cosine = dot_product / (query_magnitude * row["magnitude"])
            scored.append(SearchHit(self._variant_row_to_symbol(row), cosine, f"vector:{model}"))
        scored.sort(key=lambda hit: (-hit.score, hit.symbol.build_variant, hit.symbol.variant_id))
        return tuple(scored[:limit])


def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values:
        raise ValueError("embedding vector must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding vector must contain only finite values")
    if not any(value != 0.0 for value in values):
        raise ValueError("embedding vector magnitude must be greater than zero")
    return values


def _cfg_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_CFG_PAGE_SIZE:
        raise ValueError(f"CFG page limit must be between 1 and {MAX_CFG_PAGE_SIZE}")
    return limit


def _call_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_CALL_PAGE_SIZE:
        raise ValueError(f"call page limit must be between 1 and {MAX_CALL_PAGE_SIZE}")
    return limit


def _span_payload(span: SourceSpan) -> dict[str, object]:
    return {
        "path": str(span.path),
        "start_line": span.start_line,
        "end_line": span.end_line,
        "start_column": span.start_column,
        "end_column": span.end_column,
    }


def _span_json(span: SourceSpan | None) -> str | None:
    if span is None:
        return None
    return json.dumps(
        {
            "path": str(span.path),
            "start_line": span.start_line,
            "end_line": span.end_line,
            "start_column": span.start_column,
            "end_column": span.end_column,
        },
        sort_keys=True,
    )


def _span_from_json(payload: str | None) -> SourceSpan | None:
    if payload is None:
        return None
    value = json.loads(payload)
    return _span_from_payload(value)


def _required_span_from_json(payload: str) -> SourceSpan:
    span = _span_from_json(payload)
    if span is None:  # pragma: no cover - NOT NULL storage invariant
        raise ValueError("required source span is missing")
    return span


def _span_from_payload(value: dict[str, object]) -> SourceSpan:
    return SourceSpan(
        Path(value["path"]),
        int(value["start_line"]),
        int(value["end_line"]),
        int(value["start_column"]),
        int(value["end_column"]),
    )


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:32]}"


def _execute_script(connection: sqlite3.Connection, script: str) -> None:
    """Execute a multi-statement script without sqlite3's implicit pre-script commit."""

    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statement = pending.strip()
            if statement:
                connection.execute(statement)
            pending = ""
    if pending.strip():
        raise ValueError("incomplete SQLite migration statement")
