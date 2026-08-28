"""Transactional SQLite persistence, FTS5, graph traversal, and vector search."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
import sqlite3
import struct
import zlib
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from cpp_context_engine.analysis.interprocedural import solve_interprocedural
from cpp_context_engine.models import (
    DEFAULT_BUILD_VARIANT,
    BoundedCfgResult,
    BuildScope,
    BuildVariant,
    CallArgumentBinding,
    CallDispatchKind,
    CallResultBinding,
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
    DataAccess,
    DataAccessKind,
    DataFlowAnalysis,
    DataFlowCertainty,
    DataFlowEvidence,
    DataFlowRelation,
    FunctionSummary,
    GraphDirection,
    GraphEdge,
    GraphRelation,
    InterproceduralFlow,
    InterproceduralFlowKind,
    MacroExpansionFrame,
    MemoryLocation,
    MemoryLocationKind,
    OccurrenceKind,
    SearchHit,
    SearchQuery,
    SourceSpan,
    SummaryEffect,
    SummaryEffectKind,
    SummaryReturnOrigin,
    SummaryReturnOriginKind,
    SymbolKind,
    SymbolOccurrence,
)

if TYPE_CHECKING:
    from cpp_context_engine.ingestion.protocols import IngestionBatch

SCHEMA_VERSION = 12
DEFAULT_EMBEDDING_TEXT_CHARS = 32_000
EMBEDDING_BATCH_SIZE = 128
MAX_CFG_PAGE_SIZE = 10_000
MAX_CALL_PAGE_SIZE = 10_000
SUMMARY_PAYLOAD_ENCODING = "zlib-json-v1"
MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_SUMMARY_PAYLOAD_RECORDS = 65_536
_TRANSLATION_UNIT_DELETE_ORDER = (
    "interprocedural_flows",
    "call_argument_bindings",
    "call_result_bindings",
    "summary_effects",
    "summary_return_origins",
    "data_flow_evidence",
    "call_targets",
    "data_accesses",
    "function_summaries",
    "memory_locations",
    "data_flow_analyses",
    "cfg_edges",
    "cfg_elements",
    "cfg_blocks",
    "cfg_graphs",
    "callsites",
    "occurrences",
    "edges",
    "symbol_variants",
    "translation_unit_symbols",
    "dependencies",
)


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


class SummaryPayloadError(RuntimeError):
    """A persisted propagated-summary payload is corrupt or exceeds hard limits."""


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
        self._connection.create_function(
            "_cpp_context_cosine", 4, _sqlite_cosine, deterministic=True
        )
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
        if current <= 6:
            self._migrate_v7()
        if current <= 7:
            self._migrate_v8()
        if current <= 8:
            self._migrate_v9()
        if current <= 9:
            self._migrate_v10()
        if current <= 10:
            self._migrate_v11_and_v12()
        elif current == 11:
            self._migrate_v12()

    def _migrate_v11_and_v12(self) -> None:
        """Upgrade v10 through both dependent schema steps as one transaction."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._migrate_v11(manage_transaction=False)
            self._migrate_v12(manage_transaction=False)
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

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
            # Pin this migration's boundary: a later migration may still roll back.
            self._connection.execute("PRAGMA user_version = 6")
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_v7(self) -> None:
        """Add bounded intraprocedural data-flow evidence atomically."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            _execute_script(
                self._connection,
                """
                CREATE TABLE data_flow_analyses (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    complete INTEGER NOT NULL,
                    incomplete_reasons_json TEXT NOT NULL,
                    iteration_count INTEGER NOT NULL,
                    max_iterations INTEGER NOT NULL,
                    max_alias_targets INTEGER NOT NULL,
                    max_access_path_depth INTEGER NOT NULL,
                    max_locations INTEGER NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (project_id, graph_id),
                    FOREIGN KEY (project_id, graph_id)
                        REFERENCES cfg_graphs(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE memory_locations (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type_name TEXT NOT NULL,
                    declaration_symbol_id TEXT,
                    base_location_id TEXT,
                    access_path_json TEXT NOT NULL,
                    is_volatile INTEGER NOT NULL,
                    is_atomic INTEGER NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY (project_id, analysis_id)
                        REFERENCES data_flow_analyses(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, graph_id)
                        REFERENCES cfg_graphs(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, declaration_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, base_location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE data_accesses (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    cfg_element_id TEXT,
                    location_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    span_json TEXT,
                    expression TEXT NOT NULL,
                    pointee_symbol_ids_json TEXT NOT NULL,
                    points_to_complete INTEGER NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY (project_id, analysis_id)
                        REFERENCES data_flow_analyses(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, graph_id)
                        REFERENCES cfg_graphs(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, block_id)
                        REFERENCES cfg_blocks(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, cfg_element_id)
                        REFERENCES cfg_elements(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE data_flow_evidence (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    certainty TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    source_access_id TEXT,
                    target_access_id TEXT,
                    source_location_id TEXT,
                    target_location_id TEXT,
                    evidence_span_json TEXT,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    CHECK (
                        (source_access_id IS NOT NULL AND target_access_id IS NOT NULL
                         AND source_location_id IS NULL AND target_location_id IS NULL)
                        OR
                        (source_access_id IS NULL AND target_access_id IS NULL
                         AND source_location_id IS NOT NULL AND target_location_id IS NOT NULL)
                    ),
                    FOREIGN KEY (project_id, analysis_id)
                        REFERENCES data_flow_analyses(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, graph_id)
                        REFERENCES cfg_graphs(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, source_access_id)
                        REFERENCES data_accesses(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, target_access_id)
                        REFERENCES data_accesses(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, source_location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, target_location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE INDEX data_flow_analyses_scope
                    ON data_flow_analyses(project_id, build_variant, graph_id);
                CREATE INDEX memory_locations_analysis_order
                    ON memory_locations(project_id, analysis_id, kind, name, id);
                CREATE INDEX data_accesses_analysis_order
                    ON data_accesses(project_id, analysis_id, block_id, sequence, id);
                CREATE INDEX data_flow_evidence_analysis_order
                    ON data_flow_evidence(project_id, analysis_id, relation, id);
                """,
            )
            # Protocol-v3 native rows have no def-use or points-to facts and must be
            # refreshed through the normal incremental path after this migration.
            self._connection.execute(
                "UPDATE translation_units SET advanced_facts_complete = 0 "
                "WHERE analysis_backend = 'clang-libtooling'"
            )
            self._connection.execute("PRAGMA user_version = 7")
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_v8(self) -> None:
        """Add body-variant summaries and cross-call evidence atomically."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            _execute_script(
                self._connection,
                """
                CREATE TABLE function_summaries (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    function_symbol_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    parameter_modes_json TEXT NOT NULL,
                    parameter_location_ids_json TEXT NOT NULL,
                    local_complete INTEGER NOT NULL,
                    local_incomplete_reasons_json TEXT NOT NULL,
                    complete INTEGER NOT NULL,
                    incomplete_reasons_json TEXT NOT NULL,
                    recursive INTEGER NOT NULL,
                    iteration_count INTEGER NOT NULL,
                    max_scc_iterations INTEGER NOT NULL,
                    max_scc_size INTEGER NOT NULL,
                    max_summary_effects INTEGER NOT NULL,
                    solution_hash TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (project_id, graph_id),
                    FOREIGN KEY (project_id, function_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, graph_id)
                        REFERENCES cfg_graphs(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, analysis_id)
                        REFERENCES data_flow_analyses(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, translation_unit_id)
                        REFERENCES translation_units(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE summary_effects (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    summary_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    location_kind TEXT NOT NULL,
                    certainty TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    parameter_index INTEGER,
                    access_path_json TEXT NOT NULL,
                    location_id TEXT,
                    source_access_id TEXT,
                    is_local INTEGER NOT NULL,
                    via_callsite_id TEXT,
                    target_symbol_id TEXT,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY (project_id, summary_id)
                        REFERENCES function_summaries(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, source_access_id)
                        REFERENCES data_accesses(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, via_callsite_id)
                        REFERENCES callsites(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, target_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE summary_return_origins (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    summary_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    certainty TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    location_kind TEXT,
                    parameter_index INTEGER,
                    access_path_json TEXT NOT NULL,
                    location_id TEXT,
                    callsite_id TEXT,
                    is_local INTEGER NOT NULL,
                    via_callsite_id TEXT,
                    target_symbol_id TEXT,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY (project_id, summary_id)
                        REFERENCES function_summaries(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, callsite_id)
                        REFERENCES callsites(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, via_callsite_id)
                        REFERENCES callsites(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, target_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE call_argument_bindings (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    caller_summary_id TEXT NOT NULL,
                    callsite_id TEXT NOT NULL,
                    argument_index INTEGER NOT NULL,
                    location_id TEXT,
                    location_kind TEXT NOT NULL,
                    parameter_index INTEGER,
                    access_path_json TEXT NOT NULL,
                    writeback_candidate INTEGER NOT NULL,
                    complete INTEGER NOT NULL,
                    incomplete_reason TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (project_id, caller_summary_id, callsite_id, argument_index),
                    FOREIGN KEY (project_id, caller_summary_id)
                        REFERENCES function_summaries(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, callsite_id)
                        REFERENCES callsites(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE call_result_bindings (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    caller_summary_id TEXT NOT NULL,
                    callsite_id TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    definition_access_id TEXT NOT NULL,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    UNIQUE (project_id, caller_summary_id, callsite_id),
                    FOREIGN KEY (project_id, caller_summary_id)
                        REFERENCES function_summaries(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, callsite_id)
                        REFERENCES callsites(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, definition_access_id)
                        REFERENCES data_accesses(project_id, id) ON DELETE CASCADE
                );

                CREATE TABLE interprocedural_flows (
                    project_id INTEGER NOT NULL,
                    id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    caller_summary_id TEXT NOT NULL,
                    callee_summary_id TEXT NOT NULL,
                    callsite_id TEXT NOT NULL,
                    target_symbol_id TEXT NOT NULL,
                    target_certainty TEXT NOT NULL,
                    certainty TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    argument_index INTEGER,
                    caller_location_id TEXT,
                    callee_location_id TEXT,
                    caller_access_id TEXT,
                    translation_unit_id TEXT NOT NULL,
                    build_configuration_id TEXT NOT NULL,
                    build_variant TEXT NOT NULL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY (project_id, caller_summary_id)
                        REFERENCES function_summaries(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, callee_summary_id)
                        REFERENCES function_summaries(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, callsite_id)
                        REFERENCES callsites(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, target_symbol_id)
                        REFERENCES symbols(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, caller_location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, callee_location_id)
                        REFERENCES memory_locations(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id, caller_access_id)
                        REFERENCES data_accesses(project_id, id) ON DELETE CASCADE
                );

                CREATE INDEX function_summaries_scope
                    ON function_summaries(project_id, build_variant, function_symbol_id, id);
                CREATE INDEX summary_effects_summary_order
                    ON summary_effects(project_id, summary_id, is_local DESC, kind, id);
                CREATE INDEX summary_return_origins_summary_order
                    ON summary_return_origins(project_id, summary_id, is_local DESC, kind, id);
                CREATE INDEX call_argument_bindings_site_order
                    ON call_argument_bindings(project_id, callsite_id, argument_index, id);
                CREATE INDEX interprocedural_flows_caller_order
                    ON interprocedural_flows(project_id, caller_summary_id, kind, id);
                CREATE INDEX interprocedural_flows_callee_order
                    ON interprocedural_flows(project_id, callee_summary_id, kind, id);
                """,
            )
            self._connection.execute(
                "UPDATE translation_units SET advanced_facts_complete = 0 "
                "WHERE analysis_backend = 'clang-libtooling'"
            )
            self._connection.execute("PRAGMA user_version = 8")
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_v9(self) -> None:
        """Index symbol refresh and translation-unit cascade lookup prefixes."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            indexes = (
                (
                    "symbol_variants_symbol_preference",
                    "symbol_variants",
                    "project_id, symbol_id, is_definition DESC, build_variant, translation_unit_id",
                ),
                ("edges_tu", "edges", "project_id, translation_unit_id"),
                ("symbol_variants_tu", "symbol_variants", "project_id, translation_unit_id"),
                ("cfg_blocks_tu", "cfg_blocks", "project_id, translation_unit_id"),
                ("cfg_elements_tu", "cfg_elements", "project_id, translation_unit_id"),
                ("cfg_edges_tu", "cfg_edges", "project_id, translation_unit_id"),
                ("call_targets_tu", "call_targets", "project_id, translation_unit_id"),
                (
                    "data_flow_analyses_tu",
                    "data_flow_analyses",
                    "project_id, translation_unit_id",
                ),
                (
                    "memory_locations_tu",
                    "memory_locations",
                    "project_id, translation_unit_id",
                ),
                ("data_accesses_tu", "data_accesses", "project_id, translation_unit_id"),
                (
                    "data_flow_evidence_tu",
                    "data_flow_evidence",
                    "project_id, translation_unit_id",
                ),
                (
                    "function_summaries_tu",
                    "function_summaries",
                    "project_id, translation_unit_id",
                ),
            )
            tables = {
                row[0]
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            for name, table, columns in indexes:
                if table not in tables:
                    continue
                self._connection.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({columns})")
            # Parent deletes consult every child key. A missing prefix index turns
            # changed-TU cascades into repeated full-table scans.
            for table in sorted(tables):
                foreign_keys: dict[int, list[tuple[int, str]]] = {}
                for row in self._connection.execute(f"PRAGMA foreign_key_list({table})"):
                    foreign_keys.setdefault(int(row[0]), []).append((int(row[1]), str(row[3])))
                index_columns = [
                    tuple(
                        str(column[2])
                        for column in self._connection.execute(f"PRAGMA index_info({index[1]})")
                    )
                    for index in self._connection.execute(f"PRAGMA index_list({table})")
                ]
                for key in foreign_keys.values():
                    columns = tuple(column for _, column in sorted(key))
                    if any(candidate[: len(columns)] == columns for candidate in index_columns):
                        continue
                    suffix = "_".join(columns)
                    name = f"fk_lookup_{table}_{suffix}"
                    if not all(
                        value.replace("_", "").isalnum() for value in (table, name, *columns)
                    ):
                        raise RuntimeError("schema contains an unsafe foreign-key identifier")
                    joined = ", ".join(columns)
                    self._connection.execute(
                        f"CREATE INDEX IF NOT EXISTS {name} ON {table}({joined})"
                    )
                    index_columns.append(columns)
            self._connection.execute("PRAGMA user_version = 9")
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_v10(self) -> None:
        """Remove the obsolete duplicate symbol snapshot from TU membership rows."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            table = self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'translation_unit_symbols'"
            ).fetchone()
            if table is not None:
                columns = {
                    row[1]
                    for row in self._connection.execute(
                        "PRAGMA table_info(translation_unit_symbols)"
                    )
                }
                if "snapshot_json" in columns:
                    self._connection.execute(
                        "ALTER TABLE translation_unit_symbols DROP COLUMN snapshot_json"
                    )
            self._connection.execute("PRAGMA user_version = 10")
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_v12(self, *, manage_transaction: bool = True) -> None:
        """Content-address vectors while retaining variant-scoped search rows."""

        try:
            if manage_transaction:
                self._connection.execute("BEGIN IMMEDIATE")
            columns = {
                row[1] for row in self._connection.execute("PRAGMA table_info(variant_embeddings)")
            }
            if "configuration_id" in columns:
                # Migration tests may rewind user_version while retaining newer,
                # independently unrelated tables. Keep that already-normalized data.
                self._connection.execute("PRAGMA user_version = 12")
                if manage_transaction:
                    self._connection.commit()
                return
            has_legacy_embeddings = bool(columns)
            if has_legacy_embeddings:
                self._connection.execute(
                    "ALTER TABLE variant_embeddings RENAME TO variant_embeddings_v10"
                )
            _execute_script(
                self._connection,
                """
                CREATE TABLE embedding_vectors (
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    configuration_id TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    magnitude REAL NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (
                        project_id, model, configuration_id, dimensions, content_hash
                    )
                );

                CREATE TABLE variant_embeddings (
                    project_id INTEGER NOT NULL,
                    variant_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    configuration_id TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    PRIMARY KEY (project_id, variant_id, model, configuration_id),
                    FOREIGN KEY (project_id, variant_id)
                        REFERENCES symbol_variants(project_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (
                        project_id, model, configuration_id, dimensions, content_hash
                    )
                        REFERENCES embedding_vectors(
                            project_id, model, configuration_id, dimensions, content_hash
                        )
                );

                CREATE INDEX variant_embeddings_search
                    ON variant_embeddings(
                        project_id, model, configuration_id, dimensions, variant_id
                    );
                CREATE INDEX variant_embeddings_content
                    ON variant_embeddings(
                        project_id, model, configuration_id, dimensions, content_hash
                    );
                """,
            )
            if not has_legacy_embeddings:
                self._connection.execute("PRAGMA user_version = 12")
                if manage_transaction:
                    self._connection.commit()
                return
            cursor = self._connection.execute(
                """
                SELECT embeddings.project_id, embeddings.variant_id, embeddings.model,
                       embeddings.dimensions, embeddings.magnitude, embeddings.vector,
                       variants.snapshot_json
                FROM variant_embeddings_v10 embeddings
                JOIN symbol_variants variants
                  ON variants.project_id = embeddings.project_id
                 AND variants.id = embeddings.variant_id
                ORDER BY embeddings.project_id, embeddings.model,
                         embeddings.dimensions, embeddings.variant_id
                """
            )
            while rows := cursor.fetchmany(EMBEDDING_BATCH_SIZE):
                for row in rows:
                    if str(row["model"]).startswith("openai-compatible:"):
                        # v10 did not persist the hosted endpoint, so those vectors
                        # cannot be assigned a complete configuration identity safely.
                        continue
                    text = _embedding_text_from_snapshot(row["snapshot_json"])
                    content_hash = _embedding_content_hash(text)
                    configuration_id = row["model"]
                    existing = self._connection.execute(
                        """
                        SELECT content_text, magnitude, vector FROM embedding_vectors
                        WHERE project_id = ? AND model = ? AND configuration_id = ?
                          AND dimensions = ? AND content_hash = ?
                        """,
                        (
                            row["project_id"],
                            row["model"],
                            configuration_id,
                            row["dimensions"],
                            content_hash,
                        ),
                    ).fetchone()
                    if existing is not None and existing["content_text"] != text:
                        raise RuntimeError("embedding content hash collision during migration")
                    if existing is not None and (
                        existing["magnitude"] != row["magnitude"]
                        or existing["vector"] != row["vector"]
                    ):
                        raise RuntimeError(
                            "equal legacy embedding inputs have different vectors"
                        )
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO embedding_vectors(
                            project_id, model, configuration_id, dimensions, content_hash,
                            content_text, magnitude, vector
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["project_id"],
                            row["model"],
                            configuration_id,
                            row["dimensions"],
                            content_hash,
                            text,
                            row["magnitude"],
                            row["vector"],
                        ),
                    )
                    self._connection.execute(
                        """
                        INSERT INTO variant_embeddings(
                            project_id, variant_id, model, configuration_id,
                            dimensions, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["project_id"],
                            row["variant_id"],
                            row["model"],
                            configuration_id,
                            row["dimensions"],
                            content_hash,
                        ),
                    )
            self._connection.execute("DROP TABLE variant_embeddings_v10")
            self._connection.execute("PRAGMA user_version = 12")
        except BaseException:
            if manage_transaction:
                self._connection.rollback()
            raise
        else:
            if manage_transaction:
                self._connection.commit()

    def _migrate_v11(self, *, manage_transaction: bool = True) -> None:
        """Compact propagated summary solutions while retaining relational local inputs."""

        try:
            if manage_transaction:
                self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_solution_payloads (
                    project_id INTEGER NOT NULL,
                    summary_id TEXT NOT NULL,
                    encoding TEXT NOT NULL,
                    effect_count INTEGER NOT NULL,
                    origin_count INTEGER NOT NULL,
                    uncompressed_bytes INTEGER NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY (project_id, summary_id),
                    CHECK (effect_count >= 0 AND origin_count >= 0),
                    CHECK (uncompressed_bytes >= 0),
                    FOREIGN KEY (project_id, summary_id)
                        REFERENCES function_summaries(project_id, id) ON DELETE CASCADE
                )
                """
            )
            projects = self._connection.execute(
                """
                SELECT project_id FROM summary_effects WHERE is_local = 0
                UNION
                SELECT project_id FROM summary_return_origins WHERE is_local = 0
                ORDER BY project_id
                """
            ).fetchall()
            for project in projects:
                project_id = int(project["project_id"])
                effects = (
                    self._row_to_summary_effect(row)
                    for row in self._connection.execute(
                        """
                        SELECT * FROM summary_effects
                        WHERE project_id = ? AND is_local = 0
                          AND NOT EXISTS (
                              SELECT 1 FROM summary_solution_payloads payloads
                              WHERE payloads.project_id = summary_effects.project_id
                                AND payloads.summary_id = summary_effects.summary_id
                          )
                        ORDER BY summary_id, id
                        """,
                        (project_id,),
                    )
                )
                origins = (
                    self._row_to_summary_return_origin(row)
                    for row in self._connection.execute(
                        """
                        SELECT * FROM summary_return_origins
                        WHERE project_id = ? AND is_local = 0
                          AND NOT EXISTS (
                              SELECT 1 FROM summary_solution_payloads payloads
                              WHERE payloads.project_id = summary_return_origins.project_id
                                AND payloads.summary_id = summary_return_origins.summary_id
                          )
                        ORDER BY summary_id, id
                        """,
                        (project_id,),
                    )
                )
                self._write_summary_solution_groups(
                    project_id,
                    _ordered_summary_groups(effects),
                    _ordered_summary_groups(origins),
                )
            if projects:
                # SQLite resolves every foreign-key target even for an empty DELETE;
                # partial legacy schemas with no propagated rows need no cleanup.
                self._connection.execute("DELETE FROM summary_effects WHERE is_local = 0")
                self._connection.execute("DELETE FROM summary_return_origins WHERE is_local = 0")
            self._connection.execute("PRAGMA user_version = 11")
        except BaseException:
            if manage_transaction:
                self._connection.rollback()
            raise
        else:
            if manage_transaction:
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
    ) -> int:
        """Atomically replace changed units and optionally remove stale units."""

        selected_variant = build_variant or (
            batch.build_variants[0]
            if batch.build_variants
            else BuildVariant(DEFAULT_BUILD_VARIANT, Path("."))
        )
        return self.apply_ingestion_batches(
            project_root,
            (batch,),
            current_translation_unit_ids=current_translation_unit_ids,
            changed_translation_unit_ids=frozenset(unit.id for unit in batch.translation_units),
            build_variant=selected_variant,
        )

    def apply_ingestion_batches(
        self,
        project_root: Path,
        batches: Iterable[IngestionBatch],
        *,
        current_translation_unit_ids: frozenset[str] | None = None,
        changed_translation_unit_ids: frozenset[str] | None = None,
        build_variant: BuildVariant | None = None,
    ) -> int:
        """Atomically stage and retire TU-sized batches before global finalization."""

        root = str(project_root.resolve(strict=False))
        selected_variant = build_variant or BuildVariant(DEFAULT_BUILD_VARIANT, Path("."))
        # Canonical symbols are derived once from the newly inserted variants at
        # the end of this transaction. Deferral keeps child fact insertion valid
        # without a redundant provisional canonical-symbol write pass.
        self._reset_ingestion_tracking()
        self._connection.execute("PRAGMA defer_foreign_keys = ON")
        try:
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
                removed_ids = (
                    existing - current_translation_unit_ids
                    if current_translation_unit_ids is not None
                    else set()
                )
                replaced_ids = removed_ids | set(changed_translation_unit_ids or ())
                remaining_changed_ids = (
                    set(changed_translation_unit_ids)
                    if changed_translation_unit_ids is not None
                    else None
                )
                if replaced_ids:
                    self._track_replaced_units(project_id, selected_variant.name, replaced_ids)
                    self._delete_translation_units(project_id, replaced_ids)

                for batch in batches:
                    batch_unit_ids = {unit.id for unit in batch.translation_units}
                    if remaining_changed_ids is not None:
                        unexpected = batch_unit_ids - remaining_changed_ids
                        if unexpected:
                            raise ValueError(
                                "ingestion stream returned an unexpected or duplicate "
                                "translation unit"
                            )
                        remaining_changed_ids -= batch_unit_ids
                    if changed_translation_unit_ids is None and batch_unit_ids:
                        self._track_replaced_units(
                            project_id, selected_variant.name, batch_unit_ids
                        )
                        self._delete_translation_units(project_id, batch_unit_ids)
                    self._stage_ingestion_batch(project_id, batch)
                    # Release TU-local tuples before requesting the next batch;
                    # Python for-loops otherwise retain the previous loop value.
                    del batch

                if remaining_changed_ids:
                    raise ValueError("ingestion stream ended before every changed translation unit")

                self._refresh_indexed_override_candidates(project_id, selected_variant.name)
                affected_functions = {
                    row[0]
                    for row in self._connection.execute(
                        "SELECT id FROM temp._ingestion_affected_functions ORDER BY id"
                    )
                }
                affected_functions |= self._reverse_summary_callers(
                    project_id, selected_variant.name, affected_functions
                )
                invalidated_summaries = self._refresh_summary_solutions(
                    project_id, selected_variant.name, affected_functions
                )
                self._refresh_tracked_symbols(project_id)
                self._delete_orphans(project_id)
                self._delete_orphan_embedding_vectors(project_id)
                self._connection.execute(
                    """
                    UPDATE build_variants SET reindex_required = 0
                    WHERE project_id = ? AND name = ?
                    """,
                    (project_id, selected_variant.name),
                )
                # Cleanup is part of the same transaction as publication. A
                # cleanup error can therefore still roll the entire run back.
                self._clear_ingestion_tracking()
                return invalidated_summaries
        finally:
            if self._connection.in_transaction:
                self._connection.rollback()

    def _reset_ingestion_tracking(self) -> None:
        self._connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _ingestion_affected_symbols "
            "(id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        self._connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _ingestion_affected_functions "
            "(id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        self._clear_ingestion_tracking()
        self._connection.commit()

    def _clear_ingestion_tracking(self) -> None:
        self._connection.execute("DELETE FROM temp._ingestion_affected_symbols")
        self._connection.execute("DELETE FROM temp._ingestion_affected_functions")

    def _track_replaced_units(
        self, project_id: int, build_variant: str, unit_ids: Iterable[str]
    ) -> None:
        ids = sorted(set(unit_ids))
        for offset in range(0, len(ids), 500):
            chunk = ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            self._connection.execute(
                f"""
                INSERT OR IGNORE INTO temp._ingestion_affected_symbols(id)
                SELECT DISTINCT symbol_id FROM translation_unit_symbols
                WHERE project_id = ? AND translation_unit_id IN ({placeholders})
                """,
                (project_id, *chunk),
            )
            functions = self._summary_functions_from_units(project_id, set(chunk))
            functions |= self._reverse_summary_callers(project_id, build_variant, functions)
            self._connection.executemany(
                "INSERT OR IGNORE INTO temp._ingestion_affected_functions(id) VALUES (?)",
                ((function_id,) for function_id in functions),
            )

    def _stage_ingestion_batch(self, project_id: int, batch: IngestionBatch) -> None:
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

        symbols = tuple(batch.symbols)
        self._connection.executemany(
            """
            INSERT OR IGNORE INTO translation_unit_symbols(
                project_id, translation_unit_id, symbol_id, is_definition
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    symbol.translation_unit_id,
                    symbol.id,
                    int(bool(symbol.metadata.get("is_definition"))),
                )
                for symbol in symbols
            ),
        )
        self._connection.executemany(
            "INSERT OR IGNORE INTO temp._ingestion_affected_symbols(id) VALUES (?)",
            ((symbol.id,) for symbol in symbols),
        )
        self._connection.executemany(
            "INSERT OR IGNORE INTO temp._ingestion_affected_functions(id) VALUES (?)",
            ((summary.function_symbol_id,) for summary in batch.function_summaries),
        )
        self._put_symbol_variants(project_id, symbols)
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
        self._put_data_flow_facts(
            project_id,
            batch.data_flow_analyses,
            batch.memory_locations,
            batch.data_accesses,
            batch.data_flow_evidence,
        )
        self._put_summary_facts(
            project_id,
            batch.function_summaries,
            batch.summary_effects,
            batch.summary_return_origins,
            batch.call_argument_bindings,
            batch.call_result_bindings,
            batch.interprocedural_flows,
        )

    def _refresh_tracked_symbols(self, project_id: int) -> None:
        cursor = self._connection.execute(
            "SELECT id FROM temp._ingestion_affected_symbols ORDER BY id"
        )
        while rows := cursor.fetchmany(500):
            self._refresh_symbols(project_id, {row[0] for row in rows})

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

    def build_variants(
        self, project_root: Path | None = None, *, limit: int | None = None
    ) -> tuple[BuildVariant, ...]:
        try:
            project_id = self._project_id(project_root)
        except KeyError:
            return ()
        if limit is not None and not 1 <= limit <= 10_000:
            raise ValueError("build variant limit must be in [1, 10000]")
        limit_sql = " LIMIT ?" if limit is not None else ""
        parameters: tuple[object, ...] = (project_id, limit) if limit is not None else (project_id,)
        return tuple(
            BuildVariant(
                row["name"],
                Path(row["compilation_database"] or "."),
                row["target"],
                row["platform"],
                json.loads(row["metadata_json"]),
            )
            for row in self._connection.execute(
                "SELECT * FROM build_variants WHERE project_id = ? ORDER BY name" + limit_sql,
                parameters,
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
            self._delete_orphan_embedding_vectors(project_id)
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
        self._connection.execute(
            f"""
            DELETE FROM variant_embeddings
            WHERE project_id = ? AND variant_id IN (
                SELECT id FROM symbol_variants
                WHERE project_id = ? AND translation_unit_id IN ({placeholders})
            )
            """,
            (project_id, project_id, *ids),
        )
        # Delete leaf facts in bulk before their parents. Letting SQLite walk
        # several overlapping ON DELETE CASCADE paths per TU scaled superlinearly
        # for a shared-header reindex of the measured 200-TU smoke workload.
        for table in _TRANSLATION_UNIT_DELETE_ORDER:
            self._connection.execute(
                f"""
                DELETE FROM {table}
                WHERE project_id = ? AND translation_unit_id IN ({placeholders})
                """,
                (project_id, *ids),
            )
        self._connection.execute(
            f"DELETE FROM translation_units WHERE project_id = ? AND id IN ({placeholders})",
            (project_id, *ids),
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

    def _summary_functions_from_units(self, project_id: int, unit_ids: set[str]) -> set[str]:
        if not unit_ids:
            return set()
        placeholders = ",".join("?" for _ in unit_ids)
        return {
            row[0]
            for row in self._connection.execute(
                f"""
                SELECT DISTINCT function_symbol_id FROM function_summaries
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
            WHERE symbols.qualified_name IS NOT excluded.qualified_name
               OR symbols.kind IS NOT excluded.kind
               OR symbols.path IS NOT excluded.path
               OR symbols.start_line IS NOT excluded.start_line
               OR symbols.end_line IS NOT excluded.end_line
               OR symbols.start_column IS NOT excluded.start_column
               OR symbols.end_column IS NOT excluded.end_column
               OR symbols.signature IS NOT excluded.signature
               OR symbols.documentation IS NOT excluded.documentation
               OR symbols.source_hash IS NOT excluded.source_hash
               OR symbols.source_text IS NOT excluded.source_text
               OR symbols.build_configuration_id IS NOT excluded.build_configuration_id
               OR symbols.metadata_json IS NOT excluded.metadata_json
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

    def _put_canonical_symbols(
        self,
        project_id: int,
        symbols: Iterable[CodeSymbol],
        *,
        prefer_definition: bool = True,
    ) -> None:
        """Upsert canonical symbols without one preference query per symbol."""

        selected = tuple(symbols)
        existing_definitions: set[str] = set()
        if prefer_definition:
            ids = sorted({symbol.id for symbol in selected})
            for offset in range(0, len(ids), 500):
                chunk = ids[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                existing_definitions.update(
                    row["id"]
                    for row in self._connection.execute(
                        f"""
                        SELECT id FROM symbols
                        WHERE project_id = ? AND id IN ({placeholders})
                          AND json_extract(metadata_json, '$.is_definition') = 1
                        """,
                        (project_id, *chunk),
                    )
                )
        rows = (
            symbol
            for symbol in selected
            if not (
                prefer_definition
                and symbol.id in existing_definitions
                and not symbol.metadata.get("is_definition")
            )
        )
        self._connection.executemany(
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
            WHERE symbols.qualified_name IS NOT excluded.qualified_name
               OR symbols.kind IS NOT excluded.kind
               OR symbols.path IS NOT excluded.path
               OR symbols.start_line IS NOT excluded.start_line
               OR symbols.end_line IS NOT excluded.end_line
               OR symbols.start_column IS NOT excluded.start_column
               OR symbols.end_column IS NOT excluded.end_column
               OR symbols.signature IS NOT excluded.signature
               OR symbols.documentation IS NOT excluded.documentation
               OR symbols.source_hash IS NOT excluded.source_hash
               OR symbols.source_text IS NOT excluded.source_text
               OR symbols.build_configuration_id IS NOT excluded.build_configuration_id
               OR symbols.metadata_json IS NOT excluded.metadata_json
            """,
            (
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
                )
                for symbol in rows
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
        self._put_symbol_variants(project_id, (symbol,))

    def _put_symbol_variants(self, project_id: int, symbols: Iterable[CodeSymbol]) -> None:
        records: list[tuple[CodeSymbol, str, str]] = []
        for symbol in symbols:
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
            records.append((symbol, variant_id, snapshot))
        existing: dict[str, str] = {}
        variant_ids = [variant_id for _, variant_id, _ in records]
        for offset in range(0, len(variant_ids), 500):
            chunk = variant_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            existing.update(
                (row["id"], row["snapshot_json"])
                for row in self._connection.execute(
                    f"""
                    SELECT id, snapshot_json FROM symbol_variants
                    WHERE project_id = ? AND id IN ({placeholders})
                    """,
                    (project_id, *chunk),
                )
            )
            self._connection.execute(
                f"""
                DELETE FROM symbol_variant_fts
                WHERE project_id = ? AND variant_id IN ({placeholders})
                """,
                (project_id, *chunk),
            )
        changed = [
            variant_id
            for _, variant_id, snapshot in records
            if variant_id in existing and existing[variant_id] != snapshot
        ]
        for offset in range(0, len(changed), 500):
            chunk = changed[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            self._connection.execute(
                f"""
                DELETE FROM variant_embeddings
                WHERE project_id = ? AND variant_id IN ({placeholders})
                """,
                (project_id, *chunk),
            )
        self._connection.executemany(
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
                (
                    project_id,
                    variant_id,
                    symbol.id,
                    symbol.build_variant,
                    symbol.build_configuration_id,
                    symbol.translation_unit_id,
                    int(bool(symbol.metadata.get("is_definition"))),
                    snapshot,
                )
                for symbol, variant_id, snapshot in records
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO symbol_variant_fts(
                project_id, variant_id, symbol_id, build_variant,
                qualified_name, signature, documentation, source_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    variant_id,
                    symbol.id,
                    symbol.build_variant,
                    symbol.qualified_name,
                    symbol.signature,
                    symbol.documentation,
                    symbol.source_text,
                )
                for symbol, variant_id, _ in records
            ),
        )

    def _refresh_symbols(self, project_id: int, symbol_ids: set[str]) -> None:
        preferred: dict[str, CodeSymbol] = {}
        ids = sorted(symbol_ids)
        for offset in range(0, len(ids), 500):
            chunk = ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            for row in self._connection.execute(
                f"""
                SELECT symbol_id, snapshot_json FROM symbol_variants
                WHERE project_id = ? AND symbol_id IN ({placeholders})
                ORDER BY symbol_id, is_definition DESC, build_variant, translation_unit_id
                """,
                (project_id, *chunk),
            ):
                if row["symbol_id"] not in preferred:
                    preferred[row["symbol_id"]] = self._snapshot_symbol(row["snapshot_json"])
        self._put_canonical_symbols(project_id, preferred.values(), prefer_definition=False)

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
                    # Schema v6 requires a JSON payload; JSON null preserves a
                    # truthful missing spelling span without rebuilding the table.
                    _span_json(site.spelling_span) or "null",
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

    def _put_data_flow_facts(
        self,
        project_id: int,
        analyses: Iterable[DataFlowAnalysis],
        locations: Iterable[MemoryLocation],
        accesses: Iterable[DataAccess],
        evidence: Iterable[DataFlowEvidence],
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO data_flow_analyses(
                project_id, id, graph_id, complete, incomplete_reasons_json,
                iteration_count, max_iterations, max_alias_targets,
                max_access_path_depth, max_locations, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    analysis.id,
                    analysis.graph_id,
                    int(analysis.complete),
                    json.dumps(analysis.incomplete_reasons),
                    analysis.iteration_count,
                    analysis.max_iterations,
                    analysis.max_alias_targets,
                    analysis.max_access_path_depth,
                    analysis.max_locations,
                    analysis.translation_unit_id,
                    analysis.build_configuration_id,
                    analysis.build_variant,
                )
                for analysis in analyses
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO memory_locations(
                project_id, id, analysis_id, graph_id, kind, name, type_name,
                declaration_symbol_id, base_location_id, access_path_json,
                is_volatile, is_atomic, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    location.id,
                    location.analysis_id,
                    location.graph_id,
                    location.kind.value,
                    location.name,
                    location.type_name,
                    location.declaration_symbol_id,
                    location.base_location_id,
                    json.dumps(location.access_path),
                    int(location.is_volatile),
                    int(location.is_atomic),
                    location.translation_unit_id,
                    location.build_configuration_id,
                    location.build_variant,
                )
                for location in sorted(locations, key=lambda item: (len(item.access_path), item.id))
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO data_accesses(
                project_id, id, analysis_id, graph_id, block_id,
                cfg_element_id, location_id, kind, sequence, span_json,
                expression, pointee_symbol_ids_json, points_to_complete,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    access.id,
                    access.analysis_id,
                    access.graph_id,
                    access.block_id,
                    access.cfg_element_id,
                    access.location_id,
                    access.kind.value,
                    access.sequence,
                    _span_json(access.span),
                    access.expression,
                    json.dumps(access.pointee_symbol_ids),
                    int(access.points_to_complete),
                    access.translation_unit_id,
                    access.build_configuration_id,
                    access.build_variant,
                )
                for access in accesses
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO data_flow_evidence(
                project_id, id, analysis_id, graph_id, relation, certainty,
                reason, source_access_id, target_access_id,
                source_location_id, target_location_id, evidence_span_json,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.analysis_id,
                    item.graph_id,
                    item.relation.value,
                    item.certainty.value,
                    item.reason,
                    item.source_access_id,
                    item.target_access_id,
                    item.source_location_id,
                    item.target_location_id,
                    _span_json(item.evidence_span),
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in evidence
            ),
        )

    def _put_summary_facts(
        self,
        project_id: int,
        summaries: Iterable[FunctionSummary],
        effects: Iterable[SummaryEffect],
        origins: Iterable[SummaryReturnOrigin],
        argument_bindings: Iterable[CallArgumentBinding],
        result_bindings: Iterable[CallResultBinding],
        flows: Iterable[InterproceduralFlow],
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO function_summaries(
                project_id, id, function_symbol_id, graph_id, analysis_id,
                parameter_modes_json, parameter_location_ids_json,
                local_complete, local_incomplete_reasons_json, complete,
                incomplete_reasons_json, recursive, iteration_count,
                max_scc_iterations, max_scc_size, max_summary_effects,
                solution_hash, translation_unit_id, build_configuration_id,
                build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.function_symbol_id,
                    item.graph_id,
                    item.analysis_id,
                    json.dumps(item.parameter_modes),
                    json.dumps(item.parameter_location_ids),
                    int(item.local_complete),
                    json.dumps(item.local_incomplete_reasons),
                    int(item.complete),
                    json.dumps(item.incomplete_reasons),
                    int(item.recursive),
                    item.iteration_count,
                    item.max_scc_iterations,
                    item.max_scc_size,
                    item.max_summary_effects,
                    item.solution_hash,
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in summaries
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO summary_effects(
                project_id, id, summary_id, kind, location_kind, certainty,
                reason, parameter_index, access_path_json, location_id,
                source_access_id, is_local, via_callsite_id, target_symbol_id,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.summary_id,
                    item.kind.value,
                    item.location_kind.value,
                    item.certainty.value,
                    item.reason,
                    item.parameter_index,
                    json.dumps(item.access_path),
                    item.location_id,
                    item.source_access_id,
                    int(item.is_local),
                    item.via_callsite_id,
                    item.target_symbol_id,
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in effects
                if item.is_local
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO summary_return_origins(
                project_id, id, summary_id, kind, certainty, reason,
                location_kind, parameter_index, access_path_json, location_id,
                callsite_id, is_local, via_callsite_id, target_symbol_id,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.summary_id,
                    item.kind.value,
                    item.certainty.value,
                    item.reason,
                    item.location_kind.value if item.location_kind else None,
                    item.parameter_index,
                    json.dumps(item.access_path),
                    item.location_id,
                    item.callsite_id,
                    int(item.is_local),
                    item.via_callsite_id,
                    item.target_symbol_id,
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in origins
                if item.is_local
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO call_argument_bindings(
                project_id, id, caller_summary_id, callsite_id, argument_index,
                location_id, location_kind, parameter_index, access_path_json,
                writeback_candidate, complete, incomplete_reason,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.caller_summary_id,
                    item.callsite_id,
                    item.argument_index,
                    item.location_id,
                    item.location_kind.value,
                    item.parameter_index,
                    json.dumps(item.access_path),
                    int(item.writeback_candidate),
                    int(item.complete),
                    item.incomplete_reason,
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in argument_bindings
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO call_result_bindings(
                project_id, id, caller_summary_id, callsite_id, location_id,
                definition_access_id, translation_unit_id,
                build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.caller_summary_id,
                    item.callsite_id,
                    item.location_id,
                    item.definition_access_id,
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in result_bindings
            ),
        )
        self._put_interprocedural_flows(project_id, flows)

    def _write_summary_solution_payload(
        self,
        project_id: int,
        summary_id: str,
        effects: Sequence[SummaryEffect],
        origins: Sequence[SummaryReturnOrigin],
    ) -> None:
        if not effects and not origins:
            return
        effect_count, origin_count, raw_size, payload_hash, payload = _encode_summary_payload(
            summary_id, effects, origins
        )
        self._connection.execute(
            """
            INSERT INTO summary_solution_payloads(
                project_id, summary_id, encoding, effect_count, origin_count,
                uncompressed_bytes, payload_hash, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                summary_id,
                SUMMARY_PAYLOAD_ENCODING,
                effect_count,
                origin_count,
                raw_size,
                payload_hash,
                payload,
            ),
        )

    def _put_summary_solution_payloads(
        self,
        project_id: int,
        summary_ids: set[str],
        effects: Iterable[SummaryEffect],
        origins: Iterable[SummaryReturnOrigin],
    ) -> None:
        """Persist sorted solver output one summary at a time without relational expansion."""

        self._write_summary_solution_groups(
            project_id,
            _propagated_summary_groups(effects, summary_ids),
            _propagated_summary_groups(origins, summary_ids),
        )

    def _write_summary_solution_groups(
        self,
        project_id: int,
        effect_groups: Iterator[tuple[str, tuple[SummaryEffect | SummaryReturnOrigin, ...]]],
        origin_groups: Iterator[tuple[str, tuple[SummaryEffect | SummaryReturnOrigin, ...]]],
    ) -> None:
        effect_item = next(effect_groups, None)
        origin_item = next(origin_groups, None)
        while effect_item is not None or origin_item is not None:
            keys = [item[0] for item in (effect_item, origin_item) if item is not None]
            summary_id = min(keys)
            grouped_effects: tuple[SummaryEffect, ...] = ()
            grouped_origins: tuple[SummaryReturnOrigin, ...] = ()
            if effect_item is not None and effect_item[0] == summary_id:
                grouped_effects = effect_item[1]
                effect_item = next(effect_groups, None)
            if origin_item is not None and origin_item[0] == summary_id:
                grouped_origins = origin_item[1]
                origin_item = next(origin_groups, None)
            self._write_summary_solution_payload(
                project_id, summary_id, grouped_effects, grouped_origins
            )

    def _put_interprocedural_flows(
        self, project_id: int, flows: Iterable[InterproceduralFlow]
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO interprocedural_flows(
                project_id, id, kind, caller_summary_id, callee_summary_id,
                callsite_id, target_symbol_id, target_certainty, certainty,
                reason, argument_index, caller_location_id, callee_location_id,
                caller_access_id, translation_unit_id, build_configuration_id,
                build_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    item.id,
                    item.kind.value,
                    item.caller_summary_id,
                    item.callee_summary_id,
                    item.callsite_id,
                    item.target_symbol_id,
                    item.target_certainty.value,
                    item.certainty.value,
                    item.reason,
                    item.argument_index,
                    item.caller_location_id,
                    item.callee_location_id,
                    item.caller_access_id,
                    item.translation_unit_id,
                    item.build_configuration_id,
                    item.build_variant,
                )
                for item in flows
            ),
        )

    def _reverse_summary_callers(
        self, project_id: int, build_variant: str, function_ids: set[str]
    ) -> set[str]:
        closure = set(function_ids)
        frontier = set(function_ids)
        while frontier:
            placeholders = ",".join("?" for _ in frontier)
            callers = {
                row[0]
                for row in self._connection.execute(
                    f"""
                    SELECT DISTINCT sites.owner_symbol_id
                    FROM call_targets targets
                    JOIN callsites sites
                      ON sites.project_id = targets.project_id
                     AND sites.id = targets.callsite_id
                    WHERE targets.project_id = ? AND targets.build_variant = ?
                      AND targets.target_symbol_id IN ({placeholders})
                    """,
                    (project_id, build_variant, *sorted(frontier)),
                )
            }
            frontier = callers - closure
            closure.update(frontier)
        return closure - function_ids

    def _forward_summary_callees(
        self, project_id: int, build_variant: str, function_ids: set[str]
    ) -> set[str]:
        closure = set(function_ids)
        frontier = set(function_ids)
        while frontier:
            placeholders = ",".join("?" for _ in frontier)
            callees = {
                row[0]
                for row in self._connection.execute(
                    f"""
                    SELECT DISTINCT targets.target_symbol_id
                    FROM callsites sites
                    JOIN call_targets targets
                      ON targets.project_id = sites.project_id
                     AND targets.callsite_id = sites.id
                    WHERE sites.project_id = ? AND sites.build_variant = ?
                      AND sites.owner_symbol_id IN ({placeholders})
                    """,
                    (project_id, build_variant, *sorted(frontier)),
                )
            }
            frontier = callees - closure
            closure.update(frontier)
        return closure

    def _refresh_summary_solutions(
        self, project_id: int, build_variant: str, affected_functions: set[str]
    ) -> int:
        if not affected_functions:
            return 0
        indexed_functions = {
            row[0]
            for row in self._connection.execute(
                """
                SELECT DISTINCT function_symbol_id FROM function_summaries
                WHERE project_id = ? AND build_variant = ?
                """,
                (project_id, build_variant),
            )
        }
        if indexed_functions <= affected_functions:
            # A full build/header reindex already includes every indexed callee;
            # walking the same call graph again only repeats thousands of lookups.
            selected_functions = indexed_functions
        else:
            selected_functions = self._forward_summary_callees(
                project_id, build_variant, affected_functions
            )
        placeholders = ",".join("?" for _ in selected_functions)
        summary_rows = self._connection.execute(
            f"""
            SELECT * FROM function_summaries
            WHERE project_id = ? AND build_variant = ?
              AND function_symbol_id IN ({placeholders})
            ORDER BY id
            """,
            (project_id, build_variant, *sorted(selected_functions)),
        ).fetchall()
        summaries = tuple(self._row_to_function_summary(row) for row in summary_rows)
        if not summaries:
            return 0
        summary_ids = {item.id for item in summaries}
        summary_placeholders = ",".join("?" for _ in summary_ids)
        effects = tuple(
            self._row_to_summary_effect(row)
            for row in self._connection.execute(
                f"""
                SELECT * FROM summary_effects
                WHERE project_id = ? AND is_local = 1
                  AND summary_id IN ({summary_placeholders}) ORDER BY id
                """,
                (project_id, *sorted(summary_ids)),
            )
        )
        origins = tuple(
            self._row_to_summary_return_origin(row)
            for row in self._connection.execute(
                f"""
                SELECT * FROM summary_return_origins
                WHERE project_id = ? AND is_local = 1
                  AND summary_id IN ({summary_placeholders}) ORDER BY id
                """,
                (project_id, *sorted(summary_ids)),
            )
        )
        arguments = tuple(
            self._row_to_call_argument_binding(row)
            for row in self._connection.execute(
                f"""
                SELECT * FROM call_argument_bindings
                WHERE project_id = ? AND caller_summary_id IN ({summary_placeholders})
                ORDER BY id
                """,
                (project_id, *sorted(summary_ids)),
            )
        )
        results = tuple(
            self._row_to_call_result_binding(row)
            for row in self._connection.execute(
                f"""
                SELECT * FROM call_result_bindings
                WHERE project_id = ? AND caller_summary_id IN ({summary_placeholders})
                ORDER BY id
                """,
                (project_id, *sorted(summary_ids)),
            )
        )
        sites = tuple(
            self._row_to_callsite(row)
            for row in self._connection.execute(
                f"""
                SELECT * FROM callsites
                WHERE project_id = ? AND build_variant = ?
                  AND owner_symbol_id IN ({placeholders}) ORDER BY id
                """,
                (project_id, build_variant, *sorted(selected_functions)),
            )
        )
        site_ids = {item.id for item in sites}
        if site_ids:
            site_placeholders = ",".join("?" for _ in site_ids)
            targets = tuple(
                self._row_to_call_target(row)
                for row in self._connection.execute(
                    f"""
                    SELECT * FROM call_targets WHERE project_id = ?
                      AND callsite_id IN ({site_placeholders}) ORDER BY id
                    """,
                    (project_id, *sorted(site_ids)),
                )
            )
        else:
            targets = ()
        solution = solve_interprocedural(
            summaries, effects, origins, arguments, results, sites, targets
        )
        impacted_ids = {
            item.id for item in summaries if item.function_symbol_id in affected_functions
        }
        if not impacted_ids:
            return 0
        impacted_placeholders = ",".join("?" for _ in impacted_ids)
        parameters = (project_id, *sorted(impacted_ids))
        self._connection.execute(
            f"DELETE FROM interprocedural_flows WHERE project_id = ? "
            f"AND caller_summary_id IN ({impacted_placeholders})",
            parameters,
        )
        self._connection.execute(
            f"DELETE FROM summary_effects WHERE project_id = ? AND is_local = 0 "
            f"AND summary_id IN ({impacted_placeholders})",
            parameters,
        )
        self._connection.execute(
            f"DELETE FROM summary_return_origins WHERE project_id = ? AND is_local = 0 "
            f"AND summary_id IN ({impacted_placeholders})",
            parameters,
        )
        self._connection.execute(
            f"DELETE FROM summary_solution_payloads WHERE project_id = ? "
            f"AND summary_id IN ({impacted_placeholders})",
            parameters,
        )
        solved = [item for item in solution.summaries if item.id in impacted_ids]
        self._connection.executemany(
            """
            UPDATE function_summaries SET
                complete = ?, incomplete_reasons_json = ?, recursive = ?,
                iteration_count = ?, max_scc_iterations = ?, max_scc_size = ?,
                max_summary_effects = ?, solution_hash = ?
            WHERE project_id = ? AND id = ?
            """,
            (
                (
                    int(item.complete),
                    json.dumps(item.incomplete_reasons),
                    int(item.recursive),
                    item.iteration_count,
                    item.max_scc_iterations,
                    item.max_scc_size,
                    item.max_summary_effects,
                    item.solution_hash,
                    project_id,
                    item.id,
                )
                for item in solved
            ),
        )
        self._put_summary_facts(
            project_id,
            (),
            (),
            (),
            (),
            (),
            (item for item in solution.flows if item.caller_summary_id in impacted_ids),
        )
        self._put_summary_solution_payloads(
            project_id,
            impacted_ids,
            solution.effects,
            solution.return_origins,
        )
        return len(impacted_ids)

    @staticmethod
    def _row_to_function_summary(row: sqlite3.Row) -> FunctionSummary:
        return FunctionSummary(
            id=row["id"],
            function_symbol_id=row["function_symbol_id"],
            graph_id=row["graph_id"],
            analysis_id=row["analysis_id"],
            parameter_modes=tuple(json.loads(row["parameter_modes_json"])),
            parameter_location_ids=tuple(json.loads(row["parameter_location_ids_json"])),
            local_complete=bool(row["local_complete"]),
            local_incomplete_reasons=tuple(json.loads(row["local_incomplete_reasons_json"])),
            complete=bool(row["complete"]),
            incomplete_reasons=tuple(json.loads(row["incomplete_reasons_json"])),
            recursive=bool(row["recursive"]),
            iteration_count=row["iteration_count"],
            max_scc_iterations=row["max_scc_iterations"],
            max_scc_size=row["max_scc_size"],
            max_summary_effects=row["max_summary_effects"],
            solution_hash=row["solution_hash"],
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_summary_effect(row: sqlite3.Row) -> SummaryEffect:
        return SummaryEffect(
            id=row["id"],
            summary_id=row["summary_id"],
            kind=SummaryEffectKind(row["kind"]),
            location_kind=MemoryLocationKind(row["location_kind"]),
            certainty=DataFlowCertainty(row["certainty"]),
            reason=row["reason"],
            parameter_index=row["parameter_index"],
            access_path=tuple(json.loads(row["access_path_json"])),
            location_id=row["location_id"],
            source_access_id=row["source_access_id"],
            is_local=bool(row["is_local"]),
            via_callsite_id=row["via_callsite_id"],
            target_symbol_id=row["target_symbol_id"],
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_summary_return_origin(row: sqlite3.Row) -> SummaryReturnOrigin:
        return SummaryReturnOrigin(
            id=row["id"],
            summary_id=row["summary_id"],
            kind=SummaryReturnOriginKind(row["kind"]),
            certainty=DataFlowCertainty(row["certainty"]),
            reason=row["reason"],
            location_kind=(
                MemoryLocationKind(row["location_kind"]) if row["location_kind"] else None
            ),
            parameter_index=row["parameter_index"],
            access_path=tuple(json.loads(row["access_path_json"])),
            location_id=row["location_id"],
            callsite_id=row["callsite_id"],
            is_local=bool(row["is_local"]),
            via_callsite_id=row["via_callsite_id"],
            target_symbol_id=row["target_symbol_id"],
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_call_argument_binding(row: sqlite3.Row) -> CallArgumentBinding:
        return CallArgumentBinding(
            id=row["id"],
            caller_summary_id=row["caller_summary_id"],
            callsite_id=row["callsite_id"],
            argument_index=row["argument_index"],
            location_id=row["location_id"],
            location_kind=MemoryLocationKind(row["location_kind"]),
            parameter_index=row["parameter_index"],
            access_path=tuple(json.loads(row["access_path_json"])),
            writeback_candidate=bool(row["writeback_candidate"]),
            complete=bool(row["complete"]),
            incomplete_reason=row["incomplete_reason"],
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_call_result_binding(row: sqlite3.Row) -> CallResultBinding:
        return CallResultBinding(
            id=row["id"],
            caller_summary_id=row["caller_summary_id"],
            callsite_id=row["callsite_id"],
            location_id=row["location_id"],
            definition_access_id=row["definition_access_id"],
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    def _refresh_indexed_override_candidates(self, project_id: int, build_variant: str) -> None:
        """Rebuild build-wide virtual candidates after any incremental TU replacement."""

        derivation = "indexed_override_candidate"
        self._connection.execute(
            """
            DELETE FROM call_targets
            WHERE project_id = ? AND build_variant = ? AND derivation = ?
            """,
            (project_id, build_variant, derivation),
        )
        # Override edges can live in a different TU from an unchanged callsite, so deriving
        # candidates only from the current ingestion batch loses targets after incremental edits.
        rows = self._connection.execute(
            """
            WITH RECURSIVE override_closure(base_id, target_id) AS (
                SELECT target_id, source_id FROM edges
                WHERE project_id = ? AND build_variant = ? AND relation = 'overrides'
                UNION
                SELECT closure.base_id, edges.source_id
                FROM override_closure AS closure
                JOIN edges
                  ON edges.project_id = ?
                 AND edges.build_variant = ?
                 AND edges.relation = 'overrides'
                 AND edges.target_id = closure.target_id
            )
            SELECT sites.id AS callsite_id, closure.target_id,
                   sites.expansion_span_json, sites.translation_unit_id,
                   sites.build_configuration_id, sites.build_variant
            FROM callsites AS sites
            JOIN override_closure AS closure
              ON closure.base_id = sites.static_target_symbol_id
            WHERE sites.project_id = ? AND sites.build_variant = ?
              AND sites.dispatch_kind = 'virtual'
              AND NOT EXISTS (
                  SELECT 1 FROM call_targets AS existing
                  WHERE existing.project_id = sites.project_id
                    AND existing.callsite_id = sites.id
                    AND existing.target_symbol_id = closure.target_id
              )
            ORDER BY sites.id, closure.target_id
            """,
            (project_id, build_variant, project_id, build_variant, project_id, build_variant),
        ).fetchall()
        reason = (
            "target overrides the statically selected virtual method in this build; "
            "the value is deterministic ranking evidence, not a probability"
        )
        self._connection.executemany(
            """
            INSERT INTO call_targets(
                project_id, id, callsite_id, target_symbol_id, certainty,
                confidence, confidence_reason, derivation, evidence_span_json,
                translation_unit_id, build_configuration_id, build_variant
            ) VALUES (?, ?, ?, ?, 'possible', 0.5, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    project_id,
                    _stable_id(
                        "call_target",
                        row["build_variant"],
                        row["build_configuration_id"],
                        row["translation_unit_id"],
                        row["callsite_id"],
                        row["target_id"],
                        CallTargetCertainty.POSSIBLE.value,
                        derivation,
                    ),
                    row["callsite_id"],
                    row["target_id"],
                    reason,
                    derivation,
                    row["expansion_span_json"],
                    row["translation_unit_id"],
                    row["build_configuration_id"],
                    row["build_variant"],
                )
                for row in rows
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

    def get_symbols(
        self,
        symbol_ids: Iterable[str],
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
    ) -> tuple[CodeSymbol | None, ...]:
        """Resolve symbols in bounded SQL batches while preserving input order."""

        requested = tuple(symbol_ids)
        if not requested:
            return ()
        try:
            project_id = self._project_id(project_root)
        except KeyError:
            return tuple(None for _ in requested)
        names = self._scope_names(build_scope)
        scope_placeholders = ",".join("?" for _ in names)
        resolved: dict[str, CodeSymbol] = {}
        unique_ids = sorted(set(requested))
        for offset in range(0, len(unique_ids), 400):
            chunk = unique_ids[offset : offset + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"""
                SELECT * FROM symbol_variants
                WHERE project_id = ? AND build_variant IN ({scope_placeholders})
                  AND (id IN ({placeholders}) OR symbol_id IN ({placeholders}))
                ORDER BY is_definition DESC, build_variant, translation_unit_id, id
                """,
                (project_id, *names, *chunk, *chunk),
            )
            chunk_ids = set(chunk)
            canonical_matches: dict[str, CodeSymbol] = {}
            exact_matches: dict[str, CodeSymbol] = {}
            for row in rows:
                symbol = self._variant_row_to_symbol(row)
                if row["id"] in chunk_ids:
                    exact_matches[row["id"]] = symbol
                if row["symbol_id"] in chunk_ids:
                    canonical_matches.setdefault(row["symbol_id"], symbol)
            resolved.update(canonical_matches)
            # A variant ID is an exact identity; it must win even in the
            # pathological case where it equals another symbol's canonical ID.
            resolved.update(exact_matches)
        missing = [symbol_id for symbol_id in unique_ids if symbol_id not in resolved]
        if DEFAULT_BUILD_VARIANT in names:
            for offset in range(0, len(missing), 500):
                chunk = missing[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                resolved.update(
                    (row["id"], self._row_to_symbol(row))
                    for row in self._connection.execute(
                        f"""
                        SELECT * FROM symbols
                        WHERE project_id = ? AND id IN ({placeholders})
                        """,
                        (project_id, *chunk),
                    )
                )
        return tuple(resolved.get(symbol_id) for symbol_id in requested)

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

    def call_evidence(
        self,
        symbol_id: str,
        *,
        incoming: bool,
        project_root: Path | None = None,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> BoundedCfgResult[tuple[CallSite, CallTarget]]:
        """Return bounded callsite/target evidence, with certain targets ranked first."""

        limit = _call_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        endpoint = "targets.target_symbol_id" if incoming else "sites.owner_symbol_id"
        rows = self._connection.execute(
            f"""
            SELECT sites.*, targets.id AS target_id,
                   targets.target_symbol_id AS target_target_symbol_id,
                   targets.certainty AS target_certainty,
                   targets.confidence AS target_confidence,
                   targets.confidence_reason AS target_confidence_reason,
                   targets.derivation AS target_derivation,
                   targets.evidence_span_json AS target_evidence_span_json,
                   targets.translation_unit_id AS target_translation_unit_id,
                   targets.build_configuration_id AS target_build_configuration_id,
                   targets.build_variant AS target_build_variant
            FROM callsites sites
            JOIN call_targets targets
              ON targets.project_id = sites.project_id AND targets.callsite_id = sites.id
            WHERE sites.project_id = ? AND {endpoint} = ?
              AND sites.build_variant IN ({placeholders})
            ORDER BY CASE targets.certainty WHEN 'certain' THEN 0 ELSE 1 END,
                     targets.confidence DESC, sites.build_variant,
                     json_extract(sites.expansion_span_json, '$.path'),
                     json_extract(sites.expansion_span_json, '$.start_line'),
                     json_extract(sites.expansion_span_json, '$.start_column'),
                     sites.id, targets.target_symbol_id, targets.id
            LIMIT ?
            """,
            (project_id, symbol_id, *names, limit + 1),
        ).fetchall()
        items = tuple(
            (
                self._row_to_callsite(row),
                CallTarget(
                    id=row["target_id"],
                    callsite_id=row["id"],
                    target_symbol_id=row["target_target_symbol_id"],
                    certainty=CallTargetCertainty(row["target_certainty"]),
                    confidence=row["target_confidence"],
                    confidence_reason=row["target_confidence_reason"],
                    derivation=row["target_derivation"],
                    evidence_span=_required_span_from_json(row["target_evidence_span_json"]),
                    translation_unit_id=row["target_translation_unit_id"],
                    build_configuration_id=row["target_build_configuration_id"],
                    build_variant=row["target_build_variant"],
                ),
            )
            for row in rows[:limit]
        )
        return BoundedCfgResult(items, len(rows) > limit)

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
            spelling_span=_span_from_json(row["spelling_span_json"]),
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

    def data_flow_analyses(
        self,
        graph_id: str | None = None,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> BoundedCfgResult[DataFlowAnalysis]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        graph_sql = ""
        parameters: list[object] = [project_id, *names]
        if graph_id is not None:
            graph_sql = " AND graph_id = ?"
            parameters.append(graph_id)
        parameters.append(limit + 1)
        rows = self._connection.execute(
            f"""
            SELECT * FROM data_flow_analyses
            WHERE project_id = ? AND build_variant IN ({placeholders}){graph_sql}
            ORDER BY build_variant, graph_id, id LIMIT ?
            """,
            parameters,
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_data_flow_analysis(row) for row in rows[:limit]),
            len(rows) > limit,
        )

    def memory_locations(
        self,
        analysis_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 1_000,
    ) -> BoundedCfgResult[MemoryLocation]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM memory_locations
            WHERE project_id = ? AND analysis_id = ?
              AND build_variant IN ({placeholders})
            ORDER BY kind, name, id LIMIT ?
            """,
            (project_id, analysis_id, *names, limit + 1),
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_memory_location(row) for row in rows[:limit]), len(rows) > limit
        )

    def data_accesses(
        self,
        analysis_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 2_000,
    ) -> BoundedCfgResult[DataAccess]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT accesses.* FROM data_accesses accesses
            JOIN cfg_blocks blocks
              ON blocks.project_id = accesses.project_id AND blocks.id = accesses.block_id
            WHERE accesses.project_id = ? AND accesses.analysis_id = ?
              AND accesses.build_variant IN ({placeholders})
            ORDER BY blocks.block_index, accesses.sequence, accesses.id LIMIT ?
            """,
            (project_id, analysis_id, *names, limit + 1),
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_data_access(row) for row in rows[:limit]), len(rows) > limit
        )

    def data_flow_evidence(
        self,
        analysis_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 2_000,
    ) -> BoundedCfgResult[DataFlowEvidence]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM data_flow_evidence
            WHERE project_id = ? AND analysis_id = ?
              AND build_variant IN ({placeholders})
            ORDER BY CASE certainty WHEN 'certain' THEN 0 ELSE 1 END,
                     relation, source_access_id, source_location_id,
                     target_access_id, target_location_id, id LIMIT ?
            """,
            (project_id, analysis_id, *names, limit + 1),
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_data_flow_evidence(row) for row in rows[:limit]),
            len(rows) > limit,
        )

    def function_summaries(
        self,
        function_symbol_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> BoundedCfgResult[FunctionSummary]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM function_summaries
            WHERE project_id = ? AND function_symbol_id = ?
              AND build_variant IN ({placeholders})
            ORDER BY build_variant, build_configuration_id, translation_unit_id, id LIMIT ?
            """,
            (project_id, function_symbol_id, *names, limit + 1),
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_function_summary(row) for row in rows[:limit]), len(rows) > limit
        )

    def summary_effects(
        self,
        summary_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 1_000,
    ) -> BoundedCfgResult[SummaryEffect]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM summary_effects
            WHERE project_id = ? AND summary_id = ?
              AND is_local = 1
              AND build_variant IN ({placeholders})
            ORDER BY CASE certainty WHEN 'certain' THEN 0 ELSE 1 END,
                     is_local DESC, kind, id LIMIT ?
            """,
            (project_id, summary_id, *names, limit + 1),
        ).fetchall()
        local = tuple(self._row_to_summary_effect(row) for row in rows)
        propagated, _origins = self._summary_solution_payload(project_id, summary_id, names)
        combined = tuple(sorted((*local, *propagated), key=_summary_effect_order))
        return BoundedCfgResult(combined[:limit], len(combined) > limit)

    def summary_return_origins(
        self,
        summary_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 1_000,
    ) -> BoundedCfgResult[SummaryReturnOrigin]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM summary_return_origins
            WHERE project_id = ? AND summary_id = ?
              AND is_local = 1
              AND build_variant IN ({placeholders})
            ORDER BY CASE certainty WHEN 'certain' THEN 0 ELSE 1 END,
                     is_local DESC, kind, id LIMIT ?
            """,
            (project_id, summary_id, *names, limit + 1),
        ).fetchall()
        local = tuple(self._row_to_summary_return_origin(row) for row in rows)
        _effects, propagated = self._summary_solution_payload(project_id, summary_id, names)
        combined = tuple(sorted((*local, *propagated), key=_summary_origin_order))
        return BoundedCfgResult(combined[:limit], len(combined) > limit)

    def _summary_solution_payload(
        self, project_id: int, summary_id: str, build_variants: tuple[str, ...]
    ) -> tuple[tuple[SummaryEffect, ...], tuple[SummaryReturnOrigin, ...]]:
        placeholders = ",".join("?" for _ in build_variants)
        row = self._connection.execute(
            f"""
            SELECT payloads.encoding, payloads.effect_count, payloads.origin_count,
                   payloads.uncompressed_bytes, payloads.payload_hash,
                   length(payloads.payload) AS compressed_bytes,
                   CASE WHEN length(payloads.payload) <= ?
                        THEN payloads.payload END AS payload
            FROM summary_solution_payloads payloads
            JOIN function_summaries summaries
              ON summaries.project_id = payloads.project_id
             AND summaries.id = payloads.summary_id
            WHERE payloads.project_id = ? AND payloads.summary_id = ?
              AND summaries.build_variant IN ({placeholders})
            """,
            (
                MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES,
                project_id,
                summary_id,
                *build_variants,
            ),
        ).fetchone()
        if row is None:
            return (), ()
        if row["compressed_bytes"] > MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES:
            raise SummaryPayloadError("summary payload compressed-size limit exceeded")
        return _decode_summary_payload(
            summary_id,
            encoding=row["encoding"],
            effect_count=row["effect_count"],
            origin_count=row["origin_count"],
            uncompressed_bytes=row["uncompressed_bytes"],
            payload_hash=row["payload_hash"],
            payload=row["payload"],
        )

    def interprocedural_flows(
        self,
        summary_id: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        limit: int = 1_000,
    ) -> BoundedCfgResult[InterproceduralFlow]:
        limit = _cfg_limit(limit)
        project_id = self._project_id(project_root)
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        rows = self._connection.execute(
            f"""
            SELECT * FROM interprocedural_flows
            WHERE project_id = ?
              AND (caller_summary_id = ? OR callee_summary_id = ?)
              AND build_variant IN ({placeholders})
            ORDER BY CASE certainty WHEN 'certain' THEN 0 ELSE 1 END,
                     CASE target_certainty WHEN 'certain' THEN 0 ELSE 1 END,
                     kind, callsite_id, id LIMIT ?
            """,
            (project_id, summary_id, summary_id, *names, limit + 1),
        ).fetchall()
        return BoundedCfgResult(
            tuple(self._row_to_interprocedural_flow(row) for row in rows[:limit]),
            len(rows) > limit,
        )

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

    @staticmethod
    def _row_to_data_flow_analysis(row: sqlite3.Row) -> DataFlowAnalysis:
        return DataFlowAnalysis(
            id=row["id"],
            graph_id=row["graph_id"],
            complete=bool(row["complete"]),
            incomplete_reasons=tuple(json.loads(row["incomplete_reasons_json"])),
            iteration_count=row["iteration_count"],
            max_iterations=row["max_iterations"],
            max_alias_targets=row["max_alias_targets"],
            max_access_path_depth=row["max_access_path_depth"],
            max_locations=row["max_locations"],
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_memory_location(row: sqlite3.Row) -> MemoryLocation:
        return MemoryLocation(
            id=row["id"],
            analysis_id=row["analysis_id"],
            graph_id=row["graph_id"],
            kind=MemoryLocationKind(row["kind"]),
            name=row["name"],
            type_name=row["type_name"],
            declaration_symbol_id=row["declaration_symbol_id"],
            base_location_id=row["base_location_id"],
            access_path=tuple(json.loads(row["access_path_json"])),
            is_volatile=bool(row["is_volatile"]),
            is_atomic=bool(row["is_atomic"]),
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_data_access(row: sqlite3.Row) -> DataAccess:
        return DataAccess(
            id=row["id"],
            analysis_id=row["analysis_id"],
            graph_id=row["graph_id"],
            block_id=row["block_id"],
            location_id=row["location_id"],
            kind=DataAccessKind(row["kind"]),
            sequence=row["sequence"],
            cfg_element_id=row["cfg_element_id"],
            span=_span_from_json(row["span_json"]),
            expression=row["expression"],
            pointee_symbol_ids=tuple(json.loads(row["pointee_symbol_ids_json"])),
            points_to_complete=bool(row["points_to_complete"]),
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_data_flow_evidence(row: sqlite3.Row) -> DataFlowEvidence:
        return DataFlowEvidence(
            id=row["id"],
            analysis_id=row["analysis_id"],
            graph_id=row["graph_id"],
            relation=DataFlowRelation(row["relation"]),
            certainty=DataFlowCertainty(row["certainty"]),
            reason=row["reason"],
            source_access_id=row["source_access_id"],
            target_access_id=row["target_access_id"],
            source_location_id=row["source_location_id"],
            target_location_id=row["target_location_id"],
            evidence_span=_span_from_json(row["evidence_span_json"]),
            translation_unit_id=row["translation_unit_id"],
            build_configuration_id=row["build_configuration_id"],
            build_variant=row["build_variant"],
        )

    @staticmethod
    def _row_to_interprocedural_flow(row: sqlite3.Row) -> InterproceduralFlow:
        return InterproceduralFlow(
            id=row["id"],
            kind=InterproceduralFlowKind(row["kind"]),
            caller_summary_id=row["caller_summary_id"],
            callee_summary_id=row["callee_summary_id"],
            callsite_id=row["callsite_id"],
            target_symbol_id=row["target_symbol_id"],
            target_certainty=CallTargetCertainty(row["target_certainty"]),
            certainty=DataFlowCertainty(row["certainty"]),
            reason=row["reason"],
            argument_index=row["argument_index"],
            caller_location_id=row["caller_location_id"],
            callee_location_id=row["callee_location_id"],
            caller_access_id=row["caller_access_id"],
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
        configuration_id: str | None = None,
    ) -> None:
        symbol = self.get_symbol(symbol_id, project_root, build_scope=build_scope)
        if symbol is None or not symbol.variant_id:
            raise KeyError(f"symbol variant is not indexed: {symbol_id}")
        text = _embedding_text(symbol)
        with self.embedding_write_session(project_root):
            self.put_content_embeddings(
                ((symbol.variant_id, text, vector),),
                model,
                project_root,
                build_scope=build_scope,
                configuration_id=configuration_id,
            )

    def put_embeddings(
        self,
        entries: Iterable[tuple[str, Sequence[float]]],
        model: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        configuration_id: str | None = None,
    ) -> None:
        """Validate and persist one embedding batch in a single transaction."""

        selected_entries = tuple(entries)
        content_entries: list[tuple[str, str, Sequence[float]]] = []
        for symbol_id, vector in selected_entries:
            symbol = self.get_symbol(symbol_id, project_root, build_scope=build_scope)
            if symbol is None or not symbol.variant_id:
                raise KeyError(f"symbol variant is not indexed: {symbol_id}")
            content_entries.append((symbol.variant_id, _embedding_text(symbol), vector))
        with self.embedding_write_session(project_root):
            self.put_content_embeddings(
                content_entries,
                model,
                project_root,
                build_scope=build_scope,
                configuration_id=configuration_id,
            )

    @contextmanager
    def embedding_write_session(self, project_root: Path | None = None) -> Iterator[None]:
        """Publish all embedding batches together or roll every batch back."""

        if self._connection.in_transaction:
            raise RuntimeError("embedding write sessions cannot be nested")
        project_id = self._project_id(project_root)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._delete_orphan_embedding_vectors(project_id)
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def attach_existing_embeddings(
        self,
        entries: Sequence[tuple[str, str]],
        model: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        configuration_id: str | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Attach content already in the vector pool and return content misses."""

        if not self._connection.in_transaction:
            raise RuntimeError("embedding attachment requires an embedding write session")
        if not entries:
            return ()
        project_id = self._project_id(project_root)
        configuration = configuration_id or model
        self._validate_embedding_variants(project_id, (entry[0] for entry in entries), build_scope)
        dimensions = self._embedding_configuration_dimensions(project_id, model, configuration)
        if not dimensions:
            return tuple(entries)
        if len(dimensions) != 1:
            raise RuntimeError(f"embedding configuration {configuration!r} has mixed dimensions")
        dimension = next(iter(dimensions))
        texts_by_hash: dict[str, str] = {}
        for _, text in entries:
            content_hash = _embedding_content_hash(text)
            previous = texts_by_hash.setdefault(content_hash, text)
            if previous != text:
                raise ValueError("embedding content hash collision")
        existing: set[str] = set()
        hashes = sorted(texts_by_hash)
        for offset in range(0, len(hashes), 500):
            chunk = hashes[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            for row in self._connection.execute(
                f"""
                SELECT content_hash, content_text FROM embedding_vectors
                WHERE project_id = ? AND model = ?
                  AND configuration_id = ? AND dimensions = ?
                  AND content_hash IN ({placeholders})
                """,
                (project_id, model, configuration, dimension, *chunk),
            ):
                if texts_by_hash[row["content_hash"]] != row["content_text"]:
                    raise ValueError("embedding content hash collision")
                existing.add(row["content_hash"])
        attached = [
            (
                project_id,
                variant_id,
                model,
                configuration,
                dimension,
                _embedding_content_hash(text),
            )
            for variant_id, text in entries
            if _embedding_content_hash(text) in existing
        ]
        self._connection.executemany(
            """
            INSERT INTO variant_embeddings(
                project_id, variant_id, model, configuration_id, dimensions, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, variant_id, model, configuration_id) DO UPDATE SET
                dimensions = excluded.dimensions,
                content_hash = excluded.content_hash
            """,
            attached,
        )
        return tuple(
            (variant_id, text)
            for variant_id, text in entries
            if _embedding_content_hash(text) not in existing
        )

    def put_content_embeddings(
        self,
        entries: Iterable[tuple[str, str, Sequence[float]]],
        model: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        configuration_id: str | None = None,
    ) -> None:
        """Validate and store one bounded content/vector batch in the active session."""

        if not self._connection.in_transaction:
            raise RuntimeError("embedding writes require an embedding write session")
        selected = tuple(entries)
        if not selected:
            return
        project_id = self._project_id(project_root)
        configuration = configuration_id or model
        self._validate_embedding_variants(project_id, (entry[0] for entry in selected), build_scope)
        expected_dimensions = self._embedding_configuration_dimensions(
            project_id, model, configuration
        )
        if len(expected_dimensions) > 1:
            raise RuntimeError(f"embedding configuration {configuration!r} has mixed dimensions")
        expected = next(iter(expected_dimensions)) if expected_dimensions else None
        vectors: dict[str, tuple[str, int, float, bytes]] = {}
        references: list[tuple[object, ...]] = []
        for variant_id, text, vector in selected:
            normalized = _validate_vector(vector)
            if expected is None:
                expected = len(normalized)
            if len(normalized) != expected:
                raise ValueError(
                    f"embedding model {model!r} already uses dimension "
                    f"{expected}, not {len(normalized)}"
                )
            content_hash = _embedding_content_hash(text)
            magnitude = math.sqrt(sum(value * value for value in normalized))
            encoded = struct.pack(f"<{len(normalized)}d", *normalized)
            previous = vectors.setdefault(
                content_hash,
                (text, len(normalized), magnitude, encoded),
            )
            if previous[0] != text:
                raise ValueError("embedding content hash collision")
            if previous[3] != encoded:
                raise ValueError("equal embedding inputs produced different vectors")
            references.append(
                (project_id, variant_id, model, configuration, len(normalized), content_hash)
            )
        for content_hash, (text, dimensions, magnitude, encoded) in vectors.items():
            existing = self._connection.execute(
                """
                SELECT content_text, vector FROM embedding_vectors
                WHERE project_id = ? AND model = ? AND configuration_id = ?
                  AND dimensions = ? AND content_hash = ?
                """,
                (project_id, model, configuration, dimensions, content_hash),
            ).fetchone()
            if existing is not None:
                if existing["content_text"] != text:
                    raise ValueError("embedding content hash collision")
                if existing["vector"] != encoded:
                    raise ValueError("equal embedding inputs produced different vectors")
                continue
            self._connection.execute(
                """
                INSERT INTO embedding_vectors(
                    project_id, model, configuration_id, dimensions, content_hash,
                    content_text, magnitude, vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    model,
                    configuration,
                    dimensions,
                    content_hash,
                    text,
                    magnitude,
                    encoded,
                ),
            )
        self._connection.executemany(
            """
            INSERT INTO variant_embeddings(
                project_id, variant_id, model, configuration_id, dimensions, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, variant_id, model, configuration_id) DO UPDATE SET
                dimensions = excluded.dimensions,
                content_hash = excluded.content_hash
            """,
            references,
        )

    def _embedding_configuration_dimensions(
        self, project_id: int, model: str, configuration_id: str
    ) -> set[int]:
        return {
            int(row[0])
            for row in self._connection.execute(
                """
                SELECT DISTINCT dimensions FROM embedding_vectors
                WHERE project_id = ? AND model = ? AND configuration_id = ?
                """,
                (project_id, model, configuration_id),
            )
        }

    def _validate_embedding_variants(
        self,
        project_id: int,
        variant_ids: Iterable[str],
        build_scope: BuildScope | tuple[str, ...] | None,
    ) -> None:
        selected = sorted(set(variant_ids))
        names = self._scope_names(build_scope)
        scope_placeholders = ",".join("?" for _ in names)
        found: set[str] = set()
        for offset in range(0, len(selected), 500):
            chunk = selected[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            found.update(
                row[0]
                for row in self._connection.execute(
                    f"""
                    SELECT id FROM symbol_variants
                    WHERE project_id = ? AND build_variant IN ({scope_placeholders})
                      AND id IN ({placeholders})
                    """,
                    (project_id, *names, *chunk),
                )
            )
        missing = set(selected) - found
        if missing:
            raise KeyError(f"symbol variant is not indexed: {min(missing)}")

    def _delete_orphan_embedding_vectors(self, project_id: int) -> None:
        self._connection.execute(
            """
            DELETE FROM embedding_vectors
            WHERE project_id = ? AND NOT EXISTS (
                SELECT 1 FROM variant_embeddings references_
                WHERE references_.project_id = embedding_vectors.project_id
                  AND references_.configuration_id = embedding_vectors.configuration_id
                  AND references_.model = embedding_vectors.model
                  AND references_.dimensions = embedding_vectors.dimensions
                  AND references_.content_hash = embedding_vectors.content_hash
            )
            """,
            (project_id,),
        )

    def missing_embedding_symbol_ids(
        self,
        model: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        configuration_id: str | None = None,
    ) -> tuple[str, ...]:
        """Return stable IDs whose current source snapshot has no vector for ``model``."""

        project_id = self._project_id(project_root)
        configuration = configuration_id or model
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
                     AND embeddings.configuration_id = ?
                    WHERE variants.project_id = ?
                      AND variants.build_variant IN ({placeholders})
                      AND embeddings.variant_id IS NULL
                    ORDER BY variants.build_variant, variants.symbol_id
                    """,
                    (model, configuration, project_id, *names),
                )
            )
        )

    def missing_embedding_variant_ids(
        self,
        model: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        configuration_id: str | None = None,
    ) -> tuple[str, ...]:
        project_id = self._project_id(project_root)
        configuration = configuration_id or model
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
                 AND embeddings.configuration_id = ?
                WHERE variants.project_id = ?
                  AND variants.build_variant IN ({placeholders})
                  AND embeddings.variant_id IS NULL
                ORDER BY variants.build_variant, variants.id
                """,
                (model, configuration, project_id, *names),
            )
        )

    def iter_missing_embedding_variant_id_batches(
        self,
        model: str,
        project_root: Path | None = None,
        *,
        build_scope: BuildScope | tuple[str, ...] | None = None,
        configuration_id: str | None = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> Iterator[tuple[str, ...]]:
        if batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        project_id = self._project_id(project_root)
        configuration = configuration_id or model
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        last_variant = ""
        while True:
            rows = self._connection.execute(
                f"""
                SELECT variants.id FROM symbol_variants variants
                LEFT JOIN variant_embeddings embeddings
                 ON embeddings.project_id = variants.project_id
                 AND embeddings.variant_id = variants.id
                 AND embeddings.model = ?
                 AND embeddings.configuration_id = ?
                WHERE variants.project_id = ?
                  AND variants.build_variant IN ({placeholders})
                  AND variants.id > ?
                  AND embeddings.variant_id IS NULL
                ORDER BY variants.id
                LIMIT ?
                """,
                (model, configuration, project_id, *names, last_variant, batch_size),
            ).fetchall()
            if not rows:
                return
            batch = tuple(row[0] for row in rows)
            yield batch
            last_variant = batch[-1]

    def embedding_count(
        self,
        model: str,
        project_root: Path | None = None,
        *,
        configuration_id: str | None = None,
    ) -> int:
        project_id = self._project_id(project_root)
        configuration = configuration_id or model
        return int(
            self._connection.execute(
                """
                SELECT count(*) FROM variant_embeddings
                WHERE project_id = ? AND model = ? AND configuration_id = ?
                """,
                (project_id, model, configuration),
            ).fetchone()[0]
        )

    def embedding_vector_count(
        self,
        model: str,
        project_root: Path | None = None,
        *,
        configuration_id: str | None = None,
    ) -> int:
        project_id = self._project_id(project_root)
        configuration = configuration_id or model
        return int(
            self._connection.execute(
                """
                SELECT count(*) FROM embedding_vectors
                WHERE project_id = ? AND model = ? AND configuration_id = ?
                """,
                (project_id, model, configuration),
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
        configuration_id: str | None = None,
    ) -> Sequence[SearchHit]:
        if limit <= 0:
            raise ValueError("vector search limit must be greater than zero")
        query_vector = _validate_vector(vector)
        query_magnitude = math.sqrt(sum(value * value for value in query_vector))
        project_id = self._project_id(project_root)
        configuration = configuration_id or model
        names = self._scope_names(build_scope)
        placeholders = ",".join("?" for _ in names)
        dimensions = {
            row[0]
            for row in self._connection.execute(
                f"""
                SELECT DISTINCT vectors.dimensions FROM variant_embeddings embeddings
                JOIN embedding_vectors vectors
                  ON vectors.project_id = embeddings.project_id
                 AND vectors.model = embeddings.model
                 AND vectors.configuration_id = embeddings.configuration_id
                 AND vectors.dimensions = embeddings.dimensions
                 AND vectors.content_hash = embeddings.content_hash
                JOIN symbol_variants variants
                  ON variants.project_id = embeddings.project_id
                 AND variants.id = embeddings.variant_id
                WHERE embeddings.project_id = ? AND embeddings.model = ?
                  AND embeddings.configuration_id = ?
                  AND variants.build_variant IN ({placeholders})
                """,
                (project_id, model, configuration, *names),
            )
        }
        if dimensions and dimensions != {len(query_vector)}:
            raise ValueError(
                f"query dimension {len(query_vector)} does not match model {model!r} "
                f"dimension {next(iter(dimensions))}"
            )
        query_blob = struct.pack(f"<{len(query_vector)}d", *query_vector)
        selected = self._connection.execute(
            f"""
            SELECT variants.id, variants.build_variant,
                   _cpp_context_cosine(
                       vectors.vector, vectors.magnitude, ?, ?
                   ) AS score
            FROM variant_embeddings embeddings
            JOIN embedding_vectors vectors
              ON vectors.project_id = embeddings.project_id
             AND vectors.model = embeddings.model
             AND vectors.configuration_id = embeddings.configuration_id
             AND vectors.dimensions = embeddings.dimensions
             AND vectors.content_hash = embeddings.content_hash
            JOIN symbol_variants variants
              ON variants.project_id = embeddings.project_id
             AND variants.id = embeddings.variant_id
            WHERE embeddings.project_id = ? AND embeddings.model = ?
              AND embeddings.configuration_id = ?
              AND embeddings.dimensions = ?
              AND variants.build_variant IN ({placeholders})
            ORDER BY score DESC, variants.build_variant, variants.id
            LIMIT ?
            """,
            (
                query_blob,
                query_magnitude,
                project_id,
                model,
                configuration,
                len(query_vector),
                *names,
                limit,
            ),
        ).fetchall()
        symbols = self.get_symbols(
            (row["id"] for row in selected),
            project_root,
            build_scope=build_scope,
        )
        return tuple(
            SearchHit(symbol, row["score"], f"vector:{model}")
            for row, symbol in zip(selected, symbols, strict=True)
            if symbol is not None
        )


def _embedding_text(symbol: CodeSymbol, max_text_chars: int = DEFAULT_EMBEDDING_TEXT_CHARS) -> str:
    return "\n".join(
        part
        for part in (
            symbol.qualified_name,
            symbol.signature,
            symbol.documentation,
            symbol.source_text,
        )
        if part
    )[:max_text_chars]


def _embedding_text_from_snapshot(
    snapshot_json: str, max_text_chars: int = DEFAULT_EMBEDDING_TEXT_CHARS
) -> str:
    snapshot = json.loads(snapshot_json)
    return "\n".join(
        str(part)
        for part in (
            snapshot.get("qualified_name", ""),
            snapshot.get("signature", ""),
            snapshot.get("documentation", ""),
            snapshot.get("source_text", ""),
        )
        if part
    )[:max_text_chars]


def _embedding_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values:
        raise ValueError("embedding vector must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding vector must contain only finite values")
    if not any(value != 0.0 for value in values):
        raise ValueError("embedding vector magnitude must be greater than zero")
    return values


@lru_cache(maxsize=128)
def _decode_vector_blob(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 8}d", blob)


def _sqlite_cosine(
    candidate_blob: bytes,
    candidate_magnitude: float,
    query_blob: bytes,
    query_magnitude: float,
) -> float:
    return sum(
        map(operator.mul, _decode_vector_blob(candidate_blob), _decode_vector_blob(query_blob))
    ) / (candidate_magnitude * query_magnitude)


def _cfg_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_CFG_PAGE_SIZE:
        raise ValueError(f"CFG page limit must be between 1 and {MAX_CFG_PAGE_SIZE}")
    return limit


def _call_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_CALL_PAGE_SIZE:
        raise ValueError(f"call page limit must be between 1 and {MAX_CALL_PAGE_SIZE}")
    return limit


def _summary_effect_order(item: SummaryEffect) -> tuple[int, int, str, str]:
    return (
        0 if item.certainty == DataFlowCertainty.CERTAIN else 1,
        0 if item.is_local else 1,
        item.kind.value,
        item.id,
    )


def _summary_origin_order(item: SummaryReturnOrigin) -> tuple[int, int, str, str]:
    return (
        0 if item.certainty == DataFlowCertainty.CERTAIN else 1,
        0 if item.is_local else 1,
        item.kind.value,
        item.id,
    )


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
    if value is None:
        return None
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


def _encode_summary_payload(
    summary_id: str,
    effects: Sequence[SummaryEffect],
    origins: Sequence[SummaryReturnOrigin],
) -> tuple[int, int, int, str, bytes]:
    """Return deterministic bounded metadata and bytes for one propagated solution."""

    if len(effects) + len(origins) > MAX_SUMMARY_PAYLOAD_RECORDS:
        raise SummaryPayloadError("summary payload record limit exceeded")
    effect_records: list[list[object]] = []
    for item in effects:
        if item.is_local or item.summary_id != summary_id:
            raise SummaryPayloadError("summary payload contains an invalid propagated effect")
        effect_records.append(
            [
                item.id,
                item.kind.value,
                item.location_kind.value,
                item.certainty.value,
                item.reason,
                item.parameter_index,
                list(item.access_path),
                item.location_id,
                item.source_access_id,
                item.via_callsite_id,
                item.target_symbol_id,
                item.translation_unit_id,
                item.build_configuration_id,
                item.build_variant,
            ]
        )
    origin_records: list[list[object]] = []
    for item in origins:
        if item.is_local or item.summary_id != summary_id:
            raise SummaryPayloadError("summary payload contains an invalid propagated origin")
        origin_records.append(
            [
                item.id,
                item.kind.value,
                item.certainty.value,
                item.reason,
                item.location_kind.value if item.location_kind else None,
                item.parameter_index,
                list(item.access_path),
                item.location_id,
                item.callsite_id,
                item.via_callsite_id,
                item.target_symbol_id,
                item.translation_unit_id,
                item.build_configuration_id,
                item.build_variant,
            ]
        )
    raw = json.dumps(
        [effect_records, origin_records],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(raw) > MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES:
        raise SummaryPayloadError("summary payload decompressed-size limit exceeded")
    compressed = zlib.compress(raw, level=6)
    if len(compressed) > MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES:
        raise SummaryPayloadError("summary payload compressed-size limit exceeded")
    return (
        len(effect_records),
        len(origin_records),
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        compressed,
    )


def _propagated_summary_groups(
    records: Iterable[SummaryEffect | SummaryReturnOrigin],
    selected_summary_ids: set[str],
) -> Iterator[tuple[str, tuple[SummaryEffect | SummaryReturnOrigin, ...]]]:
    """Group propagated solver output deterministically with one record list per group."""

    current_id: str | None = None
    current: list[SummaryEffect | SummaryReturnOrigin] = []
    selected = (
        item for item in records if not item.is_local and item.summary_id in selected_summary_ids
    )
    for item in sorted(selected, key=lambda value: (value.summary_id, value.id)):
        if current_id is not None and item.summary_id != current_id:
            yield current_id, tuple(current)
            current = []
        current_id = item.summary_id
        current.append(item)
    if current_id is not None:
        yield current_id, tuple(current)


def _ordered_summary_groups(
    records: Iterable[SummaryEffect | SummaryReturnOrigin],
) -> Iterator[tuple[str, tuple[SummaryEffect | SummaryReturnOrigin, ...]]]:
    """Group a relational stream while rejecting non-deterministic row order."""

    current_id: str | None = None
    current: list[SummaryEffect | SummaryReturnOrigin] = []
    previous_key: tuple[str, str] | None = None
    for item in records:
        key = (item.summary_id, item.id)
        if previous_key is not None and key < previous_key:
            raise SummaryPayloadError("stored propagated summaries are not ordered")
        if current_id is not None and item.summary_id != current_id:
            yield current_id, tuple(current)
            current = []
        current_id = item.summary_id
        current.append(item)
        previous_key = key
    if current_id is not None:
        yield current_id, tuple(current)


def _decode_summary_payload(
    summary_id: str,
    *,
    encoding: str,
    effect_count: int,
    origin_count: int,
    uncompressed_bytes: int,
    payload_hash: str,
    payload: bytes,
) -> tuple[tuple[SummaryEffect, ...], tuple[SummaryReturnOrigin, ...]]:
    """Validate and decode one payload without allowing unbounded allocation."""

    try:
        if encoding != SUMMARY_PAYLOAD_ENCODING:
            raise SummaryPayloadError("unsupported summary payload encoding")
        if not isinstance(effect_count, int) or not isinstance(origin_count, int):
            raise SummaryPayloadError("summary payload has invalid record counts")
        if effect_count < 0 or origin_count < 0:
            raise SummaryPayloadError("summary payload has invalid record counts")
        if effect_count + origin_count > MAX_SUMMARY_PAYLOAD_RECORDS:
            raise SummaryPayloadError("summary payload record limit exceeded")
        if not isinstance(uncompressed_bytes, int) or not (
            0 <= uncompressed_bytes <= MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES
        ):
            raise SummaryPayloadError("summary payload decompressed-size limit exceeded")
        if not isinstance(payload, bytes) or len(payload) > MAX_SUMMARY_PAYLOAD_COMPRESSED_BYTES:
            raise SummaryPayloadError("summary payload compressed-size limit exceeded")

        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload, MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES + 1)
        if decoder.unconsumed_tail or len(raw) > MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES:
            raise SummaryPayloadError("summary payload decompressed-size limit exceeded")
        remaining = MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES + 1 - len(raw)
        raw += decoder.flush(remaining)
        if len(raw) > MAX_SUMMARY_PAYLOAD_UNCOMPRESSED_BYTES:
            raise SummaryPayloadError("summary payload decompressed-size limit exceeded")
        if not decoder.eof or decoder.unused_data:
            raise SummaryPayloadError("summary payload compression stream is malformed")
        if len(raw) != uncompressed_bytes:
            raise SummaryPayloadError("summary payload decompressed size does not match metadata")
        if hashlib.sha256(raw).hexdigest() != payload_hash:
            raise SummaryPayloadError("summary payload hash does not match its contents")

        document = json.loads(raw)
        if not isinstance(document, list) or len(document) != 2:
            raise SummaryPayloadError("summary payload document has an invalid shape")
        effect_records, origin_records = document
        if not isinstance(effect_records, list) or not isinstance(origin_records, list):
            raise SummaryPayloadError("summary payload record groups have an invalid shape")
        if len(effect_records) != effect_count or len(origin_records) != origin_count:
            raise SummaryPayloadError("summary payload record counts do not match metadata")

        effects = tuple(
            _summary_effect_from_payload(summary_id, record) for record in effect_records
        )
        origins = tuple(
            _summary_origin_from_payload(summary_id, record) for record in origin_records
        )
        return effects, origins
    except SummaryPayloadError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, zlib.error) as error:
        raise SummaryPayloadError("summary payload is malformed") from error


def _summary_effect_from_payload(summary_id: str, record: object) -> SummaryEffect:
    values = _summary_payload_record(record, 14)
    return SummaryEffect(
        id=_payload_string(values[0]),
        summary_id=summary_id,
        kind=SummaryEffectKind(_payload_string(values[1])),
        location_kind=MemoryLocationKind(_payload_string(values[2])),
        certainty=DataFlowCertainty(_payload_string(values[3])),
        reason=_payload_string(values[4]),
        parameter_index=_payload_optional_index(values[5]),
        access_path=_payload_string_tuple(values[6]),
        location_id=_payload_optional_string(values[7]),
        source_access_id=_payload_optional_string(values[8]),
        is_local=False,
        via_callsite_id=_payload_optional_string(values[9]),
        target_symbol_id=_payload_optional_string(values[10]),
        translation_unit_id=_payload_string(values[11]),
        build_configuration_id=_payload_string(values[12]),
        build_variant=_payload_string(values[13]),
    )


def _summary_origin_from_payload(summary_id: str, record: object) -> SummaryReturnOrigin:
    values = _summary_payload_record(record, 14)
    location_kind = _payload_optional_string(values[4])
    return SummaryReturnOrigin(
        id=_payload_string(values[0]),
        summary_id=summary_id,
        kind=SummaryReturnOriginKind(_payload_string(values[1])),
        certainty=DataFlowCertainty(_payload_string(values[2])),
        reason=_payload_string(values[3]),
        location_kind=MemoryLocationKind(location_kind) if location_kind else None,
        parameter_index=_payload_optional_index(values[5]),
        access_path=_payload_string_tuple(values[6]),
        location_id=_payload_optional_string(values[7]),
        callsite_id=_payload_optional_string(values[8]),
        is_local=False,
        via_callsite_id=_payload_optional_string(values[9]),
        target_symbol_id=_payload_optional_string(values[10]),
        translation_unit_id=_payload_string(values[11]),
        build_configuration_id=_payload_string(values[12]),
        build_variant=_payload_string(values[13]),
    )


def _summary_payload_record(record: object, length: int) -> list[object]:
    if not isinstance(record, list) or len(record) != length:
        raise SummaryPayloadError("summary payload record has an invalid shape")
    return record


def _payload_string(value: object) -> str:
    if not isinstance(value, str):
        raise SummaryPayloadError("summary payload string field has an invalid type")
    return value


def _payload_optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _payload_string(value)


def _payload_optional_index(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SummaryPayloadError("summary payload index has an invalid value")
    return value


def _payload_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SummaryPayloadError("summary payload access path has an invalid shape")
    return tuple(value)
