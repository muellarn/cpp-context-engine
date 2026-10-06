from __future__ import annotations

import json
import sqlite3
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

import cpp_context_engine.storage.sqlite as storage
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    BuildScope,
    BuildVariant,
    CodeSymbol,
    SearchQuery,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.storage.sqlite import SQLiteStore


def _batch(root: Path, unit_id: str, variant: str = "default") -> IngestionBatch:
    path = root / "shared.hpp"
    configuration = BuildConfiguration(
        id=f"configuration-{unit_id}",
        source_path=path,
        directory=root,
        arguments=("clang++", str(path)),
        command_hash=f"command-{unit_id}",
        build_variant=variant,
    )
    unit = TranslationUnit(
        id=unit_id,
        build_configuration_id=configuration.id,
        source_path=path,
        content_hash="source-hash",
        dependencies=((path, "source-hash"),),
        build_variant=variant,
    )
    shared = CodeSymbol(
        id="shared-symbol",
        qualified_name="zeta_größe",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(path, 1, 2048, 1, 30),
        signature="int zeta_größe(const char *名前)",
        documentation="Vollständiger Text: café, 名前, 🦉.",
        source_text="// café 名前 🦉\nint repeated_value = 42;\n" * 2048,
        source_hash="body-hash",
        build_configuration_id=configuration.id,
        translation_unit_id=unit.id,
        build_variant=variant,
        metadata={"is_definition": True, "nested": {"names": ["größe", "名前", "🦉"]}},
    )
    other = replace(
        shared,
        id="other-symbol",
        qualified_name="alpha",
        source_text="int alpha() { return 1; }",
        signature="int alpha()",
    )
    return IngestionBatch((configuration,), (unit,), (shared, other), (), ())


def _put(store: SQLiteStore, root: Path, batch: IngestionBatch) -> None:
    store.apply_ingestion(
        root,
        batch,
        build_variant=BuildVariant(
            batch.translation_units[0].build_variant, root / "commands.json"
        ),
    )


def test_large_unicode_snapshot_is_compressed_and_lossless(tmp_path: Path) -> None:
    batch = _batch(tmp_path, "one")
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _put(store, tmp_path, batch)
        snapshot = store._connection.execute(  # noqa: SLF001 - persisted representation contract
            "SELECT snapshot_json FROM symbol_variant_snapshots WHERE symbol_id = 'shared-symbol'"
        ).fetchone()[0]
        assert isinstance(snapshot, bytes), "symbol snapshots must not retain full JSON TEXT"
        assert len(snapshot) < len(batch.symbols[0].source_text.encode("utf-8")) // 4
        actual = store.symbols()[1]
        assert actual == replace(batch.symbols[0], variant_id=actual.variant_id)


def test_variants_order_search_reopen_update_delete_and_deep_parity(tmp_path: Path) -> None:
    database = tmp_path / "index.db"
    first = _batch(tmp_path, "one", "alpha")
    identical = _batch(tmp_path, "two", "alpha")
    different = _batch(tmp_path, "three", "beta")
    different = replace(
        different,
        symbols=(replace(different.symbols[0], source_text="int BetaOnly = 7;"),)
        + different.symbols[1:],
    )
    scope = BuildScope(("alpha", "beta"))
    with SQLiteStore(database, project_root=tmp_path) as store:
        for batch in (first, identical, different):
            _put(store, tmp_path, batch)
            store.validate_deep_navigation_parity(tmp_path, (batch,))
        before = store.symbols(build_scope=scope)
        assert len(before) == 6
        assert list(before) == sorted(
            before,
            key=lambda symbol: (symbol.qualified_name, symbol.build_variant, symbol.variant_id),
        )
        shared = [symbol for symbol in before if symbol.id == "shared-symbol"]
        assert len({symbol.variant_id for symbol in shared}) == 3
        assert len({symbol.source_text for symbol in shared}) == 2
        assert store.search(SearchQuery("BetaOnly"), build_scope=BuildScope.single("beta"))
        assert not store.search(SearchQuery("BetaOnly"), build_scope=BuildScope.single("alpha"))
        hits = store.search(SearchQuery("repeated_value"), build_scope=scope)
        store._rebuild_variant_fts()  # noqa: SLF001 - exercise compressed FTS rebuild
        assert store.search(SearchQuery("repeated_value"), build_scope=scope) == hits
        rows = store._connection.execute(  # noqa: SLF001 - embedding extraction equality
            "SELECT * FROM symbol_variant_snapshots"
        )
        for row in rows:
            snapshot = storage._full_variant_snapshot(row)
            symbol = store._snapshot_symbol(snapshot)  # noqa: SLF001
            assert storage._embedding_text_from_snapshot(snapshot) == storage._embedding_text(
                symbol
            )

    with SQLiteStore(database, project_root=tmp_path) as store:
        assert store.symbols(build_scope=scope) == before
        changed = replace(
            first,
            symbols=(replace(first.symbols[0], source_text="int UpdatedOnly = 9;"),)
            + first.symbols[1:],
        )
        _put(store, tmp_path, changed)
        store.validate_deep_navigation_parity(tmp_path, (changed, identical, different))
        assert store.search(SearchQuery("UpdatedOnly"), build_scope=scope)
        assert len(store.search(SearchQuery("repeated_value"), build_scope=scope)) == 1
        assert store.remove_build_variant("beta")
        assert len(store.symbols(build_scope=scope)) == 4
        assert not store.search(SearchQuery("BetaOnly"), build_scope=scope)


def test_compressed_bytes_are_not_semantic_identity(tmp_path: Path) -> None:
    batch = _batch(tmp_path, "one")
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _put(store, tmp_path, batch)
        row = store._connection.execute(  # noqa: SLF001
            "SELECT snapshot_id, snapshot_json FROM symbol_variant_snapshots "
            "WHERE symbol_id = 'shared-symbol'"
        ).fetchone()
        raw = storage._decode_symbol_snapshot(row[1]).encode("utf-8")
        alternate = (
            storage._SYMBOL_SNAPSHOT_ZLIB_V1 + struct.pack(">I", len(raw)) + zlib.compress(raw, 9)
        )
        assert alternate != row[1]
        store._connection.execute(  # noqa: SLF001
            "UPDATE symbol_snapshot_contents SET snapshot_json = ? WHERE id = ?",
            (alternate, row[0]),
        )
        store._connection.commit()  # noqa: SLF001
        store.put_embedding("shared-symbol", "fixture", [1.0, 0.0])
        store.validate_deep_navigation_parity(tmp_path, (batch,))
        project_id = store._project_id()  # noqa: SLF001
        with store._connection:  # noqa: SLF001
            store._put_symbol_variant(project_id, batch.symbols[0])  # noqa: SLF001
        assert store.embedding_count("fixture") == 1


def _materialize_v18_snapshots(store: SQLiteStore | sqlite3.Connection) -> None:
    """Build a real pre-pool table, preserving all FKs, FTS and public contents."""
    connection = store if isinstance(store, sqlite3.Connection) else store._connection
    connection.row_factory = sqlite3.Row
    if connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE name='symbol_variant_fts_source'"
    ).fetchone():
        # Legacy fixtures must also have the old internally stored FTS content;
        # keeping the new view would reference the tables this fixture replaces.
        connection.create_function(
            "_cpp_context_snapshot_field",
            2,
            lambda blob, field: json.loads(storage._decode_symbol_snapshot(blob))[field],
        )
        documents = [
            tuple(row)
            for row in connection.execute("SELECT rowid,* FROM symbol_variant_fts ORDER BY rowid")
        ]
        storage._execute_script(
            connection,
            """
            DROP TABLE symbol_variant_fts;
            DROP VIEW symbol_variant_fts_source;
            DROP TABLE symbol_variant_fts_rows;
            CREATE VIRTUAL TABLE symbol_variant_fts USING fts5(
                project_id UNINDEXED,variant_id UNINDEXED,symbol_id UNINDEXED,
                build_variant UNINDEXED,qualified_name,signature,documentation,source_text,
                tokenize='unicode61');
        """,
        )
        connection.executemany(
            "INSERT INTO symbol_variant_fts(rowid,project_id,variant_id,symbol_id,build_variant,"
            "qualified_name,signature,documentation,source_text) VALUES(?,?,?,?,?,?,?,?,?)",
            documents,
        )
        connection.commit()
    columns = {row[1] for row in connection.execute("PRAGMA table_info(symbol_variants)")}
    if "snapshot_id" not in columns:
        assert "snapshot_json" in columns
        return
    rows = [
        (*tuple(row)[:7], storage._full_variant_snapshot(row))
        for row in connection.execute("SELECT * FROM symbol_variant_snapshots ORDER BY id")
    ]
    indexes = [
        r[0]
        for r in connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type='index' AND tbl_name='symbol_variants' "
            "AND sql IS NOT NULL AND name != 'symbol_variants_snapshot'"
        )
    ]
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("BEGIN IMMEDIATE")
    storage._execute_script(
        connection,
        """
        DROP VIEW symbol_variant_snapshots;
        CREATE TABLE symbol_variants_v18 (
            project_id INTEGER NOT NULL, id TEXT NOT NULL, symbol_id TEXT NOT NULL,
            build_variant TEXT NOT NULL, build_configuration_id TEXT NOT NULL,
            translation_unit_id TEXT NOT NULL, is_definition INTEGER NOT NULL,
            snapshot_json TEXT NOT NULL,
            PRIMARY KEY(project_id,id),
            UNIQUE(project_id,build_variant,translation_unit_id,symbol_id),
            FOREIGN KEY(project_id,symbol_id) REFERENCES symbols(project_id,id) ON DELETE CASCADE,
            FOREIGN KEY(project_id,translation_unit_id)
                REFERENCES translation_units(project_id,id) ON DELETE CASCADE
        );
    """,
    )
    connection.executemany("INSERT INTO symbol_variants_v18 VALUES(?,?,?,?,?,?,?,?)", rows)
    storage._execute_script(
        connection,
        """
        DROP TABLE symbol_variants;
        ALTER TABLE symbol_variants_v18 RENAME TO symbol_variants;
        DROP TABLE symbol_snapshot_contents;
    """,
    )
    for statement in indexes:
        connection.execute(statement)
    connection.execute("PRAGMA user_version=18")
    connection.commit()
    connection.execute("PRAGMA foreign_keys=ON")


def _legacy_database(database: Path, root: Path) -> tuple[tuple[object, ...], ...]:
    with SQLiteStore(database, project_root=root) as store:
        _put(store, root, _batch(root, "one"))
        _materialize_v18_snapshots(store)
        for row in store._connection.execute(  # noqa: SLF001
            "SELECT rowid, snapshot_json FROM symbol_variants"
        ):
            store._connection.execute(  # noqa: SLF001
                "UPDATE symbol_variants SET snapshot_json = ? WHERE rowid = ?",
                (storage._decode_symbol_snapshot(row[1]), row[0]),
            )
        store._connection.execute("PRAGMA user_version = 17")  # noqa: SLF001
        store._connection.commit()  # noqa: SLF001
        return tuple(
            tuple(row)
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT * FROM symbol_variants ORDER BY rowid"
            )
        )


def test_v17_migration_preserves_exact_snapshots_and_fts(tmp_path: Path) -> None:
    database = tmp_path / "legacy.db"
    before = _legacy_database(database, tmp_path)
    with SQLiteStore(database, project_root=tmp_path) as store:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 22  # noqa: SLF001
        after = tuple(
            (*tuple(row)[:7], storage._full_variant_snapshot(row))
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT * FROM symbol_variant_snapshots ORDER BY id"
            )
        )
        assert all(
            isinstance(row[0], bytes)
            for row in store._connection.execute(
                "SELECT snapshot_json FROM symbol_snapshot_contents"
            )
        )
        assert sorted(after) == sorted(before)
        store.validate_deep_navigation_parity(tmp_path, (_batch(tmp_path, "one"),))
        assert store.search(SearchQuery("repeated_value"))
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001


@pytest.mark.parametrize("stage", ["row", "publication"])
def test_v18_migration_failure_rolls_back_rows_and_version(
    tmp_path: Path, monkeypatch, stage
) -> None:
    database = tmp_path / "legacy.db"
    before = _legacy_database(database, tmp_path)

    def fail(self, checkpoint):
        if checkpoint == stage:
            raise RuntimeError("injected migration failure")

    with monkeypatch.context() as patch:
        patch.setattr(SQLiteStore, "_snapshot_migration_checkpoint", fail)
        with pytest.raises(RuntimeError, match="injected migration failure"):
            SQLiteStore(database, project_root=tmp_path)
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 17
        assert tuple(connection.execute("SELECT * FROM symbol_variants ORDER BY rowid")) == before
    with SQLiteStore(database, project_root=tmp_path) as store:
        assert len(store.symbols()) == 2


def _encoded(raw: bytes, size: int | None = None) -> bytes:
    return (
        storage._SYMBOL_SNAPSHOT_ZLIB_V1
        + struct.pack(">I", len(raw) if size is None else size)
        + zlib.compress(raw)
    )


@pytest.mark.parametrize(
    "kind",
    [
        "header",
        "version",
        "truncated",
        "trailing",
        "concatenated",
        "corrupt",
        "size",
        "bomb",
        "utf8",
    ],
)
def test_decoder_rejects_invalid_compressed_snapshots(kind: str) -> None:
    good = _encoded(b'{"qualified_name":"test"}')
    payloads = {
        "header": b"CSS\x01",
        "version": b"CSS\x02" + good[4:],
        "truncated": good[:-1],
        "trailing": good + b"trailing",
        "concatenated": good + zlib.compress(b"second"),
        "corrupt": good[:8] + b"not-zlib",
        "size": _encoded(b"short", 100),
        "bomb": _encoded(b"x" * 100_000, 8),
        "utf8": _encoded(b"\xff"),
    }
    with pytest.raises(RuntimeError):
        storage._decode_symbol_snapshot(payloads[kind])


def test_snapshot_codec_limits_and_legacy_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "SYMBOL_SNAPSHOT_MAX_BYTES", 128)
    for payload in (_encoded(b"ok", 129), b"x" * 129, "x" * 129, "🦉" * 40):
        with pytest.raises(RuntimeError):
            storage._decode_symbol_snapshot(payload)
    for text in ("x" * 129, "🦉" * 40):
        with pytest.raises(RuntimeError):
            storage._encode_symbol_snapshot(text)
    assert storage._encode_symbol_snapshot("{}") == "{}"
    assert storage._decode_symbol_snapshot("{}") == "{}"
    text = json.dumps({"name": "🦉" * 20}, ensure_ascii=False)
    assert storage._decode_symbol_snapshot(storage._encode_symbol_snapshot(text)) == text


def test_compression_reduces_physical_database_without_changing_rows(
    tmp_path: Path, monkeypatch, record_property
) -> None:
    sizes = {}
    snapshots = {}
    for name, compressed in (("legacy", False), ("compressed", True)):
        database = tmp_path / f"{name}.db"
        with monkeypatch.context() as patch:
            if not compressed:
                patch.setattr(storage, "_encode_symbol_snapshot", lambda value: value)
            with SQLiteStore(database, project_root=tmp_path) as store:
                for index in range(6):
                    batch = _batch(tmp_path, str(index))
                    # Isolate compression from the separately tested pool deduplication.
                    batch = replace(
                        batch,
                        symbols=tuple(
                            replace(symbol, metadata={**symbol.metadata, "distinct": index})
                            for symbol in batch.symbols
                        ),
                    )
                    _put(store, tmp_path, batch)
                snapshots[name] = store.symbols()
                store._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # noqa: SLF001
                store._connection.execute("VACUUM")  # noqa: SLF001 - fair physical comparison
                store._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # noqa: SLF001
        sizes[name] = database.stat().st_size
    assert snapshots["compressed"] == snapshots["legacy"]
    record_property("legacy_database_bytes", sizes["legacy"])
    record_property("compressed_database_bytes", sizes["compressed"])
    assert sizes["compressed"] < sizes["legacy"] * 0.8, sizes
