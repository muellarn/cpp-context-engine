import json
import sqlite3
from dataclasses import replace

import pytest
from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.kicad_canary import semantic_snapshot
from cpp_context_engine.models import OccurrenceKind, SourceSpan, SymbolOccurrence
from cpp_context_engine.storage.sqlite import SQLiteStore


def _legacy_occurrence_layout(connection):
    connection.execute("""CREATE TABLE occurrences_legacy (
        project_id INTEGER NOT NULL, translation_unit_id TEXT NOT NULL,
        id TEXT NOT NULL, symbol_id TEXT NOT NULL, enclosing_symbol_id TEXT,
        kind TEXT NOT NULL, path TEXT NOT NULL, start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL, start_column INTEGER NOT NULL, end_column INTEGER NOT NULL,
        build_variant TEXT NOT NULL, build_configuration_id TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        PRIMARY KEY(project_id,translation_unit_id,id),
        FOREIGN KEY(project_id,translation_unit_id)
            REFERENCES translation_units(project_id,id) ON DELETE CASCADE,
        FOREIGN KEY(project_id,symbol_id) REFERENCES symbols(project_id,id) ON DELETE CASCADE)""")
    connection.execute("""INSERT INTO occurrences_legacy
        SELECT o.* FROM occurrences o JOIN occurrence_units u
            ON u.project_id=o.project_id AND u.translation_unit_id=o.translation_unit_id
        JOIN occurrence_records r ON r.project_id=o.project_id AND r.unit_key=u.key AND r.id=o.id
        ORDER BY r.record_key""")
    connection.execute("DROP VIEW occurrences")
    for name in (
        "occurrence_records",
        "occurrence_contexts",
        "occurrence_units",
    ):
        connection.execute(f"DROP TABLE {name}")
    connection.execute("ALTER TABLE occurrences_legacy RENAME TO occurrences")
    connection.execute("CREATE INDEX occurrences_symbol ON occurrences(project_id,symbol_id)")
    connection.execute("PRAGMA user_version=23")
    connection.commit()


def _seed(store, root, unit="unit"):
    batch = _batch(root, unit)
    occurrence = SymbolOccurrence(
        "arbitrary ID λ",
        batch.symbols[0].id,
        SourceSpan(root / "grüße.hpp", 1, 2, 3, 4),
        OccurrenceKind.REFERENCE,
        enclosing_symbol_id="not a canonical symbol",
        translation_unit_id=unit,
        build_configuration_id="exact legacy mismatch",
        metadata={"unicode": "名前"},
    )
    _put(store, root, replace(batch, occurrences=(occurrence,)))
    return occurrence


def test_repeated_occurrence_context_uses_bounded_physical_storage(tmp_path):
    count = 8192
    unit = "tu_" + "a" * 32
    symbol = "symbol_" + "b" * 32
    batch = _batch(tmp_path, unit)
    symbols = (replace(batch.symbols[0], id=symbol), batch.symbols[1])
    occurrences = tuple(
        SymbolOccurrence(
            id=f"occ_{index:032x}",
            symbol_id=symbol,
            enclosing_symbol_id="legacy unenforced enclosing ID",
            kind=OccurrenceKind.REFERENCE,
            span=SourceSpan(tmp_path / "shared-header.hpp", index + 1, index + 1, 1, 4),
            translation_unit_id=unit,
            build_configuration_id="config_" + "c" * 32,
        )
        for index in range(count)
    )
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _put(store, tmp_path, replace(batch, symbols=symbols, occurrences=occurrences))
        assert store.occurrences(symbol) == occurrences
        occupied = store._connection.execute(
            "SELECT sum(pgsize) FROM dbstat WHERE name='occurrences' "
            "OR name LIKE 'occurrences_%' OR name LIKE 'occurrence_%' "
            "OR name LIKE 'sqlite_autoindex_occurrence%'"
        ).fetchone()[0]
        # Preserve full FK child lookup indexes and the direct symbol FK. The
        # budget requires about 40% savings over the measured 466-byte legacy row.
        assert occupied <= count * 280, (occupied, count, occupied / count)


def test_occurrence_migration_exact_provenance_enclosing_and_order_ties(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        item = _seed(store, tmp_path)
        with store._connection:
            store._put_occurrences(
                1, (replace(item, id="z"), replace(item, id="a", enclosing_symbol_id=None))
            )
        before = store.occurrences(item.symbol_id)
        _legacy_occurrence_layout(store._connection)
        rows = [
            tuple(r) for r in store._connection.execute("SELECT * FROM occurrences ORDER BY rowid")
        ]
    with SQLiteStore(path, project_root=tmp_path) as store:
        assert store.occurrences(item.symbol_id) == before
        assert [tuple(r) for r in store._connection.execute("SELECT * FROM occurrences")] == rows
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []
        store._connection.execute("DELETE FROM symbols WHERE id=?", (item.symbol_id,))
        assert store.occurrences(item.symbol_id) == ()
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_occurrence_replace_identity_ignores_context_across_chunks_and_rolls_back(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        item = _seed(store, tmp_path)
        store._connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        project = store._project_id(tmp_path)
        final = replace(
            item,
            build_configuration_id="different",
            enclosing_symbol_id=None,
            span=replace(item.span, path=tmp_path / "other.hpp"),
            metadata={"final": True},
        )
        with store._connection:
            store._put_occurrences(project, (replace(item, id=f"row-{i}") for i in range(600)))
            store._put_occurrences(project, (item, final))
        assert len(store.occurrences(item.symbol_id)) == 601
        assert final in store.occurrences(item.symbol_id)
        counts = [
            store._connection.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in (
                "occurrence_records",
                "occurrence_contexts",
                "occurrence_units",
            )
        ]

        def failing():
            yield from (
                replace(item, id=f"new-{i}", build_configuration_id="fresh") for i in range(600)
            )
            raise RuntimeError("incomplete stream")

        with pytest.raises(RuntimeError, match="incomplete stream"), store._connection:
            store._put_occurrences(project, failing())
        assert counts == [
            store._connection.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in (
                "occurrence_records",
                "occurrence_contexts",
                "occurrence_units",
            )
        ]
        # Multirow REPLACE checks the final statement state, not an overwritten
        # intermediate symbol. This succeeds on the legacy storage too.
        with store._connection:
            store._put_occurrences(project, (replace(item, symbol_id="absent"), final))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"), store._connection:
            store._put_occurrences(project, (replace(item, symbol_id="absent"),))


def test_occurrence_project_and_unit_context_isolation_and_bulk_delete(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        item = _seed(store, tmp_path)
        _seed(store, tmp_path, "other-unit")
        other = tmp_path / "other"
        _seed(store, other)
        other_key = store._connection.execute(
            "SELECT key FROM occurrence_contexts WHERE project_id=2"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"), store._connection:
            store._connection.execute(
                "UPDATE occurrence_records SET context_key=? WHERE project_id=1", (other_key,)
            )
        keys = [
            r[0]
            for r in store._connection.execute(
                "SELECT key FROM occurrence_units WHERE project_id=1 ORDER BY key"
            )
        ]
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"), store._connection:
            store._connection.execute(
                "UPDATE occurrence_records SET unit_key=?,id=id||'-moved' WHERE unit_key=?",
                (keys[1], keys[0]),
            )
        with store._connection:
            store._delete_translation_units(1, ("unit",))
        assert len(store.occurrences(item.symbol_id)) == 1
        assert len(store.occurrences(item.symbol_id, project_root=other)) == 1
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_occurrence_migration_ddl_failure_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        _legacy_occurrence_layout(store._connection)
        before = [tuple(r) for r in store._connection.execute("SELECT * FROM occurrences")]
    original = SQLiteStore._migrate_v24

    def fail(store):
        store._connection.set_authorizer(
            lambda action, name, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_CREATE_INDEX and name == "occurrences_symbol"
                else sqlite3.SQLITE_OK
            )
        )
        try:
            original(store)
        finally:
            store._connection.set_authorizer(None)
            store.close()

    monkeypatch.setattr(SQLiteStore, "_migrate_v24", fail)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        SQLiteStore(path, project_root=tmp_path)
    with sqlite3.connect(path) as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == 23
        assert c.execute("SELECT * FROM occurrences").fetchall() == before
        assert not c.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='occurrence_records'"
        ).fetchone()
        assert c.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


def test_occurrence_semantic_digest_uses_all_logical_values_not_mapping_keys(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        before = semantic_snapshot(path)
        assert before["counts"]["occurrences"] == 1
        with store._connection:
            store._connection.execute("PRAGMA defer_foreign_keys=ON")
            store._connection.execute("UPDATE occurrence_units SET key=key+1000")
            store._connection.execute(
                "UPDATE occurrence_contexts SET unit_key=unit_key+1000,key=key+1000"
            )
            store._connection.execute(
                "UPDATE occurrence_records SET unit_key=unit_key+1000,"
                "context_key=context_key+1000,record_key=record_key+1000"
            )
        assert semantic_snapshot(path) == before
        with store._connection:
            store._connection.execute("UPDATE occurrence_records SET metadata_json='{}'")
        assert (
            semantic_snapshot(path)["table_digests"]["occurrences"]
            != before["table_digests"]["occurrences"]
        )


def test_occurrence_query_constrains_symbol_before_visiting_rows(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        item = _seed(store, tmp_path)
        sql = []
        store._connection.set_trace_callback(sql.append)
        store.occurrences(item.symbol_id)
        store._connection.set_trace_callback(None)
        query = next(s for s in sql if "FROM occurrence_records r" in s)
        plan = "\n".join(r[3] for r in store._connection.execute("EXPLAIN QUERY PLAN " + query))
        assert "SEARCH r USING INDEX occurrences_symbol (project_id=? AND symbol_id=?)" in plan


@pytest.mark.parametrize("mode", ["immediate", "deferred", "off"])
@pytest.mark.parametrize("split", [False, True])
def test_replaced_missing_symbol_matches_legacy_statement_boundaries(tmp_path, mode, split):
    outcomes = []
    for legacy in (True, False):
        root = tmp_path / "project"
        with SQLiteStore(tmp_path / f"{legacy}.db", project_root=root) as store:
            item = _seed(store, root)
            c = store._connection
            if legacy:
                _legacy_occurrence_layout(c)
            c.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
            if mode == "off":
                c.execute("PRAGMA foreign_keys=OFF")
            elif mode == "deferred":
                c.execute("PRAGMA defer_foreign_keys=ON")
            items = [replace(item, symbol_id="absent")]
            if split:
                items.extend(replace(item, id=f"filler-{i}") for i in range(72))
            items.append(item)
            failed = False
            try:
                with c:
                    if legacy:
                        store._insert_rows(
                            "INSERT OR REPLACE INTO occurrences VALUES {values}",
                            (
                                (
                                    1,
                                    o.translation_unit_id,
                                    o.id,
                                    o.symbol_id,
                                    o.enclosing_symbol_id,
                                    o.kind.value,
                                    str(o.span.path),
                                    o.span.start_line,
                                    o.span.end_line,
                                    o.span.start_column,
                                    o.span.end_column,
                                    o.build_variant,
                                    o.build_configuration_id,
                                    json.dumps(dict(o.metadata), sort_keys=True),
                                )
                                for o in items
                            ),
                            columns=14,
                        )
                    else:
                        store._put_occurrences(1, items)
            except sqlite3.IntegrityError:
                failed = True
            assert c.execute("PRAGMA foreign_key_check").fetchall() == []
            outcomes.append(
                (
                    failed,
                    [
                        tuple(r)
                        for r in c.execute(
                            "SELECT * FROM occurrences ORDER BY project_id,translation_unit_id,id"
                        )
                    ],
                )
            )
    assert outcomes[0] == outcomes[1]


def test_occurrence_tu_plan_is_bounded_and_ties_survive_reopen_vacuum(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        item = _seed(store, tmp_path)
        with store._connection:
            store._put_occurrences(
                1, (replace(item, id="z"), replace(item, id="a"), replace(item, id="z"))
            )
        before = store.occurrences(item.symbol_id)
        plan = "\n".join(
            r[3]
            for r in store._connection.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM occurrences "
                "WHERE project_id=? AND translation_unit_id=?",
                (1, "unit"),
            )
        )
        assert "translation_unit_id=?" in plan
        assert "unit_key=?" in plan
        store._connection.execute("VACUUM")
    with SQLiteStore(path, project_root=tmp_path) as store:
        assert store.occurrences(item.symbol_id) == before


@pytest.mark.parametrize(
    "field", ["translation_unit_id", "build_configuration_id", "build_variant"]
)
def test_missing_required_occurrence_context_cannot_be_silently_ignored(tmp_path, field):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        item = _seed(store, tmp_path)
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"), store._connection:
            store._put_occurrences(1, (replace(item, **{field: None}), item))
        assert store.occurrences(item.symbol_id) == (item,)
