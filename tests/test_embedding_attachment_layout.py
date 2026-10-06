import sqlite3

import pytest
from test_sqlite_storage import _batch

from cpp_context_engine.storage.sqlite import SQLiteStore


def _seed(store, root):
    store.apply_ingestion(root, _batch(root))
    store.put_embeddings((("symbol-alpha", (1.0, 0.0)), ("file-a", (0.0, 1.0))), "fixture")


def test_attachment_layout_avoids_redundant_prefix_index(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        names = {
            row[1] for row in store._connection.execute("PRAGMA index_list(variant_embeddings)")
        }
        assert "variant_embeddings_search" not in names
        assert "variant_embeddings_content" in names
        assert store.embedding_count("fixture") == 2
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_attachment_primary_key_is_the_only_table_storage(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        row = next(
            row
            for row in store._connection.execute("PRAGMA table_list")
            if row[1] == "variant_embeddings"
        )
        assert row[4] == 1, "attachment rows duplicate their wide primary keys"


def _legacy_layout(connection):
    definition = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE name='variant_embeddings'"
    ).fetchone()[0]
    connection.execute("ALTER TABLE variant_embeddings RENAME TO attachments_new")
    for name in ("variant_embeddings_content", "variant_embeddings_search"):
        connection.execute(f"DROP INDEX IF EXISTS {name}")
    connection.execute(definition.replace(" WITHOUT ROWID", ""))
    connection.execute("INSERT INTO variant_embeddings SELECT * FROM attachments_new")
    connection.execute("DROP TABLE attachments_new")
    for suffix, final in (("content", "content_hash"), ("search", "variant_id")):
        connection.execute(
            f"CREATE INDEX variant_embeddings_{suffix} ON variant_embeddings"
            f"(project_id,model,configuration_id,dimensions,{final})"
        )
    connection.execute("PRAGMA user_version=21")
    connection.commit()


def test_attachment_migration_preserves_rows_search_and_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        expected = store.search_vector((1.0, 0.0), model="fixture")
        _legacy_layout(store._connection)
        rows = [
            tuple(row)
            for row in store._connection.execute(
                "SELECT * FROM variant_embeddings "
                "ORDER BY project_id,variant_id,model,configuration_id"
            )
        ]
    original = SQLiteStore._migrate_v22

    def fail_index(store):
        store._connection.set_authorizer(
            lambda action, name, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_CREATE_INDEX and name == "variant_embeddings_content"
                else sqlite3.SQLITE_OK
            )
        )
        try:
            original(store)
        finally:
            store._connection.set_authorizer(None)
            store.close()

    with monkeypatch.context() as patch:
        patch.setattr(SQLiteStore, "_migrate_v22", fail_index)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            SQLiteStore(path, project_root=tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 21
        assert (
            connection.execute(
                "SELECT * FROM variant_embeddings "
                "ORDER BY project_id,variant_id,model,configuration_id"
            ).fetchall()
            == rows
        )
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='variant_embeddings_search'"
        ).fetchone()
    with SQLiteStore(path, project_root=tmp_path) as store:
        assert [
            tuple(row)
            for row in store._connection.execute(
                "SELECT * FROM variant_embeddings "
                "ORDER BY project_id,variant_id,model,configuration_id"
            )
        ] == rows
        assert store.search_vector((1.0, 0.0), model="fixture") == expected
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert store._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_attachment_queries_ignore_unselected_configurations(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        connection = store._connection
        costs = []
        expected = store.search_vector((1.0, 0.0), model="fixture")
        for upper in (32, 1024):
            for index in range(0 if upper == 32 else 32, upper):
                configuration = f"excluded-{index}"
                connection.execute(
                    "INSERT INTO embedding_vectors SELECT project_id,model,?,dimensions,"
                    "content_hash,content_text,magnitude,vector_encoding,vector "
                    "FROM embedding_vectors WHERE configuration_id='fixture'",
                    (configuration,),
                )
                connection.execute(
                    "INSERT INTO variant_embeddings SELECT project_id,variant_id,model,?,"
                    "dimensions,content_hash FROM variant_embeddings "
                    "WHERE configuration_id='fixture'",
                    (configuration,),
                )
            connection.commit()
            steps = 0

            def count():
                nonlocal steps
                steps += 1
                return 0

            connection.set_progress_handler(count, 1)
            try:
                assert store.embedding_count("fixture") == 2
                assert store.missing_embedding_variant_ids("fixture") == ()
                assert store.search_vector((1.0, 0.0), model="fixture") == expected
            finally:
                connection.set_progress_handler(None, 0)
            costs.append(steps)
        assert costs[1] <= costs[0] + 64, costs


def test_attachment_content_and_variant_lookups_are_indexed(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        connection = store._connection
        for predicate, expected in (
            (
                "project_id=1 AND variant_id='v' AND model='fixture' "
                "AND configuration_id='fixture'",
                "variant_id=?",
            ),
            (
                "project_id=1 AND model='fixture' AND configuration_id='fixture' "
                "AND dimensions=2 AND content_hash='h'",
                "content_hash=?",
            ),
            (
                "project_id=1 AND model='fixture' AND configuration_id='fixture'",
                "configuration_id=?",
            ),
        ):
            plan = [
                r[3]
                for r in connection.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM variant_embeddings WHERE " + predicate
                )
            ]
            assert any(expected in step for step in plan), plan
            assert not any("SCAN variant_embeddings" in step for step in plan), plan
        variant_id = connection.execute(
            "SELECT variant_id FROM variant_embeddings LIMIT 1"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO variant_embeddings SELECT * FROM variant_embeddings LIMIT 1"
            )
        connection.rollback()
        connection.execute("DELETE FROM symbol_variants WHERE id=?", (variant_id,))
        assert connection.execute("SELECT count(*) FROM variant_embeddings").fetchone()[0] == 1
