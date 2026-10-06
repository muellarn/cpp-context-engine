import hashlib
import math
import sqlite3
import struct
import zlib

import pytest
from embedding_fixtures import materialize_v24_embeddings
from test_sqlite_storage import _batch, _open_migration_store

from cpp_context_engine.kicad_canary import _encode_digest_value, semantic_snapshot
from cpp_context_engine.storage.sqlite import SQLiteStore


def _seed(store, root):
    root.mkdir(parents=True, exist_ok=True)
    store.apply_ingestion(root, _batch(root))
    for model, configuration in (("first", "config"), ("second", "config"), ("first", "other")):
        store.put_embeddings(
            (("symbol-alpha", (1.0, 0.0)), ("file-a", (0.0, 1.0))),
            model,
            root,
            configuration_id=configuration,
        )


def test_embedding_attachments_intern_only_namespace_and_content_identity(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        connection = store._connection
        kinds = dict(connection.execute("SELECT name,type FROM sqlite_schema"))
        assert kinds["embedding_vectors"] == kinds["variant_embeddings"] == "view"
        assert [
            row[1] for row in connection.execute("PRAGMA table_info(embedding_attachment_records)")
        ] == ["project_id", "variant_id", "namespace_id", "content_id"]
        assert (
            next(
                row[4]
                for row in connection.execute("PRAGMA table_list")
                if row[1] == "embedding_attachment_records"
            )
            == 1
        )
        assert connection.execute("SELECT count(*) FROM embedding_namespaces").fetchone()[0] == 3
        assert (
            connection.execute("SELECT count(*) FROM embedding_content_records").fetchone()[0] == 6
        )
        assert store.embedding_count("first", configuration_id="config") == 2
        assert store.embedding_count("first", configuration_id="other") == 2
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_embedding_record_fks_enforce_project_and_namespace(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        other = tmp_path / "other-project"
        _seed(store, other)
        connection = store._connection
        records = connection.execute(
            "SELECT project_id,namespace_id,id FROM embedding_content_records ORDER BY id"
        ).fetchall()
        project, namespace, content = records[0]
        variant = connection.execute(
            "SELECT id FROM symbol_variants WHERE project_id=? LIMIT 1", (project,)
        ).fetchone()[0]
        for foreign in records:
            if foreign[0] == project and foreign[1] == namespace:
                continue
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
                connection.execute(
                    "UPDATE embedding_attachment_records SET content_id=? "
                    "WHERE project_id=? AND variant_id=? AND namespace_id=?",
                    (foreign[2], project, variant, namespace),
                )
            connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute("DELETE FROM embedding_content_records WHERE id=?", (content,))
        connection.rollback()


def test_semantic_snapshot_keeps_both_logical_embedding_views(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        result = semantic_snapshot(path, _connection=store._connection)
        assert result["counts"]["embedding_vectors"] == 6
        assert result["counts"]["variant_embeddings"] == 6
        assert not set(result["counts"]) & {
            "embedding_namespaces",
            "embedding_content_records",
            "embedding_attachment_records",
        }
        before = result["table_digests"]
        variant = store.get_symbol("symbol-alpha").variant_id
        with store.embedding_write_session():
            store.put_content_embeddings(
                ((variant, "replacement content", (0.0, 1.0)),),
                "first",
                configuration_id="config",
            )
        result = semantic_snapshot(path, _connection=store._connection)
        assert result["table_digests"]["embedding_vectors"] != before["embedding_vectors"]
        assert result["counts"]["variant_embeddings"] == 6


def test_missing_iterator_reloads_namespace_after_cleanup_and_recreation(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        store.apply_ingestion(tmp_path, _batch(tmp_path))
        variants = sorted(store.missing_embedding_variant_ids("first"))
        with store.embedding_write_session():
            store.put_content_embeddings(((variants[-1], "old", (1.0, 0.0)),), "first")
            store.put_content_embeddings(((variants[-1], "keep", (1.0, 0.0)),), "other")
        project = store._project_id()
        original = store._embedding_namespace(project, "first", "first")
        batches = store.iter_missing_embedding_variant_id_batches("first", batch_size=1)
        assert next(batches) == (variants[0],)
        with store._connection:
            store._connection.execute(
                "DELETE FROM embedding_attachment_records WHERE namespace_id=?", (original,)
            )
            store._delete_orphan_embedding_vectors(project)
        with store.embedding_write_session():
            store.put_content_embeddings(
                ((variant, "new", (1.0, 0.0)) for variant in variants), "first"
            )
        assert store._embedding_namespace(project, "first", "first") != original
        assert list(batches) == []


def test_semantic_vectors_do_not_sort_project_payloads(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        statements = []
        store._connection.set_trace_callback(statements.append)
        semantic_snapshot(tmp_path / "index.db", _connection=store._connection)
        store._connection.set_trace_callback(None)
        reads = [
            sql for sql in statements if 'FROM "embedding_vectors"' in sql and "ORDER BY" in sql
        ]
        assert len(reads) == 3
        for sql in reads:
            bytecode = list(store._connection.execute("EXPLAIN " + sql))
            assert not any(row[1].startswith("Sorter") for row in bytecode), sql


def _logical_rows(connection):
    return {
        table: [
            tuple(row)
            for row in connection.execute(
                f"SELECT * FROM {table} ORDER BY " + ",".join(map(str, range(1, width + 1)))
            )
        ]
        for table, width in (("embedding_vectors", 9), ("variant_embeddings", 6))
    }


def _legacy_embeddings(path, root):
    with SQLiteStore(path, project_root=root) as store:
        _seed(store, root)
        _seed(store, root / "other")
        ranking = store.search_vector((1.0, 0.0), model="first", configuration_id="config")
        materialize_v24_embeddings(store._connection)
        with store._connection:
            for model, configuration, dimension, encoding in (
                ("first", "config", 3, 0),
                ("名前", "ä-config", 32, 1),
                ("a-model", "z-config", 128, 1),
            ):
                vector = struct.pack(f"<{dimension}d", *([1.0] * dimension))
                store._connection.execute(
                    "INSERT INTO embedding_vectors VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        1,
                        model,
                        configuration,
                        dimension,
                        f"orphan-{dimension}",
                        "text\0名前",
                        math.sqrt(dimension),
                        encoding,
                        zlib.compress(vector) if encoding else vector,
                    ),
                )
        rows = _logical_rows(store._connection)
        snapshot = semantic_snapshot(path, _connection=store._connection)
        return rows, snapshot, ranking


def test_v24_namespace_migration_is_lossless_and_keys_survive_vacuum(tmp_path):
    path = tmp_path / "index.db"
    rows, before, ranking = _legacy_embeddings(path, tmp_path)
    with SQLiteStore(path, project_root=tmp_path) as store:
        assert _logical_rows(store._connection) == rows
        after = semantic_snapshot(path, _connection=store._connection)
        assert after == {**before, "schema_version": 25}
        assert store.search_vector((1.0, 0.0), model="first", configuration_id="config") == ranking
        keys = list(
            map(
                tuple,
                store._connection.execute(
                    "SELECT id,project_id,namespace_id,content_hash "
                    "FROM embedding_content_records ORDER BY id"
                ),
            )
        )
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert store._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        store._connection.execute("VACUUM")
    with SQLiteStore(path, project_root=tmp_path) as store:
        assert (
            list(
                map(
                    tuple,
                    store._connection.execute(
                        "SELECT id,project_id,namespace_id,content_hash "
                        "FROM embedding_content_records ORDER BY id"
                    ),
                )
            )
            == keys
        )
        assert _logical_rows(store._connection) == rows


@pytest.mark.parametrize(
    "stage", ["namespace-created", "namespace-copied", "namespace-validated", "commit"]
)
def test_namespace_migration_failure_keeps_exact_legacy_database(tmp_path, stage):
    path = tmp_path / "index.db"
    rows, _, _ = _legacy_embeddings(path, tmp_path)
    store = _open_migration_store(path, tmp_path)
    connection = store._connection
    schema = list(map(tuple, connection.execute("SELECT * FROM sqlite_schema ORDER BY name")))

    def fail(candidate):
        if candidate == stage:
            raise RuntimeError("injected namespace migration failure")

    store._embedding_migration_checkpoint = fail
    if stage == "commit":
        connection.set_authorizer(
            lambda action, value, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT"
                else sqlite3.SQLITE_OK
            )
        )
    try:
        with pytest.raises((RuntimeError, sqlite3.DatabaseError), match="injected|authorized"):
            store._migrate_v25()
        connection.set_authorizer(None)
        assert not connection.in_transaction
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 24
        assert (
            list(map(tuple, connection.execute("SELECT * FROM sqlite_schema ORDER BY name")))
            == schema
        )
        assert _logical_rows(connection) == rows
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


@pytest.mark.parametrize("operation", ["put", "attach"])
def test_attachment_last_input_wins_across_999_parameter_batches(tmp_path, operation):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        store.apply_ingestion(tmp_path, _batch(tmp_path))
        variant = store.missing_embedding_variant_ids("fixture")[0]
        store._connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        entries = tuple((variant, f"text-{index}", (1.0, 0.0)) for index in range(1050))
        with store.embedding_write_session():
            store.put_content_embeddings(entries, "fixture")
            if operation == "attach":
                assert (
                    store.attach_existing_embeddings(
                        tuple((v, text) for v, text, _ in reversed(entries)), "fixture"
                    )
                    == ()
                )
        text = store._connection.execute(
            "SELECT c.content_text FROM embedding_attachment_records a "
            "JOIN embedding_content_records c ON c.id=a.content_id"
        ).fetchone()[0]
        assert text == ("text-1049" if operation == "put" else "text-0")
        assert store.embedding_count("fixture") == store.embedding_vector_count("fixture") == 1
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("failure", ["provider", "orphan", "commit"])
def test_embedding_session_rolls_back_all_normalized_tables(tmp_path, monkeypatch, failure):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        expected = _logical_rows(store._connection)
        counts = {
            name: store._connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in (
                "embedding_namespaces",
                "embedding_content_records",
                "embedding_attachment_records",
            )
        }

        def fail(*_):
            raise RuntimeError("injected session failure")

        if failure != "provider":
            monkeypatch.setattr(
                store,
                "_delete_orphan_embedding_vectors"
                if failure == "orphan"
                else "_commit_embedding_session",
                fail,
            )
        with (
            pytest.raises(RuntimeError, match="injected session"),
            store.embedding_write_session(),
        ):
            store.put_content_embeddings(
                ((store.get_symbol("symbol-alpha").variant_id, "new", (1.0, 0.0)),), "new"
            )
            if failure == "provider":
                fail()
        assert not store._connection.in_transaction
        assert _logical_rows(store._connection) == expected
        for name, count in counts.items():
            assert store._connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0] == count


@pytest.mark.parametrize("parent", ["variant", "unit", "project"])
def test_normalized_embeddings_preserve_parent_cascades_and_orphan_cleanup(tmp_path, parent):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _seed(store, tmp_path)
        _seed(store, tmp_path / "other")
        project = store._project_id(tmp_path)
        with store._connection:
            if parent == "project":
                store._connection.execute("DELETE FROM projects WHERE id=?", (project,))
            elif parent == "unit":
                store._delete_translation_units(project, ("unit-a",))
            else:
                store._connection.execute(
                    "DELETE FROM symbol_variants WHERE project_id=? AND symbol_id='symbol-alpha'",
                    (project,),
                )
            store._delete_orphan_embedding_vectors(project)
        expected = 3 if parent == "variant" else 0
        for table in ("embedding_content_records", "embedding_attachment_records"):
            assert (
                store._connection.execute(
                    f"SELECT count(*) FROM {table} WHERE project_id=?", (project,)
                ).fetchone()[0]
                == expected
            )
            assert (
                store._connection.execute(
                    f"SELECT count(*) FROM {table} WHERE project_id!=?", (project,)
                ).fetchone()[0]
                == 6
            )
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_namespace_migration_rejects_dangling_content_without_dropping_rows(tmp_path):
    path = tmp_path / "index.db"
    _legacy_embeddings(path, tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE variant_embeddings SET content_hash='missing'")
        before = _logical_rows(connection)
    store = _open_migration_store(path, tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            store._migrate_v25()
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 24
        assert _logical_rows(store._connection) == before
        assert not store._connection.in_transaction
    finally:
        store._connection.close()


def test_semantic_namespace_order_matches_all_nine_columns_and_snapshot(tmp_path):
    path = tmp_path / "index.db"
    rows, _, _ = _legacy_embeddings(path, tmp_path)
    with SQLiteStore(path, project_root=tmp_path) as store:
        digest = hashlib.sha256(f"embedding_vectors\0{len(rows['embedding_vectors'])}\0".encode())
        for row in rows["embedding_vectors"]:
            for value in row:
                digest.update(_encode_digest_value(value))
            digest.update(b"\xff")
        before = semantic_snapshot(path, _connection=store._connection)
        assert before["table_digests"]["embedding_vectors"] == digest.hexdigest()
        with store._connection:
            store._connection.execute("PRAGMA defer_foreign_keys=ON")
            store._connection.execute("UPDATE embedding_namespaces SET id=10000-id")
            store._connection.execute(
                "UPDATE embedding_content_records SET namespace_id=10000-namespace_id"
            )
            store._connection.execute(
                "UPDATE embedding_attachment_records SET namespace_id=10000-namespace_id"
            )
        assert semantic_snapshot(path, _connection=store._connection) == before
        writer = sqlite3.connect(path)
        triggered = []

        def concurrent_change(sql):
            if 'FROM "embedding_vectors"' in sql and "ORDER BY" in sql and not triggered:
                triggered.append(True)
                writer.execute("UPDATE embedding_content_records SET content_text='changed'")
                writer.commit()

        store._connection.set_trace_callback(concurrent_change)
        try:
            assert semantic_snapshot(path, _connection=store._connection) == before
        finally:
            store._connection.set_trace_callback(None)
            writer.close()
        assert triggered
        assert not store._connection.in_transaction
        assert semantic_snapshot(path, _connection=store._connection) != before
