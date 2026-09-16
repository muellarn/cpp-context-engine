from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from test_deep_storage_semantics import _two_tu_summary_batch
from test_streaming_indexer import _database, _semantic_dump

from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion.deep import DeepMaterializer, MaterializeDeepRequest
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.ingestion.native import (
    NativeAnalyzerClient,
    NativeClangIngestor,
    _file_digest,
)
from cpp_context_engine.models import BuildVariant, IndexProfile
from cpp_context_engine.storage.sqlite import SQLiteStore


@pytest.mark.parametrize("profile", [IndexProfile.FULL, IndexProfile.NAVIGATION])
@pytest.mark.parametrize("legacy", [False, True])
def test_native_binary_upgrade_reindexes_unchanged_tus(
    tmp_path: Path, monkeypatch, profile, legacy
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    cdb = _database(root, 1)
    binary = tmp_path / "analyzer"
    binary.write_bytes(b"analyzer A")
    client = NativeAnalyzerClient(binary, profile=profile)
    monkeypatch.setattr(client, "probe", lambda: object())
    monkeypatch.setattr(client, "analyze_stream", lambda *_args, **_kwargs: None)
    old_identity = _file_digest(binary)
    ingestor = NativeClangIngestor(client, profile=profile)
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        indexer = ProjectIndexer(ingestor, store, profile=profile)
        for name in ("debug", "release"):
            assert (
                indexer.index(
                    root, cdb, build_variant=BuildVariant(name, cdb)
                ).indexed_translation_units
                == 1
            )
        if legacy:
            store._connection.execute("UPDATE translation_units SET analyzer_identity = ''")  # noqa: SLF001
            store._connection.commit()  # noqa: SLF001
        binary.write_bytes(b"analyzer B")
        result = indexer.index(root, cdb, build_variant=BuildVariant("debug", cdb))
        assert result.indexed_translation_units == 1
        assert result.skipped_translation_units == 0
        assert (
            indexer.index(
                root, cdb, build_variant=BuildVariant("debug", cdb)
            ).skipped_translation_units
            == 1
        )
        states = store.translation_unit_states(root, build_scope=("debug", "release"))
        assert {
            state.analyzer_identity for state in states.values() if state.build_variant == "debug"
        } == {_file_digest(binary)}
        assert {
            state.analyzer_identity for state in states.values() if state.build_variant == "release"
        } == {"" if legacy else old_identity}


@pytest.mark.parametrize("producer", ["legacy", "old", "current", "race"])
def test_full_deep_reuse_checks_producer(tmp_path: Path, monkeypatch, producer) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _navigation, deep = _two_tu_summary_batch(root)
    binary = tmp_path / "analyzer"
    binary.write_bytes(b"current analyzer")
    identity = _file_digest(binary)
    stored_identity = {"legacy": "", "old": "old-analyzer"}.get(producer, identity)
    deep = replace(
        deep,
        translation_units=tuple(
            replace(unit, analyzer_identity=stored_identity) for unit in deep.translation_units
        ),
    )
    cdb = root / "compile_commands.json"
    cdb.write_text("[]")
    config = AppConfig(
        project_root=root,
        index_directory=tmp_path,
        database_path=tmp_path / "index.db",
        compilation_database=cdb,
        clang_analyzer_path=binary,
        index_profile=IndexProfile.FULL,
    )
    with SQLiteStore(config.database_path, project_root=root) as store:
        store.apply_ingestion(root, deep)
        materializer = DeepMaterializer(config, store)
        monkeypatch.setattr(materializer, "_revalidate", lambda *_args: None)
        monkeypatch.setattr(
            materializer,
            "_load_configurations",
            lambda *_args: {item.id: item for item in deep.build_configurations},
        )
        if producer == "race":
            publish = store.publish_full_profile_materialization

            def changed_before_lock(*args, **kwargs):
                store._connection.execute(
                    "UPDATE translation_units SET analyzer_identity = 'raced'"
                )  # noqa: SLF001
                store._connection.commit()  # noqa: SLF001
                return publish(*args, **kwargs)

            monkeypatch.setattr(store, "publish_full_profile_materialization", changed_before_lock)
        if producer == "current":
            result = materializer.materialize(MaterializeDeepRequest(symbol_id="caller", max_tus=2))
            assert result.cache_hit and result.provenance.analyzer_identity == identity
            return
        with pytest.raises(RuntimeError, match="full index changed|analyzer.*refresh"):
            materializer.materialize(MaterializeDeepRequest(symbol_id="caller", max_tus=2))
        assert store._connection.execute("SELECT count(*) FROM deep_tu_cache").fetchone()[0] == 0  # noqa: SLF001
        assert {
            state.analyzer_identity for state in store.translation_unit_states(root).values()
        } == {"raced" if producer == "race" else stored_identity}


def test_binary_change_during_generation_rolls_back(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    cdb = _database(root, 1)
    binary = tmp_path / "analyzer"
    binary.write_bytes(b"A")
    client = NativeAnalyzerClient(binary)
    monkeypatch.setattr(client, "probe", lambda: object())
    monkeypatch.setattr(client, "analyze_stream", lambda *_args, **_kwargs: None)
    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        indexer = ProjectIndexer(NativeClangIngestor(client), store)
        indexer.index(root, cdb)
        before = _semantic_dump(store)
        binary.write_bytes(b"B")
        monkeypatch.setattr(
            client, "analyze_stream", lambda *_args, **_kwargs: binary.write_bytes(b"C")
        )
        with pytest.raises(RuntimeError, match="analyzer changed during indexing"):
            indexer.index(root, cdb)
        assert _semantic_dump(store) == before


@pytest.mark.parametrize("deny_version_commit", [False, True])
def test_v16_migration_preserves_unknown_tu_provenance(tmp_path: Path, deny_version_commit) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _navigation, deep = _two_tu_summary_batch(root)
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=root) as store:
        store.apply_ingestion(root, deep)
        store._connection.execute("ALTER TABLE translation_units DROP COLUMN analyzer_identity")  # noqa: SLF001
        store._connection.execute("PRAGMA user_version = 16")  # noqa: SLF001
        store._connection.commit()  # noqa: SLF001
        if deny_version_commit:
            connection = store._connection  # noqa: SLF001 - inject failure after ALTER TABLE
            connection.set_authorizer(
                lambda action, first, second, *_: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_PRAGMA
                    and first == "user_version"
                    and second == "17"
                    else sqlite3.SQLITE_OK
                )
            )
            with pytest.raises(sqlite3.DatabaseError, match="authorized"):
                store._migrate_v17()  # noqa: SLF001
            connection.set_authorizer(None)
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 16
            assert "analyzer_identity" not in {
                row[1] for row in connection.execute("PRAGMA table_info(translation_units)")
            }
    with SQLiteStore(path, project_root=root) as store:
        states = store.translation_unit_states(root)
        assert len(states) == 2
        assert all(state.analyzer_identity == "" for state in states.values())
        assert all(state.index_profile is IndexProfile.FULL for state in states.values())
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 17  # noqa: SLF001
