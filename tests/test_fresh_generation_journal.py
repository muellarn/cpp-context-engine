from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from cpp_context_engine import runtime
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion.compilation_database import translation_unit_id
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.kicad_canary import _database_size, _directory_size
from cpp_context_engine.models import (
    BuildConfiguration,
    CodeSymbol,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.storage import SQLiteStore


def _project(tmp_path: Path) -> AppConfig:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "large.cpp"
    source.write_text("int value() { return 1; }\n", encoding="utf-8")
    cdb = root / "compile_commands.json"
    cdb.write_text(
        json.dumps(
            [{"directory": str(root), "file": str(source), "arguments": ["c++", str(source)]}]
        ),
        encoding="utf-8",
    )
    return AppConfig(
        project_root=root,
        compilation_database=cdb,
        index_directory=tmp_path / "output",
        database_path=tmp_path / "output" / "index.db",
        embedding_dimensions=16,
    )


class _PayloadIngestor:
    analysis_backend = "fixture"
    advanced_facts_complete = False

    def iter_configuration_batches(self, _root: Path, configurations):
        for configuration in configurations:
            yield self.batch(configuration)

    @staticmethod
    def batch(configuration: BuildConfiguration) -> IngestionBatch:
        source = configuration.source_path
        content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        unit = TranslationUnit(
            translation_unit_id(configuration), configuration.id, source, content_hash
        )
        symbol = CodeSymbol(
            "large-symbol",
            "large_symbol",
            SymbolKind.FUNCTION,
            SourceSpan(source, 1, 1),
            source_text="// payload " + "x" * (2 * 1024 * 1024),
            source_hash=content_hash,
            build_configuration_id=configuration.id,
            translation_unit_id=unit.id,
            metadata={"is_definition": True},
        )
        return IngestionBatch((configuration,), (unit,), (symbol,), (), ())


def test_fresh_index_does_not_duplicate_committed_database_in_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _project(tmp_path)
    measurements: list[tuple[int, int]] = []

    class MeasuredStore(SQLiteStore):
        def apply_ingestion_batches(self, *args, **kwargs):
            result = super().apply_ingestion_batches(*args, **kwargs)
            connection = self._connection
            logical = (
                connection.execute("PRAGMA page_count").fetchone()[0]
                * connection.execute("PRAGMA page_size").fetchone()[0]
            )
            measurements.append((_database_size(self.path), logical))
            return result

    monkeypatch.setattr(runtime, "SQLiteStore", MeasuredStore)
    monkeypatch.setattr(runtime, "ClangIngestor", lambda **_kwargs: _PayloadIngestor())

    runtime.index_project(config)

    assert measurements
    for active, logical in measurements:
        assert active < logical + 1024 * 1024, (active, logical)
    with closing(sqlite3.connect(config.database_path)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_active_database_budget_includes_rollback_journal(tmp_path: Path) -> None:
    database = tmp_path / "index.db"
    for suffix, size in (("", 100), ("-wal", 200), ("-shm", 300), ("-journal", 400)):
        Path(f"{database}{suffix}").write_bytes(b"x" * size)

    assert _database_size(database) == 1000


def test_private_success_publishes_only_after_wal_restore_and_close(tmp_path: Path) -> None:
    database = tmp_path / "index.db"
    with SQLiteStore.indexing_generation(database) as store:
        staged = store.path
        assert staged != database
        assert staged.parent.stat().st_mode & 0o777 == 0o700
        assert not database.exists()
        assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        with store._connection:
            store._connection.execute("CREATE TABLE proof(value INTEGER)")
            store._connection.execute("INSERT INTO proof VALUES (7)")
        assert not database.exists()

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        store._connection.execute("SELECT 1")
    assert not staged.parent.exists()
    with closing(sqlite3.connect(database)) as reader:
        assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert reader.execute("SELECT value FROM proof").fetchone() == (7,)


def test_failed_private_generation_rolls_back_and_retains_wal_evidence(tmp_path: Path) -> None:
    database = tmp_path / "index.db"
    with (
        pytest.raises(RuntimeError, match="injected"),
        SQLiteStore.indexing_generation(database) as store,
    ):
        staged = store.path
        store._connection.execute("BEGIN IMMEDIATE")
        store._connection.execute("CREATE TABLE uncommitted(value INTEGER)")
        store._connection.execute("INSERT INTO uncommitted VALUES (7)")
        raise RuntimeError("injected generation failure")

    assert not database.exists()
    assert staged.is_file()
    with closing(sqlite3.connect(staged)) as reader:
        assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert reader.execute(
            "SELECT count(*) FROM sqlite_master WHERE name = 'uncommitted'"
        ).fetchone() == (0,)
    assert _database_size(database) == staged.stat().st_size


@pytest.mark.parametrize("existing_rows", [False, True])
def test_existing_even_empty_database_keeps_wal_and_readers(
    tmp_path: Path, existing_rows: bool
) -> None:
    database = tmp_path / "index.db"
    with SQLiteStore(database) as setup, setup._connection:
        setup._connection.execute("CREATE TABLE proof(value INTEGER)")
        if existing_rows:
            setup._connection.execute("INSERT INTO proof VALUES (1)")
    with closing(sqlite3.connect(database)) as reader:
        reader.execute("BEGIN")
        initial = reader.execute("SELECT count(*) FROM proof").fetchone()[0]
        with SQLiteStore.indexing_generation(database) as writer:
            assert writer.path == database
            assert writer._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            with writer._connection:
                writer._connection.execute("INSERT INTO proof VALUES (2)")
            assert reader.execute("SELECT count(*) FROM proof").fetchone()[0] == initial
        reader.rollback()
        assert reader.execute("SELECT count(*) FROM proof").fetchone()[0] == initial + 1
    assert not list(tmp_path.glob(".index.db.fresh-*"))


def test_destination_created_during_index_is_never_overwritten(tmp_path: Path) -> None:
    database = tmp_path / "index.db"
    with pytest.raises(FileExistsError), SQLiteStore.indexing_generation(database) as store:
        staged = store.path
        database.write_bytes(b"other writer's generation")

    assert database.read_bytes() == b"other writer's generation"
    assert staged.is_file()
    with closing(sqlite3.connect(staged)) as reader:
        assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_unowned_destination_sidecars_prevent_publication(tmp_path: Path, suffix: str) -> None:
    database = tmp_path / "index.db"
    sidecar = Path(f"{database}{suffix}")
    sidecar.write_bytes(b"other writer")
    with pytest.raises(FileExistsError, match="unowned"), SQLiteStore.indexing_generation(database):
        pytest.fail("must reject before creating a private database")
    assert sidecar.read_bytes() == b"other writer"
    assert not database.exists()


def test_private_database_and_journal_are_counted_once_across_publication(tmp_path: Path) -> None:
    database = tmp_path / "index.db"
    private = tmp_path / ".index.db.fresh-fixture"
    private.mkdir()
    staged = private / database.name
    staged.write_bytes(b"x" * 100)
    Path(f"{staged}-journal").write_bytes(b"j" * 200)
    assert _database_size(database) == 300
    os.link(staged, database)
    assert _database_size(database) == 300
    assert _directory_size(tmp_path) == 300
