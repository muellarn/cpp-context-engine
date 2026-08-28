from __future__ import annotations

import hashlib
import json
import threading
import weakref
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    translation_unit_id,
)
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    BuildVariant,
    CodeSymbol,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.storage.sqlite import SQLiteStore


def _database(root: Path, count: int) -> Path:
    entries = []
    for index in range(count):
        source = root / f"unit-{index:03d}.cpp"
        source.write_text(f"int value_{index} = {index};\n", encoding="utf-8")
        entries.append(
            {
                "directory": str(root),
                "file": str(source),
                "arguments": ["clang++", "-c", str(source)],
            }
        )
    path = root / "compile_commands.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _batch(configuration: BuildConfiguration, *, payload_bytes: int = 0) -> IngestionBatch:
    source = configuration.source_path
    content = source.read_bytes()
    unit_id = translation_unit_id(configuration)
    unit = TranslationUnit(
        id=unit_id,
        build_configuration_id=configuration.id,
        source_path=source,
        content_hash=hashlib.sha256(content).hexdigest(),
        dependencies=((source, hashlib.sha256(content).hexdigest()),),
        build_variant=configuration.build_variant,
        analysis_backend="stream-fixture",
        advanced_facts_complete=True,
    )
    symbol = CodeSymbol(
        id=f"symbol-{configuration.id}",
        qualified_name=source.stem,
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(source, 1, 1, 1, len(content) + 1),
        signature=f"int {source.stem}()",
        source_text="x" * payload_bytes,
        source_hash=unit.content_hash,
        build_configuration_id=configuration.id,
        translation_unit_id=unit.id,
        build_variant=configuration.build_variant,
        metadata={"is_definition": True},
    )
    return IngestionBatch((configuration,), (unit,), (symbol,), (), ())


class _StreamingOnlyIngestor:
    analysis_backend = "stream-fixture"
    advanced_facts_complete = True

    def __init__(self, *, fail_after: int | None = None, payload_bytes: int = 0) -> None:
        self.fail_after = fail_after
        self.payload_bytes = payload_bytes
        self.yielded = 0

    def ingest_configurations(
        self, _project_root: Path, _configurations: Iterable[BuildConfiguration]
    ) -> IngestionBatch:
        raise AssertionError("ProjectIndexer retained the legacy project-wide batch path")

    def iter_configuration_batches(
        self, _project_root: Path, configurations: Iterable[BuildConfiguration]
    ) -> Iterator[IngestionBatch]:
        for configuration in configurations:
            if self.fail_after is not None and self.yielded == self.fail_after:
                raise RuntimeError("injected streamed ingestion failure")
            self.yielded += 1
            yield _batch(configuration, payload_bytes=self.payload_bytes)


def _semantic_dump(store: SQLiteStore) -> dict[str, tuple[tuple[object, ...], ...]]:
    connection = store._connection  # noqa: SLF001 - durable equivalence evidence
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%fts%' ORDER BY name"
        )
    ]
    snapshot: dict[str, tuple[tuple[object, ...], ...]] = {}
    for table in tables:
        columns = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})")
            if row[1] != "indexed_at"
        ]
        projection = ", ".join(columns)
        ordering = ", ".join(str(index) for index in range(1, len(columns) + 1))
        snapshot[table] = tuple(
            tuple(row)
            for row in connection.execute(f"SELECT {projection} FROM {table} ORDER BY {ordering}")
        )
    return snapshot


def test_project_indexer_consumes_tu_batches_without_project_wide_batch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = _database(root, 120)
    ingestor = _StreamingOnlyIngestor(payload_bytes=16 * 1024)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        result = ProjectIndexer(ingestor, store).index(root, database)

        assert result.indexed_translation_units == 120
        assert result.indexed_symbols == 120
        assert len(store.translation_unit_states(root)) == 120
    assert ingestor.yielded == 120


def test_staged_batch_is_released_before_waiting_for_the_next_batch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = _database(root, 2)
    second_requested = threading.Event()
    release_second = threading.Event()
    failures: list[BaseException] = []
    first_payload: list[weakref.ReferenceType[str]] = []

    class Payload(str):
        pass

    def first_batch(configuration: BuildConfiguration) -> IngestionBatch:
        batch = _batch(configuration)
        payload = Payload("x" * (1024 * 1024))
        first_payload.append(weakref.ref(payload))
        return replace(batch, symbols=(replace(batch.symbols[0], source_text=payload),))

    class BlockingIngestor(_StreamingOnlyIngestor):
        def iter_configuration_batches(
            self, _project_root: Path, configurations: Iterable[BuildConfiguration]
        ) -> Iterator[IngestionBatch]:
            for index, configuration in enumerate(configurations):
                if index == 1:
                    second_requested.set()
                    assert release_second.wait(timeout=5)
                yield first_batch(configuration) if index == 0 else _batch(configuration)

    def index_in_background() -> None:
        try:
            with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
                ProjectIndexer(BlockingIngestor(), store).index(root, database)
        except BaseException as error:  # pragma: no cover - surfaced below
            failures.append(error)

    worker = threading.Thread(target=index_in_background)
    worker.start()
    assert second_requested.wait(timeout=5)
    try:
        assert len(first_payload) == 1
        assert first_payload[0]() is None
    finally:
        release_second.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []


def test_stream_failure_rolls_back_all_staged_units_and_orphans(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = _database(root, 110)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        ProjectIndexer(_StreamingOnlyIngestor(), store).index(root, database)
        before = _semantic_dump(store)
        for index in range(110):
            changed = root / f"unit-{index:03d}.cpp"
            changed.write_text(
                changed.read_text(encoding="utf-8") + "// changed\n", encoding="utf-8"
            )

        with pytest.raises(RuntimeError, match="injected streamed ingestion failure"):
            ProjectIndexer(_StreamingOnlyIngestor(fail_after=100), store).index(root, database)

        assert _semantic_dump(store) == before
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001


def test_tracking_cleanup_failure_happens_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = _database(root, 2)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        ProjectIndexer(_StreamingOnlyIngestor(), store).index(root, database)
        before = _semantic_dump(store)
        changed = root / "unit-000.cpp"
        changed.write_text(changed.read_text(encoding="utf-8") + "// changed\n", encoding="utf-8")
        original_clear = store._clear_ingestion_tracking  # noqa: SLF001
        calls = 0

        def fail_final_cleanup() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected tracking cleanup failure")
            original_clear()

        monkeypatch.setattr(store, "_clear_ingestion_tracking", fail_final_cleanup)

        with pytest.raises(RuntimeError, match="tracking cleanup failure"):
            ProjectIndexer(_StreamingOnlyIngestor(), store).index(root, database)

        assert calls == 2
        assert _semantic_dump(store) == before
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001


def test_incomplete_stream_rolls_back_instead_of_publishing_missing_units(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = _database(root, 3)

    class IncompleteIngestor(_StreamingOnlyIngestor):
        def iter_configuration_batches(
            self, _project_root: Path, configurations: Iterable[BuildConfiguration]
        ) -> Iterator[IngestionBatch]:
            for configuration in tuple(configurations)[:2]:
                yield _batch(configuration)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        with pytest.raises(ValueError, match="before every changed translation unit"):
            ProjectIndexer(IncompleteIngestor(), store).index(root, database)

        assert not store.has_project(root)
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []  # noqa: SLF001


def test_separate_reader_sees_previous_generation_while_batches_are_staged(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = _database(root, 2)
    database_path = tmp_path / "index.db"
    staged = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    class BlockingIngestor(_StreamingOnlyIngestor):
        def iter_configuration_batches(
            self, _project_root: Path, configurations: Iterable[BuildConfiguration]
        ) -> Iterator[IngestionBatch]:
            for index, configuration in enumerate(configurations):
                yield _batch(configuration)
                if index == 0:
                    staged.set()
                    assert release.wait(timeout=5)

    with SQLiteStore(database_path, project_root=root) as writer:
        ProjectIndexer(_StreamingOnlyIngestor(), writer).index(root, database)
        before = _semantic_dump(writer)
        for index in range(2):
            source = root / f"unit-{index:03d}.cpp"
            source.write_text(source.read_text(encoding="utf-8") + "// changed\n", encoding="utf-8")

        def index_in_background() -> None:
            try:
                ProjectIndexer(BlockingIngestor(), writer).index(root, database)
            except BaseException as error:  # pragma: no cover - surfaced below
                failures.append(error)

        worker = threading.Thread(target=index_in_background)
        worker.start()
        assert staged.wait(timeout=5)
        try:
            with SQLiteStore(database_path, project_root=root) as reader:
                assert _semantic_dump(reader) == before
        finally:
            release.set()
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert failures == []
        assert _semantic_dump(writer) != before


def test_streamed_storage_matches_single_project_batch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = _database(root, 2)

    default_configurations = CompilationDatabase.load(database).configurations
    alpha = _batch(default_configurations[0])
    beta = _batch(default_configurations[1])
    combined = IngestionBatch(
        alpha.build_configurations + beta.build_configurations,
        alpha.translation_units + beta.translation_units,
        alpha.symbols + beta.symbols,
        (),
        (),
    )
    all_ids = frozenset(unit.id for unit in combined.translation_units)
    variant = BuildVariant("default", database)

    with SQLiteStore(tmp_path / "single.db", project_root=root) as single:
        single.apply_ingestion(
            root,
            combined,
            current_translation_unit_ids=all_ids,
            build_variant=variant,
        )
        expected = _semantic_dump(single)

    with SQLiteStore(tmp_path / "streamed.db", project_root=root) as streamed:
        streamed.apply_ingestion_batches(
            root,
            iter((alpha, beta)),
            current_translation_unit_ids=all_ids,
            build_variant=variant,
        )
        actual = _semantic_dump(streamed)

    assert actual == expected


def test_streamed_build_variants_remain_isolated_during_stale_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = _database(root, 2)
    alpha_configurations = CompilationDatabase.load(database, build_variant="alpha").configurations
    beta_configurations = CompilationDatabase.load(database, build_variant="beta").configurations
    alpha_batches = tuple(_batch(configuration) for configuration in alpha_configurations)
    beta_batches = tuple(_batch(configuration) for configuration in beta_configurations)

    with SQLiteStore(tmp_path / "index.db", project_root=root) as store:
        store.apply_ingestion_batches(
            root,
            alpha_batches,
            current_translation_unit_ids=frozenset(
                unit.id for batch in alpha_batches for unit in batch.translation_units
            ),
            changed_translation_unit_ids=frozenset(
                unit.id for batch in alpha_batches for unit in batch.translation_units
            ),
            build_variant=BuildVariant("alpha", database),
        )
        store.apply_ingestion_batches(
            root,
            beta_batches,
            current_translation_unit_ids=frozenset(
                unit.id for batch in beta_batches for unit in batch.translation_units
            ),
            changed_translation_unit_ids=frozenset(
                unit.id for batch in beta_batches for unit in batch.translation_units
            ),
            build_variant=BuildVariant("beta", database),
        )
        beta_before = tuple(
            store._connection.execute(  # noqa: SLF001 - isolation snapshot
                "SELECT id, snapshot_json FROM symbol_variants "
                "WHERE build_variant = 'beta' ORDER BY id"
            )
        )

        store.apply_ingestion_batches(
            root,
            (),
            current_translation_unit_ids=frozenset(),
            changed_translation_unit_ids=frozenset(),
            build_variant=BuildVariant("alpha", database),
        )

        variants = {
            row[0]
            for row in store._connection.execute(  # noqa: SLF001 - isolation evidence
                "SELECT DISTINCT build_variant FROM translation_units"
            )
        }
        beta_after = tuple(
            store._connection.execute(  # noqa: SLF001 - isolation snapshot
                "SELECT id, snapshot_json FROM symbol_variants "
                "WHERE build_variant = 'beta' ORDER BY id"
            )
        )

    assert variants == {"beta"}
    assert beta_after == beta_before
