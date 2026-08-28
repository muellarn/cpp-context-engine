from __future__ import annotations

from pathlib import Path

import pytest

from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    CodeSymbol,
    SearchQuery,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.storage.sqlite import SQLiteStore


def _batch(root: Path) -> IngestionBatch:
    source = root / "source.cpp"
    source.write_text("int alpha() { return 7; }\n", encoding="utf-8")
    configuration = BuildConfiguration("build", source, root, ("c++", "source.cpp"), "command")
    unit = TranslationUnit("unit", configuration.id, source, "content", ((source, "content"),))
    symbol = CodeSymbol(
        "alpha",
        "alpha",
        SymbolKind.FUNCTION,
        SourceSpan(source, 1, 1, 1, 26),
        "int alpha()",
        documentation="important answer",
        source_text="int alpha() { return 7; }",
        build_configuration_id=configuration.id,
        translation_unit_id=unit.id,
        metadata={"is_definition": True},
    )
    return IngestionBatch((configuration,), (unit,), (symbol,), (), ())


class _RecordingStore(SQLiteStore):
    def __init__(self, *args: object, fail_at: str | None = None, **kwargs: object) -> None:
        self.steps: list[str] = []
        self.fail_at = fail_at
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def _execute_deferred_schema_step(self, step: str, sql: str) -> None:
        self.steps.append(step)
        if step == self.fail_at:
            raise RuntimeError(f"injected failure at {step}")
        super()._execute_deferred_schema_step(step, sql)


class _OnlineIndexStore(SQLiteStore):
    def _should_defer_fresh_generation(self) -> bool:
        return False


def _logical_rows(store: SQLiteStore) -> dict[str, tuple[tuple[object, ...], ...]]:
    tables = (
        "projects",
        "build_configurations",
        "build_variants",
        "translation_units",
        "dependencies",
        "translation_unit_symbols",
        "symbol_variants",
        "symbols",
    )
    result: dict[str, tuple[tuple[object, ...], ...]] = {}
    for table in tables:
        columns = [row[1] for row in store._connection.execute(f"PRAGMA table_info({table})")]  # noqa: SLF001
        selected = [column for column in columns if column != "indexed_at"]
        result[table] = tuple(
            tuple(row)
            for row in store._connection.execute(  # noqa: SLF001
                f"SELECT {', '.join(selected)} FROM {table} ORDER BY {', '.join(selected)}"
            )
        )
    return result


def test_every_sqlite_index_has_an_explicit_fresh_generation_classification(
    tmp_path: Path,
) -> None:
    with SQLiteStore(tmp_path / "index.db") as store:
        classifications = store._classify_schema_indexes()  # noqa: SLF001
        indexes = {
            (table, row[1])
            for table in (
                item[0]
                for item in store._connection.execute(  # noqa: SLF001
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            )
            for row in store._connection.execute(f"PRAGMA index_list({table})")  # noqa: SLF001
        }

    assert set(classifications) == indexes
    assert all(item.reason for item in classifications.values())
    assert all(
        not item.deferred
        for item in classifications.values()
        if item.unique or item.origin in {"pk", "u"}
    )


def test_only_the_first_project_generation_defers_indexes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with _RecordingStore(tmp_path / "index.db") as store:
        store.apply_ingestion(root, _batch(root))
        first_steps = tuple(store.steps)
        assert any(step.startswith("drop-index:") for step in first_steps)
        assert any(step.startswith("create-index:") for step in first_steps)
        assert "fts:variant-rebuild" in first_steps
        assert "fts:variant-integrity" in first_steps
        assert store.search_symbols(SearchQuery("alpha"))[0].symbol.id == "alpha"
        assert store.search(SearchQuery("important answer"))[0].symbol.id == "alpha"

        store.steps.clear()
        store.apply_ingestion(root, _batch(root))
        assert store.steps == []


def test_deferred_generation_matches_online_indexes_and_public_queries(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    with (
        SQLiteStore(tmp_path / "deferred.db") as deferred,
        _OnlineIndexStore(tmp_path / "online.db") as online,
    ):
        deferred.apply_ingestion(root, _batch(root))
        online.apply_ingestion(root, _batch(root))

        assert _logical_rows(deferred) == _logical_rows(online)
        assert deferred._classify_schema_indexes() == online._classify_schema_indexes()  # noqa: SLF001
        for query in (SearchQuery("alpha"), SearchQuery("important answer")):
            assert deferred.search(query) == online.search(query)
            assert deferred.search_symbols(query) == online.search_symbols(query)
        deferred_plan = tuple(
            tuple(row)
            for row in deferred._connection.execute(  # noqa: SLF001
                "EXPLAIN QUERY PLAN SELECT * FROM symbol_variants "
                "WHERE project_id = 1 AND symbol_id = 'alpha' "
                "ORDER BY is_definition DESC, build_variant, translation_unit_id"
            )
        )
        online_plan = tuple(
            tuple(row)
            for row in online._connection.execute(  # noqa: SLF001
                "EXPLAIN QUERY PLAN SELECT * FROM symbol_variants "
                "WHERE project_id = 1 AND symbol_id = 'alpha' "
                "ORDER BY is_definition DESC, build_variant, translation_unit_id"
            )
        )
        assert deferred_plan == online_plan


def test_readers_never_observe_the_indexless_staging_generation(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "index.db"
    with SQLiteStore(database) as writer, SQLiteStore(database) as reader:
        committed_indexes = set(reader._classify_schema_indexes())  # noqa: SLF001

        def batches() -> object:
            writer_indexes = set(writer._classify_schema_indexes())  # noqa: SLF001
            assert writer_indexes < committed_indexes
            assert set(reader._classify_schema_indexes()) == committed_indexes  # noqa: SLF001
            assert reader._connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0  # noqa: SLF001
            yield _batch(root)

        writer.apply_ingestion_batches(root, batches())  # type: ignore[arg-type]
        assert set(reader._classify_schema_indexes()) == committed_indexes  # noqa: SLF001
        assert reader._connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1  # noqa: SLF001


def test_every_deferred_publication_step_rolls_back_the_whole_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "index.db"
    with _RecordingStore(database) as store:
        deferred = tuple(
            item.name
            for item in store._classify_schema_indexes().values()  # noqa: SLF001
            if item.deferred
        )
        committed_indexes = set(store._classify_schema_indexes())  # noqa: SLF001
        failure_steps = (
            tuple(f"drop-index:{name}" for name in sorted(deferred))
            + tuple(f"create-index:{name}" for name in sorted(deferred))
            + ("fts:variant-rebuild", "fts:variant-integrity")
        )
        for step in failure_steps:
            store.fail_at = step
            with pytest.raises(RuntimeError, match="injected failure"):
                store.apply_ingestion(root, _batch(root))
            assert store._connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0  # noqa: SLF001
            assert set(store._classify_schema_indexes()) == committed_indexes  # noqa: SLF001

        store.fail_at = None
        store.apply_ingestion(root, _batch(root))
        assert store.get_symbol("alpha") is not None
