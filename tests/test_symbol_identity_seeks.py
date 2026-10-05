from pathlib import Path

import pytest

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


@pytest.mark.parametrize("bulk", [False, True])
def test_symbol_identity_lookups_seek_both_keys(tmp_path: Path, bulk: bool):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        store._ensure_project(str(tmp_path))
        store._connection.commit()
        store.put_symbols(
            (
                CodeSymbol(
                    "needle", "needle", SymbolKind.FUNCTION, SourceSpan(tmp_path / "a.cpp", 1, 1)
                ),
            )
        )
        statements = []
        store._connection.set_trace_callback(statements.append)
        if bulk:
            store.get_symbols(("needle", "missing", "needle"))
        else:
            store.get_symbol("needle")
        store._connection.set_trace_callback(None)
        query = next(sql for sql in statements if "FROM symbol_variants" in sql)
        plan = [row[3] for row in store._connection.execute("EXPLAIN QUERY PLAN " + query)]
        assert any("symbol_id=?" in step for step in plan), plan
        assert any("AND id=?" in step for step in plan), plan


def _ingest(store, root, unit_id, specs, build="default"):
    path = root / "a.cpp"
    config = BuildConfiguration(
        unit_id, path, root, ("c++", str(path)), unit_id, build_variant=build
    )
    unit = TranslationUnit(unit_id, config.id, path, "hash", build_variant=build)
    symbols = tuple(
        CodeSymbol(
            canonical,
            name,
            SymbolKind.FUNCTION,
            SourceSpan(path, 1, 1),
            build_configuration_id=config.id,
            translation_unit_id=unit.id,
            build_variant=build,
            variant_id=variant,
            metadata={"is_definition": definition},
        )
        for canonical, variant, name, definition in specs
    )
    store.apply_ingestion(
        root,
        IngestionBatch((config,), (unit,), symbols, (), ()),
        build_variant=BuildVariant(build, root / "commands.json"),
    )


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("scope", [("default",), ("alternate",), ("default", "alternate")])
def test_symbol_seeks_match_legacy_identity_preferences(tmp_path, bulk, scope):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _ingest(
            store,
            tmp_path,
            "one",
            (
                ("shared", "v-decl", "declaration", False),
                ("same", "same", "self-match", True),
                ("collision", "v-canonical", "canonical-collision", True),
            ),
        )
        _ingest(
            store,
            tmp_path,
            "two",
            (
                ("shared", "v-def", "definition", True),
                ("different", "collision", "exact-collision", False),
            ),
        )
        _ingest(store, tmp_path, "three", (("shared", "v-alt", "alternate", True),), "alternate")
        _ingest(store, tmp_path / "other", "one", (("shared", "v-def", "foreign", True),))
        legacy = CodeSymbol(
            "legacy", "legacy", SymbolKind.FUNCTION, SourceSpan(tmp_path / "a.cpp", 1, 1)
        )
        store.put_symbols((legacy,))
        requested = (
            "shared",
            "same",
            "v-decl",
            "v-def",
            "v-alt",
            "collision",
            "missing",
            "legacy",
            "shared",
        )
        expected = []
        placeholders = ",".join("?" for _ in scope)
        for identity in requested:
            rows = list(
                store._connection.execute(
                    "SELECT * FROM symbol_variants WHERE project_id = ? "
                    f"AND build_variant IN ({placeholders}) AND (symbol_id = ? OR id = ?) "
                    "ORDER BY is_definition DESC, build_variant, translation_unit_id"
                    + (", id" if bulk else ""),
                    (store._project_id(tmp_path), *scope, identity, identity),
                )
            )
            if rows:
                row = next((r for r in rows if r["id"] == identity), rows[0]) if bulk else rows[0]
                expected.append(store._variant_row_to_symbol(row))
            else:
                row = store._connection.execute(
                    "SELECT * FROM symbols WHERE project_id=? AND id=?",
                    (store._project_id(tmp_path), identity),
                ).fetchone()
                expected.append(store._row_to_symbol(row) if row and "default" in scope else None)
        actual = (
            store.get_symbols(requested, build_scope=scope)
            if bulk
            else tuple(store.get_symbol(identity, build_scope=scope) for identity in requested)
        )
        assert actual == tuple(expected)
        assert not any(symbol and symbol.qualified_name == "foreign" for symbol in actual)
        if "default" in scope:
            assert actual[5].qualified_name == (
                "exact-collision" if bulk else "canonical-collision"
            )


def test_bulk_identity_lookup_decodes_only_selected_variants(tmp_path, monkeypatch):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        for number in range(3):
            _ingest(store, tmp_path, str(number), (("shared", f"v{number}", "name", True),))
        decoded = []
        original = store._variant_row_to_symbol

        def decode(row):
            decoded.append(row["id"])
            return original(row)

        monkeypatch.setattr(store, "_variant_row_to_symbol", decode)
        assert store.get_symbols(("shared", "shared"))[0].variant_id == "v0"
        assert decoded == ["v0"]


def test_single_identity_collision_preserves_existing_tie_preference(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _ingest(
            store,
            tmp_path,
            "same-tu",
            (
                ("needle", "zzz", "first-canonical", True),
                ("zzz", "needle", "second-canonical", True),
            ),
        )
        assert store.get_symbol("needle").qualified_name == "first-canonical"
        assert store.get_symbol("zzz").qualified_name == "first-canonical"
        assert store.get_symbols(("needle",))[0].qualified_name == "second-canonical"


def test_bulk_identity_seeks_keep_parameter_batches_bounded(tmp_path):
    import sqlite3

    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _ingest(
            store, tmp_path, "many", tuple((f"s{i}", f"v{i}", f"name{i}", True) for i in range(405))
        )
        store._connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        requested = tuple(f"v{i}" for i in reversed(range(405))) + ("v0", "missing", "s404")
        actual = store.get_symbols(requested)
        assert [symbol.variant_id if symbol else None for symbol in actual] == [
            *requested[:-3],
            "v0",
            None,
            "v404",
        ]
