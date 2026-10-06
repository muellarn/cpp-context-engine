from dataclasses import replace

import pytest
from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.models import SearchQuery
from cpp_context_engine.storage.sqlite import SQLiteStore


def test_new_variants_do_not_serialize_unused_full_snapshots(tmp_path, monkeypatch):
    calls = []
    original = SQLiteStore._symbol_snapshot

    def tracked(symbol, *, provenance=True):
        calls.append(provenance)
        return original(symbol, provenance=provenance)

    monkeypatch.setattr(SQLiteStore, "_symbol_snapshot", staticmethod(tracked))
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        batch = _batch(tmp_path, "one")
        _put(store, tmp_path, batch)
        assert calls.count(False) == len(batch.symbols)
        assert calls.count(True) == 0, "new IDs have no full snapshot to compare"
        actual = store.symbols()
        assert {s.id for s in actual} == {s.id for s in batch.symbols}
        assert store.search(SearchQuery("repeated_value"))
        assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("change", ["unchanged", "content", "provenance", "roundtrip"])
def test_existing_variants_keep_exact_invalidation_and_effective_ids(tmp_path, change):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        batch = _batch(tmp_path, "one")
        _put(store, tmp_path, batch)
        symbol = batch.symbols[0]
        store.put_embedding(symbol.id, "fixture", [1.0, 0.0])
        actual = next(s for s in store.symbols() if s.id == symbol.id)
        assert symbol.variant_id == "" and actual.variant_id
        incoming = (symbol,)
        if change == "content":
            incoming = (replace(symbol, source_text="int ChangedOnly = 7;"),)
        elif change == "provenance":
            # Keep the same effective ID while changing its full provenance.
            incoming = (
                replace(
                    symbol,
                    variant_id=actual.variant_id,
                    build_configuration_id="changed-configuration",
                ),
            )
        elif change == "roundtrip":
            incoming = (replace(symbol, source_text="int IntermediateOnly = 7;"), symbol)
        # This internal path is also used by deferred fresh ingestion. Defer FTS
        # here so repeated IDs exercise the per-input invalidation independently.
        store._defer_variant_fts = True
        with store._connection:
            store._put_symbol_variants(store._project_id(), incoming)
            store._rebuild_variant_fts()
        store._defer_variant_fts = False
        assert store.embedding_count("fixture") == int(change == "unchanged")
        current = next(s for s in store.symbols() if s.id == symbol.id)
        assert current == replace(incoming[-1], variant_id=actual.variant_id)


@pytest.mark.parametrize("failure", ["iterator", "serialization"])
def test_lazy_serialization_errors_restore_fts_facts_and_embeddings(tmp_path, monkeypatch, failure):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        batch = _batch(tmp_path, "one")
        _put(store, tmp_path, batch)
        store.put_embedding(batch.symbols[0].id, "fixture", [1.0, 0.0])
        before_symbols = store.symbols()
        before_hits = store.search(SearchQuery("repeated_value"))
        tables = ("symbol_variants", "symbol_snapshot_contents", "variant_embeddings")
        before_rows = {
            table: tuple(tuple(row) for row in store._connection.execute(f"SELECT * FROM {table}"))
            for table in tables
        }
        original = SQLiteStore._symbol_snapshot

        def fail_snapshot(symbol, *, provenance=True):
            if not provenance:
                raise ValueError("snapshot serialization failed")
            return original(symbol, provenance=provenance)

        def incoming():
            yield replace(batch.symbols[0], source_text="int ChangedOnly = 7;")
            if failure == "iterator":
                raise ValueError("input iterator failed")

        if failure == "serialization":
            monkeypatch.setattr(SQLiteStore, "_symbol_snapshot", staticmethod(fail_snapshot))
        with pytest.raises(ValueError, match="failed"), store._connection:
            store._put_symbol_variants(store._project_id(), incoming())
        assert store.symbols() == before_symbols
        assert store.search(SearchQuery("repeated_value")) == before_hits
        assert {
            table: tuple(tuple(row) for row in store._connection.execute(f"SELECT * FROM {table}"))
            for table in tables
        } == before_rows


def test_legacy_unsplit_snapshot_comparison_is_not_replaced_by_content_identity(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        batch = _batch(tmp_path, "one")
        _put(store, tmp_path, batch)
        symbol = batch.symbols[0]
        actual = next(s for s in store.symbols() if s.id == symbol.id)
        legacy = store._symbol_snapshot(actual).replace('"documentation":', '"documentation" :')
        project = store._project_id()
        with store._connection:
            (snapshot_id,) = store._intern_symbol_snapshots(project, ((0, legacy),))
            store._connection.execute(
                "UPDATE symbol_variants SET snapshot_id=? WHERE project_id=? AND id=?",
                (snapshot_id, project, actual.variant_id),
            )
        assert next(s for s in store.symbols() if s.id == symbol.id) == actual
        store.put_embedding(symbol.id, "fixture", [1.0, 0.0])
        with store._connection:
            store._put_symbol_variant(project, symbol)
        # The original contract compares complete JSON, including legacy formatting.
        assert store.embedding_count("fixture") == 0


def test_duplicate_new_variant_ids_keep_the_last_value(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        batch = _batch(tmp_path, "one")
        _put(store, tmp_path, batch)
        symbol = replace(batch.symbols[0], variant_id="new-explicit-id", build_variant="other")
        final = replace(symbol, source_text="int FinalOnly = 42;")
        store._defer_variant_fts = True
        with store._connection:
            store._put_symbol_variants(store._project_id(), (symbol, final))
            row = store._connection.execute(
                "SELECT * FROM symbol_variant_snapshots WHERE id=?", (symbol.variant_id,)
            ).fetchone()
            assert store._snapshot_symbol(row["snapshot_json"], row) == final
            store._rebuild_variant_fts()
        store._defer_variant_fts = False
