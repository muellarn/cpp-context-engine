from dataclasses import replace

import pytest
from test_symbol_snapshot_compression import _batch, _put

import cpp_context_engine.storage.sqlite as storage
from cpp_context_engine.models import BuildScope
from cpp_context_engine.search.embeddings import DeterministicLocalEmbeddingProvider
from cpp_context_engine.search.vector import SQLiteVectorSearch
from cpp_context_engine.storage.sqlite import SQLiteStore


@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_missing_embeddings_decode_each_shared_snapshot_once(tmp_path, monkeypatch, batch_size):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        for index in range(8):
            _put(store, tmp_path, _batch(tmp_path, f"unit-{index}"))
        contents = store._connection.execute(
            "SELECT count(DISTINCT snapshot_id) FROM symbol_variants"
        ).fetchone()[0]
        assert contents == 2
        calls = []
        statements = []
        store._connection.set_trace_callback(statements.append)
        original = storage._embedding_text_from_snapshot

        def counted(snapshot, limit):
            calls.append(snapshot)
            return original(snapshot, limit)

        monkeypatch.setattr(storage, "_embedding_text_from_snapshot", counted)
        search = SQLiteVectorSearch(
            store,
            DeterministicLocalEmbeddingProvider(32),
            project_root=tmp_path,
            batch_size=batch_size,
        )
        assert search.index_missing() == 16
        assert search.index_missing() == 0
        assert (
            store._connection.execute("SELECT count(*) FROM variant_embeddings").fetchone()[0] == 16
        )
        assert len(calls) == contents
        payload_reads = [
            sql
            for sql in statements
            if sql.lstrip().upper().startswith("SELECT") and "snapshot_json" in sql
        ]
        assert len(payload_reads) == contents, (
            "read the stored payload once per content, not per page"
        )


@pytest.mark.parametrize("limit", [17, 32_000])
def test_grouped_embedding_projection_matches_symbol_oracle(tmp_path, limit):
    root = tmp_path / "project"
    root.mkdir()
    provider = DeterministicLocalEmbeddingProvider(32)
    scope = BuildScope(("alpha", "beta"))
    results = []
    for label in ("symbols", "grouped"):
        with SQLiteStore(tmp_path / f"{label}.db", project_root=root) as store:
            for index, build in enumerate(("alpha", "beta", "alpha", "excluded")):
                batch = _batch(root, f"unit-{index}", build)
                if index == 2:
                    batch = replace(
                        batch,
                        symbols=(
                            replace(batch.symbols[0], source_text="unique"),
                            *batch.symbols[1:],
                        ),
                    )
                _put(store, root, batch)
            search = SQLiteVectorSearch(
                store,
                provider,
                project_root=root,
                build_scope=scope,
                batch_size=3,
                max_text_chars=limit,
            )
            identifiers = store.missing_embedding_variant_ids(provider.model_id, build_scope=scope)
            assert len(identifiers) == 6
            assert search.index(identifiers[:1]) == 1
            # A different configuration must neither count as present nor be lost.
            store.put_embedding(
                identifiers[1],
                provider.model_id,
                (1.0,) * 32,
                configuration_id="other-config",
                build_scope=scope,
            )
            missing = store.missing_embedding_variant_ids(provider.model_id, build_scope=scope)
            assert len(missing) == 5
            assert (search.index(missing) if label == "symbols" else search.index_missing()) == 5
            assert search.index_missing() == 0
            assert (
                len(
                    store.missing_embedding_variant_ids(
                        provider.model_id, build_scope=BuildScope.single("excluded")
                    )
                )
                == 2
            )
            results.append(
                tuple(
                    tuple(
                        tuple(row)
                        for row in store._connection.execute(
                            f"SELECT * FROM {table} ORDER BY 1,2,3,4"
                        )
                    )
                    for table in ("variant_embeddings", "embedding_vectors")
                )
            )
    assert results[0] == results[1]


def test_grouped_embedding_pages_use_content_range_seeks(tmp_path):
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        for index in range(8):
            _put(store, tmp_path, _batch(tmp_path, f"unit-{index}"))
        statements = []
        store._connection.set_trace_callback(statements.append)
        assert (
            SQLiteVectorSearch(
                store, DeterministicLocalEmbeddingProvider(32), batch_size=3
            ).index_missing()
            == 16
        )
        store._connection.set_trace_callback(None)
        paging = [sql for sql in statements if "SELECT variants.rowid AS record_key" in sql]
        assert any("variants.rowid>" in sql for sql in paging)
        assert any("variants.snapshot_id>" in sql for sql in paging)
        for sql in paging:
            plan = [row[3] for row in store._connection.execute("EXPLAIN QUERY PLAN " + sql)]
            assert any(
                "SEARCH variants USING INDEX symbol_variants_snapshot" in part for part in plan
            ), plan
            assert not any("TEMP B-TREE" in part for part in plan), plan
        seeks = [
            next(sql for sql in paging if suffix in sql)
            for suffix in ("variants.rowid>", "variants.snapshot_id>")
        ]

        def measured(sql):
            steps = 0

            def tick():
                nonlocal steps
                steps += 1
                return 0

            store._connection.set_progress_handler(tick, 1)
            try:
                rows = tuple(tuple(row) for row in store._connection.execute(sql))
            finally:
                store._connection.set_progress_handler(None, 0)
            return rows, steps

        before = [measured(sql) for sql in seeks]
        unit = dict(store._connection.execute("SELECT * FROM translation_units LIMIT 1").fetchone())
        variant = dict(
            store._connection.execute(
                "SELECT * FROM symbol_variants ORDER BY snapshot_id,rowid LIMIT 1"
            ).fetchone()
        )
        with store._connection:
            for index in range(1024):
                name = f"excluded-prefix-{index}"
                new_unit = dict(unit, id=name)
                store._connection.execute(
                    f"INSERT INTO translation_units({','.join(new_unit)}) "
                    f"VALUES({','.join('?' for _ in new_unit)})",
                    tuple(new_unit.values()),
                )
                new_variant = dict(
                    variant, id=name, translation_unit_id=name, build_variant="excluded"
                )
                store._connection.execute(
                    f"INSERT INTO symbol_variants(rowid,{','.join(new_variant)}) "
                    f"VALUES(?,{','.join('?' for _ in new_variant)})",
                    (-index - 1, *new_variant.values()),
                )
        assert not store._connection.execute("PRAGMA foreign_key_check").fetchall()
        after = [measured(sql) for sql in seeks]
        for (old_rows, old_steps), (new_rows, new_steps) in zip(before, after, strict=True):
            assert new_rows == old_rows
            assert new_steps <= old_steps + 80, (old_steps, new_steps)


def test_grouped_embedding_failure_rolls_back_preceding_content_group(tmp_path):
    class FailingProvider:
        model_id = "failing"
        configuration_id = "failing-config"
        calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("second content group")
            return DeterministicLocalEmbeddingProvider(32).embed(texts)

    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        for index in range(3):
            _put(store, tmp_path, _batch(tmp_path, f"unit-{index}"))
        with pytest.raises(ValueError, match="second content group"):
            SQLiteVectorSearch(store, FailingProvider(), batch_size=2).index_missing()
        assert not store._connection.in_transaction
        assert not store._connection.execute("SELECT * FROM variant_embeddings").fetchall()
        assert not store._connection.execute("SELECT * FROM embedding_vectors").fetchall()
        assert (
            SQLiteVectorSearch(
                store, DeterministicLocalEmbeddingProvider(32), batch_size=2
            ).index_missing()
            == 6
        )
