import pytest
from test_symbol_snapshot_compression import _batch, _put

import cpp_context_engine.storage.sqlite as storage
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
