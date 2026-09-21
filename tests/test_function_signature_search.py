from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import CodeSymbol, SearchHit, SearchQuery, SourceSpan, SymbolKind
from cpp_context_engine.retrieval.hybrid import HybridRetriever, _Candidate
from cpp_context_engine.storage import SQLiteStore
from cpp_context_engine.storage.source import FilesystemSourceReader
from cpp_context_engine.storage.sqlite import _embedding_text


def test_declaration_signature_changes_search_ranking_and_embedding_not_source(
    tmp_path: Path,
) -> None:
    source = "int ordinary(int value = 3) {\n  return value + body_only_token;\n}"
    path = tmp_path / "ordinary.cpp"
    path.write_text(source, encoding="utf-8")
    old = CodeSymbol(
        id="usr:ordinary",
        qualified_name="ordinary",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(path, 1, 3),
        signature=source,
        source_text=source,
    )
    new = replace(old, signature="int ordinary(int value = 3)")
    query = SearchQuery("body_only_token")
    reader = FilesystemSourceReader(tmp_path)
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        store.apply_ingestion(tmp_path, IngestionBatch((), (), (), (), ()))
        store.put_symbols((old,))
        assert [hit.symbol.id for hit in store.search_symbols(query)] == [old.id]
        store.put_symbols((new,))
        assert store.search_symbols(query) == ()
        assert [hit.symbol.id for hit in store.search(query)] == [new.id]
        assert reader.read_symbol(new) == source

        # Isolate the existing signature-overlap contribution, not an unstable
        # end-to-end ordering influenced by FTS/vector candidate generation.
        retriever = HybridRetriever(
            lexical_search=store,
            symbol_search=store,
            vector_search=store,
            symbol_store=store,
            source_reader=reader,
        )
        scores = []
        for symbol in (old, new):
            candidate = _Candidate(SearchHit(symbol, 1.0, "fixture"), [])
            ranked = retriever._rerank(query.text, {symbol.id: candidate}, update_scores=True)
            scores.append(ranked[0].hit.score)
        assert scores == pytest.approx([1.35, 1.0])

    assert _embedding_text(old) != _embedding_text(new)
    assert _embedding_text(old).count("body_only_token") == 2
    assert _embedding_text(new).count("body_only_token") == 1
    assert _embedding_text(new).endswith(source)
