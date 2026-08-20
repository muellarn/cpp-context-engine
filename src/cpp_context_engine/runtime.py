"""Composition root wiring concrete compiler, storage, retrieval, and LLM adapters."""

from __future__ import annotations

from dataclasses import dataclass

from cpp_context_engine.api import ContextRetrievalService, IterativeAnswerService
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion import ClangIngestor, IndexingResult, ProjectIndexer
from cpp_context_engine.llm import LLMProvider, OpenAICompatibleProvider
from cpp_context_engine.retrieval import HybridRetriever, RetrievalConfig
from cpp_context_engine.search import (
    DeterministicLocalEmbeddingProvider,
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SQLiteLexicalSearch,
    SQLiteSymbolSearch,
    SQLiteVectorSearch,
)
from cpp_context_engine.storage import FilesystemSourceReader, SQLiteStore


@dataclass(frozen=True, slots=True)
class IndexOperationResult:
    indexing: IndexingResult
    embedded_symbols: int
    embedding_model: str


@dataclass(slots=True)
class Runtime:
    """One open project runtime; callers must close it when no longer serving."""

    config: AppConfig
    store: SQLiteStore
    vector_search: SQLiteVectorSearch
    retrieval_service: ContextRetrievalService
    answer_service: IterativeAnswerService | None

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def embedding_provider(config: AppConfig) -> EmbeddingProvider:
    if config.embedding_provider == "local":
        return DeterministicLocalEmbeddingProvider(config.embedding_dimensions)
    if not config.embedding_base_url or not config.embedding_model:
        raise ValueError(
            "openai embeddings require CPP_CONTEXT_EMBEDDING_BASE_URL and "
            "CPP_CONTEXT_EMBEDDING_MODEL (or matching CLI options)"
        )
    return OpenAICompatibleEmbeddingProvider(
        base_url=config.embedding_base_url,
        model=config.embedding_model,
        api_key=config.embedding_api_key,
        timeout_seconds=config.provider_timeout_seconds,
    )


def llm_provider(config: AppConfig) -> LLMProvider:
    if not config.llm_base_url or not config.llm_model:
        raise ValueError(
            "LLM answering requires CPP_CONTEXT_LLM_BASE_URL and CPP_CONTEXT_LLM_MODEL "
            "(or matching CLI options)"
        )
    return OpenAICompatibleProvider(
        base_url=config.llm_base_url,
        model=config.llm_model,
        api_key=config.llm_api_key,
        timeout_seconds=config.provider_timeout_seconds,
    )


def index_project(config: AppConfig) -> IndexOperationResult:
    """Incrementally index compiler facts, then only missing/current vectors."""

    assert config.database_path is not None
    assert config.compilation_database is not None
    if not config.project_root.is_dir():
        raise ValueError(f"project directory does not exist: {config.project_root}")
    provider = embedding_provider(config)
    with SQLiteStore(config.database_path, project_root=config.project_root) as store:
        ingestor = ClangIngestor(library_file=config.libclang_library_file)
        indexing = ProjectIndexer(ingestor, store).index(
            config.project_root, config.compilation_database
        )
        vector_search = SQLiteVectorSearch(store, provider, project_root=config.project_root)
        missing = store.missing_embedding_symbol_ids(provider.model_id, config.project_root)
        vector_search.index(missing)
        return IndexOperationResult(indexing, len(missing), provider.model_id)


def build_runtime(
    config: AppConfig,
    *,
    llm: LLMProvider | None = None,
    require_llm: bool = False,
) -> Runtime:
    """Wire project-scoped adapters to retrieval and optional answer services."""

    assert config.database_path is not None
    store = SQLiteStore(config.database_path, project_root=config.project_root)
    try:
        if not store.has_project(config.project_root):
            raise ValueError(
                f"project is not indexed in {config.database_path}; run 'cpp-context index' first"
            )
        provider = embedding_provider(config)
        vector = SQLiteVectorSearch(store, provider, project_root=config.project_root)
        retriever = HybridRetriever(
            lexical_search=SQLiteLexicalSearch(store, config.project_root),
            symbol_search=SQLiteSymbolSearch(store, config.project_root),
            vector_search=vector,
            symbol_store=store,
            source_reader=FilesystemSourceReader(config.project_root),
            graph=store,
            config=RetrievalConfig(
                search_limit=config.retrieval_limit,
                candidate_limit=max(config.retrieval_limit, config.retrieval_limit * 4),
                seed_limit=min(6, config.retrieval_limit),
            ),
        )
        retrieval = ContextRetrievalService(
            retriever,
            default_max_context_tokens=config.max_context_tokens,
            max_context_tokens=max(64_000, config.max_context_tokens),
        )
        selected_llm = llm
        if selected_llm is None and (require_llm or (config.llm_base_url and config.llm_model)):
            selected_llm = llm_provider(config)
        answer = IterativeAnswerService(retrieval, selected_llm) if selected_llm else None
        return Runtime(config, store, vector, retrieval, answer)
    except Exception:
        store.close()
        raise
