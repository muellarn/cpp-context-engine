# C++ Context Engine

C++ Context Engine is a foundation for compiler-aware retrieval over large C++
codebases. It is intended to combine exact symbols and code relationships with
lexical and vector search so that an LLM can load a small, connected set of
relevant source locations instead of the entire repository.

The repository provides Clang-based C++ ingestion, transactional SQLite storage,
FTS5 lexical search, and a provider-neutral cosine-vector index. Typed adapter
boundaries keep compiler ingestion, persistence, and search replaceable. Its
retrieval layer fuses lexical, compiler-symbol, and vector candidates, follows a
bounded set of code-graph relationships, and assembles cited source context for
an LLM without loading the whole repository.

## Requirements

- Python 3.11 or newer
- A C++ compilation database (`compile_commands.json`)
- libclang 18 and the optional `clang` Python extra for compiler-aware ingestion

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,clang]'
cpp-context --help
cpp-context doctor
```

Without installing the package, run the CLI from the checkout with:

```bash
PYTHONPATH=src python -m cpp_context_engine --help
```

Run the checks:

```bash
ruff format --check .
ruff check .
pytest
```

Pytest capture can be disabled with `pytest -s` on environments whose temporary
directory is managed by an external cleanup process.

## Architecture

The package is split along the intended processing pipeline:

- `ingestion`: validated compilation databases, libclang analysis, and incremental indexing
- `storage`: SQLite symbols, occurrences, graph edges, dependency state, FTS5, and vectors
- `search`: lexical and provider-neutral vector candidate generation
- `graph`: code relationship storage and traversal
- `retrieval`: reciprocal-rank fusion, seed reranking, bounded graph expansion,
  hub downranking, and connected context packing
- `llm`: a generic OpenAI-compatible chat-completions provider and a
  deterministic network-free test provider
- `api`: transport-neutral retrieval/answer services and an optional FastAPI
  transport

Shared immutable domain types live in `models.py`. Application paths and
environment-driven settings live in `config.py`. Python `Protocol` interfaces
keep implementations replaceable and tests easy to isolate.

See [Compiler-aware indexing](docs/indexing.md) for the ingestion/storage API,
incremental behavior, libclang configuration, and current guarantees.

## Retrieval behavior

`HybridRetriever` accepts the three search adapters plus the symbol store, source
reader, and code graph. It then:

1. deduplicates and fuses ranked candidates with reciprocal rank fusion;
2. selects top seeds and reranks them using symbol-name and signature overlap;
3. expands configured graph edge types with hard depth, node, edge, and per-node
   budgets;
4. downranks high-degree hubs and preserves the path from each seed;
5. reads and packs connected source excerpts under both token-derived and
   absolute character budgets.

Each packed item contains a stable symbol ID, file and line range, selection
reason, and graph path. Adapter failures are returned as sanitized diagnostics;
one failing candidate source does not hide successful sources.

`FilesystemSourceReader` confines reads to an explicit project root and rejects
path traversal and oversized files.

## LLM and HTTP API

Install the HTTP transport without development tools using:

```bash
python -m pip install -e '.[api]'
```

`OpenAICompatibleProvider` calls a configured `/chat/completions` endpoint with
an explicit timeout and response-size limit. It makes exactly one request per
completion, performs no hidden retries, and never includes its API key in its
representation or sanitized provider errors. The API key should be supplied by
the application from a secret store or environment variable; it must not be
committed.

`ContextRetrievalService` enforces the public context limit. An
`IterativeAnswerService` can ask the model for either a final JSON answer or one
more precise search, with a hard step limit. Reported source IDs are validated
against context that was actually read before they become structured citations.

Create the optional FastAPI app around already-configured services:

```python
from cpp_context_engine.api.http import create_app

app = create_app(
    retrieval_service=retrieval_service,
    answer_service=answer_service,
)
```

The routes are:

- `GET /health`
- `POST /v1/context` for source context and provenance
- `POST /v1/answer` for a bounded answer with validated citations

The package deliberately does not create a global provider or read an API key at
import time. Applications own adapter construction and secret loading.
