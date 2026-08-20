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

## Quickstart

Requirements are Python 3.11+, Clang/libclang 18, and a C++ compilation database.
On Ubuntu, `clang-18` and `libclang-18-dev` provide the native compiler pieces.
Install the engine with compiler and HTTP support:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
```

Generate `compile_commands.json` with your real build configuration. For CMake:

```bash
cmake -S /path/to/project -B /path/to/project/build \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /path/to/project/build
```

Meson users can run `meson setup build` (Meson writes a compilation database in
the build directory). For other build systems, tools such as Bear can capture
compiler invocations. Do not hand-write a database for a non-trivial project: its
include paths, defines, target flags, and generated headers are part of the index.

If libclang is not discovered automatically, select the native library explicitly:

```bash
export LIBCLANG_LIBRARY_FILE=/usr/lib/llvm-18/lib/libclang.so
```

Then diagnose, index, and search:

```bash
cpp-context doctor \
  --project /path/to/project \
  --compile-commands /path/to/project/build/compile_commands.json

cpp-context index /path/to/project \
  --compile-commands /path/to/project/build/compile_commands.json \
  --db /path/to/project/.cpp-context/index.db

cpp-context search "where is a packet validated before decoding?" \
  --project /path/to/project \
  --db /path/to/project/.cpp-context/index.db
```

`index` is incremental: unchanged translation units and already-current vectors
are skipped. `search --json` returns stable symbol IDs, source files and line
ranges, fused scores, selection reasons, and every code-graph hop used to connect
context. It is fully local by default.

### Offline and hosted embeddings

The default `local` embedding provider is deterministic, network-free feature
hashing. It improves identifier/token matching and makes offline operation and
tests reproducible, but it has **substantially weaker semantic understanding**
than a trained code embedding model. FTS5 symbol/source search and compiler graph
expansion remain active in both modes.

To use an OpenAI-compatible embedding endpoint, configure it before indexing and
use the same configuration when searching:

```bash
export CPP_CONTEXT_EMBEDDING_PROVIDER=openai
export CPP_CONTEXT_EMBEDDING_BASE_URL=https://provider.example/v1
export CPP_CONTEXT_EMBEDDING_MODEL=code-embedding-model
export CPP_CONTEXT_EMBEDDING_API_KEY='read-from-your-secret-store'
cpp-context index /path/to/project
```

The endpoint receives bounded symbol text through `POST /embeddings`. Requests
have explicit timeouts and no hidden retries. API keys are excluded from object
representations, CLI output, and sanitized errors.

### Ask questions and serve the API

`ask` uses an OpenAI-compatible chat-completions endpoint. The model receives only
the bounded, connected excerpts selected by retrieval; citations are accepted
only for source IDs that were actually read.

```bash
export CPP_CONTEXT_PROJECT_ROOT=/path/to/project
export CPP_CONTEXT_DATABASE=/path/to/project/.cpp-context/index.db
export CPP_CONTEXT_LLM_BASE_URL=https://provider.example/v1
export CPP_CONTEXT_LLM_MODEL=your-chat-model
export CPP_CONTEXT_LLM_API_KEY='read-from-your-secret-store'

cpp-context ask "How does an incoming packet reach the decoder?"
cpp-context serve --host 127.0.0.1 --port 8000
```

`serve` exposes `GET /health`, `POST /v1/context`, and `POST /v1/answer`.
The answer route returns HTTP 503 when the server was started without LLM
configuration; local context search continues to work.

## Configuration

CLI path/provider flags override these environment variables. Secrets have no CLI
flag so they do not accidentally appear in shell history or process listings.

| Variable | Purpose | Default |
| --- | --- | --- |
| `CPP_CONTEXT_PROJECT_ROOT` | project root | current directory |
| `CPP_CONTEXT_INDEX_DIRECTORY` | local index directory | `<project>/.cpp-context` |
| `CPP_CONTEXT_DATABASE` | SQLite file | `<index-directory>/index.db` |
| `CPP_CONTEXT_COMPILE_COMMANDS` | compilation database | `<project>/build/compile_commands.json` |
| `LIBCLANG_LIBRARY_FILE` | exact compatible native libclang | auto-discovered |
| `CPP_CONTEXT_MAX_TOKENS` | default packed context budget | `16000` |
| `CPP_CONTEXT_RETRIEVAL_LIMIT` | candidates per search backend | `20` |
| `CPP_CONTEXT_EMBEDDING_PROVIDER` | `local` or `openai` | `local` |
| `CPP_CONTEXT_EMBEDDING_DIMENSIONS` | local feature-hash vector size | `384` |
| `CPP_CONTEXT_EMBEDDING_BASE_URL` | OpenAI-compatible embedding API | unset |
| `CPP_CONTEXT_EMBEDDING_MODEL` | hosted embedding model | unset |
| `CPP_CONTEXT_EMBEDDING_API_KEY` | hosted embedding secret | unset |
| `CPP_CONTEXT_LLM_BASE_URL` | OpenAI-compatible chat API | unset |
| `CPP_CONTEXT_LLM_MODEL` | chat model | unset |
| `CPP_CONTEXT_LLM_API_KEY` | chat API secret | unset |
| `CPP_CONTEXT_PROVIDER_TIMEOUT` | HTTP timeout in seconds | `30` |
| `CPP_CONTEXT_SERVE_HOST` / `CPP_CONTEXT_SERVE_PORT` | bind address | `127.0.0.1` / `8000` |

## Development

```bash
python -m pip install -e '.[dev,clang,api]'
ruff format --check .
ruff check .
pytest
python -m compileall -q src tests
```

Without installing, use `PYTHONPATH=src python -m cpp_context_engine --help`.
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

Install only the HTTP transport without development tools using:

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

The CLI composition root creates project-scoped SQLite lexical, symbol, vector,
graph, source-reader, retrieval, and optional LLM adapters. Library consumers can
still inject replacements. The package never creates a provider or reads a key at
import time.

## Current limits

- Compiler facts are limited to code visible in compilation-database translation
  units and project-local headers; system/library implementations are excluded.
- SQLite cosine search scans vectors for the configured project/model. This is a
  practical MVP, not yet an ANN index for multi-million-symbol repositories.
- Source changes are noticed by `index`; search does not watch the filesystem.
- Context token counts use a conservative character estimate rather than a
  provider-specific tokenizer.
- Model answers still require review. Compiler-derived locations and graph paths
  improve provenance but do not prove behavioral correctness.
