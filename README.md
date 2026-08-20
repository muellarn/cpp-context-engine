# C++ Context Engine

C++ Context Engine is a foundation for compiler-aware retrieval over large C++
codebases. It is intended to combine exact symbols and code relationships with
lexical and vector search so that an LLM can load a small, connected set of
relevant source locations instead of the entire repository.

The repository provides Clang-based C++ ingestion, transactional SQLite storage,
FTS5 lexical search, and a provider-neutral local cosine-vector index. Retrieval
ranking, LLM integrations, and HTTP transports remain independent behind typed
protocols.

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

## Architecture

The package is split along the intended processing pipeline:

- `ingestion`: validated compilation databases, libclang analysis, and incremental indexing
- `storage`: SQLite symbols, occurrences, graph edges, dependency state, FTS5, and vectors
- `search`: lexical and provider-neutral vector candidate generation
- `graph`: code relationship storage and traversal
- `retrieval`: candidate fusion, graph expansion, and context packing
- `llm`: model-provider adapters
- `api`: transport-neutral service request/response contracts

Shared immutable domain types live in `models.py`. Application paths and
environment-driven settings live in `config.py`. The modules currently expose
Python `Protocol` interfaces, keeping implementations replaceable and tests easy
to isolate.

See [Compiler-aware indexing](docs/indexing.md) for the ingestion/storage API,
incremental behavior, libclang configuration, and current guarantees.
