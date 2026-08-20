# C++ Context Engine

C++ Context Engine is a foundation for compiler-aware retrieval over large C++
codebases. It is intended to combine exact symbols and code relationships with
lexical and vector search so that an LLM can load a small, connected set of
relevant source locations instead of the entire repository.

The current repository deliberately contains only the stable architecture and a
diagnostic CLI. Concrete Clang/SCIP ingestion, persistence, search indexes,
retrieval ranking, LLM integrations, and an HTTP transport can be implemented
independently behind typed protocols.

## Requirements

- Python 3.11 or newer
- A C++ compilation database (`compile_commands.json`) for future ingestion

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
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

- `ingestion`: C++/compiler index adapters producing normalized symbols
- `storage`: durable symbol, source, and metadata persistence
- `search`: lexical and vector candidate generation
- `graph`: code relationship storage and traversal
- `retrieval`: candidate fusion, graph expansion, and context packing
- `llm`: model-provider adapters
- `api`: transport-neutral service request/response contracts

Shared immutable domain types live in `models.py`. Application paths and
environment-driven settings live in `config.py`. The modules currently expose
Python `Protocol` interfaces, keeping implementations replaceable and tests easy
to isolate.

## Planned first vertical slice

1. Read a compilation database and ingest symbols through a Clang/SCIP adapter.
2. Persist symbols and relationships in SQLite with FTS5 lexical search.
3. Add a local embedding index as a separate `VectorSearch` implementation.
4. Fuse candidates, expand call/type relationships, and pack bounded context.
5. Expose retrieval through the API service and an LLM tool interface.
