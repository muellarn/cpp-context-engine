# Compiler-aware indexing

`ProjectIndexer` combines a validated JSON compilation database, libclang, and
`SQLiteStore`. The LLM and transport layers are deliberately not involved.

Install the optional compiler binding and point the adapter at libclang when it
is not on the platform's default library path:

```bash
python -m pip install -e '.[clang]'
export LIBCLANG_LIBRARY_FILE=/usr/lib/llvm-18/lib/libclang.so
```

The library also searches common LLVM installation directories. The Python
binding and native library must have compatible major versions.

## Indexing from Python

```python
from pathlib import Path

from cpp_context_engine.ingestion import ClangIngestor, ProjectIndexer
from cpp_context_engine.storage import SQLiteStore

root = Path("/workspace/project")
with SQLiteStore(root / ".cpp-context" / "index.db", project_root=root) as store:
    result = ProjectIndexer(ClangIngestor(), store).index(
        root, root / "build" / "compile_commands.json"
    )
```

Compilation-database entries are checked before parsing. Exactly one of
`arguments` and `command` must be present, source files must exist, and relative
paths are resolved from the database location/entry working directory. The exact
compiler invocation is retained while driver-only output/dependency flags are
removed from the arguments sent to libclang.

Compiler errors abort a translation unit and raise `TranslationUnitError`. Its
message contains the source path, normalized parser arguments, diagnostic
severity, exact location, diagnostic text, and warning option when available.

## Persisted facts

Semantic records cover files, functions, methods, classes, structs, enums,
namespaces, variables, aliases, and macros. Each symbol has a stable USR-derived
ID where Clang supplies a USR, an exact source range, source text/hash,
documentation, signature, build configuration, and metadata. Occurrences retain
declaration/reference/call/type/macro-expansion ranges.

The graph stores `CONTAINS`, `REFERENCES`, `CALLS`, `INHERITS`, `OVERRIDES`,
`USES_TYPE`, and project-local `INCLUDES` relationships. Override edges use
libclang's native override API. System declarations and system include graph
nodes are intentionally excluded.

SQLite updates are atomic per indexing run. Translation units retain their
compiler-command, source, project-header dependency, and diagnostic state. An
unchanged run performs no parsing. A changed project header invalidates every
translation unit that recorded it; removed compilation commands cascade their
now-unreferenced symbols, occurrences, graph edges, embeddings, and FTS rows.
Symbols seen by multiple translation units retain origin mappings so updating or
removing one unit does not discard facts still used by another.

FTS5 searches names, signatures, documentation, and exact source text. Embeddings
are stored by model ID and dimension. `SQLiteVectorSearch` accepts any provider
implementing `EmbeddingProvider`; `SQLiteStore.search_vector` computes true cosine
similarity and rejects empty, non-finite, zero-magnitude, or dimension-mismatched
vectors.
