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

## Full Clang-18 analyzer companion

The optional `native/clang-analyzer` executable uses Clang LibTooling's full AST,
`SourceManager`, and `PPCallbacks`. Configure it with CMake's installed LLVM and
Clang package files, then select it explicitly:

```bash
cmake -S native/clang-analyzer -B build/clang-analyzer \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
cmake --build build/clang-analyzer
cpp-context doctor --clang-analyzer build/clang-analyzer/cpp-context-clang-analyzer
cpp-context index /workspace/project \
  --clang-analyzer build/clang-analyzer/cpp-context-clang-analyzer
```

Protocol 1 is newline-delimited JSON. A process must receive `hello` first and
returns `hello` with analyzer version, Clang major, and capabilities. An analysis
then returns `begin`, zero or more `fact` records, and `complete`. Fact records
are `file`, `symbol`, `occurrence`, `edge`, and `include`; references use stable
USR or location-derived keys that the Python adapter converts to canonical IDs.
Macro expansions carry independent `spelling_span` and `expansion_span` objects.
Symbols can carry `template_kind`, `template_arguments`,
`is_lambda_call_operator`, and `stable_lambda_key` metadata.

Only stdout is protocol data. Native diagnostics use stderr. The adapter accepts
no command string, invokes no shell, confines fact paths to the project, validates
the handshake before analysis, and enforces operator-owned timeout and byte
limits. This companion currently supports Linux and exactly Clang major 18.

The libclang path remains a baseline fallback. Baseline symbols and occurrences
are explicitly marked `analysis_backend=libclang-baseline` and
`advanced_facts_complete=false`; selecting a validated companion invalidates and
reindexes such translation units. CFG, indirect calls, and dataflow remain out of
scope until their dedicated analysis stages are implemented.

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

For multiple configurations, bind each database to a named `BuildVariant` and
index it independently:

```python
from cpp_context_engine.models import BuildScope, BuildVariant, SearchQuery

debug = BuildVariant("debug", root / "build-debug" / "compile_commands.json")
release = BuildVariant("release", root / "build-release" / "compile_commands.json")
with SQLiteStore(root / ".cpp-context" / "index.db", project_root=root) as store:
    indexer = ProjectIndexer(ClangIngestor(), store)
    indexer.index(root, debug.compilation_database, build_variant=debug)
    indexer.index(root, release.compilation_database, build_variant=release)
    hits = store.search(SearchQuery("packet handler"), build_scope=BuildScope(("debug", "release")))
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

Schema v3 separates canonical Clang symbol identity from deduplicated
build/configuration/translation-unit `symbol_variants`. Occurrences and graph edges
carry the same provenance. Graph edges have stable evidence IDs, so repeated calls
between the same endpoint symbols remain distinct callsites. Build-filtered FTS,
vector, symbol and graph reads use `BuildScope`; union results retain their build
labels. `SQLiteStore.remove_build_variant` is the only operation that removes an
entire named build.

FTS5 searches names, signatures, documentation, and exact source text. Embeddings
are stored by model ID and dimension. `SQLiteVectorSearch` accepts any provider
implementing `EmbeddingProvider`; `SQLiteStore.search_vector` computes true cosine
similarity and rejects empty, non-finite, zero-magnitude, or dimension-mismatched
vectors.
