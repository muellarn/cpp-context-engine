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

Protocol 2 is newline-delimited JSON. A process must receive `hello` first and
returns `hello` with analyzer version, Clang major, and capabilities. An analysis
then returns `begin`, zero or more `fact` records, and `complete`. Fact records
are `file`, `symbol`, `occurrence`, `edge`, `include`, and the versioned
`cfg_graph_v1`, `cfg_block_v1`, `cfg_element_v1`, and `cfg_edge_v1` types;
references use stable
USR or location-derived keys that the Python adapter converts to canonical IDs.
Macro expansions carry independent `spelling_span` and `expansion_span` objects.
Symbols can carry `template_kind`, `template_arguments`,
`is_lambda_call_operator`, and `stable_lambda_key` metadata.

Only stdout is protocol data. Native diagnostics use stderr. The adapter accepts
no command string, invokes no shell, confines fact paths to the project, validates
the handshake before analysis, and enforces operator-owned timeout and byte
limits. This companion currently supports Linux and exactly Clang major 18. Its
handshake must include `function_cfg_v1`; `cpp-context doctor` exposes that as
`cfg_facts_available=true`.

### Function control-flow graphs

For every project-local function definition, the companion calls
`clang::CFG::buildCFG` with one fixed profile. It disables trivial-false-edge
pruning and enables all-statement retention, constructor initializers, implicit
and temporary destructors, lifetime ends, loop exits, scopes, static-initializer
branches, `new` allocators, default initializer expressions, rich constructors,
elided-constructor marking, and virtual-base branches. EH edges are enabled only
when the concrete compiler invocation enables C++ exceptions. Every graph stores
the complete option profile.

CFG graphs, blocks, elements, and edges are independent domain records and SQLite
tables, not synthetic `CodeSymbol` records. Stable IDs include the build variant,
build configuration, translation unit, function identity, Clang block index, and
element or successor position. Blocks retain entry reachability and unreachable
blocks. Elements retain build/TU provenance and independent spelling/expansion
spans when Clang maps those locations inside the project. Terminators and labels
remain block facts. Edges are classified as `fallthrough`, `true`, `false`,
`case`, `default`, `loop_back`, `break`, `continue`, `return`, `goto`, or
`exception`; infeasible alternate successors are retained and marked.

Clang 18 exposes one function exit block rather than distinct normal and uncaught
exception sinks. `normal_exit_block_id` therefore identifies that exit and
`exceptional_exit_block_id` is null. Catch dispatch and supported EH flow are
still stored as exception edges. Exact block shape, lifetime elements, and EH
edges depend on the pinned Clang version and concrete build configuration.

`SQLiteStore.cfg_graphs`, `cfg_blocks`, `cfg_elements`, and `cfg_edges` are
bounded, build-scoped reads with deterministic ordering and an explicit
`truncated` flag. Dedicated CLI/API/MCP CFG tools belong to the later interface
issue.

The libclang path remains a baseline fallback. Baseline symbols and occurrences
are explicitly marked `analysis_backend=libclang-baseline` and
`advanced_facts_complete=false`; selecting a validated companion invalidates and
reindexes such translation units. The baseline does not emit CFG facts. Indirect
calls, def-use/dataflow, and dead-code judgments remain out of scope.

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

Schema v5 adds build/TU-specific CFG graph, block, element, and edge tables.
Replacing or removing a translation unit cascades only its CFG rows; other
translation units and build variants remain intact.

FTS5 searches names, signatures, documentation, and exact source text. Embeddings
are stored by model ID and dimension. `SQLiteVectorSearch` accepts any provider
implementing `EmbeddingProvider`; `SQLiteStore.search_vector` computes true cosine
similarity and rejects empty, non-finite, zero-magnitude, or dimension-mismatched
vectors.
