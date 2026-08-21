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

Protocol 5 is newline-delimited JSON. A process must receive `hello` first and
returns `hello` with analyzer version, Clang major, and capabilities. An analysis
then returns `begin`, zero or more `fact` records, and `complete`. Fact records
are `file`, `symbol`, `occurrence`, `edge`, `include`, and the versioned
`cfg_graph_v1`, `cfg_block_v1`, `cfg_element_v1`, `cfg_edge_v1`, `callsite_v1`,
`call_target_v1`, `data_flow_analysis_v1`, `memory_location_v1`,
`data_access_v1`, and `data_flow_evidence_v1` types;
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
`cfg_facts_available=true`. The callsite capabilities similarly produce
`call_facts_available=true`.
The `intraprocedural_dataflow_v1` and `points_to_v1` capabilities similarly
produce `data_flow_facts_available=true`.

The handshake advertises optional `gzip_jsonl_v1` transport support without
changing protocol version 5, fact schemas, or stable IDs. Probes and old clients
remain plain JSONL. A new client requests gzip only after a plain probe advertised
it; a new client talking to an old companion therefore remains plain. The native
sink suppresses duplicate sort keys before serialization and emits first-seen
facts incrementally through a bounded gzip level-1 writer. The Python adapter
decompresses and parses fragmented records incrementally into disk-backed
fact-kind registries. Cross-reference validation and domain construction happen
only after a matching successful `complete`, so malformed, cancelled, timed-out,
or limit-exhausted responses cannot produce a durable partial batch.

Compressed wire bytes, decoded bytes, one decoded record, stderr, and wall time
have independent hard limits. Exhaustion is an indexing error; relational facts
are never truncated. Companion processes run in a killable process group and are
joined on cancellation or failure. `CPP_CONTEXT_ANALYZER_MAX_WORKERS` is a hard
concurrency bound and defaults conservatively to one because per-TU memory varies
widely. The [large-TU benchmark](benchmarks/large-tu.md) documents the local KiCad
Clipper reproduction; it is deliberately not a CI workload.

### Callsites and C++ dispatch

The companion stores every syntactic call separately, including consecutive
identical calls. A callsite records its owner, dispatch form, static target when
known, independent spelling and expansion ranges, the innermost-to-outermost
project-macro expansion stack, and exact build/configuration/TU provenance.
`target_set_complete` is false whenever Clang cannot prove a closed target set;
`unresolved_reason` then states the gap instead of silently dropping the call.

Each target edge records `certain` or `possible`, a deterministic confidence in
`[0,1]`, its confidence reason, derivation, evidence range, and build provenance.
Confidence is ranking evidence, not a runtime probability. Direct AST calls,
qualified virtual calls, final dispatch, and targets proven by
`CXXMethodDecl::getDevirtualizedMethod` are certain. Otherwise the static virtual
method and build-local transitive overrides are possible; the set remains
incomplete because external unindexed derived types can add overriders. Concrete
lambda, generic-lambda specialization, and function-object `operator()` targets
are retained.

Function and class template specializations and instantiations retain template
kind, arguments, any Clang-provided point of instantiation, and
`SPECIALIZES`/`INSTANTIATES` graph edges. Calls emitted through project macros
retain expansion frames and a `GENERATED_BY_MACRO` relation. Local function- and
member-pointer assignments are propagated through the CFG. A complete singleton
target set is certain; every target in a non-singleton or incomplete set is
possible. Copies and conditional target sets are preserved, a known null pointer
has a complete empty target set, and unknown parameters or unsupported expressions
remain explicit incomplete indirect calls. Dependent, uninstantiated template
calls likewise remain explicit and never gain a certain target.

`SQLiteStore.callsites`, `get_callsite`, and `call_targets` are build-scoped,
bounded internal reads with deterministic ordering and explicit truncation.
The analysis service exposes bounded caller/callee evidence through CLI, HTTP,
and MCP while retaining certainty, confidence, completeness, and provenance.

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
`truncated` flag. CLI, HTTP, and MCP CFG tools apply aggregate graph, block,
element, and edge budgets across the selected build scope.

### Intraprocedural data flow and points-to facts

Each function CFG has one build-specific fixed-point result. Locations distinguish
parameters, locals, globals, function returns, call returns, known dereferences,
field paths, and an explicit unknown location. Access facts distinguish parameter
definitions, initialization, assignment, compound assignment, increment/decrement,
call returns, unknown clobbers, ordinary reads, call arguments, return values, and
conditions. Evidence connects reaching and overwritten definitions; references
and resolved dereferences add must- or may-alias evidence.

The analysis has deterministic hard limits: 64 fixed-point iterations, 64 targets
per alias/points-to set, eight access-path components, and 4096 locations per
function. Exhaustion is recorded in `incomplete_reasons`; it never silently drops
precision while claiming completeness. Pointer arithmetic, unions, reinterpret
casts, address escape, external call effects, volatile/atomic storage, inline
assembly, and unknown lvalues are modeled conservatively and likewise make the
affected function explicitly incomplete. These are compiler evidence facts only;
the analyzer does not label code dead, redundant, or buggy.

Data-flow rows retain the same build/configuration/TU provenance as their CFG and
cascade atomically when a translation unit or build variant is replaced. Public
CLI, HTTP, and MCP queries apply aggregate analysis, location, access, and evidence
budgets across the selected build scope.

These completeness flags describe the documented static-analysis model, not every
possible C++ execution. The [soundness and completeness matrix](soundness-and-completeness.md)
defines how to interpret certainty, confidence, target-set completeness, unknown
effects, dynamic behavior, and build variants.

The libclang path remains a baseline fallback. Baseline symbols and occurrences
are explicitly marked `analysis_backend=libclang-baseline` and
`advanced_facts_complete=false`; selecting a validated companion invalidates and
reindexes such translation units. The baseline does not emit CFG, callsite, or
data-flow facts. Dead-code and other semantic judgments remain outside the analyzer.

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

Schema v6 adds separate callsite and call-target tables with foreign keys to
symbols and translation units. TU replacement and build removal cascade their
call facts. Existing native rows are marked incomplete on migration so they
cannot masquerade as complete dispatch evidence; the migration is atomic.

Schema v7 adds analysis, memory-location, access, and evidence tables with foreign
keys to CFGs, blocks, elements, symbols, and translation units. Old native rows
are marked incomplete so the normal incremental path refreshes them with protocol
v4 facts. The migration and every TU replacement are atomic.

Schema v8 stores a function summary for each build/configuration/TU body variant,
plus local and propagated effects, return origins, call argument/result bindings,
and cross-call flow evidence. Protocol v5 retains callsite, concrete target
certainty, and full build provenance on every cross-call flow. The solver processes
call-graph SCCs under deterministic iteration, SCC-size, and effect-count limits;
unknown or external targets and exhausted limits are explicit incomplete reasons.
Named builds are solved independently. On TU replacement, reverse call-graph
invalidation selects the changed functions and their transitive callers, then adds
only the callees required to solve those affected summaries; unrelated summary
solution hashes remain unchanged.

Schema v9 adds lookup indexes for canonical-symbol refresh, foreign-key checks,
and translation-unit replacement. Schema v10 removes the redundant symbol snapshot
from translation-unit membership rows; canonical symbols continue to be derived
from the versioned build/TU `symbol_variants` snapshots.

FTS5 searches names, signatures, documentation, and exact source text. Embeddings
are stored by model ID and dimension. `SQLiteVectorSearch` accepts any provider
implementing `EmbeddingProvider`; `SQLiteStore.search_vector` computes true cosine
similarity and rejects empty, non-finite, zero-magnitude, or dimension-mismatched
vectors.
