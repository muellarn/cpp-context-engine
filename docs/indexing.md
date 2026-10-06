# Compiler-aware indexing

`ProjectIndexer` combines a validated JSON compilation database, libclang, and
`SQLiteStore`. The LLM and transport layers are deliberately not involved.

The first navigation ingestion into an owned, unpublished private database uses
batch foreign-key validation when both complete current and changed TU sets are
known and equal. Primary/unique constraints, triggers and rollback journaling
remain active; nonunique lookup indexes are rebuilt before finalization. Full
foreign-key, integrity and FTS checks precede commit. Online foreign-key enforcement
is restored and verified outside the transaction before the database can be used
or published. This path temporarily uses a 128 MiB SQLite page-cache target.
Existing databases, unknown/replacing streams, subsequent ingestions, full-profile
and Deep updates retain normal online foreign-key/cascade behavior. Failed rollback
or enforcement restoration closes the private connection and prevents publication.

Install the optional compiler binding and point the adapter at libclang when it
is not on the platform's default library path:

```bash
python -m pip install -e '.[clang]'
export LIBCLANG_LIBRARY_FILE=/usr/lib/llvm-18/lib/libclang.so
```

The library also searches common LLVM installation directories. The Python
binding and native library must have compatible major versions.

## Full Clang-18 analyzer companion

Schema v17 records the producing native analyzer's SHA256 per translation unit,
including its existing build/configuration scope. After upgrading the analyzer,
run normal incremental `index` for each enabled build variant. Changed identities
and legacy TUs with unknown provenance are reindexed even if sources and commands
are unchanged; matching identities keep incremental hits. Full-to-Deep reuse
rejects stale or unknown producers and requires this refresh first. Deep overlays
retain their own binary-bound provenance without relabeling retained navigation
facts. No wire-protocol change is required for a semantic analyzer fix.

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
The mandatory `compact_structural_keys_v1` capability projects the fixed CFG,
analysis, access, memory-location, evidence and effect key families to typed
SHA-256 references only when shorter than their original UTF-8 spelling. Native
solver keys, ordering, cap selection and deduplication remain unchanged. The adapter
reconstructs original identities from structural fields; memory definitions carry
their original `identity_key` once because external offsets and field USRs cannot
be reconstructed from names. Disk registries stay compact; existing typed builder
passes resolve references and reject unknown, conflicting or duplicate aliases.
Persisted IDs, ordering and summary hashes remain unchanged. Return-origin identity
keys retain their original spelling. Both peers
must confirm this capability during `hello`: rebuild the companion together with
the Python package. Old clients are rejected before analysis, not silently given
keys they would interpret as different graph identities.
Macro expansions carry independent `spelling_span` and `expansion_span` objects.
Every emitted span is validated against one Clang file buffer and ordered byte
offsets. Genuinely cross-file endpoints use Clang's deterministic file-character
range when available. If independent macro spelling endpoints instead mix a use
site with an earlier definition in the same file, the invalid spelling span is
omitted; the separately mapped expansion span and macro stack remain available.
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

### Index profiles and coverage

`full` remains the default profile and retains all existing facts. The explicit
`navigation` profile is negotiated through the optional `analysis_profiles_v1`
capability without changing protocol version 5. Full requests omit the profile
field; among compatible companions, navigation fails
before analysis if the companion does not advertise profile support.

Navigation emits only project/build/TU and dependency state, files/includes,
symbols, occurrences, semantic edges, callsites, call targets, and the inputs
used for the unchanged embedding/search path. It emits no CFG, data-flow,
summary, binding, or interprocedural-flow facts. Schema v13 records the selected
profile on each build and translation unit plus independent navigation, CFG,
data-flow, and summary coverage flags. A profile transition reindexes affected
translation units in the normal atomic replacement transaction, so a failed
transition preserves the prior generation and a successful transition cannot
leave stale deep rows visible. Deep API and MCP queries inspect this coverage and
return a structured unavailable result when the selected build scope is not deep.

The handshake advertises optional `gzip_jsonl_v1` transport support without
changing protocol version 5, fact schemas, or stable IDs. Probes remain plain JSONL.
Clients request gzip only after a compatible plain probe advertised it; a
compatible companion without gzip support remains plain. The native
sink suppresses duplicate sort keys before serialization and emits first-seen
facts incrementally through a bounded gzip level-1 writer. The Python adapter
decompresses and parses fragmented records incrementally into compact,
length-framed, disk-backed fact-kind registries. Those registries reuse already
validated objects instead of decoding every fact from JSON a second time.
Analyzer capacity is refilled as soon as a process completes; bounded conversion
batches overlap in compilation-database admission order, and SQLite publication
keeps that order. Cross-reference validation and domain construction happen only after a
matching successful `complete`, so malformed, cancelled, timed-out, or
limit-exhausted responses cannot produce a durable partial batch.

Compressed wire bytes, decoded bytes, one decoded record, stderr, and wall time
have independent hard limits. Exhaustion is an indexing error; relational facts
are never truncated. Companion processes run in a killable process group and are
joined on cancellation or failure. `CPP_CONTEXT_ANALYZER_MAX_WORKERS` is a hard
concurrency bound and defaults conservatively to one because per-TU memory varies
widely. Completed registries, spool bytes, spool files, and converted domain
batches have separate hard bounds. Their environment variables are documented in
the README; omitted spool limits are derived from the worker and decoded-byte
limits. The [large-TU benchmark](benchmarks/large-tu.md) documents the local KiCad
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

The optional protocol-v5 `symbol_text_chunks_v1` capability preserves large symbol
text without enlarging the record limit. The client confirms it explicitly in the
analysis handshake. Otherwise the companion retains its original single-record
format; upgraded clients also accept older companions. There is no signature,
source-text, model, schema or MCP change.

When negotiated, symbol `signature`, `documentation` and `source_text` fields
larger than 64 KiB of UTF-8 are replaced by a `text_chunks` descriptor containing
each field's byte length and SHA-256. Immediately following `symbol_text_chunk_v1`
facts contain the same symbol key, field name, zero-based index and UTF-8 `text`
of at most 64 KiB. Fields follow signature/documentation/source order. Fragments
cannot interleave with another symbol or fact; missing, duplicated, reordered,
unreferenced or inconsistent data fails the entire analysis. Length declarations
do not allocate buffers. JSON escaping, fragment headers and all text still count
toward the unchanged wire/decoded budgets, and each registry frame remains bounded
and charged to the shared spool budget. Oversized non-text records still fail.

The indexing path spools validated physical fragments separately and reconstructs
one logical symbol only in the batch builder. Public native-client callbacks and
`analyze()` results retain the original complete logical facts and their order.
No reassembled large symbol is written back into a registry frame. Normal binary
producer hashing invalidates results produced by the previous companion.

Semantic records cover files, functions, methods, classes, structs, enums,
namespaces, variables, aliases, and macros. Each symbol has a stable USR-derived
ID where Clang supplies a USR, an exact source range, source text/hash,
documentation, signature, build configuration, and metadata. Occurrences retain
declaration/reference/call/type/macro-expansion ranges.

Native function and function-template `signature` values describe declarations,
not their outer bodies or constructor initializer lists. Clang still prints
parameter defaults, qualifiers, attributes, templates and constraints, including
lambda bodies within those declaration expressions. `source_text`, physical
spans, symbol identities and non-signature facts retain the complete indexed
source. Other symbol kinds retain their existing printing behavior.

This intentionally changes MCP-visible signatures, symbol-only search and the
signature contribution to hybrid ranking. Body-only terms remain in lexical
source search and full-code retrieval, but no longer count as function-signature
matches. Embedding text includes the new signature and the unchanged source, so
its content identity also changes. After upgrading the native binary, run normal
incremental indexing for every enabled build variant: the existing binary-SHA256
producer identity (including compiled header changes) forces affected TUs to be
refreshed, and content-addressed embeddings use the refreshed text. No protocol,
schema or output-budget change is involved. This correction does not by itself
prove that a previously oversized KiCad TU now fits the decoded-output limit.

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

Canonical refresh seeks the preferred variant ID before loading its full snapshot,
in bounded batches of requested symbols. Definition, build, and translation-unit
preference ordering is unchanged; discarded variants no longer transfer their
snapshot payloads to Python. Persisted legacy snapshot provenance is preserved.

Schema v3 separates canonical Clang symbol identity from deduplicated
build/configuration/translation-unit `symbol_variants`. Occurrences and graph edges
carry the same provenance. Graph edges have stable evidence IDs, so repeated calls
between the same endpoint symbols remain distinct callsites. Build-filtered FTS,
vector, symbol and graph reads use `BuildScope`; union results retain their build
labels. `SQLiteStore.remove_build_variant` is the only operation that removes an
entire named build.

Bulk symbol lookup first seeks the requested variant IDs within the project, then
filters those candidates to the requested builds before resolving identities.
Applying the build filter inside the ID seek can make SQLite scan an entire build
for as few as three requested IDs. Canonical-ID lookup retains its scoped index;
exact-ID precedence, result order, and legacy default-build fallback are unchanged.

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

Schema v11 stores propagated interprocedural effects and return origins as bounded,
deterministic compressed payloads while retaining relational local facts and the
existing query API. Schema v12 separates variant-to-vector references from a project-local,
content-addressed vector pool. Its key includes the public model, the complete
non-secret provider configuration identity, dimension, and SHA-256 of the exact
bounded embedding text; the text is retained to detect a hash collision. Equal
inputs across translation units or named builds therefore share one vector while
search and stale cleanup remain variant- and build-scoped. Missing IDs, symbol
loads, provider calls, validation, and writes are processed in fixed-size batches
inside one transaction, followed by deterministic orphan-vector collection.
Legacy hosted vectors are invalidated during the v12 migration because schema v11 did not
persist their endpoint identity; local vectors retain their exact search behavior.

Schema v13 adds explicit build/TU profile and fact-family coverage. Legacy native
rows are migrated as `full` only where their prior advanced-completeness marker
proved deep coverage; other rows remain explicitly incomplete.

Schema v14 stores each canonical little-endian Float64 embedding either raw or,
when it is strictly smaller, with deterministic zlib level-3 compression. Search
decodes to the same canonical bytes, so cosine scores and tie ordering do not
change. Migration validates every bounded vector and reference atomically. SQLite
does not return freed pages to the filesystem automatically; to reclaim physical
space in an existing migrated database, stop all users, make a verified backup,
and run an explicit `VACUUM`. Fresh databases need no compaction step.

Schema v18 stores symbol-variant snapshots as versioned zlib level-3 payloads when
smaller than their original JSON. Decoding enforces a 512-MiB UTF-8 bound and rejects
invalid, truncated, or trailing compressed data. Legacy JSON snapshots migrate in
one transaction; every source/signature byte, ID, metadata field, and build/TU
membership is retained. Search ordering, FTS content, embeddings, and navigation
parity use decoded snapshots. This does not deduplicate distinct variants or
redesign FTS. As with vector migration, existing files may need explicit offline
compaction to reclaim freed pages; snapshot compression ratios are not whole-index
size reductions or a guarantee that a full project fits a particular disk budget.

Schema v19 shares identical immutable snapshot contents within each project.
New variant IDs serialize only the shared content. A complete provenance-bearing
snapshot is constructed only when an existing ID needs its exact invalidation
comparison; legacy JSON differences and intermediate duplicate updates still
invalidate embeddings under the original rules.
Every variant row and its public ID, build, configuration and translation-unit
provenance remains separate. Four provenance fields move out of the shared JSON
only when all are present, match the variant columns and use the canonical JSON
serialization. A format flag identifies this split representation. Other legacy
JSON remains byte-for-byte intact, including absent or differing provenance.
Hash matches are checked against exact decoded content; collisions or corrupt
payloads fail the transaction. Only new contents are compressed, in bounded
batches capped at 128 records and 16 MiB of payload (one larger record is handled
alone). Both encoded migration input and decoded content have byte bounds.
Unreferenced contents are reclaimed during transactional orphan cleanup.
Migration preserves foreign keys and existing FTS documents atomically; FTS itself
is unchanged. Semantic acceptance hashes contents and content references, not
insertion-order-dependent pool rowids. Pool payload savings alone are not a
physical database-size estimate or a Whole-index capacity guarantee.

Schema v20 consolidates the four overlapping graph endpoint indexes into two:
`(project_id, source_id, build_variant, relation)` and the matching target index.
Their endpoint prefix supports foreign-key lookups without scanning a project,
while scoped graph reads still seek endpoint, build and relation. The edge table,
its primary key, TU index, evidence IDs and every provenance field are unchanged.
One narrow partial index covers build-wide override-closure seeds and contains
only override edges, avoiding scans of unrelated builds or non-override evidence.
Index replacement is transactional. Existing database files retain freed pages
for reuse; the index-byte reduction is not an automatic file-size reduction.

FTS5 searches names, signatures, documentation, and exact source text.
Schema v21 removes the variant FTS content copy: a narrow, stable integer document
mapping connects each FTS row to the original variant snapshot. Every variant
retains its own document, token positions, lengths and BM25 statistics. The eight
columns, tokenizer and weights do not change. Migration compares every old content
field and the complete document mapping before atomic replacement. FTS validation
uses `integrity-check` with `rank=1`, including the external content. Updates remove
old tokens before changing snapshots; fresh deferred rebuilds use `delete-all`.
The content reader caches only one document up to 64 KiB encoded and 262,144 text
characters; larger documents are decoded without retaining them in that cache.
Existing files need compaction to shrink; occupied-page savings alone do not prove
that a full index will fit available physical storage.

Embeddings are stored by content, model/configuration identity, and dimension. `SQLiteVectorSearch` accepts any provider
implementing `EmbeddingProvider`; `SQLiteStore.search_vector` computes true cosine
similarity and rejects empty, non-finite, zero-magnitude, or dimension-mismatched
vectors.

Schema v22 stores each embedding attachment in its composite primary-key B-tree
(`WITHOUT ROWID`) and keeps one covering content index. The former search index
duplicated the content index's project/model/configuration/dimension prefix;
variant lookups retain the primary key. The content index includes the remaining
primary-key columns, so net savings include its increased width. Migration is
transactional, checks exact row parity and foreign keys, and preserves all vector
identities, configuration/dimension boundaries, cascades, cosine scores and ties.

Schema v23 stores graph facts with project-local integer endpoint references and
an interned exact TU/configuration pair. The `edges` read view exposes the same
eight logical columns, including arbitrary historical evidence IDs and provenance;
public graph ordering, scopes, duplicate-ID behavior and cascades are unchanged.
The physical records retain the `(project_id, id)` primary key. Endpoint indexes
use integer references, with a separate partial override index. Migration checks
bidirectional exact row parity and all new foreign keys inside one transaction.
Fresh writes intern bounded batches, and TU replacement removes leaf facts before
their provenance rows. This layout does not deduplicate or discard edge evidence.

Schema v24 interns occurrence TU/configuration/build/path context while retaining
all fourteen logical values through the `occurrences` read view. The original
`(project, TU, occurrence ID)` replacement identity and direct canonical-symbol
foreign key remain; enclosing-symbol strings intentionally have no such foreign
key. Migration preserves row order keys, exact provenance (including historical
configuration mismatches), and verifies bidirectional row parity and foreign keys
atomically. Bounded writes retain input order, including multirow replacements;
public ordering ties retain the original row-order key. Every child foreign key
keeps a complete lookup index. Semantic validation hashes the logical view, not
the allocation-dependent map or row keys.

Missing-embedding projection traverses variants by immutable snapshot-content ID
and row-order key. It loads and decodes a shared stored snapshot once per group,
retaining only one bounded projected text rather than one payload per variant in
the batch. Attachments remain per variant, and all batches publish in the same
atomic embedding session. Provider batch order can differ; exact vector-byte
parity is verified for the deterministic local provider, not promised for remote
providers whose output depends on batch order or mutable external state.
