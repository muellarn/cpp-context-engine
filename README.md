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
On Ubuntu, `clang-18`, `libclang-18-dev`, and `llvm-18-dev` provide the compiler pieces.
Install the engine with compiler, HTTP, and MCP support:

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

For full AST, SourceManager, preprocessor, template, and lambda facts, build the
small version-locked LibTooling companion. No native executable is vendored:

```bash
cmake -S native/clang-analyzer -B build/clang-analyzer \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
cmake --build build/clang-analyzer --parallel
export CPP_CONTEXT_CLANG_ANALYZER="$PWD/build/clang-analyzer/cpp-context-clang-analyzer"
```

The companion is currently supported on Linux with LLVM/Clang 18. It uses a
versioned JSONL stdin/stdout protocol; stdout contains protocol records only and
Clang diagnostics go to stderr. The Python adapter validates protocol version,
Clang major, and required capabilities before indexing, invokes the executable
without a shell, bounds input/output/stderr bytes, and enforces a timeout.
It also persists per-function Clang CFG graphs, blocks, source and implicit
lifetime elements, terminators, entry reachability, and typed successor edges
with exact build/TU provenance. `doctor --json` reports `cfg_facts_available`.

Project indexing consumes completed translation units in compilation-database
order and retires each in-memory batch after staging it in SQLite. The number of
active or completed analyzer batches is therefore bounded by the configured
worker limit rather than project size. Publication is still one transaction:
other database connections see the previous complete index until commit, and an
analyzer error or cancellation rolls the entire run back. The SQLite WAL can grow
with a large run, so operators should keep sufficient temporary disk space.

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

The default `full` profile is unchanged. For a navigation-first index, explicitly
select `--profile navigation` (or pass `profile="navigation"` to the MCP
`index_project` tool). It preserves files, includes, symbols, references, graph
edges, calls, targets, embeddings, and ranking, while omitting CFG, data-flow,
summary, and binding rows. Coverage is stored per translation unit and build;
deep tools return a structured unavailable result rather than treating omitted
facts as a complete empty analysis. Switching profiles is an incremental,
transactional reindex that removes stale deep facts.

### Multiple build variants

Index each relevant compilation database under a stable operator-owned name:

```bash
cpp-context index /path/to/project \
  --build debug=/path/to/project/build-debug/compile_commands.json \
  --build release=/path/to/project/build-release/compile_commands.json

cpp-context search "feature handler" --project /path/to/project --build debug
cpp-context search "feature handler" --project /path/to/project \
  --build debug --build release --json
cpp-context builds --project /path/to/project --json
cpp-context cfg SYMBOL_ID --project /path/to/project --build debug --json
cpp-context flow SYMBOL_ID --project /path/to/project --build debug --json
```

A single-build query is strictly filtered. A union query returns separate evidence
for each build and labels every result with `build_variant` (and a stable
`variant_id`). Reindexing one name removes stale translation units only from that
variant. Remove a variant explicitly with `cpp-context remove-build NAME`.

### Reading analysis evidence

Call and flow results are evidence, not runtime proofs. On call edges, `certain`
means Clang selected that target for the indexed build and call form; `possible`
means the target is supported but not uniquely proven. `confidence` is a stable
ranking signal, not a probability. `target_set_complete` applies to one callsite in
one build: `false` means the world is open and additional runtime targets may
exist. A bounded or truncated query is also never evidence that omitted targets do
not exist.

Data-flow `complete` flags cover only the analyzer's documented model and limits.
Inspect `incomplete_reasons` and summary completeness before drawing conclusions,
and keep separate build variants separate unless a union is intentional. See the
[soundness and completeness matrix](docs/soundness-and-completeness.md) for dynamic
loading, external code, assembly, undefined behavior, aliases, wrappers, registries,
non-local jumps, and build-specific limitations.

For a long-running MCP operator, configure paths with repeated `--build NAME=PATH`
or `CPP_CONTEXT_BUILDS=debug=/path/debug/compile_commands.json,release=/path/release/compile_commands.json`.
`CPP_CONTEXT_BUILD_SCOPE=debug,release` selects the server-visible union. MCP callers
can select neither filesystem paths nor unconfigured builds.

Databases created before schema v3 are migrated as the `default` build. Baseline
search remains available, but advanced build/TU provenance is marked as requiring
one reindex; run the normal `index` command before relying on build-filtered facts.

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

Embedding vectors are content-addressed by the exact bounded text, provider/model
configuration, and vector dimension. Build-specific symbol variants keep separate
references but share one bit-identical vector when those inputs match. Generation,
validation, persistence, and missing-vector discovery run in bounded batches; a
failed batch rolls back the complete embedding update. Schema-v11 local vectors
are migrated into this shared pool. Legacy hosted vectors are regenerated once
because older databases did not retain the endpoint identity needed for safe reuse.

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

`serve` exposes `GET /health`, `GET /v1/builds`, `POST /v1/context`,
`POST /v1/answer`, `POST /v1/calls`, `POST /v1/cfg`, and `POST /v1/flow`.
Every analysis response identifies its selected build or explicit union and keeps
call certainty, completeness, derivation, confidence-ranking evidence, and
build/configuration/translation-unit provenance separate from source text.
The answer route returns HTTP 503 when the server was started without LLM
configuration; local context search continues to work.

## MCP server

Install the optional official MCP Python SDK v2 integration without the HTTP API:

```bash
python -m pip install -e '.[clang,mcp]'
```

The MCP server is permanently bound to paths and providers selected by its operator.
Tools cannot supply a project root, database, compilation database, or arbitrary file
path. Configure those values when launching the process, then use stdio (the default):

```bash
export CPP_CONTEXT_PROJECT_ROOT=/path/to/project
export CPP_CONTEXT_DATABASE=/path/to/project/.cpp-context/index.db
export CPP_CONTEXT_COMPILE_COMMANDS=/path/to/project/build/compile_commands.json
cpp-context-mcp
```

The equivalent main CLI command accepts non-secret operator options:

```bash
cpp-context mcp \
  --project /path/to/project \
  --db /path/to/project/.cpp-context/index.db \
  --compile-commands /path/to/project/build/compile_commands.json
```

The server may start before an index exists. An MCP client can call `index_project`,
then `list_builds`, `search_code`, `read_symbol`, `neighbors`, `callers`, `callees`,
`control_flow`, and `data_flow`. `ask_code` is available when an LLM endpoint is
configured. Existing tool names remain unchanged. Every tool has a Pydantic structured
output schema. Locations contain exact symbol/callsite IDs, project-relative POSIX
paths, and one-based line ranges. Query length, build count, result/evidence count,
graph depth/fanout, packed context, source size, and answer steps all have hard
server-side limits. A tool may select only a subset of the operator-enabled build
scope; omitting `builds` returns that scope and labels multi-build results as a union.
For MCP call edges, interpret `certainty`, `confidence`, and `target_set_complete`
with the evidence semantics above; in particular, confidence is ranking-only and
an incomplete target set must not be treated as exhaustive.

For Codex or another stdio-capable MCP client, add a local server using the generic
command/environment shape below. The exact settings file or UI varies by client:

```json
{
  "mcpServers": {
    "cpp-context": {
      "command": "/absolute/path/to/cpp-context-mcp",
      "env": {
        "CPP_CONTEXT_PROJECT_ROOT": "/path/to/project",
        "CPP_CONTEXT_DATABASE": "/path/to/project/.cpp-context/index.db",
        "CPP_CONTEXT_COMPILE_COMMANDS": "/path/to/project/build/compile_commands.json"
      }
    }
  }
}
```

Streamable HTTP is optional and binds to localhost by default:

```bash
cpp-context mcp --transport streamable-http --host 127.0.0.1 --port 8765
# MCP endpoint: http://127.0.0.1:8765/mcp
```

The built-in Streamable HTTP mode does not add application authentication. Keep it on
localhost unless a trusted deployment adds TLS, authentication, authorization, and
network controls. Binding a non-loopback address emits a warning.

### External data disclosure

Local feature-hash embeddings and all graph/source tools are network-free. If
`CPP_CONTEXT_EMBEDDING_PROVIDER=openai` is configured, `index_project` sends bounded
symbol text and `search_code` sends its query to that OpenAI-compatible endpoint. If
an LLM is configured, `ask_code` sends the question and selected source excerpts to
that endpoint. Do not enable hosted providers for code that their data-handling terms
do not permit you to transmit. Provider secrets are read only from the environment;
there are no secret CLI flags, and sanitized MCP errors omit provider messages and
absolute host paths.

## Configuration

CLI path/provider flags override these environment variables. Secrets have no CLI
flag so they do not accidentally appear in shell history or process listings.

| Variable | Purpose | Default |
| --- | --- | --- |
| `CPP_CONTEXT_PROJECT_ROOT` | project root | current directory |
| `CPP_CONTEXT_INDEX_DIRECTORY` | local index directory | `<project>/.cpp-context` |
| `CPP_CONTEXT_DATABASE` | SQLite file | `<index-directory>/index.db` |
| `CPP_CONTEXT_COMPILE_COMMANDS` | compilation database | `<project>/build/compile_commands.json` |
| `CPP_CONTEXT_BUILDS` | comma-separated named databases (`NAME=PATH`) | unset |
| `CPP_CONTEXT_BUILD_SCOPE` | comma-separated query/MCP build names | `default` |
| `CPP_CONTEXT_INDEX_PROFILE` | indexing profile: `full` or opt-in `navigation` | `full` |
| `LIBCLANG_LIBRARY_FILE` | exact compatible native libclang | auto-discovered |
| `CPP_CONTEXT_CLANG_ANALYZER` | Clang-18 LibTooling companion executable | unset (baseline mode) |
| `CPP_CONTEXT_ANALYZER_TIMEOUT` | per-companion wall timeout in seconds | `75` |
| `CPP_CONTEXT_ANALYZER_MAX_INPUT_BYTES` | maximum JSONL request bytes | `1048576` |
| `CPP_CONTEXT_ANALYZER_MAX_OUTPUT_BYTES` | maximum compressed/plain wire bytes | `67108864` |
| `CPP_CONTEXT_ANALYZER_MAX_DECODED_BYTES` | maximum decoded protocol bytes | `268435456` |
| `CPP_CONTEXT_ANALYZER_MAX_RECORD_BYTES` | maximum decoded JSONL record bytes | `16777216` |
| `CPP_CONTEXT_ANALYZER_MAX_STDERR_BYTES` | maximum diagnostic bytes | `262144` |
| `CPP_CONTEXT_ANALYZER_MAX_WORKERS` | maximum concurrent companion processes | `1` |
| `CPP_CONTEXT_ANALYZER_MAX_SPOOL_REGISTRIES` | maximum analyzed/converting/queued TUs | `2 × workers` |
| `CPP_CONTEXT_ANALYZER_MAX_SPOOL_BYTES` | maximum compact fact-spool bytes | `registries × decoded bytes` |
| `CPP_CONTEXT_ANALYZER_MAX_SPOOL_FILES` | maximum fact-spool file descriptors | `registries × 64` |
| `CPP_CONTEXT_ANALYZER_MAX_DOMAIN_BATCHES` | maximum converted TU batches awaiting SQLite | `2` |
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
| `CPP_CONTEXT_SERVE_HOST` / `CPP_CONTEXT_SERVE_PORT` | HTTP API/MCP bind address | `127.0.0.1` / `8000` |

## Development

```bash
python -m pip install -e '.[all,dev]'
ruff format --check .
ruff check .
pytest
python -m compileall -q src tests
```

For quick iteration, run the focused Python tests without either real Clang
backend. The native command retains companion semantics, transport, cancellation,
timeout, malformed-stream, and determinism coverage. Run the full command before
merging changes that cross those boundaries:

```bash
# Fast focused tests
pytest -m 'not native and not clang'

# Native companion integration tests
pytest -m native

# Complete local suite (Clang 18, no skipped tests expected)
TMPDIR=/tmp LIBCLANG_PATH=/usr/lib/llvm-18/lib pytest
```

The complete local suite has a hard 3:00-minute acceptance limit. Run it once in
an isolated environment; exceeding 180 seconds is a test-performance failure.

Immutable native fixture facts are content-keyed and reused only within one test
session. Analyzer/protocol, fixture and compilation-database contents, relevant
environment and client transport/limits form the identity. Process-lifetime and
determinism checks deliberately launch fresh companions.
The local timing method and current Ryzen 5 1600/WSL measurements are recorded in
[Test-suite performance](docs/testing.md).

Without installing, use `PYTHONPATH=src python -m cpp_context_engine --help`.
Pytest capture can be disabled with `pytest -s` on environments whose temporary
directory is managed by an external cleanup process.

The deterministic 100-translation-unit benchmark is local-only and is not part of
GitHub Actions. Its generated project, SQLite index, and JSON report are disposable
artifacts. See [Local benchmark methodology](docs/benchmarks/README.md) for the
smoke command, budgets, recorded measurements, and the optional unexecuted
1000-translation-unit profile.

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
- `mcp`: an optional official-SDK v2 server with project-bound, structured,
  bounded tools over stdio or Streamable HTTP

Shared immutable domain types live in `models.py`. Application paths and
environment-driven settings live in `config.py`. Python `Protocol` interfaces
keep implementations replaceable and tests easy to isolate.

Without `CPP_CONTEXT_CLANG_ANALYZER`, indexing continues through the portable
libclang CIndex baseline and marks `advanced_facts_complete=false`. Configuring a
companion makes protocol/Clang/capability mismatches hard pre-index failures and
forces baseline translation units to be reindexed. The companion preserves named
build provenance and emits the existing symbol/occurrence/direct-call/include/
inheritance/override facts plus separate macro spelling and expansion spans,
template-instantiation metadata, and stable lambda call-operator metadata. CFGs
use separate schema-v5 records rather than fake symbols and retain branch,
switch, loop, jump, return, and supported exception flow. Direct, virtual,
devirtualized, lambda/functor, template, and macro-generated calls use separate
schema-v6 callsite/target evidence with explicit certainty and completeness.
Schema v7 adds bounded intraprocedural definitions, uses, reaching definitions,
overwrites, access paths, alias evidence, and local function/member-pointer target
sets. Schema v8 adds body-variant function summaries and bounded interprocedural
argument, return, reference/pointer writeback, field, and global-effect evidence.
The deterministic solver is isolated per named build, handles recursive call-graph
components with hard limits, keeps possible dispatch possible, and marks unknown,
external, or budget-limited paths incomplete. Incremental updates recompute only
changed summaries and their transitive callers plus required callees. It emits
evidence only: dead-code and redundant-call judgments remain the LLM's
responsibility. Shared bounded contracts expose these facts consistently through
the store/service layer, stable CLI JSON, HTTP, and MCP. Certain call and flow
evidence ranks before possible evidence without discarding the possible paths.

See [Compiler-aware indexing](docs/indexing.md) for the ingestion/storage API,
incremental behavior, libclang configuration, and current guarantees. Analysis
claims and open-world limits are summarized in the
[soundness and completeness matrix](docs/soundness-and-completeness.md).

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
reason, build label, and graph path. Call-derived expansion reasons also include
compact certainty, confidence-ranking, and derivation labels. Adapter failures are
returned as sanitized diagnostics;
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
    analysis_service=analysis_service,
)
```

The routes are:

- `GET /health`
- `GET /v1/builds` for safe build labels and active scope
- `POST /v1/context` for source context and provenance
- `POST /v1/answer` for a bounded answer with validated citations
- `POST /v1/calls` for evidence-rich caller/callee targets
- `POST /v1/cfg` for bounded control-flow graphs
- `POST /v1/flow` for bounded local and interprocedural data-flow evidence

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
- One MCP process serves one configured project. It serializes indexing and shared
  SQLite access for correctness; long indexing calls therefore temporarily queue
  search and graph calls.
