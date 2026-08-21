# Local generated-workload benchmark

This benchmark is intentionally local-only. GitHub Actions continues to run the
normal project checks and does not generate the 100×2 or 1000×3 workloads.
Generated C++ sources, indexes, and JSON reports are disposable and ignored by
Git; do not commit them.

The separate [large-TU transport benchmark](large-tu.md) reproduces the KiCad
Clipper case used to verify bounded streaming, compression, and cancellation.

## Reproduction

Build the pinned Clang 18 companion in Release mode, activate an installed project
environment, and run the smoke profile:

```bash
cmake -S native/clang-analyzer -B build/clang-analyzer \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
cmake --build build/clang-analyzer --config Release --parallel 2

cpp-context-benchmark \
  --profile smoke \
  --clang-analyzer build/clang-analyzer/cpp-context-clang-analyzer \
  --compiler clang++-18 \
  --query-iterations 50 \
  --output benchmark-results/smoke.json \
  --enforce-budgets
```

The `smoke` profile deterministically generates 100 translation units, two named
build variants, and 40 functions per translation unit. The source includes direct,
virtual, template, lambda, function-pointer, branch/loop CFG, and local data-flow
facts. Compilation databases use relative paths, and the source hash is independent
of the temporary root. The harness creates the project and index under a fresh
native Linux temporary directory and removes them on exit.

The optional reference profile is reproducible but was not run for this issue:

```bash
cpp-context-benchmark \
  --profile reference \
  --clang-analyzer build/clang-analyzer/cpp-context-clang-analyzer \
  --compiler clang++-18 \
  --query-iterations 50 \
  --output benchmark-results/reference.json
```

It generates 1000 translation units, three builds, and 40 functions per unit. It
must remain opt-in and local; it is not a release or CI gate.

## Method

One invocation records:

1. cold indexing of every TU/build pair;
2. an unchanged warm run, which must index zero TUs and zero embeddings;
3. a shared-header mutation, which must reindex every TU/build pair;
4. peak aggregate RSS sampled from the Python process and all live descendants at
   100 ms intervals, with Linux `getrusage` as an exit-race fallback;
5. SQLite database, WAL, and shared-memory bytes;
6. fact-table counts and analyzer capabilities;
7. 50 nearest-rank p50/p95 samples for single-build retrieval, union retrieval,
   high-fanout incoming calls, union CFG, and union data flow.

Every query uses the public service surface and asserts aggregate result, fanout,
context, graph, and evidence limits. The JSON shape is versioned by
[`report-schema-v1.json`](report-schema-v1.json). The report records hardware, OS,
Python, LLVM/Clang, analyzer protocol, source hash, commit, measurements, budgets,
fact counts, and bounded-union checks. A failed budget remains a failed boolean;
`--enforce-budgets` returns non-zero instead of hiding it.

The v1 smoke budgets are cold ≤180 s, warm ≤10 s, aggregate peak RSS ≤2 GiB,
every query p95 ≤1 s, and database footprint ≤1 GiB. Header reindex time is
recorded but currently has no pass/fail budget.

## Recorded local measurements

The last complete 100×2 report was run on 2026-08-21 in WSL2 on an AMD Ryzen 5
1600 (11 logical CPUs exposed), 16,769,753,088 bytes of memory, Python 3.12.3,
LLVM/Clang 18.1.3, and the Issue #11 worktree based on `1733efa`. It completed
before the final canonical-symbol no-op-write optimization.

| Measurement | Result | Budget | Status |
| --- | ---: | ---: | --- |
| Cold index | 204.316 s | 180 s | fail |
| Warm no-op | 0.550 s | 10 s | pass |
| Shared-header reindex | 267.030 s | recorded only | n/a |
| Peak process-tree RSS | 2,084,192,256 B | 2,147,483,648 B | pass |
| Database footprint | 1,046,192,128 B | 1,073,741,824 B | pass |
| Single-build retrieval p95 | 1.866 s | 1 s | fail |
| Union retrieval p95 | 1.812 s | 1 s | fail |
| High-fanout calls p95 | 0.147 s | 1 s | pass |
| Union CFG p95 | 0.063 s | 1 s | pass |
| Union data flow p95 | 0.075 s | 1 s | pass |

The generated JSON report was deliberately not committed, and the retained handoff
measurements do not include its fact-count table. New complete invocations record
those counts in the schema-defined report.

After the complete run, a separate 100×2 cold phase with the canonical-symbol
`ON CONFLICT ... WHERE` guard completed in 187.733 s. Profiling around that change
showed unnecessary canonical upsert write time falling from about 217 s to 2.6 s.
This was a cold-phase-only observation: warm, header, RSS, database, query, and fact
metrics were not rerun, so it is not presented as a complete post-fix report. The
cold phase still exceeded the 180 s budget.

The implementation also batches symbol/embedding reads and writes, bounds native
TU concurrency, caches validated analyzer paths, groups solver inputs in one pass,
adds foreign-key/canonical lookup indexes, and bulk-deletes TU facts in dependency
order. Focused tests cover these changes; the historical measurements above are
not silently reattributed to them.
