# Local large-translation-unit transport benchmark

This benchmark is a local acceptance workload for Issue #20. It is not part of
GitHub Actions and does not modify or build KiCad. The measured source is KiCad
commit `c6135c62abfef364a2e046565cf510573c8962c5`, file
`thirdparty/clipper2/Clipper2Lib/src/clipper.engine.cpp`, analyzed by the Release
Clang 18 companion with:

```text
-DUSINGZ -Ithirdparty/clipper2/Clipper2Lib/include -O3 -DNDEBUG
-std=gnu++17 -Wall -Wextra -Wpedantic
```

The project root is the KiCad checkout and the working directory is that root.
The complete local KiCad compilation database can also be used, but these pinned
arguments reproduce the canonical 112,218-fact baseline independently of a
particular configured KiCad build.

## Recorded acceptance run

The run was recorded on 2026-08-21 in WSL2 with Clang 18.1.3. Canonical JSON
objects exclude `hello`, `begin`, and `complete` and are compared after sorting
keys and records.

| Measurement | Result | Acceptance | Status |
| --- | ---: | ---: | --- |
| Canonical facts | 112,218 | matches baseline | pass |
| Old/new canonical multiset | identical | identical | pass |
| Decoded protocol bytes | 141,302,930 B | recorded | n/a |
| gzip level-1 wire bytes | 5,467,734 B | ≤8 MiB | pass |
| Native peak RSS | 246,648 KiB | ≤768 MiB | pass |
| Python client peak RSS | 165,044,224 B | ≤512 MiB | pass |
| Process-tree peak RSS | 269,774,848 B | ≤1.25 GiB | pass |
| First fact | 12.296 s | before completion | pass |
| Companion completion | 46.044 s | ≤75 s | pass |
| Full client/domain/solver run | 51.908 s | recorded | n/a |
| Real cancellation cleanup | 0.129 s | ≤2 s | pass |

For comparison, the previous eager sink produced the same 112,218 facts and
141,302,914 plain bytes in 44.25 seconds, but retained 2,504,344 KiB peak RSS and
did not emit facts until the final flush. The new fact multiset and stable IDs are
unchanged; only record order and transport framing differ.

Native RSS was measured with `/usr/bin/time -v`. The process-tree measurement
sampled the Python client and all live descendants from `/proc` every 20 ms while
running the complete ingestion and interprocedural solver. First-fact timing used
an incremental gzip decoder on the companion pipe. Cancellation set the same
event used by multi-TU orchestration and verified that the analysis thread joined
and no companion remained.

Temporary raw output and index artifacts belong under a disposable native Linux
temporary directory and must not be committed. Limit-failure, malformed-stream,
plain/gzip equivalence, cleanup, and worker-bound behavior are covered by small
deterministic tests in the normal suite; this KiCad workload remains local-only.
