# Test-suite performance

The suite separates quick Python checks from real Clang integration without
removing assertions:

```bash
pytest -m 'not native and not clang'
pytest -m native
TMPDIR=/tmp LIBCLANG_PATH=/usr/lib/llvm-18/lib pytest
```

The complete suite is accepted only when one isolated run finishes in at most
3:00 minutes with zero skips. A duration above 180 seconds is a failure, not a
reason to relabel a smaller subset as the full suite.

Native tests use a session-local fixture cache. Its key includes the analyzer
binary and protocol, the complete fixture and compilation-database contents,
the normalized build configuration, relevant environment, resource limits, and
transport mode. Immutable projects stored on the WSL-mounted Windows filesystem
are copied to `/tmp` for analysis; fact paths are restored before validation.
Cached artifacts are read-only, deserialized for each consumer, and removed at
session end.

Determinism tests still perform two independent analyses. Cancellation, timeout,
process-group cleanup, malformed streams, output limits, incomplete streams,
stdio MCP, and mutable incremental-index fixtures do not reuse semantic results.

Every native test module discovers the companion through one shared helper. Set
`CPP_CONTEXT_TEST_ANALYZER` to select an explicit binary; otherwise the helper
uses `build/clang-analyzer/cpp-context-clang-analyzer`. The selected path must be
a regular executable file whose handshake advertises protocol version 5, Clang
18, and all required capabilities. A missing, non-executable, or incompatible
binary is a suite-configuration error rather than a test skip. This keeps a
misconfigured native run visible and guarantees that a successful run has zero
skips.

## Ryzen 5 1600 / WSL measurement

Measured on 2026-08-28 with Clang 18, a Release companion, `TMPDIR=/tmp`, no
`PYTHONPATH`, and a fresh editable `[all,dev]` environment. Runs were performed
alone on the host.

Before fixture relocation, a controlled native profile was stopped after
174.88 seconds with only four tests complete; the initial data-flow fixture took
60.80 seconds. This reproduced the issue's earlier 12–22 minute full-suite range
and identified `/mnt/c` fixture analysis as the remaining bottleneck.

After optimization:

| Run | Result | Pytest time | Wall time | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Native integration | 65 passed | 33.47 s | 37.66 s | 187,104 KiB |
| Full suite 1 | 179 passed | 53.20 s | 60.44 s | 207,340 KiB |
| Full suite 2 | 179 passed | 52.54 s | 59.74 s | 206,072 KiB |

All three runs had zero skips. Both full runs began with an empty session cache;
after each run there were no cache directories, analyzer processes, or inherited
30-second process-test descendants left behind.
