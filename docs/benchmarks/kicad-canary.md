# KiCad navigation canary

This local-only harness is the fail-fast acceptance path for Issue #29. It uses
an existing compilation database and never builds KiCad. Large runs do not run
in CI.

## Input preflight

Preflight validates every CDB entry and records its SHA-256 without starting an
analyzer or creating an index:

```bash
cpp-context-kicad-canary \
  --project-root /home/arno/git/kicad \
  --compile-commands /tmp/cpp-context-kicad-cdb.V6Xvt5/compile_commands.json \
  --preflight-only
```

Out-of-source CMake builds legitimately contain generated translation units
outside the source root. Preflight classifies these as `generated_build_source`
or `external_source`; it does not reject them. Numeric 1/4/16/32 gates select
only real source-root TUs in original CDB order so a generated prefix cannot
silently change the canary. The later `all` gate retains the exact full CDB,
including generated entries.

The current analyzer confines searchable facts to `project_root`. It can process
an out-of-tree generated TU and retain facts from project-local headers, but the
generated source's own definitions are not searchable. That separate product
limitation must be fixed with explicitly authorized generated-source roots or
CDB-owned paths; setting the project root to `/` is unsafe and unsupported.

## Progressive navigation gates

Use a fresh output directory and the analyzer built from the exact revision:

```bash
cpp-context-kicad-canary \
  --project-root /home/arno/git/kicad \
  --compile-commands /tmp/cpp-context-kicad-cdb.V6Xvt5/compile_commands.json \
  --clang-analyzer /absolute/path/to/cpp-context-clang-analyzer \
  --output-directory /tmp/cpp-context-kicad-nav-a \
  --gates 1,4,16,32 \
  --gate-timeouts 1:60,4:90,16:120,32:150 \
  --workers 8 \
  --query compareVersionStrings
```

The 32-TU acceptance limit is 150 seconds. The smaller limits are deliberately
strict discovery guardrails. Every gate uses a new database and records:

- each staged TU, elapsed time, rate and ETA;
- peak process-tree RSS and swap, active SQLite/WAL/SHM bytes and total gate
  disk bytes;
- input/subset CDB digests, selected raw indices, engine/project commits, native
  analyzer digest/protocol/capabilities, and SQLite integrity;
- stable semantic table counts/digest plus exact ranked query IDs and scores.

Any swap, RSS above 2.5 GiB, active database above 550 MiB, output above 1 GiB,
hard gate timeout, or ten seconds with no TU/DB/CPU progress terminates the whole
worker process group. A gate contains a `.running` marker until every check has
passed; failures are renamed `.failed` and only success writes `SUCCESS`. The
source CDB digest is rechecked before and after the run.

Run the same gates a second time against the first report. Semantic rows, IDs,
coverage fields, embeddings and search rankings must match exactly:

```bash
cpp-context-kicad-canary \
  --project-root /home/arno/git/kicad \
  --compile-commands /tmp/cpp-context-kicad-cdb.V6Xvt5/compile_commands.json \
  --clang-analyzer /absolute/path/to/cpp-context-clang-analyzer \
  --output-directory /tmp/cpp-context-kicad-nav-b \
  --gates 1,4,16,32 \
  --gate-timeouts 1:60,4:90,16:120,32:150 \
  --workers 8 \
  --query compareVersionStrings \
  --baseline-report /tmp/cpp-context-kicad-nav-a/report.json
```

Only after both progressive runs pass may the complete navigation run start:

```bash
cpp-context-kicad-canary \
  --project-root /home/arno/git/kicad \
  --compile-commands /tmp/cpp-context-kicad-cdb.V6Xvt5/compile_commands.json \
  --clang-analyzer /absolute/path/to/cpp-context-clang-analyzer \
  --output-directory /tmp/cpp-context-kicad-nav-full \
  --gates all \
  --gate-timeouts all:5400 \
  --workers 8 \
  --query compareVersionStrings
```

At ten minutes a projection above the 90-minute hard limit terminates the run.
At thirty minutes a projection above the 60-minute target also terminates it.

Bounded real-KiCad deep materialization, cache/restart, invalidation,
cancellation, and multi-build checks are a separate gate dependent on Issue #45.
They must not be inferred from a navigation report and are intentionally not
executed by this harness yet.

After Issue #45 is merged, an exclusive full-profile bench32 correctness gate
may be run when no other benchmark is using the host:

```bash
cpp-context-kicad-canary \
  --project-root /home/arno/git/kicad \
  --compile-commands /tmp/cpp-context-kicad-cdb.V6Xvt5/compile_commands.json \
  --clang-analyzer /absolute/path/to/cpp-context-clang-analyzer \
  --output-directory /tmp/cpp-context-kicad-full-32 \
  --profile full \
  --gates 32 \
  --gate-timeouts 32:360 \
  --workers 8 \
  --database-limit-mib 1350 \
  --disk-limit-mib 2048 \
  --query compareVersionStrings
```

This uses the same pinned 32 project-source entries as the navigation gate and
retains its complete SQLite database. The report requires
`analysis_backend=clang-libtooling`, `advanced_facts_complete=1`, all independent
deep coverage flags, and emits separate deterministic hashes for summaries,
effects, return origins, interprocedural flows, and compact solution payloads.
It is not a prerequisite for the navigation-first closure gate and must not run
concurrently with Issue #47 measurements.
