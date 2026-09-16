# KiCad navigation canary

This local-only harness is the fail-fast acceptance path for Issue #29. It uses
an existing compilation database and never builds KiCad. Large runs do not run
in CI.

## Input preflight

Preflight validates every CDB entry, including source-file existence, and records
raw entries, normalized configurations, canonical unique translation units and
the CDB SHA-256 without starting an analyzer or creating an index. Set
`KICAD_SOURCE` and `KICAD_BUILD` to the prepared source and build directories:

```bash
cpp-context-kicad-canary \
  --project-root "$KICAD_SOURCE" \
  --compile-commands "$KICAD_BUILD/compile_commands.json" \
  --preflight-only
```

Out-of-source CMake builds legitimately contain generated translation units
outside the source root. Preflight classifies these as `generated_build_source`
or `external_source`; it does not reject them. Numeric 1/4/16/32 gates select
only canonical, unique source-root TUs in original CDB order so a generated or
duplicate prefix cannot silently change the canary. The later `all` gate retains
the exact full raw CDB, including generated entries and duplicate rows. Its
`raw_cdb_entries` report field counts those rows, while `translation_units` and
progress use the loader's normalized unique compiler configurations.

Classification does not authorize access. Before an `all` run, repeat
`--generated-source-root` for the narrow generated directories that contain
every out-of-tree CDB source. The harness canonicalizes and validates those
operator-provided directories during preflight, then binds them only to the
`all` gate's `default` build variant. Missing coverage fails before an analyzer
starts. Numeric gates need no generated-root authorization and never inherit it.
Setting the project root or a generated root to `/` is unsafe and unsupported.

To validate the complete gate's source boundary without starting an analyzer:

```bash
cpp-context-kicad-canary \
  --project-root "$KICAD_SOURCE" \
  --compile-commands "$KICAD_BUILD/compile_commands.json" \
  --generated-source-root "$KICAD_GENERATED_ROOT" \
  --gates all \
  --preflight-only
```

## Progressive navigation gates

The supervised canary is Linux-only because its hard resource limits and
identity-safe process-tree cleanup require `/proc`. Unsupported platforms fail
before the analyzer starts. Use a fresh output directory and set
`CLANG_ANALYZER` and `CANARY_OUTPUT` to paths owned by this run:

```bash
cpp-context-kicad-canary \
  --project-root "$KICAD_SOURCE" \
  --compile-commands "$KICAD_BUILD/compile_commands.json" \
  --clang-analyzer "$CLANG_ANALYZER" \
  --output-directory "$CANARY_OUTPUT/navigation-a" \
  --gates 1,4,16,32 \
  --gate-timeouts 1:60,4:90,16:120,32:150 \
  --total-gate-timeouts 1:90,4:120,16:150,32:180 \
  --workers 8 \
  --query compareVersionStrings
```

The 32-TU index/worker acceptance limit remains 150 seconds. Independent
verification runs in a separate process under the same resource supervisor,
using only the remaining end-to-end budget. Total defaults are 90/120/150/180
seconds for 1/4/16/32 TUs and 5,400 seconds for `all`; these are deadlines, not
forecast factors. `elapsed_seconds` retains worker/index time;
`validation_elapsed_seconds` and `total_elapsed_seconds` report the separate
verification cost and overall gate duration. Custom worker timeouts need a
deliberate `--total-gate-timeouts` selection too.

Each supervised stage also retains `phase-timings-index.json` or
`phase-timings-validation.json`, including on a failed gate. The worker timestamps
boundaries with the same host's monotonic clock; supervisor pipe-delivery delays
are not charged to phases. The fixed, non-overlapping phases are:

- `tu_processing`: index setup and TU processing until **all** selected TUs have
  been staged, not merely until the most recent successful TU;
- `post_tu_finalization`: remaining generator checks/cleanup, global index
  finalization and commit, ending when embedding work starts;
- `embeddings`: embedding work and private-generation close/publication;
- `producer_checks`: ranking, canonical snapshots, provenance and integrity checks;
- `validation`: the separate independent validator, including its publication checks.

Complete phases have start/end offsets from the shared gate start and a duration.
Not-started and incomplete phases have a null duration, never a fabricated zero;
`observed_seconds` for an incomplete phase is only the interval through its last
confirmed worker event, not a completed measurement. Startup, inter-process gaps,
and supervisor cleanup are outside these phase intervals; the existing total
elapsed time remains authoritative and is not added to the phase sum. Analyzer
slot times overlap TU processing and must not be added either.

Snapshots are atomically retained at transitions, at most once per five seconds
between transitions, and during supervisor cleanup. They retain configured input
pins/budgets, resource peaks and available existing counts. Embedding counts are
processed variant records, not unique provider computations; incomplete counts
remain null. Successful reports include `phase_measurements` and
`validation_phase_measurements`. Historical reports are not rewritten. These are
measurement artifacts, not forecasts: the unknown-total-ETA guard stays fail-closed
until representative navigation samples and tail-scaling evidence support a
separately reviewed calibration method.

Index-stage measurements additionally retain `post_tu_operations` for the exact
`restore_deferred_indexes` and `refresh_summaries` calls. Their worker-side start
and completion events use the same monotonic clock and atomic snapshots. An
exception or cancellation leaves the current operation incomplete (null end and
duration); a later operation remains not started. Index restoration is omitted
for a non-fresh generation and remains `not_started`, not a fabricated success.

These intervals are **inside** `post_tu_finalization`, never added to its duration.
`post_tu_unattributed_seconds` accounts for the remaining phase time only once the
whole phase is complete; it stays null after an interruption. Relationship updates,
other finalization work and commit are not silently attributed to either call.
Completed operations mean that their calls returned, not that the transaction
committed or publication succeeded. Observer errors preserve transaction rollback.
These additive fields do not change historical reports, budgets or forecasts.

The smaller limits are strict discovery guardrails. Each new database records:

- each staged TU, phase, elapsed time and TU-only ETA (not a total forecast);
- peak process-tree RSS and swap, active SQLite/WAL/SHM bytes and total gate
  disk bytes;
- input/subset CDB digests, selected raw indices, engine/project commits, native
  analyzer digest/protocol/capabilities, and SQLite integrity;
- stable semantic table counts/digest, exact ranked query IDs and scores, and
  order-sensitive public calls/CFG/data-flow result digests.

For numeric navigation samples, swap, RSS above 2.5 GiB, active database above
550 MiB, artifact-directory output above 1 GiB,
hard gate timeout, or ten seconds with no TU/DB/CPU progress terminates the whole
worker process tree. A gate contains a `.running` marker until every check has
passed; failures are renamed `.failed` and only success writes `SUCCESS`. The
source CDB digest is rechecked before publication. Baseline provenance, exact
gate membership, semantic facts, public result ordering and analyzer identity
are also checked before `SUCCESS` is written.

RSS/swap supervision includes the coordinator, producer and independent
validator. A failed validator leaves no `SUCCESS`. Physical hashes still guard
same-artifact verification/publication, not cross-run parity of volatile bytes.
`artifact_directory_peak_bytes` (also retained as `peak_disk_bytes`) covers
the gate directory including private staging/journals. Native `/tmp` spools
are separate: `native_spool_budget_bytes` reports their existing aggregate hard
allowance, 4 GiB at eight workers. Directory usage is not total machine disk use.

Run the same gates a second time against the first report. Semantic rows, IDs,
coverage fields, embeddings and search rankings must match exactly:

```bash
cpp-context-kicad-canary \
  --project-root "$KICAD_SOURCE" \
  --compile-commands "$KICAD_BUILD/compile_commands.json" \
  --clang-analyzer "$CLANG_ANALYZER" \
  --output-directory "$CANARY_OUTPUT/navigation-b" \
  --gates 1,4,16,32 \
  --gate-timeouts 1:60,4:90,16:120,32:150 \
  --total-gate-timeouts 1:90,4:120,16:150,32:180 \
  --workers 8 \
  --query compareVersionStrings \
  --baseline-report "$CANARY_OUTPUT/navigation-a/report.json"
```

Only after progressive/parity and representative main-source/generated-source
checks pass may a complete run be considered. `all` requires explicit artifact
budgets. The fixed initial ceiling is 48 GiB active DB / 64 GiB gate directory,
plus the separate unchanged spool allowance, on the prepared host with about
749 GiB free. These are ceilings, not size forecasts; never raise them mid-run.
Numeric bench32 limits are unchanged.

The prepared CDB covers 2,252 compiler configurations / 2,194 canonical sources
(2,153 source-root and 41 generated), not every possible KiCad build. Bind
`/home/arno/git/kicad` as `KICAD_SOURCE` and
`/home/arno/.local/state/cpp-context-engine/kicad-c6135c6` as `KICAD_BUILD` and
`KICAD_GENERATED_ROOT`, then repeat complete preflight before the full command:

```bash
cpp-context-kicad-canary \
  --project-root "$KICAD_SOURCE" \
  --compile-commands "$KICAD_BUILD/compile_commands.json" \
  --generated-source-root "$KICAD_GENERATED_ROOT" \
  --clang-analyzer "$CLANG_ANALYZER" \
  --output-directory "$CANARY_OUTPUT/navigation-full" \
  --gates all \
  --gate-timeouts all:5400 \
  --total-gate-timeouts all:5400 \
  --workers 8 \
  --rss-limit-mib 2560 \
  --database-limit-mib 49152 \
  --disk-limit-mib 65536 \
  --query compareVersionStrings
```

At ten minutes a projection above the 90-minute hard limit terminates the run.
At thirty minutes a projection above the 60-minute target also terminates it.
TU throughput is only a lower bound: unmeasured embeddings/verification do not
become zero after the last TU. If total remaining work cannot be estimated at a
decision checkpoint, the run fails closed with an unknown-projection reason.
This means missing forecast evidence, not proven slowness. Until there is a
calibrated tail estimate, a full run crossing the ten-minute checkpoint remains
no-go. Do not invent multipliers or extrapolate the third-party-heavy prefix32
as a reliable whole-project forecast.

Bounded real-KiCad deep materialization, cache/restart, invalidation,
cancellation, and multi-build checks are a separate gate dependent on Issue #45.
They must not be inferred from a navigation report and are intentionally not
executed by this harness yet.

After Issue #45 is merged, an exclusive full-profile bench32 correctness gate
may be run when no other benchmark is using the host:

```bash
cpp-context-kicad-canary \
  --project-root "$KICAD_SOURCE" \
  --compile-commands "$KICAD_BUILD/compile_commands.json" \
  --clang-analyzer "$CLANG_ANALYZER" \
  --output-directory "$CANARY_OUTPUT/full-32" \
  --profile full \
  --gates 32 \
  --gate-timeouts 32:360 \
  --total-gate-timeouts 32:420 \
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
