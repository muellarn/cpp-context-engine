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

The index timing file also contains `ingestion_pipeline.configurations`, keyed by
the selected compilation-database index. Its overlapping per-TU intervals distinguish
`native_transport_registry` (native execution, transport and registry ingestion),
`conversion` (Python batch conversion and registry cleanup), and `consumer_staging`
(including the existing validation/counting wrappers, **not** pure SQLite time).
Worker timestamps are retained even when delivery is delayed. Missing observations
are `unknown`; open intervals are `incomplete` with null end/duration. A completed
interval's `outcome` can still be failed/cancelled and does not imply gate success.
These bounded observations share the existing five-second atomic snapshot cadence,
global-phase flushes and final cleanup flush, not a write per TU event. If the
supervisor is killed abruptly, only its last atomic snapshot is available; no final
write or completion event is guaranteed. Existing success and resource gates remain
unchanged.

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
  --runtime-calibration "$CANARY_OUTPUT/runtime-calibration.json" \
  --gates all \
  --gate-timeouts all:5400 \
  --total-gate-timeouts all:5400 \
  --workers 8 \
  --rss-limit-mib 8192 \
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

### Empirical whole-runtime calibration (#100)

`--runtime-calibration` is supported only for one complete navigation `all` gate.
It consumes one successful fixed risk census H, two increasing nested fit samples
from the remainder R, and one disjoint R holdout. It uses their existing reports
and retained CDB/worker-spec files, not their databases. No complete real H/R
calibration has yet been accepted; a failed H with successful TU staging is still
ineligible because its remaining phases and validation did not complete. The
implementation and offline tests alone do not authorize a whole-KiCad run.

The JSON input has `schema: "cpp-context-runtime-calibration-v2"`, the unchanged
whole `source_cdb_sha256`, one descriptor in `risk`, two descriptors in `fits`,
and one in `holdout`. The old unpublished v1 contract is rejected, not silently
reinterpreted as unbiased evidence.
Each descriptor contains `report` (relative to the input file), its `sha256`,
and ordered `raw_indices` into the original whole CDB. Copy those original
objects unchanged into each sample's CDB and run the existing `all` gate with
the same workers, queries, generated roots and embedding dimensions. Keep the
sample report's local `selected_raw_indices` unchanged at `0..N-1`; only the
bundle records the different whole-CDB provenance indices. No report rewriting
or new gate-selection mode is needed. Keep the
published `SUCCESS`, report, subset CDB and worker specification. Failed gates,
partial phases, omitted configurations, deep facts or resource violations do not
qualify. Engine/analyzer/schema/project and exact compile commands must match;
only documentation-only Git differences are compatible with measured evidence.
Changes to producer, canary or model code require a new compatible evidence set.

The existing 16-configuration risk set H is fixed at original indices
`9,16,110,168,862,864,1266,1366,1771,1998,2026,2034,2093,2133,2155,2240`.
It covers known difficult regions; it is deliberately not a representative fit
sample. This contract is limited to the pinned 2252-configuration workload.
`calibration_cohorts(2252, source_cdb_sha256)` returns H and the predetermined R
indices: one uniform pseudorandom draw of 47 configurations without replacement
from the remaining 2236, using the CDB digest as the fixed seed. Its first 16 and
first 32 entries define nested fits; its final 15 define the disjoint holdout.
Each cohort is sorted into original CDB order before writing the unchanged
objects. No coverage forcing, reseeding, favorable rerolls, or overlap with H is
allowed; the loader independently regenerates and checks the exact selections.
Source-group risk coverage stays in H. A uniform small sample is still empirical,
not proof of cost representativeness. Prefix NAV32 is not an R sample.

For R, the fixed empirical estimate uses the largest measured phase cost per work unit:
TU pipeline wall time per configuration; post-TU time per indexed symbol,
occurrence, edge, callsite and call target; embedding time per processed record; separate
producer/validator time per peak database byte. Overlapping TU intervals are
never added. R work amounts use the largest measured amount per R configuration
times 2236. Add H's measured TU wall time, work amounts and nonphase overhead exactly
once; never multiply its forced heavy TU sample by the whole-project population.
For each tail phase, apply the highest observed H/R seconds-per-work-unit rate
to the combined H-plus-projected-R workload. This retains H's expensive tail
behavior rather than assuming separate finalization times simply add.
Positive time with zero work units retains a positive floor; the largest measured
R nonphase startup/gap/cleanup cost is also retained. The independent R holdout
must satisfy every R amount, phase and total estimate without borrowing H's budget.
There is no guessed safety multiplier or mathematical completion guarantee.
This is an empirical composition, not proof that nonlinear finalization
on their union is bounded by the measured rates. Existing live work/phase envelopes must
reject deviations. The earlier real failures remain attached to their original
code/data pins and cannot supply any of these four successful reports.

A valid estimate over 90 minutes rejects before starting. After validating H and
R16, the loader already rejects if `H_TU + (R16_TU / 16) * 2236 > 5400`; adding the
second max-rate fit or nonnegative tail costs cannot rescue that model. This is
an early mathematical no-go for the proposed model, not a claimed lower bound on
actual runtime. R32/holdout reports are not read after that rejection. Missing
later evidence never authorizes a Whole run.

During the run, all
unstarted tail phases retain their estimated costs; completed phases contribute
their actual time. Exceeding a measured work/phase envelope invalidates the
estimate, including at the final validator result. Zero TU progress still leaves
the checkpoint unknown. The additional observed `spent * remaining / staged`
Whole-pipeline rest estimate remains unchanged and is maximized with the
calibrated rest. It is a mixed H/R throughput check, not an R-only rate. A heavy
early prefix can still conservatively cause NO-GO; no guessed H time or sum of
overlapping Native/Conversion intervals is subtracted to hide that limitation.
The existing 10-/30-minute rejection policies, hard
90-minute deadline, swap prohibition and resource/atomicity checks remain in
force. `empirical_projected_total_seconds` is retained in the existing timing
artifact; it is not a promise that an unobserved larger workload will finish.

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
