# Real KiCad summary refresh (issue #47)

This local benchmark depends on the issue #29 canary harness in the baseline and
candidate revisions. Its original input is a successful **full-profile** real KiCad
`gate-32/index.db`, its `SUCCESS` marker and the parent `report.json`. Navigation
databases and synthetic graphs are not acceptance workloads. The report pins the
32 source TUs, compiler commands, KiCad commit, native analyzer binary, coverage,
database artifact and canonical semantic hashes. Include fmt/template cases in
the retained corpus and the existing focused semantic regressions.

Alternatively, use the distinct `summary-input.json` and adjacent `index.db`
produced by the [independent full32 summary-input validator](summary-input.md)
(#82). That manifest must have completed guard/cleanup evidence, exact producer
pins, all 28 semantic table digests and available deterministic public summaries.
The replay driver rejects incomplete manifests and `candidate.json`; it never
opens the manifest's failed `source_database` or reuses that original inode.
This path does not require or create canary `SUCCESS`, complete embeddings or a
successful whole index. Existing canary reports still require `SUCCESS`.

Use the same committed driver, Python environment, input report and quiet machine
for all three commands below. `BASELINE` and `CANDIDATE` are clean worktrees on
the main revision containing #29 and the #47 revision respectively. `CANARY` is
the retained full canary output directory. Run trials sequentially: `RUNS` needs
space for one active copy plus WAL/spool headroom; use native Linux storage.
Before the next trial, preserve the complete small report (all table digests and
public answers), guard evidence and logs. Verify the completed trial's own database
path/inode is distinct from the input, then remove only that derived database.
Keep reports, the baseline report and the original/validated input unchanged.
No Clang process is started by the refresh driver.

For a validated summary input, replace each `--input-report` value below with its
`summary-input.json`. Use the same input for baseline and both fresh candidate
copies. External supervision must still bound the complete invocation, including
untimed copying, validation, RSS, swap, visible files, anonymous temporary files
and cleanup; the refresh alarm is not a replacement for those guards.

Four flushed `summary-replay:` start markers carry monotonic timestamps: `copy`
(including manifest and source-artifact checks), `validation`, `refresh`, and
`aftercheck`. A supervisor may report successive marker differences; an unfinished
last interval has no duration. Only `measurement.seconds` is the isolated committed
refresh timer; startup, copying, checks and cleanup must not be attributed to it.

```bash
PYTHONPATH="$BASELINE/src" timeout --signal=INT --kill-after=5s 175s "$PYTHON" \
  "$CANDIDATE/tools/benchmark_summary_refresh.py" \
  --input-report "$CANARY/report.json" --output-directory "$RUNS/baseline" \
  --max-refresh-seconds 90
# Secure baseline evidence and clean only its verified own copy before continuing.
PYTHONPATH="$CANDIDATE/src" timeout --signal=INT --kill-after=5s 175s "$PYTHON" \
  "$CANDIDATE/tools/benchmark_summary_refresh.py" \
  --input-report "$CANARY/report.json" --output-directory "$RUNS/trial-1" \
  --baseline-report "$RUNS/baseline/report.json"
# Secure trial-1 evidence and clean only its verified own copy before continuing.
PYTHONPATH="$CANDIDATE/src" timeout --signal=INT --kill-after=5s 175s "$PYTHON" \
  "$CANDIDATE/tools/benchmark_summary_refresh.py" \
  --input-report "$CANARY/report.json" --output-directory "$RUNS/trial-2" \
  --baseline-report "$RUNS/baseline/report.json"
```

The outer allowance is 175 seconds for work plus five seconds for cleanup,
180 seconds hard, including copying and before/after validation. It does not
change the candidate's strict internal 30-second refresh limit. Stop on failure;
do not automatically retry or advance to another trial.

Each invocation copies the pinned input using SQLite backup. Input validation,
copying and exact hashes are outside the timer. The timer includes loading solver
inputs, solving, payload persistence and transaction commit, with a hard 30-second
alarm for candidate runs. Timeout rolls back the transaction. Two fresh candidate
runs must each finish **strictly below 30 seconds** and exactly match both the
input and baseline SQL facts and compressed payload bytes. The shared semantic
snapshot covers effects, origins, flows, truncation/completeness, solution hashes,
JSON array ordering and unchanged call/build facts. The driver also compares the
public data-flow service's summary/effect/origin/flow order for each pinned query.
Baseline runs are labelled explicitly and capped at 90 seconds; they are not
candidate acceptance results. Candidate runs require a baseline report and never
accept a refresh budget above 30 seconds.

Reports record the loaded engine revision, analyzer provenance, elapsed refresh
time, database footprint, integrity and process peak RSS (including untimed copy
and hash validation). Do not count copy/hash time as a solver regression or claim
a synthetic run as the acceptance result. Existing focused regression tests
remain the evidence for incremental closure, multiple builds and cancellation;
this benchmark stays out of CI.
