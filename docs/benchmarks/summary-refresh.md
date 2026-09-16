# Real KiCad summary refresh (issue #47)

This local benchmark depends on the issue #29 canary harness (PR #56) in the
baseline and candidate revisions; merge #56 and rebase #47 before running it.
First retain a successful **full-profile** real KiCad
`gate-32/index.db`, its `SUCCESS` marker and the parent `report.json`. Navigation
databases and synthetic graphs are not acceptance workloads. The report pins the
32 source TUs, compiler commands, KiCad commit, native analyzer binary, coverage,
database artifact and canonical semantic hashes. Include fmt/template cases in
the retained corpus and the existing focused semantic regressions.

Use the same committed driver, Python environment, input report and quiet machine
for all three commands below. `BASELINE` and `CANDIDATE` are clean worktrees on
the main revision containing #29 and the #47 revision respectively. `CANARY` is
the retained full canary output directory. `RUNS` must have space for three copies
of the input database plus WAL/spool headroom; use native Linux storage. No Clang
process is started by the refresh driver.

```bash
PYTHONPATH="$BASELINE/src" timeout --kill-after=5s 180s "$PYTHON" \
  "$CANDIDATE/tools/benchmark_summary_refresh.py" \
  --input-report "$CANARY/report.json" --output-directory "$RUNS/baseline" \
  --max-refresh-seconds 90
PYTHONPATH="$CANDIDATE/src" timeout --kill-after=5s 90s "$PYTHON" \
  "$CANDIDATE/tools/benchmark_summary_refresh.py" \
  --input-report "$CANARY/report.json" --output-directory "$RUNS/trial-1" \
  --baseline-report "$RUNS/baseline/report.json"
PYTHONPATH="$CANDIDATE/src" timeout --kill-after=5s 90s "$PYTHON" \
  "$CANDIDATE/tools/benchmark_summary_refresh.py" \
  --input-report "$CANARY/report.json" --output-directory "$RUNS/trial-2" \
  --baseline-report "$RUNS/baseline/report.json"
```

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
