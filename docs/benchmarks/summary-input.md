# Embedding-independent summary benchmark input (#82)

This local-only prerequisite for #47 validates **full32 facts and summaries**, not
an index. The original producer remains FAILED. There is no canary `SUCCESS`, no
embedding-completeness claim, and no relaxation of baseline plus two independent
candidate exact-parity/<30-second replay requirements. Replay integration belongs
to #47; this command never refreshes summaries or invokes the analyzer.

Before a real operation, review and separately authorize the exact input paths,
producer/source/analyzer pins, fresh output directory and resource limits. Small
offline fixtures are the only automated tests. Never open the failed original in
SQLite, even with `mode=ro` or `immutable=1`.

The request JSON contains these absolute paths:

```json
{
  "failed_directory": "/retained/full-32/.gate-32.failed",
  "database": "/retained/full-32/.gate-32.failed/.index.db.fresh-EXACT/index.db",
  "source_compilation_database": "/original/compile_commands.json",
  "producer_engine": "/clean/checkout/of/the/exact/producer/commit"
}
```

The retained directory must contain its unchanged worker specification, phase
timings, selected compilation database and failure marker. All 32 TUs and the
post-TU commit must be complete, with a subsequent embedding-phase transition.
The source CDB must reproduce exactly the original first-32-source workload.

Run from a reviewed engine checkout, for example with an explicitly approved
120-second total envelope (117 seconds of worker time, three for cleanup):

```sh
timeout --signal=INT --kill-after=3s 117s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m cpp_context_engine.summary_input --request /approved/request.json \
  --output-directory /fresh/private/output --seconds 120 \
  --rss-limit-mib 2560 --disk-limit-mib 2048
```

Linux OFD read locks conflict with SQLite's writer byte locks and survive closing
separate hashing FDs. The complete database/journal/WAL/SHM set is copied with the
same basenames and stable identities/hashes. Unexpected files, symlinks, active
writers and WAL databases without already-existing lockable WAL/SHM are rejected.
This intentionally supports the retained fresh-generation rollback-journal case,
not general database salvage. Recovery is ordinary SQLite rollback **on the copy**;
there is no repair, migration, provenance filling or partial-table recovery.

The copy receives full integrity/FK/schema checks, exact command/TU identity and
full-profile coverage checks, analyzer/source/dependency pin checks, all 28 existing
semantic table digests (including whatever embeddings were committed), payload and
solution-hash validation, and deterministic public summaries for up to eight
qualified-name/ID-ordered functions. Truncation and ordering are included in their
public response digests. Local facts are reconciled with propagated payloads using
the existing summary limits; empty propagated results legitimately have no payload.

The supervisor reuses canary limits/process identity cleanup. RSS includes the
supervisor; swap must remain zero; disk includes journals and open unlinked SQLite
temporary files. Only `summary-input.json`, published after clean worker exit and
guard completion, identifies a validated input. `candidate.json` and any partial
files are not acceptance artifacts. The immutable source-file hashes and new
copied-database artifact hash are separate; source and copy need not have the same
physical hash after normal transaction rollback. Retain all failure evidence.
