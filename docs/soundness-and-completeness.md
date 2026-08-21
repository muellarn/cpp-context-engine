# Soundness and completeness

The engine stores bounded compiler evidence. It does not prove whole-program
behavior, absence of calls, dead code, safety, or equivalence. Conclusions are
valid only for the indexed compilation-database entries, selected build variants,
the pinned Clang model, and the limits reported with each result.

## Evidence vocabulary

- `certainty=certain` means Clang selected the target for that indexed callsite and
  build. It does not prove the call executes at runtime.
- `certainty=possible` retains a supported target that static analysis could not
  uniquely prove. Possible evidence must not be discarded merely because certain
  evidence also exists elsewhere.
- `confidence` is a deterministic ordering hint derived from the evidence form. It
  is not a measured or Bayesian probability.
- `target_set_complete=true` means the analyzer proved a closed set for that one
  callsite and build. `false` plus `unresolved_reason` means additional targets may
  exist. Missing, baseline, legacy, or truncated evidence is not a complete set.
- Data-flow and summary `complete` flags apply only to the implemented model.
  `incomplete_reasons`, binding completeness, query `truncated`, and selected build
  scope remain part of every interpretation.

## Matrix

| Construct or boundary | Evidence retained | Completeness behavior | Unsupported conclusion |
| --- | --- | --- | --- |
| Dynamic loading (`dlopen`, `LoadLibrary`, symbol lookup) | Ordinary calls, strings, and explicit references visible in indexed source may be searchable. | Loaded modules and symbol-name resolution are not enumerated; affected target sets remain open unless independently proven closed. | No indexed target means no runtime target. |
| External binaries and libraries | Declarations and calls visible to the translation unit remain; project-local source locations are retained. | System/library bodies are excluded. External or unindexed callees make summaries incomplete and unknown side effects are conservative clobbers. | The external call is pure, safe, or has no callback/writeback effect. |
| Inline assembly | The statement remains in CFG/source evidence. | Data flow adds an unknown clobber and `inline_assembly` incomplete reason. Instruction semantics are not decoded. | Values, aliases, or reachability across the assembly are fully known. |
| Undefined behavior | Pre-UB syntax, CFG, calls, and data-flow evidence may still be recorded. | The engine does not detect or model all undefined behavior and cannot define executions after UB. | Evidence proves the program has defined behavior or a unique runtime outcome. |
| Pointer arithmetic and unknown indexing | Known bases and surrounding accesses may remain. | Unknown offsets collapse to unknown locations with `pointer_arithmetic` or `pointer_arithmetic_or_unknown_index`; alias and summary completeness are reduced. | Two accesses cannot alias, or a write cannot reach an omitted location. |
| Unions and overlapping storage | Field/access-path syntax is retained. | Union storage is conservative and reports `union_storage`; active-member and byte-level overlap are not fully reconstructed. | Field-distinct paths are disjoint or the active member is proven. |
| Registries, callbacks, and dependency injection | Explicit function/member-pointer assignments and visible registration calls may produce targets and graph edges. | Runtime keys, constructor order, reflection, generated registration, external mutation, and later replacement can leave target sets incomplete. | A registry lookup has only the statically visible targets. |
| `setjmp`/`longjmp` and other non-local transfer | Calls and the ordinary Clang CFG around them are retained. | No dedicated cross-function non-local-jump edge or restored-state model is provided. Treat reachability and data flow across such calls as incomplete even when no specialized reason is emitted. | Ordinary successor edges exhaust all control transfers. |
| Unknown or semantic wrappers | A wrapper with an indexed body is analyzed like other functions; visible argument, return, and writeback evidence can propagate. | External, generated, intrinsic, reflection-like, variadic, or unsupported wrappers may reduce binding/summary completeness and introduce unknown effects. | A wrapper preserves identity, ownership, nullness, or purity unless evidence says so. |
| Build variants, feature flags, targets, and platforms | Facts retain build/configuration/TU provenance; single-build queries filter strictly and union queries label each item. | Only explicitly indexed compilation databases are represented. A union combines evidence but does not invent evidence for unindexed variants. | A result from one build applies to every build, or union absence proves global absence. |

## Query and scope limits

Public retrieval, call, CFG, and data-flow queries enforce aggregate result and
evidence budgets across a selected union. `truncated=true`, exhausted graph
budgets, packed-context limits, or a subset of configured builds make negative
answers incomplete. The libclang baseline sets `advanced_facts_complete=false`
and does not emit the companion's CFG, callsite, points-to, or summary evidence.

The safe workflow is to select the exact build scope, inspect provenance and all
completeness fields, retain possible evidence, and phrase negative conclusions as
bounded observations rather than whole-program proofs.
