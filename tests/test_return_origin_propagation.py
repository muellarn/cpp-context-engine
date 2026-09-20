from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from cpp_context_engine.analysis import interprocedural
from cpp_context_engine.analysis.interprocedural import InterproceduralLimits
from cpp_context_engine.models import (
    CallDispatchKind,
    CallResultBinding,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    DataFlowCertainty,
    FunctionSummary,
    InterproceduralFlowKind,
    MemoryLocationKind,
    SourceSpan,
    SummaryEffect,
    SummaryEffectKind,
    SummaryReturnOrigin,
    SummaryReturnOriginKind,
)
from cpp_context_engine.storage.sqlite import _encode_summary_payload


def _inputs(*, recursive: bool = False, overwrite: bool = False) -> tuple:
    caller = FunctionSummary(
        id="caller",
        function_symbol_id="caller-function",
        graph_id="caller-graph",
        analysis_id="caller-analysis",
        parameter_modes=(),
        parameter_location_ids=(),
        local_complete=True,
        local_incomplete_reasons=(),
        complete=True,
        incomplete_reasons=(),
        recursive=False,
        iteration_count=0,
        max_scc_iterations=32,
        max_scc_size=128,
        max_summary_effects=1024,
        translation_unit_id="unit",
        build_configuration_id="config",
    )
    leaves = tuple(
        replace(
            caller,
            id=name,
            function_symbol_id=f"{name}-function",
            graph_id=f"{name}-graph",
            analysis_id=f"{name}-analysis",
        )
        for name in ("leaf-a", "leaf-b")
    )
    span = SourceSpan(Path("returns.cpp"), 1, 1)
    sites = tuple(
        CallSite(
            id=name,
            owner_symbol_id=caller.function_symbol_id,
            dispatch_kind=CallDispatchKind.DIRECT,
            spelling_span=span,
            expansion_span=span,
            target_set_complete=True,
            callee_text=name,
            translation_unit_id="unit",
            build_configuration_id="config",
        )
        for name in (("first", "later", "self") if recursive else ("first", "later"))
    )
    targets = tuple(
        CallTarget(
            id=f"target-{site}-{callee.id}",
            callsite_id=site,
            target_symbol_id=callee.function_symbol_id,
            certainty=CallTargetCertainty.CERTAIN,
            confidence=1.0,
            confidence_reason="fixture target",
            derivation="direct",
            evidence_span=span,
            translation_unit_id="unit",
            build_configuration_id="config",
        )
        for site, callee in (
            [("first", leaves[0]), ("first", leaves[1]), ("later", leaves[0])]
            + ([("self", caller)] if recursive else [])
        )
    )
    leaf_origin = SummaryReturnOrigin(
        id="leaf-origin",
        summary_id=leaves[0].id,
        kind=SummaryReturnOriginKind.LOCATION,
        certainty=DataFlowCertainty.CERTAIN,
        reason="local return",
        location_kind=MemoryLocationKind.GLOBAL,
        location_id="global-a",
        translation_unit_id="unit",
        build_configuration_id="config",
    )
    origins = [
        leaf_origin,
        replace(
            leaf_origin,
            id="unknown-origin",
            kind=SummaryReturnOriginKind.UNKNOWN,
            location_kind=None,
            location_id=None,
        ),
        replace(
            leaf_origin,
            id="other-leaf-origin",
            summary_id=leaves[1].id,
            location_id="global-b",
            certainty=DataFlowCertainty.POSSIBLE,
        ),
    ]
    marker = replace(
        leaf_origin,
        id="first-marker",
        summary_id=caller.id,
        kind=SummaryReturnOriginKind.CALL_RESULT,
        location_kind=None,
        location_id=None,
        callsite_id="first",
    )
    later_marker = replace(marker, id="later-marker", callsite_id="later")
    if overwrite:
        # Earlier propagation overwrites the later marker; a static site set is unsafe.
        overwritten_id = interprocedural._propagate_origin(
            caller, sites[0], targets[0], leaves[0], leaf_origin, {}, body_ambiguous=False
        ).id
        later_marker = replace(later_marker, id=overwritten_id)
    origins.extend(
        (
            marker,
            replace(marker, id="duplicate-first-marker"),
            later_marker,
            replace(marker, id="unrelated-marker", callsite_id="absent"),
            replace(
                leaf_origin, id="local-caller-origin", summary_id=caller.id, callsite_id="later"
            ),
            replace(
                leaf_origin,
                id="propagated-caller-origin",
                summary_id=caller.id,
                callsite_id="later",
                is_local=False,
                via_callsite_id="prior",
                target_symbol_id=leaves[0].function_symbol_id,
            ),
        )
    )
    if recursive:
        origins.append(replace(marker, id="self-marker", callsite_id="self"))
    effect = SummaryEffect(
        id="leaf-effect",
        summary_id=leaves[0].id,
        kind=SummaryEffectKind.WRITE,
        location_kind=MemoryLocationKind.GLOBAL,
        certainty=DataFlowCertainty.CERTAIN,
        reason="local write",
        location_id="global-a",
        source_access_id="access",
        translation_unit_id="unit",
        build_configuration_id="config",
    )
    results = tuple(
        CallResultBinding(
            id=f"result-{site.id}",
            caller_summary_id=caller.id,
            callsite_id=site.id,
            location_id=f"result-location-{site.id}",
            definition_access_id=f"result-{site.id}",
            translation_unit_id="unit",
            build_configuration_id="config",
        )
        for site in sites
    )
    return (
        (*leaves, caller),
        (effect, replace(effect, id="caller-effect", summary_id=caller.id)),
        tuple(origins),
        (),
        results,
        sites,
        targets,
    )


def _fingerprint(solution) -> str:
    payloads = []
    for summary in solution.summaries:
        effects = tuple(
            item for item in solution.effects if item.summary_id == summary.id and not item.is_local
        )
        origins = tuple(
            item
            for item in solution.return_origins
            if item.summary_id == summary.id and not item.is_local
        )
        *metadata, payload = _encode_summary_payload(summary.id, effects, origins)
        payloads.append((summary.id, metadata, payload.hex()))
    # Cover every model field, tuple ordering, solution hashes and exact payload bytes.
    raw = json.dumps((asdict(solution), payloads), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize("duplicate_unknown", (False, True))
def test_return_flows_construct_only_last_location_winners(monkeypatch, duplicate_unknown) -> None:
    inputs = list(_inputs())
    source = next(item for item in inputs[2] if item.id == "leaf-origin")
    inputs[2] += tuple(
        replace(
            source,
            id=f"zz-duplicate-location-{index}",
            access_path=(f"field-{index}",),
            certainty=DataFlowCertainty.POSSIBLE if index == 2 else DataFlowCertainty.CERTAIN,
        )
        for index in range(3)
    )
    if duplicate_unknown:
        unknown = next(item for item in inputs[2] if item.id == "unknown-origin")
        inputs[2] += (replace(unknown, id="zz-unknown", certainty=DataFlowCertainty.POSSIBLE),)
    created = []
    original = interprocedural._flow

    def counted(*args, **kwargs):
        flow = original(*args, **kwargs)
        if flow.kind == InterproceduralFlowKind.RETURN_TO_CALLER:
            created.append(flow)
        return flow

    monkeypatch.setattr(interprocedural, "_flow", counted)
    solution = interprocedural.solve_interprocedural(*inputs)
    fingerprint = _fingerprint(solution)
    assert len(created) == 5, fingerprint
    # Keep the first location's position but the last origin's certainty, per edge.
    assert [(item.callsite_id, item.callee_location_id) for item in created] == [
        ("first", "global-a"),
        ("first", None),
        ("first", "global-b"),
        ("later", "global-a"),
        ("later", None),
    ]
    assert all(
        item.certainty == DataFlowCertainty.POSSIBLE
        for item in created
        if item.callee_location_id == "global-a" or duplicate_unknown
    )
    assert solution.flows == tuple(sorted(created, key=lambda item: item.id))
    # Full models, solution hashes and encoded bytes captured from main 5f8cde2.
    assert fingerprint == (
        "c9ae7e87b1a73a4bcc338825a311fbc6986fb9a33af779873599740ab48e7388"
        if duplicate_unknown
        else "ab1ca16792b8c1669ff1c5c3efdde9eacff713215aa2a5d2d43e0e1acfbc7215"
    )


def test_return_flow_projection_stays_within_each_build() -> None:
    inputs = _inputs()
    expected = interprocedural.solve_interprocedural(*inputs)
    both_builds = tuple(
        records + tuple(replace(item, build_variant="alternative") for item in records)
        for records in inputs
    )
    combined = interprocedural.solve_interprocedural(*both_builds)
    for variant in ("default", "alternative"):
        assert tuple(item for item in combined.flows if item.build_variant == variant) == tuple(
            replace(item, build_variant=variant) for item in expected.flows
        )


def test_return_flow_creation_preserves_cancellation(monkeypatch) -> None:
    created = False
    original = interprocedural._flow

    def mark_created(*args, **kwargs):
        nonlocal created
        flow = original(*args, **kwargs)
        created = True
        return flow

    def check_cancelled():
        if created:
            raise InterruptedError("cancelled after return-flow construction")

    monkeypatch.setattr(interprocedural, "_flow", mark_created)
    with pytest.raises(InterruptedError, match="cancelled after return-flow construction"):
        interprocedural.solve_interprocedural(*_inputs(), check_cancelled=check_cancelled)


@pytest.mark.parametrize(
    ("case", "limits", "expected"),
    (
        (
            "acyclic",
            InterproceduralLimits(),
            "9f18b2e67e90786e570b40ae619d00e6694eebd4a599b6733a6ec1c5566733d9",
        ),
        (
            "iteration-cap",
            InterproceduralLimits(max_scc_iterations=1),
            "78a2bff3a4008e8643ce07f1989e50f15ff2428a4bd2f35fbb6aa2776ac79061",
        ),
        (
            "effect-cap",
            InterproceduralLimits(max_summary_effects=1),
            "7820d6e7046f69fb469c32f55cc76c06d2fc86289b53baa4a2857ab440961a5d",
        ),
        (
            "recursive",
            InterproceduralLimits(),
            "bb1eb08c7a30b8a8f69838b9f2b02c9a9711fb9e85ab4be84af3b6d90a88e763",
        ),
        (
            "recursive-cap",
            InterproceduralLimits(max_scc_iterations=1),
            "880533fe806c3c27eb8849fd083e607168f5adfe4e7dc07d304530f2df6d1e90",
        ),
        (
            "overwrite",
            InterproceduralLimits(),
            "dc114c4db5e0a4f3c7fe9ed7af968c9833ad345429055d62bebda229c18ee934",
        ),
    ),
)
def test_return_markers_trigger_one_propagation_per_callsite(
    monkeypatch,
    case: str,
    limits: InterproceduralLimits,
    expected: str,
) -> None:
    inputs = _inputs(recursive=case.startswith("recursive"), overwrite=case == "overwrite")
    original = interprocedural._propagate_origin
    original_effect = interprocedural._propagate_effect
    calls = 0
    transfers = 0

    def counted(caller, site, target, callee, origin, *args, **kwargs):
        nonlocal calls
        if site.id == "first" and origin.id == "leaf-origin":
            calls += 1
        return original(caller, site, target, callee, origin, *args, **kwargs)

    def counted_effect(caller, site, target, callee, effect, *args, **kwargs):
        nonlocal transfers
        if site.id == "first" and effect.id == "leaf-effect":
            transfers += 1
        return original_effect(caller, site, target, callee, effect, *args, **kwargs)

    monkeypatch.setattr(interprocedural, "_propagate_origin", counted)
    monkeypatch.setattr(interprocedural, "_propagate_effect", counted_effect)
    solution = interprocedural.solve_interprocedural(*inputs, limits=limits)
    caller = next(item for item in solution.summaries if item.id == "caller")
    # Goldens captured from main b0c00105 before removing duplicate propagation.
    assert _fingerprint(solution) == expected
    assert calls == transfers > 0
    assert caller.recursive == case.startswith("recursive")
    if case.endswith("cap"):
        reason = (
            "summary_effect_cap_exceeded" if case == "effect-cap" else "scc_iteration_cap_exceeded"
        )
        assert reason in caller.incomplete_reasons
    if case == "overwrite":
        assert not any(item.via_callsite_id == "later" for item in solution.return_origins)
    assert tuple(item.id for item in solution.return_origins) == tuple(
        sorted(item.id for item in solution.return_origins)
    )
