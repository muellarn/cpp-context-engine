from __future__ import annotations

from dataclasses import replace

import pytest
from test_return_origin_propagation import _fingerprint, _inputs

from cpp_context_engine.analysis import interprocedural
from cpp_context_engine.models import (
    CallArgumentBinding,
    DataFlowCertainty,
    InterproceduralFlowKind,
    MemoryLocationKind,
    SummaryEffectKind,
)


def _writeback_inputs(location):
    inputs = list(_inputs())
    inputs[0] = tuple(
        replace(
            item,
            parameter_modes=("reference", "pointer", "value", "reference"),
            parameter_location_ids=("p0", "p1", "p2", "p3"),
        )
        if item.id == "leaf-a"
        else item
        for item in inputs[0]
    )
    source = next(item for item in inputs[1] if item.id == "leaf-effect")
    source = replace(
        source,
        location_kind=MemoryLocationKind.PARAMETER,
        parameter_index=0,
        location_id=location,
    )
    inputs[1] = tuple(
        replace(
            source,
            id=f"m-{index}",
            source_access_id=f"access-{index}",
            access_path=(f"member-{index}",),
            certainty=DataFlowCertainty.POSSIBLE if index == 3 else DataFlowCertainty.CERTAIN,
        )
        for index in range(4)
    ) + (
        replace(source, id="n-other-parameter", source_access_id="other", parameter_index=1),
        replace(source, id="z-read", source_access_id="read", kind=SummaryEffectKind.READ),
        replace(source, id="z-value", source_access_id="value", parameter_index=2),
        replace(source, id="z-unbound", source_access_id="unbound", parameter_index=3),
    )
    inputs[3] = tuple(
        CallArgumentBinding(
            id=f"binding-{site}-{index}",
            caller_summary_id="caller",
            callsite_id=site,
            argument_index=index,
            location_id=f"caller-storage-{index}",
            location_kind=MemoryLocationKind.GLOBAL,
            parameter_index=None,
            access_path=(),
            writeback_candidate=True,
            complete=True,
            translation_unit_id="unit",
            build_configuration_id="config",
        )
        for site in ("first", "later")
        for index in (0, 1)
    )
    return inputs


@pytest.mark.parametrize("location", ("shared-storage", None))
def test_writeback_constructs_only_last_eligible_location_winners(monkeypatch, location):
    created = []
    original = interprocedural._flow

    def counted(*args, **kwargs):
        flow = original(*args, **kwargs)
        if flow.kind == InterproceduralFlowKind.WRITEBACK:
            created.append(flow)
        return flow

    monkeypatch.setattr(interprocedural, "_flow", counted)
    solution = interprocedural.solve_interprocedural(*_writeback_inputs(location))
    fingerprint = _fingerprint(solution)
    assert len(created) == 4, fingerprint
    assert [(item.callsite_id, item.argument_index) for item in created] == [
        ("first", 0),
        ("first", 1),
        ("later", 0),
        ("later", 1),
    ]
    assert all(item.callee_location_id == location for item in created)
    assert all(
        item.certainty
        == (DataFlowCertainty.POSSIBLE if item.argument_index == 0 else DataFlowCertainty.CERTAIN)
        for item in created
    )
    assert tuple(
        item for item in solution.flows if item.kind == InterproceduralFlowKind.WRITEBACK
    ) == tuple(sorted(created, key=lambda item: item.id))
    # Complete models, solution hashes and encoded payloads from main c1f5468.
    assert fingerprint == (
        "e717f9df40ad69d4116a2376fd7b72276180a5d3efd4d5437cc86018135968f5"
        if location is None
        else "b0e3c5a92cd90f3b7b54fae035031a7d5dc9eeff136988ec189ae06f5b4029fb"
    )


def test_writeback_projection_respects_binding_eligibility():
    inputs = _writeback_inputs(None)
    inputs[3] = tuple(
        replace(item, writeback_candidate=False) if item.argument_index == 1 else item
        for item in inputs[3]
    )
    solution = interprocedural.solve_interprocedural(*inputs)
    writebacks = tuple(
        item for item in solution.flows if item.kind == InterproceduralFlowKind.WRITEBACK
    )
    assert len(writebacks) == 2
    assert all(item.argument_index == 0 for item in writebacks)


def test_writeback_projection_keeps_parameter_mode_validation():
    inputs = _writeback_inputs(None)
    inputs[1] += (replace(inputs[1][0], id="zz-invalid", parameter_index=4),)
    with pytest.raises(IndexError):
        interprocedural.solve_interprocedural(*inputs)


def test_writeback_projection_is_scoped_to_each_build():
    inputs = _writeback_inputs(None)
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


def test_writeback_projection_preserves_cancellation(monkeypatch):
    created = False
    original = interprocedural._flow

    def mark_created(*args, **kwargs):
        nonlocal created
        flow = original(*args, **kwargs)
        if flow.kind == InterproceduralFlowKind.WRITEBACK:
            created = True
        return flow

    def check_cancelled():
        if created:
            raise InterruptedError("cancelled after writeback construction")

    monkeypatch.setattr(interprocedural, "_flow", mark_created)
    with pytest.raises(InterruptedError, match="cancelled after writeback construction"):
        interprocedural.solve_interprocedural(
            *_writeback_inputs(None), check_cancelled=check_cancelled
        )


@pytest.mark.parametrize("cancel_after,expected_calls", [(1, 256), (257, 512)])
def test_writeback_winner_construction_polls_every_256(monkeypatch, cancel_after, expected_calls):
    inputs = _writeback_inputs(None)
    inputs[1] = tuple(
        replace(
            inputs[1][0],
            id=f"effect-{index:03d}",
            source_access_id=f"access-{index:03d}",
            location_id=f"location-{index:03d}",
        )
        for index in range(600)
    )
    calls = 0
    original = interprocedural._flow

    def counted(*args, **kwargs):
        nonlocal calls
        flow = original(*args, **kwargs)
        if flow.kind == InterproceduralFlowKind.WRITEBACK:
            calls += 1
        return flow

    def check_cancelled():
        if calls >= cancel_after:
            raise InterruptedError("cancelled while constructing writeback winners")

    monkeypatch.setattr(interprocedural, "_flow", counted)
    with pytest.raises(InterruptedError, match="constructing writeback winners"):
        interprocedural.solve_interprocedural(*inputs, check_cancelled=check_cancelled)
    assert calls == expected_calls
