"""Deterministic bounded function-summary propagation over compiler call evidence."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, replace

from cpp_context_engine.models import (
    CallArgumentBinding,
    CallResultBinding,
    CallSite,
    CallTarget,
    CallTargetCertainty,
    DataFlowCertainty,
    FunctionSummary,
    InterproceduralFlow,
    InterproceduralFlowKind,
    MemoryLocationKind,
    SummaryEffect,
    SummaryEffectKind,
    SummaryReturnOrigin,
    SummaryReturnOriginKind,
)


@dataclass(frozen=True, slots=True)
class InterproceduralLimits:
    max_scc_iterations: int = 32
    max_scc_size: int = 128
    max_summary_effects: int = 1024

    def __post_init__(self) -> None:
        if min(self.max_scc_iterations, self.max_scc_size, self.max_summary_effects) <= 0:
            raise ValueError("interprocedural limits must be positive")


@dataclass(frozen=True, slots=True)
class InterproceduralSolution:
    summaries: tuple[FunctionSummary, ...]
    effects: tuple[SummaryEffect, ...]
    return_origins: tuple[SummaryReturnOrigin, ...]
    flows: tuple[InterproceduralFlow, ...]


def _id(prefix: str, *values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:32]}"


def _certainty(target: CallTargetCertainty, evidence: DataFlowCertainty) -> DataFlowCertainty:
    if target == CallTargetCertainty.CERTAIN and evidence == DataFlowCertainty.CERTAIN:
        return DataFlowCertainty.CERTAIN
    return DataFlowCertainty.POSSIBLE


def _tarjan(nodes: tuple[str, ...], adjacency: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indexes: dict[str, int] = {}
    lows: dict[str, int] = {}
    result: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = lows[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(adjacency.get(node, ())):
            if target not in indexes:
                visit(target)
                lows[node] = min(lows[node], lows[target])
            elif target in on_stack:
                lows[node] = min(lows[node], indexes[target])
        if lows[node] != indexes[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        result.append(tuple(sorted(component)))

    for node in sorted(nodes):
        if node not in indexes:
            visit(node)
    return tuple(result)


def solve_interprocedural(
    summaries: tuple[FunctionSummary, ...],
    local_effects: tuple[SummaryEffect, ...],
    local_return_origins: tuple[SummaryReturnOrigin, ...],
    argument_bindings: tuple[CallArgumentBinding, ...],
    result_bindings: tuple[CallResultBinding, ...],
    callsites: tuple[CallSite, ...],
    call_targets: tuple[CallTarget, ...],
    *,
    limits: InterproceduralLimits | None = None,
) -> InterproceduralSolution:
    """Solve per-build summaries; an omitted fact always reduces completeness."""

    selected_limits = limits or InterproceduralLimits()
    solved_summaries: list[FunctionSummary] = []
    solved_effects: list[SummaryEffect] = []
    solved_origins: list[SummaryReturnOrigin] = []
    solved_flows: list[InterproceduralFlow] = []
    variants = sorted({item.build_variant for item in summaries})
    for variant in variants:
        variant_summaries = tuple(item for item in summaries if item.build_variant == variant)
        solution = _solve_variant(
            variant_summaries,
            tuple(item for item in local_effects if item.build_variant == variant),
            tuple(item for item in local_return_origins if item.build_variant == variant),
            tuple(item for item in argument_bindings if item.build_variant == variant),
            tuple(item for item in result_bindings if item.build_variant == variant),
            tuple(item for item in callsites if item.build_variant == variant),
            tuple(item for item in call_targets if item.build_variant == variant),
            selected_limits,
        )
        solved_summaries.extend(solution.summaries)
        solved_effects.extend(solution.effects)
        solved_origins.extend(solution.return_origins)
        solved_flows.extend(solution.flows)
    return InterproceduralSolution(
        tuple(sorted(solved_summaries, key=lambda item: item.id)),
        tuple(sorted(solved_effects, key=lambda item: item.id)),
        tuple(sorted(solved_origins, key=lambda item: item.id)),
        tuple(sorted(solved_flows, key=lambda item: item.id)),
    )


def _solve_variant(
    summaries: tuple[FunctionSummary, ...],
    local_effects: tuple[SummaryEffect, ...],
    local_origins: tuple[SummaryReturnOrigin, ...],
    bindings: tuple[CallArgumentBinding, ...],
    result_bindings: tuple[CallResultBinding, ...],
    callsites: tuple[CallSite, ...],
    targets: tuple[CallTarget, ...],
    limits: InterproceduralLimits,
) -> InterproceduralSolution:
    summary_by_id = {item.id: item for item in summaries}
    summaries_by_function: dict[str, list[FunctionSummary]] = defaultdict(list)
    for summary in summaries:
        summaries_by_function[summary.function_symbol_id].append(summary)
    sites_by_owner: dict[tuple[str, str, str], list[CallSite]] = defaultdict(list)
    for site in callsites:
        sites_by_owner[
            (site.owner_symbol_id, site.translation_unit_id, site.build_configuration_id)
        ].append(site)
    targets_by_site: dict[str, list[CallTarget]] = defaultdict(list)
    for target in targets:
        targets_by_site[target.callsite_id].append(target)
    bindings_by_pair = {
        (item.caller_summary_id, item.callsite_id, item.argument_index): item for item in bindings
    }
    result_by_pair = {(item.caller_summary_id, item.callsite_id): item for item in result_bindings}

    def owner_sites(caller: FunctionSummary) -> tuple[CallSite, ...]:
        # One canonical symbol can have different inline/ODR bodies in several TUs.
        # Mixing their callsites would invent effects that do not occur in this body.
        return tuple(
            sites_by_owner.get(
                (
                    caller.function_symbol_id,
                    caller.translation_unit_id,
                    caller.build_configuration_id,
                ),
                (),
            )
        )

    def callee_bodies(
        caller: FunctionSummary, target: CallTarget
    ) -> tuple[tuple[FunctionSummary, ...], bool]:
        candidates = tuple(
            sorted(summaries_by_function.get(target.target_symbol_id, ()), key=lambda item: item.id)
        )
        same_tu = tuple(
            item
            for item in candidates
            if item.translation_unit_id == caller.translation_unit_id
            and item.build_configuration_id == caller.build_configuration_id
        )
        selected = same_tu or candidates
        return selected, len(selected) > 1

    adjacency: dict[str, set[str]] = defaultdict(set)
    call_edges: dict[str, list[tuple[CallSite, CallTarget, FunctionSummary, bool]]] = defaultdict(
        list
    )
    for caller in summaries:
        for site in owner_sites(caller):
            for target in targets_by_site.get(site.id, ()):
                callees, body_ambiguous = callee_bodies(caller, target)
                for callee in callees:
                    adjacency[caller.id].add(callee.id)
                    call_edges[caller.id].append((site, target, callee, body_ambiguous))
    components = _tarjan(tuple(summary_by_id), adjacency)
    recursive_ids = {
        member for component in components if len(component) > 1 for member in component
    } | {node for node in summary_by_id if node in adjacency.get(node, set())}
    oversized_ids = {
        member
        for component in components
        if len(component) > limits.max_scc_size
        for member in component
    }

    local_effect_map: dict[str, tuple[SummaryEffect, ...]] = defaultdict(tuple)
    local_origin_map: dict[str, tuple[SummaryReturnOrigin, ...]] = defaultdict(tuple)
    for summary in summaries:
        local_effect_map[summary.id] = tuple(
            sorted(
                (item for item in local_effects if item.summary_id == summary.id),
                key=lambda x: x.id,
            )
        )
        local_origin_map[summary.id] = tuple(
            sorted(
                (item for item in local_origins if item.summary_id == summary.id),
                key=lambda x: x.id,
            )
        )

    current_effects = dict(local_effect_map)
    current_origins = dict(local_origin_map)
    current_reasons = {summary.id: set(summary.local_incomplete_reasons) for summary in summaries}
    iteration_counts: dict[str, int] = {}

    def transfer(
        caller: FunctionSummary,
    ) -> tuple[tuple[SummaryEffect, ...], tuple[SummaryReturnOrigin, ...], set[str]]:
        reasons = set(caller.local_incomplete_reasons)
        if caller.id in oversized_ids:
            reasons.add("scc_size_cap_exceeded")
        effects = {item.id: item for item in local_effect_map[caller.id]}
        origins = {item.id: item for item in local_origin_map[caller.id]}
        for site in owner_sites(caller):
            site_targets = targets_by_site.get(site.id, ())
            if not site.target_set_complete or not site_targets:
                reasons.add("unknown_or_external_call_target")
            for target in site_targets:
                callees, body_ambiguous = callee_bodies(caller, target)
                if not callees:
                    reasons.add("external_or_unindexed_callee_body")
                    reasons.add("unknown_or_external_call_target")
                if body_ambiguous:
                    reasons.add("multiple_callee_body_variants")
                for callee in callees:
                    for parameter_index in range(len(callee.parameter_location_ids)):
                        binding = bindings_by_pair.get((caller.id, site.id, parameter_index))
                        # Dropping an unbound parameter would hide downstream effects while
                        # incorrectly leaving the caller summary complete.
                        if binding is None:
                            reasons.add("missing_call_argument_binding")
                        elif not binding.complete:
                            reasons.add("incomplete_call_argument_binding")
                    if current_reasons.get(callee.id):
                        reasons.add("callee_summary_incomplete")
                    for effect in current_effects.get(callee.id, ()):
                        propagated = _propagate_effect(
                            caller,
                            site,
                            target,
                            callee,
                            effect,
                            bindings_by_pair,
                            body_ambiguous=body_ambiguous,
                        )
                        if propagated is not None:
                            effects[propagated.id] = propagated
                    for origin in tuple(origins.values()):
                        if (
                            origin.kind != SummaryReturnOriginKind.CALL_RESULT
                            or origin.callsite_id != site.id
                        ):
                            continue
                        for callee_origin in current_origins.get(callee.id, ()):
                            propagated_origin = _propagate_origin(
                                caller,
                                site,
                                target,
                                callee,
                                callee_origin,
                                bindings_by_pair,
                                body_ambiguous=body_ambiguous,
                            )
                            origins[propagated_origin.id] = propagated_origin
        ordered_effects = tuple(sorted(effects.values(), key=lambda item: item.id))
        if len(ordered_effects) > limits.max_summary_effects:
            ordered_effects = ordered_effects[: limits.max_summary_effects]
            reasons.add("summary_effect_cap_exceeded")
        return ordered_effects, tuple(sorted(origins.values(), key=lambda item: item.id)), reasons

    # Tarjan emits callees before callers for this caller-to-callee graph, so each
    # component consumes stable downstream summaries and iterates only its recursive SCC.
    for component in components:
        members = tuple(summary_by_id[summary_id] for summary_id in component)
        component_converged = False
        iteration_limit = (
            1 if any(item.id in oversized_ids for item in members) else limits.max_scc_iterations
        )
        for iteration_count in range(1, iteration_limit + 1):
            next_values = {caller.id: transfer(caller) for caller in members}
            if all(
                next_values[caller.id]
                == (
                    current_effects[caller.id],
                    current_origins[caller.id],
                    current_reasons[caller.id],
                )
                for caller in members
            ):
                component_converged = True
            for caller in members:
                (
                    current_effects[caller.id],
                    current_origins[caller.id],
                    current_reasons[caller.id],
                ) = next_values[caller.id]
                iteration_counts[caller.id] = iteration_count
            if component_converged:
                break
        if not component_converged and not any(item.id in oversized_ids for item in members):
            for caller in members:
                current_reasons[caller.id].add("scc_iteration_cap_exceeded")

    final_summaries: list[FunctionSummary] = []
    for summary in summaries:
        reasons = tuple(sorted(current_reasons[summary.id]))
        fingerprint = _solution_hash(
            current_effects[summary.id], current_origins[summary.id], reasons
        )
        final_summaries.append(
            replace(
                summary,
                complete=not reasons,
                incomplete_reasons=reasons,
                recursive=summary.id in recursive_ids,
                iteration_count=iteration_counts.get(summary.id, 0),
                max_scc_iterations=limits.max_scc_iterations,
                max_scc_size=limits.max_scc_size,
                max_summary_effects=limits.max_summary_effects,
                solution_hash=fingerprint,
            )
        )

    flows: dict[str, InterproceduralFlow] = {}
    for caller in summaries:
        for site, target, callee, body_ambiguous in call_edges.get(caller.id, ()):
            for index, callee_location in enumerate(callee.parameter_location_ids):
                binding = bindings_by_pair.get((caller.id, site.id, index))
                if binding is None:
                    continue
                flow = _flow(
                    InterproceduralFlowKind.ARGUMENT_TO_PARAMETER,
                    caller,
                    callee,
                    site,
                    target,
                    DataFlowCertainty.CERTAIN
                    if binding.complete and not body_ambiguous
                    else DataFlowCertainty.POSSIBLE,
                    "caller argument is bound to this concrete callee parameter",
                    argument_index=index,
                    caller_location_id=binding.location_id,
                    callee_location_id=callee_location,
                )
                flows[flow.id] = flow
            for effect in current_effects.get(callee.id, ()):
                if effect.kind != SummaryEffectKind.WRITE or effect.parameter_index is None:
                    continue
                mode = callee.parameter_modes[effect.parameter_index]
                if mode not in {"reference", "pointer", "rvalue_reference"}:
                    continue
                binding = bindings_by_pair.get((caller.id, site.id, effect.parameter_index))
                if binding is None or not binding.writeback_candidate:
                    continue
                flow = _flow(
                    InterproceduralFlowKind.WRITEBACK,
                    caller,
                    callee,
                    site,
                    target,
                    effect.certainty
                    if binding.complete and not body_ambiguous
                    else DataFlowCertainty.POSSIBLE,
                    "callee parameter side effect may write caller storage",
                    argument_index=effect.parameter_index,
                    caller_location_id=binding.location_id,
                    callee_location_id=effect.location_id,
                )
                flows[flow.id] = flow
            result = result_by_pair.get((caller.id, site.id))
            if result is not None:
                for origin in current_origins.get(callee.id, ()):
                    flow = _flow(
                        InterproceduralFlowKind.RETURN_TO_CALLER,
                        caller,
                        callee,
                        site,
                        target,
                        origin.certainty if not body_ambiguous else DataFlowCertainty.POSSIBLE,
                        "callee return origin reaches the caller's call-result definition",
                        caller_location_id=result.location_id,
                        callee_location_id=origin.location_id,
                        caller_access_id=result.definition_access_id,
                    )
                    flows[flow.id] = flow
    return InterproceduralSolution(
        tuple(sorted(final_summaries, key=lambda item: item.id)),
        tuple(item for key in sorted(current_effects) for item in current_effects[key]),
        tuple(item for key in sorted(current_origins) for item in current_origins[key]),
        tuple(sorted(flows.values(), key=lambda item: item.id)),
    )


def _propagate_effect(
    caller: FunctionSummary,
    site: CallSite,
    target: CallTarget,
    callee: FunctionSummary,
    effect: SummaryEffect,
    bindings: dict[tuple[str, str, int], CallArgumentBinding],
    *,
    body_ambiguous: bool,
) -> SummaryEffect | None:
    location_kind = effect.location_kind
    parameter_index = effect.parameter_index
    access_path = effect.access_path
    location_id = effect.location_id
    if parameter_index is not None:
        binding = bindings.get((caller.id, site.id, parameter_index))
        if binding is None:
            return None
        location_kind = binding.location_kind
        parameter_index = binding.parameter_index
        access_path = (binding.access_path + effect.access_path)[:8]
        location_id = binding.location_id
    elif location_kind not in {MemoryLocationKind.GLOBAL, MemoryLocationKind.FIELD}:
        return None
    certainty = _certainty(
        target.certainty,
        effect.certainty if not body_ambiguous else DataFlowCertainty.POSSIBLE,
    )
    # Key by the originating local access, not the previous propagated row ID: recursive
    # SCCs otherwise mint a fresh semantic duplicate on every fixed-point iteration.
    return SummaryEffect(
        id=_id(
            "summary_effect",
            caller.id,
            effect.kind,
            location_kind,
            parameter_index,
            access_path,
            site.id,
            target.target_symbol_id,
            effect.source_access_id,
        ),
        summary_id=caller.id,
        kind=effect.kind,
        location_kind=location_kind,
        certainty=certainty,
        reason="callee effect propagated through compiler-resolved call target",
        parameter_index=parameter_index,
        access_path=access_path,
        location_id=location_id,
        source_access_id=effect.source_access_id,
        is_local=False,
        via_callsite_id=site.id,
        target_symbol_id=target.target_symbol_id,
        translation_unit_id=site.translation_unit_id,
        build_configuration_id=site.build_configuration_id,
        build_variant=site.build_variant,
    )


def _propagate_origin(
    caller: FunctionSummary,
    site: CallSite,
    target: CallTarget,
    callee: FunctionSummary,
    origin: SummaryReturnOrigin,
    bindings: dict[tuple[str, str, int], CallArgumentBinding],
    *,
    body_ambiguous: bool,
) -> SummaryReturnOrigin:
    del callee
    location_kind = origin.location_kind
    parameter_index = origin.parameter_index
    access_path = origin.access_path
    location_id = origin.location_id
    if parameter_index is not None:
        binding = bindings.get((caller.id, site.id, parameter_index))
        if binding is None:
            location_kind = MemoryLocationKind.UNKNOWN
            parameter_index = None
            access_path = ()
            location_id = None
        else:
            location_kind = binding.location_kind
            parameter_index = binding.parameter_index
            access_path = (binding.access_path + origin.access_path)[:8]
            location_id = binding.location_id
    certainty = _certainty(
        target.certainty,
        origin.certainty if not body_ambiguous else DataFlowCertainty.POSSIBLE,
    )
    return SummaryReturnOrigin(
        id=_id(
            "summary_return",
            caller.id,
            site.id,
            target.target_symbol_id,
            location_kind,
            parameter_index,
            access_path,
            location_id,
        ),
        summary_id=caller.id,
        kind=SummaryReturnOriginKind.LOCATION
        if location_kind is not None
        else SummaryReturnOriginKind.UNKNOWN,
        certainty=certainty,
        reason="callee return origin propagated through compiler-resolved call target",
        location_kind=location_kind,
        parameter_index=parameter_index,
        access_path=access_path,
        location_id=location_id,
        is_local=False,
        via_callsite_id=site.id,
        target_symbol_id=target.target_symbol_id,
        translation_unit_id=site.translation_unit_id,
        build_configuration_id=site.build_configuration_id,
        build_variant=site.build_variant,
    )


def _flow(
    kind: InterproceduralFlowKind,
    caller: FunctionSummary,
    callee: FunctionSummary,
    site: CallSite,
    target: CallTarget,
    evidence_certainty: DataFlowCertainty,
    reason: str,
    *,
    argument_index: int | None = None,
    caller_location_id: str | None = None,
    callee_location_id: str | None = None,
    caller_access_id: str | None = None,
) -> InterproceduralFlow:
    certainty = _certainty(target.certainty, evidence_certainty)
    return InterproceduralFlow(
        id=_id(
            "interprocedural_flow",
            kind,
            caller.id,
            callee.id,
            site.id,
            target.target_symbol_id,
            argument_index,
            caller_location_id,
            callee_location_id,
            caller_access_id,
        ),
        kind=kind,
        caller_summary_id=caller.id,
        callee_summary_id=callee.id,
        callsite_id=site.id,
        target_symbol_id=target.target_symbol_id,
        target_certainty=target.certainty,
        certainty=certainty,
        reason=reason,
        argument_index=argument_index,
        caller_location_id=caller_location_id,
        callee_location_id=callee_location_id,
        caller_access_id=caller_access_id,
        translation_unit_id=site.translation_unit_id,
        build_configuration_id=site.build_configuration_id,
        build_variant=site.build_variant,
    )


def _solution_hash(
    effects: tuple[SummaryEffect, ...],
    origins: tuple[SummaryReturnOrigin, ...],
    reasons: tuple[str, ...],
) -> str:
    return _id(
        "solution",
        tuple((item.id, item.certainty.value) for item in effects),
        tuple((item.id, item.certainty.value) for item in origins),
        reasons,
    )
