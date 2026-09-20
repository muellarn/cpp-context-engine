from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from test_cfg_spool_replay import _builder, _fixture

from cpp_context_engine.ingestion import native


def _definitions():
    facts = _fixture()
    graph = facts[0]["key"]
    block = graph + ":block:0"
    analysis = "data-flow:" + graph
    location = analysis + ":memory:local:value"
    facts.extend(
        [
            {"fact": "data_flow_analysis_v1", "key": analysis, "graph_key": graph},
            {
                "fact": "memory_location_v1",
                "key": location,
                "identity_key": location,
                "analysis_key": analysis,
                "graph_key": graph,
            },
            {
                "fact": "data_access_v1",
                "key": block + ":access:0",
                "block_key": block,
                "sequence": 0,
                "analysis_key": analysis,
                "graph_key": graph,
                "location_key": location,
            },
        ]
    )
    return facts


def _compact(fact):
    result = dict(fact)
    for field, family in native._WIRE_REFERENCE_FIELDS.items():
        if field in result:
            result[field] = native._wire_key(result[field], family)
    family = native._WIRE_FACT_KEYS.get(result["fact"])
    if family and "key" in result:
        result["key"] = native._wire_key(result["key"], family)
    return result


def test_registered_wire_identities_are_hashed_once_without_changing_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = _definitions()
    wire = [_compact(fact) for fact in legacy]
    calls = Counter()
    original = native._wire_key

    def counted(identity, family):
        calls[family, identity] += 1
        return original(identity, family)

    monkeypatch.setattr(native, "_wire_key", counted)
    builder = _builder(tmp_path)
    builder._prepare_wire_keys(tuple(reversed(wire)))
    prepared = calls.copy()
    assert sum(prepared.values()) == 13
    assert Counter(family for family, _ in prepared) == dict(g=2, b=4, e=4, d=1, m=1, a=1)
    restored = [builder._restore_wire_fact(fact) for fact in wire]
    assert restored == [
        {key: value for key, value in fact.items() if key != "identity_key"} for fact in legacy
    ]
    assert calls - prepared == Counter()


def test_prepared_identity_map_rejects_mutation(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    wire = [_compact(fact) for fact in _definitions()]
    builder._prepare_wire_keys(wire)
    with pytest.raises(TypeError):
        builder.wire_identities["g", wire[0]["key"]] = "changed"


@pytest.mark.parametrize("family", ["v", "s"])
def test_unregistered_evidence_and_effect_keys_remain_hashed_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    definitions = _definitions()
    analysis, location, access = [fact["key"] for fact in definitions[-3:]]
    if family == "v":
        identity = f"{analysis}:evidence:may_alias:{location}:{location}"
        fact = {
            "fact": "data_flow_evidence_v1",
            "key": identity,
            "analysis_key": analysis,
            "relation": "may_alias",
            "source_location_key": location,
            "target_location_key": location,
        }
        changed_field = "relation"
        changed_value = "must_alias"
    else:
        summary = "summary:" + analysis
        identity = f"{summary}:effect:read:{access}"
        fact = {
            "fact": "summary_effect_v1",
            "key": identity,
            "summary_key": summary,
            "kind": "read",
            "source_access_key": access,
        }
        changed_field = "kind"
        changed_value = "write"
    wire = _compact(fact)
    builder = _builder(tmp_path)
    builder._prepare_wire_keys([_compact(item) for item in definitions])
    calls = []
    original = native._wire_key

    def counted(identity, family):
        calls.append((identity, family))
        return original(identity, family)

    monkeypatch.setattr(native, "_wire_key", counted)
    assert builder._restore_wire_fact(wire) == fact
    assert calls == [(identity, family)]
    with pytest.raises(native.AnalyzerProtocolError, match="compact fact identity is inconsistent"):
        builder._restore_wire_fact(dict(wire, **{changed_field: changed_value}))
    assert len(calls) == 2
    assert calls[-1] != calls[0]


@pytest.mark.parametrize("field", ["key", "graph_key"])
def test_restore_rejects_unknown_registered_alias(tmp_path: Path, field: str) -> None:
    wire = [_compact(fact) for fact in _definitions()]
    builder = _builder(tmp_path)
    builder._prepare_wire_keys(wire)
    block = next(fact for fact in wire if fact["fact"] == "cfg_block_v1")
    family = "b" if field == "key" else "g"
    with pytest.raises(native.AnalyzerProtocolError, match="references an unknown identity"):
        builder._restore_wire_fact(dict(block, **{field: family + ":" + "0" * 64}))


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("cfg_graph_v1", "function_key", 42),
        ("cfg_block_v1", "index", -1),
    ],
)
def test_prepare_keeps_malformed_definition_errors(tmp_path: Path, kind, field, value) -> None:
    wire = [_compact(fact) for fact in _definitions()]
    next(fact for fact in wire if fact["fact"] == kind)[field] = value
    with pytest.raises(native.AnalyzerProtocolError) as failure:
        _builder(tmp_path)._prepare_wire_keys(wire)
    assert str(failure.value) == "analyzer record has invalid " + field
