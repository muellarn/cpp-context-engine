from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from cpp_context_engine.ingestion import native
from cpp_context_engine.models import BuildConfiguration


def _fixture():
    facts = []
    for name in ("first", "second"):
        function = "usr:" + name * 20
        graph = "cfg:" + function
        facts.append(
            {
                "fact": "cfg_graph_v1",
                "key": graph,
                "function_key": function,
                "entry_block_key": graph + ":block:0",
                "normal_exit_block_key": graph + ":block:1",
                "fact_schema_version": 1,
                "clang_major": 18,
                "build_options": {},
            }
        )
        for index in (0, 1):
            block = f"{graph}:block:{index}"
            facts.append(
                {
                    "fact": "cfg_block_v1",
                    "key": block,
                    "graph_key": graph,
                    "index": index,
                    "role": "entry" if index == 0 else "normal_exit",
                    "reachable": True,
                }
            )
            facts.append(
                {
                    "fact": "cfg_element_v1",
                    "key": block + ":element:0",
                    "graph_key": graph,
                    "block_key": block,
                    "index": 0,
                    "kind": "statement",
                    "text": name,
                    "metadata": {"checked": True},
                }
            )
        facts.append(
            {
                "fact": "cfg_edge_v1",
                "graph_key": graph,
                "source_block_key": graph + ":block:0",
                "target_block_key": graph + ":block:1",
                "successor_index": 0,
                "kind": "fallthrough",
                "feasible": True,
            }
        )
    return facts


def _builder(root: Path):
    configuration = BuildConfiguration("cfg-replay", root / "fixture.cpp", root, (), "test")
    builder = native._FactBatchBuilder(root, configuration)
    builder.keys.update({"usr:" + name * 20: "symbol_" + name for name in ("first", "second")})
    return builder


@pytest.mark.parametrize("compact", [False, True])
def test_cfg_element_spool_reads_preserve_exact_models_and_forward_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compact: bool
) -> None:
    facts = _fixture()
    if compact:
        for fact in facts:
            for field, family in native._WIRE_REFERENCE_FIELDS.items():
                if field in fact:
                    fact[field] = native._wire_key(fact[field], family)
            family = native._WIRE_FACT_KEYS.get(fact["fact"])
            if family and "key" in fact:
                fact["key"] = native._wire_key(fact["key"], family)
    decoded = Counter()
    original_loads = native.marshal.loads

    def counted_loads(payload):
        record = original_loads(payload)
        decoded[record["fact"]] += 1
        return record

    monkeypatch.setattr(native.marshal, "loads", counted_loads)
    with native._FactRegistry() as registry:
        # Dependents precede their definitions in the supplied stream.
        for fact in reversed(facts):
            registry.add(fact)
        builder = _builder(tmp_path)
        builder._prepare_wire_keys(registry)
        result = builder._cfg_facts(registry)
    assert tuple(map(len, result)) == (2, 4, 4, 2)
    # Golden covers every model field, identity and tuple order, before this change.
    assert hashlib.sha256(repr(result).encode()).hexdigest() == (
        "1b04b9345d5a975fd6bd754c284881ffc31ad2137555a79b76f1e6169ea2146c"
    )
    assert decoded["cfg_element_v1"] == 4 * 3
    assert decoded["cfg_edge_v1"] == 2 * 2


@pytest.mark.parametrize(
    "other_error,expected",
    [
        (None, "analyzer CFG facts have inconsistent graph references"),
        ("endpoint", "analyzer CFG exceptional exit key is invalid"),
        ("later_element", "analyzer record has invalid index"),
        ("edge", "analyzer CFG facts have inconsistent graph references"),
        ("graph_model", "analyzer CFG facts have inconsistent graph references"),
    ],
)
def test_cross_graph_membership_keeps_diagnostic_precedence(
    tmp_path: Path, other_error: str | None, expected: str
) -> None:
    facts = _fixture()
    graphs = [fact for fact in facts if fact["fact"] == "cfg_graph_v1"]
    elements = [fact for fact in facts if fact["fact"] == "cfg_element_v1"]
    # A later valid element must not overwrite the earlier mismatch indicator.
    elements[0]["graph_key"] = graphs[1]["key"]
    if other_error == "endpoint":
        graphs[0]["exceptional_exit_block_key"] = 42
    elif other_error == "later_element":
        elements[-1]["index"] = -1
    elif other_error == "edge":
        next(fact for fact in facts if fact["fact"] == "cfg_edge_v1")["source_block_key"] = 42
    elif other_error == "graph_model":
        graphs[0]["fact_schema_version"] = 2
    with native._FactRegistry() as registry:
        for fact in facts:
            registry.add(fact)
        builder = _builder(tmp_path)
        builder._prepare_wire_keys(registry)
        with pytest.raises(native.AnalyzerProtocolError) as failure:
            builder._cfg_facts(registry)
    assert str(failure.value) == expected
