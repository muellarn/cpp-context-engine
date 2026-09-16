from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary
from native_cache import fresh_native_client
from test_native_analyzer import _fake_hello
from wire_identity_fixture import create_wire_identity_fixture

from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.ingestion.native import (
    AnalyzerProtocolError,
    NativeAnalyzerClient,
    _FactBatchBuilder,
    _wire_key,
)

pytestmark = pytest.mark.native


@pytest.mark.parametrize(
    "invalid", ["duplicate", "conflict", "unknown", "wrong_family", "self_short", "self_full"]
)
def test_wire_identity_registration_fails_closed(invalid: str) -> None:
    root = Path(__file__).parent / "fixtures" / "template_dataflow_project"
    config = CompilationDatabase.load(root / "compile_commands.json").configurations[0]
    builder = _FactBatchBuilder(root.resolve(), config)
    function = "usr:" + "long_function_identity_" * 5
    key = _wire_key("cfg:" + function, "g")
    graph = {"fact": "cfg_graph_v1", "key": key, "function_key": function}
    if invalid.startswith("self_"):
        identity = "m:any" if invalid == "self_short" else "m:" + "f" * 64
        location = {"fact": "memory_location_v1", "key": identity, "identity_key": identity}
        with pytest.raises(AnalyzerProtocolError, match="inconsistent or duplicated"):
            builder._prepare_wire_keys([location])
    elif invalid in {"duplicate", "conflict"}:
        second = dict(graph)
        if invalid == "conflict":
            second["function_key"] += "_other"
        with pytest.raises(AnalyzerProtocolError, match="inconsistent or duplicated"):
            builder._prepare_wire_keys([graph, second])
    else:
        block = {
            "fact": "cfg_block_v1",
            "key": _wire_key("x" * 80, "b"),
            "graph_key": key if invalid == "unknown" else _wire_key("x" * 80, "m"),
            "index": 0,
        }
        with pytest.raises(AnalyzerProtocolError):
            builder._prepare_wire_keys([block])


def test_wire_key_threshold_counts_utf8_bytes() -> None:
    assert _wire_key("ä" * 33, "g") == "ä" * 33
    assert len(_wire_key("ä" * 34, "g")) == 66


def test_extended_wire_capability_is_required() -> None:
    hello = _fake_hello()
    hello["capabilities"].remove("compact_structural_keys_v1")
    hello["capabilities"].append("compact_access_keys_v1")
    with pytest.raises(AnalyzerProtocolError, match="compact_structural_keys_v1"):
        NativeAnalyzerClient._validate_handshake(hello)


def test_wire_projection_preserves_capped_and_external_location_identities(tmp_path: Path) -> None:
    root = create_wire_identity_fixture(tmp_path)
    configuration = CompilationDatabase.load(root / "compile_commands.json").configurations[0]
    facts = fresh_native_client(analyzer_binary(), timeout_seconds=15).analyze(root, configuration)
    analyses = [fact for fact in facts if fact["fact"] == "data_flow_analysis_v1"]
    assert any("alias_target_cap_exceeded" in fact["incomplete_reasons"] for fact in analyses)
    locations = [fact for fact in facts if fact["fact"] == "memory_location_v1"]
    assert any(
        fact["kind"] == "field" and "external_record.field" in fact["name"] for fact in locations
    )
    assert any(
        fact["kind"] == "global"
        and fact["name"] == "external_storage"
        and "declaration_key" not in fact
        for fact in locations
    )
    assert all(len(fact["key"].encode()) <= 66 for fact in locations)
    assert any(
        "external_storage:" in fact["identity_key"] for fact in locations if "identity_key" in fact
    )
    assert any(
        ":memory:field:" in fact["identity_key"] and "@FI@field" in fact["identity_key"]
        for fact in locations
        if "identity_key" in fact
    )
