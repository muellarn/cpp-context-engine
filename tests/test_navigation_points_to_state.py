from __future__ import annotations

from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary
from native_cache import fresh_native_client

from cpp_context_engine.ingestion.native import _FactBatchBuilder
from cpp_context_engine.models import BuildConfiguration, IndexProfile

pytestmark = pytest.mark.native


@pytest.fixture
def points_to_project(tmp_path: Path) -> tuple[Path, BuildConfiguration]:
    source = tmp_path / "points_to.cpp"
    source.write_text(
        """int first(int value) { return value + 1; }
int second(int value) { return value + 2; }
using Callback = int (*)(int);
struct Holder { int value; };
int branch_loop(Holder &object, bool choose, int count) {
  int *address = &object.value;
  Callback callback = first;
  while (count-- > 0) {
    if (choose) callback = second;
    else callback = first;
    *address += count;
  }
  return callback(object.value);
}
int field_rhs(bool choose) {
  Holder object{4};
  int *address = &object.value;
  Callback callback = choose ? first : second;
  *address = 5;
  return callback(*address);
}
int nullable(bool choose) {
  Callback callback = choose ? first : nullptr;
  return callback ? callback(2) : 0;
}
""",
        encoding="utf-8",
    )
    return tmp_path, BuildConfiguration(
        id="navigation-points-to-state",
        source_path=source,
        directory=tmp_path,
        arguments=("clang++", "-std=c++17", str(source)),
        command_hash="navigation-points-to-state",
    )


def test_navigation_preserves_cyclic_points_to_and_rhs_location_facts(
    points_to_project: tuple[Path, BuildConfiguration],
) -> None:
    root, configuration = points_to_project
    streams = {}
    for profile in (IndexProfile.FULL, IndexProfile.NAVIGATION):
        facts = fresh_native_client(analyzer_binary(), timeout_seconds=5, profile=profile).analyze(
            root, configuration
        )
        _FactBatchBuilder(root, configuration, profile=profile).build(facts)
        streams[profile] = facts

    navigation_kinds = {
        "file",
        "include",
        "symbol",
        "occurrence",
        "edge",
        "callsite_v1",
        "call_target_v1",
        "callsite_resolution_v1",
    }
    navigation = streams[IndexProfile.NAVIGATION]
    assert (
        tuple(fact for fact in streams[IndexProfile.FULL] if fact["fact"] in navigation_kinds)
        == navigation
    )
    symbols = {
        fact["key"]: fact["qualified_name"] for fact in navigation if fact["fact"] == "symbol"
    }
    assert "Holder::value" in symbols.values()
    calls = {
        fact["key"]: symbols[fact["owner_key"]]
        for fact in navigation
        if fact["fact"] == "callsite_v1"
    }
    targets: dict[str, set[str]] = {}
    resolutions = set()
    for fact in navigation:
        if fact["fact"] == "call_target_v1":
            targets.setdefault(calls[fact["callsite_key"]], set()).add(symbols[fact["target_key"]])
            assert fact["certainty"] == "possible"
        elif fact["fact"] == "callsite_resolution_v1":
            resolutions.add(calls[fact["callsite_key"]])
            assert fact["target_set_complete"] is True
            assert fact["unresolved_reason"] == ""
    assert targets == {
        "branch_loop": {"first", "second"},
        "field_rhs": {"first", "second"},
        "nullable": {"first"},
    }
    assert resolutions == set(targets)
    assert any(fact["fact"] == "data_flow_analysis_v1" for fact in streams[IndexProfile.FULL])
    assert all(fact["fact"] in navigation_kinds for fact in navigation)
