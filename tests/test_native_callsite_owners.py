from __future__ import annotations

from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary

from cpp_context_engine.ingestion import NativeAnalyzerClient
from cpp_context_engine.ingestion.native import _FactBatchBuilder
from cpp_context_engine.models import BuildConfiguration

pytestmark = pytest.mark.native


def _analyze(tmp_path: Path, text: str):
    source = tmp_path / "owners.cpp"
    source.write_text(text, encoding="utf-8")
    configuration = BuildConfiguration(
        "owners", source, tmp_path, ("clang++", "-std=c++17", str(source)), "owners-command"
    )
    facts = NativeAnalyzerClient(analyzer_binary(), timeout_seconds=10).analyze(
        tmp_path, configuration
    )
    return configuration, facts


def test_shared_template_member_calls_keep_each_concrete_owner(tmp_path: Path) -> None:
    configuration, facts = _analyze(
        tmp_path,
        "struct Buffer { int size() const { return 7; } };\n"
        "struct Result {\n"
        "  Buffer buffer;\n"
        "  template<class T> int size_for() const { return buffer.size(); }\n"
        "};\n"
        "int use(Result& result) {\n"
        "  return result.size_for<int>() + result.size_for<double>() + result.size_for<char>();\n"
        "}\n",
    )
    sites = {fact["key"]: fact for fact in facts if fact["fact"] == "callsite_v1"}
    references = {fact["callsite_key"] for fact in facts if fact.get("callsite_key")}
    assert references <= sites.keys(), sorted(references - sites.keys())
    owners = {
        fact["function_key"]
        for fact in facts
        if fact["fact"] == "function_summary_v1" and "@F@size_for<" in fact["function_key"]
    }
    assert len(owners) == 3
    member_sites = [site for site in sites.values() if site["callee_text"] == "buffer.size()"]
    assert owners <= {site["owner_key"] for site in member_sites}
    for owner in owners:
        assert any(
            fact["fact"] == "call_result_binding_v1"
            and sites[fact["callsite_key"]]["owner_key"] == owner
            for fact in facts
        )
    batch = _FactBatchBuilder(tmp_path, configuration).build(facts)
    assert batch.call_result_bindings


def test_lambda_capture_initializers_and_nested_bodies_keep_distinct_owners(tmp_path: Path) -> None:
    configuration, facts = _analyze(
        tmp_path,
        "int capture_outer() { return 1; }\n"
        "int capture_inner() { return 2; }\n"
        "int body_outer() { return 3; }\n"
        "int body_inner() { return 4; }\n"
        "int ordinary() { return body_outer(); }\n"
        "int nested() {\n"
        "  auto outer = [x = capture_outer()] {\n"
        "    auto inner = [y = capture_inner()] { return y + body_inner(); };\n"
        "    return x + inner() + body_outer();\n"
        "  };\n"
        "  return outer();\n"
        "}\n",
    )
    symbols = {fact["key"]: fact for fact in facts if fact["fact"] == "symbol"}
    sites = [fact for fact in facts if fact["fact"] == "callsite_v1"]

    def owner(callee: str, line: int):
        site = next(
            site
            for site in sites
            if site["callee_text"] == callee and site["expansion_span"]["start_line"] == line
        )
        return symbols[site["owner_key"]]

    assert owner("body_outer()", 5)["qualified_name"] == "ordinary"
    assert owner("capture_outer()", 7)["qualified_name"] == "nested"
    outer = owner("body_outer()", 9)
    inner = owner("body_inner()", 8)
    assert outer["metadata"]["is_lambda_call_operator"]
    assert inner["metadata"]["is_lambda_call_operator"]
    assert outer["key"] != inner["key"]
    assert owner("capture_inner()", 8)["key"] == outer["key"]
    _FactBatchBuilder(tmp_path, configuration).build(facts)
