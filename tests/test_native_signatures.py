from __future__ import annotations

from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary
from native_cache import fresh_native_client

from cpp_context_engine.ingestion.native import _FactBatchBuilder
from cpp_context_engine.models import BuildConfiguration, IndexProfile

pytestmark = pytest.mark.native


@pytest.fixture
def signature_project(tmp_path: Path) -> tuple[Path, BuildConfiguration]:
    source = tmp_path / "signatures.cpp"
    source.write_text(
        """int ordinary(int value = 3) {
  int body_only_token = value + 7;
  return body_only_token;
}
int ordinary(double value) { return static_cast<int>(value); }
struct Widget {
  int field;
  explicit Widget(int value = 4) : field(value + 11) { field += 1; }
  ~Widget() { field = 12; }
  int operator()(int value = 2) const & noexcept { return field + value; }
  explicit operator bool() const noexcept { return field != 0; }
  Widget &operator=(const Widget &) = default;
  void removed() = delete;
  template<class T> auto convert(T value) const -> T
    requires requires(T item) { item + item; } { return value + value; }
};
template<class T> T twice(T value = T{2}) { return value + value; }
template<> int twice<int>(int value) { return value + 13; }
auto trailing(int value) noexcept -> long { return value + 14; }
int default_lambda(int value = [] { return 15; }()) { return value + 16; }
bool exception_spec() noexcept([] { return true; }()) { return false; }
auto lambda_type() -> decltype([] { return 31; }()) { return 32; }
template<int N = [] { return 33; }()> int lambda_template() { return N; }
template<class T> requires ([] { return true; }())
T leading_requires(T value) { return value; }
struct TryConstructor {
  int field;
  explicit([] { return true; }()) TryConstructor(int value = [] { return 34; }())
    try : field(value) { field += 35; } catch (...) { throw; }
};
[[nodiscard]] inline constexpr int decorated(int value = 1) noexcept { return value; }
#define DEFINE_FUNCTION(name) int name(int value) { return value + 17; }
DEFINE_FUNCTION(macro_function)
namespace nested { struct Record { int member() { return 18; } }; }
int instantiate() { Widget w; return twice<int>(w.convert(ordinary(2))) + w(); }
""",
        encoding="utf-8",
    )
    return tmp_path, BuildConfiguration(
        id="signatures",
        source_path=source,
        directory=tmp_path,
        arguments=("clang++", "-std=c++20", str(source)),
        command_hash="signatures",
    )


@pytest.mark.parametrize("profile", [IndexProfile.FULL, IndexProfile.NAVIGATION])
def test_function_signatures_preserve_declarations_and_complete_source(
    signature_project: tuple[Path, BuildConfiguration], profile: IndexProfile
) -> None:
    root, configuration = signature_project
    facts = fresh_native_client(analyzer_binary(), timeout_seconds=5, profile=profile).analyze(
        root, configuration
    )
    symbols = [fact for fact in facts if fact["fact"] == "symbol"]

    def signatures(name: str) -> list[str]:
        return [
            " ".join(fact["signature"].split())
            for fact in symbols
            if fact["qualified_name"] == name
        ]

    ordinary = next(fact for fact in symbols if "body_only_token" in fact["source_text"])
    assert ordinary["signature"] == "int ordinary(int value = 3)"
    assert "return body_only_token;" in ordinary["source_text"]
    metadata = ordinary["metadata"]
    physical_source = Path(ordinary["span"]["path"]).read_bytes()
    assert (
        ordinary["source_text"]
        == physical_source[metadata["start_offset"] : metadata["end_offset_exclusive"]].decode()
    )
    assert "int ordinary(double value)" in signatures("ordinary")
    assert "explicit Widget(int value = 4)" in signatures("Widget::Widget")
    assert "~Widget() noexcept" in signatures("Widget::~Widget")
    assert "int operator()(int value = 2) const & noexcept" in signatures("Widget::operator()")
    assert "explicit operator bool() const noexcept" in signatures("Widget::operator bool")
    assert any(value.endswith(" = default") for value in signatures("Widget::operator="))
    assert "void removed() = delete" in signatures("Widget::removed")
    assert "auto trailing(int value) noexcept -> long" in signatures("trailing")
    assert "int macro_function(int value)" in signatures("macro_function")
    assert any("template <class T>" in value and "T{2}" in value for value in signatures("twice"))
    assert any("template<>" in value and "twice<int>" in value for value in signatures("twice"))
    assert any(
        "const -> T requires requires (T item) { item + item; }" in value
        for value in signatures("Widget::convert")
    )
    # A lambda body inside a default argument belongs to the declaration, unlike
    # the outer function body. A brace-stripper would lose this information.
    default_lambda = signatures("default_lambda")
    assert len(default_lambda) == 1
    assert "return 15;" in default_lambda[0]
    assert "return value + 16;" not in default_lambda[0]
    assert any("return true;" in value for value in signatures("exception_spec"))
    assert any("return 31;" in value for value in signatures("lambda_type"))
    assert any("return 33;" in value for value in signatures("lambda_template"))
    assert any(
        "requires ([] { return true; }())" in value for value in signatures("leading_requires")
    )
    constructor = signatures("TryConstructor::TryConstructor")
    assert any(
        "explicit([] { return true; }())" in value and "return 34;" in value
        for value in constructor
    )
    assert all("field(value)" not in value and "catch" not in value for value in constructor)
    assert any(
        "inline constexpr" in value and '[[nodiscard("")]]' in value
        for value in signatures("decorated")
    )
    for fact in symbols:
        if fact["kind"] in {"function", "method"}:
            assert "body_only_token" not in fact["signature"]
            assert "return value + value;" not in fact["signature"]
            assert "field += 1" not in fact["signature"]
    assert any(
        "field += 1;" in fact["source_text"]
        for fact in symbols
        if fact["qualified_name"] == "Widget::Widget"
    )
    assert any(
        "catch (...) { throw; }" in fact["source_text"]
        for fact in symbols
        if fact["qualified_name"] == "TryConstructor::TryConstructor"
    )
    # Do not propagate terse printing to unrelated declaration kinds.
    assert any("field += 1" in value for value in signatures("Widget"))
    assert any("return 18;" in value for value in signatures("nested"))
    _FactBatchBuilder(root, configuration, profile=profile).build(facts)
