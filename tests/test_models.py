from pathlib import Path

import pytest

from cpp_context_engine.models import (
    BuildScope,
    BuildVariant,
    CodeSymbol,
    SearchQuery,
    SourceSpan,
    SymbolKind,
)


def test_source_span_rejects_reversed_coordinates() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        SourceSpan(Path("parser.cpp"), start_line=12, end_line=10)


def test_symbol_metadata_is_copied_and_immutable() -> None:
    metadata = {"language": "cpp"}
    symbol = CodeSymbol(
        id="cxx:parser",
        qualified_name="net::Parser",
        kind=SymbolKind.CLASS,
        span=SourceSpan(Path("parser.hpp"), 1, 20),
        metadata=metadata,
    )
    metadata["language"] = "changed"

    assert symbol.metadata["language"] == "cpp"
    with pytest.raises(TypeError):
        symbol.metadata["language"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize("text", ["", "   "])
def test_search_query_requires_text(text: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SearchQuery(text)


def test_build_scope_has_hard_public_count_and_name_limits() -> None:
    with pytest.raises(ValueError, match="must not exceed 16"):
        BuildScope(tuple(f"build-{index}" for index in range(17)))
    with pytest.raises(ValueError, match="must not exceed 128"):
        BuildVariant("x" * 129, Path("compile_commands.json"))
