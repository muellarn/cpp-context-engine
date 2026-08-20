from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    CompilationDatabaseError,
    libclang_arguments,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cpp_project"


def test_loads_arguments_and_shell_command_entries() -> None:
    database = CompilationDatabase.load(FIXTURE / "compile_commands.json")

    assert len(database.configurations) == 2
    first = database.configurations[0]
    assert first.source_path == (FIXTURE / "src" / "model.cpp").resolve()
    assert first.directory == FIXTURE.resolve()
    assert first.arguments[0] == "clang++"
    assert first.output is None

    parser_arguments = libclang_arguments(first)
    assert "-c" not in parser_arguments
    assert "-o" not in parser_arguments
    assert "src/model.cpp" not in parser_arguments
    assert f"-I{(FIXTURE / 'include').resolve()}" in parser_arguments


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({}, "top-level JSON value must be an array"),
        ([], "compilation database is empty"),
        ([{"directory": ".", "file": "missing.cpp", "arguments": ["c++"]}], "does not exist"),
        (
            [
                {
                    "directory": ".",
                    "file": "source.cpp",
                    "arguments": ["c++"],
                    "command": "c++ source.cpp",
                }
            ],
            "exactly one",
        ),
    ],
)
def test_rejects_malformed_databases(tmp_path: Path, payload: object, expected: str) -> None:
    (tmp_path / "source.cpp").write_text("int main() {}", encoding="utf-8")
    path = tmp_path / "compile_commands.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CompilationDatabaseError, match=expected):
        CompilationDatabase.load(path)
