from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cpp_context_engine.api import AnswerRequest, QueryRequest
from cpp_context_engine.api.http import create_app
from cpp_context_engine.cli import main
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion import ClangUnavailableError
from cpp_context_engine.llm import DeterministicFakeProvider
from cpp_context_engine.runtime import build_runtime, index_project

FIXTURE = Path(__file__).parent / "fixtures" / "cpp_project"


def _config(project: Path, database: Path) -> AppConfig:
    return AppConfig(
        project_root=project,
        index_directory=database.parent,
        database_path=database,
        compilation_database=project / "compile_commands.json",
        embedding_dimensions=96,
        max_context_tokens=4_000,
    )


@pytest.mark.clang
def test_real_cpp_index_search_graph_and_fake_llm_answer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    config = _config(project, tmp_path / "index.db")
    try:
        first = index_project(config)
    except ClangUnavailableError as exc:
        pytest.skip(str(exc))

    assert first.indexing.indexed_translation_units == 2
    assert first.embedded_symbols > 0
    second = index_project(config)
    assert second.indexing.indexed_translation_units == 0
    assert second.embedded_symbols == 0

    question = "How does run reach Derived compute?"
    with build_runtime(config) as runtime:
        context = runtime.retrieval_service.query(QueryRequest(question, 4_000)).context

    names = {item.hit.symbol.qualified_name for item in context.items}
    assert "demo::Derived::compute" in names
    assert "run" in names
    assert any(item.path for item in context.items)
    cited = next(
        item.hit.symbol.id for item in context.items if item.hit.symbol.qualified_name == "run"
    )
    fake = DeterministicFakeProvider(
        json.dumps(
            {
                "action": "answer",
                "answer": "run constructs Derived and calls compute.",
                "source_ids": [cited, "invented-id"],
            }
        )
    )
    with build_runtime(config, llm=fake) as runtime:
        assert runtime.answer_service is not None
        answer = runtime.answer_service.answer(AnswerRequest(question))

        app = create_app(
            retrieval_service=runtime.retrieval_service,
            answer_service=runtime.answer_service,
        )
        api_context = TestClient(app).post("/v1/context", json={"query": question})
        api_answer = TestClient(app).post("/v1/answer", json={"query": question})

    assert answer.answer.startswith("run constructs")
    assert [source.symbol_id for source in answer.sources] == [cited]
    assert "Symbol-ID:" in fake.calls[0][0]
    assert api_context.status_code == 200
    assert any(item["qualified_name"] == "run" for item in api_context.json()["items"])
    assert api_answer.status_code == 200
    assert api_answer.json()["sources"][0]["symbol_id"] == cited

    assert (
        main(
            [
                "search",
                question,
                "--project",
                str(project),
                "--db",
                str(config.database_path),
                "--embedding-dimensions",
                "96",
                "--json",
            ]
        )
        == 0
    )


@pytest.mark.clang
def test_cli_index_and_ask_use_real_index_with_fake_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    database = tmp_path / "cli.db"
    arguments = [
        "index",
        str(project),
        "--compile-commands",
        str(project / "compile_commands.json"),
        "--db",
        str(database),
        "--embedding-dimensions",
        "96",
        "--json",
    ]
    try:
        exit_code = main(arguments)
    except ClangUnavailableError as exc:
        pytest.skip(str(exc))
    captured = capsys.readouterr()
    if exit_code == 2 and "clang Python bindings" in captured.err:
        pytest.skip("clang Python bindings are not installed")
    assert exit_code == 0
    indexed = json.loads(captured.out)
    assert indexed["indexed_translation_units"] == 2

    config = _config(project, database)
    with build_runtime(config) as runtime:
        context = runtime.retrieval_service.query(QueryRequest("run compute", 4_000)).context
    cited = next(
        item.hit.symbol.id for item in context.items if item.hit.symbol.qualified_name == "run"
    )
    fake = DeterministicFakeProvider(
        json.dumps({"action": "answer", "answer": "connected", "source_ids": [cited]})
    )
    monkeypatch.setattr("cpp_context_engine.runtime.llm_provider", lambda _config: fake)

    assert (
        main(
            [
                "ask",
                "run compute",
                "--project",
                str(project),
                "--db",
                str(database),
                "--embedding-dimensions",
                "96",
                "--json",
            ]
        )
        == 0
    )
    answered = json.loads(capsys.readouterr().out)
    assert answered["answer"] == "connected"
    assert answered["sources"][0]["symbol_id"] == cited


def test_runtime_rejects_unindexed_project_with_actionable_error(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "missing.db")

    with pytest.raises(ValueError, match="cpp-context index"):
        build_runtime(config)


def test_openai_embedding_configuration_requires_endpoint_and_model(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = replace(_config(project, tmp_path / "index.db"), embedding_provider="openai")

    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
        index_project(config)
