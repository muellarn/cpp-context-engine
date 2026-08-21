from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion import IndexingResult
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.llm import DeterministicFakeProvider
from cpp_context_engine.models import (
    BuildConfiguration,
    CodeSymbol,
    GraphEdge,
    GraphRelation,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.runtime import IndexOperationResult, Runtime, build_runtime
from cpp_context_engine.storage import SQLiteStore

FIXTURE = Path(__file__).parent / "fixtures" / "cpp_project"


def _config(project: Path, database: Path) -> AppConfig:
    return AppConfig(
        project_root=project,
        index_directory=database.parent,
        database_path=database,
        compilation_database=project / "compile_commands.json",
        embedding_dimensions=32,
        max_context_tokens=4_000,
    )


def _seed_index(config: AppConfig) -> None:
    source = config.project_root / "src" / "sample.cpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "int callee() { return 7; }\nint caller() { return callee(); }\n",
        encoding="utf-8",
    )
    build = BuildConfiguration(
        id="build",
        source_path=source,
        directory=config.project_root,
        arguments=("c++", "src/sample.cpp"),
        command_hash="command",
    )
    unit = TranslationUnit(
        id="unit",
        build_configuration_id=build.id,
        source_path=source,
        content_hash="content",
    )
    callee = CodeSymbol(
        id="cxx:callee",
        qualified_name="callee",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(source, 1, 1),
        signature="int callee()",
        source_text="int callee() { return 7; }",
        source_hash="callee",
        build_configuration_id=build.id,
        translation_unit_id=unit.id,
        metadata={"is_definition": True},
    )
    caller = CodeSymbol(
        id="cxx:caller",
        qualified_name="caller",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(source, 2, 2),
        signature="int caller()",
        source_text="int caller() { return callee(); }",
        source_hash="caller",
        build_configuration_id=build.id,
        translation_unit_id=unit.id,
        metadata={"is_definition": True},
    )
    batch = IngestionBatch(
        build_configurations=(build,),
        translation_units=(unit,),
        symbols=(callee, caller),
        occurrences=(),
        edges=(GraphEdge(caller.id, callee.id, GraphRelation.CALLS, unit.id),),
    )
    assert config.database_path is not None
    with SQLiteStore(config.database_path, project_root=config.project_root) as store:
        store.apply_ingestion(
            config.project_root,
            batch,
            current_translation_unit_ids=frozenset({unit.id}),
        )


def _fake_index(config: AppConfig) -> IndexOperationResult:
    _seed_index(config)
    return IndexOperationResult(
        IndexingResult(
            indexed_translation_units=1,
            skipped_translation_units=0,
            removed_translation_units=0,
            indexed_symbols=2,
            indexed_occurrences=0,
            indexed_edges=1,
        ),
        embedded_symbols=0,
        embedding_model="fixture-local",
    )


def test_in_memory_mcp_missing_index_then_index_search_read_and_graph(
    tmp_path: Path, monkeypatch
) -> None:
    from cpp_context_engine.mcp import server as mcp_server

    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "index.db")
    runtimes: list[Runtime] = []

    def tracked_runtime(selected: AppConfig) -> Runtime:
        runtime = build_runtime(selected)
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(mcp_server, "run_project_index", _fake_index)
    monkeypatch.setattr(mcp_server, "build_runtime", tracked_runtime)
    server = mcp_server.create_mcp_server(config)

    async def scenario() -> None:
        async with Client(server, mode="legacy") as client:
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == {
                "index_project",
                "list_builds",
                "control_flow",
                "data_flow",
                "search_code",
                "read_symbol",
                "neighbors",
                "callers",
                "callees",
                "ask_code",
            }
            assert tools["index_project"].input_schema["properties"] == {}
            for tool in tools.values():
                assert tool.output_schema is not None
                assert tool.annotations is not None
                assert tool.annotations.destructive_hint is False
                forbidden = {"project", "db", "database", "compile_commands", "path", "file"}
                assert forbidden.isdisjoint(tool.input_schema.get("properties", {}))
            assert tools["search_code"].input_schema["properties"]["query"]["maxLength"] == 2048
            assert tools["neighbors"].input_schema["properties"]["depth"]["maximum"] == 3
            assert tools["search_code"].annotations.open_world_hint is False

            missing = await client.call_tool("search_code", {"query": "callee"})
            assert missing.is_error
            assert str(tmp_path) not in missing.content[0].text
            assert "index_project" in missing.content[0].text

            indexed = await client.call_tool("index_project", {})
            assert not indexed.is_error
            assert indexed.structured_content is not None
            assert indexed.structured_content["indexed_symbols"] == 2

            searched = await client.call_tool(
                "search_code",
                {"query": "callee", "max_results": 1, "max_context_tokens": 1000},
            )
            assert not searched.is_error
            assert searched.structured_content is not None
            first = searched.structured_content["items"][0]
            assert first["symbol"]["symbol_id"] == "cxx:callee"
            assert first["symbol"]["location"] == {
                "path": "src/sample.cpp",
                "start_line": 1,
                "end_line": 1,
            }
            assert str(project) not in json.dumps(searched.structured_content)

            read = await client.call_tool("read_symbol", {"symbol_id": "cxx:caller"})
            assert not read.is_error
            assert read.structured_content["source_text"] == "int caller() { return callee(); }"

            callers = await client.call_tool("callers", {"symbol_id": "cxx:callee"})
            assert callers.structured_content["direction"] == "incoming"
            assert callers.structured_content["edges"][0]["source"]["symbol_id"] == "cxx:caller"
            callers_by_variant = await client.call_tool(
                "callers", {"symbol_id": first["symbol"]["variant_id"]}
            )
            assert callers_by_variant.structured_content["edges"][0]["source"]["symbol_id"] == (
                "cxx:caller"
            )
            callees = await client.call_tool("callees", {"symbol_id": "cxx:caller"})
            assert callees.structured_content["direction"] == "outgoing"
            assert callees.structured_content["edges"][0]["target"]["symbol_id"] == "cxx:callee"

            invalid = await client.call_tool("neighbors", {"symbol_id": "cxx:caller", "depth": 4})
            assert invalid.is_error
            unknown = await client.call_tool("read_symbol", {"symbol_id": "does-not-exist"})
            assert unknown.is_error
            assert str(tmp_path) not in unknown.content[0].text

    anyio.run(scenario)
    assert runtimes
    with pytest.raises(sqlite3.ProgrammingError):
        runtimes[-1].store.get_symbol("cxx:callee")


def test_mcp_rejects_indexed_source_outside_project_without_leaking_path(tmp_path: Path) -> None:
    from cpp_context_engine.mcp.server import create_mcp_server

    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "index.db")
    _seed_index(config)
    outside = tmp_path / "secret.cpp"
    outside.write_text("secret", encoding="utf-8")
    assert config.database_path is not None
    with SQLiteStore(config.database_path, project_root=project) as store:
        original = store.get_symbol("cxx:callee")
        assert original is not None
        store.put_symbols((replace(original, id="cxx:outside", span=SourceSpan(outside, 1, 1)),))

    async def scenario() -> None:
        async with Client(create_mcp_server(config), mode="legacy") as client:
            result = await client.call_tool("read_symbol", {"symbol_id": "cxx:outside"})
            assert result.is_error
            assert str(tmp_path) not in result.content[0].text
            assert "secret.cpp" not in result.content[0].text

    anyio.run(scenario)


def test_mcp_callers_reports_when_per_node_fanout_truncates_results(tmp_path: Path) -> None:
    from cpp_context_engine.mcp.server import create_mcp_server

    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "index.db")
    _seed_index(config)
    assert config.database_path is not None
    with SQLiteStore(config.database_path, project_root=project) as store:
        template = store.get_symbol("cxx:caller")
        assert template is not None
        callers = tuple(
            replace(
                template,
                id=f"cxx:caller-{index}",
                qualified_name=f"caller_{index}",
                source_hash=f"caller-{index}",
            )
            for index in range(21)
        )
        store.put_symbols(callers)
        store.put_edges(
            tuple(
                GraphEdge(symbol.id, "cxx:callee", GraphRelation.CALLS, "unit")
                for symbol in callers
            )
        )

    async def scenario() -> None:
        async with Client(create_mcp_server(config), mode="legacy") as client:
            result = await client.call_tool(
                "callers", {"symbol_id": "cxx:callee", "max_results": 20}
            )
            assert not result.is_error
            assert len(result.structured_content["edges"]) == 20
            assert result.structured_content["truncated"] is True

    anyio.run(scenario)


def test_mcp_ask_uses_bounded_sources_and_marks_external_provider(
    tmp_path: Path, monkeypatch
) -> None:
    from cpp_context_engine.mcp import server as mcp_server

    project = tmp_path / "project"
    project.mkdir()
    config = replace(
        _config(project, tmp_path / "index.db"),
        llm_base_url="https://provider.invalid/v1",
        llm_model="fixture-chat",
    )
    _seed_index(config)
    fake = DeterministicFakeProvider(
        json.dumps(
            {
                "action": "answer",
                "answer": f"See {project}/src/sample.cpp.",
                "source_ids": ["cxx:caller"],
            }
        )
    )
    monkeypatch.setattr(
        mcp_server,
        "build_runtime",
        lambda selected: build_runtime(selected, llm=fake),
    )

    async def scenario() -> None:
        async with Client(mcp_server.create_mcp_server(config), mode="legacy") as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert tools["ask_code"].annotations.open_world_hint is True
            assert tools["search_code"].annotations.open_world_hint is False
            result = await client.call_tool(
                "ask_code",
                {"query": "What calls callee?", "max_context_tokens": 1000, "max_steps": 1},
            )
            assert not result.is_error
            assert result.structured_content["answer"] == "See src/sample.cpp."
            assert result.structured_content["sources"][0]["location"]["path"] == "src/sample.cpp"

    anyio.run(scenario)
    assert "Location: src/sample.cpp:" in fake.calls[0][0]
    assert str(project) not in fake.calls[0][0]


def test_hosted_embedding_tools_are_marked_open_world(tmp_path: Path) -> None:
    from cpp_context_engine.mcp.server import create_mcp_server

    project = tmp_path / "project"
    project.mkdir()
    config = replace(
        _config(project, tmp_path / "missing.db"),
        embedding_provider="openai",
        embedding_base_url="https://provider.invalid/v1",
        embedding_model="fixture-embedding",
    )

    async def scenario() -> None:
        async with Client(create_mcp_server(config), mode="legacy") as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert tools["index_project"].annotations.open_world_hint is True
            assert tools["search_code"].annotations.open_world_hint is True
            assert tools["read_symbol"].annotations.open_world_hint is False

    anyio.run(scenario)


def test_mcp_sanitizes_provider_failures(tmp_path: Path, monkeypatch) -> None:
    from cpp_context_engine.mcp import server as mcp_server

    project = tmp_path / "project"
    project.mkdir()
    config = replace(
        _config(project, tmp_path / "index.db"),
        llm_base_url="https://provider.invalid/v1",
        llm_model="fixture-chat",
    )
    _seed_index(config)

    class FailingProvider:
        def complete(self, _prompt: str, *, tools=()) -> str:
            raise RuntimeError(f"provider-secret at {tmp_path}/private")

    monkeypatch.setattr(
        mcp_server,
        "build_runtime",
        lambda selected: build_runtime(selected, llm=FailingProvider()),
    )

    async def scenario() -> None:
        async with Client(mcp_server.create_mcp_server(config), mode="legacy") as client:
            result = await client.call_tool(
                "ask_code",
                {"query": "What calls callee?", "max_context_tokens": 1000, "max_steps": 1},
            )
            assert result.is_error
            rendered = result.content[0].text
            assert "provider-secret" not in rendered
            assert str(tmp_path) not in rendered
            assert "Code answering failed" in rendered

    anyio.run(scenario)


@pytest.mark.clang
def test_real_clang_indexing_through_mcp_tool(tmp_path: Path) -> None:
    from cpp_context_engine.ingestion.clang import _discover_libclang
    from cpp_context_engine.mcp.server import create_mcp_server

    if _discover_libclang() is None:
        pytest.skip("compatible libclang is unavailable")
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    config = _config(project, tmp_path / "clang-index.db")

    async def scenario() -> None:
        async with Client(create_mcp_server(config), mode="legacy") as client:
            indexed = await client.call_tool("index_project", {})
            assert not indexed.is_error, indexed.content
            assert indexed.structured_content["indexed_translation_units"] == 2
            searched = await client.call_tool(
                "search_code",
                {"query": "Derived compute", "max_results": 10, "max_context_tokens": 2000},
            )
            assert not searched.is_error
            names = {
                item["symbol"]["qualified_name"] for item in searched.structured_content["items"]
            }
            assert "demo::Derived::compute" in names
            assert all(
                not item["symbol"]["location"]["path"].startswith("/")
                for item in searched.structured_content["items"]
            )

    anyio.run(scenario)


def test_real_stdio_transport_is_protocol_clean_and_returns_structured_content(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "index.db")
    _seed_index(config)
    environment = {
        "CPP_CONTEXT_PROJECT_ROOT": str(project),
        "CPP_CONTEXT_DATABASE": str(config.database_path),
        "CPP_CONTEXT_COMPILE_COMMANDS": str(config.compilation_database),
        "CPP_CONTEXT_EMBEDDING_DIMENSIONS": "32",
    }
    stderr_path = tmp_path / "mcp-stderr.log"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cpp_context_engine.mcp.server"],
        env=environment,
        cwd=Path(__file__).parents[1],
    )

    with stderr_path.open("w+", encoding="utf-8") as stderr:

        async def scenario() -> None:
            async with Client(stdio_client(parameters, errlog=stderr), mode="legacy") as client:
                listed = await client.list_tools()
                assert "search_code" in {tool.name for tool in listed.tools}
                result = await client.call_tool(
                    "search_code",
                    {"query": "callee", "max_results": 1, "max_context_tokens": 1000},
                )
                assert not result.is_error
                assert result.structured_content["items"][0]["symbol"]["symbol_id"] == "cxx:callee"

        anyio.run(scenario)
        stderr.seek(0)
        assert "Traceback" not in stderr.read()


def test_streamable_http_transport_smoke(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = _config(project, tmp_path / "index.db")
    _seed_index(config)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = os.environ | {
        "CPP_CONTEXT_PROJECT_ROOT": str(project),
        "CPP_CONTEXT_DATABASE": str(config.database_path),
        "CPP_CONTEXT_EMBEDDING_DIMENSIONS": "32",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cpp_context_engine.mcp.server",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    async def scenario() -> None:
        last_error: Exception | None = None
        # Importing the SDK and native bindings from a Windows-mounted worktree can
        # take several seconds before Uvicorn binds, especially on a cold cache.
        for _attempt in range(150):
            try:
                async with Client(
                    f"http://127.0.0.1:{port}/mcp", mode="legacy", read_timeout_seconds=2
                ) as client:
                    result = await client.call_tool(
                        "search_code",
                        {"query": "callee", "max_results": 1, "max_context_tokens": 1000},
                    )
                    assert not result.is_error
                    return
            except Exception as exc:
                last_error = exc
                await anyio.sleep(0.1)
        raise AssertionError(f"Streamable HTTP server did not become ready: {last_error!r}")

    try:
        anyio.run(scenario)
    finally:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
    assert stdout == ""
    assert "Traceback" not in stderr
