"""Product CLI for indexing, connected search, answering, serving, and diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from cpp_context_engine import __version__
from cpp_context_engine.api import AnswerRequest, QueryRequest
from cpp_context_engine.config import AppConfig
from cpp_context_engine.runtime import build_runtime, index_project


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpp-context",
        description="Compiler-aware retrieval for large C++ codebases.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")

    doctor = commands.add_parser("doctor", help="show setup and runtime diagnostics")
    _add_project_options(doctor, positional=False, include_compile_commands=True)
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    index = commands.add_parser("index", help="incrementally index a C++ project")
    _add_project_options(index, positional=True, include_compile_commands=True)
    _add_embedding_options(index)
    index.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    search = commands.add_parser("search", help="find connected symbols and source context")
    search.add_argument("query", help="natural-language, identifier, or signature query")
    _add_project_options(search, positional=False)
    _add_embedding_options(search)
    search.add_argument("--max-context-tokens", type=int)
    search.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    ask = commands.add_parser("ask", help="answer a code question with validated sources")
    ask.add_argument("query", help="question about the indexed C++ project")
    _add_project_options(ask, positional=False)
    _add_embedding_options(ask)
    _add_llm_options(ask)
    ask.add_argument("--max-context-tokens", type=int)
    ask.add_argument("--max-steps", type=int)
    ask.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    serve = commands.add_parser("serve", help="serve the wired FastAPI application")
    _add_project_options(serve, positional=False)
    _add_embedding_options(serve)
    _add_llm_options(serve)
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    return parser


def _add_project_options(
    parser: argparse.ArgumentParser, *, positional: bool, include_compile_commands: bool = False
) -> None:
    if positional:
        parser.add_argument("project", nargs="?", type=Path, help="C++ project root")
    else:
        parser.add_argument("--project", type=Path, help="C++ project root")
    parser.add_argument("--db", type=Path, help="SQLite index path")
    if include_compile_commands:
        parser.add_argument("--compile-commands", type=Path, help="path to compile_commands.json")
        parser.add_argument("--libclang", type=Path, help="path to compatible libclang")


def _add_embedding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-provider", choices=("local", "openai"), help="vector provider")
    parser.add_argument("--embedding-base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--embedding-model", help="embedding model name")
    parser.add_argument("--embedding-dimensions", type=int, help="local feature-hash dimensions")


def _add_llm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--llm-model", help="chat completion model name")


def _resolved_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig.from_environment()
    project_arg = getattr(args, "project", None)
    project = (project_arg or base.project_root).expanduser().resolve(strict=False)
    explicit_db = getattr(args, "db", None)
    if explicit_db is not None:
        database = explicit_db.expanduser().resolve(strict=False)
    elif project_arg is not None and "CPP_CONTEXT_DATABASE" not in os.environ:
        database = project / ".cpp-context" / "index.db"
    else:
        database = base.database_path
    explicit_commands = getattr(args, "compile_commands", None)
    if explicit_commands is not None:
        compilation_database = explicit_commands.expanduser().resolve(strict=False)
    elif project_arg is not None and "CPP_CONTEXT_COMPILE_COMMANDS" not in os.environ:
        compilation_database = project / "build" / "compile_commands.json"
    else:
        compilation_database = base.compilation_database
    return replace(
        base,
        project_root=project,
        index_directory=database.parent if database else project / ".cpp-context",
        database_path=database,
        compilation_database=compilation_database,
        libclang_library_file=getattr(args, "libclang", None) or base.libclang_library_file,
        embedding_provider=getattr(args, "embedding_provider", None) or base.embedding_provider,
        embedding_base_url=getattr(args, "embedding_base_url", None) or base.embedding_base_url,
        embedding_model=getattr(args, "embedding_model", None) or base.embedding_model,
        embedding_dimensions=getattr(args, "embedding_dimensions", None)
        or base.embedding_dimensions,
        llm_base_url=getattr(args, "llm_base_url", None) or base.llm_base_url,
        llm_model=getattr(args, "llm_model", None) or base.llm_model,
        serve_host=getattr(args, "host", None) or base.serve_host,
        serve_port=getattr(args, "port", None) or base.serve_port,
    )


def _doctor(config: AppConfig, *, as_json: bool) -> int:
    from cpp_context_engine.ingestion.clang import _discover_libclang

    supported_python = sys.version_info >= (3, 11)
    try:
        clang_bindings = importlib.util.find_spec("clang.cindex") is not None
    except ModuleNotFoundError:
        clang_bindings = False
    discovered = config.libclang_library_file or _discover_libclang()
    report: dict[str, Any] = {
        "clang_bindings_installed": clang_bindings,
        "compile_commands": str(config.compilation_database),
        "compile_commands_exists": bool(
            config.compilation_database and config.compilation_database.is_file()
        ),
        "database": str(config.database_path),
        "database_exists": bool(config.database_path and config.database_path.is_file()),
        "embedding_provider": config.embedding_provider,
        "index_directory": str(config.index_directory),
        "libclang": str(discovered) if discovered else None,
        "libclang_found": bool(discovered and discovered.is_file()),
        "llm_configured": bool(config.llm_base_url and config.llm_model),
        "project_root": str(config.project_root),
        "project_root_exists": config.project_root.is_dir(),
        "python": sys.version.split()[0],
        "python_supported": supported_python,
        "status": "ok" if supported_python and config.project_root.is_dir() else "error",
    }
    if as_json:
        print(json.dumps(report, sort_keys=True))
    else:
        for label, value in report.items():
            print(f"{label.replace('_', ' ')}: {value}")
        if not clang_bindings:
            print("hint: install the compiler adapter with 'pip install cpp-context-engine[clang]'")
        if not report["compile_commands_exists"]:
            print("hint: generate compile_commands.json or pass --compile-commands")
    return 0 if report["status"] == "ok" else 1


def _run_index(config: AppConfig, *, as_json: bool) -> int:
    result = index_project(config)
    payload = {
        **asdict(result.indexing),
        "embedded_symbols": result.embedded_symbols,
        "embedding_model": result.embedding_model,
        "database": str(config.database_path),
        "project_root": str(config.project_root),
    }
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            "indexed "
            f"{payload['indexed_translation_units']} translation unit(s), "
            f"skipped {payload['skipped_translation_units']}, "
            f"removed {payload['removed_translation_units']}"
        )
        print(
            f"symbols: {payload['indexed_symbols']}; graph edges: {payload['indexed_edges']}; "
            f"new embeddings: {payload['embedded_symbols']} ({payload['embedding_model']})"
        )
        print(f"database: {payload['database']}")
    return 0


def _context_payload(bundle: Any) -> dict[str, Any]:
    return {
        "query": bundle.query,
        "estimated_tokens": bundle.estimated_tokens,
        "truncated": bundle.truncated,
        "diagnostics": list(bundle.diagnostics),
        "results": [
            {
                "symbol_id": item.hit.symbol.id,
                "qualified_name": item.hit.symbol.qualified_name,
                "kind": item.hit.symbol.kind.value,
                "path": str(item.hit.symbol.span.path),
                "start_line": item.hit.symbol.span.start_line,
                "end_line": item.hit.symbol.span.end_line,
                "score": item.hit.score,
                "reason": item.reason,
                "graph_path": [
                    {
                        "source_id": step.source_id,
                        "target_id": step.target_id,
                        "relation": step.relation.value,
                    }
                    for step in item.path
                ],
            }
            for item in bundle.items
        ],
    }


def _run_search(config: AppConfig, args: argparse.Namespace) -> int:
    with build_runtime(config) as runtime:
        response = runtime.retrieval_service.query(
            QueryRequest(args.query, args.max_context_tokens)
        )
    payload = _context_payload(response.context)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        if not payload["results"]:
            print("no connected source context found")
        for position, item in enumerate(payload["results"], start=1):
            print(
                f"{position}. {item['qualified_name']} "
                f"[{item['score']:.6f}] {item['path']}:{item['start_line']}-{item['end_line']}"
            )
            print(f"   reason: {item['reason']}")
            if item["graph_path"]:
                path = " -> ".join(
                    f"{step['source_id']} -[{step['relation']}]-> {step['target_id']}"
                    for step in item["graph_path"]
                )
                print(f"   graph: {path}")
        for diagnostic in payload["diagnostics"]:
            print(f"diagnostic: {diagnostic}", file=sys.stderr)
    return 0


def _run_ask(config: AppConfig, args: argparse.Namespace) -> int:
    with build_runtime(config, require_llm=True) as runtime:
        assert runtime.answer_service is not None
        response = runtime.answer_service.answer(
            AnswerRequest(args.query, args.max_context_tokens, args.max_steps)
        )
    payload = {
        "answer": response.answer,
        "complete": response.complete,
        "steps": response.steps,
        "sources": [
            {
                "symbol_id": source.symbol_id,
                "qualified_name": source.qualified_name,
                "path": str(source.path),
                "start_line": source.start_line,
                "end_line": source.end_line,
            }
            for source in response.sources
        ],
        "diagnostics": list(response.diagnostics),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(response.answer)
        if response.sources:
            print("\nSources:")
            for source in payload["sources"]:
                print(
                    f"- {source['qualified_name']} — "
                    f"{source['path']}:{source['start_line']}-{source['end_line']}"
                )
    return 0


def _run_serve(config: AppConfig) -> int:
    try:
        import uvicorn

        from cpp_context_engine.api.http import create_app
    except ImportError as exc:
        raise RuntimeError("serve requires 'pip install cpp-context-engine[api]'") from exc
    runtime = build_runtime(config)
    app = create_app(
        retrieval_service=runtime.retrieval_service,
        answer_service=runtime.answer_service,
    )
    app.state.cpp_context_runtime = runtime
    try:
        uvicorn.run(app, host=config.serve_host, port=config.serve_port)
    finally:
        runtime.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, rendering expected setup/provider failures without a traceback."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        config = _resolved_config(args)
        if args.command == "doctor":
            return _doctor(config, as_json=args.json)
        if args.command == "index":
            return _run_index(config, as_json=args.json)
        if args.command == "search":
            return _run_search(config, args)
        if args.command == "ask":
            return _run_ask(config, args)
        if args.command == "serve":
            return _run_serve(config)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2
