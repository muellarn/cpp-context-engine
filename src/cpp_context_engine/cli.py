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
from cpp_context_engine.api import AnswerRequest, CfgRequest, FlowRequest, QueryRequest
from cpp_context_engine.config import AppConfig
from cpp_context_engine.models import BuildScope, BuildVariant, IndexProfile
from cpp_context_engine.runtime import build_runtime, index_project
from cpp_context_engine.storage import SQLiteStore


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
    index.add_argument(
        "--build",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named compilation database; repeat to index multiple variants",
    )

    remove_build = commands.add_parser("remove-build", help="remove one indexed build variant")
    remove_build.add_argument("name", help="build variant name")
    _add_project_options(remove_build, positional=False)

    builds = commands.add_parser("builds", help="list indexed build variants and active scope")
    _add_project_options(builds, positional=False)
    builds.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    builds.add_argument("--build", action="append", default=[], metavar="NAME")

    cfg = commands.add_parser("cfg", help="read bounded control-flow evidence for a function")
    cfg.add_argument("symbol_id", help="stable function symbol ID")
    _add_project_options(cfg, positional=False)
    cfg.add_argument("--build", action="append", default=[], metavar="NAME")
    cfg.add_argument("--max-graphs", type=int, default=5)
    cfg.add_argument("--max-blocks", type=int, default=100)
    cfg.add_argument("--max-elements", type=int, default=500)
    cfg.add_argument("--max-edges", type=int, default=500)
    cfg.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    flow = commands.add_parser("flow", help="read bounded data-flow evidence for a function")
    flow.add_argument("symbol_id", help="stable function symbol ID")
    _add_project_options(flow, positional=False)
    flow.add_argument("--build", action="append", default=[], metavar="NAME")
    flow.add_argument("--max-analyses", type=int, default=5)
    flow.add_argument("--max-locations", type=int, default=200)
    flow.add_argument("--max-accesses", type=int, default=500)
    flow.add_argument("--max-evidence", type=int, default=500)
    flow.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    search = commands.add_parser("search", help="find connected symbols and source context")
    search.add_argument("query", help="natural-language, identifier, or signature query")
    _add_project_options(search, positional=False)
    _add_embedding_options(search)
    search.add_argument("--max-context-tokens", type=int)
    search.add_argument("--max-results", type=int)
    search.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    search.add_argument("--build", action="append", default=[], metavar="NAME")

    ask = commands.add_parser("ask", help="answer a code question with validated sources")
    ask.add_argument("query", help="question about the indexed C++ project")
    _add_project_options(ask, positional=False)
    _add_embedding_options(ask)
    _add_llm_options(ask)
    ask.add_argument("--max-context-tokens", type=int)
    ask.add_argument("--max-steps", type=int)
    ask.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ask.add_argument("--build", action="append", default=[], metavar="NAME")

    serve = commands.add_parser("serve", help="serve the wired FastAPI application")
    _add_project_options(serve, positional=False)
    _add_embedding_options(serve)
    _add_llm_options(serve)
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--build", action="append", default=[], metavar="NAME")

    mcp = commands.add_parser("mcp", help="serve the configured project over MCP")
    _add_project_options(mcp, positional=False, include_compile_commands=True)
    _add_embedding_options(mcp)
    _add_llm_options(mcp)
    mcp.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    mcp.add_argument("--host", help="HTTP bind address (default: configured localhost)")
    mcp.add_argument("--port", type=int, help="HTTP port")
    mcp.add_argument(
        "--build",
        action="append",
        default=[],
        metavar="NAME[=PATH]",
        help="operator-owned build scope or named compilation database",
    )
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
        parser.add_argument(
            "--profile", choices=tuple(IndexProfile), help="index profile (default: full)"
        )
        parser.add_argument("--compile-commands", type=Path, help="path to compile_commands.json")
        parser.add_argument("--libclang", type=Path, help="path to compatible libclang")
        parser.add_argument(
            "--clang-analyzer", type=Path, help="path to the Clang-18 LibTooling companion"
        )
        parser.add_argument("--analyzer-timeout", type=float, help="per-process timeout seconds")
        parser.add_argument("--analyzer-max-input-bytes", type=int)
        parser.add_argument("--analyzer-max-output-bytes", type=int)
        parser.add_argument("--analyzer-max-decoded-bytes", type=int)
        parser.add_argument("--analyzer-max-record-bytes", type=int)
        parser.add_argument("--analyzer-max-stderr-bytes", type=int)
        parser.add_argument("--analyzer-max-workers", type=int)
        parser.add_argument("--analyzer-max-spool-registries", type=int)
        parser.add_argument("--analyzer-max-spool-bytes", type=int)
        parser.add_argument("--analyzer-max-spool-files", type=int)
        parser.add_argument("--analyzer-max-domain-batches", type=int)


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
    build_arguments = getattr(args, "build", [])
    configured_variants = base.build_variants
    build_scope = base.build_scope
    if args.command in {"index", "mcp"} and any("=" in item for item in build_arguments):
        configured_variants = _parse_build_variants(build_arguments)
        build_scope = BuildScope(tuple(variant.name for variant in configured_variants))
        compilation_database = configured_variants[0].compilation_database
    elif explicit_commands is not None or (
        project_arg is not None
        and "CPP_CONTEXT_COMPILE_COMMANDS" not in os.environ
        and "CPP_CONTEXT_BUILDS" not in os.environ
    ):
        configured_variants = (BuildVariant("default", compilation_database),)
    if build_arguments and not any("=" in item for item in build_arguments):
        build_scope = BuildScope(tuple(build_arguments))
    return replace(
        base,
        project_root=project,
        index_directory=database.parent if database else project / ".cpp-context",
        database_path=database,
        compilation_database=compilation_database,
        build_variants=configured_variants,
        build_scope=build_scope,
        index_profile=IndexProfile(getattr(args, "profile", None) or base.index_profile),
        libclang_library_file=getattr(args, "libclang", None) or base.libclang_library_file,
        clang_analyzer_path=getattr(args, "clang_analyzer", None) or base.clang_analyzer_path,
        analyzer_timeout_seconds=getattr(args, "analyzer_timeout", None)
        or base.analyzer_timeout_seconds,
        analyzer_max_input_bytes=getattr(args, "analyzer_max_input_bytes", None)
        or base.analyzer_max_input_bytes,
        analyzer_max_output_bytes=getattr(args, "analyzer_max_output_bytes", None)
        or base.analyzer_max_output_bytes,
        analyzer_max_decoded_bytes=getattr(args, "analyzer_max_decoded_bytes", None)
        or base.analyzer_max_decoded_bytes,
        analyzer_max_record_bytes=getattr(args, "analyzer_max_record_bytes", None)
        or base.analyzer_max_record_bytes,
        analyzer_max_stderr_bytes=getattr(args, "analyzer_max_stderr_bytes", None)
        or base.analyzer_max_stderr_bytes,
        analyzer_max_workers=getattr(args, "analyzer_max_workers", None)
        or base.analyzer_max_workers,
        analyzer_max_spool_registries=_argument_or_default(
            args, "analyzer_max_spool_registries", base.analyzer_max_spool_registries
        ),
        analyzer_max_spool_bytes=_argument_or_default(
            args, "analyzer_max_spool_bytes", base.analyzer_max_spool_bytes
        ),
        analyzer_max_spool_files=_argument_or_default(
            args, "analyzer_max_spool_files", base.analyzer_max_spool_files
        ),
        analyzer_max_domain_batches=_argument_or_default(
            args, "analyzer_max_domain_batches", base.analyzer_max_domain_batches
        ),
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


def _argument_or_default(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _parse_build_variants(values: Sequence[str]) -> tuple[BuildVariant, ...]:
    variants: list[BuildVariant] = []
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("--build must use NAME=PATH when configuring compilation databases")
        variants.append(BuildVariant(name.strip(), Path(raw_path.strip())))
    return tuple(variants)


def _doctor(config: AppConfig, *, as_json: bool) -> int:
    from cpp_context_engine.ingestion.clang import _discover_libclang
    from cpp_context_engine.ingestion.native import NativeAnalyzerClient

    supported_python = sys.version_info >= (3, 11)
    try:
        clang_bindings = importlib.util.find_spec("clang.cindex") is not None
    except ModuleNotFoundError:
        clang_bindings = False
    discovered = config.libclang_library_file or _discover_libclang()
    analyzer_report: dict[str, Any] = {
        "clang_analyzer": str(config.clang_analyzer_path) if config.clang_analyzer_path else None,
        "clang_analyzer_executable": False,
        "clang_analyzer_protocol": None,
        "clang_analyzer_version": None,
        "clang_analyzer_clang_major": None,
        "clang_analyzer_capabilities": [],
        "advanced_facts_complete": False,
        "cfg_facts_available": False,
        "call_facts_available": False,
        "data_flow_facts_available": False,
        "function_summary_facts_available": False,
        "analysis_backend": "libclang-baseline",
    }
    analyzer_ok = True
    if config.clang_analyzer_path is not None:
        analyzer_report["clang_analyzer_executable"] = bool(
            config.clang_analyzer_path.is_file() and os.access(config.clang_analyzer_path, os.X_OK)
        )
        try:
            info = NativeAnalyzerClient(
                config.clang_analyzer_path,
                timeout_seconds=config.analyzer_timeout_seconds,
                max_input_bytes=config.analyzer_max_input_bytes,
                max_output_bytes=config.analyzer_max_output_bytes,
                max_decoded_bytes=config.analyzer_max_decoded_bytes,
                max_record_bytes=config.analyzer_max_record_bytes,
                max_stderr_bytes=config.analyzer_max_stderr_bytes,
            ).probe()
        except (OSError, RuntimeError, ValueError) as error:
            analyzer_ok = False
            analyzer_report["clang_analyzer_error"] = str(error)
        else:
            analyzer_report.update(
                {
                    "clang_analyzer_protocol": info.protocol_version,
                    "clang_analyzer_version": info.analyzer_version,
                    "clang_analyzer_clang_major": info.clang_major,
                    "clang_analyzer_capabilities": sorted(info.capabilities),
                    "advanced_facts_complete": True,
                    "cfg_facts_available": "function_cfg_v1" in info.capabilities,
                    "call_facts_available": {
                        "callsites_v1",
                        "dispatch_targets_v1",
                    }.issubset(info.capabilities),
                    "data_flow_facts_available": {
                        "intraprocedural_dataflow_v1",
                        "points_to_v1",
                    }.issubset(info.capabilities),
                    "function_summary_facts_available": {
                        "function_summaries_v1",
                        "interprocedural_bindings_v1",
                    }.issubset(info.capabilities),
                    "analysis_backend": "clang-libtooling",
                }
            )
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
        "status": "ok"
        if supported_python and config.project_root.is_dir() and analyzer_ok
        else "error",
        **analyzer_report,
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
        "analysis_backend": result.analysis_backend,
        "advanced_facts_complete": result.advanced_facts_complete,
        "analyzer_capabilities": list(result.analyzer_capabilities),
        "index_profile": result.index_profile.value,
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
        print(
            f"analysis backend: {payload['analysis_backend']}; "
            f"advanced facts complete: {payload['advanced_facts_complete']}"
        )
        print(f"database: {payload['database']}")
    return 0


def _run_remove_build(config: AppConfig, name: str) -> int:
    assert config.database_path is not None
    with SQLiteStore(config.database_path, project_root=config.project_root) as store:
        if not store.remove_build_variant(name, config.project_root):
            raise ValueError(f"build variant is not indexed: {name}")
    print(f"removed build variant: {name}")
    return 0


def _print_contract(payload: Any, *, as_json: bool) -> int:
    document = payload.model_dump(mode="json")
    if as_json:
        print(json.dumps(document, sort_keys=True))
    else:
        print(json.dumps(document, indent=2, sort_keys=True))
    return 0


def _run_builds(config: AppConfig, *, as_json: bool) -> int:
    with build_runtime(config) as runtime:
        result = runtime.analysis_service.list_builds()
    return _print_contract(result, as_json=as_json)


def _run_cfg(config: AppConfig, args: argparse.Namespace) -> int:
    request = CfgRequest(
        function_symbol_id=args.symbol_id,
        builds=args.build or None,
        max_graphs=args.max_graphs,
        max_blocks=args.max_blocks,
        max_elements=args.max_elements,
        max_edges=args.max_edges,
    )
    with build_runtime(config) as runtime:
        result = runtime.analysis_service.control_flow(request)
    return _print_contract(result, as_json=args.json)


def _run_flow(config: AppConfig, args: argparse.Namespace) -> int:
    request = FlowRequest(
        function_symbol_id=args.symbol_id,
        builds=args.build or None,
        max_analyses=args.max_analyses,
        max_locations=args.max_locations,
        max_accesses=args.max_accesses,
        max_evidence=args.max_evidence,
    )
    with build_runtime(config) as runtime:
        result = runtime.analysis_service.data_flow(request)
    return _print_contract(result, as_json=args.json)


def _context_payload(bundle: Any, project_root: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve(strict=False) if project_root is not None else None

    def display_path(path: Path) -> str:
        if root is None:
            return path.as_posix() if not path.is_absolute() else "<absolute-path-redacted>"
        resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
        if resolved.is_relative_to(root):
            return resolved.relative_to(root).as_posix()
        return "<outside-project>"

    return {
        "query": bundle.query,
        "scope": {
            "kind": "union" if len(bundle.build_variants) > 1 else "single",
            "label": bundle.scope_label or "build:default",
            "variants": list(bundle.build_variants or ("default",)),
        },
        "estimated_tokens": bundle.estimated_tokens,
        "truncated": bundle.truncated,
        "diagnostics": list(bundle.diagnostics),
        "results": [
            {
                "symbol_id": item.hit.symbol.id,
                "variant_id": item.hit.symbol.variant_id,
                "build_variant": item.hit.symbol.build_variant,
                "qualified_name": item.hit.symbol.qualified_name,
                "kind": item.hit.symbol.kind.value,
                "path": display_path(item.hit.symbol.span.path),
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
            QueryRequest(args.query, args.max_context_tokens, max_results=args.max_results)
        )
    payload = _context_payload(response.context, config.project_root)
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
        "scope": {
            "kind": "union" if len(response.build_variants) > 1 else "single",
            "label": response.scope_label,
            "variants": list(response.build_variants),
        },
        "complete": response.complete,
        "steps": response.steps,
        "sources": [
            {
                "symbol_id": source.symbol_id,
                "qualified_name": source.qualified_name,
                "build_variant": source.build_variant,
                "path": _project_path(source.path, config.project_root),
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


def _project_path(path: Path, project_root: Path) -> str:
    root = project_root.resolve(strict=False)
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    return (
        resolved.relative_to(root).as_posix()
        if resolved.is_relative_to(root)
        else "<outside-project>"
    )


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
        analysis_service=runtime.analysis_service,
        scoped_query=runtime.query_context,
        scoped_answer=runtime.answer_question if runtime.answer_service is not None else None,
    )
    app.state.cpp_context_runtime = runtime
    try:
        uvicorn.run(app, host=config.serve_host, port=config.serve_port)
    finally:
        runtime.close()
    return 0


def _run_mcp(config: AppConfig, args: argparse.Namespace) -> int:
    try:
        from cpp_context_engine.mcp.server import run_server
    except ImportError as exc:
        raise RuntimeError("mcp requires 'pip install cpp-context-engine[mcp]'") from exc
    return run_server(
        config,
        transport=args.transport,
        host=args.host,
        port=args.port,
    )


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
        if args.command == "remove-build":
            return _run_remove_build(config, args.name)
        if args.command == "builds":
            return _run_builds(config, as_json=args.json)
        if args.command == "cfg":
            return _run_cfg(config, args)
        if args.command == "flow":
            return _run_flow(config, args)
        if args.command == "search":
            return _run_search(config, args)
        if args.command == "ask":
            return _run_ask(config, args)
        if args.command == "serve":
            return _run_serve(config)
        if args.command == "mcp":
            return _run_mcp(config, args)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2
