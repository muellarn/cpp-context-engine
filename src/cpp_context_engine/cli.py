"""Command-line entry points for local operation and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from cpp_context_engine import __version__
from cpp_context_engine.config import AppConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpp-context",
        description="Compiler-aware retrieval for large C++ codebases.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="show configuration and runtime diagnostics")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _doctor(config: AppConfig, *, as_json: bool) -> int:
    supported_python = sys.version_info >= (3, 11)
    report = {
        "index_directory": str(config.index_directory),
        "project_root": str(config.project_root),
        "python": sys.version.split()[0],
        "python_supported": supported_python,
        "status": "ok" if supported_python else "error",
    }
    if as_json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"status: {report['status']}")
        print(f"python: {report['python']}")
        print(f"project root: {report['project_root']}")
        print(f"index directory: {report['index_directory']}")
    return 0 if supported_python else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(AppConfig.from_environment(), as_json=args.json)
    parser.print_help()
    return 0
