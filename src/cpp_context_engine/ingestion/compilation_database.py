"""Strict loading and normalization of ``compile_commands.json`` files."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cpp_context_engine.models import BuildConfiguration


class CompilationDatabaseError(ValueError):
    """Raised when a compilation database cannot be used safely."""


def _digest(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return digest.hexdigest()


def _absolute(path: str, directory: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = directory / candidate
    return candidate.resolve(strict=False)


class CompilationDatabase:
    """An immutable, validated collection of compiler invocations."""

    def __init__(self, path: Path, configurations: tuple[BuildConfiguration, ...]) -> None:
        self.path = path
        self.configurations = configurations

    @classmethod
    def load(cls, path: Path) -> CompilationDatabase:
        path = path.resolve(strict=False)
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise CompilationDatabaseError(
                f"compilation database does not exist: {path}"
            ) from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CompilationDatabaseError(
                f"cannot read compilation database {path}: {error}"
            ) from error

        if not isinstance(payload, list):
            raise CompilationDatabaseError(f"{path}: top-level JSON value must be an array")
        if not payload:
            raise CompilationDatabaseError(f"{path}: compilation database is empty")

        configurations: list[BuildConfiguration] = []
        seen: set[str] = set()
        for position, raw in enumerate(payload):
            configuration = cls._parse_entry(path, position, raw)
            if configuration.id not in seen:
                configurations.append(configuration)
                seen.add(configuration.id)
        return cls(path, tuple(configurations))

    @staticmethod
    def _parse_entry(path: Path, position: int, raw: Any) -> BuildConfiguration:
        prefix = f"{path}: entry {position}"
        if not isinstance(raw, dict):
            raise CompilationDatabaseError(f"{prefix} must be an object")

        directory_raw = raw.get("directory")
        file_raw = raw.get("file")
        if not isinstance(directory_raw, str) or not directory_raw.strip():
            raise CompilationDatabaseError(f"{prefix} has no non-empty 'directory'")
        if not isinstance(file_raw, str) or not file_raw.strip():
            raise CompilationDatabaseError(f"{prefix} has no non-empty 'file'")

        directory_candidate = Path(directory_raw)
        if not directory_candidate.is_absolute():
            directory_candidate = path.parent / directory_candidate
        directory = directory_candidate.resolve(strict=False)
        if not directory.is_dir():
            raise CompilationDatabaseError(
                f"{prefix} working directory does not exist: {directory}"
            )
        source_path = _absolute(file_raw, directory)
        if not source_path.is_file():
            raise CompilationDatabaseError(f"{prefix} source file does not exist: {source_path}")

        has_arguments = "arguments" in raw
        has_command = "command" in raw
        if has_arguments == has_command:
            raise CompilationDatabaseError(
                f"{prefix} must contain exactly one of 'arguments' or 'command'"
            )
        if has_arguments:
            arguments_raw = raw["arguments"]
            if not isinstance(arguments_raw, list) or not all(
                isinstance(argument, str) for argument in arguments_raw
            ):
                raise CompilationDatabaseError(f"{prefix} 'arguments' must be an array of strings")
            arguments = tuple(arguments_raw)
        else:
            command_raw = raw["command"]
            if not isinstance(command_raw, str) or not command_raw.strip():
                raise CompilationDatabaseError(f"{prefix} 'command' must be a non-empty string")
            try:
                arguments = tuple(shlex.split(command_raw, posix=True))
            except ValueError as error:
                raise CompilationDatabaseError(
                    f"{prefix} has an invalid shell command: {error}"
                ) from error
        if not arguments or not arguments[0].strip():
            raise CompilationDatabaseError(f"{prefix} compiler command is empty")

        output_raw = raw.get("output")
        if output_raw is not None and not isinstance(output_raw, str):
            raise CompilationDatabaseError(f"{prefix} 'output' must be a string when present")
        output = _absolute(output_raw, directory) if output_raw else None
        command_hash = _digest(
            [str(directory), str(source_path), *arguments, str(output) if output else ""]
        )
        return BuildConfiguration(
            id=f"build_{command_hash[:32]}",
            source_path=source_path,
            directory=directory,
            arguments=arguments,
            command_hash=command_hash,
            output=output,
        )


_VALUE_OPTIONS = frozenset({"-o", "-MF", "-MT", "-MQ", "--output"})
_DROP_OPTIONS = frozenset({"-c", "-M", "-MM", "-MD", "-MMD", "-MP", "-MG"})
_PATH_VALUE_OPTIONS = frozenset(
    {"-I", "-isystem", "-iquote", "-idirafter", "-include", "-imacros", "--sysroot"}
)
_JOINED_PATH_OPTIONS = ("-isystem", "-iquote", "-idirafter", "-include", "-imacros", "-I")


def libclang_arguments(configuration: BuildConfiguration) -> tuple[str, ...]:
    """Return parser arguments with driver/output-only flags removed.

    The compilation database still retains the exact original invocation.  Only
    the copy passed to libclang is normalized.
    """

    result: list[str] = []
    arguments = configuration.arguments[1:]
    skip_next = False
    path_value_option: str | None = None
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if path_value_option is not None:
            result.extend((path_value_option, str(_absolute(argument, configuration.directory))))
            path_value_option = None
            continue
        if argument in _VALUE_OPTIONS:
            skip_next = True
            continue
        if argument in _DROP_OPTIONS:
            continue
        if argument in _PATH_VALUE_OPTIONS:
            path_value_option = argument
            continue
        if argument == str(configuration.source_path):
            continue
        try:
            if _absolute(argument, configuration.directory) == configuration.source_path:
                continue
        except (OSError, ValueError):
            pass
        if argument.startswith("-o") and argument != "-Winvalid-pch":
            continue
        joined_path_option = next(
            (option for option in _JOINED_PATH_OPTIONS if argument.startswith(option)), None
        )
        if joined_path_option is not None and argument != joined_path_option:
            path = argument[len(joined_path_option) :]
            result.append(f"{joined_path_option}{_absolute(path, configuration.directory)}")
            continue
        if argument.startswith("--sysroot="):
            sysroot = _absolute(argument.removeprefix("--sysroot="), configuration.directory)
            result.append(f"--sysroot={sysroot}")
            continue
        result.append(argument)
    if path_value_option is not None:
        result.append(path_value_option)
    return tuple(result)


def translation_unit_id(configuration: BuildConfiguration) -> str:
    """Return the stable identity of one distinct compiler invocation."""

    return f"tu_{_digest([configuration.id])[:32]}"
