from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary, clear_analyzer_discovery_cache

from cpp_context_engine.ingestion.native import (
    PROTOCOL,
    PROTOCOL_VERSION,
    REQUIRED_CAPABILITIES,
    REQUIRED_CLANG_MAJOR,
)


@pytest.fixture(autouse=True)
def _isolated_discovery_cache() -> None:
    clear_analyzer_discovery_cache()
    yield
    clear_analyzer_discovery_cache()


def _analyzer_script(
    directory: Path,
    *,
    name: str = "analyzer",
    protocol: str = PROTOCOL,
    protocol_version: int = PROTOCOL_VERSION,
    clang_major: int = REQUIRED_CLANG_MAJOR,
    executable: bool = True,
) -> Path:
    path = directory / name
    hello = {
        "type": "hello",
        "protocol": protocol,
        "protocol_version": protocol_version,
        "analyzer_version": "discovery-test",
        "clang_major": clang_major,
        "capabilities": sorted(REQUIRED_CAPABILITIES),
    }
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "json.loads(sys.stdin.readline())\n"
        f"print(json.dumps({hello!r}), flush=True)\n",
        encoding="utf-8",
    )
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_configured_analyzer_precedes_repository_fallback(tmp_path: Path) -> None:
    configured = _analyzer_script(tmp_path, name="configured")

    assert (
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(configured)},
            repository_root=tmp_path / "repository-without-build",
        )
        == configured.resolve()
    )


@pytest.mark.parametrize("configured", ["missing", "directory"])
def test_invalid_configured_analyzer_is_a_suite_configuration_error(
    tmp_path: Path, configured: str
) -> None:
    candidate = tmp_path / configured
    if configured == "directory":
        candidate.mkdir()

    with pytest.raises(pytest.UsageError, match="CPP_CONTEXT_TEST_ANALYZER.*regular file"):
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(candidate)},
            repository_root=tmp_path,
        )


@pytest.mark.skipif(os.name == "nt", reason="Windows does not expose POSIX execute bits")
def test_non_executable_configured_analyzer_is_a_suite_configuration_error(
    tmp_path: Path,
) -> None:
    candidate = _analyzer_script(tmp_path, executable=False)

    with pytest.raises(pytest.UsageError, match="CPP_CONTEXT_TEST_ANALYZER.*not executable"):
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(candidate)},
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"protocol_version": PROTOCOL_VERSION + 1}, "protocol mismatch"),
        ({"clang_major": REQUIRED_CLANG_MAJOR - 1}, "Clang major mismatch"),
    ],
)
def test_incompatible_configured_analyzer_is_a_suite_configuration_error(
    tmp_path: Path, overrides: dict[str, int], message: str
) -> None:
    candidate = _analyzer_script(tmp_path, **overrides)

    with pytest.raises(pytest.UsageError, match=message):
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(candidate)},
            repository_root=tmp_path,
        )


def test_repository_fallback_is_validated_when_no_analyzer_is_configured(tmp_path: Path) -> None:
    fallback = tmp_path / "build" / "clang-analyzer" / "cpp-context-clang-analyzer"
    fallback.parent.mkdir(parents=True)
    _analyzer_script(fallback.parent, name=fallback.name)

    assert analyzer_binary(environment={}, repository_root=tmp_path) == fallback.resolve()


def test_missing_repository_fallback_is_a_suite_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(pytest.UsageError, match="repository fallback.*regular file"):
        analyzer_binary(environment={}, repository_root=tmp_path)


def test_discovery_cache_is_scoped_to_the_selected_binary(tmp_path: Path) -> None:
    first = _analyzer_script(tmp_path, name="first")
    second = _analyzer_script(tmp_path, name="second")

    assert (
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(first)}, repository_root=tmp_path
        )
        == first.resolve()
    )
    assert (
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(second)}, repository_root=tmp_path
        )
        == second.resolve()
    )
