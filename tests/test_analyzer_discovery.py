from __future__ import annotations

import os
import stat
from pathlib import Path

import analyzer_discovery
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
    capabilities: set[str] | None = None,
    executable: bool = True,
) -> Path:
    path = directory / name
    hello = {
        "type": "hello",
        "protocol": protocol,
        "protocol_version": protocol_version,
        "analyzer_version": "discovery-test",
        "clang_major": clang_major,
        "capabilities": sorted(REQUIRED_CAPABILITIES if capabilities is None else capabilities),
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


def test_invalid_explicit_analyzer_never_falls_back(tmp_path: Path) -> None:
    fallback = tmp_path / "build" / "clang-analyzer" / "cpp-context-clang-analyzer"
    fallback.parent.mkdir(parents=True)
    _analyzer_script(fallback.parent, name=fallback.name)

    with pytest.raises(pytest.UsageError, match="CPP_CONTEXT_TEST_ANALYZER.*regular file"):
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(tmp_path / "missing-explicit")},
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
        ({"protocol": "wrong-protocol"}, "protocol mismatch"),
        ({"protocol_version": PROTOCOL_VERSION + 1}, "protocol mismatch"),
        ({"clang_major": REQUIRED_CLANG_MAJOR - 1}, "Clang major mismatch"),
    ],
)
def test_incompatible_configured_analyzer_is_a_suite_configuration_error(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    candidate = _analyzer_script(tmp_path, **overrides)  # type: ignore[arg-type]

    with pytest.raises(pytest.UsageError, match=message):
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(candidate)},
            repository_root=tmp_path,
        )


def test_missing_required_capability_is_a_suite_configuration_error(tmp_path: Path) -> None:
    capabilities = set(REQUIRED_CAPABILITIES)
    capabilities.remove("symbols")
    candidate = _analyzer_script(tmp_path, capabilities=capabilities)

    with pytest.raises(pytest.UsageError, match="missing required capabilities: symbols"):
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


def test_discovery_revalidates_changed_binary_with_preserved_size_and_mtime(
    tmp_path: Path,
) -> None:
    candidate = _analyzer_script(tmp_path)
    original = candidate.stat()
    environment = {"CPP_CONTEXT_TEST_ANALYZER": str(candidate)}
    assert analyzer_binary(environment=environment, repository_root=tmp_path) == candidate.resolve()

    _analyzer_script(tmp_path, clang_major=REQUIRED_CLANG_MAJOR - 1)
    os.utime(candidate, ns=(original.st_atime_ns, original.st_mtime_ns))
    replaced = candidate.stat()
    assert (replaced.st_ino, replaced.st_size, replaced.st_mtime_ns) == (
        original.st_ino,
        original.st_size,
        original.st_mtime_ns,
    )

    with pytest.raises(pytest.UsageError, match="Clang major mismatch"):
        analyzer_binary(environment=environment, repository_root=tmp_path)


def test_discovery_rejects_binary_changed_during_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _analyzer_script(tmp_path)

    class _MutatingClient:
        def __init__(self, _binary: Path, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 10

        def probe(self) -> None:
            _analyzer_script(tmp_path, clang_major=REQUIRED_CLANG_MAJOR - 1)

    monkeypatch.setattr(analyzer_discovery, "NativeAnalyzerClient", _MutatingClient)
    with pytest.raises(pytest.UsageError, match="changed while being validated"):
        analyzer_binary(
            environment={"CPP_CONTEXT_TEST_ANALYZER": str(candidate)},
            repository_root=tmp_path,
        )
