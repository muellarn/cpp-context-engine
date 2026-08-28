"""Authoritative discovery and validation for the native test companion."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from cpp_context_engine.ingestion import NativeAnalyzerClient

_ANALYZER_ENVIRONMENT = "CPP_CONTEXT_TEST_ANALYZER"
_DEFAULT_RELATIVE_PATH = Path("build/clang-analyzer/cpp-context-clang-analyzer")


@dataclass(frozen=True, slots=True)
class _BinaryIdentity:
    path: Path
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


_validated: set[_BinaryIdentity] = set()
_validation_lock = threading.Lock()


def _candidate(environment: Mapping[str, str], repository_root: Path) -> tuple[Path, str]:
    configured = environment.get(_ANALYZER_ENVIRONMENT)
    if configured:
        return Path(configured).expanduser(), _ANALYZER_ENVIRONMENT
    return repository_root / _DEFAULT_RELATIVE_PATH, "repository fallback"


def _identity(candidate: Path, source: str) -> _BinaryIdentity:
    try:
        if not candidate.is_file():
            raise pytest.UsageError(f"{source} does not name a regular file: {candidate}")
        resolved = candidate.resolve(strict=True)
        if os.name != "nt" and not os.access(resolved, os.X_OK):
            raise pytest.UsageError(f"{source} is not executable: {resolved}")
        metadata = resolved.stat()
    except pytest.UsageError:
        raise
    except OSError as error:
        raise pytest.UsageError(f"{source} cannot be inspected: {candidate}") from error
    return _BinaryIdentity(
        path=resolved,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def analyzer_binary(
    *,
    environment: Mapping[str, str] | None = None,
    repository_root: Path | None = None,
) -> Path:
    """Return one executable, protocol-compatible analyzer or fail the native suite."""

    selected_environment = os.environ if environment is None else environment
    selected_root = Path(__file__).parents[1] if repository_root is None else repository_root
    candidate, source = _candidate(selected_environment, selected_root)
    identity = _identity(candidate, source)
    with _validation_lock:
        if identity in _validated:
            return identity.path
        try:
            NativeAnalyzerClient(identity.path, timeout_seconds=10).probe()
        except RuntimeError as error:
            raise pytest.UsageError(f"{source} is incompatible: {error}") from error
        # A concurrently rebuilt binary must not inherit validation performed on
        # different bytes observed immediately before the handshake.
        if _identity(identity.path, source) != identity:
            raise pytest.UsageError(f"{source} changed while being validated: {identity.path}")
        _validated.add(identity)
    return identity.path


def clear_analyzer_discovery_cache() -> None:
    """Reset process-local validation state for isolated discovery tests."""

    with _validation_lock:
        _validated.clear()
