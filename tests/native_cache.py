"""Session-local cache for immutable real-companion fixture analyses."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypeVar

from cpp_context_engine.ingestion import AnalyzerLimitError, NativeAnalyzerClient
from cpp_context_engine.ingestion.native import (
    PROTOCOL,
    PROTOCOL_VERSION,
    REQUIRED_CLANG_MAJOR,
    AnalyzerInfo,
)
from cpp_context_engine.models import BuildConfiguration

_T = TypeVar("_T")
_RELEVANT_ENVIRONMENT = {
    "CC",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CXX",
    "C_INCLUDE_PATH",
    "LIBRARY_PATH",
    "MACOSX_DEPLOYMENT_TARGET",
    "PATH",
    "SDKROOT",
    "SYSROOT",
}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _project_digest(project_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in project_root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(project_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_file_digest(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _environment_identity() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in _RELEVANT_ENVIRONMENT or key.startswith("CPP_CONTEXT_")
    }


class NativeFixtureCache:
    """Store immutable pickles and deserialize a private result for every consumer."""

    def __init__(self) -> None:
        self._directory = Path(tempfile.mkdtemp(prefix="cpp-context-native-tests-", dir="/tmp"))
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._loads = 0

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def loads(self) -> int:
        return self._loads

    def load(self, key: str, factory: Callable[[], _T]) -> _T:
        artifact = self._directory / f"{key}.pickle"
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            if not artifact.is_file():
                value = factory()
                temporary = self._directory / f".{key}.{os.getpid()}.tmp"
                temporary.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
                temporary.chmod(0o444)
                temporary.replace(artifact)
                with self._lock:
                    self._loads += 1
            payload = artifact.read_bytes()
        # Each caller receives a fresh object graph, so mutable metadata in otherwise
        # frozen domain records cannot leak state into a later test.
        return pickle.loads(payload)  # noqa: S301 - trusted, session-owned test artifact

    def analysis_key(
        self,
        client: NativeAnalyzerClient,
        project_root: Path,
        configuration: BuildConfiguration,
    ) -> str:
        root = project_root.resolve(strict=True)
        binary = client.binary.resolve(strict=True)
        client_identity = {
            name: getattr(client, name)
            for name in (
                "timeout_seconds",
                "max_input_bytes",
                "max_output_bytes",
                "max_decoded_bytes",
                "max_record_bytes",
                "max_stderr_bytes",
                "prefer_compression",
            )
        }
        configuration_identity = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(configuration).items()
        }
        identity = {
            "analyzer": {"path": str(binary), "sha256": _file_digest(binary)},
            "protocol": {
                "name": PROTOCOL,
                "version": PROTOCOL_VERSION,
                "clang_major": REQUIRED_CLANG_MAJOR,
            },
            "client": client_identity,
            "configuration": configuration_identity,
            "environment": _environment_identity(),
            "project": {"path": str(root), "sha256": _project_digest(root)},
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return "facts-" + hashlib.sha256(encoded).hexdigest()

    def probe_key(self, client: NativeAnalyzerClient) -> str:
        binary = client.binary.resolve(strict=True)
        identity = {
            "analyzer": {"path": str(binary), "sha256": _file_digest(binary)},
            "protocol": {
                "name": PROTOCOL,
                "version": PROTOCOL_VERSION,
                "clang_major": REQUIRED_CLANG_MAJOR,
            },
            "transport": client.prefer_compression,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return "probe-" + hashlib.sha256(encoded).hexdigest()

    def stage(
        self, project_root: Path, configuration: BuildConfiguration
    ) -> tuple[Path, BuildConfiguration]:
        """Copy immutable NTFS fixtures to Linux temporary storage."""

        root = project_root.resolve(strict=True)
        try:
            source_relative = configuration.source_path.resolve(strict=True).relative_to(root)
            directory_relative = configuration.directory.resolve(strict=True).relative_to(root)
        except ValueError:
            # A configuration rooted outside the fixture is not safe to relocate.
            return root, configuration

        staged_root = self.stage_project(root)
        if staged_root == root:
            return root, configuration

        original_prefix = str(root)
        staged_prefix = str(staged_root)

        def relocate(value: str) -> str:
            if value == original_prefix or value.startswith(original_prefix + os.sep):
                return staged_prefix + value[len(original_prefix) :]
            return value

        output = configuration.output
        if output is not None:
            output = Path(relocate(str(output)))
        return staged_root, replace(
            configuration,
            source_path=staged_root / source_relative,
            directory=staged_root / directory_relative,
            arguments=tuple(relocate(argument) for argument in configuration.arguments),
            output=output,
        )

    def stage_project(self, project_root: Path) -> Path:
        root = project_root.resolve(strict=True)
        temporary_root = Path("/tmp").resolve()
        if root == temporary_root or temporary_root in root.parents:
            return root
        identity = hashlib.sha256((str(root) + "\0" + _project_digest(root)).encode()).hexdigest()
        staged_root = self._directory / f"project-{identity}"
        with self._lock:
            stage_lock = self._key_locks.setdefault(f"stage-{identity}", threading.Lock())
        with stage_lock:
            if not staged_root.is_dir():
                pending = self._directory / f".project-{identity}.{os.getpid()}.tmp"
                shutil.copytree(root, pending)
                pending.replace(staged_root)
        return staged_root

    def close(self) -> None:
        shutil.rmtree(self._directory, ignore_errors=True)


def _restore_fact_paths(value: Any, staged_root: Path, project_root: Path) -> Any:
    if isinstance(value, str):
        staged = str(staged_root)
        if value == staged or value.startswith(staged + os.sep):
            return str(project_root) + value[len(staged) :]
        return value
    if isinstance(value, list):
        return [_restore_fact_paths(item, staged_root, project_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_fact_paths(item, staged_root, project_root) for item in value)
    if isinstance(value, Mapping):
        return {
            key: _restore_fact_paths(item, staged_root, project_root) for key, item in value.items()
        }
    return value


class StagedNativeAnalyzerClient:
    """Run immutable fixtures on the Linux filesystem and restore original paths."""

    def __init__(self, cache: NativeFixtureCache, client: NativeAnalyzerClient) -> None:
        self._cache = cache
        self._client = client

    def probe(self, *, refresh: bool = False) -> AnalyzerInfo:
        return self._client.probe(refresh=refresh)

    def analyze(
        self, project_root: Path, configuration: BuildConfiguration
    ) -> tuple[Mapping[str, Any], ...]:
        facts: list[Mapping[str, Any]] = []
        self.analyze_stream(project_root, configuration, facts.append)
        return tuple(facts)

    def analyze_stream(
        self,
        project_root: Path,
        configuration: BuildConfiguration,
        on_fact: Callable[[Mapping[str, Any]], None],
        *,
        cancelled: threading.Event | None = None,
    ) -> None:
        root = project_root.resolve(strict=True)
        staged_root, staged_configuration = self._cache.stage(root, configuration)

        def restore(fact: Mapping[str, Any]) -> None:
            restored = _restore_fact_paths(fact, staged_root, root)
            if not isinstance(restored, Mapping):
                raise TypeError("restored analyzer fact must remain a mapping")
            on_fact(restored)

        self._client.analyze_stream(
            staged_root,
            staged_configuration,
            restore,
            cancelled=cancelled,
        )


class CachedNativeAnalyzerClient(StagedNativeAnalyzerClient):
    """Native client facade that reuses only successful immutable fixture facts."""

    def probe(self, *, refresh: bool = False) -> AnalyzerInfo:
        if refresh:
            return self._client.probe(refresh=True)
        info = self._cache.load(self._cache.probe_key(self._client), self._client.probe)
        # Avoid a redundant probe when this facade has to perform a cache-miss analysis.
        self._client._info = info  # noqa: SLF001 - transparent test-only facade
        return info

    def analyze(
        self, project_root: Path, configuration: BuildConfiguration
    ) -> tuple[Mapping[str, Any], ...]:
        key = self._cache.analysis_key(self._client, project_root, configuration)

        def analyze_fresh() -> tuple[Mapping[str, Any], ...]:
            self.probe()
            facts: list[Mapping[str, Any]] = []
            StagedNativeAnalyzerClient.analyze_stream(
                self, project_root, configuration, facts.append
            )
            return tuple(facts)

        return self._cache.load(key, analyze_fresh)

    def analyze_stream(
        self,
        project_root: Path,
        configuration: BuildConfiguration,
        on_fact: Callable[[Mapping[str, Any]], None],
        *,
        cancelled: threading.Event | None = None,
    ) -> None:
        for fact in self.analyze(project_root, configuration):
            if cancelled is not None and cancelled.is_set():
                raise AnalyzerLimitError("analyzer analysis was cancelled")
            on_fact(fact)


_SESSION_CACHE = NativeFixtureCache()


def cached_native_client(binary: Path, **kwargs: object) -> CachedNativeAnalyzerClient:
    return CachedNativeAnalyzerClient(_SESSION_CACHE, NativeAnalyzerClient(binary, **kwargs))


def fresh_native_client(binary: Path, **kwargs: object) -> StagedNativeAnalyzerClient:
    """Return a raw client while reusing only its immutable validated handshake."""

    client = NativeAnalyzerClient(binary, **kwargs)
    client._info = _SESSION_CACHE.load(  # noqa: SLF001 - test-only startup optimization
        _SESSION_CACHE.probe_key(client), client.probe
    )
    return StagedNativeAnalyzerClient(_SESSION_CACHE, client)


def staged_fixture(project_root: Path) -> Path:
    return _SESSION_CACHE.stage_project(project_root)


def close_native_fixture_cache() -> None:
    _SESSION_CACHE.close()
