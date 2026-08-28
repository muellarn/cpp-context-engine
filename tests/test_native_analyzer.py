from __future__ import annotations

import gc
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import threading
import time
import weakref
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary
from native_cache import (
    CachedNativeAnalyzerClient,
    NativeFixtureCache,
    cached_native_client,
    fresh_native_client,
    staged_fixture,
)

from cpp_context_engine.cli import main
from cpp_context_engine.ingestion import (
    AnalyzerLimitError,
    AnalyzerProtocolError,
    NativeAnalyzerClient,
    NativeClangIngestor,
)
from cpp_context_engine.ingestion.clang import ClangIngestor, ClangUnavailableError
from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.ingestion.indexer import ProjectIndexer
from cpp_context_engine.ingestion.native import (
    MAX_FACT_KINDS,
    AnalyzerInfo,
    _FactBatchBuilder,
    _FactRegistry,
    _ResourceBudget,
)
from cpp_context_engine.models import (
    BuildConfiguration,
    BuildScope,
    BuildVariant,
    CfgEdgeKind,
    GraphRelation,
    OccurrenceKind,
)
from cpp_context_engine.storage import SQLiteStore

FIXTURE = Path(__file__).parent / "fixtures" / "analyzer_project"
PARITY_FIXTURE = Path(__file__).parent / "fixtures" / "cpp_project"
CFG_FIXTURE = Path(__file__).parent / "fixtures" / "cfg_project"
IMPLICIT_FIXTURE = Path(__file__).parent / "fixtures" / "implicit_project"
TEMPLATE_DATAFLOW_FIXTURE = Path(__file__).parent / "fixtures" / "template_dataflow_project"
pytestmark = pytest.mark.native


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake-analyzer"
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fake_hello(*, gzip_transport: bool = False) -> dict[str, object]:
    capabilities = [
        "direct_calls",
        "full_ast",
        "function_cfg_v1",
        "includes",
        "inherits",
        "lambda_metadata",
        "macro_provenance",
        "occurrences",
        "overrides",
        "pp_callbacks",
        "source_manager",
        "symbols",
        "template_metadata",
        "uses_type",
        "callsites_v1",
        "dispatch_targets_v1",
        "macro_expansion_stack",
        "template_relationships_v1",
        "intraprocedural_dataflow_v1",
        "points_to_v1",
        "function_summaries_v1",
        "interprocedural_bindings_v1",
    ]
    if gzip_transport:
        capabilities.append("gzip_jsonl_v1")
    return {
        "type": "hello",
        "protocol": "cpp-context-clang-facts",
        "protocol_version": 5,
        "analyzer_version": "test",
        "clang_major": 18,
        "capabilities": capabilities,
    }


def _gzip_analyzer_script(
    tmp_path: Path,
    records: list[dict[str, object]],
    *,
    fragment_size: int = 1,
) -> Path:
    hello = _fake_hello(gzip_transport=True)
    return _script(
        tmp_path,
        f"""import json, os, sys, zlib
requests = [json.loads(line) for line in sys.stdin]
hello = {hello!r}
if len(requests) == 1:
    print(json.dumps(hello, separators=(",", ":")), flush=True)
else:
    request = requests[1]
    output = [hello, {{"type": "begin", "request_id": request["request_id"]}},
              *{records!r},
              {{"type": "complete", "request_id": request["request_id"], "success": True}}]
    raw = b"".join(json.dumps(record, separators=(",", ":"), sort_keys=True).encode() + b"\\n"
                   for record in output)
    compressor = zlib.compressobj(level=1, wbits=31)
    encoded = compressor.compress(raw) + compressor.flush()
    for offset in range(0, len(encoded), {fragment_size}):
        os.write(1, encoded[offset:offset + {fragment_size}])
""",
    )


def test_fact_builder_and_native_cache_cover_all_semantic_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.cpp"
    source.write_text("int value = 1;\n", encoding="utf-8")
    configuration = BuildConfiguration(
        id="build",
        source_path=source,
        directory=tmp_path,
        arguments=("clang++", str(source)),
        command_hash="hash",
    )
    builder = _FactBatchBuilder(tmp_path, configuration)

    first = builder._path(str(source))  # noqa: SLF001 - verify the path hot-loop cache
    second = builder._path(str(source))  # noqa: SLF001 - verify the path hot-loop cache

    assert first is second

    binary = _script(tmp_path, "")
    cache = NativeFixtureCache()
    try:
        client = NativeAnalyzerClient(binary)
        baseline = cache.analysis_key(client, tmp_path, configuration)
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        assert cache.analysis_key(NativeAnalyzerClient(binary), tmp_path, configuration) != baseline
        binary.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        source.write_text("int changed = 1;\n", encoding="utf-8")
        assert cache.analysis_key(NativeAnalyzerClient(binary), tmp_path, configuration) != baseline
        source.write_text("int value = 1;\n", encoding="utf-8")
        original_cache_identity = os.getenv("CPP_CONTEXT_TEST_CACHE_IDENTITY")
        monkeypatch.setenv(
            "CPP_CONTEXT_TEST_CACHE_IDENTITY", f"{original_cache_identity}-changed-for-key-test"
        )
        assert cache.analysis_key(NativeAnalyzerClient(binary), tmp_path, configuration) != baseline
        if original_cache_identity is None:
            monkeypatch.delenv("CPP_CONTEXT_TEST_CACHE_IDENTITY")
        else:
            monkeypatch.setenv("CPP_CONTEXT_TEST_CACHE_IDENTITY", original_cache_identity)
        assert (
            cache.analysis_key(
                NativeAnalyzerClient(binary, prefer_compression=False), tmp_path, configuration
            )
            != baseline
        )
        assert (
            cache.analysis_key(
                NativeAnalyzerClient(binary),
                tmp_path,
                replace(configuration, command_hash="changed"),
            )
            != baseline
        )
        probe_baseline = cache.probe_key(NativeAnalyzerClient(binary))
        for client_kwargs in (
            {"timeout_seconds": 76},
            {"max_input_bytes": 1_048_577},
            {"max_output_bytes": 67_108_865},
            {"max_decoded_bytes": 268_435_457},
            {"max_record_bytes": 16_777_217},
            {"max_stderr_bytes": 262_145},
        ):
            limited = NativeAnalyzerClient(binary, **client_kwargs)
            assert cache.probe_key(limited) != probe_baseline
            assert cache.analysis_key(limited, tmp_path, configuration) != baseline

        monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/review-clang-runtime")
        environment_client = NativeAnalyzerClient(binary)
        assert cache.probe_key(environment_client) != probe_baseline
        assert cache.analysis_key(environment_client, tmp_path, configuration) != baseline
    finally:
        cache.close()


def test_native_cache_remaps_embedded_project_paths() -> None:
    root = FIXTURE.resolve()
    configuration = CompilationDatabase.load(root / "compile_commands.json").configurations[0]
    include = root / "include"
    sysroot = root / "sdk"
    cache = NativeFixtureCache()
    try:
        staged_root, staged = cache.stage(
            root,
            replace(
                configuration,
                arguments=(
                    *configuration.arguments,
                    f"-I{include}",
                    f"--sysroot={sysroot}",
                ),
            ),
        )
        assert staged_root != root
        assert f"-I{staged_root / 'include'}" in staged.arguments
        assert f"--sysroot={staged_root / 'sdk'}" in staged.arguments
        assert all(str(root) not in argument for argument in staged.arguments)
    finally:
        cache.close()


def test_fact_registry_bounds_and_cached_consumers_are_isolated(tmp_path: Path) -> None:
    with _FactRegistry() as facts:
        for index in range(MAX_FACT_KINDS):
            facts.add({"fact": f"kind-{index}"})

        with pytest.raises(AnalyzerLimitError, match="fact-kind registry limit"):
            facts.add({"fact": "one-kind-too-many"})

    cache = NativeFixtureCache()
    calls = 0

    def create() -> dict[str, list[str]]:
        nonlocal calls
        calls += 1
        return {"values": ["original"]}

    first = cache.load("identity", create)
    first["values"].append("mutation")
    second = cache.load("identity", create)
    artifact = cache.directory / "identity.pickle"
    assert calls == cache.loads == 1
    assert second == {"values": ["original"]}
    assert artifact.stat().st_mode & stat.S_IWUSR == 0

    concurrent_started = threading.Event()
    release_concurrent = threading.Event()
    concurrent_calls = 0
    concurrent_results: list[dict[str, list[str]]] = []

    def create_concurrently() -> dict[str, list[str]]:
        nonlocal concurrent_calls
        concurrent_calls += 1
        concurrent_started.set()
        assert release_concurrent.wait(timeout=2)
        return {"values": ["shared-source"]}

    workers = [
        threading.Thread(
            target=lambda: concurrent_results.append(
                cache.load("concurrent-identity", create_concurrently)
            )
        )
        for _ in range(2)
    ]
    workers[0].start()
    assert concurrent_started.wait(timeout=2)
    workers[1].start()
    release_concurrent.set()
    for concurrent_worker in workers:
        concurrent_worker.join(timeout=2)
        assert not concurrent_worker.is_alive()
    assert concurrent_calls == 1
    assert concurrent_results[0] == concurrent_results[1]
    assert concurrent_results[0] is not concurrent_results[1]
    concurrent_results[0]["values"].append("private-mutation")
    assert concurrent_results[1] == {"values": ["shared-source"]}

    source = tmp_path / "cached.cpp"
    source.write_text("int cached;\n", encoding="utf-8")
    configuration = BuildConfiguration(
        id="cached",
        source_path=source,
        directory=tmp_path,
        arguments=("clang++", "-c", str(source)),
        command_hash="cached",
    )

    class CountingClient(NativeAnalyzerClient):
        analyses = 0

        def probe(self, *, refresh: bool = False) -> AnalyzerInfo:
            return AnalyzerInfo("test", 5, "test", 18, frozenset())

        def analyze_stream(
            self,
            _project_root: Path,
            _configuration: BuildConfiguration,
            on_fact,
            *,
            cancelled=None,
        ) -> None:
            self.analyses += 1
            on_fact({"fact": "test", "payload": ["original"]})

    client = CountingClient(_script(tmp_path, ""))
    cached = CachedNativeAnalyzerClient(cache, client)
    result: list[tuple] = []
    worker = threading.Thread(target=lambda: result.append(cached.analyze(tmp_path, configuration)))
    worker.daemon = True
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive(), "cached analysis recursively reacquired its own key lock"
    result[0][0]["payload"].append("mutation")
    repeated = cached.analyze(tmp_path, configuration)
    assert client.analyses == 1
    assert repeated[0]["payload"] == ["original"]

    directory = cache.directory
    cache.close()
    assert not directory.exists()


def test_fact_registry_does_not_decode_validated_facts_as_json_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = 0
    original_loads = json.loads

    def counted_loads(value: object, *args: object, **kwargs: object) -> object:
        nonlocal decoded
        decoded += 1
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr("cpp_context_engine.ingestion.native.json.loads", counted_loads)
    fact = {"type": "fact", "fact": "symbol", "key": "usr:test", "nested": [1, True]}

    with _FactRegistry() as facts:
        facts.add(fact)
        assert list(facts.records("symbol")) == [fact]

    assert decoded == 0


def test_fact_registry_enforces_and_releases_shared_spool_limits() -> None:
    byte_budget = _ResourceBudget(4096, "byte budget")
    fd_budget = _ResourceBudget(1, "file budget")
    facts = _FactRegistry(
        max_bytes=4096,
        max_record_bytes=4096,
        byte_budget=byte_budget,
        fd_budget=fd_budget,
    )

    facts.add({"fact": "symbol", "key": "usr:test"})
    assert byte_budget.used > 0
    assert fd_budget.used == 1
    with pytest.raises(AnalyzerLimitError, match="file budget"):
        facts.add({"fact": "edge", "key": "edge:test"})

    facts.close()
    assert byte_budget.used == 0
    assert fd_budget.used == 0

    with (
        _FactRegistry(max_bytes=8, max_record_bytes=4096) as bounded,
        pytest.raises(AnalyzerLimitError, match="registry spool limit"),
    ):
        bounded.add({"fact": "symbol", "key": "too-large"})


@pytest.mark.parametrize(
    "limits",
    [
        {"max_spool_registries": 0},
        {"max_spool_bytes": 0},
        {"max_spool_fds": 0},
    ],
)
def test_native_pipeline_rejects_explicit_zero_spool_limits(limits: dict[str, int]) -> None:
    class EmptyClient:
        max_decoded_bytes = 1024

    with pytest.raises(ValueError, match="limits must be positive"):
        NativeClangIngestor(EmptyClient(), **limits)  # type: ignore[arg-type]


def test_native_configurations_are_analyzed_concurrently_in_input_order(tmp_path: Path) -> None:
    active = 0
    maximum_active = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    class ConcurrentClient:
        def probe(self) -> object:
            return object()

        def analyze(
            self, _root: Path, configuration: BuildConfiguration
        ) -> list[dict[str, object]]:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            barrier.wait(timeout=2)
            with lock:
                active -= 1
            return []

    configurations = []
    for index in range(3):
        source = tmp_path / f"source-{index}.cpp"
        source.write_text(f"int value_{index} = {index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
                build_variant=f"variant-{index}",
            )
        )

    batch = NativeClangIngestor(  # type: ignore[arg-type]
        ConcurrentClient(), max_workers=3
    ).ingest_configurations(tmp_path, configurations)

    assert maximum_active == 3
    assert [item.build_configuration_id for item in batch.translation_units] == [
        "build-0",
        "build-1",
        "build-2",
    ]
    assert [item.build_variant for item in batch.translation_units] == [
        "variant-0",
        "variant-1",
        "variant-2",
    ]


def test_native_configuration_batches_refill_on_completion_with_a_bounded_window(
    tmp_path: Path,
) -> None:
    started: list[int] = []
    third_started = threading.Event()
    release_first = threading.Event()
    lock = threading.Lock()

    class OutOfOrderClient:
        def probe(self) -> object:
            return object()

        def analyze(
            self, _root: Path, configuration: BuildConfiguration
        ) -> list[dict[str, object]]:
            index = int(configuration.id.removeprefix("build-"))
            with lock:
                started.append(index)
            if index == 0:
                assert third_started.wait(timeout=2)
                release_first.set()
            elif index == 2:
                third_started.set()
            return []

    configurations = []
    for index in range(8):
        source = tmp_path / f"ordered-{index}.cpp"
        source.write_text(f"int ordered_{index} = {index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
            )
        )

    batches = NativeClangIngestor(  # type: ignore[arg-type]
        OutOfOrderClient(), max_workers=2, max_spool_registries=3
    ).iter_configuration_batches(tmp_path, configurations)
    first = next(batches)

    assert release_first.is_set()
    assert [unit.build_configuration_id for unit in first.translation_units] == ["build-0"]
    assert 2 in started
    # The slow ordered item plus completed/refilled work must never exceed the
    # configured registry window while the consumer retains the first batch.
    assert len(started) <= 3
    assert [batch.translation_units[0].build_configuration_id for batch in batches] == [
        f"build-{index}" for index in range(1, 8)
    ]


def test_native_pipeline_bounds_converted_batches_retained_by_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[int] = []
    built_two = threading.Event()
    lock = threading.Lock()
    original_build = _FactBatchBuilder.build

    def counted_build(builder: _FactBatchBuilder, facts: object) -> object:
        result = original_build(builder, facts)  # type: ignore[arg-type]
        with lock:
            built.append(int(builder.configuration.id.removeprefix("build-")))
            if len(built) == 2:
                built_two.set()
        return result

    monkeypatch.setattr(_FactBatchBuilder, "build", counted_build)

    class EmptyClient:
        def probe(self) -> object:
            return object()

        def analyze(
            self, _root: Path, _configuration: BuildConfiguration
        ) -> list[dict[str, object]]:
            return []

    configurations = []
    for index in range(6):
        source = tmp_path / f"domain-{index}.cpp"
        source.write_text(f"int domain_{index} = {index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
            )
        )

    batches = NativeClangIngestor(  # type: ignore[arg-type]
        EmptyClient(),
        max_workers=2,
        max_spool_registries=4,
        max_domain_batches=2,
    ).iter_configuration_batches(tmp_path, configurations)
    next(batches)
    assert built_two.wait(timeout=2)
    time.sleep(0.05)

    assert sorted(built) == [0, 1]
    batches.close()


def test_native_pipeline_does_not_retain_every_completed_analysis_future(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analyzer_futures: list[weakref.ReferenceType[object]] = []

    class TrackingExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._delegate = ThreadPoolExecutor(*args, **kwargs)  # type: ignore[arg-type]
            self._track = kwargs.get("thread_name_prefix") == "cpp-context-analyzer"

        def submit(self, *args: object, **kwargs: object) -> object:
            future = self._delegate.submit(*args, **kwargs)  # type: ignore[arg-type]
            if self._track:
                analyzer_futures.append(weakref.ref(future))
            return future

        def shutdown(self, *args: object, **kwargs: object) -> None:
            self._delegate.shutdown(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("cpp_context_engine.ingestion.native.ThreadPoolExecutor", TrackingExecutor)

    class EmptyClient:
        def probe(self) -> object:
            return object()

        def analyze(
            self, _root: Path, _configuration: BuildConfiguration
        ) -> list[dict[str, object]]:
            return []

    configurations = []
    for index in range(80):
        source = tmp_path / f"future-{index}.cpp"
        source.write_text(f"int future_{index} = {index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
            )
        )

    batches = NativeClangIngestor(  # type: ignore[arg-type]
        EmptyClient(), max_workers=2, max_spool_registries=4
    ).iter_configuration_batches(tmp_path, configurations)
    for _index in range(40):
        next(batches)
    gc.collect()

    assert sum(reference() is not None for reference in analyzer_futures) <= 4
    batches.close()


def test_native_pipeline_does_not_convert_past_a_missing_ordered_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_first = threading.Event()
    later_analyzed = threading.Event()
    later_converted = threading.Event()
    finished = threading.Event()
    outcomes: list[object] = []
    original_build = _FactBatchBuilder.build

    def observe_build(builder: _FactBatchBuilder, facts: object) -> object:
        if builder.configuration.id == "build-1":
            later_converted.set()
        return original_build(builder, facts)  # type: ignore[arg-type]

    monkeypatch.setattr(_FactBatchBuilder, "build", observe_build)

    class OrderedClient:
        def probe(self) -> object:
            return object()

        def analyze(
            self, _root: Path, configuration: BuildConfiguration
        ) -> list[dict[str, object]]:
            if configuration.id == "build-0":
                assert release_first.wait(timeout=5)
            else:
                later_analyzed.set()
            return []

    configurations = []
    for index in range(2):
        source = tmp_path / f"ordered-conversion-{index}.cpp"
        source.write_text(f"int ordered_conversion_{index} = {index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
            )
        )

    batches = NativeClangIngestor(  # type: ignore[arg-type]
        OrderedClient(), max_workers=2, max_domain_batches=2
    ).iter_configuration_batches(tmp_path, configurations)

    def consume_first() -> None:
        try:
            outcomes.append(next(batches))
        except BaseException as error:  # pragma: no cover - surfaced below
            outcomes.append(error)
        finally:
            finished.set()

    consumer = threading.Thread(target=consume_first)
    consumer.start()
    assert later_analyzed.wait(timeout=2)
    overtook = later_converted.wait(timeout=1)
    release_first.set()
    assert finished.wait(timeout=5)
    consumer.join(timeout=1)
    batches.close()

    assert not overtook
    assert len(outcomes) == 1
    assert not isinstance(outcomes[0], BaseException)


def test_closing_native_configuration_batches_cancels_pending_workers(tmp_path: Path) -> None:
    pending_started = threading.Event()
    pending_cancelled = threading.Event()

    class CancellableClient:
        def probe(self) -> object:
            return object()

        def analyze_stream(
            self,
            _root: Path,
            configuration: BuildConfiguration,
            _consume: object,
            *,
            cancelled: threading.Event,
        ) -> None:
            if configuration.id == "build-0":
                return
            pending_started.set()
            if cancelled.wait(timeout=2):
                pending_cancelled.set()

    configurations = []
    for index in range(2):
        source = tmp_path / f"cancel-{index}.cpp"
        source.write_text(f"int cancel_{index} = {index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
            )
        )

    batches = NativeClangIngestor(  # type: ignore[arg-type]
        CancellableClient(), max_workers=2
    ).iter_configuration_batches(tmp_path, configurations)
    next(batches)
    assert pending_started.wait(timeout=2)
    batches.close()

    assert pending_cancelled.is_set()


def test_later_native_worker_failure_cancels_the_slow_ordered_worker(tmp_path: Path) -> None:
    slow_started = threading.Event()
    slow_cancelled = threading.Event()

    class OutOfOrderFailingClient:
        def probe(self) -> object:
            return object()

        def analyze_stream(
            self,
            _root: Path,
            configuration: BuildConfiguration,
            _consume: object,
            *,
            cancelled: threading.Event,
        ) -> None:
            if configuration.id == "build-0":
                slow_started.set()
                if cancelled.wait(timeout=2):
                    slow_cancelled.set()
                return
            assert slow_started.wait(timeout=2)
            raise RuntimeError("injected later worker failure")

    configurations = []
    for index in range(2):
        source = tmp_path / f"later-failure-{index}.cpp"
        source.write_text(f"int later_failure_{index} = {index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
            )
        )

    batches = NativeClangIngestor(  # type: ignore[arg-type]
        OutOfOrderFailingClient(), max_workers=2
    ).iter_configuration_batches(tmp_path, configurations)
    with pytest.raises(RuntimeError, match="injected later worker failure"):
        next(batches)

    assert slow_cancelled.is_set()


def test_native_worker_failure_cancels_pending_work_without_partial_batch(tmp_path: Path) -> None:
    started: list[int] = []
    lock = threading.Lock()
    workers_ready = threading.Barrier(7)

    class FailingClient:
        def probe(self) -> object:
            return object()

        def analyze(
            self, _root: Path, configuration: BuildConfiguration
        ) -> list[dict[str, object]]:
            index = int(configuration.id.removeprefix("build-"))
            with lock:
                started.append(index)
            workers_ready.wait(timeout=2)
            if index == 0:
                raise RuntimeError("injected worker failure")
            time.sleep(0.2)
            return []

    configurations = []
    for index in range(10):
        source = tmp_path / f"failure-{index}.cpp"
        source.write_text(f"int failure_{index} = {index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
            )
        )

    with pytest.raises(RuntimeError, match="injected worker failure"):
        NativeClangIngestor(  # type: ignore[arg-type]
            FailingClient(), max_workers=7
        ).ingest_configurations(tmp_path, configurations)

    assert sorted(started) == list(range(7))


def test_native_handshake_matches_protocol_golden() -> None:
    request = {
        "type": "hello",
        "protocol": "cpp-context-clang-facts",
        "protocol_version": 5,
        "required_clang_major": 18,
    }
    completed = subprocess.run(  # noqa: S603 - repository-built test binary
        [analyzer_binary()],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "analyzer_protocol" / "hello.json").read_text()
    )
    assert json.loads(completed.stdout) == expected
    assert completed.stderr == ""


def test_real_companion_finalizes_gzip_when_rejecting_request() -> None:
    hello = {
        "type": "hello",
        "protocol": "cpp-context-clang-facts",
        "protocol_version": 5,
        "required_clang_major": 18,
        "required_capabilities": [],
        "response_transport": "gzip_jsonl_v1",
    }
    malformed_analyze = {"type": "analyze"}
    completed = subprocess.run(
        [analyzer_binary()],
        input=(json.dumps(hello) + "\n" + json.dumps(malformed_analyze) + "\n").encode(),
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    records = [json.loads(line) for line in zlib.decompress(completed.stdout, 31).splitlines()]
    assert [record["type"] for record in records] == ["hello", "error"]
    assert records[-1]["code"] == "invalid_request"


def test_native_plain_and_gzip_create_identical_deterministic_batches() -> None:
    gzip_client = fresh_native_client(analyzer_binary(), timeout_seconds=30)
    plain_client = fresh_native_client(
        analyzer_binary(), timeout_seconds=30, prefer_compression=False
    )

    gzip_batch = NativeClangIngestor(gzip_client).ingest(FIXTURE, FIXTURE / "compile_commands.json")
    plain_batch = NativeClangIngestor(plain_client).ingest(
        FIXTURE, FIXTURE / "compile_commands.json"
    )
    repeated = NativeClangIngestor(gzip_client).ingest(FIXTURE, FIXTURE / "compile_commands.json")

    assert gzip_batch == plain_batch == repeated


def test_implicit_lambda_copy_has_no_orphan_symbol_references() -> None:
    configuration = CompilationDatabase.load(
        IMPLICIT_FIXTURE / "compile_commands.json"
    ).configurations[0]
    client = fresh_native_client(analyzer_binary(), timeout_seconds=30)
    facts = client.analyze(IMPLICIT_FIXTURE, configuration)
    repeated = client.analyze(IMPLICIT_FIXTURE, configuration)
    endpoint_keys = {fact["key"] for fact in facts if fact.get("fact") in {"file", "symbol"}}
    unknown_references = {
        key
        for fact in facts
        for field in ("symbol_key", "enclosing_key", "source_key", "target_key")
        if isinstance((key := fact.get(field)), str) and key and key not in endpoint_keys
    }
    skipped_implicit_parameter = "fallback:variable:src/implicit.cpp::11:14"

    assert facts == repeated
    assert skipped_implicit_parameter not in endpoint_keys
    assert unknown_references == set()
    _FactBatchBuilder(IMPLICIT_FIXTURE.resolve(), configuration).build(facts)


def test_reopened_namespace_preserves_each_definition_occurrence() -> None:
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]
    facts = cached_native_client(analyzer_binary(), timeout_seconds=30).analyze(
        FIXTURE, configuration
    )

    namespace_paths = {
        Path(fact["span"]["path"]).relative_to(FIXTURE.resolve()).as_posix()
        for fact in facts
        if fact.get("fact") == "occurrence"
        and fact.get("symbol_key") == "usr:c:@N@analyzer_fixture"
        and fact.get("kind") == "definition"
    }
    namespace_container_files = {
        fact["source_key"]
        for fact in facts
        if fact.get("fact") == "edge"
        and fact.get("relation") == "contains"
        and fact.get("target_key") == "usr:c:@N@analyzer_fixture"
    }

    assert namespace_paths == {"include/analysis.hpp", "src/analysis.cpp"}
    assert namespace_container_files == {
        "file:include/analysis.hpp",
        "file:src/analysis.cpp",
    }


def test_real_ast_macro_template_lambda_and_relationship_facts() -> None:
    batch = _cached_batch(FIXTURE)
    relations = {edge.relation for edge in batch.edges}

    assert {
        GraphRelation.CALLS,
        GraphRelation.INCLUDES,
        GraphRelation.INHERITS,
        GraphRelation.OVERRIDES,
        GraphRelation.REFERENCES,
        GraphRelation.USES_TYPE,
    } <= relations
    macro = next(symbol for symbol in batch.symbols if symbol.qualified_name == "APPLY_TWICE")
    expansions = [
        occurrence
        for occurrence in batch.occurrences
        if occurrence.symbol_id == macro.id and occurrence.kind == OccurrenceKind.MACRO_EXPANSION
    ]
    assert len(expansions) == 2
    assert any(
        occurrence.metadata["spelling_span"] != occurrence.metadata["expansion_span"]
        for occurrence in expansions
    )
    templates = [symbol for symbol in batch.symbols if "template_kind" in symbol.metadata]
    assert templates
    assert any(symbol.metadata["template_arguments"] for symbol in templates)
    lambdas = [
        symbol for symbol in batch.symbols if symbol.metadata.get("is_lambda_call_operator") is True
    ]
    assert len(lambdas) == 1
    assert lambdas[0].metadata["stable_lambda_key"].startswith("lambda:")
    assert all(symbol.metadata["advanced_facts_complete"] for symbol in batch.symbols)

    definition = next(
        symbol
        for symbol in batch.symbols
        if symbol.qualified_name == "analyzer_fixture::Derived::evaluate"
        and symbol.metadata["is_definition"]
    )
    source = definition.span.path.read_bytes()
    assert (
        source[
            definition.metadata["start_offset"] : definition.metadata["end_offset_exclusive"]
        ].decode()
        == definition.source_text
    )

    constructor = next(
        symbol
        for symbol in batch.symbols
        if symbol.qualified_name == "analyzer_fixture::Derived::Derived"
    )
    assert any(
        occurrence.symbol_id == constructor.id and occurrence.kind == OccurrenceKind.CALL
        for occurrence in batch.occurrences
    )


def _cfg_batch(*, database: str = "compile_commands.json", variant: str = "default"):
    return NativeClangIngestor(fresh_native_client(analyzer_binary(), timeout_seconds=30)).ingest(
        CFG_FIXTURE, CFG_FIXTURE / database, build_variant=variant
    )


def _cached_batch(
    fixture: Path,
    *,
    database: str = "compile_commands.json",
    variant: str = "default",
):
    return NativeClangIngestor(cached_native_client(analyzer_binary(), timeout_seconds=30)).ingest(
        fixture, fixture / database, build_variant=variant
    )


def _cfg_for(batch, qualified_name: str):
    symbol = next(symbol for symbol in batch.symbols if symbol.qualified_name == qualified_name)
    return next(graph for graph in batch.cfg_graphs if graph.function_symbol_id == symbol.id)


def _normalized_fact_multiset(facts):
    return Counter(json.dumps(fact, sort_keys=True, separators=(",", ":")) for fact in facts)


def _solution_hashes(batch):
    return {
        summary.function_symbol_id: summary.solution_hash for summary in batch.function_summaries
    }


def _semantic_analysis_rows(store):
    tables = (
        "cfg_graphs",
        "cfg_blocks",
        "cfg_elements",
        "cfg_edges",
        "callsites",
        "call_targets",
        "data_flow_analyses",
        "memory_locations",
        "data_accesses",
        "data_flow_evidence",
        "function_summaries",
        "summary_effects",
        "summary_return_origins",
        "call_argument_bindings",
        "call_result_bindings",
        "interprocedural_flows",
    )
    return {
        table: tuple(
            tuple(row)
            for row in store._connection.execute(  # noqa: SLF001
                f"SELECT * FROM {table} ORDER BY id"  # noqa: S608 - fixed test table names
            )
        )
        for table in tables
    }


def test_template_specialization_facts_are_exactly_deterministic_across_processes(
    tmp_path: Path,
) -> None:
    configuration = CompilationDatabase.load(
        TEMPLATE_DATAFLOW_FIXTURE / "compile_commands.json"
    ).configurations[0]
    runs = [
        tuple(
            fresh_native_client(analyzer_binary(), timeout_seconds=30).analyze(
                TEMPLATE_DATAFLOW_FIXTURE, configuration
            )
        )
        for _ in range(4)
    ]
    normalized = [_normalized_fact_multiset(facts) for facts in runs]
    assert all(facts == normalized[0] for facts in normalized[1:])

    batches = [
        NativeClangIngestor._merge_batches(  # noqa: SLF001 - exact raw-to-domain regression
            (_FactBatchBuilder(TEMPLATE_DATAFLOW_FIXTURE.resolve(), configuration).build(facts),)
        )
        for facts in runs
    ]
    assert all(batch == batches[0] for batch in batches[1:])
    assert all(_solution_hashes(batch) == _solution_hashes(batches[0]) for batch in batches[1:])
    assert all(_solution_hashes(batches[0]).values())

    database_rows = []
    for index, batch in enumerate(batches[:2]):
        database = tmp_path / f"index-{index}.db"
        with SQLiteStore(database, project_root=TEMPLATE_DATAFLOW_FIXTURE) as store:
            store.apply_ingestion(TEMPLATE_DATAFLOW_FIXTURE, batch)
            database_rows.append(_semantic_analysis_rows(store))
    assert database_rows[0] == database_rows[1]


def test_indirect_target_keeps_an_indexed_redeclaration_after_an_external_declaration(
    tmp_path: Path,
) -> None:
    external_header = tmp_path / "external.hpp"
    external_header.write_text("int external_target(int);\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text(
        '#include "../external.hpp"\n'
        "int external_target(int);\n"
        "int call_external(int value) {\n"
        "  auto target = &external_target;\n"
        "  return target(value);\n"
        "}\n",
        encoding="utf-8",
    )
    database = project / "compile_commands.json"
    database.write_text(
        json.dumps(
            [
                {
                    "directory": str(project),
                    "file": str(source),
                    "arguments": ["clang++", "-std=c++20", "-c", str(source)],
                }
            ]
        ),
        encoding="utf-8",
    )
    configuration = CompilationDatabase.load(database).configurations[0]
    facts = fresh_native_client(analyzer_binary(), timeout_seconds=30).analyze(
        project, configuration
    )
    target_key = next(
        fact["key"]
        for fact in facts
        if fact.get("fact") == "symbol" and fact.get("qualified_name") == "external_target"
    )

    assert any(
        target_key in fact.get("pointee_keys", ())
        for fact in facts
        if fact.get("fact") == "data_access_v1"
    )
    assert any(
        fact.get("target_key") == target_key
        for fact in facts
        if fact.get("fact") == "call_target_v1"
    )
    _FactBatchBuilder(project.resolve(), configuration).build(facts)


@pytest.fixture(scope="module")
def deterministic_cfg_batches():
    """Two genuinely fresh analyses shared by the CFG determinism assertions."""

    return _cfg_batch(), _cfg_batch()


def test_real_cfg_snapshot_covers_control_flow_macro_and_lifetime_facts() -> None:
    batch = _cached_batch(CFG_FIXTURE)
    snapshot = {}
    for graph in batch.cfg_graphs:
        symbol = next(symbol for symbol in batch.symbols if symbol.id == graph.function_symbol_id)
        blocks = [block for block in batch.cfg_blocks if block.graph_id == graph.id]
        edges = [edge for edge in batch.cfg_edges if edge.graph_id == graph.id]
        snapshot[symbol.qualified_name] = (
            len(blocks),
            sum(not block.reachable for block in blocks),
            Counter(edge.kind.value for edge in edges),
        )

    assert snapshot["cfg_fixture::branching"] == (
        6,
        0,
        Counter({"fallthrough": 3, "true": 1, "false": 1, "return": 1}),
    )
    assert snapshot["cfg_fixture::choose"] == (
        6,
        0,
        Counter({"return": 3, "case": 2, "default": 1, "fallthrough": 1}),
    )
    assert snapshot["cfg_fixture::loop"] == (
        11,
        0,
        Counter(
            {
                "true": 3,
                "false": 3,
                "fallthrough": 3,
                "break": 1,
                "continue": 1,
                "loop_back": 1,
                "return": 1,
            }
        ),
    )
    assert snapshot["cfg_fixture::early_return"][2]["return"] == 2
    assert snapshot["cfg_fixture::jump"][2]["goto"] == 1
    assert snapshot["cfg_fixture::exception_flow"][2]["exception"] == 3
    assert snapshot["cfg_fixture::unreachable_after_return"][1] == 1

    all_edge_kinds = {edge.kind for edge in batch.cfg_edges}
    assert set(CfgEdgeKind) <= all_edge_kinds
    graph = _cfg_for(batch, "cfg_fixture::branching")
    blocks = [block for block in batch.cfg_blocks if block.graph_id == graph.id]
    assert {block.id for block in blocks if block.role.value == "entry"} == {graph.entry_block_id}
    assert {block.id for block in blocks if block.role.value == "normal_exit"} == {
        graph.normal_exit_block_id
    }
    assert graph.exceptional_exit_block_id is None
    assert graph.build_options == {
        "add_cxx_default_init_expr_in_aggregates": True,
        "add_cxx_default_init_expr_in_ctors": True,
        "add_cxx_new_allocator": True,
        "add_eh_edges": True,
        "add_implicit_dtors": True,
        "add_initializers": True,
        "add_lifetime": True,
        "add_loop_exit": True,
        "add_rich_cxx_constructors": True,
        "add_scopes": True,
        "add_static_init_branches": True,
        "add_temporary_dtors": True,
        "add_virtual_base_branches": True,
        "always_add_all_statements": True,
        "mark_elided_cxx_constructors": True,
        "omit_implicit_value_initializers": False,
        "prune_trivially_false_edges": False,
    }
    branch_elements = [element for element in batch.cfg_elements if element.graph_id == graph.id]
    assert any(
        element.spelling_span != element.expansion_span
        for element in branch_elements
        if element.spelling_span and element.expansion_span
    )
    lifetime = _cfg_for(batch, "cfg_fixture::lifetime")
    lifetime_kinds = {
        element.kind for element in batch.cfg_elements if element.graph_id == lifetime.id
    }
    assert {"constructor", "automatic_object_destructor", "lifetime_end"} <= lifetime_kinds
    assert all(
        item.translation_unit_id == graph.translation_unit_id
        and item.build_configuration_id == graph.build_configuration_id
        and item.build_variant == graph.build_variant
        for item in (
            *blocks,
            *branch_elements,
            *(edge for edge in batch.cfg_edges if edge.graph_id == graph.id),
        )
    )


def test_cfg_macro_expression_ranges_are_ordered_and_keep_expansion_evidence(
    deterministic_cfg_batches,
) -> None:
    first, second = deterministic_cfg_batches
    graph = _cfg_for(first, "cfg_fixture::local_object_macro_ranges")
    elements = [element for element in first.cfg_elements if element.graph_id == graph.id]
    target_text = {
        "LOCAL_LEFT_OPERAND + 1",
        "value - LOCAL_CHUNK_SIZE",
        "value += LOCAL_CHUNK_SIZE",
    }
    macro_expressions = [
        element
        for element in elements
        if element.statement_class in {"BinaryOperator", "CompoundAssignOperator"}
        and element.text in target_text
    ]

    assert len(macro_expressions) == 3
    assert all(element.spelling_span is None for element in macro_expressions)
    assert all(element.expansion_span is not None for element in macro_expressions)
    assert all(
        (span.end_line, span.end_column) >= (span.start_line, span.start_column)
        for element in first.cfg_elements
        for span in (element.spelling_span, element.expansion_span)
        if span is not None
    )
    assert first == second


def test_mixed_macro_spelling_range_keeps_callsite_with_expansion_evidence(
    tmp_path: Path,
) -> None:
    batch = _cached_batch(CFG_FIXTURE)
    owner = next(
        symbol
        for symbol in batch.symbols
        if symbol.qualified_name == "cfg_fixture::local_object_macro_call_range"
    )
    callsites = [site for site in batch.callsites if site.owner_symbol_id == owner.id]

    assert len(callsites) == 1
    assert callsites[0].callee_text == "LOCAL_CALL_TARGET(value)"
    assert callsites[0].spelling_span is None
    assert callsites[0].expansion_span is not None

    with SQLiteStore(tmp_path / "index.db", project_root=CFG_FIXTURE) as store:
        store.apply_ingestion(CFG_FIXTURE, batch)
        assert store.get_callsite(callsites[0].id) == callsites[0]


def test_cfg_ids_are_deterministic_and_sqlite_reads_are_bounded(
    tmp_path: Path, deterministic_cfg_batches
) -> None:
    first, second = deterministic_cfg_batches
    for attribute in ("cfg_graphs", "cfg_blocks", "cfg_elements", "cfg_edges"):
        assert [item.id for item in getattr(first, attribute)] == [
            item.id for item in getattr(second, attribute)
        ]

    with SQLiteStore(tmp_path / "index.db", project_root=CFG_FIXTURE) as store:
        store.apply_ingestion(CFG_FIXTURE, first)
        graphs = store.cfg_graphs(limit=3)
        assert len(graphs.items) == 3
        assert graphs.truncated
        graph = _cfg_for(first, "cfg_fixture::branching")
        assert store.get_cfg_graph(graph.id) == graph
        assert store.cfg_blocks(graph.id).items == tuple(
            sorted(
                (block for block in first.cfg_blocks if block.graph_id == graph.id),
                key=lambda block: (block.index, block.id),
            )
        )
        assert store.cfg_elements(graph.id, limit=2).truncated
        assert not store.cfg_edges(graph.id).truncated
        with pytest.raises(ValueError, match="CFG page limit"):
            store.cfg_blocks(graph.id, limit=0)


def test_cfg_adapter_rejects_cross_graph_block_references() -> None:
    client = cached_native_client(analyzer_binary(), timeout_seconds=30)
    configuration = CompilationDatabase.load(CFG_FIXTURE / "compile_commands.json").configurations[
        0
    ]
    facts = list(client.analyze(CFG_FIXTURE, configuration))
    graph_keys = [fact["key"] for fact in facts if fact.get("fact") == "cfg_graph_v1"]
    element = next(fact for fact in facts if fact.get("fact") == "cfg_element_v1")
    malformed = dict(element)
    malformed["graph_key"] = next(key for key in graph_keys if key != element["graph_key"])
    facts[facts.index(element)] = malformed

    with pytest.raises(AnalyzerProtocolError, match="inconsistent graph references"):
        _FactBatchBuilder(CFG_FIXTURE.resolve(), configuration).build(facts)


def test_cfg_exception_edges_follow_build_configuration_and_build_scope(tmp_path: Path) -> None:
    enabled = _cached_batch(CFG_FIXTURE, variant="exceptions")
    disabled = _cached_batch(
        CFG_FIXTURE, database="compile_commands_no_eh.json", variant="no-exceptions"
    )
    enabled_graph = _cfg_for(enabled, "cfg_fixture::exception_flow")
    disabled_graph = _cfg_for(disabled, "cfg_fixture::exception_flow")
    assert enabled_graph.build_options["add_eh_edges"] is True
    assert disabled_graph.build_options["add_eh_edges"] is False
    assert any(
        edge.kind == CfgEdgeKind.EXCEPTION
        for edge in enabled.cfg_edges
        if edge.graph_id == enabled_graph.id
    )
    assert not any(
        edge.kind == CfgEdgeKind.EXCEPTION
        for edge in disabled.cfg_edges
        if edge.graph_id == disabled_graph.id
    )

    with SQLiteStore(tmp_path / "index.db", project_root=CFG_FIXTURE) as store:
        store.apply_ingestion(
            CFG_FIXTURE,
            enabled,
            build_variant=BuildVariant("exceptions", CFG_FIXTURE / "compile_commands.json"),
        )
        store.apply_ingestion(
            CFG_FIXTURE,
            disabled,
            build_variant=BuildVariant(
                "no-exceptions", CFG_FIXTURE / "compile_commands_no_eh.json"
            ),
        )
        assert store.cfg_graphs(
            enabled_graph.function_symbol_id, build_scope=BuildScope.single("exceptions")
        ).items == (enabled_graph,)
        assert store.cfg_graphs(
            disabled_graph.function_symbol_id,
            build_scope=BuildScope.single("no-exceptions"),
        ).items == (disabled_graph,)
        assert store.remove_build_variant("exceptions")
        assert not store.cfg_graphs(build_scope=BuildScope.single("exceptions")).items
        assert (
            store.get_cfg_graph(disabled_graph.id, build_scope=BuildScope.single("no-exceptions"))
            == disabled_graph
        )


def test_cfg_reindex_replaces_only_changed_tu_and_rolls_back_atomically(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(PARITY_FIXTURE, project)
    database = project / "compile_commands.json"
    ingestor = NativeClangIngestor(NativeAnalyzerClient(analyzer_binary(), timeout_seconds=30))
    with SQLiteStore(tmp_path / "index.db", project_root=project) as store:
        indexer = ProjectIndexer(ingestor, store)
        first = indexer.index(project, database)
        assert first.indexed_cfg_graphs > 0
        units = store._connection.execute(  # noqa: SLF001
            "SELECT id, source_path FROM translation_units ORDER BY source_path"
        ).fetchall()
        unchanged_unit = next(
            row["id"] for row in units if row["source_path"].endswith("model.cpp")
        )
        unchanged_before = tuple(
            tuple(row)
            for row in store._connection.execute(  # noqa: SLF001
                "SELECT id, function_symbol_id FROM cfg_graphs "
                "WHERE translation_unit_id = ? ORDER BY id",
                (unchanged_unit,),
            )
        )
        old_graph_count = len(store.cfg_graphs(limit=10_000).items)
        source = project / "src" / "main.cpp"
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        changed = indexer.index(project, database)
        assert changed.indexed_translation_units == 1
        assert (
            tuple(
                tuple(row)
                for row in store._connection.execute(  # noqa: SLF001
                    "SELECT id, function_symbol_id FROM cfg_graphs "
                    "WHERE translation_unit_id = ? ORDER BY id",
                    (unchanged_unit,),
                )
            )
            == unchanged_before
        )
        assert len(store.cfg_graphs(limit=10_000).items) == old_graph_count

        current = ingestor.ingest(project, database)
        broken = replace(
            current.cfg_edges[0], id="broken-cfg-edge", target_block_id="missing-block"
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.apply_ingestion(project, replace(current, cfg_edges=(*current.cfg_edges, broken)))
        assert len(store.cfg_graphs(limit=10_000).items) == old_graph_count


@pytest.mark.clang
def test_companion_preserves_baseline_canonical_ids_and_relation_parity() -> None:
    fixture = staged_fixture(PARITY_FIXTURE)
    try:
        baseline = ClangIngestor().ingest(fixture, fixture / "compile_commands.json")
    except ClangUnavailableError as error:
        pytest.skip(str(error))
    native = NativeClangIngestor(fresh_native_client(analyzer_binary(), timeout_seconds=30)).ingest(
        fixture, fixture / "compile_commands.json"
    )
    names = {"demo::Base", "demo::Derived", "demo::Derived::compute", "demo::helper", "run"}
    baseline_ids = {
        (symbol.qualified_name, symbol.id)
        for symbol in baseline.symbols
        if symbol.qualified_name in names
    }
    native_ids = {
        (symbol.qualified_name, symbol.id)
        for symbol in native.symbols
        if symbol.qualified_name in names
    }
    assert baseline_ids <= native_ids
    assert {edge.relation for edge in baseline.edges} <= {edge.relation for edge in native.edges}
    assert baseline.callsites == () and baseline.call_targets == ()
    assert all(not unit.advanced_facts_complete for unit in baseline.translation_units)


@pytest.mark.clang
def test_switching_from_baseline_to_companion_forces_reindex(tmp_path: Path) -> None:
    try:
        baseline = ClangIngestor()
    except ClangUnavailableError as error:
        pytest.skip(str(error))
    fixture = staged_fixture(PARITY_FIXTURE)
    database = fixture / "compile_commands.json"
    with SQLiteStore(tmp_path / "index.db", project_root=fixture) as store:
        first = ProjectIndexer(baseline, store).index(fixture, database)
        upgraded = ProjectIndexer(
            NativeClangIngestor(fresh_native_client(analyzer_binary(), timeout_seconds=30)),
            store,
        ).index(fixture, database)
        states = store.translation_unit_states(fixture)

    assert first.indexed_translation_units == 2
    assert upgraded.indexed_translation_units == 2
    assert {state.analysis_backend for state in states.values()} == {"clang-libtooling"}
    assert all(state.advanced_facts_complete for state in states.values())


def test_protocol_mismatch_fails_before_analysis_and_sanitizes_path(tmp_path: Path) -> None:
    marker = tmp_path / "analysis-ran"
    script = _script(
        tmp_path,
        f"""import json, sys
request = json.loads(sys.stdin.readline())
print(json.dumps({{"type":"hello","protocol":"wrong","protocol_version":99,
                   "analyzer_version":"x","clang_major":17,"capabilities":[]}}))
for line in sys.stdin:
    open({str(marker)!r}, "w").write("bad")
""",
    )

    with pytest.raises(AnalyzerProtocolError, match="protocol mismatch") as captured:
        NativeAnalyzerClient(script).probe()

    assert not marker.exists()
    assert str(tmp_path) not in str(captured.value)


def test_analyze_revalidates_the_process_handshake(tmp_path: Path) -> None:
    launches = tmp_path / "launches"
    valid = {
        "type": "hello",
        "protocol": "cpp-context-clang-facts",
        "protocol_version": 5,
        "analyzer_version": "test",
        "clang_major": 18,
        "capabilities": [
            "direct_calls",
            "full_ast",
            "function_cfg_v1",
            "includes",
            "inherits",
            "lambda_metadata",
            "macro_provenance",
            "occurrences",
            "overrides",
            "pp_callbacks",
            "source_manager",
            "symbols",
            "template_metadata",
            "uses_type",
            "callsites_v1",
            "dispatch_targets_v1",
            "macro_expansion_stack",
            "template_relationships_v1",
            "intraprocedural_dataflow_v1",
            "points_to_v1",
            "function_summaries_v1",
            "interprocedural_bindings_v1",
        ],
    }
    script = _script(
        tmp_path,
        f"""import json, pathlib, sys
counter = pathlib.Path({str(launches)!r})
launch = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(launch))
json.loads(sys.stdin.readline())
hello = {valid!r}
if launch == 2:
    hello["protocol_version"] = 99
print(json.dumps(hello), flush=True)
if launch == 2:
    request = json.loads(sys.stdin.readline())
    print(json.dumps({{"type": "begin", "request_id": request["request_id"]}}))
    print(json.dumps({{"type": "complete", "request_id": request["request_id"],
                      "success": True}}))
""",
    )
    client = NativeAnalyzerClient(script)
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]

    client.probe()
    with pytest.raises(AnalyzerProtocolError, match="protocol mismatch"):
        client.analyze(FIXTURE, configuration)


def test_fragmented_gzip_transport_preserves_protocol_v5_records(tmp_path: Path) -> None:
    fact = {"type": "fact", "fact": "test", "payload": "fragmented"}
    client = NativeAnalyzerClient(_gzip_analyzer_script(tmp_path, [fact]), timeout_seconds=2)
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]

    assert client.analyze(FIXTURE, configuration) == (fact,)
    assert "gzip_jsonl_v1" in client.probe().capabilities


def test_new_client_streams_plain_records_from_old_companion(tmp_path: Path) -> None:
    hello = _fake_hello()
    fact = {"type": "fact", "fact": "test", "payload": "plain"}
    script = _script(
        tmp_path,
        f"""import json, sys
requests = [json.loads(line) for line in sys.stdin]
print(json.dumps({hello!r}), flush=True)
if len(requests) > 1:
    request = requests[1]
    print(json.dumps({{"type": "begin", "request_id": request["request_id"]}}), flush=True)
    print(json.dumps({fact!r}), flush=True)
    print(json.dumps({{"type": "complete", "request_id": request["request_id"],
                      "success": True}}), flush=True)
""",
    )
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]

    assert NativeAnalyzerClient(script).analyze(FIXTURE, configuration) == (fact,)


@pytest.mark.parametrize(
    ("client_kwargs", "payload", "message"),
    [
        (
            {"max_output_bytes": 1024},
            "".join(f"{index:08x}" for index in range(2000)),
            "compressed output limit",
        ),
        ({"max_decoded_bytes": 1024}, "x" * 4096, "decoded output limit"),
        ({"max_record_bytes": 1024}, "x" * 4096, "record limit"),
    ],
)
def test_gzip_transport_enforces_independent_limits(
    tmp_path: Path,
    client_kwargs: dict[str, int],
    payload: str,
    message: str,
) -> None:
    fact = {"type": "fact", "fact": "test", "payload": payload}
    client = NativeAnalyzerClient(
        _gzip_analyzer_script(tmp_path, [fact], fragment_size=7),
        timeout_seconds=2,
        **client_kwargs,
    )
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]

    with pytest.raises(AnalyzerLimitError, match=message):
        client.analyze(FIXTURE, configuration)


def test_malformed_gzip_and_jsonl_are_protocol_errors(tmp_path: Path) -> None:
    hello = _fake_hello(gzip_transport=True)
    malformed_gzip = _script(
        tmp_path,
        f"""import json, os, sys
requests = [json.loads(line) for line in sys.stdin]
if len(requests) == 1:
    print(json.dumps({hello!r}), flush=True)
else:
    os.write(1, b"not-gzip")
""",
    )
    configuration = CompilationDatabase.load(FIXTURE / "compile_commands.json").configurations[0]
    client = NativeAnalyzerClient(malformed_gzip, timeout_seconds=2)

    with pytest.raises(AnalyzerProtocolError, match="malformed gzip"):
        client.analyze(FIXTURE, configuration)

    malformed_jsonl = _gzip_analyzer_script(
        tmp_path,
        [{"type": "fact", "fact": "test"}],
        fragment_size=3,
    )
    malformed_jsonl.write_text(
        malformed_jsonl.read_text().replace(
            "compressor.compress(raw)",
            "compressor.compress(raw.replace(b'\\\"fact\\\"', b'bad', 1))",
        ),
        encoding="utf-8",
    )
    client = NativeAnalyzerClient(malformed_jsonl, timeout_seconds=2)
    with pytest.raises(AnalyzerProtocolError, match="malformed JSONL"):
        client.analyze(FIXTURE, configuration)


def test_incomplete_stream_does_not_persist_partial_batch(tmp_path: Path) -> None:
    hello = _fake_hello()
    script = _script(
        tmp_path,
        f"""import json, sys
requests = [json.loads(line) for line in sys.stdin]
print(json.dumps({hello!r}), flush=True)
if len(requests) > 1:
    request = requests[1]
    print(json.dumps({{"type": "begin", "request_id": request["request_id"]}}), flush=True)
    print(json.dumps({{"type": "fact", "fact": "file", "key": "partial",
                      "path": str(request["source_path"])}}), flush=True)
""",
    )
    ingestor = NativeClangIngestor(NativeAnalyzerClient(script, timeout_seconds=2))

    with SQLiteStore(tmp_path / "index.db", project_root=FIXTURE) as store:
        with pytest.raises(AnalyzerProtocolError, match="incomplete"):
            ProjectIndexer(ingestor, store).index(FIXTURE, FIXTURE / "compile_commands.json")
        assert store.translation_unit_states(FIXTURE) == {}


def test_broken_analyzer_stdin_still_obeys_timeout(tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        "import os, time\nos.close(0)\ntime.sleep(2)\n",
    )
    client = NativeAnalyzerClient(script, timeout_seconds=0.05)
    request = {"payload": "x" * 200_000}

    started = time.monotonic()
    with pytest.raises(AnalyzerLimitError, match="timeout"):
        client._invoke((request,), output_limit=1024)

    assert time.monotonic() - started < 1


def test_timeout_and_output_limits_are_hard(tmp_path: Path) -> None:
    timeout = _script(tmp_path, "import time\ntime.sleep(10)\n")
    with pytest.raises(AnalyzerLimitError, match="timeout"):
        NativeAnalyzerClient(timeout, timeout_seconds=0.05).probe()

    output = _script(tmp_path, "print('x' * 10000)\n")
    with pytest.raises(AnalyzerLimitError, match="output limit"):
        NativeAnalyzerClient(output, max_output_bytes=64).probe()

    stderr = _script(tmp_path, "import sys\nsys.stderr.write('x' * 10000)\n")
    with pytest.raises(AnalyzerLimitError, match="stderr output limit"):
        NativeAnalyzerClient(stderr, max_stderr_bytes=64).probe()


def test_timeout_kills_companion_process_group_within_two_seconds(tmp_path: Path) -> None:
    if not Path("/proc").is_dir():
        pytest.skip("process-group cleanup assertion requires Linux procfs")
    child_pid = tmp_path / "child.pid"
    script = _script(
        tmp_path,
        f"""import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))
time.sleep(30)
""",
    )

    started = time.monotonic()
    with pytest.raises(AnalyzerLimitError, match="timeout"):
        NativeAnalyzerClient(script, timeout_seconds=0.2).probe()

    elapsed = time.monotonic() - started
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 1
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert elapsed < 2
    assert not Path(f"/proc/{pid}").exists()


def test_parent_exit_still_kills_inherited_pipe_descendant(tmp_path: Path) -> None:
    if not Path("/proc").is_dir():
        pytest.skip("process-group cleanup assertion requires Linux procfs")
    child_pid = tmp_path / "orphan.pid"
    script = _script(
        tmp_path,
        f"""import pathlib, subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))
""",
    )

    started = time.monotonic()
    with pytest.raises(AnalyzerProtocolError, match="handshake"):
        NativeAnalyzerClient(script, timeout_seconds=5).probe()

    elapsed = time.monotonic() - started
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 1
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert elapsed < 3
    assert not Path(f"/proc/{pid}").exists()


def test_sqlite_round_trips_macro_provenance(tmp_path: Path) -> None:
    batch = _cached_batch(FIXTURE)
    macro = next(symbol for symbol in batch.symbols if symbol.qualified_name == "APPLY_TWICE")
    with SQLiteStore(tmp_path / "index.db", project_root=FIXTURE) as store:
        store.apply_ingestion(FIXTURE, batch)
        occurrence = next(
            item
            for item in store.occurrences(macro.id)
            if item.kind == OccurrenceKind.MACRO_EXPANSION
        )
        assert occurrence.metadata["spelling_span"]
        assert occurrence.metadata["expansion_span"]


def test_doctor_checks_real_companion(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "doctor",
                "--project",
                str(FIXTURE),
                "--compile-commands",
                str(FIXTURE / "compile_commands.json"),
                "--clang-analyzer",
                str(analyzer_binary()),
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["clang_analyzer_executable"] is True
    assert report["clang_analyzer_protocol"] == 5
    assert report["clang_analyzer_clang_major"] == 18
    assert report["advanced_facts_complete"] is True
    assert report["cfg_facts_available"] is True
    assert report["call_facts_available"] is True
    assert report["data_flow_facts_available"] is True
    assert report["function_summary_facts_available"] is True
    assert "function_cfg_v1" in report["clang_analyzer_capabilities"]
    assert "macro_provenance" in report["clang_analyzer_capabilities"]
    assert "callsites_v1" in report["clang_analyzer_capabilities"]
    assert "intraprocedural_dataflow_v1" in report["clang_analyzer_capabilities"]
    assert "function_summaries_v1" in report["clang_analyzer_capabilities"]
    assert "interprocedural_bindings_v1" in report["clang_analyzer_capabilities"]


def test_cli_index_reports_companion_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = staged_fixture(FIXTURE)
    assert (
        main(
            [
                "index",
                str(fixture),
                "--compile-commands",
                str(fixture / "compile_commands.json"),
                "--clang-analyzer",
                str(analyzer_binary()),
                "--db",
                str(tmp_path / "index.db"),
                "--embedding-dimensions",
                "32",
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["analysis_backend"] == "clang-libtooling"
    assert report["advanced_facts_complete"] is True
    assert "macro_provenance" in report["analyzer_capabilities"]
