from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from analyzer_discovery import analyzer_binary

from cpp_context_engine.analysis.interprocedural import solve_interprocedural
from cpp_context_engine.api import CfgRequest, FlowRequest
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion import MaterializeDeepRequest, MaterializeDeepResult
from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.ingestion.deep import (
    DeepCancellation,
    DeepMaterializer,
    DeepRequestControl,
    _file_digest,
)
from cpp_context_engine.ingestion.native import (
    AnalyzerLimitError,
    NativeClangIngestor,
    _FactRegistry,
    _JsonlStreamDecoder,
    _ResourceBudget,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import IndexProfile
from cpp_context_engine.runtime import build_runtime, index_project
from cpp_context_engine.storage import SQLiteStore
from cpp_context_engine.storage.sqlite import DeepTranslationUnitTarget

FIXTURE = Path(__file__).parent / "fixtures" / "analyzer_project"


def _config(
    project: Path,
    database: Path,
    analyzer: Path,
    profile: IndexProfile = IndexProfile.NAVIGATION,
) -> AppConfig:
    return AppConfig(
        project_root=project,
        index_directory=database.parent,
        database_path=database,
        compilation_database=project / "compile_commands.json",
        clang_analyzer_path=analyzer,
        analyzer_max_workers=2,
        index_profile=profile,
        embedding_dimensions=32,
    )


@pytest.mark.native
def test_navigation_symbol_materializes_deep_overlay_and_restart_hits_cache(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    analyzer = analyzer_binary()
    config = _config(project, tmp_path / "index.db", analyzer)
    indexed = index_project(config)
    assert indexed.index_profile is IndexProfile.NAVIGATION

    with build_runtime(config) as runtime:
        row = runtime.store._connection.execute(  # noqa: SLF001 - stable fixture lookup
            "SELECT id FROM symbols WHERE qualified_name = 'analyzer_fixture::construct' "
            "ORDER BY id LIMIT 1"
        ).fetchone()
        assert row is not None
        symbol_id = row[0]
        before = runtime.analysis_service.control_flow(CfgRequest(function_symbol_id=symbol_id))
        assert not before.available
        assert before.required_action == "materialize_deep_analysis"

        materialized = runtime.materialize_deep(
            MaterializeDeepRequest(symbol_id=symbol_id, max_tus=1, max_wall_seconds=60)
        )
        assert materialized.status in {"complete", "partial"}
        assert len(materialized.units) == 1
        after = runtime.analysis_service.control_flow(
            CfgRequest(
                function_symbol_id=symbol_id,
                materialization_id=materialized.materialization_id,
            )
        )
        flow = runtime.analysis_service.data_flow(
            FlowRequest(
                function_symbol_id=symbol_id,
                materialization_id=materialized.materialization_id,
            )
        )
        assert after.available and after.graphs
        assert flow.available and flow.analyses
        stale_token = runtime.analysis_service.control_flow(
            CfgRequest(function_symbol_id=symbol_id, materialization_id="stale-generation")
        )
        assert not stale_token.available
        assert stale_token.required_action == "materialize_deep_analysis"
        assert runtime.store.build_index_profiles(project)["default"] is IndexProfile.NAVIGATION

    def forbidden_probe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("restart cache hit must not probe or launch Clang")

    monkeypatch.setattr(
        "cpp_context_engine.ingestion.deep.NativeAnalyzerClient.probe", forbidden_probe
    )
    with build_runtime(config) as restarted:
        cached = restarted.materialize_deep(
            MaterializeDeepRequest(symbol_id=symbol_id, max_tus=1, max_wall_seconds=60)
        )
        assert cached.status == "cache_hit"
        assert cached.cache_hit

    source = project / "src" / "analysis.cpp"
    source_text = source.read_text()
    source.write_text(source_text + "\n// invalidate deep cache\n")
    with build_runtime(config) as stale, pytest.raises(RuntimeError, match="source changed"):
        stale.materialize_deep(
            MaterializeDeepRequest(symbol_id=symbol_id, max_tus=1, max_wall_seconds=60)
        )
    source.write_text(source_text)

    header = project / "include" / "analysis.hpp"
    header_text = header.read_text()
    header.write_text(header_text + "\n// invalidate dependent deep cache\n")
    with build_runtime(config) as stale, pytest.raises(RuntimeError, match="dependency changed"):
        stale.materialize_deep(
            MaterializeDeepRequest(symbol_id=symbol_id, max_tus=1, max_wall_seconds=60)
        )
    header.write_text(header_text)

    compilation_database = project / "compile_commands.json"
    commands = json.loads(compilation_database.read_text())
    commands[0]["arguments"].insert(2, "-DDEEP_CACHE_CHANGED=1")
    compilation_database.write_text(json.dumps(commands))
    with (
        build_runtime(config) as stale,
        pytest.raises(RuntimeError, match="navigation index is stale"),
    ):
        stale.materialize_deep(
            MaterializeDeepRequest(symbol_id=symbol_id, max_tus=1, max_wall_seconds=60)
        )


def test_deep_materialization_limit_contracts_reject_before_analysis(tmp_path: Path) -> None:
    request = MaterializeDeepRequest(symbol_id="symbol", max_tus=4)
    assert request.max_wall_seconds == 120
    assert request.max_decoded_bytes == 512 * 1024 * 1024
    assert request.max_spool_bytes == 512 * 1024 * 1024
    assert request.max_spool_files == 128
    with pytest.raises(ValueError):
        MaterializeDeepRequest(symbol_id="symbol", builds=[])


def test_identical_deep_requests_coalesce_one_leader(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    analyzer = tmp_path / "analyzer"
    analyzer.write_bytes(b"fixture")
    config = _config(project, tmp_path / "index.db", analyzer)
    entered = threading.Event()
    waiter_entered = threading.Event()
    release = threading.Event()
    calls = 0
    with SQLiteStore(config.database_path, project_root=project) as store:
        materializer = DeepMaterializer(config, store)
        original_result = __import__("concurrent.futures").futures.Future.result

        def observed_result(future, *args, **kwargs):
            waiter_entered.set()
            return original_result(future, *args, **kwargs)

        monkeypatch.setattr("cpp_context_engine.ingestion.deep.Future.result", observed_result)

        def run(_request: MaterializeDeepRequest, _cancelled=None) -> MaterializeDeepResult:
            nonlocal calls
            calls += 1
            entered.set()
            assert release.wait(5)
            return MaterializeDeepResult(
                status="complete",
                materialization_id="generation",
                root_symbol_id="symbol",
                units=[],
                closure_complete=True,
                known_tus=0,
                omitted_tus=0,
                cache_hit=False,
                elapsed_seconds=0.0,
                provenance={
                    "analyzer_identity": "fixture",
                    "analyzer_version": "fixture",
                    "protocol": "cpp-context-clang-facts",
                    "protocol_version": 5,
                    "fact_schema_version": 15,
                    "profile": "full",
                    "build_scope": ["default"],
                    "closure_generation_id": "fixture",
                },
            )

        monkeypatch.setattr(materializer, "_materialize", run)
        request = MaterializeDeepRequest(symbol_id="symbol", max_wall_seconds=10)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(materializer.materialize, request)
            assert entered.wait(5)
            second = pool.submit(materializer.materialize, request)
            assert waiter_entered.wait(5)
            release.set()
            assert first.result().materialization_id == "generation"
            assert second.result().materialization_id == "generation"
        assert calls == 1
        assert materializer._inflight == {}  # noqa: SLF001 - cleanup contract


def test_deep_leader_error_reaches_waiter_and_clears_inflight(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    analyzer = tmp_path / "analyzer"
    analyzer.write_bytes(b"fixture")
    config = _config(project, tmp_path / "index.db", analyzer)
    entered = threading.Event()
    waiter_entered = threading.Event()
    release = threading.Event()
    with SQLiteStore(config.database_path, project_root=project) as store:
        materializer = DeepMaterializer(config, store)
        original_result = __import__("concurrent.futures").futures.Future.result

        def observed_result(future, *args, **kwargs):
            waiter_entered.set()
            return original_result(future, *args, **kwargs)

        def fail(_request: MaterializeDeepRequest, _cancelled=None) -> MaterializeDeepResult:
            entered.set()
            assert release.wait(5)
            raise RuntimeError("leader failed")

        monkeypatch.setattr("cpp_context_engine.ingestion.deep.Future.result", observed_result)
        monkeypatch.setattr(materializer, "_materialize", fail)
        request = MaterializeDeepRequest(symbol_id="symbol", max_wall_seconds=10)
        with ThreadPoolExecutor(max_workers=2) as pool:
            leader = pool.submit(materializer.materialize, request)
            assert entered.wait(5)
            waiter = pool.submit(materializer.materialize, request)
            assert waiter_entered.wait(5)
            release.set()
            with pytest.raises(RuntimeError, match="leader failed"):
                leader.result()
            with pytest.raises(RuntimeError, match="leader failed"):
                waiter.result()
        assert materializer._inflight == {}  # noqa: SLF001 - failure cleanup contract


@pytest.mark.native
def test_full_profile_materialization_persists_usable_token_without_analyzer_probe(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    analyzer = analyzer_binary()
    config = _config(project, tmp_path / "index.db", analyzer, IndexProfile.FULL)
    index_project(config)
    with build_runtime(config) as runtime:
        row = runtime.store._connection.execute(  # noqa: SLF001
            "SELECT id FROM symbols WHERE qualified_name = 'analyzer_fixture::construct' "
            "ORDER BY id LIMIT 1"
        ).fetchone()
        assert row is not None

        def forbidden_probe(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("full-profile facts must not launch Clang")

        monkeypatch.setattr(
            "cpp_context_engine.ingestion.deep.NativeAnalyzerClient.probe", forbidden_probe
        )
        result = runtime.materialize_deep(MaterializeDeepRequest(symbol_id=row[0], max_tus=1))
        assert result.status == "cache_hit"
        token = result.materialization_id
        cfg = runtime.analysis_service.control_flow(
            CfgRequest(function_symbol_id=row[0], materialization_id=token)
        )
        flow = runtime.analysis_service.data_flow(
            FlowRequest(function_symbol_id=row[0], materialization_id=token)
        )
        assert cfg.available and cfg.graphs
        assert flow.available and flow.analyses
        assert (
            runtime.store._connection.execute(  # noqa: SLF001
                "SELECT count(*) FROM deep_materializations"
            ).fetchone()[0]
            == 1
        )

    with build_runtime(config) as restarted:
        cfg = restarted.analysis_service.control_flow(
            CfgRequest(function_symbol_id=row[0], materialization_id=token)
        )
        flow = restarted.analysis_service.data_flow(
            FlowRequest(function_symbol_id=row[0], materialization_id=token)
        )
        assert cfg.available and cfg.graphs
        assert flow.available and flow.analyses


def test_closure_stops_at_one_omitted_tu_and_expands_only_rooted_functions(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    analyzer = tmp_path / "analyzer"
    analyzer.write_bytes(b"fixture")
    config = _config(project, tmp_path / "index.db", analyzer)

    def target(index: int, distance: int) -> DeepTranslationUnitTarget:
        path = project / f"unit-{index}.cpp"
        path.write_text("")
        return DeepTranslationUnitTarget(
            translation_unit_id=f"unit-{index}",
            build_configuration_id=f"build-{index}",
            build_variant="default",
            command_hash=f"command-{index}",
            source_path=path,
            content_hash="",
            dependencies=(),
            distance=distance,
        )

    calls: list[tuple[str, tuple[str, ...]]] = []
    with SQLiteStore(config.database_path, project_root=project) as store:
        materializer = DeepMaterializer(config, store)

        def callees(current, owners, _root, **_kwargs):
            calls.append((current.translation_unit_id, owners))
            index = int(current.translation_unit_id.rsplit("-", 1)[1])
            return ((f"function-{index + 1}", target(index + 1, current.distance + 1)),)

        monkeypatch.setattr(store, "deep_symbol_callees", callees)
        selected, known, omitted = materializer._closure(  # noqa: SLF001
            (target(0, 0),),
            "root-function",
            2,
            DeepRequestControl(time.monotonic(), 5, DeepCancellation()),
        )
    assert [item.translation_unit_id for item in selected] == ["unit-0", "unit-1"]
    assert known == 3
    assert omitted == 1
    assert calls == [
        ("unit-0", ("root-function",)),
        ("unit-1", ("function-1",)),
    ]


def test_header_fanout_counts_all_definition_roots_before_truncation(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    analyzer = tmp_path / "analyzer"
    analyzer.write_bytes(b"fixture")
    config = _config(project, tmp_path / "index.db", analyzer)

    def target(index: int) -> DeepTranslationUnitTarget:
        path = project / f"unit-{index}.cpp"
        path.write_text("")
        return DeepTranslationUnitTarget(
            translation_unit_id=f"unit-{index}",
            build_configuration_id=f"build-{index}",
            build_variant="default",
            command_hash=f"command-{index}",
            source_path=path,
            content_hash="",
            dependencies=(),
        )

    with SQLiteStore(config.database_path, project_root=project) as store:
        materializer = DeepMaterializer(config, store)
        monkeypatch.setattr(store, "deep_symbol_callees", lambda *_args, **_kwargs: ())
        selected, known, omitted = materializer._closure(  # noqa: SLF001
            tuple(target(index) for index in range(5)),
            "inline-header-function",
            2,
            DeepRequestControl(time.monotonic(), 5, DeepCancellation()),
        )

    assert [item.translation_unit_id for item in selected] == ["unit-0", "unit-1"]
    assert known == 5
    assert omitted == 3


def test_hard_deadline_interrupts_hash_closure_merge_solver_and_db(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "unit.cpp"
    source.write_bytes(b"x" * (1024 * 1024 + 1))
    analyzer = tmp_path / "analyzer"
    analyzer.write_bytes(b"fixture")
    config = _config(project, tmp_path / "index.db", analyzer)
    expired = DeepRequestControl(time.monotonic() - 2, 1, DeepCancellation())
    target = DeepTranslationUnitTarget(
        translation_unit_id="unit",
        build_configuration_id="build",
        build_variant="default",
        command_hash="command",
        source_path=source,
        content_hash="content",
        dependencies=(),
    )

    with pytest.raises(TimeoutError, match="file hashing"):
        _file_digest(source, expired)
    compilation_database = project / "compile_commands.json"
    compilation_database.write_bytes(b"[" + b" " * (1024 * 1024) + b"]")
    with pytest.raises(TimeoutError, match="compilation database load"):
        CompilationDatabase.load(
            compilation_database,
            check_cancelled=lambda: expired.check("compilation database load"),
        )
    with SQLiteStore(config.database_path, project_root=project) as store:
        materializer = DeepMaterializer(config, store)
        with pytest.raises(TimeoutError, match="closure"):
            materializer._closure((target,), "symbol", 2, expired)  # noqa: SLF001
        with pytest.raises(TimeoutError, match="navigation parity"):
            store.validate_deep_navigation_parity(project, (), request_control=expired)
    empty = IngestionBatch((), (), (), (), ())
    with pytest.raises(TimeoutError, match="batch merge"):
        NativeClangIngestor._merge_batches(  # noqa: SLF001
            (empty,), check_callback=lambda: expired.check("batch merge")
        )
    with pytest.raises(TimeoutError, match="summary solver"):
        solve_interprocedural(
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            check_cancelled=lambda: expired.check("summary solver"),
        )


def test_admission_queue_time_counts_against_request_deadline(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    analyzer = tmp_path / "analyzer"
    analyzer.write_bytes(b"fixture")
    config = _config(project, tmp_path / "index.db", analyzer)
    with SQLiteStore(config.database_path, project_root=project) as store:
        materializer = DeepMaterializer(config, store)
        materializer._admission.acquire()  # noqa: SLF001 - deterministic queue saturation
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="admission queue"):
                materializer.materialize(
                    MaterializeDeepRequest(symbol_id="never-started", max_wall_seconds=1)
                )
        finally:
            materializer._admission.release()  # noqa: SLF001
    assert time.monotonic() - started < 1.5


def test_request_aggregate_decoded_and_parallel_spool_budgets_cleanup() -> None:
    decoded = _ResourceBudget(5, "aggregate decoded limit")
    records: list[dict[str, object]] = []
    first = _JsonlStreamDecoder(
        gzip_transport=False,
        max_wire_bytes=100,
        max_decoded_bytes=100,
        max_record_bytes=100,
        on_record=records.append,
        decoded_budget=decoded,
    )
    second = _JsonlStreamDecoder(
        gzip_transport=False,
        max_wire_bytes=100,
        max_decoded_bytes=100,
        max_record_bytes=100,
        on_record=records.append,
        decoded_budget=decoded,
    )
    first.feed(b"{}\n")
    with pytest.raises(AnalyzerLimitError, match="aggregate decoded limit"):
        second.feed(b"{}\n")
    assert decoded.used == 3

    spool_bytes = _ResourceBudget(4096, "spool bytes")
    spool_fds = _ResourceBudget(2, "spool fds")
    registries = tuple(
        _FactRegistry(
            max_bytes=4096,
            max_record_bytes=4096,
            byte_budget=spool_bytes,
            fd_budget=spool_fds,
        )
        for _ in range(2)
    )
    try:
        registries[0].add({"fact": "symbol", "key": "first"})
        registries[1].add({"fact": "edge", "key": "second"})
        assert spool_bytes.used > 0
        assert spool_fds.used == 2
    finally:
        for registry in registries:
            registry.close()
    assert spool_bytes.used == 0
    assert spool_fds.used == 0


@pytest.mark.parametrize(
    "limit",
    [
        {"max_decoded_bytes": 1},
        {"max_spool_bytes": 1},
        {"max_spool_files": 1},
    ],
)
@pytest.mark.native
def test_deep_resource_limit_failure_publishes_nothing(
    tmp_path: Path, limit: dict[str, int]
) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    analyzer = analyzer_binary()
    config = _config(project, tmp_path / "index.db", analyzer)
    index_project(config)
    with build_runtime(config) as runtime:
        row = runtime.store._connection.execute(  # noqa: SLF001
            "SELECT id FROM symbols WHERE qualified_name = 'analyzer_fixture::construct' "
            "ORDER BY id LIMIT 1"
        ).fetchone()
        assert row is not None
        with pytest.raises(AnalyzerLimitError, match="limit|exceed|decoded|spool|file"):
            runtime.materialize_deep(
                MaterializeDeepRequest(symbol_id=row[0], max_tus=1, max_wall_seconds=30, **limit)
            )
        assert (
            runtime.store._connection.execute(  # noqa: SLF001
                "SELECT count(*) FROM deep_materializations"
            ).fetchone()[0]
            == 0
        )
