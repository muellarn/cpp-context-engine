from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cpp_context_engine import kicad_canary
from cpp_context_engine.api import DataFlowResult, FlowRequest
from cpp_context_engine.ingestion import AnalyzerPipelineEvent, AnalyzerSlotIdleError
from cpp_context_engine.kicad_canary import (
    CanaryLimits,
    _compare_baseline,
    _process_group_live,
    _terminate_process_group,
    _validate_profile_provenance,
    database_provenance,
    inspect_compilation_database,
    select_gate_entries,
    semantic_snapshot,
    write_subset_database,
)
from cpp_context_engine.models import IndexProfile


def _write_cdb(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _pipeline_event(
    sequence: int,
    at: float,
    kind: str,
    *,
    slot_id: int | None = None,
    configuration_index: int | None = None,
    outcome: str | None = None,
    unscheduled_count: int = 4,
    held_registries: int = 0,
    max_spool_registries: int = 4,
    slot_count: int = 2,
) -> AnalyzerPipelineEvent:
    return AnalyzerPipelineEvent(
        sequence=sequence,
        monotonic_seconds=at,
        kind=kind,
        slot_id=slot_id,
        configuration_index=configuration_index,
        outcome=outcome,
        unscheduled_count=unscheduled_count,
        held_registries=held_registries,
        max_spool_registries=max_spool_registries,
        slot_count=slot_count,
    )


def _successful_pipeline_report(
    *, configuration_count: int, slot_count: int, max_idle_seconds: float
) -> dict[str, object]:
    return {
        "protocol": "analyzer_pipeline",
        "clock": "monotonic",
        "event_count": configuration_count * 2 + 1,
        "configuration_count": configuration_count,
        "configuration_outcomes": [
            {"configuration_index": index, "outcome": "succeeded"}
            for index in range(configuration_count)
        ],
        "max_idle_seconds": max_idle_seconds,
        "slots": [
            {
                "slot_id": slot_id,
                "state": "idle",
                "configuration_index": slot_id,
                "outcome": "succeeded",
                "maximum_idle_seconds": 0.0,
                "maximum_idle_cause": None,
                "idle_cause": None,
            }
            for slot_id in range(slot_count)
        ],
    }


def _write_final_canary_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE translation_units (
                analysis_backend TEXT,
                advanced_facts_complete INTEGER,
                index_profile TEXT,
                navigation_facts_complete INTEGER,
                cfg_facts_complete INTEGER,
                data_flow_facts_complete INTEGER,
                summary_facts_complete INTEGER,
                indexed_at TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO translation_units VALUES "
            "('clang-libtooling', 0, 'navigation', 1, 0, 0, 0, 'before')"
        )
        connection.execute("CREATE TABLE build_variants (name TEXT, index_profile TEXT)")
        connection.execute("INSERT INTO build_variants VALUES ('default', 'navigation')")


@pytest.fixture
def inline_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing artifact fault injection local; subprocess guards have separate tests."""

    def validate(spec_path, _directory, _limits, _total):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        kicad_canary._validate_gate_spec(spec)
        return {
            "validated": True,
            "process_group_clean": True,
            "elapsed_seconds": 0,
            "total_elapsed_seconds": 0,
            "peak_rss_bytes": 0,
            "peak_swap_bytes": 0,
            "peak_database_bytes": 0,
            "peak_disk_bytes": 0,
        }

    monkeypatch.setattr(kicad_canary, "_run_validation_supervised", validate)


def test_canary_child_wires_exact_analyzer_protocol_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    event = _pipeline_event(1, 1.0, "scheduling_state")

    class Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def probe(self) -> object:
            return object()

    def ingestor(*_args, observer=None, **_kwargs):
        assert observer is not None
        observer(event)
        raise RuntimeError("stop after observer wiring")

    monkeypatch.setattr(kicad_canary, "NativeAnalyzerClient", Client)
    monkeypatch.setattr(kicad_canary, "NativeClangIngestor", ingestor)
    spec = tmp_path / "worker.json"
    spec.write_text(
        json.dumps(
            {
                "project_root": str(tmp_path),
                "compilation_database": str(tmp_path / "compile_commands.json"),
                "database": str(tmp_path / "index.db"),
                "analyzer": str(tmp_path / "analyzer"),
                "translation_units": 1,
                "workers": 1,
                "analyzer_timeout_seconds": 1,
                "embedding_dimensions": 1,
                "queries": [],
                "profile": "navigation",
            }
        ),
        encoding="utf-8",
    )

    assert kicad_canary._run_worker(spec) == 2
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert payloads[0] == event.to_protocol_payload()
    assert payloads[1]["event"] == "error"


def test_canary_monitor_detects_hidden_idle_slot_from_child_protocol() -> None:
    clock = _Clock()
    monitor = kicad_canary._AnalyzerTelemetryMonitor(
        max_idle_seconds=10.0, expected_configurations=4, clock=clock
    )
    events = (
        _pipeline_event(1, 0.0, "scheduling_state"),
        _pipeline_event(2, 0.0, "analyzer_started", slot_id=0, configuration_index=0),
        _pipeline_event(3, 0.0, "analyzer_started", slot_id=1, configuration_index=1),
        _pipeline_event(
            4,
            1.0,
            "analyzer_finished",
            slot_id=0,
            configuration_index=0,
            outcome="succeeded",
        ),
        _pipeline_event(
            5,
            5.0,
            "analyzer_finished",
            slot_id=1,
            configuration_index=1,
            outcome="succeeded",
        ),
        _pipeline_event(6, 5.0, "analyzer_started", slot_id=1, configuration_index=2),
    )
    for event in events:
        monitor.observe_payload(event.to_protocol_payload())

    clock.now = 12.0
    with pytest.raises(AnalyzerSlotIdleError, match=r"slot 0.*11\.000"):
        monitor.check()


def test_canary_monitor_pauses_for_full_spool_and_fully_scheduled_work() -> None:
    clock = _Clock()
    monitor = kicad_canary._AnalyzerTelemetryMonitor(
        max_idle_seconds=10.0, expected_configurations=1, clock=clock
    )
    monitor.observe_payload(
        _pipeline_event(
            1,
            0.0,
            "scheduling_state",
            unscheduled_count=2,
            held_registries=0,
            max_spool_registries=2,
            slot_count=1,
        ).to_protocol_payload()
    )
    clock.now = 5.0
    monitor.check()
    monitor.observe_payload(
        _pipeline_event(
            2,
            5.0,
            "scheduling_state",
            unscheduled_count=2,
            held_registries=2,
            max_spool_registries=2,
            slot_count=1,
        ).to_protocol_payload()
    )
    clock.now = 20.0
    monitor.check()
    monitor.observe_payload(
        _pipeline_event(
            3,
            20.0,
            "scheduling_state",
            unscheduled_count=1,
            held_registries=1,
            max_spool_registries=2,
            slot_count=1,
        ).to_protocol_payload()
    )
    clock.now = 29.0
    monitor.check()
    monitor.observe_payload(
        _pipeline_event(
            4,
            29.0,
            "scheduling_state",
            unscheduled_count=0,
            held_registries=1,
            max_spool_registries=2,
            slot_count=1,
        ).to_protocol_payload()
    )
    clock.now = 100.0
    monitor.check()
    monitor.observe_payload(
        _pipeline_event(
            5,
            100.0,
            "analyzer_started",
            slot_id=0,
            configuration_index=0,
            unscheduled_count=0,
            held_registries=1,
            max_spool_registries=2,
            slot_count=1,
        ).to_protocol_payload()
    )
    monitor.observe_payload(
        _pipeline_event(
            6,
            100.0,
            "analyzer_finished",
            slot_id=0,
            configuration_index=0,
            outcome="succeeded",
            unscheduled_count=0,
            held_registries=1,
            max_spool_registries=2,
            slot_count=1,
        ).to_protocol_payload()
    )

    assert monitor.success_report()["slots"][0]["maximum_idle_seconds"] == 9.0


def test_canary_monitor_rejects_missing_malformed_sequence_and_active_slot() -> None:
    clock = _Clock()
    missing = kicad_canary._AnalyzerTelemetryMonitor(
        max_idle_seconds=10.0, expected_configurations=1, clock=clock
    )
    with pytest.raises(RuntimeError, match="missing"):
        missing.success_report()
    with pytest.raises(ValueError, match="missing fields"):
        missing.observe_payload({"event": "analyzer_pipeline"})

    sequence = kicad_canary._AnalyzerTelemetryMonitor(
        max_idle_seconds=10.0, expected_configurations=1, clock=clock
    )
    sequence.observe_payload(_pipeline_event(1, 0.0, "scheduling_state").to_protocol_payload())
    with pytest.raises(ValueError, match="sequence"):
        sequence.observe_payload(_pipeline_event(3, 0.0, "scheduling_state").to_protocol_payload())

    active = kicad_canary._AnalyzerTelemetryMonitor(
        max_idle_seconds=10.0, expected_configurations=1, clock=clock
    )
    active.observe_payload(_pipeline_event(1, 0.0, "scheduling_state").to_protocol_payload())
    active.observe_payload(
        _pipeline_event(
            2, 0.0, "analyzer_started", slot_id=0, configuration_index=0
        ).to_protocol_payload()
    )
    with pytest.raises(RuntimeError, match="active slot"):
        active.success_report()


@pytest.mark.parametrize("first_outcome", ["failed", "cancelled"])
def test_canary_monitor_rejects_non_successful_history_after_slot_reuse(
    first_outcome: str,
) -> None:
    clock = _Clock()
    monitor = kicad_canary._AnalyzerTelemetryMonitor(
        max_idle_seconds=10.0,
        expected_configurations=2,
        clock=clock,
    )
    events = (
        _pipeline_event(1, 0.0, "scheduling_state", slot_count=1),
        _pipeline_event(2, 0.0, "analyzer_started", slot_id=0, configuration_index=0, slot_count=1),
        _pipeline_event(
            3,
            1.0,
            "analyzer_finished",
            slot_id=0,
            configuration_index=0,
            outcome=first_outcome,
            slot_count=1,
        ),
        _pipeline_event(4, 1.0, "analyzer_started", slot_id=0, configuration_index=1, slot_count=1),
        _pipeline_event(
            5,
            2.0,
            "analyzer_finished",
            slot_id=0,
            configuration_index=1,
            outcome="succeeded",
            slot_count=1,
        ),
    )
    for event in events:
        monitor.observe_payload(event.to_protocol_payload())
    clock.now = 2.0

    with pytest.raises(RuntimeError, match="non-successful"):
        monitor.success_report()


def test_canary_monitor_success_report_is_deterministic_and_has_provenance() -> None:
    clock = _Clock()
    events = (
        _pipeline_event(1, 0.0, "scheduling_state", unscheduled_count=0),
        _pipeline_event(
            2,
            0.0,
            "analyzer_started",
            slot_id=1,
            configuration_index=1,
            unscheduled_count=0,
        ),
        _pipeline_event(
            3,
            0.0,
            "analyzer_started",
            slot_id=0,
            configuration_index=0,
            unscheduled_count=0,
        ),
        _pipeline_event(
            4,
            1.0,
            "analyzer_finished",
            slot_id=1,
            configuration_index=1,
            outcome="succeeded",
            unscheduled_count=0,
        ),
        _pipeline_event(
            5,
            2.0,
            "analyzer_finished",
            slot_id=0,
            configuration_index=0,
            outcome="succeeded",
            unscheduled_count=0,
        ),
    )
    reports = []
    clock.now = 2.0
    for _index in range(2):
        monitor = kicad_canary._AnalyzerTelemetryMonitor(
            max_idle_seconds=10.0, expected_configurations=2, clock=clock
        )
        for event in events:
            monitor.observe_payload(event.to_protocol_payload())
        reports.append(monitor.success_report())

    assert reports[0] == reports[1]
    assert reports[0]["protocol"] == "analyzer_pipeline"
    assert reports[0]["event_count"] == len(events)
    assert [slot["slot_id"] for slot in reports[0]["slots"]] == [0, 1]
    assert all(slot["state"] == "idle" for slot in reports[0]["slots"])


def test_preflight_classifies_out_of_tree_generated_sources_without_rejecting_them(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "src" / "main.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("int main() { return 0; }\n", encoding="utf-8")
    build = tmp_path / "build"
    generated = build / "messages.pb.cc"
    generated.parent.mkdir()
    generated.write_text("int generated() { return 1; }\n", encoding="utf-8")
    external = tmp_path / "vendor" / "vendor.cc"
    external.parent.mkdir()
    external.write_text("int vendor() { return 2; }\n", encoding="utf-8")
    cdb = build / "compile_commands.json"
    _write_cdb(
        cdb,
        [
            {
                "directory": str(project),
                "file": str(source),
                "arguments": ["c++", "-c", str(source)],
            },
            {
                "directory": str(build),
                "file": "messages.pb.cc",
                "arguments": ["c++", "-c", "messages.pb.cc"],
            },
            {
                "directory": str(external.parent),
                "file": str(external),
                "arguments": ["c++", "-c", str(external)],
            },
        ],
    )

    inspection = inspect_compilation_database(project, cdb)

    assert inspection.entry_count == 3
    assert inspection.classification_counts == {
        "project_source": 1,
        "generated_build_source": 1,
        "external_source": 1,
    }
    assert [entry.raw_index for entry in select_gate_entries(inspection, 1)] == [0]
    assert [entry.raw_index for entry in select_gate_entries(inspection, "all")] == [0, 1, 2]


def test_all_gate_forwards_only_explicit_generated_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inline_validator: None
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    build = tmp_path / "build"
    generated_root = build / "generated"
    generated_root.mkdir(parents=True)
    generated = generated_root / "messages.pb.cc"
    generated.write_text("int generated() {}\n", encoding="utf-8")
    cdb = build / "compile_commands.json"
    _write_cdb(
        cdb,
        [
            {"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]},
            {
                "directory": str(generated_root),
                "file": str(generated),
                "arguments": ["c++", str(generated)],
            },
        ],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    observed_roots: list[list[str]] = []

    def supervised(spec_path, _gate_directory, _limits, total_tus):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        observed_roots.append(spec["generated_source_roots"])
        return {
            "completed_translation_units": total_tus,
            "process_group_clean": True,
            "analyzer_pipeline": _successful_pipeline_report(
                configuration_count=total_tus,
                slot_count=1,
                max_idle_seconds=1,
            ),
            "database_provenance": {
                "translation_unit_groups": [
                    {
                        "analysis_backend": "clang-libtooling",
                        "advanced_facts_complete": 0,
                        "index_profile": "navigation",
                        "navigation_facts_complete": 1,
                        "cfg_facts_complete": 0,
                        "data_flow_facts_complete": 0,
                        "summary_facts_complete": 0,
                        "translation_units": total_tus,
                    }
                ],
                "build_variants": [{"name": "default", "index_profile": "navigation"}],
            },
        }

    monkeypatch.setattr(kicad_canary, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(kicad_canary, "_run_supervised", supervised)
    monkeypatch.setattr(kicad_canary, "_revalidate_final_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(
        kicad_canary,
        "_validated_artifact_publication",
        lambda *_args: kicad_canary.contextlib.nullcontext(),
    )

    kicad_canary.run_canary(
        project_root=project,
        compilation_database=cdb,
        analyzer=analyzer,
        output_directory=tmp_path / "output",
        gates=(1, "all"),
        gate_timeouts={"1": 1.0, "all": 1.0},
        workers=1,
        analyzer_timeout_seconds=1,
        embedding_dimensions=1,
        queries=(),
        rss_bytes=1,
        database_bytes=1,
        disk_bytes=1,
        no_progress_seconds=1,
        generated_source_roots=(Path("generated"),),
    )

    assert observed_roots == [[], [str(generated_root.resolve())]]


def test_all_gate_requires_explicit_generated_root_before_analyzer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    generated = tmp_path / "build" / "generated.cpp"
    generated.parent.mkdir()
    generated.write_text("int generated() {}\n", encoding="utf-8")
    cdb = generated.parent / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(generated.parent), "file": str(generated), "arguments": ["c++"]}],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    monkeypatch.setattr(
        kicad_canary,
        "_run_supervised",
        lambda *_args: (_ for _ in ()).throw(AssertionError("analyzer must not start")),
    )

    with pytest.raises(ValueError, match="explicit generated source root"):
        kicad_canary.run_canary(
            project_root=project,
            compilation_database=cdb,
            analyzer=analyzer,
            output_directory=tmp_path / "output",
            gates=("all",),
            gate_timeouts={"all": 1.0},
            workers=1,
            analyzer_timeout_seconds=1,
            embedding_dimensions=1,
            queries=(),
            rss_bytes=1,
            database_bytes=1,
            disk_bytes=1,
            no_progress_seconds=1,
        )

    assert (
        kicad_canary.main(
            [
                "--project-root",
                str(project),
                "--compile-commands",
                str(cdb),
                "--gates",
                "all",
                "--preflight-only",
            ]
        )
        == 2
    )
    assert "explicit generated source root" in capsys.readouterr().err


def test_gate_subset_is_deterministic_and_retains_original_working_directory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "src"
    source.mkdir(parents=True)
    for name in ("a.cpp", "b.cpp", "c.cpp"):
        (source / name).write_text(f"int {name[0]}();\n", encoding="utf-8")
    build = project / "build"
    cdb = build / "compile_commands.json"
    _write_cdb(
        cdb,
        [
            {
                "directory": "..",
                "file": f"src/{name}",
                "arguments": ["c++", "-c", f"src/{name}"],
            }
            for name in ("a.cpp", "b.cpp", "c.cpp")
        ],
    )
    inspection = inspect_compilation_database(project, cdb)
    selected = select_gate_entries(inspection, 2)
    subset = tmp_path / "gate" / "compile_commands.json"

    metadata = write_subset_database(cdb, selected, subset)
    payload = json.loads(subset.read_text(encoding="utf-8"))

    assert [entry.raw_index for entry in selected] == [0, 1]
    assert all(Path(entry["directory"]).is_absolute() for entry in payload)
    assert payload[0]["file"] == "src/a.cpp"
    assert len(metadata.sha256) == 64
    assert metadata.raw_entry_count == metadata.normalized_configuration_count == 2


def test_all_gate_retains_raw_duplicates_but_counts_normalized_configurations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inline_validator: None
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    first = project / "first.cpp"
    second = project / "second.cpp"
    first.write_text("int first();\n", encoding="utf-8")
    second.write_text("int second();\n", encoding="utf-8")
    duplicate = {
        "directory": str(project),
        "file": "first.cpp",
        "arguments": ["c++", "-c", "first.cpp"],
    }
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [
            duplicate,
            duplicate,
            {
                "directory": str(project),
                "file": "second.cpp",
                "arguments": ["c++", "-c", "second.cpp"],
            },
        ],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    observed: dict[str, object] = {}

    def supervised(spec_path, gate_directory, _limits, total_tus):
        observed["spec"] = json.loads(spec_path.read_text(encoding="utf-8"))
        observed["total_tus"] = total_tus
        observed["subset"] = json.loads(
            (gate_directory / "compile_commands.json").read_text(encoding="utf-8")
        )
        return {
            "completed_translation_units": 2,
            "process_group_clean": True,
            "analyzer_pipeline": _successful_pipeline_report(
                configuration_count=2,
                slot_count=1,
                max_idle_seconds=1,
            ),
            "database_provenance": {
                "translation_unit_groups": [
                    {
                        "analysis_backend": "clang-libtooling",
                        "advanced_facts_complete": 0,
                        "index_profile": "navigation",
                        "navigation_facts_complete": 1,
                        "cfg_facts_complete": 0,
                        "data_flow_facts_complete": 0,
                        "summary_facts_complete": 0,
                        "translation_units": 2,
                    }
                ],
                "build_variants": [{"name": "default", "index_profile": "navigation"}],
            },
        }

    monkeypatch.setattr(kicad_canary, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(kicad_canary, "_run_supervised", supervised)
    monkeypatch.setattr(kicad_canary, "_revalidate_final_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(
        kicad_canary,
        "_validated_artifact_publication",
        lambda *_args: kicad_canary.contextlib.nullcontext(),
    )

    report = kicad_canary.run_canary(
        project_root=project,
        compilation_database=cdb,
        analyzer=analyzer,
        output_directory=tmp_path / "output",
        gates=("all",),
        gate_timeouts={"all": 1.0},
        workers=1,
        analyzer_timeout_seconds=1,
        embedding_dimensions=1,
        queries=("first",),
        rss_bytes=1,
        database_bytes=1,
        disk_bytes=1,
        no_progress_seconds=1,
    )

    gate = report["gates"][0]
    assert len(observed["subset"]) == 3
    assert observed["total_tus"] == 2
    assert observed["spec"]["translation_units"] == 2
    assert gate["selected_raw_indices"] == [0, 1, 2]
    assert gate["raw_cdb_entries"] == 3
    assert gate["translation_units"] == 2


def test_preflight_rejects_missing_sources_and_numeric_gates_deduplicate_tus(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "same.cpp"
    source.write_text("int same();\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    repeated = {
        "directory": str(project),
        "file": str(source),
        "arguments": ["c++", "-c", str(source)],
    }
    _write_cdb(cdb, [repeated, repeated])

    inspection = inspect_compilation_database(project, cdb)

    assert inspection.public_report()["canonical_translation_unit_count"] == 1
    assert inspection.public_report()["numeric_gate_eligible_count"] == 1
    with pytest.raises(ValueError, match="found 1"):
        select_gate_entries(inspection, 2)

    _write_cdb(
        cdb,
        [
            {
                "directory": str(project),
                "file": "missing.cpp",
                "arguments": ["c++", "-c", "missing.cpp"],
            }
        ],
    )
    with pytest.raises(ValueError, match="source file does not exist"):
        inspect_compilation_database(project, cdb)


def test_baseline_accepts_independent_artifacts_with_equal_semantics(tmp_path: Path) -> None:
    gates: list[dict[str, object]] = []
    for position, timestamp in enumerate(("first", "second")):
        database = tmp_path / str(position) / "index.db"
        database.parent.mkdir()
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE translation_units (id TEXT PRIMARY KEY, indexed_at TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO translation_units VALUES ('tu', ?)", (timestamp,))
            connection.execute(
                "CREATE TABLE build_variants (name TEXT PRIMARY KEY, compilation_database TEXT)"
            )
            connection.execute(
                "INSERT INTO build_variants VALUES ('default', ?)",
                (str(database.parent / "compile_commands.json"),),
            )
        connection.close()
        gates.append(
            {
                "gate": 1,
                "semantic_snapshot": semantic_snapshot(database),
                "database_artifact_sha256": kicad_canary._database_artifact_digest(database).sha256,
            }
        )

    assert gates[0]["semantic_snapshot"] == gates[1]["semantic_snapshot"]
    assert gates[0]["database_artifact_sha256"] != gates[1]["database_artifact_sha256"]
    kicad_canary._compare_baseline_gate(gates[0], gates[1])


@pytest.mark.parametrize("phase", ["index", "embeddings"])
@pytest.mark.parametrize("checkpoint", [601, 1801])
def test_checkpoint_rejects_unknown_remaining_work(
    phase: str, checkpoint: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    events = [{"event": "phase", "name": "index"}]
    if phase == "embeddings":
        events.extend(
            [
                {"event": "tu_staged", "completed": 1},
                {"event": "phase", "name": "embeddings"},
            ]
        )
    phase_events = len(events)
    events.append({"event": "result", "result": {}})

    class Process:
        pid = 999999
        stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in events))
        stderr = io.StringIO("")

        def poll(self):
            return 0

        def wait(self, **_kwargs):
            return 0

    samples = 0

    def database_size(_path):
        nonlocal samples
        samples += 1
        if samples >= phase_events:
            clock.now = checkpoint
        return 0

    monkeypatch.setattr(kicad_canary.time, "monotonic", clock)
    monkeypatch.setattr(kicad_canary.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(kicad_canary, "_database_size", database_size)
    monkeypatch.setattr(kicad_canary, "_directory_size", lambda _path: 0)
    monkeypatch.setattr(
        kicad_canary, "_process_tree_metrics", lambda _pid: kicad_canary._TreeMetrics(0, 0, 0, ())
    )
    monkeypatch.setattr(kicad_canary, "_terminate_process_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        kicad_canary,
        "_AnalyzerTelemetryMonitor",
        lambda **_kwargs: SimpleNamespace(check=lambda: None, success_report=lambda: {}),
    )
    (tmp_path / "spec.json").write_text('{"full_project": true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="projection.*unknown|unknown.*projection"):
        kicad_canary._run_supervised(
            tmp_path / "spec.json", tmp_path, CanaryLimits(wall_seconds=5400), 1
        )


def test_post_worker_validation_cannot_publish_after_gate_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inline_validator: None
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]}],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    clock = _Clock()

    def validation(**_kwargs):
        clock.now = 2

    monkeypatch.setattr(kicad_canary.time, "monotonic", clock)
    monkeypatch.setattr(kicad_canary, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(kicad_canary, "_revalidate_final_artifacts", validation)
    monkeypatch.setattr(kicad_canary, "_validate_profile_provenance", lambda *_args: None)
    monkeypatch.setattr(
        kicad_canary,
        "_validated_artifact_publication",
        lambda *_args: kicad_canary.contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        kicad_canary,
        "_run_supervised",
        lambda *_args: {
            "completed_translation_units": 1,
            "process_group_clean": True,
            "database_provenance": {},
            "analyzer_pipeline": _successful_pipeline_report(
                configuration_count=1, slot_count=1, max_idle_seconds=1
            ),
        },
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="wall|deadline"):
        kicad_canary.run_canary(
            project_root=project,
            compilation_database=cdb,
            analyzer=analyzer,
            output_directory=output,
            gates=(1,),
            gate_timeouts={"1": 1.0},
            workers=1,
            analyzer_timeout_seconds=1,
            embedding_dimensions=1,
            queries=(),
            rss_bytes=1,
            database_bytes=1,
            disk_bytes=1,
            no_progress_seconds=1,
        )
    assert not (output / "gate-1/SUCCESS").exists()


@pytest.mark.parametrize("fault", ["deadline", "rss", "swap"])
def test_validator_subprocess_limits_reap_owned_descendants(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = subprocess.Popen
    metrics = kicad_canary._process_tree_metrics
    workers: list[subprocess.Popen] = []
    descendants: set[int] = set()
    limits = CanaryLimits(wall_seconds=5)

    def launch_validator(_command, **kwargs):
        worker = launch(
            [
                sys.executable,
                "-c",
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
                "start_new_session=True); time.sleep(30)",
            ],
            **kwargs,
        )
        workers.append(worker)
        return worker

    def sampled(root):
        tree = metrics(root)
        if workers and root == workers[0].pid:
            descendants.update(pid for pid in tree.live_pids if pid != root)
        if root == os.getpid() and descendants:
            return kicad_canary.replace(
                tree,
                rss=limits.rss_bytes + 1 if fault == "rss" else tree.rss,
                swap=1 if fault == "swap" else tree.swap,
            )
        return tree

    monkeypatch.setattr(kicad_canary.subprocess, "Popen", launch_validator)
    monkeypatch.setattr(kicad_canary, "_process_tree_metrics", sampled)
    spec = tmp_path / "validation-spec.json"
    spec.write_text(
        json.dumps(
            {
                "supervision_stage": "validation",
                "gate_started_monotonic": kicad_canary.time.monotonic(),
                "total_wall_seconds": 0.4 if fault == "deadline" else 3,
            }
        ),
        encoding="utf-8",
    )
    reason = {"deadline": "end-to-end", "rss": "RSS", "swap": "swap"}[fault]

    with pytest.raises(RuntimeError, match=reason):
        kicad_canary._run_validation_supervised(spec, tmp_path, limits, 1)

    assert descendants
    assert all(worker.poll() is not None for worker in workers)
    assert all(not _process_group_live(pid) for pid in descendants)
    assert not (tmp_path / "SUCCESS").exists()


def test_validation_child_never_constructs_native_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called: list[str] = []
    spec = tmp_path / "validation-spec.json"
    spec.write_text('{"supervision_stage":"validation"}', encoding="utf-8")
    monkeypatch.setattr(
        kicad_canary, "_validate_gate_spec", lambda _spec: called.append("validator")
    )
    monkeypatch.setattr(
        kicad_canary,
        "NativeAnalyzerClient",
        lambda *_args, **_kwargs: pytest.fail("validator must not start a producer"),
    )

    assert kicad_canary._run_worker(spec) == 0
    assert called == ["validator"]
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result == {"event": "result", "result": {"validated": True}}


@pytest.mark.parametrize("missing", ["database", "disk", "both"])
def test_all_cli_requires_explicit_artifact_budgets_before_run(
    missing: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(kicad_canary, "run_canary", lambda **_kwargs: pytest.fail("must not run"))
    args = [
        "--project-root",
        "/unused/project",
        "--compile-commands",
        "/unused/cdb.json",
        "--clang-analyzer",
        "/unused/analyzer",
        "--output-directory",
        "/unused/output",
        "--gates",
        "all",
    ]
    if missing == "database":
        args.extend(["--disk-limit-mib", "65536"])
    elif missing == "disk":
        args.extend(["--database-limit-mib", "49152"])
    assert kicad_canary.main(args) == 2
    assert "explicit" in capsys.readouterr().err


def test_sample_worker_and_total_deadlines_remain_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(kicad_canary, "run_canary", lambda **kwargs: observed.update(kwargs) or {})
    assert (
        kicad_canary.main(
            [
                "--project-root",
                "/unused/project",
                "--compile-commands",
                "/unused/cdb.json",
                "--clang-analyzer",
                "/unused/analyzer",
                "--output-directory",
                "/unused/output",
                "--gates",
                "32",
            ]
        )
        == 0
    )
    assert observed["gate_timeouts"]["32"] == 150
    assert observed["total_gate_timeouts"]["32"] == 180
    assert observed["database_bytes"] == 550 * 1024**2
    assert observed["disk_bytes"] == 1024**3


def test_full_database_provenance_requires_clang_and_every_deep_coverage_flag(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE translation_units (
                analysis_backend TEXT,
                advanced_facts_complete INTEGER,
                index_profile TEXT,
                navigation_facts_complete INTEGER,
                cfg_facts_complete INTEGER,
                data_flow_facts_complete INTEGER,
                summary_facts_complete INTEGER
            )
            """
        )
        connection.execute(
            "INSERT INTO translation_units VALUES ('clang-libtooling', 1, 'full', 1, 1, 1, 1)"
        )
        connection.execute("CREATE TABLE build_variants (name TEXT, index_profile TEXT)")
        connection.execute("INSERT INTO build_variants VALUES ('default', 'full')")

    provenance = database_provenance(database)

    _validate_profile_provenance(provenance, IndexProfile.FULL, 1)
    assert provenance["translation_unit_groups"][0]["advanced_facts_complete"] == 1


def test_baseline_comparison_pins_revision_cdb_selection_semantics_and_ranking() -> None:
    report = {
        "engine_commit": "engine",
        "project_commit": "project",
        "profile": "navigation",
        "workers": 8,
        "embedding_dimensions": 32,
        "input": {"cdb_sha256": "cdb"},
        "gates": [
            {
                "gate": 1,
                "selection": "project_source_prefix",
                "raw_cdb_entries": 1,
                "translation_units": 1,
                "selected_raw_indices": [16],
                "subset_cdb_sha256": "subset",
                "semantic_snapshot": {"digest": "facts"},
                "rankings": {"query": ["symbol"]},
                "public_orderings": {"query": {"digest": "ordered"}},
                "summary_orderings": {"query": {"digest": "summary-ordered"}},
                "database_provenance": {"coverage": "complete"},
                "database_artifact_sha256": "database",
                "database_sidecar_policy": kicad_canary.DATABASE_ARTIFACT_POLICY,
                "database_integrity": "ok",
                "analyzer": {"sha256": "analyzer"},
            }
        ],
    }
    baseline = json.loads(json.dumps(report))

    _compare_baseline(report, baseline)
    baseline["gates"][0]["selected_raw_indices"] = [0]
    with pytest.raises(RuntimeError, match="selected_raw_indices"):
        _compare_baseline(report, baseline)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda baseline: baseline.update(workers=4), "workers"),
        (lambda baseline: baseline["gates"].clear(), "gate set"),
        (
            lambda baseline: baseline["gates"][0].update(raw_cdb_entries=2),
            "raw_cdb_entries",
        ),
        (
            lambda baseline: baseline["gates"][0].update(translation_units=2),
            "translation_units",
        ),
        (
            lambda baseline: baseline["gates"][0]["analyzer"].update(sha256="different"),
            "analyzer",
        ),
        (
            lambda baseline: baseline["gates"][0]["database_provenance"].update(
                coverage="different"
            ),
            "database_provenance",
        ),
        (
            lambda baseline: baseline["gates"][0].update(database_integrity="different"),
            "database_integrity",
        ),
        (
            lambda baseline: baseline["gates"][0]["public_orderings"]["query"].update(
                digest="different"
            ),
            "public_orderings",
        ),
        (
            lambda baseline: baseline["gates"][0]["summary_orderings"]["query"].update(
                digest="different"
            ),
            "summary_orderings",
        ),
    ],
)
def test_baseline_comparison_requires_exact_provenance_and_gate_set(
    mutation: object, message: str
) -> None:
    report = {
        "engine_commit": "engine",
        "project_commit": "project",
        "profile": "full",
        "workers": 8,
        "embedding_dimensions": 32,
        "input": {"cdb_sha256": "cdb"},
        "gates": [
            {
                "gate": 32,
                "selection": "project_source_prefix",
                "raw_cdb_entries": 32,
                "translation_units": 32,
                "selected_raw_indices": list(range(32)),
                "subset_cdb_sha256": "subset",
                "semantic_snapshot": {"digest": "facts"},
                "rankings": {"query": ["symbol"]},
                "public_orderings": {"query": {"digest": "ordered"}},
                "summary_orderings": {"query": {"digest": "summary-ordered"}},
                "database_provenance": {"coverage": "complete"},
                "database_artifact_sha256": "database",
                "database_sidecar_policy": kicad_canary.DATABASE_ARTIFACT_POLICY,
                "database_integrity": "ok",
                "analyzer": {"sha256": "analyzer"},
            }
        ],
    }
    baseline = json.loads(json.dumps(report))

    mutation(baseline)  # type: ignore[operator]

    with pytest.raises(RuntimeError, match=message):
        _compare_baseline(report, baseline)


def test_ordered_public_result_digest_preserves_list_order() -> None:
    first = {"calls": [{"target": "a"}, {"target": "b"}]}
    reversed_result = {"calls": list(reversed(first["calls"]))}

    assert kicad_canary._ordered_public_result_digest(first) != (
        kicad_canary._ordered_public_result_digest(reversed_result)
    )


def test_public_summary_ordering_uses_service_order_and_baseline_pins_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effects = [{"effect_id": "first"}, {"effect_id": "second"}]

    def collect(effect_order: list[dict[str, str]]) -> dict[str, object]:
        flow = DataFlowResult.model_validate(
            {
                "function_symbol_id": "symbol",
                "scope": {"variants": ["default"], "kind": "single", "label": "default"},
                "truncated": False,
                "analyses": [
                    {
                        "analysis_id": "analysis",
                        "graph_id": "graph",
                        "complete": True,
                        "incomplete_reasons": [],
                        "iteration_count": 1,
                        "provenance": {
                            "build_variant": "default",
                            "build_configuration_id": "configuration",
                            "translation_unit_id": "translation-unit",
                        },
                        "locations": [],
                        "accesses": [],
                        "evidence": [],
                        "summary_complete": True,
                        "summary_incomplete_reasons": [],
                        "effects": [
                            {
                                **effect,
                                "kind": "read",
                                "location_kind": "parameter",
                                "certainty": "certain",
                                "reason": "fixture",
                                "parameter_index": 0,
                                "access_path": [],
                                "is_local": True,
                                "via_callsite_id": None,
                                "target_symbol_id": None,
                            }
                            for effect in effect_order
                        ],
                        "return_origins": [],
                        "interprocedural": [],
                    }
                ],
            }
        )

        class Service:
            def __init__(self) -> None:
                self.requests: list[object] = []

            def control_flow(self, _request: object) -> dict[str, object]:
                return {"blocks": []}

            def data_flow(self, request: object) -> object:
                self.requests.append(request)
                return flow

            def calls(self, _request: object) -> dict[str, object]:
                return {"calls": []}

        service = Service()
        hit = SimpleNamespace(symbol=SimpleNamespace(id="symbol"), score=1.0, source="lexical")
        runtime = SimpleNamespace(
            analysis_service=service,
            query_context=lambda _request: SimpleNamespace(
                context=SimpleNamespace(items=[SimpleNamespace(hit=hit, reason="pinned")])
            ),
        )
        monkeypatch.setattr(
            kicad_canary,
            "build_runtime",
            lambda _config: kicad_canary.contextlib.nullcontext(runtime),
        )
        config = SimpleNamespace(
            build_scope=SimpleNamespace(variants=("default",)),
            index_profile=IndexProfile.FULL,
        )

        _rankings, _public_orderings, summaries = kicad_canary._ranking_canaries(config, ("query",))

        assert len(service.requests) == 1
        assert isinstance(service.requests[0], FlowRequest)
        return summaries

    forward = collect(effects)
    reverse = collect(list(reversed(effects)))
    assert sorted(effect["effect_id"] for effect in effects) == sorted(
        effect["effect_id"] for effect in reversed(effects)
    )
    assert forward != reverse
    with pytest.raises(RuntimeError, match="summary_orderings"):
        kicad_canary._compare_baseline_gate(
            {"gate": 1, "summary_orderings": forward},
            {"gate": 1, "summary_orderings": reverse},
        )


def test_public_summary_ordering_reports_navigation_unavailability() -> None:
    flow = DataFlowResult.model_validate(
        {
            "function_symbol_id": "symbol",
            "scope": {"variants": ["default"], "kind": "single", "label": "default"},
            "analyses": [],
            "truncated": False,
            "available": False,
            "unavailable_reason": "deep facts are not materialized",
            "required_action": "materialize_deep_analysis",
        }
    )

    evidence = kicad_canary._public_summary_ordering(flow, required=False)

    assert evidence == {
        "symbol_id": "symbol",
        "available": False,
        "unavailable_reason": "deep facts are not materialized",
        "required_action": "materialize_deep_analysis",
    }
    with pytest.raises(RuntimeError, match="full-profile public summary"):
        kicad_canary._public_summary_ordering(flow, required=True)


def test_baseline_mismatch_never_publishes_gate_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inline_validator: None
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]}],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    inspection = inspect_compilation_database(project, cdb)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "engine_commit": "revision",
                "project_commit": "revision",
                "profile": "navigation",
                "workers": 1,
                "embedding_dimensions": 1,
                "input": {"cdb_sha256": inspection.sha256},
                "gates": [{"gate": 1}],
            }
        ),
        encoding="utf-8",
    )
    provenance = {
        "translation_unit_groups": [
            {
                "analysis_backend": "clang-libtooling",
                "advanced_facts_complete": 0,
                "index_profile": "navigation",
                "navigation_facts_complete": 1,
                "cfg_facts_complete": 0,
                "data_flow_facts_complete": 0,
                "summary_facts_complete": 0,
                "translation_units": 1,
            }
        ],
        "build_variants": [{"name": "default", "index_profile": "navigation"}],
    }
    monkeypatch.setattr(kicad_canary, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(
        kicad_canary,
        "_run_supervised",
        lambda *_args: {
            "completed_translation_units": 1,
            "process_group_clean": True,
            "analyzer_pipeline": _successful_pipeline_report(
                configuration_count=1,
                slot_count=1,
                max_idle_seconds=1,
            ),
            "database_provenance": provenance,
        },
    )
    monkeypatch.setattr(
        kicad_canary,
        "_compare_baseline_gate",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("parity mismatch")),
    )
    monkeypatch.setattr(kicad_canary, "_revalidate_final_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(
        kicad_canary,
        "_validated_artifact_publication",
        lambda *_args: kicad_canary.contextlib.nullcontext(),
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="parity mismatch"):
        kicad_canary.run_canary(
            project_root=project,
            compilation_database=cdb,
            analyzer=analyzer,
            output_directory=output,
            gates=(1,),
            gate_timeouts={"1": 1.0},
            workers=1,
            analyzer_timeout_seconds=1,
            embedding_dimensions=1,
            queries=("main",),
            rss_bytes=1,
            database_bytes=1,
            disk_bytes=1,
            no_progress_seconds=1,
            baseline_report=baseline,
        )

    failed = output / ".gate-1.failed"
    assert (failed / ".running").is_file()
    assert not (failed / "SUCCESS").exists()


def test_setup_failure_after_running_marker_is_atomically_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]}],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    monkeypatch.setattr(kicad_canary, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(
        kicad_canary,
        "write_subset_database",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("sensitive failure at /private/build/root")
        ),
    )
    monkeypatch.setattr(
        kicad_canary,
        "_run_supervised",
        lambda *_args: pytest.fail("worker must not start after setup failure"),
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="sensitive failure"):
        kicad_canary.run_canary(
            project_root=project,
            compilation_database=cdb,
            analyzer=analyzer,
            output_directory=output,
            gates=(1,),
            gate_timeouts={"1": 1.0},
            workers=1,
            analyzer_timeout_seconds=1,
            embedding_dimensions=1,
            queries=("main",),
            rss_bytes=1,
            database_bytes=1,
            disk_bytes=1,
            no_progress_seconds=1,
        )

    failed = output / ".gate-1.failed"
    assert failed.is_dir()
    assert (failed / ".running").is_file()
    assert not (failed / "SUCCESS").exists()
    failure = (failed / "FAILURE.json").read_text(encoding="utf-8")
    assert "gate_setup" in failure
    assert "RuntimeError" in failure
    assert "private/build/root" not in failure


def test_unclean_process_tree_never_publishes_gate_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]}],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    monkeypatch.setattr(kicad_canary, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(kicad_canary, "_validate_profile_provenance", lambda *_args: None)
    monkeypatch.setattr(
        kicad_canary,
        "_run_supervised",
        lambda *_args: {
            "completed_translation_units": 1,
            "process_group_clean": False,
            "database_provenance": {},
        },
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="process tree"):
        kicad_canary.run_canary(
            project_root=project,
            compilation_database=cdb,
            analyzer=analyzer,
            output_directory=output,
            gates=(1,),
            gate_timeouts={"1": 1.0},
            workers=1,
            analyzer_timeout_seconds=1,
            embedding_dimensions=1,
            queries=("main",),
            rss_bytes=1,
            database_bytes=1,
            disk_bytes=1,
            no_progress_seconds=1,
        )

    failed = output / ".gate-1.failed"
    assert failed.is_dir()
    assert not (failed / "SUCCESS").exists()


def test_missing_analyzer_telemetry_never_publishes_gate_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]}],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    monkeypatch.setattr(kicad_canary, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(kicad_canary, "_validate_profile_provenance", lambda *_args: None)
    monkeypatch.setattr(
        kicad_canary,
        "_run_supervised",
        lambda *_args: {
            "completed_translation_units": 1,
            "process_group_clean": True,
            "database_provenance": {},
        },
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="telemetry"):
        kicad_canary.run_canary(
            project_root=project,
            compilation_database=cdb,
            analyzer=analyzer,
            output_directory=output,
            gates=(1,),
            gate_timeouts={"1": 1.0},
            workers=1,
            analyzer_timeout_seconds=1,
            embedding_dimensions=1,
            queries=("main",),
            rss_bytes=1,
            database_bytes=1,
            disk_bytes=1,
            no_progress_seconds=1,
        )

    failed = output / ".gate-1.failed"
    assert failed.is_dir()
    assert not (failed / "SUCCESS").exists()


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "child-result",
        "database-after-hash",
        "subset-after-hash",
        "wal-after-child-hash",
        "active-writer",
        "writer-before-publish",
    ],
)
def test_parent_revalidation_accepts_exact_evidence_and_rejects_tampering(
    tamper: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inline_validator: None
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]}],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    open_connections: list[sqlite3.Connection] = []

    def supervised(_spec_path, gate_directory, _limits, _total_tus):
        database = gate_directory / "index.db"
        _write_final_canary_database(database)
        writer: sqlite3.Connection | None = None
        if tamper in {"wal-after-child-hash", "active-writer"}:
            writer = sqlite3.connect(database)
            assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
            open_connections.append(writer)
        result = {
            "completed_translation_units": 1,
            "process_group_clean": True,
            "analyzer_pipeline": _successful_pipeline_report(
                configuration_count=1,
                slot_count=1,
                max_idle_seconds=1,
            ),
            "rankings": {},
            "public_orderings": {},
            "summary_orderings": {},
            "semantic_snapshot": semantic_snapshot(database),
            "database_provenance": database_provenance(database),
            "database_artifact_sha256": kicad_canary._database_artifact_digest(database).sha256,
            "database_sidecar_policy": kicad_canary.DATABASE_ARTIFACT_POLICY,
            "database_integrity": "ok",
            "analyzer": {"sha256": kicad_canary._sha256(analyzer)},
        }
        if tamper == "child-result":
            result["semantic_snapshot"] = {"digest": "forged-child-result"}
        elif tamper == "database-after-hash":
            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE translation_units SET navigation_facts_complete = 0")
        elif tamper == "subset-after-hash":
            subset = gate_directory / "compile_commands.json"
            subset.write_text(subset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        elif tamper == "wal-after-child-hash":
            assert writer is not None
            writer.execute("UPDATE translation_units SET indexed_at = 'after'")
            writer.commit()
        elif tamper == "active-writer":
            assert writer is not None
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE translation_units SET indexed_at = 'after'")
        return result

    monkeypatch.setattr(kicad_canary, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(kicad_canary, "_run_supervised", supervised)
    monkeypatch.setattr(kicad_canary, "_ranking_canaries", lambda *_args: ({}, {}, {}))
    output = tmp_path / "output"
    if tamper == "writer-before-publish":
        original_sha256 = kicad_canary._sha256
        source_hash_count = 0

        def sha256_with_late_writer(path: Path) -> str:
            nonlocal source_hash_count
            result = original_sha256(path)
            if path.resolve() == cdb.resolve():
                source_hash_count += 1
                if source_hash_count == 3:
                    writer = sqlite3.connect(output / "gate-1" / "index.db")
                    writer.execute("BEGIN IMMEDIATE")
                    writer.execute("UPDATE translation_units SET indexed_at = 'after'")
                    open_connections.append(writer)
            return result

        monkeypatch.setattr(kicad_canary, "_sha256", sha256_with_late_writer)

    def run() -> dict[str, object]:
        return kicad_canary.run_canary(
            project_root=project,
            compilation_database=cdb,
            analyzer=analyzer,
            output_directory=output,
            gates=(1,),
            gate_timeouts={"1": 1.0},
            workers=1,
            analyzer_timeout_seconds=1,
            embedding_dimensions=1,
            queries=(),
            rss_bytes=1,
            database_bytes=1,
            disk_bytes=1,
            no_progress_seconds=1,
        )

    try:
        if tamper is None:
            run()
            assert (output / "gate-1" / "SUCCESS").is_file()
            return

        with pytest.raises(RuntimeError, match="parent artifact revalidation"):
            run()
    finally:
        for connection in open_connections:
            connection.rollback()
            connection.close()

    failed = output / ".gate-1.failed"
    assert failed.is_dir()
    assert not (failed / "SUCCESS").exists()


def test_limits_fail_immediately_on_swap_or_resource_growth() -> None:
    limits = CanaryLimits(
        wall_seconds=150.0,
        rss_bytes=2_500 * 1024**2,
        database_bytes=550 * 1024**2,
        disk_bytes=1024**3,
        no_progress_seconds=10.0,
    )
    assert limits.violation(elapsed=1, rss=1, swap=4096, database=1, disk=1) == "swap used"
    assert "RSS" in (
        limits.violation(elapsed=1, rss=limits.rss_bytes + 1, swap=0, database=1, disk=1) or ""
    )
    assert "database" in (
        limits.violation(elapsed=1, rss=1, swap=0, database=limits.database_bytes + 1, disk=1) or ""
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    ["wall_seconds", "rss_bytes", "database_bytes", "disk_bytes", "no_progress_seconds"],
)
def test_canary_limits_reject_nonfinite_values(field: str, value: float) -> None:
    arguments = {
        "wall_seconds": 1.0,
        "rss_bytes": 1,
        "database_bytes": 1,
        "disk_bytes": 1,
        "no_progress_seconds": 1.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="finite and positive"):
        CanaryLimits(**arguments)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_timeout_parser_rejects_nonfinite_values(value: str) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        kicad_canary._parse_timeouts(f"1:{value}")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_rejects_nonfinite_limits_before_run(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(
        kicad_canary,
        "run_canary",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("run must not start")),
    )

    assert (
        kicad_canary.main(
            [
                "--project-root",
                str(project),
                "--compile-commands",
                str(tmp_path / "compile_commands.json"),
                "--clang-analyzer",
                str(tmp_path / "analyzer"),
                "--output-directory",
                str(tmp_path / "output"),
                f"--rss-limit-mib={value}",
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workers", 0),
        ("analyzer_timeout_seconds", float("nan")),
        ("embedding_dimensions", 0),
        ("rss_bytes", 0),
        ("database_bytes", 0),
        ("disk_bytes", 0),
        ("no_progress_seconds", float("inf")),
        ("gate_timeouts", {"1": float("-inf")}),
        ("total_gate_timeouts", {"1": float("inf")}),
    ],
)
def test_run_configuration_rejects_invalid_numbers_before_supervisor(
    field: str,
    value: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]}],
    )
    analyzer = tmp_path / "analyzer"
    analyzer.write_text("#!/bin/sh\n", encoding="utf-8")
    analyzer.chmod(0o755)
    arguments: dict[str, object] = {
        "project_root": project,
        "compilation_database": cdb,
        "analyzer": analyzer,
        "output_directory": tmp_path / "output",
        "gates": (1,),
        "gate_timeouts": {"1": 1.0},
        "workers": 1,
        "analyzer_timeout_seconds": 1.0,
        "embedding_dimensions": 1,
        "queries": (),
        "rss_bytes": 1,
        "database_bytes": 1,
        "disk_bytes": 1,
        "no_progress_seconds": 1.0,
    }
    arguments[field] = value
    monkeypatch.setattr(
        kicad_canary,
        "_run_supervised",
        lambda *_args: (_ for _ in ()).throw(AssertionError("supervisor must not start")),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        kicad_canary.run_canary(**arguments)


def test_database_integrity_uses_full_check_and_preserves_foreign_key_check() -> None:
    class Cursor:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    class Connection:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def execute(self, command: str) -> Cursor:
            self.commands.append(command)
            if command == "PRAGMA integrity_check":
                return Cursor([("index entry missing",)])
            return Cursor([])

    connection = Connection()
    with pytest.raises(RuntimeError, match="integrity"):
        kicad_canary._validate_database_integrity(connection)

    assert connection.commands == ["PRAGMA integrity_check", "PRAGMA foreign_key_check"]
    assert "PRAGMA quick_check" not in connection.commands


@pytest.mark.parametrize("suffix", ["-journal", "-unexpected"])
def test_database_artifact_rejects_forbidden_sidecars(tmp_path: Path, suffix: str) -> None:
    database = tmp_path / "index.db"
    _write_final_canary_database(database)
    Path(f"{database}{suffix}").write_bytes(b"unsafe")

    with pytest.raises(RuntimeError, match="rollback journal|unexpected SQLite sidecar"):
        kicad_canary._database_artifact_digest(database)


def test_process_group_cleanup_reaps_worker_descendants() -> None:
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
                "start_new_session=True); "
                "print(child.pid, flush=True); time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert worker.stdout is not None
    child_pid = int(worker.stdout.readline().strip())
    child_group = os.getpgid(child_pid)

    _terminate_process_group(worker)

    assert worker.poll() is not None
    assert not _process_group_live(worker.pid)
    assert not _process_group_live(child_group)


def test_observed_process_identities_are_cumulative() -> None:
    groups: set[kicad_canary._ProcessGroupIdentity] = set()
    processes: set[kicad_canary._ProcessIdentity] = set()
    first_group = kicad_canary._ProcessGroupIdentity(11, 101)
    second_group = kicad_canary._ProcessGroupIdentity(12, 102)
    first_process = kicad_canary._ProcessIdentity(21, 201)
    second_process = kicad_canary._ProcessIdentity(22, 202)

    kicad_canary._remember_process_tree(
        groups,
        processes,
        kicad_canary._TreeMetrics(1, 0, 1, (21,), (first_process,), (first_group,)),
    )
    kicad_canary._remember_process_tree(
        groups,
        processes,
        kicad_canary._TreeMetrics(1, 0, 2, (22,), (second_process,), (second_group,)),
    )

    assert groups == {first_group, second_group}
    assert processes == {first_process, second_process}


def test_supervised_run_rejects_unmeasurable_platform_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kicad_canary, "_linux_supervisor_available", lambda: False)

    with pytest.raises(RuntimeError, match="no analyzer was started"):
        kicad_canary._require_supervisor_platform()


def test_recycled_process_identity_is_not_considered_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = kicad_canary._ProcessIdentity(42, 100)
    monkeypatch.setattr(
        kicad_canary,
        "_read_process_identity",
        lambda _pid: kicad_canary._ProcessIdentity(42, 101),
    )

    assert not kicad_canary._process_identity_live(expected)


def test_preflight_cli_never_requires_or_starts_an_analyzer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    source = project / "main.cpp"
    project.mkdir()
    source.write_text("int main() {}\n", encoding="utf-8")
    cdb = tmp_path / "compile_commands.json"
    _write_cdb(
        cdb,
        [{"directory": str(project), "file": str(source), "arguments": ["c++", str(source)]}],
    )
    monkeypatch.setattr(
        kicad_canary,
        "run_canary",
        lambda **_arguments: (_ for _ in ()).throw(AssertionError("index must not start")),
        raising=False,
    )

    assert (
        kicad_canary.main(
            [
                "--project-root",
                str(project),
                "--compile-commands",
                str(cdb),
                "--preflight-only",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["classification_counts"]["project_source"] == 1
