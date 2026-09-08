from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from cpp_context_engine import kicad_canary
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_semantic_snapshot_ignores_volatile_timestamp_and_database_location(tmp_path: Path) -> None:
    digests: list[str] = []
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
        digests.append(semantic_snapshot(database)["digest"])

    assert digests[0] == digests[1]


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
                "database_provenance": {"coverage": "complete"},
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
            lambda baseline: baseline["gates"][0]["public_orderings"]["query"].update(
                digest="different"
            ),
            "public_orderings",
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
                "database_provenance": {"coverage": "complete"},
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


def test_baseline_mismatch_never_publishes_gate_success(
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
            "database_provenance": provenance,
        },
    )
    monkeypatch.setattr(
        kicad_canary,
        "_compare_baseline_gate",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("parity mismatch")),
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
