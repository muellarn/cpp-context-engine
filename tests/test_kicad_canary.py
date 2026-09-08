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

    digest = write_subset_database(cdb, selected, subset)
    payload = json.loads(subset.read_text(encoding="utf-8"))

    assert [entry.raw_index for entry in selected] == [0, 1]
    assert all(Path(entry["directory"]).is_absolute() for entry in payload)
    assert payload[0]["file"] == "src/a.cpp"
    assert len(digest) == 64


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
        "embedding_dimensions": 32,
        "input": {"cdb_sha256": "cdb"},
        "gates": [
            {
                "gate": 1,
                "selection": "project_source_prefix",
                "selected_raw_indices": [16],
                "subset_cdb_sha256": "subset",
                "semantic_snapshot": {"digest": "facts"},
                "rankings": {"query": ["symbol"]},
            }
        ],
    }
    baseline = json.loads(json.dumps(report))

    _compare_baseline(report, baseline)
    baseline["gates"][0]["selected_raw_indices"] = [0]
    with pytest.raises(RuntimeError, match="selected_raw_indices"):
        _compare_baseline(report, baseline)


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
