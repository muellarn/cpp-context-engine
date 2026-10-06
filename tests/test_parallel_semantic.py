import sqlite3
import sys
import time

import pytest
from test_compact_occurrences import _seed

from cpp_context_engine import kicad_canary as canary
from cpp_context_engine.storage.sqlite import SQLiteStore


def _minimal(path, journal="DELETE"):
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA journal_mode={journal}")
        connection.execute("CREATE TABLE symbols(project_id,id,payload,updated_at)")
        connection.execute("CREATE TABLE edges(project_id,id,payload)")
        connection.executemany(
            "INSERT INTO symbols VALUES(?,?,?,?)",
            [(1, "名前", b"\x00\xff", "ignored"), (2, "café", None, "ignored")],
        )
        connection.executemany(
            "INSERT INTO edges VALUES(?,?,?)", [(1, "z", 1.0), (1, "a", 1), (2, "", "\x00")]
        )


def _parallel(path):
    with canary._database_writer_exclusion(path) as owner:
        return canary._parallel_semantic_snapshot(
            path, owner, deadline_monotonic=time.monotonic() + 20
        )


@pytest.mark.parametrize("journal", ["DELETE", "WAL"])
def test_parallel_semantic_preserves_every_value_and_full_digest(tmp_path, monkeypatch, journal):
    path = tmp_path / "index.db"
    _minimal(path, journal)
    expected = canary.semantic_snapshot(path)
    processes = []
    original = canary.subprocess.Popen

    def observed(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(canary.subprocess, "Popen", observed)
    assert _parallel(path) == expected
    assert len(processes) == 2, "run disjoint tables in two fresh interpreters"
    assert all(process.poll() == 0 for process in processes)
    with sqlite3.connect(path, timeout=0) as writer:
        writer.execute("UPDATE symbols SET payload='changed' WHERE project_id=1")
    assert _parallel(path) != expected


def test_parallel_semantic_includes_normalized_views(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteStore(path, project_root=tmp_path) as store:
        _seed(store, tmp_path)
        store.put_embedding("shared-symbol", "model", [1.0, 0.0])
    expected = canary.semantic_snapshot(path)
    assert expected["counts"]["occurrences"] == 1
    assert expected["counts"]["embedding_vectors"] == 1
    assert _parallel(path) == expected


@pytest.mark.parametrize("journal", ["DELETE", "WAL"])
def test_parallel_semantic_blocks_writers_until_children_end(tmp_path, monkeypatch, journal):
    path = tmp_path / "index.db"
    _minimal(path, journal)
    original = canary.subprocess.Popen
    blocked = []

    def observed(*args, **kwargs):
        with (
            sqlite3.connect(path, timeout=0) as writer,
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            writer.execute("UPDATE symbols SET payload='racing'")
        blocked.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(canary.subprocess, "Popen", observed)
    expected = canary.semantic_snapshot(path)
    assert _parallel(path) == expected
    assert len(blocked) == 2
    with sqlite3.connect(path, timeout=0) as writer:
        writer.execute("UPDATE symbols SET payload='after'")


def test_serial_supplied_connection_keeps_uncommitted_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    _minimal(path)
    before = canary.semantic_snapshot(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("generic snapshots must not spawn readers")

    monkeypatch.setattr(canary.subprocess, "Popen", forbidden)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE symbols SET payload='uncommitted'")
        assert canary.semantic_snapshot(path, _connection=connection) != before
        assert connection.in_transaction
        connection.rollback()
    assert canary.semantic_snapshot(path) == before


def test_parallel_semantic_rejects_expired_deadline_before_spawn(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    _minimal(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("expired verification must not spawn")

    monkeypatch.setattr(canary.subprocess, "Popen", forbidden)
    with canary._database_writer_exclusion(path) as owner, pytest.raises(TimeoutError):
        canary._parallel_semantic_snapshot(path, owner, deadline_monotonic=time.monotonic() - 1)


def test_parallel_semantic_refuses_uncommitted_owner(tmp_path):
    path = tmp_path / "index.db"
    _minimal(path)
    with canary._database_writer_exclusion(path) as owner:
        owner.execute("UPDATE symbols SET payload='uncommitted'")
        with pytest.raises(ValueError, match="write"):
            canary._parallel_semantic_snapshot(
                path, owner, deadline_monotonic=time.monotonic() + 20
            )
        assert owner.in_transaction


@pytest.mark.parametrize("kind", ["empty", "single"])
def test_parallel_semantic_handles_missing_and_empty_tables(tmp_path, kind):
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as connection:
        if kind == "single":
            connection.execute("CREATE TABLE symbols(project_id,id,payload)")
    assert _parallel(path) == canary.semantic_snapshot(path)


def test_parallel_semantic_reaps_first_child_if_second_spawn_fails(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    _minimal(path)
    original = canary.subprocess.Popen
    started = []

    def fail_second(*args, **kwargs):
        if started:
            raise OSError("injected second child failure")
        process = original(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(canary.subprocess, "Popen", fail_second)
    with pytest.raises(OSError, match="injected"):
        _parallel(path)
    assert len(started) == 1 and started[0].poll() is not None
    with sqlite3.connect(path, timeout=0) as writer:
        writer.execute("UPDATE symbols SET payload='after cleanup'")


@pytest.mark.parametrize(
    "payload", ["not json", "{}", "x" * 65537], ids=["json", "coverage", "size"]
)
def test_parallel_semantic_fails_closed_on_invalid_child_output(tmp_path, monkeypatch, payload):
    path = tmp_path / "index.db"
    _minimal(path)
    original = canary.subprocess.Popen
    started = []

    def invalid(_command, **kwargs):
        process = original([sys.executable, "-c", f"print({payload!r})"], **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(canary.subprocess, "Popen", invalid)
    with pytest.raises((ValueError, RuntimeError)):
        _parallel(path)
    assert all(process.poll() is not None for process in started)


def test_parallel_semantic_deadline_kills_and_reaps_readers(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    _minimal(path)
    original = canary.subprocess.Popen
    started = []

    def blocked(_command, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(canary.subprocess, "Popen", blocked)
    before = time.monotonic()
    with canary._database_writer_exclusion(path) as owner:
        with pytest.raises(TimeoutError):
            canary._parallel_semantic_snapshot(path, owner, deadline_monotonic=before + 0.3)
        assert all(process.poll() is not None for process in started)
        with (
            sqlite3.connect(path, timeout=0) as writer,
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            writer.execute("UPDATE symbols SET payload='racing'")
    assert len(started) == 2 and time.monotonic() - before < 5


@pytest.mark.parametrize("parallel", [False, True])
def test_parallel_semantic_preserves_table_without_stable_columns(tmp_path, parallel):
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE symbols(indexed_at)")
        connection.execute("INSERT INTO symbols VALUES('ignored')")
        if parallel:
            connection.execute("CREATE TABLE edges(id)")
    assert _parallel(path) == canary.semantic_snapshot(path)


@pytest.mark.parametrize("stage", ["count", "single"])
def test_parallel_semantic_interrupts_parent_queries_at_deadline(tmp_path, stage):
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE input(value)")
        connection.executemany("INSERT INTO input VALUES(?)", [(i,) for i in range(50)])
        connection.execute("CREATE VIEW edges AS SELECT value FROM input WHERE slow(value)>=0")
        if stage == "count":
            connection.execute("CREATE TABLE symbols(id)")
    with canary._database_writer_exclusion(path) as owner:
        owner.create_function("slow", 1, lambda value: (time.sleep(0.01), value)[1])
        before = time.monotonic()
        with pytest.raises(TimeoutError):
            canary._parallel_semantic_snapshot(path, owner, deadline_monotonic=before + 0.02)
        assert time.monotonic() - before < 0.25
        assert owner.in_transaction
