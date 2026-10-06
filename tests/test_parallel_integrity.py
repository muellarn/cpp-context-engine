import sqlite3
import sys
import time

import pytest
from test_parallel_semantic import _minimal

from cpp_context_engine import kicad_canary as canary


def _checks(path, owner, deadline):
    combined = getattr(canary, "_parallel_database_checks", None)
    if combined is None:
        # Baseline reproduces the producer's existing serial checks, so RED
        # measures missing overlap rather than a missing helper attribute.
        snapshot = canary._parallel_semantic_snapshot(path, owner, deadline_monotonic=deadline)
        return snapshot, canary._validate_database_integrity(owner)
    return combined(path, owner, deadline_monotonic=deadline)


@pytest.mark.parametrize("journal", ["DELETE", "WAL"])
def test_integrity_overlaps_both_semantic_readers_under_writer_exclusion(
    tmp_path, monkeypatch, journal
):
    path = tmp_path / "index.db"
    _minimal(path, journal)
    expected = canary.semantic_snapshot(path)
    original = canary.subprocess.Popen
    started = []

    def observed(command, **kwargs):
        with (
            sqlite3.connect(path, timeout=0) as writer,
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            writer.execute("UPDATE symbols SET payload='racing'")
        process = original(command, **kwargs)
        started.append((command, process))
        return process

    monkeypatch.setattr(canary.subprocess, "Popen", observed)
    with canary._database_writer_exclusion(path) as owner:
        assert _checks(path, owner, time.monotonic() + 20) == (expected, "ok")
        assert len(started) == 3, "full integrity must overlap both semantic readers"
        assert sum("--_integrity-spec" in command for command, _ in started) == 1
        assert all(process.poll() == 0 for _, process in started)
        with (
            sqlite3.connect(path, timeout=0) as writer,
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            writer.execute("UPDATE symbols SET payload='racing'")
    with sqlite3.connect(path, timeout=0) as writer:
        writer.execute("UPDATE symbols SET payload='after cleanup'")


@pytest.mark.parametrize("kind", ["foreign_key", "integrity"])
def test_full_integrity_reader_rejects_corrupt_database(tmp_path, kind):
    path = tmp_path / "index.db"
    _minimal(path)
    with sqlite3.connect(path) as connection:
        if kind == "foreign_key":
            connection.execute("CREATE TABLE parent(id PRIMARY KEY)")
            connection.execute("CREATE TABLE child(id REFERENCES parent(id))")
            connection.execute("INSERT INTO child VALUES(999)")
        else:
            connection.execute("CREATE TABLE required(value)")
            connection.execute("INSERT INTO required VALUES(NULL)")
            connection.execute("PRAGMA writable_schema=ON")
            connection.execute(
                "UPDATE sqlite_master SET sql='CREATE TABLE required(value NOT NULL)' "
                "WHERE name='required'"
            )
            connection.execute("PRAGMA writable_schema=OFF")
    with sqlite3.connect(path) as connection:
        if kind == "integrity":
            assert connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]
        else:
            assert connection.execute("PRAGMA foreign_key_check").fetchall()
    with canary._database_writer_exclusion(path) as owner, pytest.raises(RuntimeError):
        _checks(path, owner, time.monotonic() + 20)


@pytest.mark.parametrize("payload", ["{}", '{"integrity":"bad"}', '{"integrity":"ok","extra":1}'])
def test_integrity_response_is_strict_and_readers_are_reaped(tmp_path, monkeypatch, payload):
    path = tmp_path / "index.db"
    _minimal(path)
    original = canary.subprocess.Popen
    processes = []

    def invalid(command, **kwargs):
        if "--_integrity-spec" in command:
            command = [sys.executable, "-c", f"print({payload!r})"]
        process = original(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(canary.subprocess, "Popen", invalid)
    with canary._database_writer_exclusion(path) as owner:
        with pytest.raises(RuntimeError, match="integrity"):
            _checks(path, owner, time.monotonic() + 20)
        assert len(processes) == 3 and all(p.poll() is not None for p in processes)


@pytest.mark.parametrize("failure", ["startup", "peer", "deadline"])
def test_combined_checks_kill_and_reap_all_readers_before_unlock(tmp_path, monkeypatch, failure):
    path = tmp_path / "index.db"
    _minimal(path)
    original = canary.subprocess.Popen
    processes = []

    def injected(command, **kwargs):
        if "--_integrity-spec" in command:
            if failure == "startup":
                raise OSError("injected third reader start failure")
            if failure == "peer":
                command = [sys.executable, "-c", "raise SystemExit(3)"]
            else:
                command = [sys.executable, "-c", "import time; time.sleep(60)"]
        else:
            command = [sys.executable, "-c", "import time; time.sleep(60)"]
        process = original(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(canary.subprocess, "Popen", injected)
    before = time.monotonic()
    with canary._database_writer_exclusion(path) as owner:
        error = {"startup": OSError, "peer": RuntimeError, "deadline": TimeoutError}[failure]
        with pytest.raises(error):
            _checks(path, owner, before + (0.6 if failure == "deadline" else 20))
        assert len(processes) == (2 if failure == "startup" else 3)
        assert all(p.poll() is not None for p in processes)
        with (
            sqlite3.connect(path, timeout=0) as writer,
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            writer.execute("UPDATE symbols SET payload='racing'")
    assert time.monotonic() - before < 5


def test_integrity_reader_requires_future_finite_deadline(tmp_path):
    path = tmp_path / "index.db"
    _minimal(path)
    for deadline in (-1, float("nan"), float("inf")):
        with pytest.raises(TimeoutError):
            canary._integrity_reader({"database": str(path), "deadline_monotonic": deadline})


@pytest.mark.parametrize("tables", [0, 1])
def test_tiny_databases_still_receive_integrity_checks(tmp_path, tables):
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as connection:
        if tables:
            connection.execute("CREATE TABLE symbols(id)")
    with canary._database_writer_exclusion(path) as owner:
        assert _checks(path, owner, time.monotonic() + 20) == (canary.semantic_snapshot(path), "ok")
