import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from cpp_context_engine import summary_input


def test_anonymous_scan_tolerates_permission_error_only_after_process_exit(tmp_path, monkeypatch):
    process = tmp_path / "123"
    descriptors = process / "fd"
    descriptors.mkdir(parents=True)
    (descriptors / "1").touch()
    identity = process / "stat"
    identity.write_text("123 (native worker) S " + "0 " * 18 + "77 0 0\n")
    monkeypatch.setattr(
        summary_input, "Path", lambda value: tmp_path / value.removeprefix("/proc/")
    )
    original_stat = Path.stat

    def exited_stat(path, *args, **kwargs):
        if path.parent == descriptors:
            identity.unlink()
            raise PermissionError("process exited during descriptor access")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", exited_stat)
    assert summary_input._anonymous_bytes((123,)) == 0


@pytest.fixture
def process_files(tmp_path, monkeypatch):
    process = tmp_path / "123"
    (process / "fd").mkdir(parents=True)
    for descriptor in ("1", "2", "3", "4"):
        (process / "fd" / descriptor).touch()
    (process / "stat").write_text("123 (native ) worker) S " + "0 " * 18 + "77 0 0\n")
    monkeypatch.setattr(
        summary_input, "Path", lambda value: tmp_path / value.removeprefix("/proc/")
    )
    return process


@pytest.mark.parametrize("end", ["same", "zombie", "reused", "gone", "malformed", "denied"])
@pytest.mark.parametrize("operation", ["stat", "iterdir"])
def test_scan_failure_requires_positive_original_process_exit(
    process_files, monkeypatch, end, operation
):
    process = process_files
    original = getattr(Path, operation)

    def access_failure(path, *args, **kwargs):
        target = path.parent == process / "fd" if operation == "stat" else path == process / "fd"
        if target:
            identity = process / "stat"
            if end == "gone":
                identity.unlink()
            elif end == "zombie":
                identity.write_text("123 (native) Z " + "0 " * 18 + "77 0\n")
            elif end == "reused":
                identity.write_text("123 (native) S " + "0 " * 18 + "88 0\n")
            elif end == "malformed":
                identity.write_text("invalid identity")
            elif end == "denied":
                read_text = Path.read_text

                def unreadable_identity(item, *a, **kw):
                    if item == identity:
                        raise PermissionError("identity inaccessible")
                    return read_text(item, *a, **kw)

                monkeypatch.setattr(Path, "read_text", unreadable_identity)
            raise PermissionError("descriptor access denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, access_failure)
    if end in {"zombie", "reused", "gone"}:
        assert summary_input._anonymous_bytes((123,)) == 0
    else:
        with pytest.raises(RuntimeError if end == "malformed" else PermissionError):
            summary_input._anonymous_bytes((123,))


def test_scan_counts_anonymous_regular_inodes_once(process_files, monkeypatch):
    original = Path.stat

    def metadata(path, *args, **kwargs):
        if path.parent == process_files / "fd":
            return SimpleNamespace(
                st_mode=stat.S_IFIFO if path.name == "4" else stat.S_IFREG,
                st_nlink=1 if path.name == "3" else 0,
                st_dev=1,
                st_ino=2,
                st_size=1024,
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", metadata)
    assert summary_input._anonymous_bytes((123, 123, 999)) == 1024


def test_scan_discards_successful_reads_after_identity_change(process_files, monkeypatch):
    original = Path.stat

    def metadata(path, *args, **kwargs):
        if path.parent == process_files / "fd":
            (process_files / "stat").write_text("123 (new owner) S " + "0 " * 18 + "88 0\n")
            return SimpleNamespace(st_mode=stat.S_IFREG, st_nlink=0, st_dev=1, st_ino=2, st_size=1024)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", metadata)
    assert summary_input._anonymous_bytes((123,)) == 0
