from __future__ import annotations

import json

from cpp_context_engine.cli import main


def test_main_without_command_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "Compiler-aware retrieval" in capsys.readouterr().out


def test_doctor_json_reports_supported_runtime(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["doctor", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["python_supported"] is True
    assert report["project_root"] == str(tmp_path)
