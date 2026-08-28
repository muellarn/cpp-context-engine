from __future__ import annotations

import json

import pytest

from cpp_context_engine.cli import main
from cpp_context_engine.models import IndexProfile


def test_main_without_command_prints_help(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Compiler-aware retrieval" in output
    assert "index" in output
    assert "search" in output
    assert "ask" in output
    assert "serve" in output
    assert "mcp" in output


def test_doctor_json_reports_supported_runtime(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["doctor", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["python_supported"] is True
    assert report["project_root"] == str(tmp_path)
    assert "embedding_provider" in report


@pytest.mark.parametrize(
    "option",
    [
        "--analyzer-max-spool-registries",
        "--analyzer-max-spool-bytes",
        "--analyzer-max-spool-files",
        "--analyzer-max-domain-batches",
    ],
)
def test_cli_rejects_zero_pipeline_limits(option, capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["doctor", option, "0"]) == 2
    assert "must be positive" in capsys.readouterr().err


def test_search_without_an_index_has_actionable_error(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["search", "PacketParser"]) == 2

    assert "run 'cpp-context index' first" in capsys.readouterr().err


def test_mcp_subcommand_dispatches_without_secret_flags(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    observed = {}

    def run_mcp(config, args):
        observed["project"] = config.project_root
        observed["transport"] = args.transport
        return 0

    monkeypatch.setattr("cpp_context_engine.cli._run_mcp", run_mcp)

    assert main(["mcp"]) == 0
    assert observed == {"project": tmp_path, "transport": "stdio"}


def test_index_accepts_explicit_navigation_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    observed = {}

    def run_index(config, *, as_json):
        observed["profile"] = config.index_profile
        return 0

    monkeypatch.setattr("cpp_context_engine.cli._run_index", run_index)

    assert main(["index", "--profile", "navigation"]) == 0
    assert observed == {"profile": IndexProfile.NAVIGATION}
