from __future__ import annotations

import pytest

from cpp_context_engine.config import AppConfig


def test_environment_configuration_uses_project_local_index(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CPP_CONTEXT_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CPP_CONTEXT_INDEX_DIRECTORY", raising=False)

    config = AppConfig.from_environment(cwd=tmp_path)

    assert config.project_root == tmp_path
    assert config.index_directory == tmp_path / ".cpp-context"


def test_environment_configuration_rejects_non_positive_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CPP_CONTEXT_RETRIEVAL_LIMIT", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        AppConfig.from_environment(cwd=tmp_path)
