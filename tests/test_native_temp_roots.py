import tempfile
from pathlib import Path

from native_cache import NativeFixtureCache

from cpp_context_engine.models import BuildConfiguration


def test_native_cache_preserves_configured_temporary_project_and_siblings(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external.hpp"
    external.write_text("int external();\n")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "value.hpp").write_text("int generated();\n")
    source = project / "main.cpp"
    source.write_text('#include "../external.hpp"\n#include "../generated/value.hpp"\n')
    (project / "alias.hpp").symlink_to(external)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    cache = NativeFixtureCache()
    directory = cache.directory
    try:
        configuration = BuildConfiguration(
            id="temp-roots",
            source_path=source,
            directory=project,
            arguments=("clang++", "-c", str(source)),
            command_hash="temp-roots",
            generated_source_roots=(generated,),
        )
        staged_root, staged_configuration = cache.stage(project, configuration)
        assert staged_root == project
        assert staged_configuration == configuration
        assert (staged_root / "../external.hpp").resolve() == external
        assert (staged_root / "alias.hpp").resolve() == external
        assert cache.load("sample", lambda: {"facts": ["original"]}) == {"facts": ["original"]}
    finally:
        cache.close()
    assert not directory.exists()
    assert source.exists() and external.exists() and generated.exists()


def test_native_cache_artifacts_stay_in_configured_temporary_root(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    cache = NativeFixtureCache()
    try:
        assert cache.directory.parent == tmp_path
    finally:
        cache.close()


def test_native_cache_does_not_relocate_legacy_tmp(tmp_path, monkeypatch):
    alternative = tmp_path / "configured"
    alternative.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(alternative))
    cache = NativeFixtureCache()
    try:
        assert cache.stage_project(Path("/tmp")) == Path("/tmp").resolve()
    finally:
        cache.close()
