from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
from dataclasses import replace
from pathlib import Path

import pytest

from cpp_context_engine.cli import _parser, _resolved_config
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion.compilation_database import (
    CompilationDatabase,
    CompilationDatabaseError,
)
from cpp_context_engine.ingestion.deep import DeepCancellation, DeepMaterializer, DeepRequestControl
from cpp_context_engine.ingestion.native import (
    GENERATED_SOURCE_ROOTS_CAPABILITY,
    REQUIRED_CAPABILITIES,
    AnalyzerProtocolError,
    NativeAnalyzerClient,
    _FactBatchBuilder,
)
from cpp_context_engine.ingestion.protocols import IngestionBatch
from cpp_context_engine.models import (
    BuildConfiguration,
    BuildScope,
    BuildVariant,
    CodeSymbol,
    SourceSpan,
    SymbolKind,
    TranslationUnit,
)
from cpp_context_engine.runtime import build_runtime
from cpp_context_engine.storage import FilesystemSourceReader, SQLiteStore
from cpp_context_engine.storage.source import SourceReadError


def _source_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    source = project / "src" / "main.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("int project_entry();\n", encoding="utf-8")
    build = tmp_path / "build"
    generated = build / "generated"
    generated.mkdir(parents=True)
    generated_source = generated / "messages.pb.cc"
    generated_source.write_text("int generated_entry() { return 1; }\n", encoding="utf-8")
    return project, source, build, generated_source


def _write_cdb(path: Path, sources: list[Path]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "directory": str(path.parent),
                    "file": str(source),
                    "arguments": ["clang++", "-std=c++20", "-c", str(source)],
                }
                for source in sources
            ]
        ),
        encoding="utf-8",
    )


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _provenance_batch(variant: BuildVariant, sources: tuple[Path, ...]) -> IngestionBatch:
    configurations = tuple(
        BuildConfiguration(
            id=f"configuration-{variant.name}-{index}",
            source_path=source,
            directory=variant.compilation_database.parent,
            arguments=("clang++", "-c", str(source)),
            command_hash=f"command-{variant.name}-{index}",
            build_variant=variant.name,
            generated_source_roots=variant.generated_source_roots,
        )
        for index, source in enumerate(sources)
    )
    units = tuple(
        TranslationUnit(
            id=f"unit-{variant.name}-{index}",
            build_configuration_id=configuration.id,
            source_path=configuration.source_path,
            content_hash=f"content-{variant.name}-{index}",
            build_variant=variant.name,
        )
        for index, configuration in enumerate(configurations)
    )
    return IngestionBatch(configurations, units, (), (), (), build_variants=(variant,))


def test_build_variant_canonicalizes_relative_generated_roots_against_its_cdb(
    tmp_path: Path,
) -> None:
    _project, _source, build, _generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    cdb.write_text("[]", encoding="utf-8")

    variant = BuildVariant(
        "debug",
        cdb,
        generated_source_roots=(Path("generated"), build / "generated"),
    )

    assert variant.generated_source_roots == ((build / "generated").resolve(),)


def test_build_variant_rejects_nonexistent_file_and_filesystem_generated_roots(
    tmp_path: Path,
) -> None:
    _project, _source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    cdb.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must exist"):
        BuildVariant("missing", cdb, generated_source_roots=(Path("missing"),))
    with pytest.raises(ValueError, match="directory"):
        BuildVariant("file", cdb, generated_source_roots=(generated_source,))
    with pytest.raises(ValueError, match="filesystem root"):
        BuildVariant("broad", cdb, generated_source_roots=(Path(os.sep),))


def test_environment_and_cli_bind_generated_roots_to_exact_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    other_build = tmp_path / "other-build"
    other_generated = other_build / "generated"
    other_generated.mkdir(parents=True)
    alpha_cdb = build / "compile_commands.json"
    beta_cdb = other_build / "compile_commands.json"
    alpha_cdb.write_text("[]", encoding="utf-8")
    beta_cdb.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("CPP_CONTEXT_PROJECT_ROOT", str(project))
    monkeypatch.setenv("CPP_CONTEXT_BUILDS", f"alpha={alpha_cdb},beta={beta_cdb}")
    monkeypatch.setenv(
        "CPP_CONTEXT_GENERATED_SOURCE_ROOTS",
        f"alpha=generated,beta={other_generated}",
    )

    environment = AppConfig.from_environment(cwd=project)
    assert environment.build_variants[0].generated_source_roots == (
        generated_source.parent.resolve(),
    )
    assert environment.build_variants[1].generated_source_roots == (other_generated.resolve(),)

    args = _parser().parse_args(
        [
            "index",
            str(project),
            "--build",
            f"alpha={alpha_cdb}",
            "--build",
            f"beta={beta_cdb}",
            "--generated-source-root",
            "alpha=generated",
        ]
    )
    configured = _resolved_config(args)
    assert configured.build_variants[0].generated_source_roots == (
        generated_source.parent.resolve(),
    )
    assert configured.build_variants[1].generated_source_roots == ()


def test_generated_root_configuration_rejects_ambiguous_or_unknown_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    cdb.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("CPP_CONTEXT_PROJECT_ROOT", str(project))
    monkeypatch.setenv("CPP_CONTEXT_BUILDS", f"alpha={cdb},beta={cdb}")
    monkeypatch.setenv("CPP_CONTEXT_GENERATED_SOURCE_ROOTS", str(generated_source.parent))
    with pytest.raises(ValueError, match="requires NAME=PATH"):
        AppConfig.from_environment(cwd=project)

    monkeypatch.setenv("CPP_CONTEXT_GENERATED_SOURCE_ROOTS", f"other={generated_source.parent}")
    with pytest.raises(ValueError, match="unknown or invalid build"):
        AppConfig.from_environment(cwd=project)


def test_cdb_requires_explicit_generated_root_and_binds_it_into_configuration_identity(
    tmp_path: Path,
) -> None:
    project, source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    _write_cdb(cdb, [source, generated_source])

    with pytest.raises(CompilationDatabaseError, match="authorized source roots"):
        CompilationDatabase.load(cdb, project_root=project)
    database = CompilationDatabase.load(
        cdb,
        project_root=project,
        generated_source_roots=(generated_source.parent,),
    )
    without_boundary = CompilationDatabase.load(cdb)

    assert len(database.configurations) == 2
    assert database.configurations[1].generated_source_roots == (generated_source.parent.resolve(),)
    assert database.configurations[1].id != without_boundary.configurations[1].id

    with pytest.raises(ValueError, match="project source root must be a bounded directory"):
        CompilationDatabase.load(cdb, project_root=Path(os.sep))


@pytest.mark.parametrize("escape_kind", ["traversal", "symlink"])
def test_cdb_rejects_generated_source_escape(tmp_path: Path, escape_kind: str) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    unrelated = tmp_path / "other-project" / "secret.cc"
    unrelated.parent.mkdir()
    unrelated.write_text("int secret();\n", encoding="utf-8")
    if escape_kind == "traversal":
        escaped = generated_source.parent / ".." / ".." / "other-project" / "secret.cc"
    else:
        escaped = generated_source.parent / "linked.cc"
        escaped.symlink_to(unrelated)
    cdb = build / "compile_commands.json"
    _write_cdb(cdb, [escaped])

    with pytest.raises(CompilationDatabaseError, match="authorized source roots"):
        CompilationDatabase.load(
            cdb,
            project_root=project,
            generated_source_roots=(generated_source.parent,),
        )


def test_native_request_carries_only_the_configuration_generated_roots(tmp_path: Path) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    _write_cdb(cdb, [generated_source])
    configuration = CompilationDatabase.load(
        cdb,
        project_root=project,
        generated_source_roots=(generated_source.parent,),
    ).configurations[0]
    captured = tmp_path / "captured.json"
    capabilities = sorted((*REQUIRED_CAPABILITIES, GENERATED_SOURCE_ROOTS_CAPABILITY))
    hello = {
        "type": "hello",
        "protocol": "cpp-context-clang-facts",
        "protocol_version": 5,
        "analyzer_version": "test",
        "clang_major": 18,
        "capabilities": capabilities,
    }
    analyzer = _script(
        tmp_path / "fake-analyzer",
        f"""import json, pathlib, sys
requests = [json.loads(line) for line in sys.stdin]
hello = {hello!r}
print(json.dumps(hello))
if len(requests) > 1:
    pathlib.Path({str(captured)!r}).write_text(json.dumps(requests[1]))
    print(json.dumps({{"type": "begin", "request_id": requests[1]["request_id"]}}))
    print(json.dumps({{
        "type": "complete",
        "request_id": requests[1]["request_id"],
        "success": True,
    }}))
""",
    )

    NativeAnalyzerClient(analyzer).analyze(project, configuration)
    request = json.loads(captured.read_text(encoding="utf-8"))

    assert request["generated_source_roots"] == [str(generated_source.parent.resolve())]


def test_native_rejects_generated_roots_when_companion_lacks_capability(
    tmp_path: Path,
) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    _write_cdb(cdb, [generated_source])
    configuration = CompilationDatabase.load(
        cdb,
        project_root=project,
        generated_source_roots=(generated_source.parent,),
    ).configurations[0]
    hello = {
        "type": "hello",
        "protocol": "cpp-context-clang-facts",
        "protocol_version": 5,
        "analyzer_version": "test",
        "clang_major": 18,
        "capabilities": sorted(REQUIRED_CAPABILITIES),
    }
    analyzer = _script(
        tmp_path / "old-analyzer",
        f"import json\nprint(json.dumps({hello!r}))\n",
    )

    with pytest.raises(AnalyzerProtocolError, match="does not support"):
        NativeAnalyzerClient(analyzer).analyze(project, configuration)


def test_native_rejects_source_escape_before_spawning_companion(tmp_path: Path) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    outside = tmp_path / "other-project" / "outside.cc"
    outside.parent.mkdir()
    outside.write_text("int outside();\n", encoding="utf-8")
    marker = tmp_path / "spawned"
    analyzer = _script(
        tmp_path / "must-not-run",
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
    )
    configuration = BuildConfiguration(
        id="escaped",
        source_path=outside,
        directory=build,
        arguments=("clang++", "-c", str(outside)),
        command_hash="escaped",
        generated_source_roots=(generated_source.parent,),
    )

    with pytest.raises(AnalyzerProtocolError, match="outside the authorized source roots"):
        NativeAnalyzerClient(analyzer).analyze(project, configuration)
    assert not marker.exists()


def test_deep_materializer_reloads_exact_generated_boundary_for_selected_build(
    tmp_path: Path,
) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    _write_cdb(cdb, [generated_source])
    variant = BuildVariant("generated", cdb, generated_source_roots=(generated_source.parent,))
    config = AppConfig(
        project_root=project,
        index_directory=tmp_path,
        database_path=tmp_path / "index.db",
        compilation_database=cdb,
        build_variants=(variant,),
        build_scope=BuildScope.single("generated"),
        embedding_dimensions=8,
    )
    control = DeepRequestControl(time.monotonic(), 5, DeepCancellation())

    with SQLiteStore(config.database_path, project_root=project) as store:
        configurations = DeepMaterializer(config, store)._load_configurations(  # noqa: SLF001
            config.build_scope, control
        )

    assert tuple(configurations.values())[0].generated_source_roots == (
        generated_source.parent.resolve(),
    )


def test_python_fact_boundary_accepts_generated_facts_but_rejects_other_host_files(
    tmp_path: Path,
) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    _write_cdb(cdb, [generated_source])
    configuration = CompilationDatabase.load(
        cdb,
        project_root=project,
        generated_source_roots=(generated_source.parent,),
    ).configurations[0]
    facts = (
        {
            "type": "fact",
            "fact": "file",
            "key": "file:@generated/0/messages.pb.cc",
            "path": str(generated_source),
        },
        {
            "type": "fact",
            "fact": "symbol",
            "key": "usr:generated_entry",
            "qualified_name": "generated_entry",
            "kind": "function",
            "span": {
                "path": str(generated_source),
                "start_line": 1,
                "end_line": 1,
                "start_column": 1,
                "end_column": 36,
            },
            "signature": "int generated_entry()",
            "documentation": "",
            "source_text": "int generated_entry() { return 1; }",
            "metadata": {"is_definition": True},
        },
    )

    batch = _FactBatchBuilder(project.resolve(), configuration).build(facts)

    assert {symbol.qualified_name for symbol in batch.symbols} >= {
        "@generated/0/messages.pb.cc",
        "generated_entry",
    }
    outside = tmp_path / "outside.cc"
    outside.write_text("int outside();\n", encoding="utf-8")
    escaped = ({**facts[0], "path": str(outside)},)
    with pytest.raises(RuntimeError, match="outside the authorized source roots"):
        _FactBatchBuilder(project.resolve(), configuration).build(escaped)


def test_source_reader_requires_exact_build_scoped_provenance_for_generated_files(
    tmp_path: Path,
) -> None:
    project, _source, _build, generated_source = _source_tree(tmp_path)
    neighbor = generated_source.parent / "neighbor.cc"
    neighbor.write_text("int neighbor();\n", encoding="utf-8")
    symbol = CodeSymbol(
        id="generated",
        qualified_name="generated_entry",
        kind=SymbolKind.FUNCTION,
        span=SourceSpan(generated_source, 1, 1),
    )
    reader = FilesystemSourceReader(
        project,
        generated_source_roots=(generated_source.parent,),
        allowed_paths=frozenset({generated_source}),
    )

    assert "generated_entry" in reader.read_symbol(symbol)
    with pytest.raises(SourceReadError, match="not present in the selected build provenance"):
        reader.read_symbol(
            CodeSymbol(
                id="neighbor",
                qualified_name="neighbor",
                kind=SymbolKind.FUNCTION,
                span=SourceSpan(neighbor, 1, 1),
            )
        )


def test_generated_roots_and_known_files_round_trip_per_project_and_build(tmp_path: Path) -> None:
    project, source, build, generated_source = _source_tree(tmp_path)
    other_build = tmp_path / "other-build"
    other_generated = other_build / "generated"
    other_generated.mkdir(parents=True)
    other_source = other_generated / "other.cc"
    other_source.write_text("int other();\n", encoding="utf-8")
    foreign_project = tmp_path / "foreign-project"
    foreign_project.mkdir()
    foreign_source = foreign_project / "foreign.cc"
    foreign_source.write_text("int foreign();\n", encoding="utf-8")
    alpha_cdb = build / "compile_commands.json"
    beta_cdb = other_build / "compile_commands.json"
    _write_cdb(alpha_cdb, [source, generated_source])
    _write_cdb(beta_cdb, [source, other_source])
    alpha = BuildVariant("alpha", alpha_cdb, generated_source_roots=(generated_source.parent,))
    beta = BuildVariant("beta", beta_cdb, generated_source_roots=(other_generated,))
    foreign = BuildVariant("alpha", alpha_cdb)
    database = tmp_path / "index.db"

    with SQLiteStore(database, project_root=project) as store:
        store.apply_ingestion(
            project,
            _provenance_batch(alpha, (source, generated_source)),
            build_variant=alpha,
        )
        store.apply_ingestion(
            project,
            _provenance_batch(beta, (source, other_source)),
            build_variant=beta,
        )
        store.apply_ingestion(
            foreign_project,
            _provenance_batch(foreign, (foreign_source,)),
            build_variant=foreign,
        )

        variants = {variant.name: variant for variant in store.build_variants(project)}
        assert variants["alpha"].generated_source_roots == (generated_source.parent.resolve(),)
        assert variants["beta"].generated_source_roots == (other_generated.resolve(),)
        assert store.source_paths(project, BuildScope.single("alpha")) == frozenset(
            {source.resolve(), generated_source.resolve()}
        )
        assert other_source.resolve() not in store.source_paths(project, BuildScope.single("alpha"))
        assert foreign_source.resolve() not in store.source_paths(
            project, BuildScope.single("alpha")
        )


def test_runtime_rejects_generated_root_binding_that_differs_from_persisted_build(
    tmp_path: Path,
) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    _write_cdb(cdb, [generated_source])
    persisted = BuildVariant("default", cdb, generated_source_roots=(generated_source.parent,))
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=project) as store:
        store.apply_ingestion(
            project,
            _provenance_batch(persisted, (generated_source,)),
            build_variant=persisted,
        )
    mismatched = BuildVariant("default", cdb)
    config = AppConfig(
        project_root=project,
        index_directory=tmp_path,
        database_path=database,
        compilation_database=cdb,
        build_variants=(mismatched,),
        embedding_dimensions=8,
    )

    with pytest.raises(ValueError, match="generated-source roots do not match"):
        build_runtime(config)


def test_storage_rejects_mismatched_build_boundary_atomically(tmp_path: Path) -> None:
    project, _source, build, generated_source = _source_tree(tmp_path)
    cdb = build / "compile_commands.json"
    _write_cdb(cdb, [generated_source])
    variant = BuildVariant("default", cdb, generated_source_roots=(generated_source.parent,))
    batch = _provenance_batch(variant, (generated_source,))
    mismatched = replace(
        batch,
        build_configurations=(replace(batch.build_configurations[0], generated_source_roots=()),),
    )
    with SQLiteStore(tmp_path / "index.db", project_root=project) as store:
        with pytest.raises(ValueError, match="source boundary does not match"):
            store.apply_ingestion(project, mismatched, build_variant=variant)
        assert store.build_variants(project) == ()


def test_schema_14_upgrades_generated_root_provenance_without_rewriting_old_builds(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=project):
        pass
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            DROP TABLE deep_materialization_units;
            DROP TABLE deep_tu_cache;
            DROP TABLE deep_materializations;
            """
        )
        connection.execute("ALTER TABLE build_variants DROP COLUMN generated_source_roots_json")
        connection.execute("PRAGMA user_version = 14")

    with SQLiteStore(database, project_root=project) as migrated:
        columns = {
            row["name"]
            for row in migrated._connection.execute("PRAGMA table_info(build_variants)")  # noqa: SLF001
        }
        tables = {
            row["name"]
            for row in migrated._connection.execute(  # noqa: SLF001
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = migrated._connection.execute("PRAGMA user_version").fetchone()[0]  # noqa: SLF001

    assert "generated_source_roots_json" in columns
    assert {
        "deep_materializations",
        "deep_tu_cache",
        "deep_materialization_units",
    } <= tables
    assert version == 16


@pytest.mark.parametrize("stage", ["column", "publication"])
def test_generated_root_migration_failure_rolls_back_schema(tmp_path: Path, stage: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    database = tmp_path / "index.db"
    with SQLiteStore(database, project_root=project):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE build_variants DROP COLUMN generated_source_roots_json")
        connection.execute("PRAGMA user_version = 15")

    store = SQLiteStore.__new__(SQLiteStore)
    store.path = database
    store.project_root = project
    store.build_scope = BuildScope.single()
    store._connection = sqlite3.connect(database)  # noqa: SLF001
    store._connection.row_factory = sqlite3.Row  # noqa: SLF001

    def fail_at(candidate: str) -> None:
        if candidate == stage:
            raise RuntimeError(f"injected {stage} failure")

    store._generated_roots_migration_checkpoint = fail_at  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match=f"injected {stage} failure"):
        store._migrate_v16()  # noqa: SLF001
    assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 15  # noqa: SLF001
    assert "generated_source_roots_json" not in {  # noqa: SLF001
        row[1] for row in store._connection.execute("PRAGMA table_info(build_variants)")
    }
    store._connection.close()  # noqa: SLF001
