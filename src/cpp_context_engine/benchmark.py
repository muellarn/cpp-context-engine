"""Deterministic generated-workload benchmark for the Clang analysis stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cpp_context_engine.api import CallRequest, CfgRequest, FlowRequest, QueryRequest
from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion.native import NativeAnalyzerClient
from cpp_context_engine.models import BuildScope, BuildVariant, GraphDirection, SearchQuery
from cpp_context_engine.runtime import build_runtime, index_project

REPORT_SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
DEFAULT_SEED = 0xCCE2026


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Shape of one deterministic generated C++ workload."""

    translation_units: int
    builds: int
    functions_per_translation_unit: int
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if min(self.translation_units, self.builds, self.functions_per_translation_unit) < 1:
            raise ValueError("workload dimensions must be positive")
        if self.builds > 16:
            raise ValueError("workload cannot exceed the public limit of 16 build variants")
        if self.seed < 0:
            raise ValueError("workload seed must not be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkBudgets:
    """Issue #11 acceptance budgets, expressed in report-native units."""

    cold_index_seconds: float = 180.0
    peak_rss_bytes: int = 2 * 1024**3
    warm_noop_seconds: float = 10.0
    query_p95_seconds: float = 1.0
    database_bytes: int = 1024**3


@dataclass(frozen=True, slots=True)
class GeneratedWorkload:
    project_root: Path
    build_variants: tuple[BuildVariant, ...]
    source_hash: str
    shared_header: Path


SMOKE_WORKLOAD = WorkloadSpec(100, 2, 40)
REFERENCE_WORKLOAD = WorkloadSpec(1_000, 3, 40)
DEFAULT_BUDGETS = BenchmarkBudgets()

_FACT_TABLES = (
    "build_configurations",
    "translation_units",
    "dependencies",
    "symbols",
    "symbol_variants",
    "occurrences",
    "edges",
    "cfg_graphs",
    "cfg_blocks",
    "cfg_elements",
    "cfg_edges",
    "callsites",
    "call_targets",
    "data_flow_analyses",
    "memory_locations",
    "data_accesses",
    "data_flow_evidence",
    "function_summaries",
    "summary_effects",
    "summary_return_origins",
    "call_argument_bindings",
    "call_result_bindings",
    "interprocedural_flows",
    "variant_embeddings",
)


def generate_workload(
    project_root: Path,
    spec: WorkloadSpec,
    *,
    compiler: str = "clang++-18",
) -> GeneratedWorkload:
    """Create a root-independent C++ workload and compilation databases.

    The caller owns ``project_root``. The benchmark passes a fresh temporary
    directory so neither generated sources nor indexes survive the run.
    """

    root = project_root.resolve(strict=False)
    if root.exists() and any(root.iterdir()):
        raise ValueError("generated workload directory must be empty")
    include = root / "include"
    source = root / "src"
    include.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)

    header = include / "benchmark.hpp"
    header.write_text(_header_text(), encoding="utf-8")
    for unit_index in range(spec.translation_units):
        (source / f"unit_{unit_index:04d}.cpp").write_text(
            _translation_unit_text(unit_index, spec.functions_per_translation_unit, spec.seed),
            encoding="utf-8",
        )

    variants: list[BuildVariant] = []
    for build_index in range(spec.builds):
        name = f"build-{build_index}"
        build_directory = root / "build" / name
        build_directory.mkdir(parents=True)
        database = build_directory / "compile_commands.json"
        entries = [
            {
                "directory": "../..",
                "file": f"src/unit_{unit_index:04d}.cpp",
                "arguments": [
                    compiler,
                    "-std=c++17",
                    "-Iinclude",
                    f"-DBENCH_BUILD_VARIANT={build_index}",
                    "-Wno-unused-parameter",
                    "-c",
                    f"src/unit_{unit_index:04d}.cpp",
                ],
            }
            for unit_index in range(spec.translation_units)
        ]
        database.write_text(
            json.dumps(entries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        variants.append(BuildVariant(name, database))

    return GeneratedWorkload(root, tuple(variants), workload_hash(root), header)


def workload_hash(project_root: Path) -> str:
    """Hash generated inputs by relative path and bytes, independent of temp root."""

    digest = hashlib.sha256()
    for path in sorted(item for item in project_root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(project_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _header_text() -> str:
    return """#pragma once

#ifndef BENCH_BUILD_VARIANT
#define BENCH_BUILD_VARIANT 0
#endif

namespace bench {

inline constexpr int build_bias = BENCH_BUILD_VARIANT;

struct Base {
  virtual ~Base() = default;
  virtual int apply(int value) const { return value + 1; }
};

struct Derived final : Base {
  int apply(int value) const override { return value + 2; }
};

template <int Offset>
int transform(int value) {
  return value + Offset + build_bias;
}

inline int direct_target(int value) { return value * 2 + build_bias; }
inline int high_fanout_target(int value) { return value - build_bias; }

}  // namespace bench
"""


def _translation_unit_text(unit_index: int, function_count: int, seed: int) -> str:
    lines = [
        '#include "benchmark.hpp"',
        "",
        f"namespace bench::unit_{unit_index:04d} {{",
        "",
    ]
    for function_index in range(function_count):
        offset = 1 + ((seed + unit_index * 17 + function_index * 31) % 19)
        if function_index == 0:
            lines.extend(
                (
                    f"int function_{function_index:04d}(int input, const Base& polymorphic) {{",
                    "  int value = input + build_bias;",
                    "  if ((value & 1) == 0) {",
                    f"    value = transform<{offset}>(value);",
                    "  } else {",
                    "    value = direct_target(value);",
                    "  }",
                    "  value = polymorphic.apply(value);",
                    "  auto local = [](int item) { return item + 3; };",
                    "  value = local(value);",
                    "  int (*selected)(int) = &high_fanout_target;",
                    "  value = selected(value);",
                    "  for (int iteration = 0; iteration < 2; ++iteration) {",
                    "    value += iteration;",
                    "  }",
                    "  return value;",
                    "}",
                    "",
                )
            )
        else:
            lines.extend(
                (
                    f"int function_{function_index:04d}(int input, const Base& polymorphic) {{",
                    f"  int value = input + {offset} + build_bias;",
                    "  if (value % 3 == 0) {",
                    "    value += 1;",
                    "  }",
                    "  return high_fanout_target(value);",
                    "}",
                    "",
                )
            )
    lines.extend((f"}}  // namespace bench::unit_{unit_index:04d}", ""))
    return "\n".join(lines)


def run_benchmark(
    *,
    spec: WorkloadSpec,
    analyzer: Path,
    compiler: str = "clang++-18",
    query_iterations: int = 50,
    profile: str = "custom",
    budgets: BenchmarkBudgets = DEFAULT_BUDGETS,
    commit: str | None = None,
) -> dict[str, Any]:
    """Run generation, indexing, reindexing, and bounded public queries."""

    if query_iterations < 1:
        raise ValueError("query iterations must be positive")
    analyzer = analyzer.expanduser().resolve(strict=True)
    if not analyzer.is_file() or not os.access(analyzer, os.X_OK):
        raise ValueError("Clang analyzer must be an executable file")

    with tempfile.TemporaryDirectory(
        prefix="cpp-context-benchmark-", dir=benchmark_temp_root()
    ) as temporary:
        _progress("generate workload")
        project = Path(temporary) / "project"
        generated = generate_workload(project, spec, compiler=compiler)
        database = Path(temporary) / "index" / "index.db"
        scope = BuildScope(tuple(item.name for item in generated.build_variants))
        config = AppConfig(
            project_root=project,
            index_directory=database.parent,
            database_path=database,
            compilation_database=generated.build_variants[0].compilation_database,
            build_variants=generated.build_variants,
            build_scope=scope,
            clang_analyzer_path=analyzer,
            analyzer_timeout_seconds=60.0,
            embedding_dimensions=32,
            retrieval_limit=20,
            max_context_tokens=4_000,
        )

        _progress("cold index")
        cold, cold_seconds, cold_rss = _measure(lambda: index_project(config))
        _progress("warm no-op index")
        warm, warm_seconds, warm_rss = _measure(lambda: index_project(config))
        if warm.indexing.indexed_translation_units != 0 or warm.embedded_symbols != 0:
            raise RuntimeError("warm benchmark was not a no-op")

        generated.shared_header.write_text(
            generated.shared_header.read_text(encoding="utf-8")
            + "\n// deterministic header-reindex mutation\n",
            encoding="utf-8",
        )
        _progress("shared-header reindex")
        header, header_seconds, header_rss = _measure(lambda: index_project(config))
        expected_units = spec.translation_units * spec.builds
        if header.indexing.indexed_translation_units != expected_units:
            raise RuntimeError(
                "header mutation did not reindex every translation unit: "
                f"expected {expected_units}, got {header.indexing.indexed_translation_units}"
            )

        _progress("bounded query distributions")
        query_metrics, query_checks = _benchmark_queries(config, query_iterations)
        database_bytes = _database_size(database)
        fact_counts = _fact_counts(database)
        peak_rss = max(cold_rss, warm_rss, header_rss)
        environment = _environment(analyzer, compiler, commit)

    measurements = {
        "cold_index_seconds": cold_seconds,
        "warm_noop_seconds": warm_seconds,
        "header_reindex_seconds": header_seconds,
        "peak_rss_bytes": peak_rss,
        "database_bytes": database_bytes,
        "queries": query_metrics,
    }
    _progress("benchmark complete")
    budget_values = asdict(budgets)
    budget_results = evaluate_budgets(measurements, budgets)
    return {
        "schema": "cpp-context-benchmark-report",
        "schema_version": REPORT_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "measured_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "profile": profile,
        "workload": {**asdict(spec), "source_hash": generated.source_hash},
        "environment": environment,
        "measurements": measurements,
        "fact_counts": fact_counts,
        "budgets": budget_values,
        "budget_results": budget_results,
        "all_budgets_passed": all(budget_results.values()),
        "bounded_union_checks": query_checks,
        "indexing": {
            "cold": asdict(cold.indexing),
            "warm": asdict(warm.indexing),
            "header_reindex": asdict(header.indexing),
            "analysis_backend": cold.analysis_backend,
            "advanced_facts_complete": cold.advanced_facts_complete,
            "analyzer_capabilities": list(cold.analyzer_capabilities),
        },
    }


def _measure(operation: Callable[[], Any]) -> tuple[Any, float, int]:
    sampler = _ProcessTreeRssSampler()
    with sampler:
        started = time.perf_counter()
        result = operation()
        seconds = time.perf_counter() - started
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # getrusage is retained as a fallback for a process that exits between
    # samples; the sampler captures the aggregate of concurrently live workers.
    peak_rss_bytes = max(sampler.peak_bytes, self_rss * 1024, child_rss * 1024)
    return result, seconds, peak_rss_bytes


class _ProcessTreeRssSampler:
    """Sample aggregate Linux RSS for this process and all live descendants."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _ProcessTreeRssSampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_error: object) -> None:
        self._sample()
        self._stopped.set()
        assert self._thread is not None
        self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._stopped.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, _process_tree_rss_bytes(os.getpid()))


def _process_tree_rss_bytes(root_pid: int) -> int:
    processes: dict[int, tuple[int, int]] = {}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            fields = {
                key: value.strip()
                for key, _, value in (
                    line.partition(":")
                    for line in (entry / "status").read_text(encoding="utf-8").splitlines()
                )
                if key in {"PPid", "VmRSS"}
            }
            pid = int(entry.name)
            parent = int(fields["PPid"])
            rss_bytes = int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            continue
        processes[pid] = (parent, rss_bytes)

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in processes.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def _progress(message: str) -> None:
    print(f"benchmark: {message}", file=sys.stderr, flush=True)


def benchmark_temp_root() -> Path:
    """Use native Linux temporary storage even when WSL inherits Windows TMP/TEMP."""

    native = Path("/tmp")
    if native.is_dir() and os.access(native, os.W_OK | os.X_OK):
        return native
    fallback = Path(tempfile.gettempdir()).resolve(strict=True)
    if not fallback.is_dir() or not os.access(fallback, os.W_OK | os.X_OK):
        raise ValueError("no writable benchmark temporary directory is available")
    return fallback


def _benchmark_queries(
    config: AppConfig, iterations: int
) -> tuple[dict[str, dict[str, float | int]], dict[str, bool]]:
    single = (config.build_scope.variants[0],)
    union = config.build_scope.variants
    function_name = "bench::unit_0000::function_0000"
    with build_runtime(config) as runtime:
        function_hits = runtime.store.search_symbols(
            SearchQuery(function_name, limit=10),
            config.project_root,
            build_scope=config.build_scope,
        )
        function = next(
            (item.symbol for item in function_hits if item.symbol.qualified_name == function_name),
            None,
        )
        target_hits = runtime.store.search_symbols(
            SearchQuery("bench::high_fanout_target", limit=10),
            config.project_root,
            build_scope=config.build_scope,
        )
        target = next(
            (
                item.symbol
                for item in target_hits
                if item.symbol.qualified_name == "bench::high_fanout_target"
            ),
            None,
        )
        if function is None or target is None:
            raise RuntimeError("generated benchmark symbols were not indexed")

        checks: dict[str, bool] = {}

        def single_search() -> int:
            result = runtime.query_context(
                QueryRequest(
                    function_name,
                    max_context_tokens=2_000,
                    builds=single,
                    max_results=8,
                )
            ).context
            if len(result.items) > 8 or result.estimated_tokens > 2_000:
                raise RuntimeError("single-build retrieval exceeded a public limit")
            return len(result.items)

        def union_search() -> int:
            result = runtime.query_context(
                QueryRequest(
                    function_name,
                    max_context_tokens=2_000,
                    builds=union,
                    max_results=8,
                )
            ).context
            bounded = (
                len(result.items) <= 8
                and result.estimated_tokens <= 2_000
                and result.build_variants == union
            )
            checks["retrieval"] = checks.get("retrieval", True) and bounded
            if not bounded:
                raise RuntimeError("union retrieval exceeded a public limit or leaked scope")
            return len(result.items)

        def high_fanout_calls() -> int:
            result = runtime.analysis_service.calls(
                CallRequest(
                    symbol_id=target.id,
                    direction=GraphDirection.INCOMING,
                    builds=list(union),
                    max_results=25,
                )
            )
            expected_kind = "union" if len(union) > 1 else "single"
            bounded = len(result.calls) <= 25 and result.scope.kind == expected_kind
            checks["high_fanout_calls"] = checks.get("high_fanout_calls", True) and bounded
            if not bounded:
                raise RuntimeError("union call query exceeded result or scope limits")
            return len(result.calls)

        def cfg() -> int:
            result = runtime.analysis_service.control_flow(
                CfgRequest(
                    function_symbol_id=function.id,
                    builds=list(union),
                    max_graphs=min(5, len(union)),
                    max_blocks=12,
                    max_elements=40,
                    max_edges=24,
                )
            )
            bounded = (
                len(result.graphs) <= min(5, len(union))
                and sum(len(item.blocks) for item in result.graphs) <= 12
                and sum(len(item.elements) for item in result.graphs) <= 40
                and sum(len(item.edges) for item in result.graphs) <= 24
                and result.scope.kind == ("union" if len(union) > 1 else "single")
            )
            checks["control_flow"] = checks.get("control_flow", True) and bounded
            if not bounded:
                raise RuntimeError("union CFG query exceeded a public limit")
            return sum(len(item.blocks) for item in result.graphs)

        def flow() -> int:
            result = runtime.analysis_service.data_flow(
                FlowRequest(
                    function_symbol_id=function.id,
                    builds=list(union),
                    max_analyses=min(5, len(union)),
                    max_locations=20,
                    max_accesses=40,
                    max_evidence=40,
                )
            )
            bounded = (
                len(result.analyses) <= min(5, len(union))
                and sum(len(item.locations) for item in result.analyses) <= 20
                and sum(len(item.accesses) for item in result.analyses) <= 40
                and sum(
                    len(item.evidence)
                    + len(item.effects)
                    + len(item.return_origins)
                    + len(item.interprocedural)
                    for item in result.analyses
                )
                <= 40
                and result.scope.kind == ("union" if len(union) > 1 else "single")
            )
            checks["data_flow"] = checks.get("data_flow", True) and bounded
            if not bounded:
                raise RuntimeError("union data-flow query exceeded a public limit")
            return sum(len(item.evidence) for item in result.analyses)

        operations = {
            "single_build_search": single_search,
            "union_search": union_search,
            "high_fanout_calls_union": high_fanout_calls,
            "control_flow_union": cfg,
            "data_flow_union": flow,
        }
        metrics = {
            name: _query_distribution(operation, iterations)
            for name, operation in operations.items()
        }
    return metrics, checks


def _query_distribution(operation: Callable[[], int], iterations: int) -> dict[str, float | int]:
    durations: list[float] = []
    maximum_items = 0
    for _ in range(iterations):
        started = time.perf_counter()
        maximum_items = max(maximum_items, operation())
        durations.append(time.perf_counter() - started)
    return {
        "iterations": iterations,
        "p50_seconds": percentile(durations, 0.50),
        "p95_seconds": percentile(durations, 0.95),
        "max_seconds": max(durations),
        "max_items": maximum_items,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return the nearest-rank percentile used by the benchmark schema."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def evaluate_budgets(
    measurements: Mapping[str, Any], budgets: BenchmarkBudgets = DEFAULT_BUDGETS
) -> dict[str, bool]:
    """Compare a report measurement object with the Issue #11 budgets."""

    queries = measurements.get("queries")
    if not isinstance(queries, Mapping) or not queries:
        raise ValueError("measurements must contain query distributions")
    query_p95 = max(float(item["p95_seconds"]) for item in queries.values())
    return {
        "cold_index_seconds": float(measurements["cold_index_seconds"])
        <= budgets.cold_index_seconds,
        "peak_rss_bytes": int(measurements["peak_rss_bytes"]) <= budgets.peak_rss_bytes,
        "warm_noop_seconds": float(measurements["warm_noop_seconds"]) <= budgets.warm_noop_seconds,
        "query_p95_seconds": query_p95 <= budgets.query_p95_seconds,
        "database_bytes": int(measurements["database_bytes"]) <= budgets.database_bytes,
    }


def validate_report(report: Mapping[str, Any]) -> None:
    """Validate stable required fields without adding a runtime schema dependency."""

    if report.get("schema") != "cpp-context-benchmark-report":
        raise ValueError("benchmark report has an unknown schema")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("benchmark report has an unsupported schema version")
    required = (
        "generator_version",
        "measured_at_utc",
        "profile",
        "workload",
        "environment",
        "measurements",
        "fact_counts",
        "budgets",
        "budget_results",
        "all_budgets_passed",
        "bounded_union_checks",
        "indexing",
    )
    missing = [name for name in required if name not in report]
    if missing:
        raise ValueError("benchmark report is missing fields: " + ", ".join(missing))
    if not all(report["bounded_union_checks"].values()):
        raise ValueError("benchmark report contains a failed union-bound check")
    _reject_absolute_paths(report)


def _reject_absolute_paths(value: Any) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_absolute_paths(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            _reject_absolute_paths(nested)
    elif isinstance(value, str) and (value.startswith("/") or ":\\" in value):
        raise ValueError("benchmark report must not contain absolute paths")


def _database_size(database: Path) -> int:
    return sum(
        path.stat().st_size
        for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm"))
        if path.is_file()
    )


def _write_report_atomic(output: Path, document: str) -> None:
    """Replace a report only after its complete contents reach a sibling file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fact_counts(database: Path) -> dict[str, int]:
    with sqlite3.connect(database) as connection:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in _FACT_TABLES
        }


def _environment(analyzer: Path, compiler: str, commit: str | None) -> dict[str, Any]:
    analyzer_info = NativeAnalyzerClient(analyzer).probe()
    return {
        "commit": commit or _git_commit(),
        "hardware": {
            "architecture": platform.machine() or "unknown",
            "cpu": _cpu_model(),
            "logical_cpu_count": os.cpu_count() or 0,
            "memory_bytes": _physical_memory(),
        },
        "os": platform.platform(aliased=True),
        "python": platform.python_version(),
        "llvm": _tool_version(("llvm-config-18", "--version")),
        "clang": _tool_version((compiler, "--version")),
        "analyzer": {
            "version": analyzer_info.analyzer_version,
            "protocol": analyzer_info.protocol,
            "protocol_version": analyzer_info.protocol_version,
            "clang_major": analyzer_info.clang_major,
            "capabilities": sorted(analyzer_info.capabilities),
        },
        "peak_rss_method": (
            "100 ms /proc process-tree RSS sampling with getrusage fallback; "
            "Linux KiB converted to bytes"
        ),
    }


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    return _tool_version(("git", "-C", str(repository), "rev-parse", "HEAD"))


def _tool_version(arguments: Sequence[str], *, allow_failure: bool = False) -> str:
    try:
        completed = subprocess.run(
            arguments,
            check=not allow_failure,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else "unavailable"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.casefold().startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _physical_memory() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpp-context-benchmark",
        description="Run the deterministic Clang-18 C++ context benchmark.",
    )
    parser.add_argument("--profile", choices=("smoke", "reference"), default="smoke")
    parser.add_argument("--clang-analyzer", required=True, type=Path)
    parser.add_argument("--compiler", default="clang++-18")
    parser.add_argument("--query-iterations", type=int, default=50)
    parser.add_argument("--translation-units", type=int)
    parser.add_argument("--builds", type=int)
    parser.add_argument("--functions-per-tu", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, help="write report JSON (stdout when omitted)")
    parser.add_argument(
        "--enforce-budgets",
        action="store_true",
        help="return non-zero when any acceptance budget fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base = SMOKE_WORKLOAD if args.profile == "smoke" else REFERENCE_WORKLOAD
    customized = any(
        value is not None
        for value in (
            args.translation_units,
            args.builds,
            args.functions_per_tu,
            args.seed,
        )
    )
    try:
        spec = replace(
            base,
            translation_units=(
                base.translation_units if args.translation_units is None else args.translation_units
            ),
            builds=base.builds if args.builds is None else args.builds,
            functions_per_translation_unit=(
                base.functions_per_translation_unit
                if args.functions_per_tu is None
                else args.functions_per_tu
            ),
            seed=base.seed if args.seed is None else args.seed,
        )
        report = run_benchmark(
            spec=spec,
            analyzer=args.clang_analyzer,
            compiler=args.compiler,
            query_iterations=args.query_iterations,
            profile="custom" if customized else args.profile,
        )
        validate_report(report)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    document = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(document, end="")
    else:
        try:
            _write_report_atomic(args.output, document)
        except OSError:
            # Filesystem details can contain project/user paths; keep the CLI error public-safe.
            print("error: benchmark report could not be written", file=sys.stderr)
            return 2
        print(f"benchmark report written: {args.output}", file=sys.stderr)
    return 1 if args.enforce_budgets and not report["all_budgets_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
