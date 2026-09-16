from __future__ import annotations

import importlib.util
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).parents[1] / "tools" / "benchmark_summary_refresh.py"
    spec = importlib.util.spec_from_file_location("summary_benchmark_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_driver_times_committed_transaction_and_rolls_back_timeout(driver) -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

        def solve(project_id, variant, functions):
            assert (project_id, variant, functions) == (1, "default", {"function"})
            connection.execute("INSERT INTO evidence VALUES ('complete')")
            return 1

        store = SimpleNamespace(_connection=connection, _refresh_summary_solutions=solve)
        result = driver.refresh(store, 1, {"function"}, 1.0)
        assert result["summaries_refreshed"] == 1
        assert 0 <= result["seconds"] < 1.0
        assert not connection.in_transaction

        def slow_solve(*_args):
            connection.execute("INSERT INTO evidence VALUES ('must roll back')")
            time.sleep(0.2)

        store._refresh_summary_solutions = slow_solve
        with pytest.raises(TimeoutError, match="exceeded"):
            driver.refresh(store, 1, {"function"}, 0.01)
        assert connection.execute("SELECT * FROM evidence").fetchall() == [("complete",)]
        assert not connection.in_transaction


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_refresh_driver_rejects_disabled_limits(driver, budget) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        driver.refresh(None, 1, {"function"}, budget)


@pytest.mark.parametrize("baseline,budget", [(None, 91), (Path("baseline.json"), 31)])
def test_refresh_driver_keeps_baseline_and_candidate_limits_distinct(driver, baseline, budget):
    with pytest.raises(ValueError, match="at most"):
        driver.run(Path("unused-input"), Path("unused-output"), baseline, budget)
