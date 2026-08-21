"""Bounded compiler-evidence analyses composed above native facts."""

from cpp_context_engine.analysis.interprocedural import (
    InterproceduralLimits,
    InterproceduralSolution,
    solve_interprocedural,
)

__all__ = ["InterproceduralLimits", "InterproceduralSolution", "solve_interprocedural"]
