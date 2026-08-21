"""Interfaces implemented by Clang, SCIP, or other ingestion adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cpp_context_engine.models import (
    BuildConfiguration,
    BuildVariant,
    CallArgumentBinding,
    CallResultBinding,
    CallSite,
    CallTarget,
    CfgBlock,
    CfgEdge,
    CfgElement,
    CfgGraph,
    CodeSymbol,
    DataAccess,
    DataFlowAnalysis,
    DataFlowEvidence,
    FunctionSummary,
    GraphEdge,
    InterproceduralFlow,
    MemoryLocation,
    SummaryEffect,
    SummaryReturnOrigin,
    SymbolOccurrence,
    TranslationUnit,
)


@dataclass(frozen=True, slots=True)
class IngestionBatch:
    build_configurations: tuple[BuildConfiguration, ...]
    translation_units: tuple[TranslationUnit, ...]
    symbols: tuple[CodeSymbol, ...]
    occurrences: tuple[SymbolOccurrence, ...]
    edges: tuple[GraphEdge, ...]
    build_variants: tuple[BuildVariant, ...] = ()
    cfg_graphs: tuple[CfgGraph, ...] = ()
    cfg_blocks: tuple[CfgBlock, ...] = ()
    cfg_elements: tuple[CfgElement, ...] = ()
    cfg_edges: tuple[CfgEdge, ...] = ()
    callsites: tuple[CallSite, ...] = ()
    call_targets: tuple[CallTarget, ...] = ()
    data_flow_analyses: tuple[DataFlowAnalysis, ...] = ()
    memory_locations: tuple[MemoryLocation, ...] = ()
    data_accesses: tuple[DataAccess, ...] = ()
    data_flow_evidence: tuple[DataFlowEvidence, ...] = ()
    function_summaries: tuple[FunctionSummary, ...] = ()
    summary_effects: tuple[SummaryEffect, ...] = ()
    summary_return_origins: tuple[SummaryReturnOrigin, ...] = ()
    call_argument_bindings: tuple[CallArgumentBinding, ...] = ()
    call_result_bindings: tuple[CallResultBinding, ...] = ()
    interprocedural_flows: tuple[InterproceduralFlow, ...] = ()


class Ingestor(Protocol):
    def ingest(self, project_root: Path, compilation_database: Path) -> IngestionBatch:
        """Normalize a configured C++ project into symbols and relationships."""
        ...
