"""Dictionary layouts returned by the native pipeline and equivalence bindings."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict

from molgr import _core


class NativeDiagnostics(TypedDict, total=False):
    """Empty on success. A native failure supplies its stage, code and counters."""

    code: str
    stage: str
    backend: str
    message: str
    counts: Dict[str, int]
    details: Dict[str, str]
    cause_type: str
    cause_message: str


class NativeBatchResult(TypedDict):
    """One native result keyed by the zero-based index in the input batch."""

    index: int
    molecule_data: Optional[_core.utils.MoleculeData]
    diagnostics: NativeDiagnostics


class NativeGraphSummary(TypedDict):
    """Graph invariants and completeness of the input's three atom-local electron fields."""

    formula: str
    heavy_formula: str
    atom_count: int
    hydrogen_count: int
    charge: int
    radicals: int
    organic_atoms: int
    organic_charge: int
    organic_radicals: int
    metal_states: List[Tuple[int, int]]
    known_fields: int
    deferred_centers: int


class _OptionalMoveFields(TypedDict, total=False):
    cycle: List[int]


class NativeResonanceMove(_OptionalMoveFields):
    """One resonance witness step with atom indices and the search side that generated it."""

    kind: str
    atoms: Tuple[int, int, int]
    side: int


class NativeEquivalenceResult(TypedDict):
    """Decision and evidence from bounded native atom mapping and resonance search."""

    decision: str
    method: str
    reason: str
    invariants: Dict[str, int]
    left: NativeGraphSummary
    right: NativeGraphSummary
    mapping: List[int]
    witness: List[NativeResonanceMove]
    mappings: int
    mapping_nodes: int
    mapping_limit: bool
    states_left: int
    states_right: int
    regions: int
    pruned_moves: int
    resonance_cache_hits: int
    region_cache_hits: int
    region_constraint_rejections: int
    comparable_fields: int
    resonance_limit: bool
    resonance_attempted: bool
    validation_failed: bool
    stop_reason: str
    preparation_ms: float
    mapping_ms: float
    resonance_ms: float
