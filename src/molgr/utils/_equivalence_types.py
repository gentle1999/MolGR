from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EquivalenceMethod(str, Enum):
    IDEAL = "ideal"
    INCHI_KEY = "inchi_key"
    CARBENE_ZWITTERION = "carbene_zwitterion"
    RESONANCE = "resonance"
    OPEN_SHELL_THREE_CENTER = "open_shell_three_center"


class EquivalenceDecision(str, Enum):
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    INCONCLUSIVE = "inconclusive"


class EquivalenceRelation(str, Enum):
    NORMALIZED_GRAPH_IDENTITY = "normalized_graph_identity"
    IDENTIFIER_EQUIVALENCE = "identifier_equivalence"
    CARBENE_ZWITTERION_EQUIVALENCE = "carbene_zwitterion_equivalence"
    RESONANCE_EQUIVALENCE = "resonance_equivalence"
    NONE = "none"


class InvariantStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


@dataclass
class InvariantResult:
    status: InvariantStatus
    mol1: object | None = None
    mol2: object | None = None
    reason: str = ""


@dataclass
class BoundedSearchMetadata:
    attempted: bool = False
    limit: int | None = None
    mol1_count: int = 0
    mol2_count: int = 0
    limit_reached: bool = False
    exhaustive: bool | None = None
    stop_reason: str = "not_attempted"
    supplier_limit_reached: bool = False
    local_limit_reached: bool = False
    validation_failed: bool = False
    regional_region_count: int = 0
    regional_state_count: int = 0
    regional_pruned_moves: int = 0


@dataclass
class PropertyCheck:
    mol1: int | str
    mol2: int | str
    passed: bool


@dataclass
class CanonicalSmilesDetail:
    mol1: str
    mol2: str
    use_chirality: bool


@dataclass
class CarbeneZwitterionDetail:
    mol1_normalized: str
    mol2_normalized: str


@dataclass
class ResonanceDetail:
    max_resonance: int
    mol1_resonance_count: int
    mol2_resonance_count: int
    hit_smiles: Optional[str] = None


@dataclass
class OpenShellThreeCenterDetail:
    """Mapped ``A radical-B=C <-> A=B-C radical`` structural witness."""

    mol1_atoms: tuple[int, int, int]
    mol2_atoms: tuple[int, int, int]
    mol1_radical_electrons: tuple[int, int, int]
    mol2_radical_electrons: tuple[int, int, int]
    mol1_bond_orders: tuple[float, float]
    mol2_bond_orders: tuple[float, float]


@dataclass
class TopologyMappingSearchMetadata:
    attempted: bool = False
    limit: int = 10000
    mappings_examined: int = 0
    limit_reached: bool = False


UNKNOWN_ELECTRON_METADATA = "unknown"


@dataclass
class RawMoleculeDiagnostics:
    """Read-only diagnostics captured before equivalence normalization.

    The MolGR-specific fields deliberately use ``"unknown"`` when the
    source molecule does not carry the corresponding property.  A missing
    property is not evidence for a zero-valued electronic state, especially
    for ordinary reference RDKit molecules.
    """

    formula: str
    hydrogen_count: int
    explicit_hydrogen_count: int
    formal_charge: int
    rdkit_radical_electrons: int
    metal_formal_state: tuple[tuple[int, str, int], ...]
    molgr_metal_unpaired_electrons: object = UNKNOWN_ELECTRON_METADATA
    active_lone_pair_properties: object = UNKNOWN_ELECTRON_METADATA
    unresolved_two_electron_center_properties: object = UNKNOWN_ELECTRON_METADATA
    molgr_unpaired_electrons: object = UNKNOWN_ELECTRON_METADATA


@dataclass
class EquivalenceChecks:
    formal_charge: PropertyCheck
    radical_electrons: PropertyCheck
    num_atoms: PropertyCheck
    heavy_atom_formula: PropertyCheck
    explicit_h_formula: PropertyCheck


@dataclass
class EquivalenceResult:
    decision: EquivalenceDecision = EquivalenceDecision.INCONCLUSIVE
    relation: EquivalenceRelation = EquivalenceRelation.NONE
    equivalent: bool = False
    method: Optional[EquivalenceMethod] = None
    reason: str = ""
    invariants: dict[str, InvariantResult] = field(default_factory=dict)
    contradictions: list[str] = field(default_factory=list)
    bounded_search: BoundedSearchMetadata | None = None
    checks: Optional[EquivalenceChecks] = None
    canonical_smiles: Optional[CanonicalSmilesDetail] = None
    carbene_zwitterion: Optional[CarbeneZwitterionDetail] = None
    resonance: Optional[ResonanceDetail] = None
    open_shell_three_center: Optional[OpenShellThreeCenterDetail] = None
    open_shell_three_center_mapping_search: TopologyMappingSearchMetadata | None = None
    # ``mol1``/``mol2`` are the evaluator's positional inputs.  The candidate
    # and reference aliases make the same evidence convenient for reviewer
    # callers without changing the historical positional API.
    raw_mol1: Optional[RawMoleculeDiagnostics] = None
    raw_mol2: Optional[RawMoleculeDiagnostics] = None
    raw_diagnostics: dict[str, RawMoleculeDiagnostics] = field(default_factory=dict)
    normalization_electronic_effects: dict[str, dict[str, int]] = field(default_factory=dict)
    open_shell_heuristic_triggered: bool = False
    native_stats: dict[str, object] = field(default_factory=dict)
    atom_mapping: tuple[int, ...] = ()
    resonance_witness: tuple[dict[str, object], ...] = ()


# Compatibility name retained for callers that imported the former detail type.
EquivalenceInfo = EquivalenceResult
