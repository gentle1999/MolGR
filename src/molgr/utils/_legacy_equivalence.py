# pyright: reportCallIssue=false
"""Conservative molecular-representation equivalence evaluator.

Evaluator v2 compares the normalized organic graph, metal formal-state signature,
and jointly known MolGR electronic annotations. Missing annotations do not veto
independent structural evidence. Known annotations are compared under atom
symmetry and the existing representation transformations; incomplete searches
are inconclusive, not negative evidence. Metal coordination edges are ignored
by default; callers can request a standardized coordination-connectivity check.
This is not a proof of an identical quantum electronic state.

``INCONCLUSIVE`` is a first-class result, not a synonym for ``NOT_EQUIVALENT``.
The compatibility wrapper returns ``False`` for both, so callers that need the
distinction must inspect :class:`EquivalenceResult`.
"""

from __future__ import annotations

from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Iterator, Optional, Tuple

from rdkit import Chem, rdBase
from rdkit.Chem import ResonanceMolSupplier, inchi


pt = Chem.GetPeriodicTable()


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


@dataclass(frozen=True)
class _ResonanceForms:
    forms: tuple[str, ...] = ()
    supplier_limit_reached: bool = False
    local_limit_reached: bool = False
    target_found: bool = False
    validation_failed: bool = False
    # Validated local edges from a source/supplier seed to the common witness.
    # Supplier enumeration and special normalization are not local graph edges.
    witness_path: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.forms)

    def __len__(self) -> int:
        return len(self.forms)

    @property
    def limit_reached(self) -> bool:
        return self.supplier_limit_reached or self.local_limit_reached


@dataclass(frozen=True)
class _ResonanceMatchResult:
    matched: bool
    mol1_count: int
    mol2_count: int
    hit_smiles: str | None
    supplier_limit_reached: bool = False
    local_limit_reached: bool = False
    validation_failed: bool = False
    local_witness_paths: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    regional_witness: _RegionalChargeShiftWitness | None = None

    def __iter__(self):
        # Preserve tuple unpacking for callers of this private helper.
        return iter((self.matched, self.mol1_count, self.mol2_count, self.hit_smiles))

    @property
    def limit_reached(self) -> bool:
        return self.supplier_limit_reached or self.local_limit_reached

    @property
    def exhaustive(self) -> bool:
        return not self.matched and not self.limit_reached and not self.validation_failed

    @property
    def stop_reason(self) -> str:
        if self.matched:
            return "target_found"
        if self.validation_failed:
            return "validation_failed"
        return "limit_reached" if self.limit_reached else "exhausted"


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
    resonance_flags: int
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


# Compatibility name retained for callers that imported the former detail type.
EquivalenceInfo = EquivalenceResult


_NON_METAL_ATOMIC_NUMBERS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 51, 52, 53})

_SULFIMIDE_RESONANCE_PATTERN = Chem.MolFromSmarts("[N:1]=[S:2](=[O:3])([O-:4])[!#1:5]")
_CARBENE_ZWITTERION_PATTERN = Chem.MolFromSmarts("[*-]=[*+]")
_THIOSEMICARBAZONE_RESONANCE_PATTERNS = tuple(
    pattern
    for pattern in (
        Chem.MolFromSmarts("[N:1]-[N:2]=[C:3]([S:4])[N:5]"),
        Chem.MolFromSmarts("[N:1]=[N:2]-[C:3](=[S:4])[N:5]"),
    )
    if pattern is not None
)


def _is_metal_atom(atom: Chem.Atom) -> bool:
    return int(atom.GetAtomicNum()) not in _NON_METAL_ATOMIC_NUMBERS


def _total_formal_charge(mol: Chem.Mol) -> int:
    return sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())


def _total_radical_electrons(mol: Chem.Mol) -> int:
    return sum(int(atom.GetNumRadicalElectrons()) for atom in mol.GetAtoms())


def _formula_key(mol: Chem.Mol, *, include_hydrogen: bool) -> str:
    counts = Counter(
        atom.GetSymbol() for atom in mol.GetAtoms() if include_hydrogen or atom.GetAtomicNum() != 1
    )
    if not counts:
        return ""
    ordered_symbols: list[str] = []
    if "C" in counts:
        ordered_symbols.append("C")
    if include_hydrogen and "H" in counts:
        ordered_symbols.append("H")
    ordered_symbols.extend(symbol for symbol in sorted(counts) if symbol not in ordered_symbols)
    return "".join(
        f"{symbol}{counts[symbol] if counts[symbol] != 1 else ''}" for symbol in ordered_symbols
    )


def _raw_hydrogen_count(mol: Chem.Mol) -> tuple[int, int]:
    """Return explicit and total hydrogen counts without normalizing ``mol``."""

    explicit = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
    total = explicit
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        with suppress(Exception):
            # Do not count explicit hydrogen neighbors here: they are already
            # included in ``explicit``.  ``includeNeighbors=True`` would count
            # those neighbors a second time for molecules passed through
            # ``Chem.AddHs``.
            total += int(atom.GetTotalNumHs(includeNeighbors=False))
    return explicit, total


def _raw_formula_key(mol: Chem.Mol, hydrogen_count: int) -> str:
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
    if hydrogen_count:
        counts["H"] = hydrogen_count
    ordered_symbols: list[str] = []
    if "C" in counts:
        ordered_symbols.append("C")
    if "H" in counts:
        ordered_symbols.append("H")
    ordered_symbols.extend(symbol for symbol in sorted(counts) if symbol not in ordered_symbols)
    return "".join(
        f"{symbol}{counts[symbol] if counts[symbol] != 1 else ''}" for symbol in ordered_symbols
    )


def _read_electron_property(atom: Chem.Atom, name: str, *, boolean: bool = False):
    """Read present annotations without turning absent/malformed data into zero."""

    if not atom.HasProp(name):
        return None
    try:
        value = atom.GetProp(name).strip().lower()
        if boolean:
            return {"true": True, "1": True, "false": False, "0": False}.get(value)
        count = int(value)
        return count if count >= 0 else None
    except (RuntimeError, TypeError, ValueError):
        return None


def _raw_atom_property_entries(
    mol: Chem.Mol,
    property_name: str,
    *,
    metals_only: bool = False,
    boolean: bool = False,
) -> object:
    """Collect only present atom properties, preserving missing metadata."""

    entries: list[tuple[int, object]] = []
    for atom in mol.GetAtoms():
        if metals_only and not _is_metal_atom(atom):
            continue
        if not atom.HasProp(property_name):
            continue
        value = _read_electron_property(atom, property_name, boolean=boolean)
        if value is None:
            value = UNKNOWN_ELECTRON_METADATA
        entries.append((int(atom.GetIdx()), value))
    return tuple(entries) if entries else UNKNOWN_ELECTRON_METADATA


def _raw_molecule_diagnostics(mol: Chem.Mol) -> RawMoleculeDiagnostics:
    """Capture source-state evidence before metal or octet normalization."""

    explicit_h, total_h = _raw_hydrogen_count(mol)
    metal_formal_state = tuple(
        sorted(
            (
                int(atom.GetIdx()),
                atom.GetSymbol(),
                int(atom.GetFormalCharge()),
            )
            for atom in mol.GetAtoms()
            if _is_metal_atom(atom)
        )
    )
    return RawMoleculeDiagnostics(
        formula=_raw_formula_key(mol, total_h),
        hydrogen_count=total_h,
        explicit_hydrogen_count=explicit_h,
        formal_charge=_total_formal_charge(mol),
        rdkit_radical_electrons=_total_radical_electrons(mol),
        metal_formal_state=metal_formal_state,
        molgr_metal_unpaired_electrons=_raw_atom_property_entries(
            mol,
            "MOLGR_METAL_UNPAIRED_ELECTRONS",
            metals_only=True,
        ),
        active_lone_pair_properties=_raw_atom_property_entries(
            mol,
            "MOLGR_LONE_PAIR_COUNT",
        ),
        unresolved_two_electron_center_properties=_raw_atom_property_entries(
            mol,
            "MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER",
            boolean=True,
        ),
        molgr_unpaired_electrons=_raw_atom_property_entries(
            mol,
            "MOLGR_UNPAIRED_ELECTRON_COUNT",
        ),
    )


def _safe_copy(mol: Chem.Mol) -> Chem.Mol:
    clone = Chem.Mol(mol)
    clone.UpdatePropertyCache(strict=False)
    _sanitize_if_possible(clone)
    return clone


def _sanitize_if_possible(
    mol: Chem.Mol, *, annotated_indices: tuple[int, ...] | None = None
) -> bool:
    # MolGR's explicit assignment is authoritative, but the current working
    # count may already include an allowed graph/electron transformation.
    # Preserve that count, not the original property value: restoring the latter
    # here would undo resonance moves or metal-bond bookkeeping. Unannotated
    # baseline atoms keep RDKit's existing perception behavior.
    if annotated_indices is None:
        annotated_indices = _unpaired_annotation_indices(mol)
    assigned = [
        (atom, atom.GetNumRadicalElectrons())
        for atom in (mol.GetAtomWithIdx(idx) for idx in annotated_indices)
    ]
    try:
        Chem.SanitizeMol(mol)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return False
    finally:
        for atom, count in assigned:
            atom.SetNumRadicalElectrons(count)
    return True


def _unpaired_annotation_indices(mol: Chem.Mol) -> tuple[int, ...]:
    return tuple(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if not _is_metal_atom(atom)
        and _read_electron_property(atom, "MOLGR_UNPAIRED_ELECTRON_COUNT") is not None
    )


def _kekulize_if_possible(mol: Chem.Mol, *, clear_aromatic_flags: bool = False) -> None:
    try:
        Chem.Kekulize(mol, clearAromaticFlags=clear_aromatic_flags)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        pass


def _clear_metal_radicals(mol: Chem.Mol) -> Chem.Mol:
    rw_mol = Chem.RWMol(mol)
    for atom in rw_mol.GetAtoms():
        if _is_metal_atom(atom):
            atom.SetNumRadicalElectrons(0)
    normalized = rw_mol.GetMol()
    normalized.UpdatePropertyCache(strict=False)
    return normalized


def _safe_smiles(mol: Chem.Mol, *, use_chirality: bool) -> str:
    clone = _safe_copy(mol)
    if not use_chirality:
        Chem.RemoveStereochemistry(clone)
        clone.UpdatePropertyCache(strict=False)
    # Isotopes remain chemical labels when stereochemistry is disabled.
    return Chem.MolToSmiles(clone, canonical=True, isomericSmiles=True)


def _canon_smiles(mol: Chem.Mol, use_chirality: bool) -> str:
    # MolToSmiles(canonical=True) already canonicalizes the molecular graph.
    return _safe_smiles(mol, use_chirality=use_chirality)


def _resonance_form_smiles(
    mol: Chem.Mol, use_chirality: bool, *, stereo_prepared: bool = False
) -> str:
    try:
        # ResonanceMolSupplier returns sanitized molecules. Avoid copying and
        # sanitizing every form in the hot loop, but retain the safe fallback.
        if not use_chirality and not stereo_prepared:
            # Remove stereo explicitly; isomeric output must stay enabled to
            # retain isotopes in the resonance cache and comparison keys.
            achiral = Chem.Mol(mol)
            Chem.RemoveStereochemistry(achiral)
            for bond in achiral.GetBonds():
                bond.SetBondDir(Chem.BondDir.NONE)
                bond.SetStereo(Chem.BondStereo.STEREONONE)
            achiral.UpdatePropertyCache(strict=False)
            mol = achiral
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return _canon_smiles(mol, use_chirality)


def _inchi_key(mol: Chem.Mol, *, use_chirality: bool = True) -> str | None:
    try:
        clone = _safe_copy(mol)
        if not use_chirality:
            Chem.RemoveStereochemistry(clone)
            clone.RemoveAllConformers()
            clone.UpdatePropertyCache(strict=False)
        key = inchi.MolToInchiKey(clone)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return None
    if not key:
        return None
    return key


def _remove_hs_without_sanitize(mol: Chem.Mol) -> Chem.Mol:
    try:
        stripped = Chem.RemoveHs(mol, sanitize=False)
    except TypeError:
        stripped = Chem.RemoveHs(mol)
    stripped.UpdatePropertyCache(strict=False)
    return stripped


def _add_hs_without_sanitize(mol: Chem.Mol) -> Chem.Mol:
    added = Chem.AddHs(mol)
    added.UpdatePropertyCache(strict=False)
    return added


def _standardize_metal_bonds(mol: Chem.Mol) -> Chem.Mol:
    source = Chem.Mol(_clear_metal_radicals(_add_hs_without_sanitize(mol)))
    source.UpdatePropertyCache(strict=False)
    sanitized = _sanitize_if_possible(source)
    rw_mol = Chem.RWMol(source)
    changed = False
    for bond in list(rw_mol.GetBonds()):
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        begin_is_metal = _is_metal_atom(begin_atom)
        end_is_metal = _is_metal_atom(end_atom)
        if begin_is_metal == end_is_metal:
            continue
        if bond.GetBondType() == Chem.BondType.DATIVE:
            continue

        metal_atom = begin_atom if begin_is_metal else end_atom
        ligand_atom = end_atom if begin_is_metal else begin_atom
        bond_order = max(1, int(round(bond.GetBondTypeAsDouble())))

        if bond_order == 2:
            # Return the metal-ligand double-bond electron pair to the ligand.
            # Once the metal is removed, this is the carbene-like two-electron
            # center represented by an explicit pair of radical electrons. A
            # carbene coordination bond does not change the metal valence.
            ligand_atom.SetNumRadicalElectrons(ligand_atom.GetNumRadicalElectrons() + 2)
        else:
            metal_atom.SetFormalCharge(metal_atom.GetFormalCharge() + bond_order)
            ligand_atom.SetFormalCharge(ligand_atom.GetFormalCharge() - bond_order)

        bond.SetBondType(Chem.BondType.DATIVE)
        bond.SetBondDir(Chem.BondDir.NONE)
        bond.SetIsAromatic(False)
        changed = True

    standardized = rw_mol.GetMol()
    standardized.UpdatePropertyCache(strict=False)
    if changed or not sanitized:
        _sanitize_if_possible(standardized)
    return standardized


def _remove_coordination_bonds(mol: Chem.Mol) -> Chem.Mol:
    rw_mol = Chem.RWMol(Chem.Mol(mol))
    bonds_to_remove: list[tuple[int, int]] = []
    for bond in rw_mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if _is_metal_atom(begin_atom) == _is_metal_atom(end_atom):
            continue
        if bond.GetBondType() == Chem.BondType.DATIVE:
            bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    for begin_idx, end_idx in bonds_to_remove:
        rw_mol.RemoveBond(begin_idx, end_idx)
    stripped = rw_mol.GetMol()
    stripped.UpdatePropertyCache(strict=False)
    _sanitize_if_possible(stripped)
    return stripped


def _remove_metal_atoms(mol: Chem.Mol) -> Chem.Mol:
    rw_mol = Chem.RWMol(_safe_copy(mol))
    metal_indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if _is_metal_atom(atom)]
    for atom_idx in sorted(metal_indices, reverse=True):
        rw_mol.RemoveAtom(int(atom_idx))
    stripped = rw_mol.GetMol()
    stripped.UpdatePropertyCache(strict=False)
    _sanitize_if_possible(stripped)
    return stripped


def _prepare_organic_mol(mol: Chem.Mol, *, already_standardized: bool = False) -> Chem.Mol:
    return _prepare_organic_mol_with_status(mol, already_standardized=already_standardized)[0]


def _prepare_organic_mol_with_status(
    mol: Chem.Mol, *, already_standardized: bool = False
) -> tuple[Chem.Mol, bool]:
    standardized = mol if already_standardized else _standardize_metal_bonds(mol)
    # Removing a metal also removes all of its incident bonds. No intermediate
    # disconnected metal graph is needed by the organic comparison.
    graph = Chem.RWMol(standardized)
    metal_indices = [atom.GetIdx() for atom in graph.GetAtoms() if _is_metal_atom(atom)]
    for idx in reversed(metal_indices):
        graph.RemoveAtom(idx)
    organic = graph.GetMol()
    organic.UpdatePropertyCache(strict=False)
    sanitized = _sanitize_if_possible(organic)
    return organic, sanitized


# These elements are allowed to appear in tmQMg in hypervalent, multiple-bond
# forms.  Equivalence comparison uses their charge-separated octet form only;
# reconstruction output and stored reference graphs are left untouched.
_OCTET_NORMALIZED_ATOMIC_NUMS = frozenset({7, 8, 9, 15, 16, 17, 33, 34, 35, 53})
_OCTET_VALENCE_LIMITS = {
    9: 1,
    17: 1,
    35: 1,
    53: 1,
}


def _bond_type_for_order(order: int) -> Chem.BondType:
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
    }[order]


def _normalize_nonmetal_octet(mol: Chem.Mol, *, already_sanitized: bool = False) -> Chem.Mol:
    """Canonicalize supported hypervalent nonmetals to charge-separated octets."""

    rw_mol = Chem.RWMol(mol if already_sanitized else _safe_copy(mol))
    # A single deterministic pass is insufficient when reducing one bond
    # exposes another overvalent atom, so process until no excess remains.
    changed = True
    while changed:
        changed = False
        for atom in rw_mol.GetAtoms():
            atomic_num = int(atom.GetAtomicNum())
            if atomic_num not in _OCTET_NORMALIZED_ATOMIC_NUMS:
                continue
            limit = _OCTET_VALENCE_LIMITS.get(atomic_num, 4)
            bond_order_sum = sum(
                int(round(bond.GetBondTypeAsDouble()))
                for bond in atom.GetBonds()
                if bond.GetBondType() != Chem.BondType.DATIVE
            )
            if bond_order_sum <= limit:
                continue

            reducible = sorted(
                (
                    bond
                    for bond in atom.GetBonds()
                    if int(round(bond.GetBondTypeAsDouble())) > 1
                    and not bond.GetIsAromatic()
                    and bond.GetBondType() != Chem.BondType.DATIVE
                ),
                key=lambda bond: (
                    -int(round(bond.GetBondTypeAsDouble())),
                    int(bond.GetOtherAtomIdx(atom.GetIdx())),
                ),
            )
            if not reducible:
                continue
            bond = reducible[0]
            order = int(round(bond.GetBondTypeAsDouble()))
            other = rw_mol.GetAtomWithIdx(bond.GetOtherAtomIdx(atom.GetIdx()))
            bond.SetBondType(_bond_type_for_order(order - 1))
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
            other.SetFormalCharge(other.GetFormalCharge() - 1)
            changed = True
            break

    normalized = rw_mol.GetMol()
    normalized.UpdatePropertyCache(strict=False)
    return normalized


def _prepare_resonance_source(mol: Chem.Mol, *, use_chirality: bool = True) -> Chem.Mol:
    source = _add_hs_without_sanitize(mol)
    _kekulize_if_possible(source, clear_aromatic_flags=True)
    if not use_chirality:
        # Neither the local moves nor ResonanceMolSupplier introduce stereo.
        # Clear it once on the private source, rather than copying every form.
        Chem.RemoveStereochemistry(source)
        for bond in source.GetBonds():
            bond.SetBondDir(Chem.BondDir.NONE)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    return source


class _ChargeShiftContext:
    """Reusable local search work for one fixed atom/bond numbering.

    Proposal keys describe the graph BEFORE sanitization. They only reuse a
    previously validated result of the same proposal; an unvalidated source is
    never treated as an already accepted result.
    """

    def __init__(self, source: Chem.Mol, *, use_chirality: bool, cache_limit: int = 256):
        self.use_chirality = use_chirality
        self.cache_limit = max(1, cache_limit)
        self.transition_edges = 0
        self.schema = self._schema(source)
        self.annotated_indices = _unpaired_annotation_indices(source)
        adjacency = [[] for _ in source.GetAtoms()]
        for bond in source.GetBonds():
            a, b, idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetIdx()
            adjacency[a].append((idx, b))
            adjacency[b].append((idx, a))
        self.adjacency = tuple(tuple(neighbors) for neighbors in adjacency)
        self.states: dict[Chem.Mol, tuple] = {}
        self.proposals: dict[tuple, Chem.Mol] = {}
        self.keys: dict[Chem.Mol, str] = {}
        self.finished: dict[Chem.Mol, tuple[Chem.Mol, ...]] = {}
        self.transitions: dict[tuple, tuple[Chem.Mol, ...]] = {}
        self.validation_failed = False

    def remember(self, cache, key, value):
        if key not in cache and len(cache) >= self.cache_limit:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def complete(self, mol, state, neighbors):
        if neighbors is None:
            return
        neighbors = tuple(neighbors)
        if self.transition_edges + len(neighbors) > self.cache_limit:
            self.finished.clear()
            self.transitions.clear()
            self.transition_edges = 0
        self.transition_edges += len(neighbors)
        self.remember(self.finished, mol, neighbors)
        if state is not None:
            self.remember(self.transitions, state, neighbors)

    @staticmethod
    def _schema(mol: Chem.Mol) -> tuple:
        return (
            tuple((a.GetAtomicNum(), a.GetIsotope(), a.GetAtomMapNum()) for a in mol.GetAtoms()),
            tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()),
            _unpaired_annotation_indices(mol),
        )

    def compatible(self, mol: Chem.Mol) -> bool:
        return self.schema == self._schema(mol)

    def state(self, mol: Chem.Mol) -> tuple:
        state = self.states.get(mol)
        if state is None:
            state = (
                tuple(
                    (
                        a.GetFormalCharge(),
                        a.GetNumRadicalElectrons(),
                        a.GetIsAromatic(),
                        a.GetNumExplicitHs(),
                        a.GetNoImplicit(),
                        a.GetChiralTag() if self.use_chirality else None,
                    )
                    for a in mol.GetAtoms()
                ),
                tuple(
                    (
                        b.GetBondType(),
                        b.GetIsAromatic(),
                        b.GetBondDir() if self.use_chirality else None,
                        b.GetStereo() if self.use_chirality else None,
                        tuple(b.GetStereoAtoms()) if self.use_chirality else (),
                    )
                    for b in mol.GetBonds()
                ),
            )
            self.remember(self.states, mol, state)
        return state

    def key(self, mol: Chem.Mol) -> str:
        key = self.keys.get(mol)
        if key is None:
            key = _resonance_form_smiles(mol, self.use_chirality, stereo_prepared=True)
            self.remember(self.keys, mol, key)
        return key

    def move(self, mol, state, negative, center, terminal, single_idx, double_idx):
        atoms, bonds = state
        new_atoms, new_bonds = list(atoms), list(bonds)
        new_atoms[negative] = (0, *atoms[negative][1:])
        new_atoms[terminal] = (-1, *atoms[terminal][1:])
        new_bonds[single_idx] = (Chem.BondType.DOUBLE, *bonds[single_idx][1:])
        new_bonds[double_idx] = (Chem.BondType.SINGLE, *bonds[double_idx][1:])
        proposal = (tuple(new_atoms), tuple(new_bonds))
        known = self.proposals.get(proposal)
        if known is not None:
            return known
        shifted = Chem.RWMol(mol)
        shifted.GetBondWithIdx(single_idx).SetBondType(Chem.BondType.DOUBLE)
        shifted.GetBondWithIdx(double_idx).SetBondType(Chem.BondType.SINGLE)
        shifted.GetAtomWithIdx(negative).SetFormalCharge(0)
        shifted.GetAtomWithIdx(terminal).SetFormalCharge(-1)
        form = shifted.GetMol()
        try:
            form.UpdatePropertyCache(strict=False)
            valid = _sanitize_if_possible(form, annotated_indices=self.annotated_indices)
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            valid = False
        if not valid:
            self.validation_failed = True
            return None
        self.remember(self.proposals, proposal, form)
        # Sanitization may change aromaticity/radicals beyond the edited atoms.
        # Read the actual state lazily if this form is expanded; never label it
        # with the proposed pre-sanitization state.
        return form


def _iter_charge_shift_resonance_forms(
    mol: Chem.Mol,
    *,
    annotated_indices: tuple[int, ...] | None = None,
    context: _ChargeShiftContext | None = None,
) -> Iterator[Chem.Mol]:
    """Generate validated ``[*-]-[*]=[*]`` migrations from negative centers.

    Adjacency is fixed for the search. Electronic state proposals are deduplicated
    before allocating a molecule, sanitizing it, or generating a canonical key.
    """
    if context is None:
        context = _ChargeShiftContext(mol, use_chirality=True)
        if annotated_indices is not None:
            context.annotated_indices = annotated_indices
    known = context.finished.get(mol)
    if known is not None:
        yield from known
        return
    negatives = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetFormalCharge() == -1 and atom.GetNumRadicalElectrons() == 0
    ]
    if not negatives:
        context.complete(mol, None, ())
        return
    state = None
    neighbors = []
    for negative in negatives:
        negative_idx = negative.GetIdx()
        default_valence = pt.GetDefaultValence(negative.GetAtomicNum())
        if default_valence <= 0 or negative.GetTotalValence() + 1 > default_valence:
            continue
        for single_idx, center_idx in context.adjacency[negative_idx]:
            single = mol.GetBondWithIdx(single_idx)
            if single.GetBondType() != Chem.BondType.SINGLE or single.GetIsAromatic():
                continue
            center = mol.GetAtomWithIdx(center_idx)
            if center.GetFormalCharge() or center.GetNumRadicalElectrons():
                continue
            for double_idx, terminal_idx in context.adjacency[center_idx]:
                if terminal_idx == negative_idx:
                    continue
                double = mol.GetBondWithIdx(double_idx)
                if double.GetBondType() != Chem.BondType.DOUBLE or double.GetIsAromatic():
                    continue
                terminal = mol.GetAtomWithIdx(terminal_idx)
                if terminal.GetFormalCharge() or terminal.GetNumRadicalElectrons():
                    continue
                # Most supplier forms have no eligible local move. Snapshot the
                # complete state only when a proposal will actually be made.
                if state is None:
                    state = context.state(mol)
                    known = context.transitions.get(state)
                    if known is not None:
                        context.remember(context.finished, mol, known)
                        yield from known
                        return
                form = context.move(
                    mol,
                    state,
                    negative_idx,
                    center_idx,
                    terminal_idx,
                    single_idx,
                    double_idx,
                )
                if form is not None:
                    if neighbors is not None:
                        if len(neighbors) >= context.cache_limit:
                            neighbors = None
                        else:
                            neighbors.append(form)
                    yield form
    # A consumer may stop at a witness/budget. Only memoize an expansion after
    # its generator finishes; a consumed prefix is not the full neighbor set.
    context.complete(mol, state, neighbors)


def _charge_shift_resonance_forms(mol: Chem.Mol) -> tuple[Chem.Mol, ...]:
    return tuple(_iter_charge_shift_resonance_forms(mol))


def _local_charge_shift_search(
    mol: Chem.Mol,
    *,
    use_chirality: bool,
    max_resonance: int,
    targets: frozenset[str],
    context: _ChargeShiftContext | None = None,
) -> _ResonanceForms:
    if max_resonance <= 0:
        return _ResonanceForms(local_limit_reached=True)
    source = _prepare_resonance_source(mol, use_chirality=use_chirality)
    if context is None or not context.compatible(source):
        context = _ChargeShiftContext(
            source, use_chirality=use_chirality, cache_limit=2 * max_resonance
        )
    source_key = context.key(source)
    queue = [(source, source_key)]
    discovered = {source_key}
    if source_key in targets:
        return _ResonanceForms((source_key,), target_found=True)
    index = 0
    while index < len(queue):
        form, _ = queue[index]
        index += 1
        for shifted in _iter_charge_shift_resonance_forms(form, context=context):
            key = context.key(shifted)
            if key in discovered:
                continue
            if len(queue) >= max_resonance:
                return _ResonanceForms(
                    tuple(key for _, key in queue),
                    local_limit_reached=True,
                    validation_failed=context.validation_failed,
                )
            discovered.add(key)
            queue.append((shifted, key))
            if key in targets:
                return _ResonanceForms(
                    tuple(key for _, key in queue),
                    target_found=True,
                    validation_failed=context.validation_failed,
                )
    return _ResonanceForms(
        tuple(key for _, key in queue),
        validation_failed=context.validation_failed,
    )


def _bidirectional_charge_shift_search(
    seeds: tuple[list[tuple[Chem.Mol, str]], list[tuple[Chem.Mol, str]]],
    *,
    contexts: tuple[_ChargeShiftContext, _ChargeShiftContext],
    local_budget: int,
    supplier_limited: tuple[bool, bool] = (False, False),
) -> tuple[_ResonanceForms, _ResonanceForms]:
    """Meet at a common witness, expanding the smaller BFS layer first.

    Each side follows validated outgoing migrations from its own seeds. These
    certify equivalence, but sanitization/valence guards can make the generator
    asymmetric. Therefore an empty frontier does NOT terminate the other side,
    and this is not a shortest directed-path algorithm. With no caps/failures,
    a miss requires exhausting both forward closures.

    Supplier seeds have their own budget. At most ``local_budget`` additional
    distinct states per side are retained here; no Cartesian product of the two
    closures is formed. Parents retain the two actual local witness paths.
    """
    seen: list[dict[str, Chem.Mol]] = [{}, {}]
    for i, side in enumerate(seeds):
        for mol, key in side:
            # Keep the prepared source before equivalent supplier seeds,
            # preserving its annotation and Kekule representation precedence.
            seen[i].setdefault(key, mol)
    frontiers = [list(side.items()) for side in seen]
    parents: list[dict[str, str | None]] = [dict.fromkeys(side) for side in seen]
    added = [0, 0]
    expanded = [0, 0]
    limited = [False, False]

    def results(hit=None):
        paths = [[], []]
        if hit is not None:
            for i in (0, 1):
                key = hit
                while key is not None:
                    paths[i].append(key)
                    key = parents[i][key]
                paths[i].reverse()
        return tuple(
            _ResonanceForms(
                tuple(seen[i]),
                supplier_limit_reached=supplier_limited[i],
                local_limit_reached=limited[i],
                target_found=hit is not None,
                validation_failed=contexts[i].validation_failed,
                witness_path=tuple(paths[i]),
            )
            for i in (0, 1)
        )

    common = seen[0].keys() & seen[1].keys()
    if common:
        return results(min(common))
    while frontiers[0] or frontiers[1]:
        active = [i for i in (0, 1) if frontiers[i]]
        # Tie-break by work already done so equal-width paths advance from
        # both ends. A full layer is expanded before selecting the next side.
        i = min(active, key=lambda side: (len(frontiers[side]), expanded[side]))
        layer, frontiers[i] = frontiers[i], []
        for parent_key, form in layer:
            expanded[i] += 1
            for shifted in _iter_charge_shift_resonance_forms(form, context=contexts[i]):
                key = contexts[i].key(shifted)
                if key in seen[i]:
                    continue
                if added[i] >= local_budget:
                    limited[i] = True
                    frontiers[i] = []
                    break
                added[i] += 1
                seen[i][key] = shifted
                parents[i][key] = parent_key
                if key in seen[1 - i]:
                    return results(key)
                frontiers[i].append((key, shifted))
            if limited[i]:
                break
    return results()


@lru_cache(maxsize=128)
def _cached_resonance_smiles(
    source_smiles: str,
    *,
    use_chirality: bool,
    max_resonance: int,
    resonance_flags: int,
    target_smiles: str | frozenset[str] | None = None,
) -> _ResonanceForms:
    # A target-directed prefix must never masquerade as the full enumeration.
    # Include the target(s) in the cache key; exceptions are not cached.
    source_mol = Chem.MolFromSmiles(source_smiles)
    if source_mol is None:
        raise ValueError("Cannot parse resonance source")
    result = _enumerate_resonance_forms(
        source_mol,
        use_chirality=use_chirality,
        max_resonance=max_resonance,
        resonance_flags=resonance_flags,
        target_smiles=target_smiles,
    )
    if result.validation_failed and not result.target_found:
        # Failed validation must be retryable, never memoized as a closure.
        raise ValueError("Resonance candidate validation failed")
    return result


def _enumerate_resonance_forms(
    source_mol: Chem.Mol,
    *,
    use_chirality: bool,
    max_resonance: int,
    resonance_flags: int,
    target_smiles: str | frozenset[str] | None = None,
    context: _ChargeShiftContext | None = None,
    prepared_source: Chem.Mol | None = None,
    supplied_forms: list[tuple[Chem.Mol, str]] | None = None,
    supplied_count: int = 0,
) -> _ResonanceForms:
    targets = frozenset((target_smiles,)) if isinstance(target_smiles, str) else target_smiles
    targets = targets or frozenset()
    if max_resonance <= 0:
        return _ResonanceForms(local_limit_reached=True)
    source = (
        prepared_source
        if prepared_source is not None
        else _prepare_resonance_source(source_mol, use_chirality=use_chirality)
    )
    if context is None or not context.compatible(source):
        context = _ChargeShiftContext(
            source, use_chirality=use_chirality, cache_limit=2 * max_resonance
        )
    source_key = context.key(source)

    def result(
        forms, *, supplier_limit_reached=False, local_limit_reached=False, target_found=False
    ):
        return _ResonanceForms(
            forms,
            supplier_limit_reached=supplier_limit_reached,
            local_limit_reached=local_limit_reached,
            target_found=target_found,
            validation_failed=context.validation_failed,
        )

    if source_key in targets:
        return result((source_key,), target_found=True)

    supplier_forms = [] if supplied_forms is None else supplied_forms
    supplier_count = supplied_count
    if supplied_forms is None:
        supplier = ResonanceMolSupplier(
            source, maxStructs=max_resonance, flags=Chem.ResonanceFlags(resonance_flags)
        )
        for form in supplier:
            supplier_count += 1
            if form is None:
                context.validation_failed = True
                continue
            key = context.key(form)
            supplier_forms.append((form, key))
            if key in targets:
                return result(
                    tuple(dict.fromkeys(key for _, key in supplier_forms)),
                    supplier_limit_reached=supplier_count >= max_resonance,
                    target_found=True,
                )
    for _, key in supplier_forms:
        if key in targets:
            return result(
                tuple(dict.fromkeys(key for _, key in supplier_forms)),
                supplier_limit_reached=supplier_count >= max_resonance,
                target_found=True,
            )
    # Canonical duplicates do not make a capped supplier exhaustive.
    supplier_limited = supplier_count >= max_resonance

    # Expand each seed/state once. Discovered states remain eligible for later
    # expansion; membership in this set only prevents duplicate queue entries.
    queue = [(source, source_key)]
    discovered = {source_key}
    for form, key in supplier_forms:
        if key not in discovered:
            discovered.add(key)
            queue.append((form, key))
    seen: dict[str, None] = {}
    index = 0
    while index < len(queue):
        form, _ = queue[index]
        index += 1
        for shifted in _iter_charge_shift_resonance_forms(form, context=context):
            key = context.key(shifted)
            if key not in discovered:
                discovered.add(key)
                queue.append((shifted, key))
            if key in seen:
                continue
            seen[key] = None
            if key in targets:
                return result(
                    tuple(seen), supplier_limit_reached=supplier_limited, target_found=True
                )
            # Keep the historical priority for local charge-shift forms.
            # At most L supplier seeds plus L discovered shifts are retained.
            if len(seen) >= max_resonance:
                return result(
                    tuple(seen),
                    supplier_limit_reached=supplier_limited,
                    local_limit_reached=True,
                )
    for _, key in supplier_forms:
        if key in seen:
            continue
        if len(seen) >= max_resonance:
            return result(
                tuple(seen),
                supplier_limit_reached=supplier_limited,
                local_limit_reached=True,
            )
        seen[key] = None
    return result(tuple(seen), supplier_limit_reached=supplier_limited)


def _paired_resonance_search(
    mols: tuple[Chem.Mol, Chem.Mol],
    *,
    contexts: tuple[_ChargeShiftContext, _ChargeShiftContext],
    use_chirality: bool,
    max_resonance: int,
    resonance_flags: int,
) -> tuple[_ResonanceForms, _ResonanceForms]:
    """Intersect supplier prefixes before expanding either full local closure.

    Both frontiers advance incrementally. If they do not meet, their validated
    seeds and comparison keys are reused by the local BFS; no second supplier
    construction, molecule serialization or parsing is required.
    """
    if max_resonance <= 0:
        return (_ResonanceForms(local_limit_reached=True),) * 2
    sources = tuple(_prepare_resonance_source(m, use_chirality=use_chirality) for m in mols)
    contexts = tuple(
        context
        if context.compatible(source)
        else _ChargeShiftContext(source, use_chirality=use_chirality, cache_limit=2 * max_resonance)
        for source, context in zip(sources, contexts)
    )
    seeds: list[list[tuple[Chem.Mol, str]]] = [[], []]
    seen = [{contexts[i].key(sources[i]): None} for i in (0, 1)]
    counts = [0, 0]

    def prefix(i):
        return _ResonanceForms(
            tuple(seen[i]),
            supplier_limit_reached=counts[i] >= max_resonance,
            target_found=True,
            validation_failed=contexts[i].validation_failed,
        )

    if seen[0].keys() & seen[1].keys():
        return prefix(0), prefix(1)
    suppliers = [
        iter(
            ResonanceMolSupplier(
                source,
                maxStructs=max_resonance,
                flags=Chem.ResonanceFlags(resonance_flags),
            )
        )
        for source in sources
    ]
    active = [True, True]
    while any(active):
        for i in (0, 1):
            if not active[i]:
                continue
            try:
                form = next(suppliers[i])
            except StopIteration:
                active[i] = False
                continue
            counts[i] += 1
            if counts[i] >= max_resonance:
                active[i] = False
            if form is None:
                contexts[i].validation_failed = True
                continue
            key = contexts[i].key(form)
            seeds[i].append((form, key))
            seen[i][key] = None
            if key in seen[1 - i]:
                return prefix(0), prefix(1)

    return _bidirectional_charge_shift_search(
        tuple([(sources[i], contexts[i].key(sources[i])), *seeds[i]] for i in (0, 1)),
        contexts=contexts,
        local_budget=max_resonance,
        supplier_limited=tuple(count >= max_resonance for count in counts),
    )


def _resonance_topology_key(mol: Chem.Mol, *, preserve_metal_formal_charge: bool = False) -> str:
    rw_mol = Chem.RWMol(Chem.Mol(mol))
    for atom in rw_mol.GetAtoms():
        if not (preserve_metal_formal_charge and _is_metal_atom(atom)):
            atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw_mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetBondDir(Chem.BondDir.NONE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    topology = rw_mol.GetMol()
    topology.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(topology, canonical=True, isomericSmiles=True)


def _has_defined_stereochemistry(mol: Chem.Mol) -> bool:
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return True
    return any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds())


def _component_electron_signature(
    mol: Chem.Mol, *, topology_key: str | None = None
) -> tuple[tuple[str, int, int], ...]:
    if topology_key is not None and len(Chem.GetMolFrags(mol)) == 1:
        return ((topology_key, _total_formal_charge(mol), _total_radical_electrons(mol)),)
    signature = []
    for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        signature.append(
            (
                _resonance_topology_key(fragment),
                _total_formal_charge(fragment),
                _total_radical_electrons(fragment),
            )
        )
    return tuple(sorted(signature))


def _open_shell_heuristic_applies(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    *,
    use_chirality: bool,
) -> bool:
    """Return whether only open-shell components differ in representation.

    This is ambiguity evidence, not proof of resonance reachability. Grouping by
    topology and component electron totals prevents an unrelated radical fragment
    from qualifying a representation change in a closed-shell component.
    """

    def component_representations(mol: Chem.Mol) -> dict[tuple[str, int, int], Counter[str]]:
        representations: dict[tuple[str, int, int], Counter[str]] = {}
        for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
            key = (
                _resonance_topology_key(fragment),
                _total_formal_charge(fragment),
                _total_radical_electrons(fragment),
            )
            representations.setdefault(key, Counter())[_canon_smiles(fragment, use_chirality)] += 1
        return representations

    representations_1 = component_representations(mol1)
    representations_2 = component_representations(mol2)
    if representations_1.keys() != representations_2.keys():
        return False

    representation_changed = False
    for key, forms_1 in representations_1.items():
        forms_2 = representations_2[key]
        if forms_1 == forms_2:
            continue
        representation_changed = True
        if key[2] == 0:
            return False
    return representation_changed


def _topology_query(mol: Chem.Mol) -> Chem.Mol:
    """Build an element/connectivity query without electronic-state constraints."""

    query = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        query.ReplaceAtom(
            atom.GetIdx(),
            Chem.AtomFromSmarts(f"[#{atom.GetAtomicNum()}]"),
        )
    for bond in mol.GetBonds():
        query.ReplaceBond(bond.GetIdx(), Chem.BondFromSmarts("~"))
    return query.GetMol()


def _mapped_hydrogen_count(atom: Chem.Atom) -> int:
    return int(atom.GetTotalNumHs(includeNeighbors=True))


def _bond_order_sum(atom: Chem.Atom) -> float:
    return sum(float(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds())


def _elementary_open_shell_resonance_witness(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    *,
    use_chirality: bool,
    atom_mapping: tuple[int, ...] | None = None,
) -> tuple[OpenShellThreeCenterDetail | None, TopologyMappingSearchMetadata]:
    """Find an explicit elementary ``A radical-B=C <-> A=B-C radical`` witness.

    A positive result requires a topology-preserving atom mapping and exactly
    one three-center electron shift. It deliberately excludes charge motion,
    proton motion, aromatic/non-integral bond changes, and any additional atom
    or bond change. Failure to find a mapping is only absence of this proof;
    it is not evidence of non-equivalence.
    """

    search = TopologyMappingSearchMetadata()
    radicals_1 = _total_radical_electrons(mol1)
    radicals_2 = _total_radical_electrons(mol2)
    if radicals_1 == 0 or radicals_1 != radicals_2:
        return None, search
    if mol1.GetNumAtoms() != mol2.GetNumAtoms() or mol1.GetNumBonds() != mol2.GetNumBonds():
        return None, search

    search.attempted = True
    mappings = (
        (atom_mapping,)
        if atom_mapping is not None
        else mol2.GetSubstructMatches(
            _topology_query(mol1),
            uniquify=False,
            useChirality=False,
            maxMatches=10000,
        )
    )
    search.limit_reached = len(mappings) >= search.limit
    for mapping_index, mapping in enumerate(mappings):
        search.mappings_examined = mapping_index + 1
        if len(mapping) != mol1.GetNumAtoms():
            continue

        radical_deltas: dict[int, int] = {}
        atom_mapping_valid = True
        for atom1 in mol1.GetAtoms():
            idx1 = atom1.GetIdx()
            atom2 = mol2.GetAtomWithIdx(mapping[idx1])
            if (
                atom1.GetAtomicNum() != atom2.GetAtomicNum()
                or atom1.GetIsotope() != atom2.GetIsotope()
                or atom1.GetFormalCharge() != atom2.GetFormalCharge()
                or _mapped_hydrogen_count(atom1) != _mapped_hydrogen_count(atom2)
                or (use_chirality and atom1.GetChiralTag() != atom2.GetChiralTag())
            ):
                atom_mapping_valid = False
                break
            radical_delta = int(atom2.GetNumRadicalElectrons()) - int(
                atom1.GetNumRadicalElectrons()
            )
            if radical_delta:
                radical_deltas[idx1] = radical_delta
        if not atom_mapping_valid or sorted(radical_deltas.values()) != [-1, 1]:
            continue

        changed_bonds: list[tuple[int, int, int]] = []
        bond_mapping_valid = True
        for bond1 in mol1.GetBonds():
            begin1, end1 = bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()
            bond2 = mol2.GetBondBetweenAtoms(mapping[begin1], mapping[end1])
            if bond2 is None:
                bond_mapping_valid = False
                break
            order1 = float(bond1.GetBondTypeAsDouble())
            order2 = float(bond2.GetBondTypeAsDouble())
            if order1 == order2:
                if use_chirality and bond1.GetStereo() != bond2.GetStereo():
                    bond_mapping_valid = False
                    break
                continue
            if bond1.GetIsAromatic() or bond2.GetIsAromatic():
                bond_mapping_valid = False
                break
            delta = order2 - order1
            if delta not in (-1.0, 1.0):
                bond_mapping_valid = False
                break
            if use_chirality and (
                bond1.GetStereo() != Chem.BondStereo.STEREONONE
                or bond2.GetStereo() != Chem.BondStereo.STEREONONE
            ):
                bond_mapping_valid = False
                break
            changed_bonds.append((begin1, end1, int(delta)))
        if not bond_mapping_valid or len(changed_bonds) != 2:
            continue
        if changed_bonds[0][2] == changed_bonds[1][2]:
            continue

        adjacency: dict[int, list[int]] = {}
        for begin, end, _ in changed_bonds:
            adjacency.setdefault(begin, []).append(end)
            adjacency.setdefault(end, []).append(begin)
        centers = [idx for idx, neighbors in adjacency.items() if len(neighbors) == 2]
        endpoints = {idx for idx, neighbors in adjacency.items() if len(neighbors) == 1}
        if len(centers) != 1 or endpoints != set(radical_deltas):
            continue
        center = centers[0]

        # Local Lewis-electron bookkeeping: the change in bond-order sum is
        # exactly balanced by the opposite change in radical count at every
        # atom touched by the shift.
        if any(
            abs(
                (
                    _bond_order_sum(mol1.GetAtomWithIdx(idx))
                    + mol1.GetAtomWithIdx(idx).GetNumRadicalElectrons()
                )
                - (
                    _bond_order_sum(mol2.GetAtomWithIdx(mapping[idx]))
                    + mol2.GetAtomWithIdx(mapping[idx]).GetNumRadicalElectrons()
                )
            )
            > 1e-8
            for idx in adjacency
        ):
            continue

        ordered_endpoints = sorted(endpoints)
        mol1_atoms = (ordered_endpoints[0], center, ordered_endpoints[1])
        mol2_atoms = (mapping[mol1_atoms[0]], mapping[center], mapping[mol1_atoms[2]])
        return OpenShellThreeCenterDetail(
            mol1_atoms=mol1_atoms,
            mol2_atoms=mol2_atoms,
            mol1_radical_electrons=(
                int(mol1.GetAtomWithIdx(mol1_atoms[0]).GetNumRadicalElectrons()),
                int(mol1.GetAtomWithIdx(mol1_atoms[1]).GetNumRadicalElectrons()),
                int(mol1.GetAtomWithIdx(mol1_atoms[2]).GetNumRadicalElectrons()),
            ),
            mol2_radical_electrons=(
                int(mol2.GetAtomWithIdx(mol2_atoms[0]).GetNumRadicalElectrons()),
                int(mol2.GetAtomWithIdx(mol2_atoms[1]).GetNumRadicalElectrons()),
                int(mol2.GetAtomWithIdx(mol2_atoms[2]).GetNumRadicalElectrons()),
            ),
            mol1_bond_orders=(
                float(mol1.GetBondBetweenAtoms(mol1_atoms[0], mol1_atoms[1]).GetBondTypeAsDouble()),
                float(mol1.GetBondBetweenAtoms(mol1_atoms[1], mol1_atoms[2]).GetBondTypeAsDouble()),
            ),
            mol2_bond_orders=(
                float(mol2.GetBondBetweenAtoms(mol2_atoms[0], mol2_atoms[1]).GetBondTypeAsDouble()),
                float(mol2.GetBondBetweenAtoms(mol2_atoms[1], mol2_atoms[2]).GetBondTypeAsDouble()),
            ),
        ), search
    return None, search


def _metal_signature_key(mol: Chem.Mol) -> tuple[tuple[int, int], ...]:
    signature = []
    for atom in mol.GetAtoms():
        if not _is_metal_atom(atom):
            continue
        signature.append((int(atom.GetAtomicNum()), int(atom.GetFormalCharge())))
    return tuple(sorted(signature))


def _carbene_zwitterion_normalized_smiles(mol: Chem.Mol, *, use_chirality: bool) -> str:
    normalized = _safe_copy(mol)
    rw_mol = Chem.RWMol(normalized)
    _kekulize_if_possible(rw_mol)

    if _CARBENE_ZWITTERION_PATTERN is not None:
        for match in rw_mol.GetSubstructMatches(_CARBENE_ZWITTERION_PATTERN):
            begin_idx, end_idx = (int(atom_idx) for atom_idx in match)
            bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
            atom_begin = rw_mol.GetAtomWithIdx(begin_idx)
            atom_end = rw_mol.GetAtomWithIdx(end_idx)
            if bond is None:
                continue
            if (
                pt.GetDefaultValence(atom_begin.GetAtomicNum()) - atom_begin.GetTotalValence() == 1
                and pt.GetDefaultValence(atom_end.GetAtomicNum()) - atom_end.GetTotalValence() == -1
            ):
                bond.SetBondType(Chem.BondType.SINGLE)
                atom_begin.SetFormalCharge(atom_begin.GetFormalCharge() + 1)
                atom_end.SetFormalCharge(atom_end.GetFormalCharge() - 1)
                atom_begin.SetNumRadicalElectrons(2)
                atom_end.SetNumRadicalElectrons(0)

    normalized_mol = rw_mol.GetMol()
    normalized_mol.UpdatePropertyCache(strict=False)
    _sanitize_if_possible(normalized_mol)
    return _canon_smiles(normalized_mol, use_chirality)


def _normalize_special_resonance_forms(mol: Chem.Mol) -> Chem.Mol:
    normalized = _safe_copy(mol)
    rw_mol = Chem.RWMol(normalized)

    if _SULFIMIDE_RESONANCE_PATTERN is not None:
        base = rw_mol.GetMol()
        for match in base.GetSubstructMatches(_SULFIMIDE_RESONANCE_PATTERN):
            n_idx, s_idx, _, o_minus_idx, _ = (int(atom_idx) for atom_idx in match)
            n_atom = rw_mol.GetAtomWithIdx(n_idx)
            o_minus_atom = rw_mol.GetAtomWithIdx(o_minus_idx)
            ns_bond = rw_mol.GetBondBetweenAtoms(n_idx, s_idx)
            so_bond = rw_mol.GetBondBetweenAtoms(s_idx, o_minus_idx)
            if ns_bond is None or so_bond is None:
                continue
            ns_bond.SetBondType(Chem.BondType.SINGLE)
            so_bond.SetBondType(Chem.BondType.DOUBLE)
            n_atom.SetFormalCharge(n_atom.GetFormalCharge() - 1)
            o_minus_atom.SetFormalCharge(0)

    def collapse_thiosemicarbazone_match(match: tuple[int, ...]) -> None:
        n1_idx, n2_idx, c_idx, s_idx, n3_idx = (int(atom_idx) for atom_idx in match)
        for begin_idx, end_idx in (
            (n1_idx, n2_idx),
            (n2_idx, c_idx),
            (c_idx, s_idx),
            (c_idx, n3_idx),
        ):
            bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
            if bond is not None:
                bond.SetBondType(Chem.BondType.SINGLE)
                bond.SetIsAromatic(False)
        for atom_idx in (n1_idx, n2_idx, c_idx, s_idx, n3_idx):
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetIsAromatic(False)

    base = rw_mol.GetMol()
    for pattern in _THIOSEMICARBAZONE_RESONANCE_PATTERNS:
        for match in base.GetSubstructMatches(pattern):
            collapse_thiosemicarbazone_match(match)

    normalized_mol = rw_mol.GetMol()
    normalized_mol.UpdatePropertyCache(strict=False)
    _sanitize_if_possible(normalized_mol)
    return normalized_mol


@dataclass(frozen=True)
class _RegionalChargeShiftWitness:
    molecule: Chem.Mol
    mapping: tuple[int, ...]
    steps: tuple[tuple[int, int, int], ...]
    region_count: int
    state_count: int
    distance_pruned: int


def _charge_shift_atom_mapping(left: Chem.Mol, right: Chem.Mol) -> tuple[int, ...]:
    """One label-preserving topology mapping, without canonical serialization.

    Temporary isotope colors encode immutable labels only on private topology
    copies. Electronic charges/bond orders deliberately do not constrain the
    mapping. Failure of this one alignment is not negative evidence.
    """
    if left.GetNumAtoms() != right.GetNumAtoms() or left.GetNumBonds() != right.GetNumBonds():
        return ()
    signatures = [
        [
            (
                a.GetAtomicNum(),
                a.GetIsotope(),
                a.GetAtomMapNum(),
                a.GetNumRadicalElectrons(),
                _mapped_hydrogen_count(a),
            )
            for a in mol.GetAtoms()
        ]
        for mol in (left, right)
    ]
    if Counter(signatures[0]) != Counter(signatures[1]):
        return ()

    def valid(candidate):
        return all(signatures[0][i] == signatures[1][j] for i, j in enumerate(candidate)) and all(
            right.GetBondBetweenAtoms(candidate[b.GetBeginAtomIdx()], candidate[b.GetEndAtomIdx()])
            is not None
            for b in left.GetBonds()
        )

    # Equal indices are only a candidate, never an assumed correspondence.
    # Verify invariant labels and every edge before using it as an alignment.
    identity = tuple(range(left.GetNumAtoms()))
    if valid(identity):
        return identity

    # A fixed annotation mapping supplies unique maps on the original atoms.
    # Added unmapped hydrogens can be paired by parent map and immutable label.
    # This recovers that correspondence without copying/relabeling topologies.
    def mapped_groups(mol, labels):
        groups = {}
        seen_maps = set()
        for atom, label in zip(mol.GetAtoms(), labels):
            number = atom.GetAtomMapNum()
            if number:
                if number in seen_maps:
                    return None
                seen_maps.add(number)
                key = (number,)
            elif atom.GetAtomicNum() == 1 and atom.GetDegree() == 1:
                parent = atom.GetNeighbors()[0].GetAtomMapNum()
                if not parent:
                    return None
                key = (0, parent, label)
            else:
                return None
            groups.setdefault(key, []).append(atom.GetIdx())
        return groups

    groups = mapped_groups(left, signatures[0])
    if groups is not None:
        other_groups = mapped_groups(right, signatures[1])
        if other_groups is not None and groups.keys() == other_groups.keys():
            candidate = [-1] * left.GetNumAtoms()
            for label, indices in groups.items():
                other_indices = other_groups[label]
                if len(indices) != len(other_indices):
                    break
                for i, j in zip(indices, other_indices):
                    candidate[i] = j
            else:
                if valid(candidate):
                    return tuple(candidate)

    colors = {signature: i + 1 for i, signature in enumerate(dict.fromkeys(signatures[0]))}
    topologies = []
    for mol, labels in zip((left, right), signatures):
        topology = Chem.RWMol(mol)
        for atom, label in zip(topology.GetAtoms(), labels):
            atom.SetIsotope(colors[label])
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetIsAromatic(False)
            atom.SetNoImplicit(True)
        for bond in topology.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        topologies.append(topology.GetMol())
    mapping = topologies[1].GetSubstructMatch(topologies[0], useChirality=False)
    if any(signatures[0][i] != signatures[1][j] for i, j in enumerate(mapping)):
        return ()
    return mapping


def _charge_shift_corridor(adjacency, source_charges, target_charges, max_steps):
    """Necessary spatial envelope for at most k elementary two-bond hops.

    Multi-source distances allow interchangeable anions: the nearest source
    and target need not belong to the same eventual trajectory. This weakens
    the bound but cannot exclude a trajectory of at most 2k bond traversals.
    """

    def distances(charges):
        roots = [i for i, charge in enumerate(charges) if charge == -1]
        result = dict.fromkeys(roots, 0)
        for atom in roots:
            for _, other in adjacency[atom]:
                if other not in result:
                    result[other] = result[atom] + 1
                    roots.append(other)
        return result

    from_source, to_target = distances(source_charges), distances(target_charges)
    allowed = frozenset(
        atom
        for atom in from_source.keys() & to_target.keys()
        if from_source[atom] + to_target[atom] <= 2 * max_steps
    )
    return allowed, to_target


def _search_charge_shift_region(
    atoms, bonds, source, target, adjacency, valences, budget, *, max_steps
):
    """BFS over a projected region, with separate step and state budgets.

    Unchanged boundary bonds are frozen. The two-ended 2k corridor and remaining
    bond-edit/charge-distance lower bounds prune only paths exceeding max_steps.
    The caller first tries the minimum bond-edit step budget; a restricted miss
    is not negative evidence and falls back to the general resonance search.
    """
    target_charges, target_orders = target
    allowed, to_target = _charge_shift_corridor(adjacency, source[0], target_charges, max_steps)
    parents = {source: None}
    queue = [(source, 0)]
    pruned = 0
    for state, depth in queue:
        if depth >= max_steps:
            continue
        charges, orders = state
        remaining = max_steps - depth - 1
        for negative, charge in enumerate(charges):
            if charge != -1 or negative not in allowed:
                continue
            # V - q is invariant under a supported elementary migration.
            initial_valence, default_valence = valences[negative]
            if (
                default_valence <= 0
                or initial_valence + charge - source[0][negative] + 1 > default_valence
            ):
                continue
            for single_idx, center in adjacency[negative]:
                if orders[single_idx] != 1 or charges[center] != 0:
                    continue
                for double_idx, terminal in adjacency[center]:
                    if terminal == negative or orders[double_idx] != 2 or charges[terminal] != 0:
                        continue
                    if (
                        center not in allowed
                        or terminal not in allowed
                        or to_target[terminal] > 2 * remaining
                    ):
                        pruned += 1
                        continue
                    new_charges, new_orders = list(charges), list(orders)
                    new_charges[negative], new_charges[terminal] = 0, -1
                    new_orders[single_idx], new_orders[double_idx] = 2, 1
                    # A migration can correct at most two units of bond-order
                    # difference. This also forbids undoing settled bonds when
                    # the path is already at the minimum feasible step budget.
                    if sum(abs(a - b) for a, b in zip(new_orders, target_orders)) > 2 * remaining:
                        pruned += 1
                        continue
                    child = (tuple(new_charges), tuple(new_orders))
                    if child in parents:
                        continue
                    if len(parents) - 1 >= budget:
                        return None
                    move = (
                        atoms[negative],
                        atoms[center],
                        atoms[terminal],
                        bonds[single_idx],
                        bonds[double_idx],
                    )
                    parents[child] = (state, move)
                    if child == target:
                        moves = []
                        while parents[child] is not None:
                            child, step = parents[child]
                            moves.append(step)
                        return tuple(reversed(moves)), len(parents) - 1, pruned
                    queue.append((child, depth + 1))
    return None


def _regional_charge_shift_witness(
    mol1: Chem.Mol, mol2: Chem.Mol, *, use_chirality: bool, max_resonance: int
) -> _RegionalChargeShiftWitness | None:
    """Try a factored, target-directed proof before unrestricted enumeration.

    Regions are vertex-disjoint components of changed bonds under one explicit
    atom mapping, not disconnected molecular fragments. Unchanged connecting
    bonds and spectator regions stay frozen. A successful abstract path is
    replayed on the FULL molecule with ordinary sanitization after every move;
    any unmodeled change rejects this optimization. Restricted misses, mapping
    failures and budget exhaustion never supply a negative verdict.
    """
    if max_resonance <= 1:
        return None
    negative_counts = [sum(a.GetFormalCharge() == -1 for a in m.GetAtoms()) for m in (mol1, mol2)]
    if not all(negative_counts) or negative_counts[0] != negative_counts[1]:
        return None
    # Leave isolated single-hop cases to the cheaper existing direct probe.
    if (
        negative_counts[0] == 1
        and sum(b.GetBondType() == Chem.BondType.DOUBLE for b in mol1.GetBonds()) < 2
    ):
        return None
    if use_chirality and any(_has_defined_stereochemistry(m) for m in (mol1, mol2)):
        return None
    sources = tuple(
        _add_hs_without_sanitize(m) if any(a.GetTotalNumHs() for a in m.GetAtoms()) else m
        for m in (mol1, mol2)
    )
    left, right = sources
    mapping = _charge_shift_atom_mapping(left, right)
    if not mapping:
        return None
    source_charges = tuple(a.GetFormalCharge() for a in left.GetAtoms())
    target_charges = tuple(right.GetAtomWithIdx(i).GetFormalCharge() for i in mapping)
    changed_atoms = {i for i, (a, b) in enumerate(zip(source_charges, target_charges)) if a != b}
    changed_bonds = {}
    graph = {}
    allowed_types = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
    for bond in left.GetBonds():
        a, b, idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetIdx()
        other = right.GetBondBetweenAtoms(mapping[a], mapping[b])
        if (
            bond.GetBondType() == other.GetBondType()
            and bond.GetIsAromatic() == other.GetIsAromatic()
        ):
            continue
        if (
            bond.GetBondType() not in allowed_types
            or other.GetBondType() not in allowed_types
            or bond.GetIsAromatic()
            or other.GetIsAromatic()
        ):
            return None
        changed_bonds[idx] = (int(bond.GetBondTypeAsDouble()), int(other.GetBondTypeAsDouble()))
        graph.setdefault(a, []).append((idx, b))
        graph.setdefault(b, []).append((idx, a))
    if not graph or not changed_atoms.issubset(graph):
        return None
    for idx in graph:
        atom = left.GetAtomWithIdx(idx)
        if (
            atom.GetNumRadicalElectrons()
            or atom.GetIsAromatic()
            or source_charges[idx] not in (-1, 0)
            or target_charges[idx] not in (-1, 0)
        ):
            return None
    # Independent difference components, including regions joined by unchanged
    # bonds in the same molecule. Shared atoms necessarily join the same region.
    remaining = set(graph)
    regions = []
    while remaining:
        atoms = [min(remaining)]
        remaining.remove(atoms[0])
        for atom in atoms:
            for _, other in graph[atom]:
                if other in remaining:
                    remaining.remove(other)
                    atoms.append(other)
        regions.append(tuple(atoms))
    steps = []
    used, pruned = 1, 0
    for atoms in regions:
        bonds = tuple(sorted({idx for atom in atoms for idx, _ in graph[atom]}))
        atom_indices, bond_indices = (
            {a: i for i, a in enumerate(atoms)},
            {b: i for i, b in enumerate(bonds)},
        )
        source = (
            tuple(source_charges[i] for i in atoms),
            tuple(changed_bonds[i][0] for i in bonds),
        )
        target = (
            tuple(target_charges[i] for i in atoms),
            tuple(changed_bonds[i][1] for i in bonds),
        )
        if sum(source[0]) != sum(target[0]) or sum(source[1]) != sum(target[1]):
            return None
        adjacency = tuple(
            tuple((bond_indices[b], atom_indices[a]) for b, a in graph[i]) for i in atoms
        )
        valences = tuple(
            (
                left.GetAtomWithIdx(i).GetTotalValence(),
                pt.GetDefaultValence(left.GetAtomWithIdx(i).GetAtomicNum()),
            )
            for i in atoms
        )
        # Each move edits two bonds by one order unit. Try a shortest-edit
        # proof first, then fall back rather than assume every pair admits one.
        step_limit = sum(abs(a - b) for a, b in zip(source[1], target[1])) // 2
        if step_limit > max_resonance - used:
            return None
        found = _search_charge_shift_region(
            atoms,
            bonds,
            source,
            target,
            adjacency,
            valences,
            max_resonance - used,
            max_steps=step_limit,
        )
        if found is None:
            return None
        moves, count, rejected = found
        steps.extend(moves)
        used += count
        pruned += rejected
    # Abstract reachability alone is not chemistry evidence. Replay only the
    # successful path, validating that every step has EXACTLY the modeled effect.
    context = _ChargeShiftContext(left, use_chirality=use_chirality, cache_limit=max_resonance)
    current = left
    for negative, center, terminal, single_idx, double_idx in steps:
        state = context.state(current)
        shifted = context.move(current, state, negative, center, terminal, single_idx, double_idx)
        if shifted is None or not context.compatible(shifted):
            return None
        atoms, bonds = list(state[0]), list(state[1])
        atoms[negative] = (0, *atoms[negative][1:])
        atoms[terminal] = (-1, *atoms[terminal][1:])
        bonds[single_idx] = (Chem.BondType.DOUBLE, *bonds[single_idx][1:])
        bonds[double_idx] = (Chem.BondType.SINGLE, *bonds[double_idx][1:])
        if context.state(shifted) != (tuple(atoms), tuple(bonds)):
            return None
        current = shifted
    if not _mapped_graph_identity(current, right, mapping, use_chirality=use_chirality):
        return None
    return _RegionalChargeShiftWitness(
        current, mapping, tuple(step[:3] for step in steps), len(regions), used, pruned
    )


def _resonance_match(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    *,
    use_chirality: bool,
    max_resonance: int,
    resonance_flags: Chem.ResonanceFlags,
) -> _ResonanceMatchResult:
    regional_sources = (mol1, mol2)

    # Give bounded enumeration a reproducible atom order without a SMILES
    # roundtrip, preserving annotations and explicit hydrogen identities.
    def ordered(mol):
        ranks = Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=use_chirality)
        order = sorted(
            range(mol.GetNumAtoms()),
            key=lambda idx: (mol.GetAtomWithIdx(idx).GetAtomicNum() == 1, ranks[idx]),
        )
        return Chem.RenumberAtoms(mol, order)

    mol1, mol2 = ordered(mol1), ordered(mol2)
    searches: list[_ResonanceForms] = []

    def result(
        left: _ResonanceForms, right: _ResonanceForms | None, target: str
    ) -> _ResonanceMatchResult:
        left_keys = set(left)
        right_keys = set(right) if right is not None else set()
        common = left_keys & right_keys
        hit = target if target in left_keys else min(common) if common else None
        if left.witness_path and right is not None and right.witness_path:
            # Report the meeting that the retained parent chains actually prove.
            hit = left.witness_path[-1]
        return _ResonanceMatchResult(
            hit is not None,
            len(left),
            len(right) if right is not None else 0,
            hit,
            supplier_limit_reached=any(s.supplier_limit_reached for s in searches),
            local_limit_reached=any(s.local_limit_reached for s in searches),
            validation_failed=any(s.validation_failed for s in searches),
            local_witness_paths=(left.witness_path, right.witness_path if right else ()),
        )

    contexts = (
        _ChargeShiftContext(mol1, use_chirality=use_chirality, cache_limit=2 * max_resonance),
        _ChargeShiftContext(mol2, use_chirality=use_chirality, cache_limit=2 * max_resonance),
    )
    raw_target_2 = _canon_smiles(mol2, use_chirality)
    if max_resonance <= 0:
        return _ResonanceMatchResult(False, 0, 0, None, local_limit_reached=True)
    source1 = _prepare_resonance_source(mol1, use_chirality=use_chirality)
    if not contexts[0].compatible(source1):
        contexts = (
            _ChargeShiftContext(
                source1, use_chirality=use_chirality, cache_limit=2 * max_resonance
            ),
            contexts[1],
        )
    key1 = contexts[0].key(source1)
    if key1 == raw_target_2:
        return result(_ResonanceForms((key1,), target_found=True), None, raw_target_2)
    # Check a direct local edge before preparing the other end. This is the
    # first layer the two-ended scheduler would expand anyway. On a miss its
    # validated neighbors/keys are reused from the context by the BFS below.
    direct_keys = {key1: None}
    if max_resonance > 1:
        for shifted in _iter_charge_shift_resonance_forms(source1, context=contexts[0]):
            key = contexts[0].key(shifted)
            if key in direct_keys:
                continue
            if len(direct_keys) >= max_resonance:
                break
            direct_keys[key] = None
            if key == raw_target_2:
                direct = _ResonanceForms(
                    tuple(direct_keys),
                    target_found=True,
                    validation_failed=contexts[0].validation_failed,
                    witness_path=(key1, key),
                )
                searches.append(direct)
                return result(direct, None, raw_target_2)

    # Pay for alignment only when the first layer exposes multiple distinct
    # successors: simple chains (including symmetry-collapsed independent sites)
    # are cheap to traverse. Skipping this optional proof is not a verdict.
    # Keep the original representation; preparation may Kekulize spectator rings.
    try:
        regional = (
            _regional_charge_shift_witness(
                *regional_sources, use_chirality=use_chirality, max_resonance=max_resonance
            )
            if len(direct_keys) > 2
            else None
        )
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        # This optional positive probe cannot disable the independent complete
        # search if alignment or abstract-path validation fails unexpectedly.
        regional = None
    if regional is not None:
        return _ResonanceMatchResult(
            True,
            len(regional.steps) + 1,
            1,
            _resonance_form_smiles(regional.molecule, use_chirality),
            regional_witness=regional,
        )

    source2 = _prepare_resonance_source(mol2, use_chirality=use_chirality)
    if not contexts[1].compatible(source2):
        contexts = (
            contexts[0],
            _ChargeShiftContext(
                source2, use_chirality=use_chirality, cache_limit=2 * max_resonance
            ),
        )
    left, right = _bidirectional_charge_shift_search(
        ([(source1, key1)], [(source2, contexts[1].key(source2))]),
        contexts=contexts,
        local_budget=max_resonance - 1,
    )
    searches.extend((left, right))
    raw_result = result(left, right, raw_target_2)
    if raw_result.matched:
        return raw_result

    # Keep raw charge migration before the domain-specific normalizers, which
    # intentionally collapse electronic distinctions in selected motifs.
    normalized_1 = _normalize_special_resonance_forms(mol1)
    normalized_2 = _normalize_special_resonance_forms(mol2)
    left, right = _paired_resonance_search(
        (normalized_1, normalized_2),
        contexts=contexts,
        use_chirality=use_chirality,
        max_resonance=max_resonance,
        resonance_flags=int(resonance_flags),
    )
    searches.extend((left, right))
    return result(left, right, "")


_ELECTRON_PROPERTY_MAPPING_LIMIT = 10000
_ELECTRON_PROPERTY_NAMES = (
    "MOLGR_METAL_UNPAIRED_ELECTRONS",
    "MOLGR_LONE_PAIR_COUNT",
    "MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER",
)


def _electron_annotations(mol: Chem.Mol) -> list[tuple[int | None, int | None, int | None]]:
    return [
        (
            _read_electron_property(atom, _ELECTRON_PROPERTY_NAMES[0])
            if _is_metal_atom(atom)
            else None,
            _read_electron_property(atom, _ELECTRON_PROPERTY_NAMES[1]),
            _read_electron_property(atom, _ELECTRON_PROPERTY_NAMES[2], boolean=True),
        )
        for atom in mol.GetAtoms()
    ]


def _electron_coverage(mol: Chem.Mol) -> dict[str, object]:
    annotations = _electron_annotations(mol)
    known = sum(value is not None for fields in annotations for value in fields)
    return {
        "known_fields": known,
        "unknown_fields": 3 * len(annotations) - known,
        "comparable_fields": 0,
        "deferred_centers": sum(fields[2] is True for fields in annotations),
        "mappings_examined": 0,
        "mapping_limit_reached": False,
    }


def _electronic_property_mappings(mol1: Chem.Mol, mol2: Chem.Mol):
    if _ELECTRON_PROPERTY_MAPPING_LIMIT <= 0:
        return
    query = _topology_query(mol1)
    first = mol2.GetSubstructMatch(query, useChirality=False)
    if not first:
        return
    yield first
    if _ELECTRON_PROPERTY_MAPPING_LIMIT == 1:
        return
    # Most compatible pairs accept the first mapping. Avoid materializing
    # thousands of explicit-H permutations until an alternative is needed.
    mappings = mol2.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=False,
        maxMatches=_ELECTRON_PROPERTY_MAPPING_LIMIT,
    )
    for mapping in mappings:
        if mapping != first:
            yield mapping


def _metal_bonds_follow_mapping(mol1: Chem.Mol, mol2: Chem.Mol, mapping: tuple[int, ...]) -> bool:
    """Require the same metal-incident edges under an annotation mapping."""

    def metal_bonds(mol: Chem.Mol):
        return (
            bond
            for bond in mol.GetBonds()
            if _is_metal_atom(bond.GetBeginAtom()) or _is_metal_atom(bond.GetEndAtom())
        )

    bonds1 = tuple(metal_bonds(mol1))
    bonds2 = tuple(metal_bonds(mol2))
    return len(bonds1) == len(bonds2) and all(
        mol2.GetBondBetweenAtoms(mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()])
        is not None
        for bond in bonds1
    )


def _mapped_graph_identity(
    mol1: Chem.Mol, mol2: Chem.Mol, mapping: tuple[int, ...], *, use_chirality: bool
) -> bool:
    """Prove identity under a fixed mapping without canonical serialization.

    Defined stereo uses the existing canonical fallback: raw chiral tags alone
    do not encode parity independently of atom/neighbour ordering.
    """
    if use_chirality and (_has_defined_stereochemistry(mol1) or _has_defined_stereochemistry(mol2)):
        return False
    if (
        mol1.GetNumAtoms() != mol2.GetNumAtoms()
        or mol1.GetNumBonds() != mol2.GetNumBonds()
        or len(mapping) != mol1.GetNumAtoms()
        or set(mapping) != set(range(mol2.GetNumAtoms()))
    ):
        return False
    for atom1 in mol1.GetAtoms():
        atom2 = mol2.GetAtomWithIdx(mapping[atom1.GetIdx()])
        if (
            atom1.GetAtomicNum() != atom2.GetAtomicNum()
            or atom1.GetIsotope() != atom2.GetIsotope()
            or atom1.GetFormalCharge() != atom2.GetFormalCharge()
            or atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons()
            or atom1.GetIsAromatic() != atom2.GetIsAromatic()
            or _mapped_hydrogen_count(atom1) != _mapped_hydrogen_count(atom2)
        ):
            return False
    for bond1 in mol1.GetBonds():
        begin, end = mapping[bond1.GetBeginAtomIdx()], mapping[bond1.GetEndAtomIdx()]
        bond2 = mol2.GetBondBetweenAtoms(begin, end)
        if (
            bond2 is None
            or bond1.GetBondType() != bond2.GetBondType()
            or bond1.GetIsAromatic() != bond2.GetIsAromatic()
            or (bond1.GetBondType() == Chem.BondType.DATIVE and bond2.GetBeginAtomIdx() != begin)
        ):
            return False
    return True


def _mapped_electronic_structure_match(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    mapping: tuple[int, ...],
    *,
    use_chirality: bool,
    max_resonance: int,
    resonance_flags: Chem.ResonanceFlags,
) -> tuple[bool, bool]:
    """Bind an existing structural proof to this particular annotation mapping."""

    if _mapped_graph_identity(mol1, mol2, mapping, use_chirality=use_chirality):
        return True, False
    left, right = Chem.Mol(mol1), Chem.Mol(mol2)
    for idx1, idx2 in enumerate(mapping):
        atom1, atom2 = left.GetAtomWithIdx(idx1), right.GetAtomWithIdx(idx2)
        if (
            atom1.GetIsotope() != atom2.GetIsotope()
            or _mapped_hydrogen_count(atom1) != _mapped_hydrogen_count(atom2)
            or (_is_metal_atom(atom1) and atom1.GetFormalCharge() != atom2.GetFormalCharge())
        ):
            return False, False
        # Atom maps survive the existing SMILES/resonance paths. They prevent
        # a second, incompatible symmetry mapping from supplying the proof.
        atom1.SetAtomMapNum(idx1 + 1)
        atom2.SetAtomMapNum(idx1 + 1)
    if _canon_smiles(left, use_chirality) == _canon_smiles(right, use_chirality):
        return True, False
    if _carbene_zwitterion_normalized_smiles(
        left, use_chirality=use_chirality
    ) == _carbene_zwitterion_normalized_smiles(right, use_chirality=use_chirality):
        return True, False
    witness, _ = _elementary_open_shell_resonance_witness(
        left, right, use_chirality=use_chirality, atom_mapping=mapping
    )
    if witness is not None:
        return True, False
    result = _resonance_match(
        left,
        right,
        use_chirality=use_chirality,
        max_resonance=max_resonance,
        resonance_flags=resonance_flags,
    )
    return result.matched, not result.matched and (result.limit_reached or result.validation_failed)


def _has_electron_rearrangement(mol: Chem.Mol) -> bool:
    """A conservative guard: only localized, non-conjugated conflicts are final."""

    return any(
        bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1 for bond in mol.GetBonds()
    ) or any(
        not _is_metal_atom(atom)
        and (atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0)
        and atom.GetDegree() != 0
        for atom in mol.GetAtoms()
    )


def _compare_electronic_properties(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    coverage: InvariantResult,
    *,
    use_chirality: bool,
    max_resonance: int,
    resonance_flags: Chem.ResonanceFlags,
    ignore_metal_coordination_bonds: bool,
) -> EquivalenceDecision | None:
    """Constrain positive graph evidence; never infer EQ from electron totals."""

    annotations1, annotations2 = _electron_annotations(mol1), _electron_annotations(mol2)
    # Ordinary baseline/reference molecules have no side annotations. Avoid a
    # mapping search and preserve the existing structural result in that case.
    jointly_known = any(
        any(fields[k] is not None for fields in annotations1)
        and any(fields[k] is not None for fields in annotations2)
        for k in (0, 1)
    )
    require_joint_coordination = not ignore_metal_coordination_bonds and any(
        _is_metal_atom(bond.GetBeginAtom()) or _is_metal_atom(bond.GetEndAtom())
        for mol in (mol1, mol2)
        for bond in mol.GetBonds()
    )
    if not jointly_known and not require_joint_coordination:
        coverage.reason = "No jointly known occupancy fields; structural evidence retained."
        return None

    assert isinstance(coverage.mol1, dict) and isinstance(coverage.mol2, dict)
    graphs = []
    for source in (mol1, mol2):
        graph = Chem.RWMol(source)
        # Retain metals for annotation mapping, not their coordination edges:
        # the existing structural contract compares only their state multiset.
        for bond in list(graph.GetBonds()):
            if _is_metal_atom(bond.GetBeginAtom()) or _is_metal_atom(bond.GetEndAtom()):
                graph.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        graphs.append(_normalize_nonmetal_octet(graph.GetMol()))
    left, right = graphs
    # Normalizers retain atom order and annotations, including added explicit H.
    annotations1, annotations2 = _electron_annotations(left), _electron_annotations(right)
    incomplete = False
    conflicts: list[tuple[int, int, str, object, object]] = []
    comparable_max = 0
    try:
        # In strict mode constrain mappings by coordination connectivity before
        # checking the organic/electronic witness under that SAME mapping.
        mappings = _electronic_property_mappings(
            mol1 if require_joint_coordination else left,
            mol2 if require_joint_coordination else right,
        )
        limit_reached = _ELECTRON_PROPERTY_MAPPING_LIMIT <= 0
        for index, mapping in enumerate(mappings):
            limit_reached = index + 1 >= _ELECTRON_PROPERTY_MAPPING_LIMIT
            for side in (coverage.mol1, coverage.mol2):
                side["mappings_examined"] = index + 1
                side["mapping_limit_reached"] = limit_reached
            if not ignore_metal_coordination_bonds and not _metal_bonds_follow_mapping(
                mol1, mol2, mapping
            ):
                continue
            matched, bounded = _mapped_electronic_structure_match(
                left,
                right,
                mapping,
                use_chirality=use_chirality,
                max_resonance=max_resonance,
                resonance_flags=resonance_flags,
            )
            incomplete |= bounded
            if not matched:
                continue
            mapped_conflicts = []
            comparable = 0
            for idx1, idx2 in enumerate(mapping):
                fields1, fields2 = annotations1[idx1], annotations2[idx2]
                atom1, atom2 = left.GetAtomWithIdx(idx1), right.GetAtomWithIdx(idx2)
                for k in (0, 1):
                    value1, value2 = fields1[k], fields2[k]
                    # A deferred center has no selected singlet/triplet
                    # occupancy. Its zero LP/spin placeholders cannot veto EQ.
                    if (
                        value1 is None
                        or value2 is None
                        or (k == 1 and (fields1[2] is True or fields2[2] is True))
                    ):
                        continue
                    comparable += 1
                    if value1 == value2:
                        continue
                    if k == 1:
                        # These are active pairs, NOT a full Lewis-pair census.
                        # Equal labels need no transport. A changed label must
                        # be balanced by the mapped, independently proved
                        # bond/charge/radical rearrangement.
                        electron1 = (
                            2 * value1
                            + atom1.GetFormalCharge()
                            + _bond_order_sum(atom1)
                            + atom1.GetNumRadicalElectrons()
                        )
                        electron2 = (
                            2 * value2
                            + atom2.GetFormalCharge()
                            + _bond_order_sum(atom2)
                            + atom2.GetNumRadicalElectrons()
                        )
                        if abs(electron1 - electron2) < 1e-8:
                            continue
                    mapped_conflicts.append(
                        (idx1, idx2, _ELECTRON_PROPERTY_NAMES[k], value1, value2)
                    )
            comparable_max = max(comparable_max, comparable)
            if not mapped_conflicts:
                for side in (coverage.mol1, coverage.mol2):
                    side["comparable_fields"] = comparable
                coverage.status = (
                    InvariantStatus.PASSED if comparable else InvariantStatus.NOT_EVALUATED
                )
                coverage.reason = "Compatible atom-mapped known information; unknowns retained."
                return None
            conflicts = mapped_conflicts
        incomplete |= limit_reached
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        coverage.reason = f"Electronic comparison did not complete: {type(exc).__name__}: {exc}"
        return EquivalenceDecision.INCONCLUSIVE
    for side in (coverage.mol1, coverage.mol2):
        side["comparable_fields"] = comparable_max
    if (
        conflicts
        and not incomplete
        and not _has_electron_rearrangement(left)
        and not _has_electron_rearrangement(right)
    ):
        coverage.status = InvariantStatus.FAILED
        coverage.reason = (
            f"Exhaustive mappings have irreconcilable localized annotations: {conflicts}"
        )
        return EquivalenceDecision.NOT_EQUIVALENT
    coverage.reason = (
        "Electronic mapping/resonance search reached its limit."
        if incomplete
        else f"No annotation-compatible mapped structural proof; unresolved conflicts: {conflicts}"
    )
    return EquivalenceDecision.INCONCLUSIVE


def evaluate_equivalence(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    use_chirality: bool = True,
    max_resonance: int = 50,
    resonance_flags: Chem.ResonanceFlags = Chem.ResonanceFlags.UNCONSTRAINED_CATIONS
    | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS,
    ignore_metal_coordination_bonds: bool = True,
) -> EquivalenceResult:
    """Evaluate molecular equivalence and retain the proof and invariant state.

    The result separates the comparison decision from the relation that supplied
    positive evidence. InChIKey agreement is diagnostic and, without an accepted
    structural path, remains inconclusive regardless of which input form matched.
    Component electron mismatches remain diagnostic evidence; accepted comparison
    pathways still determine the decision. Metal coordination connectivity is
    checked when ``ignore_metal_coordination_bonds`` is disabled.
    """

    # Capture the source molecules before any graph normalization.  In
    # particular, _standardize_metal_bonds intentionally creates bookkeeping
    # radicals for metal double bonds; those generated labels must never be
    # presented as the source molecule's physical open-shell state.
    raw_mol1 = _raw_molecule_diagnostics(mol1)
    raw_mol2 = _raw_molecule_diagnostics(mol2)

    # Discard consumer tracking labels only at the public comparison boundary.
    # Later mapping witnesses deliberately introduce their own atom maps to
    # bind graph and electron evidence to the same symmetry correspondence.
    # Those internal maps must survive the canonical/resonance helpers.
    mol1, mol2 = Chem.Mol(mol1), Chem.Mol(mol2)
    for mol in (mol1, mol2):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            if not _is_metal_atom(atom):
                assigned = _read_electron_property(atom, "MOLGR_UNPAIRED_ELECTRON_COUNT")
                if assigned is not None:
                    # SDF/RDKit perception may have changed the native flag after
                    # conversion. Decode the explicit source assignment once at
                    # entry; working transformations may then move electrons.
                    atom.SetNumRadicalElectrons(assigned)

    standardized_1 = _standardize_metal_bonds(mol1)
    standardized_2 = _standardize_metal_bonds(mol2)
    normalization_electronic_effects = {
        "mol1": {
            "raw_formal_charge": raw_mol1.formal_charge,
            "standardized_formal_charge": _total_formal_charge(standardized_1),
            "raw_rdkit_radical_electrons": raw_mol1.rdkit_radical_electrons,
            "standardized_rdkit_radical_electrons": _total_radical_electrons(standardized_1),
        },
        "mol2": {
            "raw_formal_charge": raw_mol2.formal_charge,
            "standardized_formal_charge": _total_formal_charge(standardized_2),
            "raw_rdkit_radical_electrons": raw_mol2.rdkit_radical_electrons,
            "standardized_rdkit_radical_electrons": _total_radical_electrons(standardized_2),
        },
    }
    normalization_electronic_effects["candidate"] = normalization_electronic_effects["mol1"]
    normalization_electronic_effects["reference"] = normalization_electronic_effects["mol2"]

    prepared_organic_1, sanitized_1 = _prepare_organic_mol_with_status(
        standardized_1, already_standardized=True
    )
    prepared_organic_2, sanitized_2 = _prepare_organic_mol_with_status(
        standardized_2, already_standardized=True
    )
    organic_1 = _normalize_nonmetal_octet(prepared_organic_1, already_sanitized=sanitized_1)
    organic_2 = _normalize_nonmetal_octet(prepared_organic_2, already_sanitized=sanitized_2)

    formal_charge_1 = _total_formal_charge(organic_1)
    formal_charge_2 = _total_formal_charge(organic_2)
    radical_electrons_1 = _total_radical_electrons(organic_1)
    radical_electrons_2 = _total_radical_electrons(organic_2)
    num_atoms_1 = organic_1.GetNumAtoms()
    num_atoms_2 = organic_2.GetNumAtoms()
    heavy_atom_formula_1 = _formula_key(organic_1, include_hydrogen=False)
    heavy_atom_formula_2 = _formula_key(organic_2, include_hydrogen=False)
    explicit_h_formula_1 = _formula_key(organic_1, include_hydrogen=True)
    explicit_h_formula_2 = _formula_key(organic_2, include_hydrogen=True)

    checks = EquivalenceChecks(
        formal_charge=PropertyCheck(
            formal_charge_1,
            formal_charge_2,
            formal_charge_1 == formal_charge_2,
        ),
        radical_electrons=PropertyCheck(
            radical_electrons_1,
            radical_electrons_2,
            radical_electrons_1 == radical_electrons_2,
        ),
        num_atoms=PropertyCheck(
            num_atoms_1,
            num_atoms_2,
            num_atoms_1 == num_atoms_2,
        ),
        heavy_atom_formula=PropertyCheck(
            heavy_atom_formula_1,
            heavy_atom_formula_2,
            heavy_atom_formula_1 == heavy_atom_formula_2,
        ),
        explicit_h_formula=PropertyCheck(
            explicit_h_formula_1,
            explicit_h_formula_2,
            explicit_h_formula_1 == explicit_h_formula_2,
        ),
    )
    invariants = {
        "raw_total_formal_charge": InvariantResult(
            InvariantStatus.PASSED
            if raw_mol1.formal_charge == raw_mol2.formal_charge
            else InvariantStatus.FAILED,
            raw_mol1.formal_charge,
            raw_mol2.formal_charge,
        ),
        "heavy_atom_formula": InvariantResult(
            InvariantStatus.PASSED if checks.heavy_atom_formula.passed else InvariantStatus.FAILED,
            checks.heavy_atom_formula.mol1,
            checks.heavy_atom_formula.mol2,
        ),
        "explicit_h_formula": InvariantResult(
            InvariantStatus.PASSED if checks.explicit_h_formula.passed else InvariantStatus.FAILED,
            checks.explicit_h_formula.mol1,
            checks.explicit_h_formula.mol2,
        ),
        "atom_count": InvariantResult(
            InvariantStatus.PASSED if checks.num_atoms.passed else InvariantStatus.FAILED,
            checks.num_atoms.mol1,
            checks.num_atoms.mol2,
        ),
        "formal_charge": InvariantResult(
            InvariantStatus.PASSED if checks.formal_charge.passed else InvariantStatus.FAILED,
            checks.formal_charge.mol1,
            checks.formal_charge.mol2,
        ),
        "radical_electrons": InvariantResult(
            InvariantStatus.PASSED if checks.radical_electrons.passed else InvariantStatus.FAILED,
            checks.radical_electrons.mol1,
            checks.radical_electrons.mol2,
        ),
        "metal_state": InvariantResult(InvariantStatus.NOT_EVALUATED),
        "metal_coordination": InvariantResult(
            InvariantStatus.NOT_EVALUATED,
            reason="Metal coordination bonds are ignored by configuration.",
        ),
        "nonmetal_connectivity": InvariantResult(InvariantStatus.NOT_EVALUATED),
        "component_electrons": InvariantResult(InvariantStatus.NOT_EVALUATED),
        "stereochemistry": InvariantResult(InvariantStatus.NOT_EVALUATED),
        "electronic_properties": InvariantResult(
            InvariantStatus.NOT_EVALUATED,
            _electron_coverage(mol1),
            _electron_coverage(mol2),
            "Known source annotations retained; comparison not yet required.",
        ),
    }
    info = EquivalenceResult(
        checks=checks,
        invariants=invariants,
        contradictions=[],
        bounded_search=BoundedSearchMetadata(limit=max_resonance),
        raw_mol1=raw_mol1,
        raw_mol2=raw_mol2,
        raw_diagnostics={
            "mol1": raw_mol1,
            "mol2": raw_mol2,
            # Positional aliases are intentional: callers that label the
            # inputs Candidate/Reference can consume the same snapshot without
            # a second normalization pass.
            "candidate": raw_mol1,
            "reference": raw_mol2,
        },
        normalization_electronic_effects=normalization_electronic_effects,
    )

    resonance_error: str | None = None

    def finish(
        decision: EquivalenceDecision,
        reason: str,
        *,
        relation: EquivalenceRelation = EquivalenceRelation.NONE,
        method: EquivalenceMethod | None = None,
    ) -> EquivalenceResult:
        if decision == EquivalenceDecision.EQUIVALENT:
            electronic = invariants["electronic_properties"]
            electronic_decision = _compare_electronic_properties(
                standardized_1,
                standardized_2,
                electronic,
                use_chirality=use_chirality,
                max_resonance=max_resonance,
                resonance_flags=resonance_flags,
                ignore_metal_coordination_bonds=ignore_metal_coordination_bonds,
            )
            if electronic_decision is not None:
                decision = electronic_decision
                reason = electronic.reason
                relation = EquivalenceRelation.NONE
                method = None
        info.decision = decision
        info.relation = relation
        info.equivalent = decision == EquivalenceDecision.EQUIVALENT
        info.method = method
        info.reason = reason
        if resonance_error is not None:
            info.reason += f" Resonance search failed: {resonance_error}."
        return info

    if not checks.heavy_atom_formula.passed:
        return finish(
            EquivalenceDecision.NOT_EQUIVALENT,
            "Not equivalent: heavy-atom element counts differ.",
        )

    metal_signature_1 = _metal_signature_key(standardized_1)
    metal_signature_2 = _metal_signature_key(standardized_2)
    invariants["metal_state"] = InvariantResult(
        InvariantStatus.PASSED
        if metal_signature_1 == metal_signature_2
        else InvariantStatus.FAILED,
        metal_signature_1,
        metal_signature_2,
    )
    if metal_signature_1 != metal_signature_2:
        return finish(
            EquivalenceDecision.NOT_EQUIVALENT,
            "Not equivalent: metal valence assignment differs.",
        )

    if not checks.explicit_h_formula.passed:
        return finish(
            EquivalenceDecision.NOT_EQUIVALENT,
            "Not equivalent: explicit-hydrogen element counts differ.",
        )

    if not checks.num_atoms.passed:
        return finish(
            EquivalenceDecision.NOT_EQUIVALENT,
            "Not equivalent: explicit-hydrogen atom counts differ.",
        )

    if raw_mol1.formal_charge != raw_mol2.formal_charge:
        return finish(
            EquivalenceDecision.NOT_EQUIVALENT,
            "Not equivalent: raw whole-molecule total formal charges differ.",
        )

    try:
        if not ignore_metal_coordination_bonds:
            coordination_1 = _resonance_topology_key(
                standardized_1, preserve_metal_formal_charge=True
            )
            coordination_2 = _resonance_topology_key(
                standardized_2, preserve_metal_formal_charge=True
            )
            coordination_matches = coordination_1 == coordination_2
            invariants["metal_coordination"] = InvariantResult(
                InvariantStatus.PASSED if coordination_matches else InvariantStatus.FAILED,
                coordination_1,
                coordination_2,
            )
            if not coordination_matches:
                return finish(
                    EquivalenceDecision.NOT_EQUIVALENT,
                    "Not equivalent: metal coordination connectivity differs.",
                )

        smiles_1 = _canon_smiles(organic_1, use_chirality)
        smiles_2 = _canon_smiles(organic_2, use_chirality)
        info.canonical_smiles = CanonicalSmilesDetail(smiles_1, smiles_2, use_chirality)
        if smiles_1 == smiles_2:
            # Identity entails these invariants. Do not construct weaker
            # topology/component representations to prove them again.
            for name in ("nonmetal_connectivity", "component_electrons", "stereochemistry"):
                invariants[name] = InvariantResult(
                    InvariantStatus.PASSED,
                    reason="Implied by normalized graph identity; separate keys not computed.",
                )
            return finish(
                EquivalenceDecision.EQUIVALENT,
                "Equivalent: canonical SMILES are identical after standardization.",
                relation=EquivalenceRelation.NORMALIZED_GRAPH_IDENTITY,
                method=EquivalenceMethod.IDEAL,
            )
        topology_1 = _resonance_topology_key(organic_1)
        topology_2 = _resonance_topology_key(organic_2)
        topology_matches = topology_1 == topology_2
        invariants["nonmetal_connectivity"] = InvariantResult(
            InvariantStatus.PASSED if topology_matches else InvariantStatus.FAILED,
            topology_1,
            topology_2,
        )
        if not topology_matches:
            return finish(
                EquivalenceDecision.NOT_EQUIVALENT,
                "Not equivalent: non-metal connectivity differs.",
            )

        # Preserve this representation-sensitive comparison as structured
        # evidence, but do not let it terminate the accepted-path evaluation.
        component_signature_1 = _component_electron_signature(organic_1, topology_key=topology_1)
        component_signature_2 = _component_electron_signature(organic_2, topology_key=topology_2)
        component_electrons_match = component_signature_1 == component_signature_2
        invariants["component_electrons"] = InvariantResult(
            InvariantStatus.PASSED if component_electrons_match else InvariantStatus.FAILED,
            component_signature_1,
            component_signature_2,
        )

        stereo_matches = True
        if use_chirality and smiles_1 != smiles_2:
            achiral_smiles_match = _canon_smiles(organic_1, False) == _canon_smiles(
                organic_2, False
            )
            stereo_matches = not achiral_smiles_match
        invariants["stereochemistry"] = InvariantResult(
            InvariantStatus.PASSED if stereo_matches else InvariantStatus.FAILED,
            smiles_1,
            smiles_2,
        )
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        return finish(
            EquivalenceDecision.INCONCLUSIVE,
            f"Inconclusive: canonical comparison failed: {type(exc).__name__}: {exc}",
        )

    identifier_match: bool | None = None

    def identifier_matches() -> bool:
        nonlocal identifier_match
        if identifier_match is None:
            # InChIKey is diagnostic evidence, not a structural proof. Compute
            # it only for conflicts or after all accepted paths have failed.
            prepared_inchi_source_1 = prepared_organic_1
            prepared_inchi_source_2 = prepared_organic_2
            if not any(_is_metal_atom(atom) for atom in mol1.GetAtoms()) and not any(
                _is_metal_atom(atom) for atom in mol2.GetAtoms()
            ):
                prepared_inchi_source_1 = mol1
                prepared_inchi_source_2 = mol2
            prepared_inchi_key_1 = _inchi_key(prepared_inchi_source_1, use_chirality=use_chirality)
            prepared_inchi_key_2 = _inchi_key(prepared_inchi_source_2, use_chirality=use_chirality)
            normalized_inchi_key_1 = _inchi_key(organic_1, use_chirality=use_chirality)
            normalized_inchi_key_2 = _inchi_key(organic_2, use_chirality=use_chirality)
            identifier_match = (
                prepared_inchi_key_1 is not None and prepared_inchi_key_1 == prepared_inchi_key_2
            ) or (
                normalized_inchi_key_1 is not None
                and normalized_inchi_key_1 == normalized_inchi_key_2
            )
        return identifier_match

    carbene_match: bool | None = None

    def carbene_matches() -> bool:
        nonlocal carbene_match
        if carbene_match is None:
            try:
                normalized_1 = _carbene_zwitterion_normalized_smiles(
                    organic_1, use_chirality=use_chirality
                )
                normalized_2 = _carbene_zwitterion_normalized_smiles(
                    organic_2, use_chirality=use_chirality
                )
                info.carbene_zwitterion = CarbeneZwitterionDetail(
                    mol1_normalized=normalized_1,
                    mol2_normalized=normalized_2,
                )
                carbene_match = normalized_1 == normalized_2
            except TimeoutError:
                raise
            except Exception:  # noqa: BLE001
                carbene_match = False
        return carbene_match

    if not component_electrons_match:
        if identifier_matches():
            info.contradictions.append("identifier_match_despite_component_electron_mismatch")
        if carbene_matches():
            info.contradictions.append(
                "carbene_normalization_match_despite_component_electron_mismatch"
            )

    if use_chirality and not stereo_matches:
        if identifier_matches():
            info.contradictions.append("identifier_match_despite_stereochemistry_mismatch")
        return finish(
            EquivalenceDecision.NOT_EQUIVALENT,
            "Not equivalent: stereochemistry differs.",
        )

    if carbene_matches():
        return finish(
            EquivalenceDecision.EQUIVALENT,
            "Equivalent: carbene/zwitterion normalization matched.",
            relation=EquivalenceRelation.CARBENE_ZWITTERION_EQUIVALENCE,
            method=EquivalenceMethod.CARBENE_ZWITTERION,
        )

    # Matching topology and component electron totals can identify an ambiguous
    # open-shell representation, but do not prove resonance reachability. Keep
    # the heuristic as structured evidence and require the bounded matcher or
    # an explicit local witness before returning EQUIVALENT.
    open_shell_predicates_match = (
        component_electrons_match
        and checks.formal_charge.passed
        and checks.radical_electrons.passed
        and radical_electrons_1 > 0
        and (
            not use_chirality
            or (
                not _has_defined_stereochemistry(organic_1)
                and not _has_defined_stereochemistry(organic_2)
            )
        )
    )
    if open_shell_predicates_match:
        info.open_shell_heuristic_triggered = _open_shell_heuristic_applies(
            organic_1,
            organic_2,
            use_chirality=use_chirality,
        )

    assert info.bounded_search is not None
    info.bounded_search.attempted = True
    try:
        resonance_result = _resonance_match(
            organic_1,
            organic_2,
            use_chirality=use_chirality,
            max_resonance=max_resonance,
            resonance_flags=resonance_flags,
        )
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        resonance_error = f"{type(exc).__name__}: {exc}"
        resonance_result = _ResonanceMatchResult(False, 0, 0, None)
    resonance_matched, mol1_count, mol2_count, hit_smiles = resonance_result
    info.bounded_search.mol1_count = mol1_count
    info.bounded_search.mol2_count = mol2_count
    info.bounded_search.limit_reached = resonance_result.limit_reached
    info.bounded_search.supplier_limit_reached = resonance_result.supplier_limit_reached
    info.bounded_search.local_limit_reached = resonance_result.local_limit_reached
    info.bounded_search.validation_failed = resonance_result.validation_failed
    info.bounded_search.exhaustive = resonance_error is None and resonance_result.exhaustive
    info.bounded_search.stop_reason = (
        "error" if resonance_error is not None else resonance_result.stop_reason
    )
    if resonance_result.regional_witness is not None:
        regional = resonance_result.regional_witness
        info.bounded_search.regional_region_count = regional.region_count
        info.bounded_search.regional_state_count = regional.state_count
        info.bounded_search.regional_pruned_moves = regional.distance_pruned
    if resonance_matched:
        info.resonance = ResonanceDetail(
            max_resonance=max_resonance,
            resonance_flags=int(resonance_flags),
            mol1_resonance_count=mol1_count,
            mol2_resonance_count=mol2_count,
            hit_smiles=hit_smiles,
        )
        return finish(
            EquivalenceDecision.EQUIVALENT,
            "Equivalent: resonance normalization matched.",
            relation=EquivalenceRelation.RESONANCE_EQUIVALENCE,
            method=EquivalenceMethod.RESONANCE,
        )

    elementary_open_shell_witness, mapping_search = _elementary_open_shell_resonance_witness(
        organic_1,
        organic_2,
        use_chirality=use_chirality,
    )
    info.open_shell_three_center_mapping_search = mapping_search
    if elementary_open_shell_witness is not None:
        info.open_shell_three_center = elementary_open_shell_witness
        return finish(
            EquivalenceDecision.EQUIVALENT,
            "Equivalent: explicit elementary three-center open-shell electron shift matched.",
            relation=EquivalenceRelation.RESONANCE_EQUIVALENCE,
            method=EquivalenceMethod.OPEN_SHELL_THREE_CENTER,
        )

    if resonance_error is not None:
        return finish(
            EquivalenceDecision.INCONCLUSIVE,
            "Inconclusive: resonance search did not complete.",
        )

    if mapping_search.limit_reached:
        return finish(
            EquivalenceDecision.INCONCLUSIVE,
            "Inconclusive: elementary open-shell topology mapping search reached its limit.",
        )

    # Standard InChI can normalize distinct Lewis and proton placements. Neither
    # key source alone proves a graph/electron-equivalence pathway.
    if identifier_matches():
        return finish(
            EquivalenceDecision.INCONCLUSIVE,
            "Inconclusive: InChIKey agreement has no independent structural witness.",
            relation=EquivalenceRelation.IDENTIFIER_EQUIVALENCE,
            method=EquivalenceMethod.INCHI_KEY,
        )

    if info.open_shell_heuristic_triggered:
        return finish(
            EquivalenceDecision.INCONCLUSIVE,
            "Inconclusive: open-shell topology and component electron counts match, "
            "but no explicit resonance witness was found.",
        )

    if info.bounded_search.limit_reached:
        return finish(
            EquivalenceDecision.INCONCLUSIVE,
            "Inconclusive: bounded resonance search reached its configured limit.",
        )

    if info.bounded_search.validation_failed:
        return finish(
            EquivalenceDecision.INCONCLUSIVE,
            "Inconclusive: resonance candidates failed validation; search is incomplete.",
        )

    return finish(
        EquivalenceDecision.NOT_EQUIVALENT,
        "Not equivalent: no standardized comparison path matched.",
    )


def check_equivalence(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    use_chirality: bool = True,
    max_resonance: int = 50,
    resonance_flags: Chem.ResonanceFlags = Chem.ResonanceFlags.UNCONSTRAINED_CATIONS
    | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS,
    ignore_metal_coordination_bonds: bool = True,
) -> Tuple[bool, EquivalenceInfo]:
    with rdBase.BlockLogs():
        result = evaluate_equivalence(
            mol1,
            mol2,
            use_chirality=use_chirality,
            max_resonance=max_resonance,
            resonance_flags=resonance_flags,
            ignore_metal_coordination_bonds=ignore_metal_coordination_bonds,
        )
    return result.decision == EquivalenceDecision.EQUIVALENT, result
