"""Native equivalence: Python prepares inputs and packages native evidence.

All mapping, normalization, stereo/electron comparison and resonance search run
in C++. No supplier, SMILES or InChI fallback is used. The previous evaluator
is retained in _legacy_equivalence for comparison only.
"""

from __future__ import annotations

from collections import Counter

from rdkit import Chem, rdBase

from molgr import _core
from molgr.fallback.utils.consts import NON_METAL_DICT
from molgr.utils._equivalence_types import (
    UNKNOWN_ELECTRON_METADATA,
    BoundedSearchMetadata,
    CanonicalSmilesDetail,
    CarbeneZwitterionDetail,
    EquivalenceChecks,
    EquivalenceDecision,
    EquivalenceInfo,
    EquivalenceMethod,
    EquivalenceRelation,
    EquivalenceResult,
    InvariantResult,
    InvariantStatus,
    OpenShellThreeCenterDetail,
    PropertyCheck,
    RawMoleculeDiagnostics,
    ResonanceDetail,
    TopologyMappingSearchMetadata,
)
from molgr.utils.converter import _convert_rdkit_graph


__all__ = [
    "UNKNOWN_ELECTRON_METADATA",
    "BoundedSearchMetadata",
    "CanonicalSmilesDetail",
    "CarbeneZwitterionDetail",
    "EquivalenceChecks",
    "EquivalenceDecision",
    "EquivalenceInfo",
    "EquivalenceMethod",
    "EquivalenceRelation",
    "EquivalenceResult",
    "InvariantResult",
    "InvariantStatus",
    "OpenShellThreeCenterDetail",
    "PropertyCheck",
    "RawMoleculeDiagnostics",
    "ResonanceDetail",
    "TopologyMappingSearchMetadata",
    "evaluate_equivalence",
    "check_equivalence",
]

_ELECTRON_PROPERTY_MAPPING_LIMIT = 10000
_FIELDS = (
    "MOLGR_METAL_UNPAIRED_ELECTRONS",
    "MOLGR_LONE_PAIR_COUNT",
    "MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER",
    "MOLGR_UNPAIRED_ELECTRON_COUNT",
)


def _read_field(atom, name, boolean=False):
    """Decode a property whose presence has already been checked."""
    try:
        value = atom.GetProp(name).strip().lower()
        if boolean:
            return {"true": True, "1": True, "false": False, "0": False}.get(value)
        count = int(value)
        return count if count >= 0 else None
    except (RuntimeError, TypeError, ValueError):
        return None


def _prepare_input(mol, use_chirality, ignore_metal_coordination_bonds=True):
    """Validate atoms and capture diagnostics/transfer rows in one source scan."""
    # AddHs creates the private graph. Only a changed radical assignment needs
    # an earlier copy; decoded MolGR annotations go directly into native rows.
    prepared = mol
    counts = Counter()
    explicit_h = total_h = charge = radicals = 0
    metals = []
    properties = [[] for _ in _FIELDS]
    atom_rows = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        z, symbol = atom.GetAtomicNum(), atom.GetSymbol()
        if atom.HasQuery() or z == 0:
            raise ValueError("MoleculeData does not support query or dummy atoms")
        metal = z not in NON_METAL_DICT
        atom_charge = atom.GetFormalCharge()
        atom_radicals = atom.GetNumRadicalElectrons()
        charge += atom_charge
        radicals += atom_radicals
        counts[symbol] += 1
        if z == 1:
            explicit_h += 1
            total_h += 1
        else:
            h = atom.GetTotalNumHs(includeNeighbors=False)
            total_h += h
            counts["H"] += h
        if metal:
            metals.append((i, symbol, atom_charge))
        unpaired = 0 if metal else None
        lone_pairs, unresolved, presence = 0, False, 0
        for index, name in enumerate(_FIELDS):
            if (index == 0 and not metal) or not atom.HasProp(name):
                continue
            value = _read_field(atom, name, boolean=index == 2)
            properties[index].append((i, value if value is not None else UNKNOWN_ELECTRON_METADATA))
            if value is None:
                continue
            if index == 0 or (index == 3 and not metal):
                unpaired = value
                presence |= 1
                if not metal and value != atom_radicals:
                    if prepared is mol:
                        prepared = Chem.Mol(mol)
                    prepared.GetAtomWithIdx(i).SetNumRadicalElectrons(value)
            elif index == 1:
                lone_pairs = value
                presence |= 2
            elif index == 2:
                unresolved = value
                presence |= 4
        atom_rows.append(
            (z, atom.GetIsotope(), atom_charge, unpaired, lone_pairs, unresolved, presence)
        )
    keys = [symbol for symbol in ("C", "H") if counts[symbol]]
    keys.extend(symbol for symbol in sorted(counts) if symbol not in keys and counts[symbol])
    formula = "".join(
        symbol + (str(counts[symbol]) if counts[symbol] != 1 else "") for symbol in keys
    )
    raw = RawMoleculeDiagnostics(
        formula=formula,
        hydrogen_count=total_h,
        explicit_hydrogen_count=explicit_h,
        formal_charge=charge,
        rdkit_radical_electrons=radicals,
        metal_formal_state=tuple(sorted(metals)),
        molgr_metal_unpaired_electrons=tuple(properties[0]) or UNKNOWN_ELECTRON_METADATA,
        active_lone_pair_properties=tuple(properties[1]) or UNKNOWN_ELECTRON_METADATA,
        unresolved_two_electron_center_properties=tuple(properties[2]) or UNKNOWN_ELECTRON_METADATA,
        molgr_unpaired_electrons=tuple(properties[3]) or UNKNOWN_ELECTRON_METADATA,
    )
    # Atom maps are not exported. Coordinates do not enter equivalence, and
    # disabled stereo needs neither perception nor extraction.
    return _convert_rdkit_graph(
        prepared,
        perceive_radicals=True,
        include_stereo=use_chirality,
        include_coordinates=False,
        atom_rows=atom_rows,
        ignore_metal_stereo=ignore_metal_coordination_bonds,
    ), raw


def _package(result, raw_left, raw_right, max_resonance):
    left, right = result["left"], result["right"]
    decision = EquivalenceDecision(result["decision"])
    method = EquivalenceMethod(result["method"]) if result["method"] else None
    relations = {
        EquivalenceMethod.IDEAL: EquivalenceRelation.NORMALIZED_GRAPH_IDENTITY,
        EquivalenceMethod.CARBENE_ZWITTERION: EquivalenceRelation.CARBENE_ZWITTERION_EQUIVALENCE,
        EquivalenceMethod.RESONANCE: EquivalenceRelation.RESONANCE_EQUIVALENCE,
    }
    statuses = {
        1: InvariantStatus.PASSED,
        0: InvariantStatus.NOT_EVALUATED,
        -1: InvariantStatus.FAILED,
    }
    invariants = {
        name: InvariantResult(statuses[status]) for name, status in result["invariants"].items()
    }
    for name, key in (
        ("raw_total_formal_charge", "charge"),
        ("heavy_atom_formula", "heavy_formula"),
        ("explicit_h_formula", "hydrogen_count"),
        ("atom_count", "atom_count"),
        ("formal_charge", "organic_charge"),
        ("radical_electrons", "organic_radicals"),
        ("metal_state", "metal_states"),
    ):
        invariants[name].mol1, invariants[name].mol2 = left[key], right[key]

    def coverage(summary):
        return {
            "known_fields": summary["known_fields"],
            "unknown_fields": 3 * summary["atom_count"] - summary["known_fields"],
            "comparable_fields": result["comparable_fields"],
            "deferred_centers": summary["deferred_centers"],
            "mappings_examined": result["mappings"],
            "mapping_limit_reached": result["mapping_limit"],
        }

    electronic = invariants["electronic_properties"]
    electronic.mol1, electronic.mol2 = coverage(left), coverage(right)
    electronic.reason = "Known fields require the same native mapping as the structural witness; missing fields stay unknown."
    checks = EquivalenceChecks(
        formal_charge=PropertyCheck(
            left["organic_charge"],
            right["organic_charge"],
            left["organic_charge"] == right["organic_charge"],
        ),
        radical_electrons=PropertyCheck(
            left["organic_radicals"],
            right["organic_radicals"],
            left["organic_radicals"] == right["organic_radicals"],
        ),
        num_atoms=PropertyCheck(
            left["organic_atoms"],
            right["organic_atoms"],
            left["organic_atoms"] == right["organic_atoms"],
        ),
        heavy_atom_formula=PropertyCheck(
            left["heavy_formula"],
            right["heavy_formula"],
            left["heavy_formula"] == right["heavy_formula"],
        ),
        explicit_h_formula=PropertyCheck(
            left["formula"], right["formula"], left["formula"] == right["formula"]
        ),
    )
    effects = {}
    for name, raw, summary in (("mol1", raw_left, left), ("mol2", raw_right, right)):
        effects[name] = {
            "raw_formal_charge": raw.formal_charge,
            "standardized_formal_charge": summary["charge"],
            "raw_rdkit_radical_electrons": raw.rdkit_radical_electrons,
            "standardized_rdkit_radical_electrons": summary["organic_radicals"],
        }
    effects["candidate"], effects["reference"] = effects["mol1"], effects["mol2"]
    return EquivalenceResult(
        decision=decision,
        relation=relations.get(method, EquivalenceRelation.NONE),
        equivalent=decision == EquivalenceDecision.EQUIVALENT,
        method=method,
        reason=result["reason"],
        invariants=invariants,
        checks=checks,
        raw_mol1=raw_left,
        raw_mol2=raw_right,
        raw_diagnostics={
            "mol1": raw_left,
            "mol2": raw_right,
            "candidate": raw_left,
            "reference": raw_right,
        },
        normalization_electronic_effects=effects,
        bounded_search=BoundedSearchMetadata(
            attempted=result["resonance_attempted"],
            limit=max_resonance,
            mol1_count=result["states_left"],
            mol2_count=result["states_right"],
            limit_reached=result["resonance_limit"] or result["mapping_limit"],
            local_limit_reached=result["resonance_limit"],
            validation_failed=result["validation_failed"],
            exhaustive=False if result["resonance_attempted"] else None,
            stop_reason=result["stop_reason"],
            regional_region_count=result["regions"],
            regional_state_count=result["states_left"] + result["states_right"],
            regional_pruned_moves=result["pruned_moves"],
        ),
        resonance=ResonanceDetail(
            max_resonance=max_resonance,
            mol1_resonance_count=result["states_left"],
            mol2_resonance_count=result["states_right"],
        )
        if method == EquivalenceMethod.RESONANCE
        else None,
        atom_mapping=tuple(result["mapping"]),
        resonance_witness=tuple(result["witness"]),
        native_stats={
            key: result[key]
            for key in (
                "mappings",
                "mapping_nodes",
                "mapping_limit",
                "preparation_ms",
                "mapping_ms",
                "resonance_ms",
                "resonance_cache_hits",
                "region_cache_hits",
                "region_constraint_rejections",
            )
        },
    )


def evaluate_equivalence(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    use_chirality: bool = False,
    max_resonance: int = 5000,
    *,
    ignore_metal_coordination_bonds: bool = True,
    max_mappings: int | None = None,
    max_mapping_nodes: int = 100000,
    max_resonance_steps: int = 24,
) -> EquivalenceResult:
    """Native comparison; unsupported or bounded cases are inconclusive.

    Stereo comparison is disabled by default; set ``use_chirality=True`` to
    require compatible stereochemical information under the accepted mapping.

    ``max_resonance`` permits twice that many visited states across both BFS
    sides, all regions and mappings. ``max_resonance_steps`` bounds each region's
    path. Legacy SMILES diagnostics stay None; native mappings and moves
    replace them.
    """
    try:
        if not 0 <= max_resonance <= 1000000 or not 0 <= max_resonance_steps <= 10000:
            raise ValueError("Invalid native resonance budget")
        left, raw_left = _prepare_input(mol1, use_chirality, ignore_metal_coordination_bonds)
        right, raw_right = _prepare_input(mol2, use_chirality, ignore_metal_coordination_bonds)
        result = _core.equivalence.evaluate(
            left,
            right,
            use_chirality,
            max_resonance,
            ignore_metal_coordination_bonds,
            _ELECTRON_PROPERTY_MAPPING_LIMIT if max_mappings is None else max_mappings,
            max_mapping_nodes,
            max_resonance_steps,
        )
        return _package(result, raw_left, raw_right, max_resonance)
    except TimeoutError:
        raise
    except (ValueError, RuntimeError, TypeError, OverflowError) as exc:
        return EquivalenceResult(
            reason=f"Input preparation/native evaluation could not be completed: {exc}",
            invariants={"electronic_properties": InvariantResult(InvariantStatus.NOT_EVALUATED)},
        )


def check_equivalence(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    use_chirality: bool = False,
    max_resonance: int = 5000,
    *,
    ignore_metal_coordination_bonds: bool = True,
    **native_options,
):
    with rdBase.BlockLogs():
        result = evaluate_equivalence(
            mol1,
            mol2,
            use_chirality=use_chirality,
            max_resonance=max_resonance,
            ignore_metal_coordination_bonds=ignore_metal_coordination_bonds,
            **native_options,
        )
    return result.equivalent, result
