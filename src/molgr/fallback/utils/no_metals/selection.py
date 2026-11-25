"""Selection helpers for no-metal fallback reconstruction."""

from __future__ import annotations

from dataclasses import astuple
from typing import Optional, Sequence, cast

from openbabel import openbabel as ob

from molgr.config import CONFIG, MolGRConfig
from molgr.fallback.state import ReconstructionState
from molgr.fallback.utils.bond_geometry import (
    azinium_geometry_conflict_count,
    is_elongated_carbonyl_pair,
)
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import mol_atoms, mol_bonds
from molgr.fallback.utils.organic_topology import (
    OrganicTopologyMetrics,
    compute_organic_topology_metrics,
)


def _score_reconstruction_candidate(
    candidate: ReconstructionState,
    *,
    config: MolGRConfig | None = None,
) -> float:
    if config is None:
        return candidate.full_score()
    return candidate.full_score(config=config)


def _annotate_no_metal_candidate_topology(
    candidate: ReconstructionState,
    *,
    config: MolGRConfig | None = None,
) -> OrganicTopologyMetrics:
    resolved_config = CONFIG if config is None else config
    topology_config = resolved_config.organic_topology
    metrics = candidate.get_cached_omol_value(
        f"organic_topology_metrics:{astuple(topology_config)!r}",
        lambda omol: compute_organic_topology_metrics(
            omol,
            topology_config,
        ),
    )
    candidate.metadata["organic_aromatic_atom_count"] = metrics.aromatic_atom_count
    candidate.metadata["organic_aromatic_ring_count"] = metrics.aromatic_ring_count
    candidate.metadata["organic_aromatic_stability_score"] = metrics.aromatic_stability_score
    candidate.metadata["organic_conjugated_atom_count"] = metrics.conjugated_atom_count
    candidate.metadata["organic_conjugated_bond_count"] = metrics.conjugated_bond_count
    candidate.metadata["organic_max_conjugated_component_size"] = (
        metrics.max_conjugated_component_size
    )
    candidate.metadata["organic_hyperconjugative_donor_count"] = (
        metrics.hyperconjugative_donor_count
    )
    candidate.metadata["organic_hyperconjugation_score"] = metrics.hyperconjugation_score
    return metrics


def _formal_charge_absolute_sum(candidate: ReconstructionState) -> int:
    """Measure charge separation on the ligand, counting both ends of each formal ion pair."""
    return int(
        candidate.get_cached_omol_value(
            "organic_formal_charge_absolute_sum",
            lambda omol: sum(
                abs(int(atom_iter.GetFormalCharge()))
                for atom_iter in mol_atoms(cast(ob.OBMol, omol.OBMol))
            ),
        )
    )


def _excess_radical_labels(candidate: ReconstructionState) -> int:
    """Count explicit unpaired electrons above the requested ligand radical budget."""
    radical_sum = int(
        candidate.get_cached_omol_value(
            "organic_radical_label_sum",
            lambda omol: sum(
                get_unpaired_electron_count(atom_iter)
                for atom_iter in mol_atoms(cast(ob.OBMol, omol.OBMol))
            ),
        )
    )
    excess = max(0, radical_sum - candidate.total_radical_electrons)
    candidate.metadata["organic_radical_label_sum"] = radical_sum
    candidate.metadata["organic_excess_radical_labels"] = excess
    return excess


def _no_metal_candidate_selection_key(
    candidate: ReconstructionState,
    *,
    config: MolGRConfig | None = None,
) -> tuple[int, int, int, int, int, float, float, float, float, int, int, float]:
    """Rank azinium geometry, charge separation and stretched carbonyls before conjugation and energy."""
    metrics = _annotate_no_metal_candidate_topology(candidate, config=config)
    score = float(candidate.metadata.get("score", float("inf")))
    formal_charge_absolute_sum = _formal_charge_absolute_sum(candidate)
    conjugation_charge_penalty = formal_charge_absolute_sum / 2.0
    adjusted_max_conjugated_component_size = (
        metrics.max_conjugated_component_size - conjugation_charge_penalty
    )
    adjusted_conjugated_atom_count = metrics.conjugated_atom_count - conjugation_charge_penalty
    adjusted_conjugated_bond_count = metrics.conjugated_bond_count - conjugation_charge_penalty
    excess_radical_labels = _excess_radical_labels(candidate)
    candidate.metadata["organic_formal_charge_absolute_sum"] = formal_charge_absolute_sum
    candidate.metadata["organic_conjugation_charge_penalty"] = conjugation_charge_penalty
    candidate.metadata["organic_adjusted_max_conjugated_component_size"] = (
        adjusted_max_conjugated_component_size
    )
    candidate.metadata["organic_adjusted_conjugated_atom_count"] = adjusted_conjugated_atom_count
    candidate.metadata["organic_adjusted_conjugated_bond_count"] = adjusted_conjugated_bond_count
    long_carbonyl_count = sum(
        bond.GetBondOrder() == 2
        and is_elongated_carbonyl_pair(bond.GetBeginAtom(), bond.GetEndAtom())
        for bond in mol_bonds(cast(ob.OBMol, candidate.omol.OBMol))
    )
    candidate.metadata["organic_long_carbonyl_multiple_bond_count"] = long_carbonyl_count
    azinium_count = azinium_geometry_conflict_count(candidate.omol.OBMol)
    candidate.metadata["organic_azinium_geometry_conflict_count"] = azinium_count
    selection_key = (
        azinium_count,
        formal_charge_absolute_sum,
        long_carbonyl_count,
        -metrics.aromatic_atom_count,
        -metrics.aromatic_ring_count,
        -metrics.aromatic_stability_score,
        -adjusted_max_conjugated_component_size,
        -adjusted_conjugated_atom_count,
        -adjusted_conjugated_bond_count,
        excess_radical_labels,
        -metrics.hyperconjugation_score,
        score,
    )
    candidate.metadata["organic_topology_selection_key"] = selection_key
    return selection_key


def _no_metal_candidate_graph_tie_break_key(
    candidate: ReconstructionState,
) -> tuple[int, ...]:
    """Return a backend-independent key for exact no-metal score ties.

    The key deliberately uses atom indices, explicit electron labels, and the
    sorted bond table only. It does not use Open Babel aromaticity flags,
    iterator order, coordinates, or resonance indices.
    """

    obmol = cast(ob.OBMol, candidate.omol.OBMol)
    atoms = tuple(
        value
        for atom in mol_atoms(obmol)
        for value in (
            int(atom.GetIdx()),
            int(atom.GetAtomicNum()),
            int(atom.GetFormalCharge()),
            int(get_unpaired_electron_count(atom)),
            int(get_lone_pair_count(atom)),
            int(has_unresolved_two_electron_center(atom)),
        )
    )
    bonds = tuple(
        value
        for begin_idx, end_idx, order in sorted(
            (
                min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                int(bond.GetBondOrder()),
            )
            for bond in mol_bonds(obmol)
        )
        for value in (begin_idx, end_idx, order)
    )
    return (int(obmol.NumAtoms()), *atoms, len(bonds) // 3, *bonds)


def select_best_no_metal_candidate(
    candidates: Sequence[ReconstructionState],
    *,
    config: MolGRConfig | None = None,
) -> Optional[ReconstructionState]:
    """Return the lowest-ranked candidate using chemistry first and graph ties last."""

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            _no_metal_candidate_selection_key(candidate, config=config),
            _no_metal_candidate_graph_tie_break_key(candidate),
        ),
    )


__all__ = [
    "_annotate_no_metal_candidate_topology",
    "_no_metal_candidate_graph_tie_break_key",
    "_no_metal_candidate_selection_key",
    "_score_reconstruction_candidate",
    "select_best_no_metal_candidate",
]
