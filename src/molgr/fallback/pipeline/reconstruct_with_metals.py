"""Metal-aware reconstruction orchestration for fallback.

The production path is:
1. Strip metals and enumerate a small set of metal valence/radical assignments.
2. Group assignments by the no-metal target they induce via meet-in-the-middle DP.
3. Reconstruct each no-metal target once, reuse that state across the bucket, and
   score the organic core with fixed UFF scoring.
4. Across no-metal charge states, prefer candidates that preserve aromaticity,
   conjugation, and charge localization on chemically plausible sites.
5. Within that organic electronic-state preference, prefer metal assignments whose
   oxidation states are better supported by the fixed local electrostatics and donor field.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import List, Optional

from openbabel import pybel

from molgr.config import MolGRConfig
from molgr.diagnostics import ReconstructionDiagnosticCollector, ReconstructionFailureCode
from molgr.fallback.state import (
    MetalCandidateState,
    MetalCandidateStateMachine,
    ReconstructionState,
)
from molgr.fallback.utils.metals import preparation, scoring, search
from molgr.fallback.utils.no_metals import preparation as no_metal_preparation
from molgr.process_guard import ensure_current_process

from . import reconstruct_without_metals


def _candidate_matches_global_electronic_state(
    candidate: MetalCandidateState,
    total_charge: int,
    total_radical_electrons: int,
) -> bool:
    """Check that metal and ligand assignments jointly account for input charge and unpaired electrons."""
    return (
        candidate.no_metal_charge_target
        + sum(int(state.valence) for state in candidate.metal_states)
        == total_charge
        and candidate.no_metal_radical_target
        + sum(int(state.radical_num) for state in candidate.metal_states)
        == total_radical_electrons
    )


def xyz2omol_state(
    xyz_block: str,
    total_charge: int = 0,
    total_radical_electrons: int = 0,
    *,
    config: MolGRConfig | None = None,
    _diagnostics: ReconstructionDiagnosticCollector | None = None,
) -> Optional[MetalCandidateState]:
    """Return the best scored metal candidate state for the input XYZ block."""

    ensure_current_process("molgr.fallback.xyz2omol_state")
    diagnostics = _diagnostics
    if diagnostics is not None:
        diagnostics.set("total_charge", int(total_charge))
        diagnostics.set("total_radical_electrons", int(total_radical_electrons))
    try:
        base_state = preparation.prepare_metal_state(
            xyz_block,
            total_charge,
            total_radical_electrons,
            config=config,
        )
    except (OSError, ValueError) as exc:
        if diagnostics is not None:
            diagnostics.fail(
                ReconstructionFailureCode.INVALID_XYZ,
                "input.parse",
                "The XYZ block could not be parsed by Open Babel.",
                cause=exc,
            )
        raise
    if diagnostics is not None:
        diagnostics.set("metal_atom_count", int(base_state.metadata.get("metal_atom_count", 0)))
        diagnostics.set(
            "metal_state_options_per_site",
            [len(options) for options in base_state.available_valence_radical_states],
        )
    state_search_groups = search._build_metal_state_search_groups(
        base_state.available_valence_radical_states,
        config=config,
    )
    layered_state_search_groups = search._build_layered_metal_state_search_groups(
        state_search_groups,
        total_radical_electrons,
        config=config,
    )
    no_metal_seed_omol: Optional[pybel.Molecule] = None
    scored_candidates: List[MetalCandidateState] = []
    winning_layer_index = 0
    for layer_index, available_valence_radical_states in enumerate(layered_state_search_groups):
        if diagnostics is not None:
            diagnostics.count("search_layers")
        grouped_candidates = search._group_candidates_by_target_dp(
            base_state.phase_history,
            available_valence_radical_states,
            total_charge,
            total_radical_electrons,
            config=config,
        )
        if diagnostics is not None:
            diagnostics.count("target_buckets", len(grouped_candidates))
            diagnostics.count(
                "metal_candidates_enumerated",
                sum(len(candidates) for candidates in grouped_candidates.values()),
            )
        if not grouped_candidates:
            if diagnostics is not None:
                diagnostics.count("layers_without_target_buckets")
            continue

        if no_metal_seed_omol is None:
            try:
                no_metal_seed_omol = no_metal_preparation._seed_omol_from_xyz(
                    base_state.no_metal_xyz_block, base_state.metal_sites
                )
            except (OSError, ValueError) as exc:
                if diagnostics is not None:
                    diagnostics.fail(
                        ReconstructionFailureCode.INVALID_XYZ,
                        "no_metal.parse",
                        "The no-metal XYZ seed could not be parsed.",
                        cause=exc,
                    )
                return None

        current_layer_scored_candidates: list[MetalCandidateState] = []
        for candidates in grouped_candidates.values():
            if not candidates:
                continue
            prototype = candidates[0]
            organic_candidates: list[ReconstructionState] = []
            try:
                no_metal_state = reconstruct_without_metals._seed_omol_to_omol_no_metal_state(
                    no_metal_seed_omol,
                    prototype.no_metal_charge_target,
                    prototype.no_metal_radical_target,
                    config=config,
                    candidate_pool=(
                        organic_candidates if base_state.available_valence_radical_states else None
                    ),
                    metal_sites=base_state.metal_sites,
                )
            except (OSError, ValueError):
                if diagnostics is not None:
                    diagnostics.count("no_metal_reconstruction_exceptions")
                continue
            if no_metal_state is None:
                if diagnostics is not None:
                    diagnostics.count("no_metal_reconstruction_none")
                continue
            if diagnostics is not None:
                diagnostics.count("no_metal_reconstruction_successes")

            # Keep ligand alternatives until metal geometry and electronic state can
            # participate in selection; the isolated-ligand winner is insufficient.
            for organic_state in organic_candidates or [no_metal_state]:
                for candidate in candidates:
                    try:
                        scored_candidate = scoring._prepare_candidate_with_no_metal_state(
                            candidate,
                            organic_state,
                            config=config,
                        )
                    except ValueError:
                        if diagnostics is not None:
                            diagnostics.count("metal_candidate_scoring_rejections")
                        continue
                    if scored_candidate.score is None:
                        if diagnostics is not None:
                            diagnostics.count("metal_candidate_missing_scores")
                        continue
                    if diagnostics is not None:
                        diagnostics.count("metal_candidates_scored")
                    current_layer_scored_candidates.append(scored_candidate)
        if not current_layer_scored_candidates:
            continue
        scored_candidates = scoring.select_ligand_alternatives(
            current_layer_scored_candidates, config=config
        )
        winning_layer_index = layer_index
        break

    if not scored_candidates:
        if diagnostics is not None:
            if diagnostics.counts.get("target_buckets", 0) == 0:
                diagnostics.fail(
                    ReconstructionFailureCode.NO_REACHABLE_METAL_STATE,
                    "metal.search",
                    "No metal-state assignment reached a valid charge/radical target.",
                )
            elif diagnostics.counts.get("no_metal_reconstruction_successes", 0) == 0:
                diagnostics.fail(
                    ReconstructionFailureCode.NO_VALID_ORGANIC_CANDIDATE,
                    "no_metal.reconstruction",
                    "Every reachable metal target failed organic graph reconstruction.",
                )
            else:
                diagnostics.fail(
                    ReconstructionFailureCode.ALL_METAL_CANDIDATES_REJECTED,
                    "metal.scoring",
                    "Organic targets were reconstructed, but every metal candidate was rejected.",
                )
        return None

    for scored_candidate in scored_candidates:
        scored_candidate.metadata["search_layer_index"] = winning_layer_index

    best_candidate = scoring.select_best_candidate(scored_candidates, config=config)
    if best_candidate is None:
        if diagnostics is not None:
            diagnostics.fail(
                ReconstructionFailureCode.ALL_METAL_CANDIDATES_REJECTED,
                "metal.selection",
                "No scored metal candidate could be selected.",
            )
        return None
    if not _candidate_matches_global_electronic_state(
        best_candidate,
        total_charge,
        total_radical_electrons,
    ):
        if diagnostics is not None:
            diagnostics.fail(
                ReconstructionFailureCode.OUTPUT_INVARIANT_BROKEN,
                "metal.selection",
                "The selected candidate does not satisfy the requested charge/radical budget.",
                candidate_total_charge=(
                    best_candidate.no_metal_charge_target
                    + sum(int(state.valence) for state in best_candidate.metal_states)
                ),
                candidate_total_radicals=(
                    best_candidate.no_metal_radical_target
                    + sum(int(state.radical_num) for state in best_candidate.metal_states)
                ),
            )
        return None
    if best_candidate.combined_omol is None:
        best_candidate.materialize_combined_omol(preparation.combine_metal_with_omol)
        winner_machine = MetalCandidateStateMachine.from_candidate_state(best_candidate)
        winner_machine.annotate("combine_metal_with_omol")
        best_candidate = winner_machine.freeze()

    winner_machine = MetalCandidateStateMachine.from_candidate_state(best_candidate)
    winner_machine.annotate("select_best_candidate")
    return winner_machine.freeze()


def xyz2omol(
    xyz_block: str,
    total_charge: int = 0,
    total_radical_electrons: int = 0,
    *,
    config: MolGRConfig | None = None,
    _diagnostics: MutableMapping[str, object] | None = None,
) -> Optional[pybel.Molecule]:
    """Reconstruct an Open Babel molecule using the Python backend.

    Parameters
    ----------
    xyz_block : str
        One XYZ frame including all hydrogen atoms, with coordinates in angstroms.
    total_charge : int, default 0
        Net charge shared by the metal centers and organic fragments.
    total_radical_electrons : int, default 0
        Target count of unpaired electrons. From the public interface this
        is ``spin_multiplicity - 1`` (zero for a singlet).
    config : MolGRConfig, optional
        Reconstruction settings passed to the candidate search.
    _diagnostics : mutable mapping, optional
        Internal output mapping populated with the failure record when the
        search raises or returns no candidate.

    Returns
    -------
    pybel.Molecule or None
        The selected candidate's combined graph, or None if no reconstruction
        was found. Use ``molgr.interface.xyz_to_rdmol`` for the finalized RDKit
        graph with coordination and stereo processing.
    """

    collector = ReconstructionDiagnosticCollector() if _diagnostics is not None else None
    try:
        candidate = xyz2omol_state(
            xyz_block,
            total_charge,
            total_radical_electrons,
            config=config,
            _diagnostics=collector,
        )
    except Exception as exc:
        if collector is not None:
            assert _diagnostics is not None
            diagnostic = collector.fail(
                ReconstructionFailureCode.BACKEND_EXCEPTION,
                "reconstruction",
                "The Python reconstruction pipeline raised an exception.",
                cause=exc,
            )
            _diagnostics.update(diagnostic.as_dict())
        raise
    if candidate is None:
        if collector is not None:
            assert _diagnostics is not None
            diagnostic = collector.failure or collector.finish(
                code=ReconstructionFailureCode.NO_VALID_RECONSTRUCTION,
                stage="reconstruction",
                message="The Python reconstruction pipeline produced no candidate.",
            )
            _diagnostics.update(diagnostic.as_dict())
        return None
    return candidate.combined_omol


__all__ = [
    "xyz2omol",
    "xyz2omol_state",
]
