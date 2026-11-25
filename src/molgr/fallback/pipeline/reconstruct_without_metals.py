"""Orchestrate no-metal reconstruction from XYZ input to one selected graph.

The pipeline has four phases:

1. Parse and deterministically prepare one seed graph.
2. Widen neighbor-radical hypotheses by discrepancy and search their resonances.
3. If no valid candidate exists, try deformed-pi and then bond-break recovery.
4. Select the best validated candidate by chemical topology, then force field,
   with a graph-only deterministic tie break.

Chemical transformations live in ``utils.no_metals`` modules. This module only
controls phase ordering, fallback policy, caching, and result materialization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple, dataclass, field, replace
from typing import Optional, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import CONFIG, MolGRConfig
from molgr.fallback.pipeline.resonance import walk_radical_resonances
from molgr.fallback.stages.break_bond import break_deformed_ene, break_one_bond
from molgr.fallback.stages.clean import clean_carbene_neighbor_unsaturated
from molgr.fallback.stages.eliminate import (
    eliminate_carbene_neighbor_heteroatom,
    eliminate_carboxyl,
    eliminate_high_positive_charge_atoms,
    eliminate_NNN,
)
from molgr.fallback.stages.fresh import fresh_omol_charge_radical
from molgr.fallback.stages.preprocess import make_connections, pre_clean, validate_omol
from molgr.fallback.state import OmolStateMachine, ReconstructionState
from molgr.fallback.utils import resonance as resonance_utils
from molgr.fallback.utils.coordination_geometry import metal_contact_distances
from molgr.fallback.utils.coordination_seeds import expand_coordination_seeds
from molgr.fallback.utils.dataclasses import MetalSite
from molgr.fallback.utils.no_metals import neighbor_radicals, preparation, recovery, selection
from molgr.fallback.utils.no_metals import (
    resonance as no_metal_resonance,
)
from molgr.fallback.utils.ob_graph import mol_atoms
from molgr.fallback.utils.tools import typed_lru_cache
from molgr.process_guard import ensure_current_process


_DEFAULT_NO_METAL_STATE_CACHE_MAXSIZE = 1024


@dataclass(frozen=True)
class _ConfigCacheToken:
    key: tuple[object, ...]
    config: MolGRConfig = field(compare=False, hash=False)


def _config_cache_token(config: MolGRConfig) -> _ConfigCacheToken:
    return _ConfigCacheToken(astuple(config), config)


@dataclass(frozen=True)
class _ResonanceLayerResult:
    """States produced at the two boundaries of one resonance search layer."""

    resonance_seeds: list[ReconstructionState]
    validated_candidates: list[ReconstructionState]


@dataclass(frozen=True)
class _PrimarySearchResult:
    """Primary candidates plus every seed retained as a possible recovery source."""

    attempted_resonance_seeds: list[ReconstructionState]
    validated_candidates: list[ReconstructionState]


def _run_linear_preparation(seed_state: ReconstructionState) -> ReconstructionState:
    """Run the non-branching front of the algorithm in its chemical order."""

    machine = OmolStateMachine.from_reconstruction_state(seed_state)

    # Establish topology and explicit electronic labels before consuming charge.
    machine.run_omol_stage("make_connections", make_connections)
    machine.run_omol_stage("pre_clean", pre_clean)
    machine.run_omol_stage(
        "fresh_omol_charge_radical_initial",
        fresh_omol_charge_radical,
    )
    machine.set_given_charge(
        "initialize_charge_budget",
        seed_state.total_charge
        - sum(atom.GetFormalCharge() for atom in mol_atoms(cast(ob.OBMol, machine.omol.OBMol))),
    )

    # Deterministic local charge reconstruction; order is chemically significant.
    machine.run_omol_charge_stage("eliminate_NNN_negative", eliminate_NNN, False)
    machine.run_omol_charge_stage(
        "eliminate_high_positive_charge_atoms",
        eliminate_high_positive_charge_atoms,
        seed_state.metadata.get("metal_contact_distances", ()),
    )
    machine.run_omol_charge_stage("eliminate_NNN_positive", eliminate_NNN, True)
    machine.run_omol_charge_stage(
        "eliminate_carboxyl",
        eliminate_carboxyl,
        seed_state.metadata.get("metal_contact_distances", ()),
    )

    # Resolve deterministic carbene patterns before any candidate branching.
    machine.run_omol_stage(
        "clean_carbene_neighbor_unsaturated_first",
        clean_carbene_neighbor_unsaturated,
    )
    machine.run_omol_charge_stage(
        "eliminate_carbene_neighbor_heteroatom",
        eliminate_carbene_neighbor_heteroatom,
    )
    machine.annotate("prepare_no_metal_seed")
    return machine.freeze_like(seed_state)


def _expand_resonance_layer(
    source_seeds: list[ReconstructionState],
    *,
    traversal_policy: no_metal_resonance.ResonanceTraversalPolicy,
    config: MolGRConfig | None,
    session: no_metal_resonance._ResonanceSearchSession | None = None,
) -> _ResonanceLayerResult:
    """Expand optional seed normalizations, then traverse and validate resonances."""

    # Each normalization is optional: retain the incoming seeds and add changed
    # variants before continuing to the next normalization.
    resonance_seeds = no_metal_resonance.expand_boron_neighbor_radical_seeds(source_seeds)
    resonance_seeds = no_metal_resonance.expand_resonance_seed_pool_stage(
        resonance_seeds,
        phase="relocate_carbene_radical_for_resonance",
        stage=clean_carbene_neighbor_unsaturated,
        uses_charge_budget=False,
    )
    # The helper owns graph traversal and deduplication; every chemical operation
    # applied to the traversed states remains declared here in execution order.
    candidates = no_metal_resonance.search_resonance_candidates(
        resonance_seeds,
        resonance_traversal_policy=traversal_policy,
        config=config,
        session=session,
        walk_stage=walk_radical_resonances,
        full_normalization_stage=resonance_utils.process_resonance,
        validation_stage=validate_omol,
        score_stage=selection._score_reconstruction_candidate,
        topology_annotation_stage=selection._annotate_no_metal_candidate_topology,
    )
    return _ResonanceLayerResult(resonance_seeds, candidates)


def _search_discrepancy_layers(
    prepared_seed: ReconstructionState,
    *,
    max_discrepancy: int,
    traversal_policy: no_metal_resonance.ResonanceTraversalPolicy,
    config: MolGRConfig | None,
) -> _PrimarySearchResult:
    """Search primary neighbor-radical layers from least to most discrepant.

    The shared session deduplicates resonance states across widening layers. The
    accumulated seeds are retained as inputs for recovery if every layer fails.
    """

    all_resonance_seeds: list[ReconstructionState] = []
    search_session = no_metal_resonance._ResonanceSearchSession()

    for discrepancy in range(max_discrepancy + 1):
        neighbor_seeds = neighbor_radicals.enumerate_neighbor_radical_seeds(
            prepared_seed,
            exact_discrepancy=discrepancy,
        )
        if discrepancy == 0 and neighbor_radicals.has_substituted_n_o_radical_pair(
            prepared_seed.omol
        ):
            # Retain the unclosed hypothesis alongside normal closure branches.
            # The common resonance/validation path must still satisfy the same
            # charge and spin budgets before this seed can compete.
            neighbor_seeds.append(prepared_seed)
        layer = _expand_resonance_layer(
            neighbor_seeds,
            traversal_policy=traversal_policy,
            config=config,
            session=search_session,
        )
        all_resonance_seeds.extend(layer.resonance_seeds)
        if layer.validated_candidates:
            return _PrimarySearchResult(
                all_resonance_seeds,
                layer.validated_candidates,
            )

    return _PrimarySearchResult(all_resonance_seeds, [])


def _search_recovery_seed_pool(
    recovery_seeds: list[ReconstructionState],
    *,
    traversal_policy: no_metal_resonance.ResonanceTraversalPolicy,
    config: MolGRConfig | None,
) -> _ResonanceLayerResult:
    """Expand and search one recovery seed pool, or return an empty layer."""

    if not recovery_seeds:
        return _ResonanceLayerResult([], [])
    return _expand_resonance_layer(
        recovery_seeds,
        traversal_policy=traversal_policy,
        config=config,
    )


def _search_recovery_tiers(
    resonance_seeds: list[ReconstructionState],
    *,
    traversal_policy: no_metal_resonance.ResonanceTraversalPolicy,
    config: MolGRConfig | None,
) -> list[ReconstructionState]:
    """Try recovery in increasing order of structural destructiveness."""

    deformed_pi_seeds = recovery.enumerate_deformed_pi_recovery_seeds(
        resonance_seeds,
        break_stage=break_deformed_ene,
    )
    deformed_pi_layer = _search_recovery_seed_pool(
        deformed_pi_seeds,
        traversal_policy=traversal_policy,
        config=config,
    )
    if deformed_pi_layer.validated_candidates:
        return deformed_pi_layer.validated_candidates

    bond_break_seeds = recovery.enumerate_bond_break_recovery_seeds(
        (*resonance_seeds, *deformed_pi_seeds),
        break_stage=break_one_bond,
    )
    bond_break_layer = _search_recovery_seed_pool(
        bond_break_seeds,
        traversal_policy=traversal_policy,
        config=config,
    )
    return bond_break_layer.validated_candidates


def _find_no_metal_candidates(
    prepared_seed: ReconstructionState,
    *,
    config: MolGRConfig | None,
) -> list[ReconstructionState]:
    """Run primary resonance search, escalating to recovery only if necessary."""

    resolved_config = CONFIG if config is None else config
    shallow_config = replace(
        resolved_config,
        resonance=replace(
            resolved_config.resonance,
            limited_discrepancy_max_discrepancy=min(
                1, resolved_config.resonance.limited_discrepancy_max_discrepancy
            ),
        ),
    )
    traversal_policy = no_metal_resonance._default_resonance_traversal_policy(shallow_config)
    max_discrepancy = max(
        0,
        int(resolved_config.resonance.neighbor_radical_max_discrepancy),
    )
    # Preserve the established shallow solution when one exists. Spend the
    # larger configured depth only before destructive recovery becomes necessary.
    primary_config = replace(
        shallow_config,
        resonance=replace(
            shallow_config.resonance, max_depth=min(2, resolved_config.resonance.max_depth)
        ),
    )
    primary = _search_discrepancy_layers(
        prepared_seed,
        max_discrepancy=max_discrepancy,
        traversal_policy=traversal_policy,
        config=primary_config,
    )
    if primary.validated_candidates:
        return primary.validated_candidates
    if resolved_config.resonance.max_depth > 2:
        deeper = no_metal_resonance.search_resonance_candidates(
            primary.attempted_resonance_seeds,
            resonance_traversal_policy=traversal_policy,
            config=resolved_config,
        )
        if deeper:
            return deeper
    full_policy = no_metal_resonance._default_resonance_traversal_policy(resolved_config)
    if resolved_config.resonance.limited_discrepancy_max_discrepancy > 1:
        widened = no_metal_resonance.search_resonance_candidates(
            primary.attempted_resonance_seeds,
            resonance_traversal_policy=full_policy,
            config=resolved_config,
        )
        if widened:
            return widened
    return _search_recovery_tiers(
        primary.attempted_resonance_seeds,
        traversal_policy=full_policy,
        config=config,
    )


def _mark_selected_candidate(candidate: ReconstructionState) -> ReconstructionState:
    result_machine = OmolStateMachine.from_reconstruction_state(candidate)
    result_machine.annotate("select_best_no_metal_candidate")
    return result_machine.freeze_like(candidate)


def _run_no_metal_pipeline_from_state(
    seed_state: ReconstructionState,
    *,
    config: MolGRConfig | None = None,
    candidate_pool: list[ReconstructionState] | None = None,
    metal_sites: Sequence[MetalSite] = (),
) -> Optional[ReconstructionState]:
    """Search each ligand occupation allowed by geometry, retaining validated alternatives for metal scoring."""
    if seed_state.total_radical_electrons < 0:
        return None

    if metal_sites:
        distances = metal_contact_distances(
            seed_state.omol.OBMol, metal_sites, (CONFIG if config is None else config).metal_scoring
        )
        seed_state = replace(
            seed_state, metadata={**seed_state.metadata, "metal_contact_distances": distances}
        )
    prepared_seed = _run_linear_preparation(seed_state)
    seeds = preparation.expand_geometry_electronic_seeds(prepared_seed)
    if metal_sites:
        geometry_config = (CONFIG if config is None else config).metal_scoring
        seeds = [
            alternative
            for seed in seeds
            for alternative in expand_coordination_seeds(seed, metal_sites, geometry_config)
        ]
    candidates = [
        candidate
        for local_seed in seeds
        for candidate in _find_no_metal_candidates(local_seed, config=config)
    ]
    if not candidates:
        return None

    if candidate_pool is not None:
        candidate_pool.extend(candidates)

    best_candidate = selection.select_best_no_metal_candidate(candidates, config=config)
    if best_candidate is None:
        return None
    return _mark_selected_candidate(best_candidate)


@typed_lru_cache(maxsize=_DEFAULT_NO_METAL_STATE_CACHE_MAXSIZE, typed=True)
def _run_no_metal_pipeline_cached(
    seed_state: ReconstructionState,
    config_token: _ConfigCacheToken,
) -> Optional[ReconstructionState]:
    return _run_no_metal_pipeline_from_state(seed_state, config=config_token.config)


def xyz_to_omol_no_metal_state(
    xyz_block: str,
    total_charge: int = 0,
    total_radical_electrons: int = 0,
    *,
    config: MolGRConfig | None = None,
    metal_sites: Sequence[MetalSite] = (),
    candidate_pool: list[ReconstructionState] | None = None,
) -> Optional[ReconstructionState]:
    """Return the best no-metal reconstruction state for the requested charge/radicals."""

    ensure_current_process("molgr.fallback.xyz_to_omol_no_metal_state")
    if metal_sites or candidate_pool is not None:
        return _seed_omol_to_omol_no_metal_state(
            preparation._seed_omol_from_xyz(xyz_block, metal_sites),
            total_charge,
            total_radical_electrons,
            config=config,
            candidate_pool=candidate_pool,
            metal_sites=metal_sites,
        )
    seed_state = preparation._seed_state(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )
    resolved_config = CONFIG if config is None else config
    return _run_no_metal_pipeline_cached(seed_state, _config_cache_token(resolved_config))


def _seed_omol_to_omol_no_metal_state(
    seed_omol: pybel.Molecule,
    total_charge: int = 0,
    total_radical_electrons: int = 0,
    *,
    config: MolGRConfig | None = None,
    candidate_pool: list[ReconstructionState] | None = None,
    metal_sites: Sequence[MetalSite] = (),
) -> Optional[ReconstructionState]:
    """Return the best no-metal state from a shared parsed seed molecule."""

    seed_state = preparation._seed_state_from_omol(
        seed_omol,
        total_charge,
        total_radical_electrons,
    )
    resolved_config = CONFIG if config is None else config
    if candidate_pool is not None or metal_sites:
        return _run_no_metal_pipeline_from_state(
            seed_state,
            config=resolved_config,
            candidate_pool=candidate_pool,
            metal_sites=metal_sites,
        )
    return _run_no_metal_pipeline_cached(seed_state, _config_cache_token(resolved_config))


def xyz_to_omol_no_metal(
    xyz_block: str,
    total_charge: int = 0,
    total_radical_electrons: int = 0,
    *,
    config: MolGRConfig | None = None,
) -> Optional[pybel.Molecule]:
    """Materialize the winning no-metal reconstruction."""

    state = xyz_to_omol_no_metal_state(
        xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    if state is None:
        return None
    return state.omol


__all__ = [
    "xyz_to_omol_no_metal",
    "xyz_to_omol_no_metal_state",
]
