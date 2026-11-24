# pyright: reportMissingImports=false

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Sequence, Tuple, cast

from openbabel import openbabel as ob

from molgr.config import CONFIG, MetalRadicalInferenceConfig, MetalScoringConfig, MolGRConfig
from molgr.fallback.utils import consts
from molgr.fallback.utils.dataclasses import FDSP
from molgr.fallback.utils.ob_graph import mol_atoms
from molgr.utils.coordination import coordination_distance_cutoff


_D_BLOCK_REARRANGEMENT_METALS = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}
_DONOR_FIELD_STRENGTH: Dict[int, float] = {
    1: 1.20,
    2: 0.00,
    5: 1.30,
    6: 1.30,
    7: 1.00,
    8: 0.85,
    9: 0.65,
    10: 0.00,
    14: 1.30,
    # Phosphines are strong sigma donors; retain a distinct boost over N/O.
    15: 1.50,
    16: 0.75,
    17: 0.55,
    18: 0.00,
    32: 1.30,
    33: 1.40,
    # Selenium donors are treated as strong-field, including after the
    # tetrahedral geometry penalty.
    34: 1.50,
    35: 0.50,
    36: 0.00,
    51: 1.35,
    52: 1.45,
    53: 0.45,
    54: 0.00,
    84: 1.40,
    85: 0.40,
    86: 0.00,
}
_GEOMETRY_FIELD_ADJUSTMENT: Dict[str, float] = {
    "free_ion": 0.0,
    "terminal": 0.0,
    "bent": 0.0,
    "linear": 0.20,
    "trigonal_planar": 0.15,
    "trigonal_pyramidal": -0.05,
    "tetrahedral": -0.25,
    "square_planar": 0.35,
    "octahedral_like": 0.05,
}
_METAL_HYDRIDE_COVALENT_TOLERANCE_ANGSTROM = 0.45
_MIN_METAL_HYDRIDE_CUTOFF_ANGSTROM = 1.25


@dataclasses.dataclass(frozen=True)
class _ShellOccupation:
    remaining_f: int
    effective_d: int
    residual_sp: int


@dataclasses.dataclass(frozen=True)
class _DonorSample:
    atom_idx: int
    atomic_num: int
    distance_angstrom: float
    vector: Tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class MetalRadicalInferenceResult:
    radical_counts: Tuple[int, ...]
    effective_d_electrons: int
    residual_sp_electrons: int
    remaining_f_electrons: int
    coordination_number: int
    geometry: str
    field_score: float
    field_strength: str


def _vector_from_atoms(metal_atom: ob.OBAtom, donor_atom: ob.OBAtom) -> Tuple[float, float, float]:
    return (
        donor_atom.GetX() - metal_atom.GetX(),
        donor_atom.GetY() - metal_atom.GetY(),
        donor_atom.GetZ() - metal_atom.GetZ(),
    )


def _vector_norm(vector: Tuple[float, float, float]) -> float:
    x, y, z = vector
    return math.sqrt(x * x + y * y + z * z)


def _clamped_covalent_radius(atomic_num: int) -> float:
    value = float(ob.GetCovalentRad(atomic_num))
    if value <= 0.0 or not math.isfinite(value):
        return 0.0
    return value


def _metal_hydride_cutoff_angstrom(metal_atomic_num: int) -> float:
    return max(
        _MIN_METAL_HYDRIDE_CUTOFF_ANGSTROM,
        _clamped_covalent_radius(metal_atomic_num)
        + _clamped_covalent_radius(1)
        + _METAL_HYDRIDE_COVALENT_TOLERANCE_ANGSTROM,
    )


def _is_direct_metal_hydride(
    metal_atom: ob.OBAtom,
    donor_atom: ob.OBAtom,
    distance: float,
) -> bool:
    return donor_atom.GetAtomicNum() == 1 and distance <= _metal_hydride_cutoff_angstrom(
        metal_atom.GetAtomicNum()
    )


def _cross(
    lhs: Tuple[float, float, float], rhs: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    )


def _dot(lhs: Tuple[float, float, float], rhs: Tuple[float, float, float]) -> float:
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _angle_degrees(lhs: Tuple[float, float, float], rhs: Tuple[float, float, float]) -> float:
    lhs_norm = _vector_norm(lhs)
    rhs_norm = _vector_norm(rhs)
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return 0.0
    cos_value = _dot(lhs, rhs) / (lhs_norm * rhs_norm)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def _planarity_distance(vectors: Sequence[Tuple[float, float, float]]) -> float:
    if len(vectors) < 3:
        return float("inf")

    best_normal: Tuple[float, float, float] | None = None
    best_norm = 0.0
    for i, lhs in enumerate(vectors):
        for rhs in vectors[i + 1 :]:
            normal = _cross(lhs, rhs)
            normal_norm = _vector_norm(normal)
            if normal_norm > best_norm:
                best_normal = normal
                best_norm = normal_norm

    if best_normal is None or best_norm <= 1e-8:
        return float("inf")

    normal = (
        best_normal[0] / best_norm,
        best_normal[1] / best_norm,
        best_normal[2] / best_norm,
    )
    return sum(abs(_dot(vector, normal)) for vector in vectors) / float(len(vectors))


def _collect_coordination_environment(
    metal_atom: ob.OBAtom,
    *,
    metal_radical_config: MetalRadicalInferenceConfig,
    metal_scoring_config: MetalScoringConfig,
) -> Tuple[_DonorSample, ...]:
    parent = cast(ob.OBMol, metal_atom.GetParent())
    donors: List[_DonorSample] = []
    for neighbor in mol_atoms(parent):
        if neighbor.GetIdx() == metal_atom.GetIdx():
            continue
        if neighbor.IsMetal():
            continue
        atomic_num = neighbor.GetAtomicNum()

        vector = _vector_from_atoms(metal_atom, neighbor)
        distance = _vector_norm(vector)
        if atomic_num <= 1:
            if not _is_direct_metal_hydride(metal_atom, neighbor, distance):
                continue
        else:
            covalent_cutoff = coordination_distance_cutoff(
                int(metal_atom.GetAtomicNum()),
                int(atomic_num),
                radius_scale=metal_scoring_config.metal_access_radius_scale,
                extra_tolerance_angstrom=(
                    metal_scoring_config.metal_coordination_extra_tolerance_angstrom
                ),
            )
            if distance > covalent_cutoff:
                continue

        donors.append(
            _DonorSample(
                atom_idx=neighbor.GetIdx(),
                atomic_num=atomic_num,
                distance_angstrom=distance,
                vector=vector,
            )
        )

    donors.sort(key=lambda donor: donor.distance_angstrom)
    return tuple(donors[: metal_radical_config.max_considered_donors])


def _classify_geometry(
    donors: Sequence[_DonorSample],
    *,
    metal_radical_config: MetalRadicalInferenceConfig,
) -> str:
    coordination_number = len(donors)
    vectors = [donor.vector for donor in donors]
    if coordination_number == 0:
        return "free_ion"
    if coordination_number == 1:
        return "terminal"
    if coordination_number == 2:
        angle = _angle_degrees(vectors[0], vectors[1])
        if angle >= metal_radical_config.linear_angle_min_degrees:
            return "linear"
        return "bent"
    if coordination_number == 3:
        if (
            _planarity_distance(vectors)
            <= metal_radical_config.trigonal_planar_planarity_tolerance_angstrom
        ):
            return "trigonal_planar"
        return "trigonal_pyramidal"
    if coordination_number == 4:
        if (
            _planarity_distance(vectors)
            <= metal_radical_config.square_planar_planarity_tolerance_angstrom
        ):
            return "square_planar"
        return "tetrahedral"
    return "octahedral_like"


def _donor_field_score(
    donors: Sequence[_DonorSample],
    *,
    geometry: str,
) -> float:
    if not donors:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for donor in donors:
        base_strength = _DONOR_FIELD_STRENGTH.get(donor.atomic_num, 0.60)
        distance_weight = 1.0 / max(donor.distance_angstrom * donor.distance_angstrom, 1.0)
        weighted_sum += base_strength * distance_weight
        total_weight += distance_weight

    if total_weight <= 0.0:
        return _GEOMETRY_FIELD_ADJUSTMENT.get(geometry, 0.0)
    return weighted_sum / total_weight + _GEOMETRY_FIELD_ADJUSTMENT.get(geometry, 0.0)


def _classify_field_strength(
    field_score: float,
    *,
    metal_radical_config: MetalRadicalInferenceConfig,
) -> str:
    ambiguity_margin = max(0.0, metal_radical_config.field_ambiguity_margin)
    if field_score >= metal_radical_config.strong_field_threshold + ambiguity_margin:
        return "strong"
    if field_score <= metal_radical_config.weak_field_threshold - ambiguity_margin:
        return "weak"
    return "ambiguous"


def _shell_occupation_after_oxidation(metal: str, valence: int) -> _ShellOccupation | None:
    fdsp: FDSP | None = consts.METAL_F_D_S_P_ELECTRONS.get(metal)
    if fdsp is None:
        return None

    f, d, s, p = fdsp.f, fdsp.d, fdsp.s, fdsp.p
    total_outer = f + d + s + p
    if valence > total_outer:
        return None

    remaining_sp = max(0, s + p - valence)
    removed_from_d = min(d, max(0, valence - (s + p)))
    remaining_d = d - removed_from_d
    removed_from_f = min(f, max(0, valence - (s + p + d)))
    remaining_f = f - removed_from_f

    promoted_to_d = 0
    if metal in _D_BLOCK_REARRANGEMENT_METALS:
        promoted_to_d = min(remaining_sp, max(0, 10 - remaining_d))

    return _ShellOccupation(
        remaining_f=remaining_f,
        effective_d=remaining_d + promoted_to_d,
        residual_sp=remaining_sp - promoted_to_d,
    )


def _candidate_d_unpaired_counts(
    effective_d: int,
    *,
    geometry: str,
    field_strength: str,
    prefer_low_spin: bool,
    allow_low_spin_alternative: bool,
) -> Tuple[int, ...]:
    if effective_d < 0 or effective_d >= len(consts.D_ELECTRONS_SPIN):
        return (0,)

    free_ion_candidates = sorted(set(consts.D_ELECTRONS_SPIN[effective_d]))
    if geometry == "square_planar":
        if effective_d == 8:
            return (0,)
        if effective_d in {7, 9}:
            return (1,)
        return (free_ion_candidates[0], free_ion_candidates[-1])

    if geometry == "tetrahedral":
        if field_strength == "strong":
            return (free_ion_candidates[0],)
        if field_strength == "ambiguous":
            return (
                (free_ion_candidates[0], free_ion_candidates[-1])
                if prefer_low_spin
                else (free_ion_candidates[-1], free_ion_candidates[0])
            )
        return (
            (free_ion_candidates[-1], free_ion_candidates[0])
            if allow_low_spin_alternative
            else (free_ion_candidates[-1],)
        )

    if len(free_ion_candidates) == 1:
        return (free_ion_candidates[0],)
    if field_strength == "strong":
        return (
            (free_ion_candidates[0], free_ion_candidates[-1])
            if allow_low_spin_alternative
            else (free_ion_candidates[0],)
        )
    if field_strength == "weak":
        return (
            (free_ion_candidates[-1], free_ion_candidates[0])
            if allow_low_spin_alternative
            else (free_ion_candidates[-1],)
        )
    return (
        (free_ion_candidates[0], free_ion_candidates[-1])
        if prefer_low_spin
        else (free_ion_candidates[-1], free_ion_candidates[0])
    )


def infer_metal_radical_state(
    metal_atom: ob.OBAtom,
    valence: int,
    *,
    config: MolGRConfig | None = None,
) -> MetalRadicalInferenceResult:
    resolved_config = CONFIG if config is None else config
    metal_radical_config = resolved_config.metal_radical_inference
    symbol = ob.GetSymbol(metal_atom.GetAtomicNum())

    occupation = _shell_occupation_after_oxidation(symbol, valence)
    if occupation is None:
        return MetalRadicalInferenceResult(
            radical_counts=(),
            effective_d_electrons=0,
            residual_sp_electrons=0,
            remaining_f_electrons=0,
            coordination_number=0,
            geometry="free_ion",
            field_score=0.0,
            field_strength="weak",
        )

    donors = _collect_coordination_environment(
        metal_atom,
        metal_radical_config=metal_radical_config,
        metal_scoring_config=resolved_config.metal_scoring,
    )
    geometry = _classify_geometry(donors, metal_radical_config=metal_radical_config)
    field_score = _donor_field_score(donors, geometry=geometry)
    field_strength = _classify_field_strength(
        field_score,
        metal_radical_config=metal_radical_config,
    )
    base_unpaired = (occupation.remaining_f + occupation.residual_sp) % 2

    d_candidates = _candidate_d_unpaired_counts(
        occupation.effective_d,
        geometry=geometry,
        field_strength=field_strength,
        prefer_low_spin=(
            field_score
            >= (
                metal_radical_config.weak_field_threshold
                + metal_radical_config.strong_field_threshold
            )
            / 2.0
        ),
        allow_low_spin_alternative=True,
    )
    radical_counts = tuple(dict.fromkeys(base_unpaired + candidate for candidate in d_candidates))
    return MetalRadicalInferenceResult(
        radical_counts=radical_counts,
        effective_d_electrons=occupation.effective_d,
        residual_sp_electrons=occupation.residual_sp,
        remaining_f_electrons=occupation.remaining_f,
        coordination_number=len(donors),
        geometry=geometry,
        field_score=field_score,
        field_strength=field_strength,
    )


def infer_metal_radical_counts(
    metal_atom: ob.OBAtom,
    valence: int,
    *,
    config: MolGRConfig | None = None,
) -> Tuple[int, ...]:
    return infer_metal_radical_state(metal_atom, valence, config=config).radical_counts


__all__ = [
    "MetalRadicalInferenceResult",
    "infer_metal_radical_counts",
    "infer_metal_radical_state",
]
