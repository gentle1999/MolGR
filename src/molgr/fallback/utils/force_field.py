from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, Union, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.utils.dataclasses import MetalAtomPosition
from molgr.fallback.utils.electrons import get_unpaired_electron_count
from molgr.fallback.utils.ob_graph import indexed_atoms, indexed_bonds
from molgr.fallback.utils.tools import typed_lru_cache


if TYPE_CHECKING:
    from molgr.fallback.state import ReconstructionState


ForceFieldAtomKey = Tuple[int, int, int, int, int, int, int, bool]
ForceFieldBondKey = Tuple[int, int, int, bool]
ForceFieldScoreKey = Tuple[Tuple[ForceFieldAtomKey, ...], Tuple[ForceFieldBondKey, ...]]
ForceFieldSetupAtomKey = Tuple[int, int, int, int, int, bool]
ForceFieldSetupKey = Tuple[Tuple[ForceFieldSetupAtomKey, ...], Tuple[ForceFieldBondKey, ...]]
OpenBabelSetupAtomKey = Tuple[int, int]
OpenBabelSetupBondKey = Tuple[int, int, int, int]
OpenBabelSetupKey = Tuple[Tuple[OpenBabelSetupAtomKey, ...], Tuple[OpenBabelSetupBondKey, ...]]
MetalStateKey = Tuple[Tuple[int, int, int, int, int, int, int], ...]
BondOrderOverrides = Mapping[Tuple[int, int], int]
ForceFieldInput = Union[pybel.Molecule, "ReconstructionState", "OmolForceFieldContext"]
_FIXED_FORCE_FIELD = "uff"
_DEFAULT_FORCE_FIELD_CACHE_MAXSIZE = 4096
_COORDINATE_SCALE = 1_000_000
_FORCE_FIELD_UNIT_TO_KJ_MOL = {
    "kj/mol": 1.0,
    "kcal/mol": 4.184,
}
_force_field_setup_lock = threading.RLock()
_force_field_setup_state: dict[str, tuple[ForceFieldSetupKey, OpenBabelSetupKey]] = {}


def _quantized_coordinate(value: float) -> int:
    return int(round(float(value) * _COORDINATE_SCALE))


def _build_score_key(omol: pybel.Molecule) -> ForceFieldScoreKey:
    obmol = cast(ob.OBMol, omol.OBMol)
    atom_keys: list[ForceFieldAtomKey] = []
    for obatom in indexed_atoms(obmol):
        atom_keys.append(
            (
                int(obatom.GetAtomicNum()),
                int(obatom.GetFormalCharge()),
                int(get_unpaired_electron_count(obatom)),
                int(obatom.GetHyb()),
                _quantized_coordinate(obatom.GetX()),
                _quantized_coordinate(obatom.GetY()),
                _quantized_coordinate(obatom.GetZ()),
                bool(obatom.IsAromatic()),
            )
        )

    bond_keys: list[ForceFieldBondKey] = []
    for obbond in indexed_bonds(obmol):
        begin_idx = int(obbond.GetBeginAtomIdx())
        end_idx = int(obbond.GetEndAtomIdx())
        if begin_idx > end_idx:
            begin_idx, end_idx = end_idx, begin_idx
        bond_keys.append(
            (begin_idx, end_idx, int(obbond.GetBondOrder()), bool(obbond.IsAromatic()))
        )
    return tuple(atom_keys), tuple(sorted(bond_keys))


def _build_force_field_setup_key(obmol: ob.OBMol) -> ForceFieldSetupKey:
    atom_keys: list[ForceFieldSetupAtomKey] = []
    for obatom in indexed_atoms(obmol):
        atom_keys.append(
            (
                int(obatom.GetAtomicNum()),
                int(obatom.GetFormalCharge()),
                int(get_unpaired_electron_count(obatom)),
                int(obatom.GetHyb()),
                int(obatom.GetExplicitDegree()),
                bool(obatom.IsAromatic()),
            )
        )

    bond_keys: list[ForceFieldBondKey] = []
    for obbond in indexed_bonds(obmol):
        begin_idx = int(obbond.GetBeginAtomIdx())
        end_idx = int(obbond.GetEndAtomIdx())
        if begin_idx > end_idx:
            begin_idx, end_idx = end_idx, begin_idx
        bond_keys.append(
            (begin_idx, end_idx, int(obbond.GetBondOrder()), bool(obbond.IsAromatic()))
        )
    return tuple(atom_keys), tuple(sorted(bond_keys))


def _build_openbabel_setup_key(obmol: ob.OBMol) -> OpenBabelSetupKey:
    atom_keys: list[OpenBabelSetupAtomKey] = []
    for obatom in indexed_atoms(obmol):
        atom_keys.append((int(obatom.GetAtomicNum()), int(obatom.GetExplicitDegree())))

    bond_keys: list[OpenBabelSetupBondKey] = []
    for obbond in indexed_bonds(obmol):
        begin_atom = cast(ob.OBAtom, obbond.GetBeginAtom())
        end_atom = cast(ob.OBAtom, obbond.GetEndAtom())
        bond_keys.append(
            (
                int(obbond.GetIdx()),
                int(obbond.GetBondOrder()),
                int(begin_atom.GetAtomicNum()),
                int(end_atom.GetAtomicNum()),
            )
        )
    return tuple(atom_keys), tuple(bond_keys)


def build_force_field_score_key(omol_or_state: ForceFieldInput) -> ForceFieldScoreKey:
    return OmolForceFieldContext.from_input(omol_or_state).score_key


def build_metal_state_key(metal_states: Sequence[MetalAtomPosition]) -> MetalStateKey:
    return tuple(
        (
            metal_state.idx,
            metal_state.element_idx,
            metal_state.valence,
            metal_state.radical_num,
            _quantized_coordinate(metal_state.position_x),
            _quantized_coordinate(metal_state.position_y),
            _quantized_coordinate(metal_state.position_z),
        )
        for metal_state in metal_states
    )


class OmolForceFieldContext:
    """Hashable context wrapper around one omol instance."""

    __slots__ = ("omol", "_score_key", "_score_key_hash", "_contains_metals")

    def __init__(
        self,
        omol: pybel.Molecule,
        *,
        score_key: ForceFieldScoreKey | None = None,
        contains_metals: bool | None = None,
    ) -> None:
        self.omol = omol
        self._score_key = score_key
        self._score_key_hash: int | None = None
        self._contains_metals = contains_metals

    @classmethod
    def from_input(cls, omol_or_state: ForceFieldInput) -> OmolForceFieldContext:
        if isinstance(omol_or_state, OmolForceFieldContext):
            return omol_or_state
        if isinstance(omol_or_state, pybel.Molecule):
            return cls(omol_or_state)

        from molgr.fallback.state import ReconstructionState

        if isinstance(omol_or_state, ReconstructionState):
            cached_key = omol_or_state.metadata.get("force_field_score_key")
            cached_score_key = cached_key if isinstance(cached_key, tuple) else None
            return cls(omol_or_state.omol, score_key=cached_score_key)

        raise TypeError(f"Unsupported force-field input: {type(omol_or_state)!r}")

    @property
    def score_key(self) -> ForceFieldScoreKey:
        if self._score_key is None:
            self._score_key = _build_score_key(self.omol)
        return self._score_key

    @property
    def contains_metals(self) -> bool:
        if self._contains_metals is None:
            self._contains_metals = _contains_metal_atoms(self.omol)
        return self._contains_metals

    def __hash__(self) -> int:
        if self._score_key_hash is None:
            self._score_key_hash = hash(self.score_key)
        return self._score_key_hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OmolForceFieldContext):
            return NotImplemented
        return self.score_key == other.score_key


@dataclass(frozen=True)
class ForceFieldEvaluation:
    raw_energy: float
    raw_unit: str
    energy_kj_mol: float
    atom_count: int
    heavy_atom_count: int
    contains_metals: bool


def _contains_metal_atoms(omol: pybel.Molecule) -> bool:
    obmol = cast(ob.OBMol, omol.OBMol)
    return any(atom.IsMetal() for atom in indexed_atoms(obmol))


def _strip_metal_atoms(omol: pybel.Molecule) -> pybel.Molecule:
    obmol = ob.OBMol(cast(ob.OBMol, omol.OBMol))
    metal_indices = [cast(int, atom.GetIdx()) for atom in indexed_atoms(obmol) if atom.IsMetal()]
    if not metal_indices:
        return pybel.Molecule(obmol)
    obmol.BeginModify()
    try:
        for atom_idx in reversed(metal_indices):
            atom = cast(Optional[ob.OBAtom], obmol.GetAtom(atom_idx))
            if atom is not None:
                obmol.DeleteAtom(atom)
    finally:
        obmol.EndModify()
    return pybel.Molecule(obmol)


def _count_heavy_atoms(obmol: ob.OBMol) -> int:
    return sum(1 for atom in indexed_atoms(obmol) if atom.GetAtomicNum() != 1)


def _force_field_energy_to_kj_mol(raw_energy: float, raw_unit: str) -> float:
    normalized_unit = raw_unit.strip().lower()
    factor = _FORCE_FIELD_UNIT_TO_KJ_MOL.get(normalized_unit)
    if factor is None:
        raise ValueError(f"Unsupported force-field energy unit: {raw_unit!r}")
    return raw_energy * factor


def _reset_force_field_setup(force_field: ob.OBForceField) -> None:
    force_field.Setup(ob.OBMol())


def _prepare_force_field_for_setup(
    force_field_name: str,
    force_field: ob.OBForceField,
    exact_setup_key: ForceFieldSetupKey,
    openbabel_setup_key: OpenBabelSetupKey,
) -> None:
    previous_keys = _force_field_setup_state.get(force_field_name)
    if previous_keys is None:
        # FindForceField returns a process-global plugin instance. Clear any
        # setup left by OpenBabel callers outside this cache before first use.
        _reset_force_field_setup(force_field)
        return

    previous_exact_key, previous_openbabel_key = previous_keys
    if previous_exact_key != exact_setup_key and previous_openbabel_key == openbabel_setup_key:
        # OpenBabel's IsSetupNeeded ignores details such as charge/aromaticity.
        # Force a rebuild when MolGR's exact setup key changed but OB would skip.
        _reset_force_field_setup(force_field)


def _build_force_field_evaluation(context: OmolForceFieldContext) -> ForceFieldEvaluation:
    contains_metals = context.contains_metals
    working_obmol = ob.OBMol(cast(ob.OBMol, context.omol.OBMol))
    working_obmol.SetAromaticPerceived(False)
    atom_count = int(working_obmol.NumAtoms())
    heavy_atom_count = _count_heavy_atoms(working_obmol)

    force_field = cast("ob.OBForceField | None", ob.OBForceField.FindForceField(_FIXED_FORCE_FIELD))
    if not force_field:
        raise ValueError("Could not evaluate force-field energy with fixed 'uff': unavailable")
    candidate_obmol = ob.OBMol(working_obmol)
    exact_setup_key = _build_force_field_setup_key(candidate_obmol)
    openbabel_setup_key = _build_openbabel_setup_key(candidate_obmol)
    with _force_field_setup_lock:
        _prepare_force_field_for_setup(
            _FIXED_FORCE_FIELD,
            force_field,
            exact_setup_key,
            openbabel_setup_key,
        )
        if not bool(force_field.Setup(candidate_obmol)):
            _force_field_setup_state.pop(_FIXED_FORCE_FIELD, None)
            raise ValueError("Could not evaluate force-field energy with fixed 'uff': setup_failed")
        _force_field_setup_state[_FIXED_FORCE_FIELD] = (
            exact_setup_key,
            openbabel_setup_key,
        )
        raw_energy = float(force_field.Energy())
        raw_unit = str(force_field.GetUnit())
    energy_kj_mol = _force_field_energy_to_kj_mol(raw_energy, raw_unit)
    return ForceFieldEvaluation(
        raw_energy=raw_energy,
        raw_unit=raw_unit,
        energy_kj_mol=energy_kj_mol,
        atom_count=atom_count,
        heavy_atom_count=heavy_atom_count,
        contains_metals=contains_metals,
    )


def _build_force_field_evaluation_cached_impl(
    context: OmolForceFieldContext,
) -> ForceFieldEvaluation:
    return _build_force_field_evaluation(context)


_force_field_evaluation_cached = typed_lru_cache(
    maxsize=_DEFAULT_FORCE_FIELD_CACHE_MAXSIZE,
    typed=True,
)(_build_force_field_evaluation_cached_impl)


def force_field_evaluation_cache_info() -> Tuple[int, int, int]:
    info = _force_field_evaluation_cached.cache_info()
    return info.hits, info.misses, info.currsize


def force_field_evaluation_cache_clear() -> None:
    _force_field_evaluation_cached.cache_clear()
    with _force_field_setup_lock:
        _force_field_setup_state.clear()


def _force_field_evaluation_from_context(
    context: OmolForceFieldContext,
) -> ForceFieldEvaluation:
    return _force_field_evaluation_cached(context)


def force_field_evaluation(omol_or_state: ForceFieldInput) -> ForceFieldEvaluation:
    context = OmolForceFieldContext.from_input(omol_or_state)
    return _force_field_evaluation_from_context(context)


def force_field_energy(omol_or_state: ForceFieldInput) -> float:
    return force_field_evaluation(omol_or_state).energy_kj_mol


def organic_force_field_evaluation(omol_or_state: ForceFieldInput) -> ForceFieldEvaluation:
    context = OmolForceFieldContext.from_input(omol_or_state)
    if context.contains_metals:
        raise ValueError(
            "Organic force-field evaluation only supports metal-free molecules; "
            "use selection_force_field_evaluation(...) for the default metal-aware policy or "
            "combined_force_field_evaluation(...) for raw full-molecule UFF diagnostics."
        )
    return _force_field_evaluation_from_context(context)


def organic_force_field_energy(omol_or_state: ForceFieldInput) -> float:
    return organic_force_field_evaluation(omol_or_state).energy_kj_mol


def combined_force_field_evaluation(omol_or_state: ForceFieldInput) -> ForceFieldEvaluation:
    return force_field_evaluation(omol_or_state)


def combined_force_field_energy(omol_or_state: ForceFieldInput) -> float:
    return combined_force_field_evaluation(omol_or_state).energy_kj_mol


def selection_force_field_evaluation(omol_or_state: ForceFieldInput) -> ForceFieldEvaluation:
    context = OmolForceFieldContext.from_input(omol_or_state)
    if context.contains_metals:
        organic_context = OmolForceFieldContext(
            _strip_metal_atoms(context.omol), contains_metals=False
        )
        return _force_field_evaluation_from_context(organic_context)
    return _force_field_evaluation_from_context(context)


def selection_force_field_energy(omol_or_state: ForceFieldInput) -> float:
    return selection_force_field_evaluation(omol_or_state).energy_kj_mol


__all__ = [
    "BondOrderOverrides",
    "ForceFieldEvaluation",
    "ForceFieldInput",
    "ForceFieldScoreKey",
    "MetalStateKey",
    "OmolForceFieldContext",
    "build_force_field_score_key",
    "build_metal_state_key",
    "combined_force_field_energy",
    "combined_force_field_evaluation",
    "force_field_energy",
    "force_field_evaluation",
    "force_field_evaluation_cache_clear",
    "force_field_evaluation_cache_info",
    "organic_force_field_energy",
    "organic_force_field_evaluation",
    "selection_force_field_energy",
    "selection_force_field_evaluation",
]
