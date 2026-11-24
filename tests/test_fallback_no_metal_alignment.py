# pyright: reportMissingImports=false

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom


pytest.importorskip("openbabel")

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import MolGRConfig
from molgr.fallback.pipeline import reconstruct_without_metals as no_metal_module
from molgr.fallback.pipeline.reconstruct_without_metals import (
    xyz_to_omol_no_metal,
    xyz_to_omol_no_metal_state,
)
from molgr.fallback.stages.break_bond import break_deformed_ene, break_one_bond
from molgr.fallback.stages.clean import (
    clean_1_4_radicals,
    clean_1_6_radicals,
    clean_carbene_neighbor_unsaturated,
    clean_neighbor_radicals,
    clean_possible_1_3_dipole,
    clean_resonances_0,
    clean_resonances_2,
    clean_resonances_3,
    clean_resonances_7,
    clean_resonances_14,
    clean_resonances_16,
    clean_resonances_17,
    clean_resonances_18,
)
from molgr.fallback.stages.eliminate import (
    eliminate_1_3_dipole_postive,
    eliminate_carbene_neighbor_heteroatom,
    eliminate_carboxyl,
    eliminate_high_positive_charge_atoms,
    eliminate_negative_charges,
    eliminate_NNN,
    eliminate_positive_charges,
    eliminate_possible_cp_like_radical_anion,
)
from molgr.fallback.stages.fresh import (
    _infer_active_electron_occupancy,
    assign_charge_radical_for_atom,
    assign_radical_dots,
    fresh_omol_charge_radical,
)
from molgr.fallback.stages.preprocess import make_connections, pre_clean
from molgr.fallback.state import ReconstructionState
from molgr.fallback.utils import resonance as resonance_utils
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.no_metals import neighbor_radicals as neighbor_radical_module
from molgr.fallback.utils.no_metals import preparation as no_metal_preparation_module
from molgr.fallback.utils.no_metals import recovery as no_metal_recovery_module
from molgr.fallback.utils.no_metals import resonance as no_metal_resonance_module
from molgr.fallback.utils.no_metals import selection as no_metal_selection_module
from molgr.fallback.utils.tools import typed_lru_cache
from molgr.utils.converter import pybel_to_rdmol
from molgr.utils.equivalence import check_equivalence
from molgr.utils.post_process import make_stereochemistry


RDLogger.DisableLog("rdApp.*")  # type: ignore


_SEED_LABEL_NORMALIZATION_XYZ = """15

O          1.09630       -0.87290       -0.00190
N          0.73010       -2.07620       -0.00290
C          0.68990       -2.76820       -1.34610
C         -0.16110       -4.04780       -1.24720
C          0.13080       -4.88630       -0.00600
C         -0.15820       -4.05110        1.23820
C          0.69300       -2.77180        1.33850
C          0.08290       -1.81050       -2.38120
C          2.15530       -3.06860       -1.73930
C          2.15940       -3.07320        1.72740
C          0.08850       -1.81700        2.37780
H          0.67210       -0.88080       -2.48180
H         -0.95860       -1.55790       -2.11210
H          0.07560       -2.32530       -3.36140
H          2.16980       -3.47910       -2.76790
"""


def _total_charge_and_radicals(mol: Chem.Mol) -> tuple[int, int]:
    charge = 0
    radicals = 0
    for atom in mol.GetAtoms():  # pyright: ignore[reportCallIssue]
        charge += int(atom.GetFormalCharge())
        radicals += int(atom.GetNumRadicalElectrons())
    return charge, radicals


def _get_ptr(obmol: ob.OBMol) -> int:
    this = getattr(obmol, "this", None)
    if this is not None:
        return int(this)  # type: ignore[arg-type]
    return int(obmol)  # type: ignore[arg-type]


def _twisted_ene(*, unresolved: bool) -> pybel.Molecule:
    obmol = ob.OBMol()
    obmol.BeginModify()
    for x, y, z in ((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 0, 1)):
        atom = obmol.NewAtom()
        atom.SetAtomicNum(6)
        atom.SetVector(x, y, z)
    obmol.AddBond(1, 2, 1)
    obmol.AddBond(2, 3, 2)
    obmol.AddBond(3, 4, 1)
    obmol.EndModify()
    set_unresolved_two_electron_center(obmol.GetAtom(1), unresolved)
    return pybel.Molecule(obmol)


@pytest.mark.parametrize(
    ("unresolved", "expected_hit", "expected_order"),
    [(False, True, 1), (True, False, 2)],
)
def test_break_deformed_ene_counts_unresolved_center_electrons_for_python_and_cpp(
    unresolved: bool,
    expected_hit: bool,
    expected_order: int,
) -> None:
    python_omol = _twisted_ene(unresolved=unresolved)
    _, python_hit = break_deformed_ene(python_omol, given_charge=2, tolerance=5.0)

    assert python_hit is expected_hit
    assert python_omol.OBMol.GetBond(2, 3).GetBondOrder() == expected_order
    assert has_unresolved_two_electron_center(python_omol.OBMol.GetAtom(1)) is unresolved

    from molgr import _core  # type: ignore

    cpp_omol = _twisted_ene(unresolved=unresolved)
    cpp_hit = _core.dev.stages.break_bond.break_deformed_ene_ptr(
        _get_ptr(cpp_omol.OBMol),
        2,
        0,
        5.0,
    )

    assert cpp_hit is expected_hit
    assert cpp_omol.OBMol.GetBond(2, 3).GetBondOrder() == expected_order
    assert has_unresolved_two_electron_center(cpp_omol.OBMol.GetAtom(1)) is unresolved


@pytest.mark.parametrize("endpoints", [(1, 2), (2, 1)])
def test_cation_bond_recovery_uses_matched_endpoints(endpoints: tuple[int, int]) -> None:
    def make_omol() -> pybel.Molecule:
        mol = ob.OBMol()
        mol.BeginModify()
        mol.NewAtom().SetAtomicNum(6)
        nitrogen = mol.NewAtom()
        nitrogen.SetAtomicNum(7)
        nitrogen.SetFormalCharge(1)
        mol.AddBond(*endpoints, 2)
        mol.EndModify()
        return pybel.Molecule(mol)

    python_omol, charge, hit = break_one_bond(make_omol(), given_charge=-2)
    assert hit and charge == -1
    assert [a.OBAtom.GetFormalCharge() for a in python_omol] == [0, 0]
    assert [get_unpaired_electron_count(a.OBAtom) for a in python_omol] == [1, 0]
    assert python_omol.OBMol.GetBond(1, 2).GetBondOrder() == 1

    from molgr import _core

    cpp_omol = make_omol()
    cpp_charge, cpp_hit = _core.dev.stages.break_bond.break_one_bond_ptr(
        _get_ptr(cpp_omol.OBMol), -2, 0
    )
    assert (cpp_charge, cpp_hit) == (charge, hit)
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(python_omol)


def test_recovery_neutralizes_cation_before_breaking_carbonyl() -> None:
    """The same deficit can be filled without reducing an unrelated C=O."""
    from molgr import _core

    def make_omol():
        return pybel.readstring("smi", "CC=O.C=[N+]=C")

    python_omol = make_omol()
    _, charge, hit = break_one_bond(python_omol, given_charge=-2)
    assert hit and charge == -1
    assert python_omol.OBMol.GetBond(2, 3).GetBondOrder() == 2
    assert python_omol.OBMol.GetAtom(5).GetFormalCharge() == 0
    cpp_omol = make_omol()
    cpp_result = _core.dev.stages.break_bond.break_one_bond_ptr(_get_ptr(cpp_omol.OBMol), -2, 0)
    assert cpp_result == (charge, hit)
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(python_omol)


@pytest.mark.parametrize(
    ("partner", "distance", "expected_hit"),
    [(1, 1.09, False), (1, 1.40, True), (6, 1.54, False), (6, 1.90, True)],
)
def test_sigma_bond_recovery_requires_stretched_xyz_geometry(
    partner: int, distance: float, expected_hit: bool
) -> None:
    from molgr import _core

    def make_omol():
        mol = ob.OBMol()
        mol.SetDimension(3)
        first, second = mol.NewAtom(), mol.NewAtom()
        first.SetAtomicNum(6)
        second.SetAtomicNum(partner)
        first.SetVector(0.0, 0.0, 0.0)
        second.SetVector(distance, 0.0, 0.0)
        mol.AddBond(1, 2, 1)
        return pybel.Molecule(mol)

    python_omol, cpp_omol = make_omol(), make_omol()
    _, charge, hit = break_one_bond(python_omol, given_charge=-2)
    assert hit is expected_hit
    assert python_omol.OBMol.NumBonds() == (0 if expected_hit else 1)
    assert [get_unpaired_electron_count(a.OBAtom) for a in python_omol] == [
        int(expected_hit),
        int(expected_hit),
    ]
    assert _core.dev.stages.break_bond.break_one_bond_ptr(_get_ptr(cpp_omol.OBMol), -2, 0) == (
        charge,
        hit,
    )
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(python_omol)


@pytest.mark.parametrize("budget", [-2, -1, 0, 1])
def test_positive_nnn_preserves_radical_for_negative_charge_budget(budget) -> None:
    from molgr import _core

    def make_omol():
        omol = pybel.readstring("smi", "CN(C)[N]N(C)C")
        fresh_omol_charge_radical(omol)
        return omol

    python_omol = make_omol()
    before = _pybel_stage_signature(python_omol)
    _, charge, hit = eliminate_NNN(python_omol, budget, True)
    assert hit is (budget >= 0)
    assert charge == budget - int(hit)
    if budget < 0:
        assert _pybel_stage_signature(python_omol) == before
    cpp_omol = make_omol()
    assert _core.dev.stages.eliminate.eliminate_nnn_ptr(_get_ptr(cpp_omol.OBMol), budget, True) == (
        charge,
        hit,
    )
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(python_omol)


@pytest.mark.parametrize(
    ("unresolved", "expected_hit", "expected_order"),
    [(False, True, 1), (True, False, 2)],
)
def test_break_one_bond_counts_unresolved_center_electrons_for_python_and_cpp(
    unresolved: bool,
    expected_hit: bool,
    expected_order: int,
) -> None:
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "C=C")
        set_unresolved_two_electron_center(omol.OBMol.GetAtom(1), unresolved)
        return omol

    python_omol = make_omol()
    _, python_charge, python_hit = break_one_bond(python_omol, given_charge=2)

    assert python_charge == 2
    assert python_hit is expected_hit
    assert python_omol.OBMol.GetBond(1, 2).GetBondOrder() == expected_order
    assert has_unresolved_two_electron_center(python_omol.OBMol.GetAtom(1)) is unresolved

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_charge, cpp_hit = _core.dev.stages.break_bond.break_one_bond_ptr(
        _get_ptr(cpp_omol.OBMol),
        2,
        0,
    )

    assert cpp_charge == 2
    assert cpp_hit is expected_hit
    assert cpp_omol.OBMol.GetBond(1, 2).GetBondOrder() == expected_order
    assert has_unresolved_two_electron_center(cpp_omol.OBMol.GetAtom(1)) is unresolved


@pytest.mark.parametrize(
    ("charge_stage", "given_charge", "expected_charge"),
    [
        ("positive", 2, 2),
        ("negative", -2, -2),
    ],
)
def test_charge_assignment_consumes_pure_unresolved_center_as_two_electrons_for_python_and_cpp(
    charge_stage: str,
    given_charge: int,
    expected_charge: int,
) -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        atom = obmol.NewAtom()
        atom.SetAtomicNum(6)
        set_unresolved_two_electron_center(atom, True)
        return pybel.Molecule(obmol)

    def run_python(omol: pybel.Molecule) -> tuple[int, bool]:
        if charge_stage == "positive":
            _, remaining_charge, hit = eliminate_positive_charges(omol, given_charge)
        else:
            _, remaining_charge, hit = eliminate_negative_charges(omol, given_charge)
        return remaining_charge, hit

    def run_cpp(omol: pybel.Molecule) -> tuple[int, bool]:
        from molgr import _core  # type: ignore

        if charge_stage == "positive":
            return _core.dev.stages.eliminate.eliminate_positive_charges_ptr(
                _get_ptr(omol.OBMol), given_charge
            )
        return _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
            _get_ptr(omol.OBMol), given_charge
        )

    python_omol = make_omol()
    python_remaining, python_hit = run_python(python_omol)
    python_atom = python_omol.OBMol.GetAtom(1)

    assert python_hit
    assert python_remaining == 0
    assert python_atom.GetFormalCharge() == expected_charge
    assert get_unpaired_electron_count(python_atom) == 0
    assert get_lone_pair_count(python_atom) == 0
    assert not has_unresolved_two_electron_center(python_atom)

    cpp_omol = make_omol()
    cpp_remaining, cpp_hit = run_cpp(cpp_omol)
    cpp_atom = cpp_omol.OBMol.GetAtom(1)

    assert cpp_hit
    assert cpp_remaining == python_remaining
    assert cpp_atom.GetFormalCharge() == expected_charge
    assert get_unpaired_electron_count(cpp_atom) == 0
    assert get_lone_pair_count(cpp_atom) == 0
    assert not has_unresolved_two_electron_center(cpp_atom)


@pytest.mark.parametrize(
    ("charge_stage", "given_charge"),
    [
        ("positive", 1),
        ("negative", -1),
        ("negative", 0),
    ],
)
def test_charge_assignment_does_not_partially_consume_unresolved_center_for_python_and_cpp(
    charge_stage: str,
    given_charge: int,
) -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        atom = obmol.NewAtom()
        atom.SetAtomicNum(6)
        set_unresolved_two_electron_center(atom, True)
        return pybel.Molecule(obmol)

    python_omol = make_omol()
    if charge_stage == "positive":
        _, python_remaining, python_hit = eliminate_positive_charges(python_omol, given_charge)
    else:
        _, python_remaining, python_hit = eliminate_negative_charges(python_omol, given_charge)
    python_atom = python_omol.OBMol.GetAtom(1)

    assert not python_hit
    assert python_remaining == given_charge
    assert python_atom.GetFormalCharge() == 0
    assert has_unresolved_two_electron_center(python_atom)

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    if charge_stage == "positive":
        cpp_remaining, cpp_hit = _core.dev.stages.eliminate.eliminate_positive_charges_ptr(
            _get_ptr(cpp_omol.OBMol), given_charge
        )
    else:
        cpp_remaining, cpp_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
            _get_ptr(cpp_omol.OBMol), given_charge
        )
    cpp_atom = cpp_omol.OBMol.GetAtom(1)

    assert not cpp_hit
    assert cpp_remaining == given_charge
    assert cpp_atom.GetFormalCharge() == 0
    assert has_unresolved_two_electron_center(cpp_atom)


def _pybel_stage_signature(
    omol: pybel.Molecule,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    obmol = omol.OBMol
    atoms = tuple(
        (atom.idx, atom.OBAtom.GetFormalCharge(), get_unpaired_electron_count(atom.OBAtom))
        for atom in omol
    )
    bonds = tuple(
        sorted(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondOrder(),
            )
            for bond in ob.OBMolBondIter(obmol)
        )
    )
    return atoms, bonds


@typed_lru_cache(maxsize=128, typed=True)
def _seed_case(smiles: str) -> tuple[str, int, int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol_h = Chem.AddHs(mol)
    embed_code = rdDistGeom.EmbedMolecule(mol_h)  # pyright: ignore[reportCallIssue]
    assert int(embed_code) == 0
    charge, radicals = _total_charge_and_radicals(mol_h)
    return Chem.MolToXYZBlock(mol_h), charge, radicals


def _load_curated_smiles() -> list[str]:
    csv_path = Path(__file__).with_name("test_cases.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    curated_rows = [1, 2, 5, 10, 17]
    return [rows[idx - 1]["smiles"] for idx in curated_rows]


@pytest.mark.parametrize("smiles", _load_curated_smiles())
def test_fallback_no_metal_reconstructs_curated_cases(smiles: str) -> None:
    xyz_block, total_charge, total_radical_electrons = _seed_case(smiles)

    result = xyz_to_omol_no_metal(xyz_block, total_charge, total_radical_electrons)

    assert result is not None

    expected = Chem.MolFromSmiles(smiles)
    assert expected is not None
    equivalent, info = check_equivalence(
        expected,
        make_stereochemistry(pybel_to_rdmol(result)),
        use_chirality=True,
        max_resonance=100,
    )
    assert equivalent, info.reason


def test_fallback_no_metal_exposes_staged_history() -> None:
    xyz_block = """2
H2
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""
    state = xyz_to_omol_no_metal_state(xyz_block, 0, 0)

    assert state is not None
    assert state.phase_history[:7] == (
        "read_xyz",
        "normalize_seed_electronic_labels",
        "make_connections",
        "pre_clean",
        "fresh_omol_charge_radical_initial",
        "initialize_charge_budget",
        "eliminate_NNN_negative",
    )
    assert "prepare_no_metal_seed" in state.phase_history
    assert state.phase_history[-1] == "select_best_no_metal_candidate"


def test_seed_state_normalizes_openbabel_charge_and_spin_labels_for_python_and_cpp() -> None:
    raw_omol = pybel.readstring("xyz", _SEED_LABEL_NORMALIZATION_XYZ)
    raw_nitrogen = raw_omol.OBMol.GetAtom(2)
    raw_oxygen = raw_omol.OBMol.GetAtom(1)
    assert raw_nitrogen.GetFormalCharge() == 1
    assert raw_oxygen.GetFormalCharge() == -1

    state = no_metal_preparation_module._seed_state(_SEED_LABEL_NORMALIZATION_XYZ, 0, 0)
    assert state.phase_history[:2] == ("read_xyz", "normalize_seed_electronic_labels")
    assert all(atom.OBAtom.GetFormalCharge() == 0 for atom in state.omol)
    assert all(get_unpaired_electron_count(atom.OBAtom) == 0 for atom in state.omol)
    prepared = no_metal_preparation_module.prepare_no_metal_seed(state)

    from molgr import _core  # type: ignore

    cpp_prepared = _core.dev.pipeline.reconstruct_without_metals.debug_prepared_no_metal_seed(
        _SEED_LABEL_NORMALIZATION_XYZ,
        0,
        0,
    )
    assert cpp_prepared is not None
    assert "[N+]" not in cpp_prepared["smiles"]
    assert "[O-]" not in cpp_prepared["smiles"]
    assert tuple(cpp_prepared["phase_history"]) == prepared.phase_history


def test_make_connections_reconnects_current_donor_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()

        atom_specs = (
            (7, (0.00, 0.00, 0.00)),
            (1, (-0.60, 0.94, 0.00)),
            (1, (-0.60, -0.47, 0.82)),
            (1, (-0.60, -0.47, -0.82)),
            (5, (1.65, 0.00, 0.00)),
            (1, (2.45, 0.94, 0.00)),
            (1, (2.45, -0.47, 0.82)),
            (1, (2.45, -0.47, -0.82)),
        )
        for atomic_num, (x, y, z) in atom_specs:
            atom = obmol.NewAtom()
            atom.SetAtomicNum(atomic_num)
            atom.SetVector(x, y, z)

        for begin_idx, end_idx in ((1, 2), (1, 3), (1, 4), (5, 6), (5, 7), (5, 8)):
            obmol.AddBond(begin_idx, end_idx, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    assert omol.OBMol.GetBond(1, 5) is None

    omol, hit = make_connections(omol)

    assert hit
    assert omol.OBMol.GetBond(1, 5) is not None
    assert omol.OBMol.GetBond(1, 5).GetBondOrder() == 1

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    assert cpp_omol.OBMol.GetBond(1, 5) is None

    cpp_hit = _core.dev.stages.preprocess.make_connections_ptr(_get_ptr(cpp_omol.OBMol), 0.15)

    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


@pytest.mark.parametrize(
    ("seed", "positions", "expected_donor"),
    [
        # P and N have equal priority. The first atom wins even when N is closer.
        ("[PH2].[NH3].[CH3]", [-1.80, 1.45, 0.0], 1),
        ("[NH3].[PH2].[CH3]", [1.45, -1.80, 0.0], 1),
        # P has priority over O even when O is first and closer.
        ("[OH].[PH2].[CH3]", [1.40, -1.80, 0.0], 2),
    ],
)
def test_make_connections_competing_donors(seed, positions, expected_donor):
    from molgr import _core

    for backend in ("python", "cpp"):
        mol = pybel.readstring("smi", seed)
        for idx, x in enumerate(positions, 1):
            mol.OBMol.GetAtom(idx).SetVector(x, 0, 0)
        if backend == "python":
            make_connections(mol)
        else:
            _core.dev.stages.preprocess.make_connections_ptr(_get_ptr(mol.OBMol), 0.15)
        carbon = mol.OBMol.GetAtom(3)
        assert [bond.GetNbrAtomIdx(carbon) for bond in ob.OBAtomBondIter(carbon)] == [
            expected_donor
        ]


@pytest.mark.parametrize(
    ("seed", "distance", "should_connect"),
    [
        ("[PH]C=[CH]", 1.82, True),
        ("[PH]C[CH]", 1.86, True),
        ("[PH]C=[CH]", 2.20, False),
        ("[PH]C=C", 1.82, False),
        ("[PH2]C=[CH]", 1.82, False),
        ("[PH]C=[N]", 1.82, True),
        ("[PH]C=[N]", 2.20, False),
        ("[PH]C[N]", 1.82, False),
    ],
)
def test_make_connections_phosphorus_single_bond(seed, distance, should_connect):
    from molgr import _core

    for backend in ("python", "cpp"):
        mol = pybel.readstring("smi", seed)
        mol.OBMol.GetAtom(1).SetVector(0, 0, 0)
        mol.OBMol.GetAtom(2).SetVector(1.20, 1.35, 0)
        mol.OBMol.GetAtom(3).SetVector(distance, 0, 0)
        assert mol.OBMol.GetBond(1, 3) is None
        if backend == "python":
            _, changed = make_connections(mol)
        else:
            changed = _core.dev.stages.preprocess.make_connections_ptr(_get_ptr(mol.OBMol), 0.15)
        assert changed == should_connect
        bond = mol.OBMol.GetBond(1, 3)
        assert (bond.GetBondOrder() if bond else None) == (1 if should_connect else None)


@pytest.mark.parametrize(
    ("seed", "partner_idx", "distance", "remove"),
    [
        ("[Si](F)(F)(F)[C]=O", 5, 2.24, True),
        ("[Si](F)(F)(F)[C]=O", 5, 1.90, False),
        ("[Si](F)(F)(F)[C]=S", 5, 2.24, False),
        ("[Si](F)(F)(F)C=O", 5, 2.24, False),  # C has another H
        ("[P](F)(F)(F)[P](F)(F)F", 5, 2.59, True),
        ("[P](F)(F)(F)[P](F)(F)F", 5, 2.24, False),
        ("[P](F)(F)(F)[P](F)F", 5, 2.59, False),
        ("[Si](F)(F)(F)[Si](F)(F)F", 5, 2.64, False),
    ],
)
def test_make_connections_long_seed_bonds(seed, partner_idx, distance, remove):
    from molgr import _core

    for backend in ("python", "cpp"):
        mol = pybel.readstring("smi", seed)
        mol.OBMol.GetAtom(1).SetVector(0, 0, 0)
        mol.OBMol.GetAtom(partner_idx).SetVector(distance, 0, 0)
        assert mol.OBMol.GetBond(1, partner_idx).GetBondOrder() == 1
        if backend == "python":
            _, changed = make_connections(mol)
        else:
            changed = _core.dev.stages.preprocess.make_connections_ptr(_get_ptr(mol.OBMol), 0.15)
        assert changed == remove
        assert (mol.OBMol.GetBond(1, partner_idx) is None) == remove


def test_make_connections_rechecks_shared_phosphorus_after_deletion():
    from molgr import _core

    for backend in ("python", "cpp"):
        mol = pybel.readstring("smi", "[P](F)(F)(F)[P](F)(F)[P](F)(F)F")
        for idx, x in [(1, -2.59), (5, 0.0), (8, 2.59)]:
            mol.OBMol.GetAtom(idx).SetVector(x, 0, 0)
        if backend == "python":
            _, changed = make_connections(mol)
        else:
            changed = _core.dev.stages.preprocess.make_connections_ptr(_get_ptr(mol.OBMol), 0.15)
        assert changed
        assert mol.OBMol.GetAtom(5).GetExplicitDegree() == 3
        assert sum(mol.OBMol.GetBond(5, idx) is not None for idx in (1, 8)) == 1


@pytest.mark.parametrize(
    ("axial_neighbor", "candidate", "axial_distance", "position", "expected"),
    [
        ("C", "[CH3]", 2.00, (0, 0, 2.02), True),
        ("O", "[OH]", 1.74, (0, 0, 1.76), True),
        ("C", "[CH3]", 2.00, (1.20, 0, 1.60), False),  # outside the vacant site
        ("C", "[CH3]", 1.85, (0, 0, 2.02), False),  # unlike the existing P–C length
        ("F", "[OH]", 1.74, (0, 0, 1.76), False),  # no P–O bond for comparison
        ("C", "[CH3]", 2.12, (0, 0, 2.14), False),  # beyond the P–C distance limit
        ("C", "[CH2]", 2.00, (0, 0, 2.02), False),  # candidate has the wrong valence
    ],
)
def test_make_connections_octahedral_phosphorus(
    axial_neighbor, candidate, axial_distance, position, expected
):
    from molgr import _core

    for backend in ("python", "cpp"):
        mol = pybel.readstring("smi", f"[P](F)(F)(F)(F){axial_neighbor}.{candidate}")
        coordinates = [
            (0, 0, 0),
            (1.7, 0, 0),
            (-1.7, 0, 0),
            (0, 1.7, 0),
            (0, -1.7, 0),
            (0, 0, -axial_distance),
            position,
        ]
        for idx, xyz in enumerate(coordinates, 1):
            mol.OBMol.GetAtom(idx).SetVector(*xyz)
        if backend == "python":
            _, changed = make_connections(mol)
        else:
            changed = _core.dev.stages.preprocess.make_connections_ptr(_get_ptr(mol.OBMol), 0.15)
        assert changed == expected
        bond = mol.OBMol.GetBond(1, 7)
        assert (bond.GetBondOrder() if bond else None) == (1 if expected else None)


@pytest.mark.parametrize(
    ("seed", "distances", "expected_partners"),
    [
        ("[OH].[CH3]", [1.44], [2]),
        ("[OH].[CH3]", [1.70], []),
        ("[OH].[CH2]", [1.44], []),
        ("[SeH3].[OH]", [1.81], [2]),
        ("[SeH4].[OH]", [1.81], []),
        ("[SeH3].[CH3]", [1.81], []),
        # Fill iodine valences four and five in successive passes, nearest first.
        ("[IH4].[OH].[OH].[OH]", [2.16, 2.10, 2.13], [3, 4]),
        ("[IH4].[CH3]", [2.10], []),
    ],
)
def test_make_connections_oxygen_partners(seed, distances, expected_partners):
    from molgr import _core

    for backend in ("python", "cpp"):
        mol = pybel.readstring("smi", seed)
        for idx, distance in enumerate([0.0, *distances], 1):
            mol.OBMol.GetAtom(idx).SetVector(distance, 0, 0)
        if backend == "python":
            _, changed = make_connections(mol)
        else:
            changed = _core.dev.stages.preprocess.make_connections_ptr(_get_ptr(mol.OBMol), 0.15)
        assert changed == bool(expected_partners)
        bonds = list(ob.OBAtomBondIter(mol.OBMol.GetAtom(1)))
        assert sorted(b.GetNbrAtomIdx(mol.OBMol.GetAtom(1)) for b in bonds) == expected_partners
        assert all(b.GetBondOrder() == 1 for b in bonds)


@pytest.mark.parametrize("atomic_num", [16, 15, 33, 9, 17, 35, 53])
@pytest.mark.parametrize("bond_order", [2, 3])
def test_pre_clean_collapses_hyper_pi_bonds_for_python_and_cpp(
    atomic_num: int,
    bond_order: int,
) -> None:
    from molgr import _core  # type: ignore

    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        hyper_pi_atom = obmol.NewAtom()
        hyper_pi_atom.SetAtomicNum(atomic_num)
        carbon = obmol.NewAtom()
        carbon.SetAtomicNum(6)
        obmol.AddBond(hyper_pi_atom.GetIdx(), carbon.GetIdx(), bond_order)
        # Substituted C=X keeps the general cleanup contract. Isolated CO/CS
        # has its own closed-shell precursor and is tested separately.
        hydrogen = obmol.NewAtom()
        hydrogen.SetAtomicNum(1)
        obmol.AddBond(carbon.GetIdx(), hydrogen.GetIdx(), 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    python_omol = make_omol()
    cpp_omol = make_omol()

    _, python_hit = pre_clean(python_omol)
    cpp_hit = _core.dev.stages.preprocess.pre_clean_ptr(_get_ptr(cpp_omol.OBMol))

    assert python_hit
    assert cpp_hit == python_hit
    assert python_omol.OBMol.GetBond(1, 2).GetBondOrder() == 1
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(python_omol)


@pytest.mark.parametrize("seed", ["[Si](F)(F)(F)(F)F", "[Si](C)(O)(O)(O)O"])
def test_pre_clean_retains_five_single_bonds_to_silicon(seed: str) -> None:
    from molgr import _core

    python_mol = pybel.readstring("smi", seed)
    cpp_mol = pybel.readstring("smi", seed)
    pre_clean(python_mol)
    _core.dev.stages.preprocess.pre_clean_ptr(_get_ptr(cpp_mol.OBMol))
    for mol in (python_mol, cpp_mol):
        assert mol.OBMol.GetAtom(1).GetFormalCharge() == 0
        assert [bond.GetBondOrder() for bond in ob.OBAtomBondIter(mol.OBMol.GetAtom(1))] == [1] * 5


def test_pre_clean_normalizes_aromatic_sulfur_oxygen_for_python_and_cpp() -> None:
    from molgr import _core  # type: ignore

    def make_omol() -> pybel.Molecule:
        return pybel.readstring("smi", "COc1ns(=O)nc1OC")

    python_omol = make_omol()
    cpp_omol = make_omol()
    sulfur_idx = next(atom.idx for atom in python_omol if atom.atomicnum == 16)
    oxygen_idx = next(
        neighbor.GetIdx()
        for neighbor in ob.OBAtomAtomIter(python_omol.OBMol.GetAtom(sulfur_idx))
        if neighbor.GetAtomicNum() == 8
    )

    assert python_omol.OBMol.GetAtom(sulfur_idx).IsAromatic()
    assert python_omol.OBMol.GetBond(sulfur_idx, oxygen_idx).GetBondOrder() == 2

    _, python_pre_clean_hit = pre_clean(python_omol)
    python_omol, python_fresh_hit = fresh_omol_charge_radical(python_omol)
    python_given_charge = -sum(atom.OBAtom.GetFormalCharge() for atom in python_omol)
    python_omol, python_given_charge, python_eliminate_hit = eliminate_negative_charges(
        python_omol,
        python_given_charge,
    )

    cpp_pre_clean_hit = _core.dev.stages.preprocess.pre_clean_ptr(_get_ptr(cpp_omol.OBMol))
    cpp_fresh_hit = _core.dev.stages.fresh.fresh_omol_charge_radical_ptr(_get_ptr(cpp_omol.OBMol))
    cpp_given_charge = -sum(atom.OBAtom.GetFormalCharge() for atom in cpp_omol)
    cpp_given_charge, cpp_eliminate_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        cpp_given_charge,
    )

    assert python_pre_clean_hit
    assert python_fresh_hit
    assert python_eliminate_hit
    assert python_given_charge == 0
    assert python_omol.OBMol.GetBond(sulfur_idx, oxygen_idx).GetBondOrder() == 1
    assert python_omol.OBMol.GetAtom(sulfur_idx).GetFormalCharge() == 1
    assert python_omol.OBMol.GetAtom(oxygen_idx).GetFormalCharge() == -1
    assert cpp_pre_clean_hit == python_pre_clean_hit
    assert cpp_fresh_hit == python_fresh_hit
    assert cpp_eliminate_hit == python_eliminate_hit
    assert cpp_given_charge == python_given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(python_omol)


def test_eliminate_nnn_negative_produces_closed_shell_azide() -> None:
    omol = pybel.readstring("smi", "[N][N][N]")

    omol, hit = fresh_omol_charge_radical(omol)

    assert hit
    assert [
        (
            atom.idx,
            atom.OBAtom.GetFormalCharge(),
            get_unpaired_electron_count(atom.OBAtom),
            has_unresolved_two_electron_center(atom.OBAtom),
        )
        for atom in omol
    ] == [
        (1, 0, 0, True),
        (2, 0, 1, False),
        (3, 0, 0, True),
    ]

    omol, given_charge, hit = eliminate_NNN(omol, 0, False)

    assert hit
    assert given_charge == 1
    assert [
        (atom.idx, atom.OBAtom.GetFormalCharge(), get_unpaired_electron_count(atom.OBAtom))
        for atom in omol
    ] == [
        (1, -1, 0),
        (2, 1, 0),
        (3, -1, 0),
    ]
    assert all(not has_unresolved_two_electron_center(atom.OBAtom) for atom in omol)


def test_eliminate_nnn_rejects_missing_classified_electrons_without_mutation() -> None:
    omol = pybel.readstring("smi", "[N][N][N]")
    for atom in omol:
        set_unpaired_electron_count(atom.OBAtom, 0)
        set_lone_pair_count(atom.OBAtom, 0)
        set_unresolved_two_electron_center(atom.OBAtom, False)
    before = _pybel_stage_signature(omol)

    omol, given_charge, hit = eliminate_NNN(omol, 0, False)

    assert not hit
    assert given_charge == 0
    assert _pybel_stage_signature(omol) == before


@pytest.mark.parametrize("unpaired_electrons", [0, 2])
def test_eliminate_carboxyl_requires_one_real_unpaired_electron_for_python_and_cpp(
    unpaired_electrons: int,
) -> None:
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "[O]C=O")
        set_unpaired_electron_count(omol.OBMol.GetAtom(1), unpaired_electrons)
        return omol

    omol = make_omol()
    before = _pybel_stage_signature(omol)
    omol, given_charge, hit = eliminate_carboxyl(omol, -1)
    assert not hit
    assert given_charge == -1
    assert _pybel_stage_signature(omol) == before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    given_charge, hit = _core.dev.stages.eliminate.eliminate_carboxyl_ptr(
        _get_ptr(cpp_omol.OBMol),
        -1,
    )
    assert not hit
    assert given_charge == -1
    assert _pybel_stage_signature(cpp_omol) == before


@pytest.mark.parametrize(
    ("smiles", "expected_radical_dots", "expected_occupancy"),
    [
        ("[B-](O)O", 0, (0, 0)),
        ("[B](O)O", 1, (1, 0)),
        ("S(=O)(C)C", 0, (0, 0)),
        ("Cl(=O)(C)", 0, (0, 0)),
        ("O", 0, (0, 0)),
    ],
)
def test_assign_charge_radical_uses_active_orbital_occupancy_for_python_and_cpp(
    smiles: str,
    expected_radical_dots: int,
    expected_occupancy: tuple[int, int],
) -> None:
    from molgr import _core  # type: ignore

    python_omol = pybel.readstring("smi", smiles)
    python_atom = python_omol.OBMol.GetAtom(1)
    cpp_omol = pybel.Molecule(ob.OBMol(python_omol.OBMol))
    cpp_atom = cpp_omol.OBMol.GetAtom(1)

    assert assign_radical_dots(python_atom) == expected_radical_dots
    assert (
        _core.dev.stages.fresh.assign_radical_dots_ptr(_get_ptr(cpp_omol.OBMol), 1)
        == expected_radical_dots
    )

    assign_charge_radical_for_atom(python_atom)
    _core.dev.stages.fresh.assign_charge_radical_for_atom_ptr(_get_ptr(cpp_omol.OBMol), 1)

    def occupancy(atom: ob.OBAtom) -> tuple[int, int]:
        return get_unpaired_electron_count(atom), get_lone_pair_count(atom)

    assert occupancy(python_atom) == expected_occupancy
    assert occupancy(cpp_atom) == expected_occupancy
    assert cpp_atom.GetFormalCharge() == python_atom.GetFormalCharge()
    assert has_unresolved_two_electron_center(cpp_atom) == has_unresolved_two_electron_center(
        python_atom
    )


@pytest.mark.parametrize(
    ("charge", "bond_orders", "expected_unpaired"),
    [
        (0, (1, 1), 1),  # Neutral boryl radical.
        (0, (1, 1, 1), 0),  # Closed-shell, electron-deficient borane.
        (-1, (1,), 1),  # An anionic lone pair plus one active electron.
        (-1, (1, 1), 0),  # Preserve the closed-shell boryl-anion convention.
        (-1, (1, 1, 1), 1),  # Three-coordinate radical anion.
        (-1, (1, 1, 1, 1), 0),  # Closed-shell borate.
        (-1, (2, 1), 1),  # Count bond orders, not just neighbors.
        (-1, (2, 1, 1), 0),
        (1, (1, 1), 0),  # Removing an electron must not create two radicals.
    ],
)
def test_boron_electronic_states_survive_refresh_for_python_and_cpp(
    charge: int, bond_orders: tuple[int, ...], expected_unpaired: int
) -> None:
    from molgr import _core  # type: ignore

    for backend in ("python", "cpp"):
        mol = ob.OBMol()
        boron = mol.NewAtom()
        boron.SetAtomicNum(5)
        boron.SetFormalCharge(charge)
        for order in bond_orders:
            neighbor = mol.NewAtom()
            neighbor.SetAtomicNum(6)
            mol.AddBond(boron.GetIdx(), neighbor.GetIdx(), order)
        # A refresh must replace stale electronic labels, then be idempotent.
        set_unpaired_electron_count(boron, 3)
        set_lone_pair_count(boron, 1)
        set_unresolved_two_electron_center(boron, True)
        for expected_changed in (True, False):
            if backend == "python":
                changed = assign_charge_radical_for_atom(boron)
            else:
                changed = _core.dev.stages.fresh.assign_charge_radical_for_atom_ptr(
                    _get_ptr(mol), 1
                )
            assert changed is expected_changed
            assert boron.GetFormalCharge() == charge
            assert get_unpaired_electron_count(boron) == expected_unpaired
            assert get_lone_pair_count(boron) == 0
            assert not has_unresolved_two_electron_center(boron)


@pytest.mark.parametrize(
    ("atomic_num", "hydrogen_count", "initial_charge", "expected_charge"),
    [
        (8, 3, -1, 1),
        (6, 5, 0, -1),
        (9, 0, 0, 0),
    ],
)
def test_assign_charge_radical_charge_normalization_matches_python_formula(
    atomic_num: int,
    hydrogen_count: int,
    initial_charge: int,
    expected_charge: int,
) -> None:
    from molgr import _core  # type: ignore

    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        atom = obmol.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetFormalCharge(initial_charge)
        for _ in range(hydrogen_count):
            hydrogen = obmol.NewAtom()
            hydrogen.SetAtomicNum(1)
            obmol.AddBond(atom.GetIdx(), hydrogen.GetIdx(), 1)
        return pybel.Molecule(obmol)

    python_omol = make_omol()
    cpp_omol = make_omol()

    python_hit = assign_charge_radical_for_atom(python_omol.OBMol.GetAtom(1))
    cpp_hit = _core.dev.stages.fresh.assign_charge_radical_for_atom_ptr(_get_ptr(cpp_omol.OBMol), 1)

    def atom_signature(atom: ob.OBAtom) -> tuple[int, int, int, bool]:
        return (
            atom.GetFormalCharge(),
            get_unpaired_electron_count(atom),
            get_lone_pair_count(atom),
            has_unresolved_two_electron_center(atom),
        )

    python_signature = atom_signature(python_omol.OBMol.GetAtom(1))
    cpp_signature = atom_signature(cpp_omol.OBMol.GetAtom(1))
    assert python_signature[0] == expected_charge
    assert cpp_signature == python_signature
    assert cpp_hit is python_hit


@pytest.mark.parametrize(
    ("total_valence", "total_degree", "electron_count", "expected_occupancy"),
    [
        (4, 3, 2, (0, 1)),
        (3, 2, 4, (0, 2)),
        (1, 1, 5, (1, 2)),
        (0, 0, 2, (2, 0)),
    ],
)
def test_active_electron_occupancy_does_not_read_openbabel_hybridization(
    total_valence: int,
    total_degree: int,
    electron_count: int,
    expected_occupancy: tuple[int, int],
) -> None:
    class TopologyOnlyAtom:
        def GetTotalValence(self) -> int:
            return total_valence

        def GetTotalDegree(self) -> int:
            return total_degree

        def GetHyb(self) -> int:
            raise AssertionError("electronic-state inference must not read GetHyb()")

    assert (
        _infer_active_electron_occupancy(TopologyOnlyAtom(), electron_count)  # type: ignore[arg-type]
        == expected_occupancy
    )


@pytest.mark.parametrize(("atomic_num", "neighbor_count"), [(6, 2), (7, 1), (14, 2), (15, 1)])
def test_assign_charge_radical_defers_two_electron_center_occupancy(
    atomic_num: int,
    neighbor_count: int,
) -> None:
    from molgr import _core  # type: ignore

    def make_two_electron_center() -> pybel.Molecule:
        obmol = ob.OBMol()
        center = obmol.NewAtom()
        center.SetAtomicNum(atomic_num)
        for _ in range(neighbor_count):
            hydrogen = obmol.NewAtom()
            hydrogen.SetAtomicNum(1)
            obmol.AddBond(1, hydrogen.GetIdx(), 1)
        return pybel.Molecule(obmol)

    python_omol = make_two_electron_center()
    cpp_omol = make_two_electron_center()
    assign_charge_radical_for_atom(python_omol.OBMol.GetAtom(1))
    _core.dev.stages.fresh.assign_charge_radical_for_atom_ptr(_get_ptr(cpp_omol.OBMol), 1)

    for atom in (python_omol.OBMol.GetAtom(1), cpp_omol.OBMol.GetAtom(1)):
        assert (get_unpaired_electron_count(atom), get_lone_pair_count(atom)) == (0, 0)
        assert has_unresolved_two_electron_center(atom)


@pytest.mark.parametrize(
    ("symbol", "xyz_atoms"),
    [
        ("C", "C 0 0 0\nH 1.08 0 0\nH -0.54 0.935 0"),
        ("N", "N 0 0 0\nH 1.02 0 0"),
        ("P", "P 0 0 0\nH 1.42 0 0"),
        ("Si", "Si 0 0 0\nH 1.48 0 0\nH -0.74 1.282 0"),
    ],
)
@pytest.mark.parametrize(
    ("total_radical_electrons", "expected_occupancy"),
    [(0, (0, 1)), (2, (2, 0))],
)
def test_two_electron_centers_resolve_at_resonance_pool_for_python_and_cpp(
    symbol: str,
    xyz_atoms: str,
    total_radical_electrons: int,
    expected_occupancy: tuple[int, int],
) -> None:
    from molgr import _core  # type: ignore

    atom_count = len(xyz_atoms.splitlines())
    xyz_block = f"{atom_count}\n{symbol} two-electron center\n{xyz_atoms}\n"
    python_state = xyz_to_omol_no_metal_state(
        xyz_block,
        total_charge=0,
        total_radical_electrons=total_radical_electrons,
    )
    cpp_data = _core.pipeline.reconstruct_without_metals.xyz_to_omol_no_metal(
        xyz_block,
        0,
        total_radical_electrons,
    )

    assert python_state is not None
    python_center = python_state.omol.OBMol.GetAtom(1)
    assert (
        get_unpaired_electron_count(python_center),
        get_lone_pair_count(python_center),
    ) == expected_occupancy
    assert not has_unresolved_two_electron_center(python_center)
    assert (cpp_data.atoms[0].radical_num, cpp_data.atoms[0].lone_pair_count) == (
        expected_occupancy
    )
    assert not cpp_data.atoms[0].unresolved_two_electron_center


def test_clean_carbene_neighbor_unsaturated_consumes_unresolved_center_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "CC=C")
        center = omol.OBMol.GetAtom(1)
        set_unpaired_electron_count(center, 0)
        set_lone_pair_count(center, 0)
        set_unresolved_two_electron_center(center, True)
        return omol

    python_omol, hit = clean_carbene_neighbor_unsaturated(make_omol())
    assert hit
    assert not has_unresolved_two_electron_center(python_omol.OBMol.GetAtom(1))
    assert [get_unpaired_electron_count(python_omol.OBMol.GetAtom(i)) for i in (1, 3)] == [1, 1]

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    assert _core.dev.stages.clean.clean_carbene_neighbor_unsaturated_ptr(_get_ptr(cpp_omol.OBMol))
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(python_omol)
    assert not has_unresolved_two_electron_center(cpp_omol.OBMol.GetAtom(1))


def test_carbene_cleanup_requeries_paths_exposed_by_a_previous_move() -> None:
    def make_omol() -> pybel.Molecule:
        mol = ob.OBMol()
        mol.BeginModify()
        for _ in range(4):
            mol.NewAtom().SetAtomicNum(6)
        for begin, end, order in ((1, 2, 1), (2, 3, 1), (3, 4, 2)):
            mol.AddBond(begin, end, order)
        mol.EndModify()
        for index in (1, 2):
            set_unresolved_two_electron_center(mol.GetAtom(index), True)
        return pybel.Molecule(mol)

    python_omol, hit = clean_carbene_neighbor_unsaturated(make_omol())
    assert hit
    # Resolving 2-3=4 exposes 1-2=3, which was absent from the first match list.
    assert [b.GetBondOrder() for b in ob.OBMolBondIter(python_omol.OBMol)] == [2, 1, 1]
    assert [get_unpaired_electron_count(a.OBAtom) for a in python_omol] == [1, 1, 1, 1]
    assert not any(has_unresolved_two_electron_center(a.OBAtom) for a in python_omol)
    assert not clean_carbene_neighbor_unsaturated(python_omol)[1]

    from molgr import _core

    cpp_omol = make_omol()
    assert _core.dev.stages.clean.clean_carbene_neighbor_unsaturated_ptr(_get_ptr(cpp_omol.OBMol))
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(python_omol)


def test_eliminate_possible_cp_like_radical_anion_converts_excess_cyclopentadienyl_radical_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(5):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, 0)
        set_unpaired_electron_count(obmol.GetAtom(5), 1)
        obmol.AddBond(1, 2, 2)
        obmol.AddBond(2, 3, 1)
        obmol.AddBond(3, 4, 2)
        obmol.AddBond(4, 5, 1)
        obmol.AddBond(5, 1, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_possible_cp_like_radical_anion(omol, 0, 0)

    assert hit
    assert given_charge == 1
    assert _pybel_stage_signature(omol) == (
        ((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, -1, 0)),
        ((1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (5, 1, 1)),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = (
        _core.dev.stages.eliminate.eliminate_possible_cp_like_radical_anion_ptr(
            _get_ptr(cpp_omol.OBMol),
            0,
            0,
        )
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


@pytest.mark.parametrize(
    ("given_charge", "total_radical_electrons", "expected_hit"),
    [
        (0, 0, True),
        (1, 0, False),
        (0, 1, False),
    ],
)
def test_clean_possible_1_3_dipole_uses_global_electron_budget_for_python_and_cpp(
    given_charge: int,
    total_radical_electrons: int,
    expected_hit: bool,
) -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for atomic_num, unpaired in ((6, 1), (7, 0), (6, 1)):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(atomic_num)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, unpaired)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    before = _pybel_stage_signature(omol)
    omol, hit = clean_possible_1_3_dipole(
        omol,
        given_charge,
        total_radical_electrons,
    )

    assert hit is expected_hit
    if expected_hit:
        assert omol.OBMol.GetBond(1, 2).GetBondOrder() == 2
        assert omol.OBMol.GetAtom(2).GetFormalCharge() == 1
        assert omol.OBMol.GetAtom(3).GetFormalCharge() == -1
        assert sum(get_unpaired_electron_count(atom.OBAtom) for atom in omol) == 0
    else:
        assert _pybel_stage_signature(omol) == before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_possible_1_3_dipole_ptr(
        _get_ptr(cpp_omol.OBMol),
        given_charge,
        total_radical_electrons,
    )

    assert cpp_hit is expected_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_clean_possible_1_3_dipole_increases_lower_average_bond_order_side() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for atomic_num, unpaired in ((6, 1), (7, 0), (6, 1), (6, 0)):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(atomic_num)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, unpaired)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 1)
        obmol.AddBond(1, 4, 2)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, hit = clean_possible_1_3_dipole(omol, 0, 0)

    assert hit
    assert omol.OBMol.GetBond(1, 2).GetBondOrder() == 1
    assert omol.OBMol.GetBond(2, 3).GetBondOrder() == 2
    assert omol.OBMol.GetAtom(1).GetFormalCharge() == -1
    assert omol.OBMol.GetAtom(2).GetFormalCharge() == 1

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_possible_1_3_dipole_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
        0,
    )

    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_clean_possible_1_3_dipole_recomputes_budget_for_fragment_pool() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(2):
            for atomic_num, unpaired in ((6, 1), (7, 0), (6, 1)):
                atom = obmol.NewAtom()
                atom.SetAtomicNum(atomic_num)
                atom.SetFormalCharge(0)
                set_unpaired_electron_count(atom, unpaired)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 1)
        obmol.AddBond(4, 5, 1)
        obmol.AddBond(5, 6, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()

    omol, hit = clean_possible_1_3_dipole(omol, 0, 2)

    assert hit
    assert sum(atom.OBAtom.GetFormalCharge() == 1 for atom in omol) == 1
    assert sum(atom.OBAtom.GetFormalCharge() == -1 for atom in omol) == 1
    assert sum(get_unpaired_electron_count(atom.OBAtom) for atom in omol) == 2

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_possible_1_3_dipole_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
        2,
    )

    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_possible_cp_like_radical_anion_rejects_diradical_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(5):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, 0)
        set_unpaired_electron_count(obmol.GetAtom(5), 2)
        obmol.AddBond(1, 2, 2)
        obmol.AddBond(2, 3, 1)
        obmol.AddBond(3, 4, 2)
        obmol.AddBond(4, 5, 1)
        obmol.AddBond(5, 1, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    before = _pybel_stage_signature(omol)
    omol, given_charge, hit = eliminate_possible_cp_like_radical_anion(omol, -2, 0)

    assert not hit
    assert given_charge == -2
    assert _pybel_stage_signature(omol) == before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = (
        _core.dev.stages.eliminate.eliminate_possible_cp_like_radical_anion_ptr(
            _get_ptr(cpp_omol.OBMol),
            -2,
            0,
        )
    )

    assert not cpp_hit
    assert cpp_given_charge == -2
    assert _pybel_stage_signature(cpp_omol) == before


def test_eliminate_1_3_dipole_postive_rejects_diradical_endpoint_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for atomic_num, formal_charge, unpaired in ((6, -1, 0), (7, 0, 0), (6, 0, 2)):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(atomic_num)
            atom.SetFormalCharge(formal_charge)
            set_unpaired_electron_count(atom, unpaired)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 2)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    before = _pybel_stage_signature(omol)
    omol, given_charge, hit = eliminate_1_3_dipole_postive(omol, 1)

    assert not hit
    assert given_charge == 1
    assert _pybel_stage_signature(omol) == before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_1_3_dipole_postive_ptr(
        _get_ptr(cpp_omol.OBMol),
        1,
    )

    assert not cpp_hit
    assert cpp_given_charge == 1
    assert _pybel_stage_signature(cpp_omol) == before


@pytest.mark.parametrize(
    ("given_charge", "expected_charge", "expected_hit"),
    [
        (-1, -1, False),
        (0, -1, True),
    ],
)
def test_eliminate_1_3_dipole_postive_uses_nonnegative_budget_for_python_and_cpp(
    given_charge: int,
    expected_charge: int,
    expected_hit: bool,
) -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for atomic_num, formal_charge, unpaired in ((6, -1, 0), (7, 0, 0), (6, 0, 1)):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(atomic_num)
            atom.SetFormalCharge(formal_charge)
            set_unpaired_electron_count(atom, unpaired)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 2)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    before = _pybel_stage_signature(omol)
    omol, remaining_charge, hit = eliminate_1_3_dipole_postive(omol, given_charge)

    assert hit is expected_hit
    assert remaining_charge == expected_charge
    if expected_hit:
        assert _pybel_stage_signature(omol) != before
        assert omol.OBMol.GetAtom(2).GetFormalCharge() == 1
        assert get_unpaired_electron_count(omol.OBMol.GetAtom(3)) == 0
    else:
        assert _pybel_stage_signature(omol) == before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_1_3_dipole_postive_ptr(
        _get_ptr(cpp_omol.OBMol),
        given_charge,
    )

    assert cpp_hit is expected_hit
    assert cpp_charge == expected_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_1_3_dipole_postive_consumes_positive_budget_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(2):
            for atomic_num, formal_charge, unpaired in ((6, -1, 0), (7, 0, 0), (6, 0, 1)):
                atom = obmol.NewAtom()
                atom.SetAtomicNum(atomic_num)
                atom.SetFormalCharge(formal_charge)
                set_unpaired_electron_count(atom, unpaired)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 2)
        obmol.AddBond(4, 5, 1)
        obmol.AddBond(5, 6, 2)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_1_3_dipole_postive(omol, 1)

    assert hit
    assert given_charge == -1
    assert sum(atom.OBAtom.GetFormalCharge() == 1 for atom in omol) == 2
    assert sum(get_unpaired_electron_count(atom.OBAtom) for atom in omol) == 0

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_1_3_dipole_postive_ptr(
        _get_ptr(cpp_omol.OBMol),
        1,
    )

    assert cpp_hit
    assert cpp_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


@pytest.mark.parametrize(
    ("given_charge", "total_radical_electrons"),
    [(-1, 0), (0, 1)],
)
def test_eliminate_possible_cp_like_radical_anion_preserves_global_electron_budget_for_python_and_cpp(
    given_charge: int,
    total_radical_electrons: int,
) -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(5):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, 0)
        set_unpaired_electron_count(obmol.GetAtom(5), 1)
        obmol.AddBond(1, 2, 2)
        obmol.AddBond(2, 3, 1)
        obmol.AddBond(3, 4, 2)
        obmol.AddBond(4, 5, 1)
        obmol.AddBond(5, 1, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    before = _pybel_stage_signature(omol)
    omol, result_charge, hit = eliminate_possible_cp_like_radical_anion(
        omol,
        given_charge,
        total_radical_electrons,
    )

    assert not hit
    assert result_charge == given_charge
    assert _pybel_stage_signature(omol) == before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    before = _pybel_stage_signature(cpp_omol)
    cpp_given_charge, cpp_hit = (
        _core.dev.stages.eliminate.eliminate_possible_cp_like_radical_anion_ptr(
            _get_ptr(cpp_omol.OBMol),
            given_charge,
            total_radical_electrons,
        )
    )

    assert not cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == before


def test_eliminate_possible_cp_like_radical_anion_recomputes_global_electron_budget_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for ring_offset in (0, 5):
            for _ in range(5):
                atom = obmol.NewAtom()
                atom.SetAtomicNum(6)
                atom.SetFormalCharge(0)
                set_unpaired_electron_count(atom, 0)
            set_unpaired_electron_count(obmol.GetAtom(ring_offset + 5), 1)
            obmol.AddBond(ring_offset + 1, ring_offset + 2, 2)
            obmol.AddBond(ring_offset + 2, ring_offset + 3, 1)
            obmol.AddBond(ring_offset + 3, ring_offset + 4, 2)
            obmol.AddBond(ring_offset + 4, ring_offset + 5, 1)
            obmol.AddBond(ring_offset + 5, ring_offset + 1, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    def cp_status(omol: pybel.Molecule) -> tuple[int, int]:
        atom_signature = _pybel_stage_signature(omol)[0]
        negative_cp_atoms = sum(
            1
            for atom_idx, formal_charge, spin in atom_signature
            if atom_idx in (5, 10) and formal_charge == -1 and spin == 0
        )
        radical_cp_atoms = sum(
            1
            for atom_idx, formal_charge, spin in atom_signature
            if atom_idx in (5, 10) and formal_charge == 0 and spin == 1
        )
        return negative_cp_atoms, radical_cp_atoms

    omol = make_omol()
    omol, given_charge, hit = eliminate_possible_cp_like_radical_anion(omol, -1, 0)

    assert hit
    assert given_charge == 1
    assert cp_status(omol) == (2, 0)

    omol = make_omol()
    omol, given_charge, hit = eliminate_possible_cp_like_radical_anion(omol, -2, 0)

    assert not hit
    assert given_charge == -2
    assert cp_status(omol) == (0, 2)

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = (
        _core.dev.stages.eliminate.eliminate_possible_cp_like_radical_anion_ptr(
            _get_ptr(cpp_omol.OBMol),
            -1,
            0,
        )
    )

    assert cpp_hit
    assert cpp_given_charge == 1
    assert cp_status(cpp_omol) == (2, 0)

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = (
        _core.dev.stages.eliminate.eliminate_possible_cp_like_radical_anion_ptr(
            _get_ptr(cpp_omol.OBMol),
            -2,
            0,
        )
    )

    assert not cpp_hit
    assert cpp_given_charge == -2
    assert cp_status(cpp_omol) == (0, 2)


@pytest.mark.parametrize(
    ("given_charge", "radical_target", "existing_radicals", "expected_assignments"),
    [
        (-1, 1, 0, 1),
        (-2, 1, 0, 1),  # Radical budget runs out before negative charge does.
        (-2, 2, 0, 2),
        (-1, 0, 0, 0),
        (-1, None, 0, 0),  # Legacy calls do not authorize new radicals.
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (-1, 1, 1, 0),  # Existing radicals count against the global target.
        (-1, 2, 1, 1),
    ],
)
def test_bv3_negative_charge_assignment_obeys_both_budgets_for_python_and_cpp(
    given_charge: int,
    radical_target: int | None,
    existing_radicals: int,
    expected_assignments: int,
) -> None:
    from molgr import _core  # type: ignore

    for backend in ("python", "cpp"):
        mol = ob.OBMol()
        boron_indices = []
        for group in range(2):
            boron = mol.NewAtom()
            boron.SetAtomicNum(5)
            boron_indices.append(boron.GetIdx())
            if group == 0 and existing_radicals:
                boron.SetFormalCharge(-1)
                set_unpaired_electron_count(boron, 1)
            for _ in range(3):
                fluorine = mol.NewAtom()
                fluorine.SetAtomicNum(9)
                mol.AddBond(boron.GetIdx(), fluorine.GetIdx(), 1)
        omol = pybel.Molecule(mol)
        bonds_before = _pybel_stage_signature(omol)[1]
        if backend == "python":
            _, remaining, hit = eliminate_negative_charges(omol, given_charge, radical_target)
        else:
            remaining, hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
                _get_ptr(mol), given_charge, -1 if radical_target is None else radical_target
            )
        assert hit is bool(expected_assignments)
        assert remaining == given_charge + expected_assignments
        assert _pybel_stage_signature(omol)[1] == bonds_before
        borons = [mol.GetAtom(idx) for idx in boron_indices]
        assert sum(atom.GetFormalCharge() for atom in borons) == -(
            existing_radicals + expected_assignments
        )
        assert sum(get_unpaired_electron_count(atom) for atom in borons) == (
            existing_radicals + expected_assignments
        )
        for atom in borons:
            assert atom.GetFormalCharge() == -get_unpaired_electron_count(atom)
            assert get_lone_pair_count(atom) == 0
            assert not has_unresolved_two_electron_center(atom)


def test_eliminate_negative_charges_prefers_single_radical_heteroatom_even_at_zero_charge_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        atom = obmol.NewAtom()
        atom.SetAtomicNum(8)
        atom.SetFormalCharge(0)
        set_unpaired_electron_count(atom, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_negative_charges(omol, 0)

    assert hit
    assert given_charge == 1
    assert _pybel_stage_signature(omol) == (((1, -1, 0),), ())

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_negative_charges_rechecks_radical_target_after_reaching_zero_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(2):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(8)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_negative_charges(
        omol,
        -1,
        total_radical_electrons=1,
    )

    assert hit
    assert given_charge == 0
    assert _pybel_stage_signature(omol) == (
        ((1, -1, 0), (2, 0, 1)),
        (),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        -1,
        1,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_negative_charges_uses_ordered_smarts_before_atom_order_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        nitrogen = obmol.NewAtom()
        nitrogen.SetAtomicNum(7)
        nitrogen.SetFormalCharge(0)
        set_unpaired_electron_count(nitrogen, 1)
        oxygen = obmol.NewAtom()
        oxygen.SetAtomicNum(8)
        oxygen.SetFormalCharge(0)
        set_unpaired_electron_count(oxygen, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_negative_charges(omol, 0)

    assert hit
    assert given_charge == 1
    assert _pybel_stage_signature(omol) == (((1, 0, 1), (2, -1, 0)), ())

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_negative_charges_uses_same_action_for_double_radical_heteroatom_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        oxygen = obmol.NewAtom()
        oxygen.SetAtomicNum(8)
        oxygen.SetFormalCharge(0)
        set_unpaired_electron_count(oxygen, 2)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_negative_charges(omol, 0)

    assert hit
    assert given_charge == 1
    assert _pybel_stage_signature(omol) == (((1, -1, 1),), ())

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_negative_charges_matches_valence_specific_oxygen_smarts_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        oxygen = obmol.NewAtom()
        oxygen.SetAtomicNum(8)
        oxygen.SetFormalCharge(0)
        set_unpaired_electron_count(oxygen, 1)
        hydrogen = obmol.NewAtom()
        hydrogen.SetAtomicNum(1)
        hydrogen.SetFormalCharge(0)
        set_unpaired_electron_count(hydrogen, 0)
        obmol.AddBond(1, 2, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_negative_charges(omol, 0)

    assert hit
    assert given_charge == 1
    assert _pybel_stage_signature(omol) == (
        ((1, -1, 0), (2, 0, 0)),
        ((1, 2, 1),),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_negative_charges_leaves_cp_aromatic_action_to_resonance_stage_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(5):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, 0)
        set_unpaired_electron_count(obmol.GetAtom(5), 1)
        oxygen = obmol.NewAtom()
        oxygen.SetAtomicNum(8)
        oxygen.SetFormalCharge(0)
        set_unpaired_electron_count(oxygen, 1)
        obmol.AddBond(1, 2, 2)
        obmol.AddBond(2, 3, 1)
        obmol.AddBond(3, 4, 2)
        obmol.AddBond(4, 5, 1)
        obmol.AddBond(5, 1, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_negative_charges(omol, 0)

    assert hit
    assert given_charge == 1
    assert _pybel_stage_signature(omol)[0] == (
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (4, 0, 0),
        (5, 0, 1),
        (6, -1, 0),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_process_resonance_runs_cp_like_stage_before_positive_charge_assignment_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(5):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, 0)
        set_unpaired_electron_count(obmol.GetAtom(5), 1)
        carbon = obmol.NewAtom()
        carbon.SetAtomicNum(6)
        carbon.SetFormalCharge(0)
        set_unpaired_electron_count(carbon, 1)
        obmol.AddBond(1, 2, 2)
        obmol.AddBond(2, 3, 1)
        obmol.AddBond(3, 4, 2)
        obmol.AddBond(4, 5, 1)
        obmol.AddBond(5, 1, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    processed, given_charge, hit = resonance_utils.process_resonance(omol, -1)

    assert hit
    assert given_charge == 0
    assert _pybel_stage_signature(processed)[0] == (
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (4, 0, 0),
        (5, -1, 0),
        (6, 0, 1),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_res_ptr, cpp_charge = _core.dev.pipeline.resonance.process_resonance_ptr(
        _get_ptr(cpp_omol.OBMol),
        -1,
    )
    try:
        cpp_processed = _core.utils.extract_molecule_data(cpp_res_ptr)
        assert cpp_charge == given_charge
        assert tuple(
            (atom.formal_charge, atom.radical_num) for atom in cpp_processed.atoms
        ) == tuple(
            (atom.OBAtom.GetFormalCharge(), get_unpaired_electron_count(atom.OBAtom))
            for atom in processed
        )
    finally:
        _core.free_obmol_ptr(cpp_res_ptr)


def test_process_resonance_cleans_possible_1_3_dipole_before_positive_charge_assignment() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for atomic_num, unpaired in ((6, 1), (7, 0), (6, 1), (6, 1)):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(atomic_num)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, unpaired)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    processed, given_charge, hit = resonance_utils.process_resonance(
        omol,
        1,
        total_charge=1,
        total_radical_electrons=0,
    )

    assert hit
    assert given_charge == 0
    assert processed.OBMol.GetBond(1, 2).GetBondOrder() == 2
    assert tuple(atom.OBAtom.GetFormalCharge() for atom in processed) == (0, 1, -1, 1)
    assert sum(get_unpaired_electron_count(atom.OBAtom) for atom in processed) == 0

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_res_ptr, cpp_charge = _core.dev.pipeline.resonance.process_resonance_ptr(
        _get_ptr(cpp_omol.OBMol),
        1,
        1,
        0,
    )
    try:
        cpp_processed = _core.utils.extract_molecule_data(cpp_res_ptr)
        assert cpp_charge == given_charge
        assert tuple(
            (atom.formal_charge, atom.radical_num) for atom in cpp_processed.atoms
        ) == tuple(
            (atom.OBAtom.GetFormalCharge(), get_unpaired_electron_count(atom.OBAtom))
            for atom in processed
        )
        assert tuple(bond.order for bond in cpp_processed.bonds) == (2, 1)
    finally:
        _core.free_obmol_ptr(cpp_res_ptr)


def test_eliminate_negative_charges_prefers_carbon_before_hydrogen_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        carbon = obmol.NewAtom()
        carbon.SetAtomicNum(6)
        carbon.SetFormalCharge(0)
        set_unpaired_electron_count(carbon, 1)
        hydrogen = obmol.NewAtom()
        hydrogen.SetAtomicNum(1)
        hydrogen.SetFormalCharge(0)
        set_unpaired_electron_count(hydrogen, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_negative_charges(omol, -1)

    assert hit
    assert given_charge == 0
    assert _pybel_stage_signature(omol)[0] == ((1, -1, 0), (2, 0, 1))

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_negative_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        -1,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_positive_charges_prefers_nitrogen_motif_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        n1 = obmol.NewAtom()
        n1.SetAtomicNum(7)
        n1.SetFormalCharge(0)
        set_unpaired_electron_count(n1, 0)
        n2 = obmol.NewAtom()
        n2.SetAtomicNum(7)
        n2.SetFormalCharge(0)
        set_unpaired_electron_count(n2, 1)
        hydrogen = obmol.NewAtom()
        hydrogen.SetAtomicNum(1)
        hydrogen.SetFormalCharge(0)
        set_unpaired_electron_count(hydrogen, 0)
        carbon = obmol.NewAtom()
        carbon.SetAtomicNum(6)
        carbon.SetFormalCharge(0)
        set_unpaired_electron_count(carbon, 1)
        obmol.AddBond(1, 2, 2)
        obmol.AddBond(1, 3, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_positive_charges(omol, 1)

    assert hit
    assert given_charge == 0
    assert _pybel_stage_signature(omol)[0] == (
        (1, 0, 0),
        (2, 1, 0),
        (3, 0, 0),
        (4, 0, 1),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_positive_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        1,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_positive_charges_prefers_tier5_dipole_bond_closure_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        carbon = obmol.NewAtom()
        carbon.SetAtomicNum(6)
        carbon.SetFormalCharge(0)
        set_unpaired_electron_count(carbon, 1)
        nitrogen = obmol.NewAtom()
        nitrogen.SetAtomicNum(7)
        nitrogen.SetFormalCharge(0)
        set_unpaired_electron_count(nitrogen, 0)
        for _ in range(2):
            hydrogen = obmol.NewAtom()
            hydrogen.SetAtomicNum(1)
            hydrogen.SetFormalCharge(0)
            obmol.AddBond(2, hydrogen.GetIdx(), 1)
        obmol.AddBond(1, 2, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_positive_charges(omol, 0)
    assert not hit
    assert given_charge == 0

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_positive_charges_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert not cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_high_positive_charge_atoms_skips_overstabilized_match_and_keeps_later_match_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        a1 = obmol.NewAtom()
        a1.SetAtomicNum(15)
        a1.SetFormalCharge(1)
        a2 = obmol.NewAtom()
        a2.SetAtomicNum(8)
        a2.SetFormalCharge(0)
        set_unpaired_electron_count(a2, 1)
        a3 = obmol.NewAtom()
        a3.SetAtomicNum(8)
        a3.SetFormalCharge(-1)
        a4 = obmol.NewAtom()
        a4.SetAtomicNum(6)
        a4.SetFormalCharge(-1)
        a5 = obmol.NewAtom()
        a5.SetAtomicNum(15)
        a5.SetFormalCharge(1)
        a6 = obmol.NewAtom()
        a6.SetAtomicNum(8)
        a6.SetFormalCharge(0)
        set_unpaired_electron_count(a6, 1)
        a7 = obmol.NewAtom()
        a7.SetAtomicNum(6)
        a7.SetFormalCharge(0)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(1, 3, 1)
        obmol.AddBond(1, 4, 1)
        obmol.AddBond(5, 6, 1)
        obmol.AddBond(5, 7, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_high_positive_charge_atoms(omol, 0)

    assert hit
    assert given_charge == 1
    assert _pybel_stage_signature(omol) == (
        (
            (1, 1, 0),
            (2, 0, 1),
            (3, -1, 0),
            (4, -1, 0),
            (5, 1, 0),
            (6, -1, 0),
            (7, 0, 0),
        ),
        ((1, 2, 1), (1, 3, 1), (1, 4, 1), (5, 6, 1), (5, 7, 1)),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_high_positive_charge_atoms_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_high_positive_charge_atoms_skips_non_singlet_donor_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        a1 = obmol.NewAtom()
        a1.SetAtomicNum(15)
        a1.SetFormalCharge(1)
        a2 = obmol.NewAtom()
        a2.SetAtomicNum(8)
        a2.SetFormalCharge(0)
        set_unpaired_electron_count(a2, 0)
        a3 = obmol.NewAtom()
        a3.SetAtomicNum(6)
        a3.SetFormalCharge(0)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(1, 3, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_high_positive_charge_atoms(omol, 0)

    assert not hit
    assert given_charge == 0
    assert _pybel_stage_signature(omol) == (
        ((1, 1, 0), (2, 0, 0), (3, 0, 0)),
        ((1, 2, 1), (1, 3, 1)),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_high_positive_charge_atoms_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert not cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_high_positive_charge_atoms_uses_electronegativity_and_stops_at_neutrality() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        center = obmol.NewAtom()
        center.SetAtomicNum(16)
        center.SetFormalCharge(1)
        for atomic_num in (7, 8, 7):
            neighbor = obmol.NewAtom()
            neighbor.SetAtomicNum(atomic_num)
            set_unpaired_electron_count(neighbor, 1)
            obmol.AddBond(center.GetIdx(), neighbor.GetIdx(), 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol, given_charge, hit = eliminate_high_positive_charge_atoms(make_omol(), -1)

    assert hit
    assert given_charge == 0
    assert [omol.OBMol.GetAtom(idx).GetFormalCharge() for idx in (2, 3, 4)] == [0, -1, 0]
    assert [get_unpaired_electron_count(omol.OBMol.GetAtom(idx)) for idx in (2, 3, 4)] == [
        1,
        0,
        1,
    ]

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_high_positive_charge_atoms_ptr(
        _get_ptr(cpp_omol.OBMol),
        -1,
    )

    assert cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_eliminate_high_positive_charge_atoms_skips_pending_unresolved_neighbor_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        atom1 = obmol.NewAtom()
        atom1.SetAtomicNum(15)
        atom1.SetFormalCharge(1)
        atom2 = obmol.NewAtom()
        atom2.SetAtomicNum(8)
        atom2.SetFormalCharge(0)
        set_unpaired_electron_count(atom2, 1)
        neighbor = obmol.NewAtom()
        neighbor.SetAtomicNum(6)
        neighbor.SetFormalCharge(0)
        set_unresolved_two_electron_center(neighbor, True)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_high_positive_charge_atoms(omol, 0)

    assert not hit
    assert given_charge == 0
    assert get_unpaired_electron_count(omol.OBMol.GetAtom(2)) == 1

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_given_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_high_positive_charge_atoms_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert not cpp_hit
    assert cpp_given_charge == given_charge
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


@pytest.mark.parametrize(
    ("atomic_num", "expected_charge", "expected_unpaired"),
    [(15, 0, 1), (16, 0, 0), (17, 1, 0)],
)
def test_clean_resonances_14_refreshes_endpoint_state_for_python_and_cpp(
    atomic_num: int,
    expected_charge: int,
    expected_unpaired: int,
) -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        atom1 = obmol.NewAtom()
        atom1.SetAtomicNum(atomic_num)
        atom1.SetFormalCharge(-1)
        nitrogen = obmol.NewAtom()
        nitrogen.SetAtomicNum(7)
        nitrogen.SetFormalCharge(1)
        obmol.AddBond(1, 2, 3)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, hit = clean_resonances_14(omol)

    assert hit
    assert _pybel_stage_signature(omol) == (
        ((1, expected_charge, expected_unpaired), (2, 0, 1)),
        ((1, 2, 2),),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_resonances_14_ptr(_get_ptr(cpp_omol.OBMol))

    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_clean_resonances_3_preserves_unrelated_explicit_electron_state() -> None:
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "[N+]=CC=C[O-].[C]")
        detached = omol.OBMol.GetAtom(6)
        set_unpaired_electron_count(detached, 2)
        set_lone_pair_count(detached, 0)
        set_unresolved_two_electron_center(detached, False)
        return omol

    omol, hit = clean_resonances_3(make_omol())
    assert hit
    assert get_unpaired_electron_count(omol.OBMol.GetAtom(6)) == 2

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    assert _core.dev.stages.clean.clean_resonances_ptr(_get_ptr(cpp_omol.OBMol))
    assert get_unpaired_electron_count(cpp_omol.OBMol.GetAtom(6)) == 2


def test_clean_resonances_16_refreshes_endpoint_state_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        atom1 = obmol.NewAtom()
        atom1.SetAtomicNum(8)
        atom1.SetFormalCharge(-1)
        set_unpaired_electron_count(atom1, 0)
        atom2 = obmol.NewAtom()
        atom2.SetAtomicNum(6)
        atom2.SetFormalCharge(0)
        set_unpaired_electron_count(atom2, 0)
        atom3 = obmol.NewAtom()
        atom3.SetAtomicNum(6)
        atom3.SetFormalCharge(0)
        set_unpaired_electron_count(atom3, 0)
        atom4 = obmol.NewAtom()
        atom4.SetAtomicNum(6)
        atom4.SetFormalCharge(0)
        set_unpaired_electron_count(atom4, 0)
        atom5 = obmol.NewAtom()
        atom5.SetAtomicNum(7)
        atom5.SetFormalCharge(1)
        set_unpaired_electron_count(atom5, 0)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(2, 3, 2)
        obmol.AddBond(3, 4, 1)
        obmol.AddBond(4, 5, 2)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, hit = clean_resonances_16(omol)

    assert hit
    assert _pybel_stage_signature(omol) == (
        ((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0)),
        ((1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1)),
    )
    assert has_unresolved_two_electron_center(omol.OBMol.GetAtom(5))

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_resonances_16_ptr(_get_ptr(cpp_omol.OBMol))

    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)
    assert has_unresolved_two_electron_center(cpp_omol.OBMol.GetAtom(5))


def test_clean_resonances_17_converts_only_ring_allene_for_python_and_cpp() -> None:
    def make_ring_allene() -> pybel.Molecule:
        return pybel.readstring("smi", "[CH2-]C1=C=CCC1")

    expected_signature = (
        ((1, 0, 0), (2, 0, 0), (3, -1, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0)),
        (
            (1, 2, 2),
            (2, 3, 1),
            (2, 6, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 6, 1),
        ),
    )

    python_ring, python_hit = clean_resonances_17(make_ring_allene())
    assert python_hit
    assert _pybel_stage_signature(python_ring) == expected_signature

    acyclic = pybel.readstring("smi", "[CH2-]C=C=C")
    acyclic_before = _pybel_stage_signature(acyclic)
    acyclic, acyclic_hit = clean_resonances_17(acyclic)
    assert not acyclic_hit
    assert _pybel_stage_signature(acyclic) == acyclic_before

    from molgr import _core  # type: ignore

    cpp_ring = make_ring_allene()
    assert _core.dev.stages.clean.clean_resonances_17_ptr(_get_ptr(cpp_ring.OBMol))
    assert _pybel_stage_signature(cpp_ring) == expected_signature

    cpp_acyclic = pybel.readstring("smi", "[CH2-]C=C=C")
    cpp_acyclic_before = _pybel_stage_signature(cpp_acyclic)
    assert not _core.dev.stages.clean.clean_resonances_17_ptr(_get_ptr(cpp_acyclic.OBMol))
    assert _pybel_stage_signature(cpp_acyclic) == cpp_acyclic_before


def test_clean_resonances_18_converts_unresolved_diazene_to_azide_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "C[N]=N[N]")
        terminal = omol.OBMol.GetAtom(4)
        set_unpaired_electron_count(terminal, 0)
        set_lone_pair_count(terminal, 0)
        set_unresolved_two_electron_center(terminal, True)
        return omol

    expected_signature = (
        ((1, 0, 0), (2, 0, 0), (3, 1, 0), (4, -1, 0)),
        ((1, 2, 1), (2, 3, 2), (3, 4, 2)),
    )

    omol, hit = clean_resonances_18(make_omol())
    assert hit
    assert _pybel_stage_signature(omol) == expected_signature
    assert not has_unresolved_two_electron_center(omol.OBMol.GetAtom(4))

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_resonances_18_ptr(_get_ptr(cpp_omol.OBMol))
    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == expected_signature
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_clean_resonances_0_does_not_reuse_stale_symmetric_match() -> None:
    def make_omol() -> pybel.Molecule:
        return pybel.readstring("smi", "[O-]C(=C([O+])[O-])[O+]")

    omol, hit = clean_resonances_0(make_omol())
    assert hit
    assert _pybel_stage_signature(omol) == (
        ((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, -1, 0), (6, 1, 0)),
        ((1, 2, 2), (2, 3, 1), (2, 6, 1), (3, 4, 2), (3, 5, 1)),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    assert _core.dev.stages.clean.clean_resonances_ptr(_get_ptr(cpp_omol.OBMol))
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_clean_resonances_2_does_not_overvalent_tetracoordinate_boron_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for atomic_num, formal_charge in (
            (8, 0),
            (6, 0),
            (9, 0),
            (6, 0),
            (7, 0),
            (5, -1),
            (9, 0),
            (7, 0),
            (7, 0),
        ):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(atomic_num)
            atom.SetFormalCharge(formal_charge)
        for begin, end, order in (
            (1, 2, 2),
            (2, 3, 1),
            (2, 4, 1),
            (4, 5, 2),
            (5, 6, 1),
            (6, 7, 1),
            (6, 8, 1),
            (6, 9, 1),
        ):
            obmol.AddBond(begin, end, order)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    python_omol = make_omol()
    python_before = _pybel_stage_signature(python_omol)
    python_omol, python_hit = clean_resonances_2(python_omol)
    python_boron = python_omol.OBMol.GetAtom(6)

    assert python_hit is False
    assert python_boron.GetFormalCharge() == -1
    assert python_boron.GetTotalValence() == 4
    assert _pybel_stage_signature(python_omol) == python_before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_before = _pybel_stage_signature(cpp_omol)
    cpp_hit = _core.dev.stages.clean.clean_resonances_ptr(_get_ptr(cpp_omol.OBMol))
    cpp_boron = cpp_omol.OBMol.GetAtom(6)

    assert cpp_hit is False
    assert cpp_boron.GetFormalCharge() == -1
    assert cpp_boron.GetTotalValence() == 4
    assert _pybel_stage_signature(cpp_omol) == cpp_before


def test_clean_resonances_7_does_not_overvalent_tetracoordinate_boron_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for atomic_num, formal_charge in (
            (5, -1),
            (6, 0),
            (6, 0),
            (6, 0),
            (6, 0),
            (6, 0),
            (6, 0),
            (9, 0),
            (9, 0),
        ):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(atomic_num)
            atom.SetFormalCharge(formal_charge)
        for begin, end, order in (
            (1, 2, 1),
            (2, 3, 2),
            (2, 4, 1),
            (4, 5, 2),
            (5, 6, 1),
            (6, 7, 2),
            (7, 1, 1),
            (1, 8, 1),
            (1, 9, 1),
        ):
            obmol.AddBond(begin, end, order)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    python_omol = make_omol()
    python_before = _pybel_stage_signature(python_omol)
    python_omol, python_hit = clean_resonances_7(python_omol)

    assert python_hit is False
    assert python_omol.OBMol.GetAtom(1).GetFormalCharge() == -1
    assert python_omol.OBMol.GetAtom(1).GetTotalValence() == 4
    assert _pybel_stage_signature(python_omol) == python_before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_before = _pybel_stage_signature(cpp_omol)
    cpp_hit = _core.dev.stages.clean.clean_resonances_ptr(_get_ptr(cpp_omol.OBMol))

    assert cpp_hit is False
    assert cpp_omol.OBMol.GetAtom(1).GetFormalCharge() == -1
    assert cpp_omol.OBMol.GetAtom(1).GetTotalValence() == 4
    assert _pybel_stage_signature(cpp_omol) == cpp_before


def test_clean_neighbor_radicals_refreshes_atom_state_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(2):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(5)
            atom.SetFormalCharge(-1)
            set_unpaired_electron_count(atom, 1)
        obmol.AddBond(1, 2, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, hit = clean_neighbor_radicals(omol, 0, 0)

    assert hit
    assert _pybel_stage_signature(omol) == (
        ((1, -1, 0), (2, -1, 0)),
        ((1, 2, 2),),
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_neighbor_radicals_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
        0,
    )

    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_clean_neighbor_radicals_resolves_radical_next_to_unresolved_center_for_python_and_cpp() -> (
    None
):
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "CC")
        radical = omol.OBMol.GetAtom(1)
        unresolved = omol.OBMol.GetAtom(2)
        set_unpaired_electron_count(radical, 1)
        set_lone_pair_count(radical, 0)
        set_unresolved_two_electron_center(radical, False)
        set_unpaired_electron_count(unresolved, 0)
        set_lone_pair_count(unresolved, 0)
        set_unresolved_two_electron_center(unresolved, True)
        return omol

    omol, hit = clean_neighbor_radicals(make_omol(), 0, 0)
    assert hit
    assert omol.OBMol.GetBond(1, 2).GetBondOrder() == 2
    assert get_unpaired_electron_count(omol.OBMol.GetAtom(1)) == 0
    assert get_unpaired_electron_count(omol.OBMol.GetAtom(2)) == 1
    assert not has_unresolved_two_electron_center(omol.OBMol.GetAtom(2))

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_neighbor_radicals_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
        0,
    )
    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)
    assert not has_unresolved_two_electron_center(cpp_omol.OBMol.GetAtom(2))


@pytest.mark.parametrize(
    ("total_radical_electrons", "expected_bond_order", "expected_unpaired"),
    [
        (0, 3, (0, 0)),
        (1, 2, (1, 1)),
        (4, 1, (0, 0)),
    ],
)
def test_clean_neighbor_radicals_uses_unresolved_center_budget_for_python_and_cpp(
    total_radical_electrons: int,
    expected_bond_order: int,
    expected_unpaired: tuple[int, int],
) -> None:
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "CC")
        for atom in omol:
            set_unpaired_electron_count(atom.OBAtom, 0)
            set_lone_pair_count(atom.OBAtom, 0)
            set_unresolved_two_electron_center(atom.OBAtom, True)
        return omol

    omol, hit = clean_neighbor_radicals(make_omol(), 0, total_radical_electrons)
    assert hit is (total_radical_electrons != 4)
    assert omol.OBMol.GetBond(1, 2).GetBondOrder() == expected_bond_order
    assert tuple(get_unpaired_electron_count(atom.OBAtom) for atom in omol) == expected_unpaired
    for atom in omol:
        assert get_lone_pair_count(atom.OBAtom) == 0
        assert has_unresolved_two_electron_center(atom.OBAtom) is (total_radical_electrons == 4)

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_neighbor_radicals_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
        total_radical_electrons,
    )
    assert cpp_hit is (total_radical_electrons != 4)
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)
    assert cpp_omol.OBMol.GetBond(1, 2).GetBondOrder() == expected_bond_order
    assert tuple(get_unpaired_electron_count(atom.OBAtom) for atom in cpp_omol) == expected_unpaired
    for atom in cpp_omol:
        assert get_lone_pair_count(atom.OBAtom) == 0
        assert has_unresolved_two_electron_center(atom.OBAtom) is (total_radical_electrons == 4)


@pytest.mark.parametrize(
    ("given_charge", "total_radical_electrons", "expected_hit"),
    [
        (0, 0, True),
        (1, 0, False),
        (0, 1, False),
    ],
)
def test_clean_neighbor_radicals_uses_global_electron_budget_for_python_and_cpp(
    given_charge: int,
    total_radical_electrons: int,
    expected_hit: bool,
) -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(2):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(5)
            atom.SetFormalCharge(-1)
            set_unpaired_electron_count(atom, 1)
        obmol.AddBond(1, 2, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    before = _pybel_stage_signature(omol)
    omol, hit = clean_neighbor_radicals(
        omol,
        given_charge,
        total_radical_electrons,
    )

    assert hit is expected_hit
    if expected_hit:
        assert omol.OBMol.GetBond(1, 2).GetBondOrder() == 2
        assert sum(get_unpaired_electron_count(atom.OBAtom) for atom in omol) == 0
    else:
        assert _pybel_stage_signature(omol) == before

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_neighbor_radicals_ptr(
        _get_ptr(cpp_omol.OBMol),
        given_charge,
        total_radical_electrons,
    )

    assert cpp_hit is expected_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


@pytest.mark.parametrize("endpoint_state", ["radicals", "unresolved", "mixed"])
def test_clean_1_4_radicals_resolves_separated_endpoint_states_for_python_and_cpp(
    endpoint_state: str,
) -> None:
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "CCCC")
        obmol = omol.OBMol
        obmol.GetBond(2, 3).SetBondOrder(2)
        states = {
            "radicals": ("r", "r"),
            "unresolved": ("m", "m"),
            "mixed": ("r", "m"),
        }[endpoint_state]
        for idx, state in zip((1, 4), states):
            atom = obmol.GetAtom(idx)
            if state in ("r", "m"):
                set_unpaired_electron_count(atom, 1 if state == "r" else 0)
                set_lone_pair_count(atom, 0)
                set_unresolved_two_electron_center(atom, state == "m")
        return omol

    omol, hit = clean_1_4_radicals(make_omol(), 0, 0)
    assert hit
    assert [omol.OBMol.GetBond(i, i + 1).GetBondOrder() for i in range(1, 4)] == [2, 1, 2]
    assert all(
        get_unpaired_electron_count(omol.OBMol.GetAtom(idx)) == 0
        and not has_unresolved_two_electron_center(omol.OBMol.GetAtom(idx))
        for idx in (1, 4)
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_1_4_radicals_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
        0,
    )
    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_clean_1_4_radicals_respects_global_electron_budget() -> None:
    omol = pybel.readstring("smi", "CCCC")
    omol.OBMol.GetBond(2, 3).SetBondOrder(2)
    for idx in (1, 4):
        set_unpaired_electron_count(omol.OBMol.GetAtom(idx), 1)
    before = _pybel_stage_signature(omol)
    omol, hit = clean_1_4_radicals(omol, 0, 1)
    assert not hit
    assert _pybel_stage_signature(omol) == before


@pytest.mark.parametrize("endpoint_state", ["radicals", "unresolved", "mixed"])
def test_clean_1_6_radicals_resolves_separated_endpoint_states_for_python_and_cpp(
    endpoint_state: str,
) -> None:
    def make_omol() -> pybel.Molecule:
        omol = pybel.readstring("smi", "CCCCCC")
        obmol = omol.OBMol
        obmol.GetBond(2, 3).SetBondOrder(2)
        obmol.GetBond(4, 5).SetBondOrder(2)
        states = {
            "radicals": ("r", "r"),
            "unresolved": ("m", "m"),
            "mixed": ("r", "m"),
        }[endpoint_state]
        for idx, state in zip((1, 6), states):
            atom = obmol.GetAtom(idx)
            set_unpaired_electron_count(atom, 1 if state == "r" else 0)
            set_lone_pair_count(atom, 0)
            set_unresolved_two_electron_center(atom, state == "m")
        return omol

    omol, hit = clean_1_6_radicals(make_omol(), 0, 0)
    assert hit
    assert [omol.OBMol.GetBond(i, i + 1).GetBondOrder() for i in range(1, 6)] == [
        2,
        1,
        2,
        1,
        2,
    ]
    assert all(
        get_unpaired_electron_count(omol.OBMol.GetAtom(idx)) == 0
        and not has_unresolved_two_electron_center(omol.OBMol.GetAtom(idx))
        for idx in (1, 6)
    )

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    cpp_hit = _core.dev.stages.clean.clean_1_6_radicals_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
        0,
    )
    assert cpp_hit
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_clean_1_6_radicals_respects_global_electron_budget() -> None:
    omol = pybel.readstring("smi", "CCCCCC")
    omol.OBMol.GetBond(2, 3).SetBondOrder(2)
    omol.OBMol.GetBond(4, 5).SetBondOrder(2)
    for idx in (1, 6):
        set_unpaired_electron_count(omol.OBMol.GetAtom(idx), 1)
    before = _pybel_stage_signature(omol)
    omol, hit = clean_1_6_radicals(omol, 0, 1)
    assert not hit
    assert _pybel_stage_signature(omol) == before


@pytest.mark.parametrize("length", [1.35, 1.48])
def test_enumerate_neighbor_radical_seeds_resolves_neighbor_radicals(length: float) -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for _ in range(2):
            atom = obmol.NewAtom()
            atom.SetAtomicNum(8)
            atom.SetFormalCharge(0)
            set_unpaired_electron_count(atom, 1)
        obmol.GetAtom(1).SetVector(0, 0, 0)
        obmol.GetAtom(2).SetVector(length, 0, 0)
        obmol.AddBond(1, 2, 1)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    state = ReconstructionState(
        omol=make_omol(),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz",),
    )

    candidates = neighbor_radical_module.enumerate_neighbor_radical_seeds(state)

    from molgr import _core  # type: ignore

    xyz_block = f"2\nOO\nO 0 0 0\nO {length} 0 0\n"
    cpp_candidates = _core.dev.pipeline.reconstruct_without_metals.debug_neighbor_radical_seeds(
        xyz_block,
        0,
        0,
    )
    if length >= 1.40:
        # Both backends defer the same elongated O2 seed without losing electrons.
        assert len(candidates) == len(cpp_candidates) == 1
        assert candidates[0].metadata["neighbor_radical_resolution"] == "none"
        assert cpp_candidates[0]["neighbor_radical_resolution"] == "none"
        assert _pybel_stage_signature(candidates[0].omol) == _pybel_stage_signature(state.omol)
        assert (
            neighbor_radical_module.enumerate_neighbor_radical_seeds(state, exact_discrepancy=1)
            == []
        )
        assert (
            _core.dev.pipeline.reconstruct_without_metals.debug_neighbor_radical_seeds(
                xyz_block, 0, 0, 1
            )
            == []
        )
        return

    assert len(candidates) == 3
    assert [candidate.metadata["neighbor_radical_resolution"] for candidate in candidates] == [
        "bond_order",
        "charge_separation",
        "charge_separation",
    ]
    assert [_pybel_stage_signature(candidate.omol) for candidate in candidates] == [
        (((1, 0, 0), (2, 0, 0)), ((1, 2, 2),)),
        (((1, 1, 0), (2, -1, 0)), ((1, 2, 1),)),
        (((1, -1, 0), (2, 1, 0)), ((1, 2, 1),)),
    ]

    assert [candidate["neighbor_radical_resolution"] for candidate in cpp_candidates] == [
        "bond_order",
        "charge_separation",
        "charge_separation",
    ]
    assert [candidate.get("positive_atom_idx") for candidate in cpp_candidates] == [None, 1, 2]
    assert [candidate.get("neighbor_radical_actions") for candidate in cpp_candidates] == [
        "bond_order:1-2",
        "charge_separation:1+2-",
        "charge_separation:2+1-",
    ]
    assert (
        len(
            _core.dev.pipeline.reconstruct_without_metals.debug_neighbor_radical_seeds(
                xyz_block,
                0,
                0,
                0,
            )
        )
        == len(
            neighbor_radical_module.enumerate_neighbor_radical_seeds(
                state,
                exact_discrepancy=0,
            )
        )
        == 1
    )
    assert (
        len(
            _core.dev.pipeline.reconstruct_without_metals.debug_neighbor_radical_seeds(
                xyz_block,
                0,
                0,
                1,
            )
        )
        == len(
            neighbor_radical_module.enumerate_neighbor_radical_seeds(
                state,
                exact_discrepancy=1,
            )
        )
        == 2
    )
    assert (
        _core.dev.pipeline.reconstruct_without_metals.debug_neighbor_radical_seeds(
            xyz_block,
            0,
            0,
            2,
        )
        == neighbor_radical_module.enumerate_neighbor_radical_seeds(
            state,
            exact_discrepancy=2,
        )
        == []
    )


def test_eliminate_carbene_neighbor_heteroatom_skips_radical_neighbors_for_python_and_cpp() -> None:
    def make_omol() -> pybel.Molecule:
        obmol = ob.OBMol()
        obmol.BeginModify()
        blocked_carbon = obmol.NewAtom()
        blocked_carbon.SetAtomicNum(6)
        blocked_carbon.SetFormalCharge(0)
        set_unresolved_two_electron_center(blocked_carbon, True)
        radical_neighbor = obmol.NewAtom()
        radical_neighbor.SetAtomicNum(6)
        radical_neighbor.SetFormalCharge(0)
        set_unpaired_electron_count(radical_neighbor, 1)
        blocked_oxygen = obmol.NewAtom()
        blocked_oxygen.SetAtomicNum(8)
        blocked_oxygen.SetFormalCharge(0)
        set_unpaired_electron_count(blocked_oxygen, 0)
        carbonyl_carbon = obmol.NewAtom()
        carbonyl_carbon.SetAtomicNum(6)
        carbonyl_carbon.SetFormalCharge(0)
        set_unresolved_two_electron_center(carbonyl_carbon, True)
        carbonyl_oxygen = obmol.NewAtom()
        carbonyl_oxygen.SetAtomicNum(8)
        carbonyl_oxygen.SetFormalCharge(0)
        set_unpaired_electron_count(carbonyl_oxygen, 0)
        set_lone_pair_count(carbonyl_oxygen, 1)
        obmol.AddBond(1, 2, 1)
        obmol.AddBond(1, 3, 1)
        obmol.AddBond(4, 5, 2)
        obmol.EndModify()
        return pybel.Molecule(obmol)

    omol = make_omol()
    omol, given_charge, hit = eliminate_carbene_neighbor_heteroatom(omol, 0)

    assert hit
    assert given_charge == 0
    assert _pybel_stage_signature(omol) == (
        ((1, 0, 0), (2, 0, 1), (3, 0, 0), (4, -1, 0), (5, 1, 0)),
        ((1, 2, 1), (1, 3, 1), (4, 5, 3)),
    )
    assert has_unresolved_two_electron_center(omol.OBMol.GetAtom(1))
    assert get_lone_pair_count(omol.OBMol.GetAtom(4)) == 0
    assert get_lone_pair_count(omol.OBMol.GetAtom(5)) == 0
    assert not has_unresolved_two_electron_center(omol.OBMol.GetAtom(4))

    from molgr import _core  # type: ignore

    cpp_omol = make_omol()
    given_charge, hit = _core.dev.stages.eliminate.eliminate_carbene_neighbor_heteroatom_ptr(
        _get_ptr(cpp_omol.OBMol),
        0,
    )

    assert hit
    assert given_charge == 0
    assert _pybel_stage_signature(cpp_omol) == _pybel_stage_signature(omol)


def test_recovery_tiers_pass_current_charge_into_break_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xyz_block = """2
H2
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""
    state = ReconstructionState(
        omol=pybel.readstring("xyz", xyz_block),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=3,
        phase_history=("read_xyz",),
    )
    state.given_charge = 7
    recorded: dict[str, tuple[int, ...]] = {}

    def record_break_deformed_ene(omol, given_charge, total_radical_electrons, tolerance):
        recorded["break_deformed_ene"] = (given_charge, total_radical_electrons, tolerance)
        set_unpaired_electron_count(omol.OBMol.GetAtom(1), 1)
        return omol, True

    def record_break_one_bond(omol, given_charge, total_radical_electrons):
        recorded["break_one_bond"] = (given_charge, total_radical_electrons)
        set_unpaired_electron_count(omol.OBMol.GetAtom(2), 2)
        return omol, given_charge, True

    monkeypatch.setattr(no_metal_recovery_module, "break_deformed_ene", record_break_deformed_ene)
    monkeypatch.setattr(no_metal_recovery_module, "break_one_bond", record_break_one_bond)
    deformed = no_metal_recovery_module.enumerate_deformed_pi_recovery_seeds([state])
    broken = no_metal_recovery_module.enumerate_bond_break_recovery_seeds([state])

    assert recorded["break_deformed_ene"] == (7, 3, 5.0)
    assert recorded["break_one_bond"] == (7, 3)
    assert deformed[0].given_charge == 7
    assert broken[0].given_charge == 7
    assert get_unpaired_electron_count(deformed[0].omol.OBMol.GetAtom(1)) == 1
    assert get_unpaired_electron_count(broken[0].omol.OBMol.GetAtom(2)) == 2
    assert "refresh_electronic_labels_after_recovery" not in deformed[0].phase_history
    assert "refresh_electronic_labels_after_recovery" not in broken[0].phase_history


def test_fallback_no_metal_reuses_resonance_score_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    no_metal_module._run_no_metal_pipeline_cached.cache_clear()

    base_state = ReconstructionState(
        omol=pybel.readstring("smi", "CC"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz",),
    )
    weaker_candidate = ReconstructionState(
        omol=pybel.readstring("smi", "C=CC=C"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz", "validate_resonance_candidate"),
        metadata={"score": 5.0},
    )
    stronger_candidate = ReconstructionState(
        omol=pybel.readstring("smi", "C=CC=C"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz", "validate_resonance_candidate"),
        metadata={"score": 2.0},
    )

    monkeypatch.setattr(
        no_metal_preparation_module, "_seed_state", lambda *args, **kwargs: base_state
    )
    monkeypatch.setattr(
        no_metal_module,
        "_run_linear_preparation",
        lambda state: state,
    )
    monkeypatch.setattr(
        neighbor_radical_module,
        "enumerate_neighbor_radical_seeds",
        lambda state, **kwargs: [state],
    )
    monkeypatch.setattr(
        no_metal_resonance_module,
        "build_resonance_seed_pool",
        lambda states: list(states),
    )
    monkeypatch.setattr(
        no_metal_resonance_module,
        "search_resonance_candidates",
        lambda states, **kwargs: [weaker_candidate, stronger_candidate],
    )
    monkeypatch.setattr(
        no_metal_selection_module,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: float(candidate.metadata["score"]),
    )

    result = xyz_to_omol_no_metal_state("cached-resonance", 0, 0)

    assert result is not None
    assert result.metadata["score"] == 2.0
    assert result.phase_history[-1] == "select_best_no_metal_candidate"


def test_no_metal_pipeline_searches_all_neighbor_seeds_in_one_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_metal_module._run_no_metal_pipeline_cached.cache_clear()

    base_state = ReconstructionState(
        omol=pybel.readstring("smi", "CC"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz",),
    )
    branch_states = [
        ReconstructionState(
            omol=pybel.readstring("smi", "CC"),
            given_charge=0,
            total_charge=0,
            total_radical_electrons=0,
            phase_history=("read_xyz", "branch_0"),
            metadata={"branch": 0, "priority": 0},
        ),
        ReconstructionState(
            omol=pybel.readstring("smi", "C=C"),
            given_charge=0,
            total_charge=0,
            total_radical_electrons=0,
            phase_history=("read_xyz", "branch_1"),
            metadata={"branch": 1, "priority": 10},
        ),
        ReconstructionState(
            omol=pybel.readstring("smi", "c1ccccc1"),
            given_charge=0,
            total_charge=0,
            total_radical_electrons=0,
            phase_history=("read_xyz", "branch_2"),
            metadata={"branch": 2, "priority": 20},
        ),
    ]
    resonance_map = {
        0: [
            ReconstructionState(
                omol=pybel.readstring("smi", "C=C"),
                given_charge=0,
                total_charge=0,
                total_radical_electrons=0,
                phase_history=("read_xyz", "validate_resonance_candidate"),
                metadata={"branch": 0, "priority": 1},
            )
        ],
        1: [
            ReconstructionState(
                omol=pybel.readstring("smi", "C=C"),
                given_charge=0,
                total_charge=0,
                total_radical_electrons=0,
                phase_history=("read_xyz", "validate_resonance_candidate"),
                metadata={"branch": 1, "priority": 2},
            )
        ],
        2: [
            ReconstructionState(
                omol=pybel.readstring("smi", "c1ccccc1"),
                given_charge=0,
                total_charge=0,
                total_radical_electrons=0,
                phase_history=("read_xyz", "validate_resonance_candidate"),
                metadata={"branch": 2, "priority": 3},
            )
        ],
    }
    resonance_calls: list[int] = []

    monkeypatch.setattr(
        no_metal_preparation_module, "_seed_state", lambda *args, **kwargs: base_state
    )
    monkeypatch.setattr(
        no_metal_module,
        "_run_linear_preparation",
        lambda state: state,
    )
    monkeypatch.setattr(
        neighbor_radical_module,
        "enumerate_neighbor_radical_seeds",
        lambda state, **kwargs: branch_states,
    )
    monkeypatch.setattr(
        no_metal_resonance_module,
        "build_resonance_seed_pool",
        lambda states: list(states),
    )

    def search_all(states, **kwargs):
        del kwargs
        resonance_calls.extend(int(state.metadata["branch"]) for state in states)
        return [
            candidate for state in states for candidate in resonance_map[state.metadata["branch"]]
        ]

    monkeypatch.setattr(
        no_metal_resonance_module,
        "search_resonance_candidates",
        search_all,
    )
    monkeypatch.setattr(
        no_metal_selection_module,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: float(candidate.metadata["priority"]),
    )
    monkeypatch.setattr(
        no_metal_selection_module,
        "_no_metal_candidate_selection_key",
        lambda candidate, **kwargs: (
            float(candidate.metadata["priority"]),
            0,
            0,
            0,
            0,
            0,
        ),
    )

    result = xyz_to_omol_no_metal_state("all-branches-resonance", 0, 0)

    assert result is not None
    assert resonance_calls == [0, 1, 2]
    assert result.phase_history[-1] == "select_best_no_metal_candidate"
    assert result.metadata["branch"] == 0
    assert result.metadata["priority"] == 1


def test_unified_resonance_pool_isolates_seed_from_normalization_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obmol = ob.OBMol()
    obmol.BeginModify()
    atom = obmol.NewAtom()
    atom.SetAtomicNum(7)
    atom.SetFormalCharge(0)
    set_unpaired_electron_count(atom, 1)
    obmol.EndModify()
    seed_state = ReconstructionState(
        omol=pybel.Molecule(obmol),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=1,
        phase_history=("read_xyz", "seed"),
        metadata={"priority": 0},
    )

    def mutate_candidate(omol, remaining_charge):
        atom = omol.OBMol.GetAtom(1)
        atom.SetFormalCharge(-1)
        set_unpaired_electron_count(atom, 0)
        return omol, remaining_charge, True

    def walk_resonances(omol, *, visit, **kwargs):
        del kwargs
        visit(
            resonance_utils.ResonanceSearchNode(
                omol,
                resonance_utils.build_resonance_state_key(omol),
                0,
                0,
            )
        )

    monkeypatch.setattr(
        no_metal_resonance_module,
        "walk_radical_resonances",
        walk_resonances,
    )
    monkeypatch.setattr(
        no_metal_resonance_module.resonance_utils,
        "process_resonance",
        mutate_candidate,
    )
    monkeypatch.setattr(no_metal_resonance_module, "validate_omol", lambda *args: True)
    monkeypatch.setattr(
        no_metal_resonance_module,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        no_metal_resonance_module,
        "_annotate_no_metal_candidate_topology",
        lambda candidate, **kwargs: None,
    )

    candidates = no_metal_resonance_module.search_resonance_candidates([seed_state])

    assert candidates
    source_atom = seed_state.omol.OBMol.GetAtom(1)
    assert source_atom.GetFormalCharge() == 0
    assert get_unpaired_electron_count(source_atom) == 1


def test_no_metal_resonance_selection_prefers_aromatic_topology_before_force_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_metal_module._run_no_metal_pipeline_cached.cache_clear()

    base_state = ReconstructionState(
        omol=pybel.readstring("smi", "CC"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz",),
    )
    lower_force_field_candidate = ReconstructionState(
        omol=pybel.readstring("smi", "C=CC=C"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz", "validate_resonance_candidate"),
        metadata={"score": 1.0},
    )
    aromatic_candidate = ReconstructionState(
        omol=pybel.readstring("smi", "c1ccccc1"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz", "validate_resonance_candidate"),
        metadata={"score": 10.0},
    )

    monkeypatch.setattr(
        no_metal_preparation_module, "_seed_state", lambda *args, **kwargs: base_state
    )
    monkeypatch.setattr(
        no_metal_module,
        "_run_linear_preparation",
        lambda state: state,
    )
    monkeypatch.setattr(
        neighbor_radical_module,
        "enumerate_neighbor_radical_seeds",
        lambda state, **kwargs: [state],
    )
    monkeypatch.setattr(
        no_metal_resonance_module,
        "build_resonance_seed_pool",
        lambda states: list(states),
    )
    monkeypatch.setattr(
        no_metal_resonance_module,
        "search_resonance_candidates",
        lambda states, **kwargs: [lower_force_field_candidate, aromatic_candidate],
    )
    monkeypatch.setattr(
        no_metal_selection_module,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: float(candidate.metadata["score"]),
    )

    result = xyz_to_omol_no_metal_state("topology-first", 0, 0)

    assert result is not None
    assert result.omol.write("can").strip() == aromatic_candidate.omol.write("can").strip()
    assert result.metadata["organic_aromatic_atom_count"] == 6
    assert result.metadata["organic_aromatic_stability_score"] == pytest.approx(1.0)
    assert result.metadata["organic_topology_selection_key"][:9] == (
        0,
        0,
        0,
        -6,
        -1,
        -1.0,
        -6,
        -6,
        -6,
    )
    assert result.phase_history[-1] == "select_best_no_metal_candidate"


def test_no_metal_resonance_selection_scores_conjugation_and_charge_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_metal_module._run_no_metal_pipeline_cached.cache_clear()

    base_state = ReconstructionState(
        omol=pybel.readstring("smi", "CC"),
        given_charge=0,
        total_charge=-2,
        total_radical_electrons=0,
        phase_history=("read_xyz",),
    )
    lower_force_field_candidate = ReconstructionState(
        omol=pybel.readstring(
            "smi",
            "c1ccc(nc1)c1ccccn1.Oc1cc(O)c2c(c1)oc(c(c2=O)[O-])c1ccc(c(c1)O)O.[Cl-]",
        ),
        given_charge=0,
        total_charge=-2,
        total_radical_electrons=0,
        phase_history=("read_xyz", "validate_resonance_candidate"),
        metadata={"score": 1168.9043528762083},
    )
    charge_separated_candidate = ReconstructionState(
        omol=pybel.readstring(
            "smi",
            "c1ccc(nc1)c1ccccn1.[OH+]=c1cc2-c(c(c1)O)c(c(c(o2)c1ccc(c(c1)O)O)[O-])[O-].[Cl-]",
        ),
        given_charge=0,
        total_charge=-2,
        total_radical_electrons=0,
        phase_history=("read_xyz", "validate_resonance_candidate"),
        metadata={"score": 1226.1886293507894},
    )

    monkeypatch.setattr(
        no_metal_preparation_module, "_seed_state", lambda *args, **kwargs: base_state
    )
    monkeypatch.setattr(
        no_metal_module,
        "_run_linear_preparation",
        lambda state: state,
    )
    monkeypatch.setattr(
        neighbor_radical_module,
        "enumerate_neighbor_radical_seeds",
        lambda state, **kwargs: [state],
    )
    monkeypatch.setattr(
        no_metal_resonance_module,
        "build_resonance_seed_pool",
        lambda states: list(states),
    )
    monkeypatch.setattr(
        no_metal_resonance_module,
        "search_resonance_candidates",
        lambda states, **kwargs: [lower_force_field_candidate, charge_separated_candidate],
    )
    monkeypatch.setattr(
        no_metal_selection_module,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: float(candidate.metadata["score"]),
    )

    result = xyz_to_omol_no_metal_state("charge-penalized", -2, 0)

    assert result is not None
    assert result.metadata["score"] == pytest.approx(1168.9043528762083)
    assert result.metadata["organic_formal_charge_absolute_sum"] == 2
    assert result.metadata["organic_conjugation_charge_penalty"] == pytest.approx(1.0)
    assert result.omol.write("can").strip() == lower_force_field_candidate.omol.write("can").strip()
    # Neutral O lone pairs participate without an artificial oxonium charge.
    assert (
        charge_separated_candidate.metadata["organic_max_conjugated_component_size"]
        == (lower_force_field_candidate.metadata["organic_max_conjugated_component_size"])
    )
    assert (
        charge_separated_candidate.metadata["organic_conjugated_atom_count"]
        == (lower_force_field_candidate.metadata["organic_conjugated_atom_count"])
    )
    assert (
        charge_separated_candidate.metadata["organic_conjugated_bond_count"]
        == (lower_force_field_candidate.metadata["organic_conjugated_bond_count"])
    )
    assert charge_separated_candidate.metadata["organic_conjugation_charge_penalty"] == (
        lower_force_field_candidate.metadata["organic_conjugation_charge_penalty"] + 1.0
    )
    assert (
        lower_force_field_candidate.metadata["organic_adjusted_max_conjugated_component_size"]
        == charge_separated_candidate.metadata["organic_adjusted_max_conjugated_component_size"] + 1
    )
    assert (
        lower_force_field_candidate.metadata["organic_adjusted_conjugated_atom_count"]
        == charge_separated_candidate.metadata["organic_adjusted_conjugated_atom_count"] + 1
    )
    assert (
        lower_force_field_candidate.metadata["organic_adjusted_conjugated_bond_count"]
        == charge_separated_candidate.metadata["organic_adjusted_conjugated_bond_count"] + 1
    )
    assert lower_force_field_candidate.metadata["organic_topology_selection_key"][1] == 2
    assert charge_separated_candidate.metadata["organic_topology_selection_key"][1] == 4
    assert (
        lower_force_field_candidate.metadata["organic_topology_selection_key"]
        < charge_separated_candidate.metadata["organic_topology_selection_key"]
    )
    assert result.phase_history[-1] == "select_best_no_metal_candidate"


def test_no_metal_selection_uses_hyperconjugation_after_stronger_electronic_metrics() -> None:
    more_hyperconjugated = ReconstructionState(
        omol=pybel.readstring("smi", "CC=CC"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={"score": 20.0},
    )
    less_hyperconjugated = ReconstructionState(
        omol=pybel.readstring("smi", "C=CCC"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={"score": 10.0},
    )

    result = no_metal_selection_module.select_best_no_metal_candidate(
        [less_hyperconjugated, more_hyperconjugated]
    )

    assert result is more_hyperconjugated
    assert more_hyperconjugated.metadata["organic_hyperconjugation_score"] == 6
    assert less_hyperconjugated.metadata["organic_hyperconjugation_score"] == 2
    assert more_hyperconjugated.metadata["organic_topology_selection_key"][-3:] == (0, -6, 20.0)
    assert less_hyperconjugated.metadata["organic_topology_selection_key"][-3:] == (0, -2, 10.0)


def test_no_metal_cached_pipeline_uses_config_in_key_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_metal_module._run_no_metal_pipeline_cached.cache_clear()

    base_state = ReconstructionState(
        omol=pybel.readstring("smi", "CC"),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("read_xyz",),
    )
    default_config = MolGRConfig()
    config_a = replace(
        default_config,
        resonance=replace(default_config.resonance, max_depth=2),
    )
    config_b = replace(
        default_config,
        resonance=replace(default_config.resonance, max_depth=3),
    )
    seen_configs = []

    monkeypatch.setattr(
        no_metal_preparation_module, "_seed_state", lambda *args, **kwargs: base_state
    )

    def fake_run_from_state(seed_state: ReconstructionState, *, config=None):
        assert seed_state is base_state
        seen_configs.append(config)
        return base_state

    monkeypatch.setattr(no_metal_module, "_run_no_metal_pipeline_from_state", fake_run_from_state)

    first = xyz_to_omol_no_metal_state("cfg-key", 0, 0, config=config_a)
    second = xyz_to_omol_no_metal_state("cfg-key", 0, 0, config=config_a)
    third = xyz_to_omol_no_metal_state("cfg-key", 0, 0, config=config_b)

    assert first is base_state
    assert second is base_state
    assert third is base_state
    assert seen_configs == [config_a, config_b]


@pytest.mark.parametrize("length,height,expected", [(1.40, 0.0, 1), (1.46, 0.0, 0), (1.40, 0.8, 0)])
def test_azinium_preference_requires_compact_planar_geometry(length, height, expected):
    import math

    from molgr.fallback.utils.bond_geometry import azinium_geometry_conflict_count

    mol = pybel.readstring("smi", "CN1C=C[CH-]C=C1").OBMol
    ids = list(mol.GetSSSR()[0]._path)
    for i, idx in enumerate(ids):
        angle = i * math.pi / 3
        mol.GetAtom(idx).SetVector(
            length * math.cos(angle), length * math.sin(angle), height if i == 0 else 0
        )
    assert azinium_geometry_conflict_count(mol) == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("substituted", True),
        ("isolated_no", False),
        ("hydroxyl", False),
        ("closed_shell", False),
        ("double_bond", False),
    ],
)
def test_unclosed_n_o_seed_scope_preserves_terminal_no_and_hydroxyl(mode, expected):
    mol = ob.OBMol()
    nitrogen, oxygen = mol.NewAtom(), mol.NewAtom()
    nitrogen.SetAtomicNum(7)
    oxygen.SetAtomicNum(8)
    for atom in (nitrogen, oxygen):
        atom.SetImplicitHCount(0)
        set_unpaired_electron_count(atom, 0 if mode == "closed_shell" else 1)
    mol.AddBond(1, 2, 2 if mode == "double_bond" else 1)
    if mode != "isolated_no":
        carbon = mol.NewAtom()
        carbon.SetAtomicNum(6)
        mol.AddBond(1, carbon.GetIdx(), 1)
    if mode == "hydroxyl":
        hydrogen = mol.NewAtom()
        hydrogen.SetAtomicNum(1)
        mol.AddBond(2, hydrogen.GetIdx(), 1)
    assert neighbor_radical_module.has_substituted_n_o_radical_pair(pybel.Molecule(mol)) == expected


def test_bent_allene_guard_preserves_linear_and_non_ring_centers():
    from molgr.fallback.utils.bond_geometry import would_form_bent_ring_allene

    mol = pybel.readstring("smi", "C1=CCCCC1").OBMol
    center, double_neighbor, partner = (mol.GetAtom(i) for i in (1, 2, 6))
    center.SetVector(0, 0, 0)
    double_neighbor.SetVector(1, 0, 0)
    partner.SetVector(-0.5, 0.866, 0)
    assert would_form_bent_ring_allene(center, partner)
    partner.SetVector(-1, 0, 0)
    assert not would_form_bent_ring_allene(center, partner)
    chain = pybel.readstring("smi", "C=CC").OBMol
    assert not would_form_bent_ring_allene(chain.GetAtom(2), chain.GetAtom(3))


@pytest.mark.parametrize(
    "atomic_num,valence,can_donate",
    [
        (7, 3, True),
        (8, 2, True),
        (15, 3, True),
        (15, 5, False),
        (16, 2, True),
        (16, 4, True),
        (16, 6, False),
    ],
)
def test_carbene_donor_requires_remaining_valence_electrons(atomic_num, valence, can_donate):
    """Neutral hypervalent atoms must not invent an additional donor pair."""
    from molgr import _core

    def make_molecule():
        mol = ob.OBMol()
        carbon = mol.NewAtom()
        carbon.SetAtomicNum(6)
        donor = mol.NewAtom()
        donor.SetAtomicNum(atomic_num)
        mol.AddBond(1, 2, 1)
        for parent in [1] + [2] * (valence - 1):
            hydrogen = mol.NewAtom()
            hydrogen.SetAtomicNum(1)
            mol.AddBond(parent, hydrogen.GetIdx(), 1)
        result = pybel.Molecule(mol)
        fresh_omol_charge_radical(result)
        assert has_unresolved_two_electron_center(carbon)
        # Exercise the donor rule on neutral closed-shell Lewis states,
        # independently of fresh's preferred sulfur charge normalization.
        donor.SetFormalCharge(0)
        assert donor.GetFormalCharge() == 0
        assert get_unpaired_electron_count(donor) == 0
        return result

    python_mol = make_molecule()
    before = _pybel_stage_signature(python_mol)
    python_mol, charge, hit = eliminate_carbene_neighbor_heteroatom(python_mol, 0)
    cpp_mol = make_molecule()
    cpp_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_carbene_neighbor_heteroatom_ptr(
        _get_ptr(cpp_mol.OBMol), 0
    )
    assert hit == cpp_hit == can_donate
    assert charge == cpp_charge == 0
    assert _pybel_stage_signature(python_mol) == _pybel_stage_signature(cpp_mol)
    if can_donate:
        assert python_mol.OBMol.GetBond(1, 2).GetBondOrder() == 2
        assert python_mol.OBMol.GetAtom(1).GetFormalCharge() == -1
        assert python_mol.OBMol.GetAtom(2).GetFormalCharge() == 1
        Chem.SanitizeMol(pybel_to_rdmol(python_mol))
    else:
        assert _pybel_stage_signature(python_mol) == before
        assert has_unresolved_two_electron_center(python_mol.OBMol.GetAtom(1))


@pytest.mark.parametrize("chalcogen", [8, 16], ids=["CO", "CS"])
@pytest.mark.parametrize("seed_order", [1, 2, 3])
def test_isolated_co_cs_share_closed_shell_donor_path(chalcogen, seed_order):
    """Seed bond perception must not decide whether neutral CO/CS can close."""
    from molgr import _core

    def make_molecule():
        mol = ob.OBMol()
        for number in [6, chalcogen]:
            mol.NewAtom().SetAtomicNum(number)
        mol.AddBond(1, 2, seed_order)
        return pybel.Molecule(mol)

    python_mol = make_molecule()
    cpp_mol = make_molecule()
    pre_clean(python_mol)
    _core.dev.stages.preprocess.pre_clean_ptr(_get_ptr(cpp_mol.OBMol))
    assert python_mol.OBMol.GetBond(1, 2).GetBondOrder() == 2
    assert _pybel_stage_signature(python_mol) == _pybel_stage_signature(cpp_mol)
    assert not pre_clean(python_mol)[1]
    assert not _core.dev.stages.preprocess.pre_clean_ptr(_get_ptr(cpp_mol.OBMol))
    fresh_omol_charge_radical(python_mol)
    _core.dev.stages.fresh.fresh_omol_charge_radical_ptr(_get_ptr(cpp_mol.OBMol))
    python_mol, charge, hit = eliminate_carbene_neighbor_heteroatom(python_mol, 0)
    cpp_charge, cpp_hit = _core.dev.stages.eliminate.eliminate_carbene_neighbor_heteroatom_ptr(
        _get_ptr(cpp_mol.OBMol), 0
    )
    assert hit and cpp_hit and charge == cpp_charge == 0
    assert _pybel_stage_signature(python_mol) == _pybel_stage_signature(cpp_mol)
    result = pybel_to_rdmol(python_mol)
    assert result.GetNumAtoms() == 2
    assert Chem.GetFormalCharge(result) == 0
    assert [a.GetFormalCharge() for a in result.GetAtoms()] == [-1, 1]
    assert sum(a.GetNumRadicalElectrons() for a in result.GetAtoms()) == 0
    assert result.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.TRIPLE
    assert all(a.GetNumImplicitHs() == 0 for a in result.GetAtoms())


def _phosphoranide_geometry(kind="seesaw", *, count=1, family="cf"):
    """Four explicit ligand neighbors with independently prescribed geometry."""
    import math

    mol = ob.OBMol()
    for fragment in range(count):
        p = mol.NewAtom()
        p.SetAtomicNum(15)
        p.SetFormalCharge(1)
        p.SetImplicitHCount(0)
        offset = 10.0 * fragment
        p.SetVector(offset, 0.0, 0.0)
        half = math.radians(50)
        vectors = [
            (0.0, 0.0, 1.7),
            (0.0, 0.0, -1.7),
            (1.7 * math.cos(half), 1.7 * math.sin(half), 0.0),
            (1.7 * math.cos(half), -1.7 * math.sin(half), 0.0),
        ]
        if kind == "tetrahedral":
            scale = 1.7 / math.sqrt(3)
            vectors = [
                (scale * x, scale * y, scale * z)
                for x, y, z in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
            ]
        elif kind in ("axial_164", "axial_166"):
            angle = math.radians(int(kind.split("_")[1]))
            vectors[1] = (1.7 * math.sin(angle), 0.0, 1.7 * math.cos(angle))
        elif kind == "zero_length":
            vectors[0] = (0.0, 0.0, 0.0)
        elif kind == "nonfinite":
            vectors[0] = (float("nan"), 0.0, 1.7)
        elif kind == "long_bond":
            vectors[0] = (0.0, 0.0, 2.0)
        elif kind == "equatorial_150":
            angle = math.radians(75)
            vectors[2:] = [
                (1.7 * math.cos(angle), sign * 1.7 * math.sin(angle), 0.0) for sign in (1, -1)
            ]
        for i, vector in enumerate(vectors):
            n = mol.NewAtom()
            elements = [9, 9, 9, 9] if family == "cf" else [8, 8, 7, 7]
            if kind == "no_axial_pair":
                elements = [6, 6, 9, 9] if family == "cf" else [7, 7, 8, 8]
            elif kind == "all_nitrogen":
                elements = [7] * 4
            elif kind == "mixed_axial":
                elements[1] = 6 if family == "cf" else 7
            elif kind == "mixed_neighbors":
                elements[0] = 8 if family == "cf" else 9
            n.SetAtomicNum(elements[i])
            n.SetImplicitHCount(0)
            n.SetVector(vector[0] + offset, vector[1], vector[2])
            mol.AddBond(p.GetIdx(), n.GetIdx(), 2 if kind == "multiple_bond" and i == 0 else 1)
        if kind == "implicit_h":
            p.SetImplicitHCount(1)
    return pybel.Molecule(mol)


@pytest.mark.parametrize(
    "kind,allowed",
    [
        ("seesaw", True),
        ("mixed_axial", None),
        ("all_nitrogen", True),
        ("axial_166", True),
        ("axial_164", False),
        ("tetrahedral", False),
        ("zero_length", False),
        ("nonfinite", False),
        ("long_bond", False),
        ("equatorial_150", False),
        ("no_axial_pair", False),
        ("mixed_neighbors", False),
        ("multiple_bond", False),
        ("implicit_h", False),
    ],
)
@pytest.mark.parametrize("family", ["cf", "on"])
def test_phosphoranide_geometry_and_refresh_parity(kind, allowed, family):
    from molgr import _core
    from molgr.fallback.utils.bond_geometry import allows_seesaw_phosphoranide

    if kind == "mixed_axial":
        allowed = family == "cf"
    python_mol = _phosphoranide_geometry(kind, family=family)
    p = python_mol.OBMol.GetAtom(1)
    assert allows_seesaw_phosphoranide(p) == allowed
    # Recognition must not overwrite the ordinary cation hypothesis.
    assign_charge_radical_for_atom(p)
    if kind not in ("multiple_bond", "implicit_h"):
        assert p.GetFormalCharge() == 1
    # An explicitly explored anion survives only within the admitted scope.
    p.SetFormalCharge(-1)
    set_unpaired_electron_count(p, 2)
    set_lone_pair_count(p, 1)
    set_unresolved_two_electron_center(p, True)
    cpp_mol = pybel.Molecule(ob.OBMol(python_mol.OBMol))
    python_hit = assign_charge_radical_for_atom(p)
    cpp_hit = _core.dev.stages.fresh.assign_charge_radical_for_atom_ptr(_get_ptr(cpp_mol.OBMol), 1)
    assert cpp_hit == python_hit
    assert _pybel_stage_signature(cpp_mol) == _pybel_stage_signature(python_mol)
    if allowed:
        assert p.GetFormalCharge() == -1
        assert get_unpaired_electron_count(p) == get_lone_pair_count(p) == 0
        assert not has_unresolved_two_electron_center(p)
    elif kind not in ("multiple_bond", "implicit_h"):
        assert p.GetFormalCharge() == 1


@pytest.mark.parametrize("count,expected", [(1, 2), (2, 4), (9, 256)])
@pytest.mark.parametrize("family", ["cf", "on"])
def test_geometry_electronic_seeds_preserve_charge_budget_and_original(count, expected, family):
    omol = _phosphoranide_geometry(count=count, family=family)
    state = ReconstructionState(omol, -2 * count, -count, 0)
    before = _pybel_stage_signature(omol)
    seeds = no_metal_preparation_module.expand_geometry_electronic_seeds(state)
    assert len(seeds) == expected and seeds[0] is state
    assert _pybel_stage_signature(omol) == before
    assert len({tuple(a.OBAtom.GetFormalCharge() for a in s.omol) for s in seeds}) == expected
    for seed in seeds:
        assert seed.given_charge + sum(a.OBAtom.GetFormalCharge() for a in seed.omol) == -count
        assert seed.total_radical_electrons == 0
    if count <= 2:
        assert any(s.given_charge == 0 for s in seeds)


@pytest.mark.parametrize("charge", [-1, 1])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_phosphoranide_exploration_respects_global_charge(charge, backend):
    from molgr.interface import xyz_to_rdmol

    omol = _phosphoranide_geometry()
    xyz = omol.write("xyz")
    result = xyz_to_rdmol(xyz, total_charge=charge, spin_multiplicity=1, backend=backend)
    assert result.GetNumAtoms() == 5 and result.GetNumBonds() == 4
    assert Chem.GetFormalCharge(result) == charge
    assert result.GetAtomWithIdx(0).GetFormalCharge() == charge
    assert all(a.GetNumRadicalElectrons() == 0 and a.GetNoImplicit() for a in result.GetAtoms())
    assert all(b.GetBondType() == Chem.BondType.SINGLE for b in result.GetBonds())


def test_cpp_phosphoranide_diagnostic_includes_production_branch():
    from molgr import _core

    xyz = _phosphoranide_geometry().write("xyz")
    summaries = _core.dev.pipeline.reconstruct_without_metals.debug_resonance_candidate_summaries(
        xyz, -1, 0, config=MolGRConfig()
    )
    assert summaries
    assert any("[P-]" in summary["smiles"] for summary in summaries)


@pytest.mark.parametrize("reverse_atoms", [False, True])
@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize("family", ["cf", "on"])
def test_phosphoranide_geometry_is_frame_and_atom_order_independent(
    reverse_atoms, transform, family
):
    from molgr import _core
    from molgr.fallback.utils.bond_geometry import allows_seesaw_phosphoranide

    source = _phosphoranide_geometry(family=family).OBMol
    for idx in (4, 5):
        carbon = source.GetAtom(idx)
        carbon.SetAtomicNum(6 if family == "cf" else 7)
        length = 1.93 if family == "cf" else 1.75
        carbon.SetVector(carbon.GetX() * length / 1.7, carbon.GetY() * length / 1.7, 0.0)
    mol = ob.OBMol()
    mapping = {}
    for idx in range(5, 0, -1) if reverse_atoms else range(1, 6):
        old = source.GetAtom(idx)
        atom = mol.NewAtom()
        atom.SetAtomicNum(old.GetAtomicNum())
        atom.SetImplicitHCount(0)
        x, y, z = old.GetX(), old.GetY(), old.GetZ()
        atom.SetVector(-z + 12.0, y - 3.0, x + 5.0) if transform else atom.SetVector(x, y, z)
        mapping[idx] = atom.GetIdx()
    for idx in range(2, 6):
        mol.AddBond(mapping[1], mapping[idx], 1)
    p = mol.GetAtom(mapping[1])
    assert allows_seesaw_phosphoranide(p)
    p.SetFormalCharge(-1)
    _core.dev.stages.fresh.assign_charge_radical_for_atom_ptr(_get_ptr(mol), p.GetIdx())
    assert p.GetFormalCharge() == -1
