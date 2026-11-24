from __future__ import annotations

from dataclasses import replace

import pytest

from molgr.config import MolGRConfig
from molgr.fallback.utils.electrons import (
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.organic_topology import (
    _additional_atom_pi_electrons,
    compute_organic_topology_metrics,
)


def _require_pybel():
    pytest.importorskip("openbabel")
    return pytest.importorskip("openbabel.pybel")


def test_continuous_conjugation_metrics_count_only_three_bond_alternation() -> None:
    pybel = _require_pybel()
    benzene = pybel.readstring("smi", "c1ccccc1")
    diene = pybel.readstring("smi", "C=CC=C")

    benzene_metrics = compute_organic_topology_metrics(benzene)
    diene_metrics = compute_organic_topology_metrics(diene)

    assert benzene_metrics.aromatic_atom_count == 6
    assert benzene_metrics.aromatic_ring_count == 1
    assert benzene_metrics.aromatic_stability_score == pytest.approx(1.0)
    assert benzene_metrics.conjugated_atom_count == 6
    assert benzene_metrics.conjugated_bond_count == 6
    assert benzene_metrics.max_conjugated_component_size == 6

    assert diene_metrics.aromatic_atom_count == 0
    assert diene_metrics.aromatic_ring_count == 0
    assert diene_metrics.aromatic_stability_score == pytest.approx(0.0)
    assert diene_metrics.conjugated_atom_count == 4
    assert diene_metrics.conjugated_bond_count == 3
    assert diene_metrics.max_conjugated_component_size == 4


def test_cumulated_double_bonds_do_not_join_orthogonal_pi_systems() -> None:
    pybel = _require_pybel()
    allene = pybel.readstring("smi", "C=C=C")
    agodeg_co1_organic = pybel.readstring("smi", "S=C1C=CC=C2C1=C=CC=C2")
    adinor_w2_organic = pybel.readstring("smi", "C=C=CC(=O)OCC")
    adinor_w4_organic = pybel.readstring("smi", "C=[C-]/C=C(\\[O-])/OCC")

    allene_metrics = compute_organic_topology_metrics(allene)
    agodeg_metrics = compute_organic_topology_metrics(agodeg_co1_organic)
    adinor_w2_metrics = compute_organic_topology_metrics(adinor_w2_organic)
    adinor_w4_metrics = compute_organic_topology_metrics(adinor_w4_organic)

    assert allene_metrics.conjugated_atom_count == 0
    assert allene_metrics.conjugated_bond_count == 0
    assert allene_metrics.max_conjugated_component_size == 0
    assert agodeg_metrics.aromatic_ring_count == 0
    assert agodeg_metrics.conjugated_atom_count == 10
    assert agodeg_metrics.conjugated_bond_count == 10
    assert agodeg_metrics.max_conjugated_component_size == 10
    # The ester O supplies a Lewis pair in both representations. It extends
    # the ester-side system without joining the orthogonal terminal C=C.
    assert adinor_w2_metrics.conjugated_atom_count == 5
    assert adinor_w2_metrics.conjugated_bond_count == 4
    assert adinor_w2_metrics.max_conjugated_component_size == 5
    assert 0 not in adinor_w2_metrics.conjugated_atom_indices
    assert adinor_w4_metrics.conjugated_atom_count == 6
    assert adinor_w4_metrics.conjugated_bond_count == 5
    assert adinor_w4_metrics.max_conjugated_component_size == 6


def test_diene_conjugation_requires_outer_substituents_to_be_coplanar() -> None:
    pybel = _require_pybel()

    def make_diene(*, twisted: bool, scale: float = 1.0):
        molecule = pybel.readstring("smi", "CC=CC=CC")
        coordinates = (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.3, 0.0, 0.0),
            (2.7, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 0.0, 1.0) if twisted else (4.0, 1.0, 0.0),
        )
        for atom_idx, coordinate in enumerate(coordinates, start=1):
            molecule.OBMol.GetAtom(atom_idx).SetVector(
                *(component * scale for component in coordinate)
            )
        return molecule

    planar = make_diene(twisted=False)
    twisted = make_diene(twisted=True)
    scaled_twisted = make_diene(twisted=True, scale=0.5)

    planar_python_metrics = compute_organic_topology_metrics(planar)
    twisted_python_metrics = compute_organic_topology_metrics(twisted)
    scaled_twisted_python_metrics = compute_organic_topology_metrics(scaled_twisted)

    from molgr import _core  # type: ignore

    planar_cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(
        int(planar.OBMol.this)
    )
    twisted_cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(
        int(twisted.OBMol.this)
    )
    scaled_twisted_cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(
        int(scaled_twisted.OBMol.this)
    )

    assert planar_python_metrics.conjugated_atom_count == 4
    assert planar_python_metrics.conjugated_bond_count == 3
    assert planar_python_metrics.max_conjugated_component_size == 4
    assert twisted_python_metrics.conjugated_atom_count == 0
    assert twisted_python_metrics.conjugated_bond_count == 0
    assert twisted_python_metrics.max_conjugated_component_size == 0
    assert scaled_twisted_python_metrics == twisted_python_metrics
    assert planar_cpp_metrics["conjugated_atom_count"] == 4
    assert planar_cpp_metrics["conjugated_bond_count"] == 3
    assert planar_cpp_metrics["max_conjugated_component_size"] == 4
    assert twisted_cpp_metrics["conjugated_atom_count"] == 0
    assert twisted_cpp_metrics["conjugated_bond_count"] == 0
    assert twisted_cpp_metrics["max_conjugated_component_size"] == 0
    assert scaled_twisted_cpp_metrics == twisted_cpp_metrics


def test_terminal_alkene_hydrogen_participates_in_conjugation_geometry_check() -> None:
    pybel = _require_pybel()
    from openbabel import openbabel as ob

    molecule = ob.OBMol()
    atomic_numbers = (1, 6, 6, 6, 6, 8)
    coordinates = (
        (-0.9671, -1.2935, 2.5769),
        (-0.6182, -1.6115, 1.5756),
        (0.5161, -0.9751, 1.0027),
        (1.8656, -0.9816, 1.0784),
        (2.7278, -0.1498, 0.2193),
        (2.3680, 0.6880, -0.6013),
    )
    for atomic_number, coordinate in zip(atomic_numbers, coordinates):
        atom = molecule.NewAtom()
        atom.SetAtomicNum(atomic_number)
        atom.SetVector(*coordinate)
    molecule.AddBond(1, 2, 1)
    molecule.AddBond(2, 3, 2)
    molecule.AddBond(3, 4, 1)
    molecule.AddBond(4, 5, 2)
    molecule.AddBond(5, 6, 1)
    omol = pybel.Molecule(molecule)

    python_metrics = compute_organic_topology_metrics(omol)

    from molgr import _core  # type: ignore

    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(omol.OBMol.this))

    assert python_metrics.conjugated_atom_count == 0
    assert python_metrics.conjugated_bond_count == 0
    assert python_metrics.max_conjugated_component_size == 0
    assert cpp_metrics["conjugated_atom_count"] == 0
    assert cpp_metrics["conjugated_bond_count"] == 0
    assert cpp_metrics["max_conjugated_component_size"] == 0


def test_allylic_charge_center_extends_conjugated_topology() -> None:
    pybel = _require_pybel()
    alternating = pybel.readstring("smi", "[O-]/C(=C\\C(=O)C(F)(F)F)/C(F)(F)F")
    charge_separated = pybel.readstring("smi", "O=C([CH+]C(=O)C(F)(F)F)C(F)(F)F")

    alternating_metrics = compute_organic_topology_metrics(alternating)
    charge_separated_metrics = compute_organic_topology_metrics(charge_separated)

    assert charge_separated_metrics.conjugated_atom_count == 5
    assert charge_separated_metrics.conjugated_bond_count == 4
    assert charge_separated_metrics.max_conjugated_component_size == 5
    assert alternating_metrics.conjugated_atom_count >= 5


@pytest.mark.parametrize("smiles", ["C=C[CH2+]", "C=C[CH2-]"])
def test_allylic_ion_gets_conjugation_reward_in_python_and_cpp(smiles: str) -> None:
    pybel = _require_pybel()
    allyl = pybel.readstring("smi", smiles)

    python_metrics = compute_organic_topology_metrics(allyl)

    from molgr import _core  # type: ignore

    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(allyl.OBMol.this))

    assert python_metrics.conjugated_atom_count == 3
    assert python_metrics.conjugated_bond_count == 2
    assert python_metrics.max_conjugated_component_size == 3
    assert cpp_metrics["conjugated_atom_count"] == 3
    assert cpp_metrics["conjugated_bond_count"] == 2
    assert cpp_metrics["max_conjugated_component_size"] == 3


@pytest.mark.parametrize(
    "smiles,expected_atoms,expected_bonds",
    [
        ("Oc1ccccc1", 7, 7),
        ("Nc1ccccc1", 7, 7),
        ("C=CO", 3, 2),
        ("COC=C", 3, 2),
        ("CC(=O)N", 3, 2),
        ("C=CCO", 0, 0),  # Saturated carbon interrupts the path.
        ("C=C[N+](C)(C)C", 0, 0),  # Ammonium has no Lewis pair.
        ("C=CP(C)C", 0, 0),  # A pyramidal P pair is not an sp2 donor.
    ],
)
def test_neutral_lewis_conjugation_preserves_topological_boundaries(
    smiles: str, expected_atoms: int, expected_bonds: int
) -> None:
    from molgr import _core

    molecule = _require_pybel().readstring("smi", smiles)
    molecule.addh()
    before = molecule.write("smi")
    python_metrics = compute_organic_topology_metrics(molecule)
    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(molecule.OBMol.this))
    assert python_metrics.conjugated_atom_count == expected_atoms
    assert python_metrics.conjugated_bond_count == expected_bonds
    assert cpp_metrics["conjugated_atom_count"] == expected_atoms
    assert cpp_metrics["conjugated_bond_count"] == expected_bonds
    assert molecule.write("smi") == before


@pytest.mark.parametrize("terminal", ["O", "S"])
def test_exocyclic_multiple_bond_remains_conjugated_with_aromatic_ring(terminal: str) -> None:
    from molgr import _core

    molecule = _require_pybel().readstring("smi", f"{terminal}=c1cccc[nH]1")
    molecule.addh()
    python_metrics = compute_organic_topology_metrics(molecule)
    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(molecule.OBMol.this))
    assert python_metrics.aromatic_atom_count == 6
    assert python_metrics.conjugated_atom_count == 7
    assert python_metrics.conjugated_bond_count == 7
    assert cpp_metrics["conjugated_atom_count"] == 7
    assert cpp_metrics["conjugated_bond_count"] == 7


@pytest.mark.parametrize("scale", [0.4, 1.0, 3.0])
@pytest.mark.parametrize("twisted", [False, True])
def test_lone_pair_donation_respects_orbital_plane(twisted: bool, scale: float) -> None:
    from openbabel import openbabel as ob

    from molgr import _core

    molecule = _require_pybel().readstring("smi", "C=CN")
    molecule.addh()
    positions = [(2.05, 1.1, 0.0), (1.4, 0.0, 0.0), (0.0, 0.0, 0.0)]
    for i, position in enumerate(positions, 1):
        molecule.OBMol.GetAtom(i).SetVector(*(scale * value for value in position))
    nitrogen = molecule.OBMol.GetAtom(3)
    hydrogens = [atom for atom in ob.OBAtomAtomIter(nitrogen) if atom.GetAtomicNum() == 1]
    for sign, atom in zip((-1, 1), hydrogens):
        position = (-0.5, 0.0, sign * 0.8) if twisted else (-0.5, sign * 0.8, 0.0)
        atom.SetVector(*(scale * value for value in position))
    python_metrics = compute_organic_topology_metrics(molecule)
    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(molecule.OBMol.this))
    assert python_metrics.conjugated_atom_count == (0 if twisted else 3)
    assert cpp_metrics["conjugated_atom_count"] == python_metrics.conjugated_atom_count


@pytest.mark.parametrize("electron_state", ["radical", "lone_pair", "unresolved"])
def test_explicit_allylic_electron_center_extends_conjugation(
    electron_state: str,
) -> None:
    pybel = _require_pybel()
    allyl = pybel.readstring("smi", "C=C[CH2]")
    center = allyl.OBMol.GetAtom(3)
    if electron_state == "radical":
        set_unpaired_electron_count(center, 1)
    elif electron_state == "lone_pair":
        set_lone_pair_count(center, 1)
    else:
        set_unresolved_two_electron_center(center, True)

    python_metrics = compute_organic_topology_metrics(allyl)

    from molgr import _core  # type: ignore

    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(allyl.OBMol.this))

    assert python_metrics.conjugated_atom_count == 3
    assert python_metrics.conjugated_bond_count == 2
    assert python_metrics.max_conjugated_component_size == 3
    assert cpp_metrics["conjugated_atom_count"] == 3
    assert cpp_metrics["conjugated_bond_count"] == 2
    assert cpp_metrics["max_conjugated_component_size"] == 3


def test_saturated_charged_center_does_not_extend_conjugation() -> None:
    pybel = _require_pybel()
    saturated_carbonium = pybel.readstring("smi", "C=C[CH3+]")

    python_metrics = compute_organic_topology_metrics(saturated_carbonium)

    from molgr import _core  # type: ignore

    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(
        int(saturated_carbonium.OBMol.this)
    )

    assert python_metrics.conjugated_atom_count == 0
    assert python_metrics.conjugated_bond_count == 0
    assert cpp_metrics["conjugated_atom_count"] == 0
    assert cpp_metrics["conjugated_bond_count"] == 0


@pytest.mark.parametrize("active_pairs", [0, 1])
@pytest.mark.parametrize("explicit_h", [False, True])
def test_neutral_sp2_lewis_pair_extends_conjugation_independently_of_search_pairs(
    active_pairs: int, explicit_h: bool
) -> None:
    pybel = _require_pybel()
    vinylamine = pybel.readstring("smi", "C=CN")
    if explicit_h:
        vinylamine.addh()
    nitrogen = vinylamine.OBMol.GetAtom(3)
    assert nitrogen.GetTotalValence() == 3
    set_lone_pair_count(nitrogen, active_pairs)

    python_metrics = compute_organic_topology_metrics(vinylamine)

    from molgr import _core  # type: ignore

    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(vinylamine.OBMol.this))

    assert python_metrics.conjugated_atom_count == 3
    assert python_metrics.conjugated_bond_count == 2
    assert python_metrics.max_conjugated_component_size == 3
    assert cpp_metrics["conjugated_atom_count"] == 3
    assert cpp_metrics["conjugated_bond_count"] == 2
    assert cpp_metrics["max_conjugated_component_size"] == 3


@pytest.mark.parametrize(
    ("smiles", "expected_donor_count", "expected_score"),
    [
        ("CC", 0, 0),
        ("C=C", 0, 0),
        ("CC=C", 1, 3),
        ("CC(=C)C", 2, 6),
    ],
)
def test_classical_hyperconjugation_is_a_separate_python_cpp_metric(
    smiles: str,
    expected_donor_count: int,
    expected_score: int,
) -> None:
    pybel = _require_pybel()
    molecule = pybel.readstring("smi", smiles)

    python_metrics = compute_organic_topology_metrics(molecule)

    from molgr import _core  # type: ignore

    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(molecule.OBMol.this))

    assert python_metrics.hyperconjugative_donor_count == expected_donor_count
    assert python_metrics.hyperconjugation_score == expected_score
    assert cpp_metrics["hyperconjugative_donor_count"] == expected_donor_count
    assert cpp_metrics["hyperconjugation_score"] == expected_score
    if expected_score:
        assert python_metrics.conjugated_atom_count == 0
        assert python_metrics.conjugated_bond_count == 0


def test_hyperconjugation_score_is_independent_of_explicit_hydrogens() -> None:
    pybel = _require_pybel()
    implicit_hydrogens = pybel.readstring("smi", "CC=C")
    explicit_hydrogens = implicit_hydrogens.clone
    explicit_hydrogens.addh()

    implicit_metrics = compute_organic_topology_metrics(implicit_hydrogens)
    explicit_metrics = compute_organic_topology_metrics(explicit_hydrogens)

    assert implicit_metrics.hyperconjugation_score == 3
    assert explicit_metrics.hyperconjugation_score == implicit_metrics.hyperconjugation_score


def test_high_absolute_formal_charge_sum_rejects_ring_aromaticity() -> None:
    pybel = _require_pybel()
    tolerated = pybel.readstring("smi", "[c-]1[c-][c-][cH][cH][cH]1")
    rejected = pybel.readstring("smi", "[c-]1[c-][c-][c-][cH][cH]1")

    tolerated_metrics = compute_organic_topology_metrics(tolerated)
    rejected_metrics = compute_organic_topology_metrics(rejected)

    assert tolerated_metrics.aromatic_atom_count == 6
    assert tolerated_metrics.aromatic_ring_count == 1
    assert rejected_metrics.aromatic_atom_count == 0
    assert rejected_metrics.aromatic_ring_count == 0


def test_fused_aromatic_system_is_not_rejected_by_individual_sssr_huckel_counts() -> None:
    pybel = _require_pybel()
    accepted = pybel.readstring("smi", "[n-]1cccc1")
    rejected = pybel.readstring("smi", "c1cc2ccccc2cc1")

    accepted_metrics = compute_organic_topology_metrics(accepted)
    rejected_metrics = compute_organic_topology_metrics(rejected)

    assert accepted_metrics.aromatic_ring_count == 1
    assert accepted_metrics.aromatic_atom_count == 5
    assert rejected_metrics.aromatic_ring_count == 2
    assert rejected_metrics.aromatic_atom_count == 10
    assert rejected_metrics.aromatic_stability_score == pytest.approx(2.0)


def test_aceput_fused_system_prefers_the_more_complete_aromatic_candidate() -> None:
    pybel = _require_pybel()
    c2 = pybel.readstring(
        "smi",
        "[Br-].Clc1c2ccc3ccc4[C-](Cl)[CH-]C=Nc4c3c2ncc1.[O+]#[C-].[O+]#[C-].[O+]#[C-]",
    )
    c3 = pybel.readstring(
        "smi",
        "[Br-].Clc1c2ccc3ccc4c(Cl)ccnc4c3c2ncc1.[O+]#[C-].[O+]#[C-].[O+]#[C-]",
    )

    c2_metrics = compute_organic_topology_metrics(c2)
    c3_metrics = compute_organic_topology_metrics(c3)

    assert c2_metrics.aromatic_ring_count == 3
    assert c2_metrics.aromatic_atom_count == 14
    assert c3_metrics.aromatic_ring_count == 4
    assert c3_metrics.aromatic_atom_count == 18
    assert c3_metrics.aromatic_stability_score > c2_metrics.aromatic_stability_score


def test_negative_ring_atom_contributes_pi_electrons_when_not_in_ring_multiple_bond() -> None:
    pybel = _require_pybel()
    omol = pybel.readstring("smi", "[N-]1C=CC=C1")
    negative_nitrogen = omol.OBMol.GetAtom(1)

    assert int(negative_nitrogen.GetAtomicNum()) == 7
    assert int(negative_nitrogen.GetFormalCharge()) == -1
    assert int(negative_nitrogen.GetHyb()) == 2
    assert (
        _additional_atom_pi_electrons(
            negative_nitrogen,
            incident_to_ring_multiple_bond=False,
        )
        == 2
    )
    assert (
        _additional_atom_pi_electrons(
            negative_nitrogen,
            incident_to_ring_multiple_bond=True,
        )
        == 0
    )


@pytest.mark.parametrize(("hybridization", "expected_electrons"), [(1, 0), (2, 2), (3, 0)])
def test_heteroatom_lone_pair_pi_contribution_requires_sp2_hybridization(
    hybridization: int,
    expected_electrons: int,
) -> None:
    pybel = _require_pybel()
    omol = pybel.readstring("smi", "N")
    nitrogen = omol.OBMol.GetAtom(1)
    nitrogen.SetHyb(hybridization)
    omol.OBMol.SetHybridizationPerceived(True)

    assert (
        _additional_atom_pi_electrons(
            nitrogen,
            incident_to_ring_multiple_bond=False,
        )
        == expected_electrons
    )


def test_topology_metrics_refresh_stale_hybridization_for_python_and_cpp() -> None:
    pybel = _require_pybel()
    pyrrole = pybel.readstring("smi", "c1cc[nH]c1")
    nitrogen = pyrrole.OBMol.GetAtom(4)
    nitrogen.SetHyb(3)
    pyrrole.OBMol.SetHybridizationPerceived(True)

    assert int(nitrogen.GetHyb()) == 3
    python_metrics = compute_organic_topology_metrics(pyrrole)

    from molgr import _core  # type: ignore

    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(pyrrole.OBMol.this))

    assert python_metrics.aromatic_ring_count == 1
    assert python_metrics.aromatic_atom_count == 5
    assert cpp_metrics["aromatic_ring_count"] == 1
    assert cpp_metrics["aromatic_atom_count"] == 5
    assert int(nitrogen.GetHyb()) == 3


def test_sp3_heteroatom_does_not_supply_aromatic_pi_electrons_for_python_and_cpp() -> None:
    pybel = _require_pybel()
    phosphole = pybel.readstring("smi", "P1C=CC=C1")
    phosphorus = phosphole.OBMol.GetAtom(1)
    phosphole.OBMol.SetHybridizationPerceived(False)

    assert int(phosphorus.GetHyb()) == 3
    python_metrics = compute_organic_topology_metrics(phosphole)

    from molgr import _core  # type: ignore

    cpp_metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(int(phosphole.OBMol.this))

    assert python_metrics.aromatic_ring_count == 0
    assert python_metrics.aromatic_atom_count == 0
    assert cpp_metrics["aromatic_ring_count"] == 0
    assert cpp_metrics["aromatic_atom_count"] == 0


def test_acasoo_like_resonance_prefers_greater_aromatic_system_coverage() -> None:
    pybel = _require_pybel()
    resonance_zero = pybel.readstring(
        "smi",
        r"C\1/2=C(/C3=C(c4c(=N3)/c(=C\3/C=CC(=N3)/C(=c\3/cc/c(=C(/C(=N2)C=C1)\c1ccccc1)/[n-]3)/c1ccccc1)/c1ccccc1c4[O-])N)\c1ccccc1",
    )
    resonance_one = pybel.readstring(
        "smi",
        r"C\1/2=C(/c3c(c4c(/C(=C\5/C=CC(=N5)/C(=c\5/cc/c(=C(/C(=N2)C=C1)\c1ccccc1)/[n-]5)/c1ccccc1)/c1ccccc1C4=O)[n-]3)N)\c1ccccc1",
    )

    zero_metrics = compute_organic_topology_metrics(resonance_zero)
    one_metrics = compute_organic_topology_metrics(resonance_one)

    assert zero_metrics.aromatic_ring_count == 5
    assert zero_metrics.aromatic_atom_count == 28
    assert zero_metrics.aromatic_stability_score == pytest.approx(5.0)
    assert one_metrics.aromatic_ring_count == 5
    assert one_metrics.aromatic_atom_count == 29
    assert one_metrics.aromatic_stability_score == pytest.approx(4.63504)
    assert one_metrics.aromatic_atom_count > zero_metrics.aromatic_atom_count
    assert one_metrics.aromatic_stability_score < zero_metrics.aromatic_stability_score


def test_aromatic_stability_scores_benzene_above_heteroaromatics() -> None:
    pybel = _require_pybel()
    benzene = pybel.readstring("smi", "c1ccccc1")
    pyridine = pybel.readstring("smi", "n1ccccc1")
    pyrrole = pybel.readstring("smi", "c1cc[nH]c1")
    thiophene = pybel.readstring("smi", "c1ccsc1")

    benzene_metrics = compute_organic_topology_metrics(benzene)
    hetero_scores = [
        compute_organic_topology_metrics(mol).aromatic_stability_score
        for mol in (pyridine, pyrrole, thiophene)
    ]

    assert benzene_metrics.aromatic_stability_score == pytest.approx(1.0)
    assert all(0.0 < score < 1.0 for score in hetero_scores)


def test_aromatic_stability_uses_configured_scale_factors() -> None:
    pybel = _require_pybel()
    pyridine = pybel.readstring("smi", "n1ccccc1")
    config = replace(
        MolGRConfig().organic_topology,
        aromatic_stability_ring_size_6_factor=0.50,
        aromatic_stability_hetero_atom_penalty=0.25,
        aromatic_stability_other_ring_max_score=0.99,
    )

    metrics = compute_organic_topology_metrics(pyridine, config=config)

    assert metrics.aromatic_stability_score == pytest.approx(0.50 * 0.75)


def test_abated_like_quad_anionic_ring_is_not_counted_as_aromatic() -> None:
    pybel = _require_pybel()
    abated_like = pybel.readstring(
        "smi",
        "C/C(=C/C(=N/c1c(C)cccc1C)/C)/[N-]c1c(C)cccc1C.C[c-]1[cH-][cH-][cH-]cc1",
    )
    neutral_reference = pybel.readstring(
        "smi",
        "C/C(=C/C(=N/c1c(C)cccc1C)/C)/[N-]c1c(C)cccc1C.Cc1ccccc1",
    )

    abated_like_metrics = compute_organic_topology_metrics(abated_like)
    neutral_reference_metrics = compute_organic_topology_metrics(neutral_reference)

    assert neutral_reference_metrics.aromatic_ring_count == 3
    assert neutral_reference_metrics.aromatic_atom_count == 18
    assert abated_like_metrics.aromatic_ring_count == 2
    assert abated_like_metrics.aromatic_atom_count == 12


def test_re_dithiocarbamate_variants_do_not_get_artificial_conjugation_gap() -> None:
    pybel = _require_pybel()
    zwitterionic = pybel.readstring(
        "smi",
        "CP(c1ccccc1)c1ccccc1.CC[N+](=C([S+])[S+])CC.[C-]#[O+].[C-]#[O+].[C-]#[O+]",
    )
    anionic = pybel.readstring(
        "smi",
        "CP(c1ccccc1)c1ccccc1.CCN(C(=S)[S-])CC.[C-]#[O+].[C-]#[O+].[C-]#[O+]",
    )

    zwitterionic_metrics = compute_organic_topology_metrics(zwitterionic)
    anionic_metrics = compute_organic_topology_metrics(anionic)

    assert (
        abs(zwitterionic_metrics.conjugated_atom_count - anionic_metrics.conjugated_atom_count) <= 1
    )
    assert (
        abs(zwitterionic_metrics.conjugated_bond_count - anionic_metrics.conjugated_bond_count) <= 1
    )
    assert (
        zwitterionic_metrics.max_conjugated_component_size
        == anionic_metrics.max_conjugated_component_size
        == 6
    )


def test_au_nhc_zwitterion_variant_keeps_conjugation_component() -> None:
    pybel = _require_pybel()
    charge_separated = pybel.readstring(
        "smi",
        "Cc1cc(C)c(c(c1)C)[N+]1=C([S+]=C=C1c1ccccc1)C1=CC=CC=C1.[Cl-]",
    )
    aromatic_zwitterion = pybel.readstring(
        "smi",
        "Cc1cc(C)c(c(c1)C)n1c([s+][c-]c1c1ccccc1)c1ccccc1.[Cl-]",
    )

    charge_separated_metrics = compute_organic_topology_metrics(charge_separated)
    aromatic_zwitterion_metrics = compute_organic_topology_metrics(aromatic_zwitterion)

    assert (
        aromatic_zwitterion_metrics.max_conjugated_component_size
        >= charge_separated_metrics.max_conjugated_component_size
    )
    assert (
        aromatic_zwitterion_metrics.conjugated_atom_count
        >= charge_separated_metrics.conjugated_atom_count
    )


@pytest.mark.parametrize(
    ("smiles", "components"),
    [
        ("C=CC=C", ((0, 1, 2, 3),)),
        ("C=CC=C.C=CC=C", ((0, 1, 2, 3), (4, 5, 6, 7))),
        ("C=CC=CC(C)(C)C=CC=C", ((0, 1, 2, 3), (7, 8, 9, 10))),
        ("C=CC=C[Si](C)(C)C=CC=C", ((0, 1, 2, 3), (7, 8, 9, 10))),
        ("C=CC=C=CC=C", ((0, 1, 2, 3), (3, 4, 5, 6))),
    ],
)
def test_conjugated_components_preserve_sigma_and_orthogonal_pi_boundaries(smiles, components):
    from molgr import _core

    mol = _require_pybel().readstring("smi", smiles)
    python = compute_organic_topology_metrics(mol)
    cpp = _core.dev.utils.compute_organic_topology_metrics_ptr(int(mol.OBMol.this))
    assert python.conjugated_atom_components == components
    assert tuple(tuple(c) for c in cpp["conjugated_atom_components"]) == components
    assert python.max_conjugated_component_size == max(map(len, components))
