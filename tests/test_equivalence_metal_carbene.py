"""Resolved metal carbenes must not acquire artificial unpaired electrons."""

import pytest
from rdkit import Chem

from molgr.utils.equivalence import evaluate_equivalence


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.AddHs(mol)


def _closed_metal_carbene(smiles="[Mo]=[CH]C#C[O-]"):
    mol = _mol(smiles)
    carbon = mol.GetAtomWithIdx(1)
    assert carbon.GetAtomicNum() == 6
    carbon.SetIntProp("MOLGR_UNPAIRED_ELECTRON_COUNT", 0)
    carbon.SetIntProp("MOLGR_LONE_PAIR_COUNT", 0)
    carbon.SetBoolProp("MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER", False)
    return mol


def _compare(left, right):
    return evaluate_equivalence(
        left,
        right,
        use_chirality=False,
        ignore_metal_coordination_bonds=True,
        max_resonance=100000,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("renumber", [False, True])
def test_paired_carbene_has_electron_conserving_acyl_resonance(reverse, renumber):
    left = _closed_metal_carbene()
    right = _mol("[Mo].C#C[C-]=O")
    if renumber:
        left = Chem.RenumberAtoms(left, list(reversed(range(left.GetNumAtoms()))))
    if reverse:
        left, right = right, left
    before = [mol.ToBinary() for mol in (left, right)]
    result = _compare(left, right)
    assert result.equivalent, result.reason
    assert result.resonance_witness
    assert [mol.ToBinary() for mol in (left, right)] == before


@pytest.mark.parametrize(
    "field",
    [
        "MOLGR_UNPAIRED_ELECTRON_COUNT",
        "MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER",
    ],
)
def test_unknown_electron_metadata_does_not_enable_paired_convention(field):
    left = _closed_metal_carbene()
    left.GetAtomWithIdx(1).ClearProp(field)
    assert not _compare(left, _mol("[Mo].C#C[C-]=O")).equivalent


def test_unresolved_center_does_not_enable_paired_convention():
    left = _closed_metal_carbene()
    left.GetAtomWithIdx(1).SetBoolProp("MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER", True)
    assert not _compare(left, _mol("[Mo].C#C[C-]=O")).equivalent


@pytest.mark.parametrize("reverse", [False, True])
def test_known_closed_carbene_is_not_an_explicit_triplet(reverse):
    left = _closed_metal_carbene()
    right = _mol("[Mo]<-[CH]C#C[O-]")
    right.GetAtomWithIdx(1).SetIntProp("MOLGR_UNPAIRED_ELECTRON_COUNT", 2)
    right.GetAtomWithIdx(1).SetBoolProp("MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER", False)
    if reverse:
        left, right = right, left
    assert not _compare(left, right).equivalent


def test_missing_metadata_on_identical_metal_carbene_is_not_a_conflict():
    left = _closed_metal_carbene()
    right = _mol("[Mo]=[CH]C#C[O-]")
    assert _compare(left, right).equivalent
    assert _compare(right, left).equivalent


@pytest.mark.parametrize("explicit_pair", [False, True])
def test_paired_metal_carbene_matches_free_carbene_without_inventing_spin(explicit_pair):
    left = _closed_metal_carbene()
    right = _mol("[Mo]<-[CH]C#C[O-]")
    if explicit_pair:
        carbon = right.GetAtomWithIdx(1)
        carbon.SetIntProp("MOLGR_UNPAIRED_ELECTRON_COUNT", 0)
        carbon.SetIntProp("MOLGR_LONE_PAIR_COUNT", 1)
        carbon.SetBoolProp("MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER", False)
    assert _compare(left, right).equivalent
    assert _compare(right, left).equivalent


def test_explicit_absence_of_carbene_pair_is_not_overridden():
    left = _closed_metal_carbene()
    right = _mol("[Mo]<-[CH]C#C[O-]")
    right.GetAtomWithIdx(1).SetIntProp("MOLGR_LONE_PAIR_COUNT", 0)
    assert not _compare(left, right).equivalent


def test_paired_convention_still_accepts_charge_separated_nhc():
    left = _closed_metal_carbene("[Mo]=C(N(C)C)N(C)C")
    right = _mol("[Mo]<-[C-](=[N+](C)C)N(C)C")
    result = _compare(left, right)
    assert result.equivalent
    assert any(move["kind"] == "paired_carbene_normalization" for move in result.resonance_witness)
    assert _compare(right, left).equivalent


def test_paired_convention_preserves_total_charge_and_metal_state_checks():
    left = _closed_metal_carbene()
    right = _mol("[Mo+2].C#C[C-]=O")
    assert not _compare(left, right).equivalent


def test_legacy_unannotated_metal_carbene_convention_is_preserved():
    assert _compare(_mol("[Mo]=[CH]C#C[O-]"), _mol("[Mo]<-[CH]C#C[O-]")).equivalent


def test_paired_carbene_still_requires_a_bounded_resonance_proof():
    result = evaluate_equivalence(
        _closed_metal_carbene(),
        _mol("[Mo].C#C[C-]=O"),
        use_chirality=False,
        ignore_metal_coordination_bonds=True,
        max_resonance=0,
    )
    assert not result.equivalent
