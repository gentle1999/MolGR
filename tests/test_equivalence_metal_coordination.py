from __future__ import annotations

from rdkit import Chem

from molgr.utils.equivalence import (
    EquivalenceDecision,
    InvariantStatus,
    check_equivalence,
    evaluate_equivalence,
)


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def test_metal_coordination_is_ignored_by_default_and_checked_on_request() -> None:
    coordinated, separated = _mol("N->[Cu]"), _mol("N.[Cu]")

    default = evaluate_equivalence(coordinated, separated, use_chirality=False)
    strict = evaluate_equivalence(
        coordinated,
        separated,
        use_chirality=False,
        ignore_metal_coordination_bonds=False,
    )
    wrapper_equivalent, wrapper = check_equivalence(
        coordinated,
        separated,
        use_chirality=False,
        ignore_metal_coordination_bonds=False,
    )

    assert default.decision == EquivalenceDecision.EQUIVALENT
    assert default.invariants["metal_coordination"].status == InvariantStatus.NOT_EVALUATED
    assert strict.decision == EquivalenceDecision.NOT_EQUIVALENT
    assert strict.invariants["metal_coordination"].status == InvariantStatus.FAILED
    assert wrapper_equivalent is False
    assert wrapper.decision == strict.decision


def test_strict_metal_coordination_checks_which_ligand_is_attached() -> None:
    nitrogen_bound = _mol("N->[Cu].O")
    oxygen_bound = _mol("N.O->[Cu]")

    assert (
        evaluate_equivalence(nitrogen_bound, oxygen_bound, use_chirality=False).decision
        == EquivalenceDecision.EQUIVALENT
    )
    strict = evaluate_equivalence(
        nitrogen_bound,
        oxygen_bound,
        use_chirality=False,
        ignore_metal_coordination_bonds=False,
    )
    assert strict.decision == EquivalenceDecision.NOT_EQUIVALENT
    assert strict.invariants["metal_coordination"].status == InvariantStatus.FAILED


def test_strict_metal_coordination_accepts_reindexed_same_graph() -> None:
    original = _mol("N->[Cu].O")
    reindexed = Chem.RenumberAtoms(original, list(reversed(range(original.GetNumAtoms()))))

    result = evaluate_equivalence(
        original,
        reindexed,
        use_chirality=False,
        ignore_metal_coordination_bonds=False,
    )
    assert result.decision == EquivalenceDecision.EQUIVALENT
    assert result.invariants["metal_coordination"].status == InvariantStatus.PASSED


def test_strict_electronic_mapping_respects_metal_attachment() -> None:
    first = _mol("N->[Fe].O->[Fe]")
    second = Chem.Mol(first)
    for mol, counts in ((first, (0, 2)), (second, (2, 0))):
        metals = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 26]
        for atom, count in zip(metals, counts):
            atom.SetIntProp("MOLGR_METAL_UNPAIRED_ELECTRONS", count)

    assert evaluate_equivalence(first, second, use_chirality=False).decision == (
        EquivalenceDecision.EQUIVALENT
    )
    strict = evaluate_equivalence(
        first,
        second,
        use_chirality=False,
        ignore_metal_coordination_bonds=False,
    )
    assert strict.decision == EquivalenceDecision.NOT_EQUIVALENT
    assert strict.invariants["metal_coordination"].status == InvariantStatus.PASSED
    assert strict.invariants["electronic_properties"].status == InvariantStatus.FAILED
