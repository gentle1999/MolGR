# pyright: reportCallIssue=false
from __future__ import annotations

import pytest
from rdkit import Chem

from molgr.utils import equivalence


METAL_UNPAIRED = "MOLGR_METAL_UNPAIRED_ELECTRONS"
LONE_PAIRS = "MOLGR_LONE_PAIR_COUNT"
UNRESOLVED = "MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER"


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _set_counts(mol: Chem.Mol, prop: str, counts: tuple[int, ...]) -> None:
    assert mol.GetNumAtoms() == len(counts)
    for atom, count in zip(mol.GetAtoms(), counts):
        atom.SetIntProp(prop, count)


def _input_state(mol: Chem.Mol) -> tuple:
    return (
        mol.GetPropsAsDict(includePrivate=True, includeComputed=False),
        tuple(
            (
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetNumExplicitHs(),
                atom.GetNoImplicit(),
                atom.GetChiralTag(),
                atom.GetPropsAsDict(includePrivate=True, includeComputed=False),
            )
            for atom in mol.GetAtoms()
        ),
        tuple(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType(),
                bond.GetStereo(),
                bond.GetPropsAsDict(includePrivate=True, includeComputed=False),
            )
            for bond in mol.GetBonds()
        ),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_same_smiles_cannot_hide_known_metal_unpaired_conflict(reverse: bool) -> None:
    low_spin = _mol("[Fe]")
    high_spin = Chem.Mol(low_spin)
    low_spin.GetAtomWithIdx(0).SetIntProp(METAL_UNPAIRED, 0)
    high_spin.GetAtomWithIdx(0).SetIntProp(METAL_UNPAIRED, 2)
    assert Chem.MolToSmiles(low_spin) == Chem.MolToSmiles(high_spin)
    mol1, mol2 = (high_spin, low_spin) if reverse else (low_spin, high_spin)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.equivalent is False
    assert result.invariants["electronic_properties"].status == equivalence.InvariantStatus.FAILED


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("smiles", "prop", "known_value"),
    [
        ("[Fe]", METAL_UNPAIRED, 0),
        ("[Fe]", METAL_UNPAIRED, 2),
        ("O", LONE_PAIRS, 0),
        ("O", LONE_PAIRS, 1),
        ("[CH2]", UNRESOLVED, 0),
        ("[CH2]", UNRESOLVED, 1),
    ],
)
def test_baseline_without_properties_is_unknown_not_zero_or_inconclusive(
    smiles: str,
    prop: str,
    known_value: int,
    reverse: bool,
) -> None:
    annotated = _mol(smiles)
    baseline = Chem.Mol(annotated)
    annotated.GetAtomWithIdx(0).SetIntProp(prop, known_value)
    mol1, mol2 = (baseline, annotated) if reverse else (annotated, baseline)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    electronic = result.invariants["electronic_properties"]
    assert electronic.status == equivalence.InvariantStatus.NOT_EVALUATED
    annotated_evidence = electronic.mol2 if reverse else electronic.mol1
    baseline_evidence = electronic.mol1 if reverse else electronic.mol2
    assert annotated_evidence["known_fields"] > 0
    assert baseline_evidence["known_fields"] == 0
    assert annotated_evidence["comparable_fields"] == 0
    assert baseline_evidence["comparable_fields"] == 0
    assert not baseline.GetAtomWithIdx(0).HasProp(prop)


@pytest.mark.parametrize("smiles", ["[Fe]", "CCO", "[CH2]C=O"])
def test_both_sides_unknown_preserve_sufficient_graph_equivalence(smiles: str) -> None:
    mol1 = _mol(smiles)
    mol2 = Chem.Mol(mol1)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert (
        result.invariants["electronic_properties"].status
        == equivalence.InvariantStatus.NOT_EVALUATED
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_known_metadata_does_not_turn_unknown_fields_into_conflicts(reverse: bool) -> None:
    mol1 = _mol("CCO")
    mol2 = Chem.Mol(mol1)
    for mol in (mol1, mol2):
        mol.GetAtomWithIdx(2).SetIntProp(LONE_PAIRS, 1)
    mol1.GetAtomWithIdx(0).SetBoolProp(UNRESOLVED, True)
    if reverse:
        mol1, mol2 = mol2, mol1

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.invariants["electronic_properties"].status != equivalence.InvariantStatus.FAILED


@pytest.mark.parametrize("reverse", [False, True])
def test_known_lone_pair_can_follow_allowed_charge_resonance(reverse: bool) -> None:
    carbon_anion = _mol("[CH2-]C=O")
    oxygen_anion = _mol("C=C[O-]")
    # These are reconstruction-active pairs, not the complete Lewis lone-pair census.
    _set_counts(carbon_anion, LONE_PAIRS, (1, 0, 0))
    _set_counts(oxygen_anion, LONE_PAIRS, (0, 0, 1))
    mol1, mol2 = (oxygen_anion, carbon_anion) if reverse else (carbon_anion, oxygen_anion)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.invariants["electronic_properties"].status == equivalence.InvariantStatus.PASSED


@pytest.mark.parametrize("reverse", [False, True])
def test_symmetric_atom_reindexing_preserves_compatible_known_metadata(reverse: bool) -> None:
    mol1 = _mol("OCCO")
    _set_counts(mol1, LONE_PAIRS, (0, 0, 0, 1))
    mol2 = Chem.RenumberAtoms(mol1, [3, 2, 1, 0])
    if reverse:
        mol1, mol2 = mol2, mol1

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.invariants["electronic_properties"].status == equivalence.InvariantStatus.PASSED


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("terminal_count", [1, 2])
def test_symmetric_alternative_mapping_is_considered_before_known_mismatch(
    terminal_count: int,
    reverse: bool,
) -> None:
    mol1 = _mol("OCCO")
    mol2 = Chem.Mol(mol1)
    _set_counts(mol1, LONE_PAIRS, (0, 0, 0, 1))
    _set_counts(mol2, LONE_PAIRS, (terminal_count, 0, 0, 0))
    if reverse:
        mol1, mol2 = mol2, mol1

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    expected = (
        equivalence.EquivalenceDecision.EQUIVALENT
        if terminal_count == 1
        else equivalence.EquivalenceDecision.NOT_EQUIVALENT
    )
    assert result.decision == expected
    expected_status = (
        equivalence.InvariantStatus.PASSED
        if terminal_count == 1
        else equivalence.InvariantStatus.FAILED
    )
    assert result.invariants["electronic_properties"].status == expected_status


@pytest.mark.parametrize("other_unresolved", [False, True])
def test_unresolved_center_is_a_separate_flag_not_a_numeric_radical_count(
    other_unresolved: bool,
) -> None:
    mol1 = _mol("[CH2]")
    mol2 = Chem.Mol(mol1)
    mol1.GetAtomWithIdx(0).SetBoolProp(UNRESOLVED, True)
    mol2.GetAtomWithIdx(0).SetBoolProp(UNRESOLVED, other_unresolved)
    radicals = mol1.GetAtomWithIdx(0).GetNumRadicalElectrons()

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.raw_mol1.rdkit_radical_electrons == radicals
    assert result.raw_mol2.rdkit_radical_electrons == radicals
    assert mol1.GetAtomWithIdx(0).GetNumRadicalElectrons() == radicals
    assert mol2.GetAtomWithIdx(0).GetNumRadicalElectrons() == radicals


@pytest.mark.parametrize("reverse", [False, True])
def test_unresolved_zero_placeholders_do_not_conflict_with_a_known_lone_pair(
    reverse: bool,
) -> None:
    deferred = _mol("[CH2]")
    deferred.GetAtomWithIdx(0).SetNumRadicalElectrons(0)
    resolved = Chem.Mol(deferred)
    deferred.GetAtomWithIdx(0).SetBoolProp(UNRESOLVED, True)
    deferred.GetAtomWithIdx(0).SetIntProp(LONE_PAIRS, 0)
    resolved.GetAtomWithIdx(0).SetBoolProp(UNRESOLVED, False)
    resolved.GetAtomWithIdx(0).SetIntProp(LONE_PAIRS, 1)
    mol1, mol2 = (resolved, deferred) if reverse else (deferred, resolved)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.invariants["electronic_properties"].status != equivalence.InvariantStatus.FAILED
    assert deferred.GetAtomWithIdx(0).GetNumRadicalElectrons() == 0
    assert deferred.GetAtomWithIdx(0).GetIntProp(LONE_PAIRS) == 0
    assert deferred.GetAtomWithIdx(0).GetBoolProp(UNRESOLVED) is True


@pytest.mark.parametrize("copy_kind", ["molecule", "binary"])
def test_evaluation_and_property_preserving_copies_keep_input_electronic_assignments(
    copy_kind: str,
) -> None:
    original = _mol("[Fe].O.[CH2]")
    original.SetProp("consumer_note", "preserve source assignments")
    original.GetAtomWithIdx(0).SetIntProp(METAL_UNPAIRED, 2)
    original.GetAtomWithIdx(1).SetIntProp(LONE_PAIRS, 1)
    original.GetAtomWithIdx(2).SetBoolProp(UNRESOLVED, True)
    before_original = _input_state(original)
    copied = (
        Chem.Mol(original)
        if copy_kind == "molecule"
        else Chem.Mol(original.ToBinary(Chem.PropertyPickleOptions.AllProps))
    )
    assert _input_state(copied) == before_original
    before_copied = _input_state(copied)

    result = equivalence.evaluate_equivalence(original, copied, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.invariants["electronic_properties"].status == equivalence.InvariantStatus.PASSED
    assert _input_state(original) == before_original
    assert _input_state(copied) == before_copied


@pytest.mark.parametrize("reverse", [False, True])
def test_bounded_electronic_mapping_is_inconclusive_not_a_known_conflict(
    monkeypatch: pytest.MonkeyPatch,
    reverse: bool,
) -> None:
    mol1 = _mol("OCCO")
    mol2 = Chem.Mol(mol1)
    _set_counts(mol1, LONE_PAIRS, (0, 0, 0, 1))
    _set_counts(mol2, LONE_PAIRS, (1, 0, 0, 0))
    if reverse:
        mol1, mol2 = mol2, mol1
    # Stop before completing any mapping, independently of candidate ordering.
    monkeypatch.setattr(equivalence, "_ELECTRON_PROPERTY_MAPPING_LIMIT", 1)

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False, max_mapping_nodes=1)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.equivalent is False
    assert (
        result.invariants["electronic_properties"].status
        == equivalence.InvariantStatus.NOT_EVALUATED
    )


def test_electronic_mapping_enumeration_exception_is_inconclusive_not_inequivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol1 = _mol("OCCO")
    mol2 = Chem.Mol(mol1)
    _set_counts(mol1, LONE_PAIRS, (0, 0, 0, 1))
    _set_counts(mol2, LONE_PAIRS, (1, 0, 0, 0))

    def fail_mapping_enumeration(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("controlled atom-mapping enumeration failure")

    monkeypatch.setattr(
        equivalence._core.equivalence,
        "evaluate",
        fail_mapping_enumeration,
    )

    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.equivalent is False
    assert (
        result.invariants["electronic_properties"].status
        == equivalence.InvariantStatus.NOT_EVALUATED
    )


@pytest.mark.parametrize(
    ("prop", "malformed"),
    [(METAL_UNPAIRED, "missing"), (LONE_PAIRS, "-1"), (UNRESOLVED, "not-a-bool")],
)
def test_malformed_electronic_metadata_remains_unknown_without_blocking_graph_eq(
    prop: str,
    malformed: str,
) -> None:
    mol1 = _mol("[Fe].O")
    mol2 = Chem.Mol(mol1)
    mol1.GetAtomWithIdx(0).SetProp(prop, malformed)
    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)
    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.invariants["electronic_properties"].mol1["known_fields"] == 0
    assert result.invariants["electronic_properties"].mol1["comparable_fields"] == 0


def test_unreconciled_conjugated_annotations_are_not_a_negative_search_proof() -> None:
    mol1 = _mol("C=C")
    mol2 = Chem.Mol(mol1)
    _set_counts(mol1, LONE_PAIRS, (0, 0))
    _set_counts(mol2, LONE_PAIRS, (1, 0))
    result = equivalence.evaluate_equivalence(mol1, mol2, use_chirality=False)
    assert result.decision == equivalence.EquivalenceDecision.INCONCLUSIVE
    assert result.invariants["electronic_properties"].status != equivalence.InvariantStatus.FAILED
