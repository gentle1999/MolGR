from __future__ import annotations

import io

import pytest
from rdkit import Chem

from molgr.utils import equivalence


UNPAIRED = "MOLGR_UNPAIRED_ELECTRON_COUNT"
LONE_PAIRS = "MOLGR_LONE_PAIR_COUNT"


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _state(mol: Chem.Mol) -> tuple:
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


def _set_counts(mol: Chem.Mol, prop: str, counts: tuple[int, ...]) -> None:
    assert mol.GetNumAtoms() == len(counts)
    for atom, count in zip(mol.GetAtoms(), counts):
        atom.SetIntProp(prop, count)


@pytest.mark.parametrize("use_chirality", [False, True])
@pytest.mark.parametrize("reindex", [False, True])
@pytest.mark.parametrize("other_mapped", [False, True])
def test_source_atom_maps_are_not_chemistry_and_inputs_remain_unchanged(
    use_chirality: bool, reindex: bool, other_mapped: bool,
) -> None:
    source = _mol("C[C@H](F)Cl")
    for idx, atom in enumerate(source.GetAtoms()):
        atom.SetAtomMapNum(101 + idx)
    other = Chem.Mol(source)
    for idx, atom in enumerate(other.GetAtoms()):
        atom.SetAtomMapNum(901 - idx if other_mapped else 0)
    if reindex:
        other = Chem.RenumberAtoms(other, [3, 1, 0, 2])
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=use_chirality)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert (_state(source), _state(other)) == before


@pytest.mark.parametrize("use_chirality", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("terminal_count", [1, 2])
def test_source_maps_do_not_block_symmetric_electronic_property_mapping(
    use_chirality: bool, reverse: bool, terminal_count: int,
) -> None:
    source = _mol("OCCO")
    _set_counts(source, LONE_PAIRS, (0, 0, 0, 1))
    for idx, atom in enumerate(source.GetAtoms()):
        atom.SetAtomMapNum(21 + idx)
    other = Chem.RenumberAtoms(source, [3, 2, 1, 0])
    _set_counts(other, LONE_PAIRS, (terminal_count, 0, 0, 0))
    for idx, atom in enumerate(other.GetAtoms()):
        atom.SetAtomMapNum(71 + idx)
    if reverse:
        source, other = other, source
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=use_chirality)

    expected = (
        equivalence.EquivalenceDecision.EQUIVALENT
        if terminal_count == 1
        else equivalence.EquivalenceDecision.NOT_EQUIVALENT
    )
    assert result.decision == expected
    status = (
        equivalence.InvariantStatus.PASSED
        if terminal_count == 1
        else equivalence.InvariantStatus.FAILED
    )
    assert result.invariants["electronic_properties"].status == status
    assert (_state(source), _state(other)) == before


@pytest.mark.parametrize("use_chirality", [False, True])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("[13CH4]", "C"),
        ("[13CH3]CO", "C[13CH2]O"),
        ("[2H]CCO", "CC([2H])O"),
        ("[2H]O[2H]", "O"),
    ],
)
def test_isotope_identity_and_position_are_not_disabled_with_stereochemistry(
    left: str, right: str, use_chirality: bool,
) -> None:
    source, other = _mol(left), _mol(right)
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=use_chirality)

    assert result.decision == equivalence.EquivalenceDecision.NOT_EQUIVALENT
    assert result.equivalent is False
    assert (_state(source), _state(other)) == before


@pytest.mark.parametrize("use_chirality", [False, True])
@pytest.mark.parametrize("smiles", ["[13CH3]CO", "[2H]CCO"])
def test_matching_isotopes_survive_reindexing_and_source_map_changes(
    smiles: str, use_chirality: bool,
) -> None:
    source = _mol(smiles)
    for idx, atom in enumerate(source.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)
    other = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    for idx, atom in enumerate(other.GetAtoms()):
        atom.SetAtomMapNum(idx + 501)
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=use_chirality)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert (_state(source), _state(other)) == before


@pytest.mark.parametrize("use_chirality", [False, True])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("C[C@H](F)Cl", "C[C@@H](F)Cl"),
        ("[2H][C@](F)(Cl)Br", "[2H][C@@](F)(Cl)Br"),
    ],
)
def test_enantiomers_are_equivalent_only_when_chirality_is_disabled(
    left: str, right: str, use_chirality: bool,
) -> None:
    source, other = _mol(left), _mol(right)
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=use_chirality)

    expected = (
        equivalence.EquivalenceDecision.NOT_EQUIVALENT
        if use_chirality
        else equivalence.EquivalenceDecision.EQUIVALENT
    )
    assert result.decision == expected
    assert (_state(source), _state(other)) == before


@pytest.mark.parametrize("count", [0, 1, 2])
def test_known_nonmetal_unpaired_count_survives_copy_and_evaluation_normalization(
    count: int,
) -> None:
    source = _mol("F[B-]F")
    boron = source.GetAtomWithIdx(1)
    boron.SetNumRadicalElectrons(count)
    boron.SetIntProp(UNPAIRED, count)
    before = _state(source)

    copied = Chem.Mol(source)
    data, _ = equivalence._prepare_input(source, use_chirality=False)
    assert data.atoms[1].radical_num == count
    assert data.atoms[1].electron_property_presence & 1

    assert copied.GetAtomWithIdx(1).GetNumRadicalElectrons() == count
    assert copied.GetAtomWithIdx(1).GetIntProp(UNPAIRED) == count
    assert _state(source) == before
    copied_before = _state(copied)

    result = equivalence.evaluate_equivalence(source, copied, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.raw_mol1 is not None
    assert result.raw_mol2 is not None
    assert result.raw_mol1.rdkit_radical_electrons == count
    assert result.raw_mol2.rdkit_radical_electrons == count
    assert result.checks is not None
    assert result.checks.radical_electrons.mol1 == count
    assert result.checks.radical_electrons.mol2 == count
    for side in ("mol1", "mol2"):
        effects = result.normalization_electronic_effects[side]
        assert effects["raw_rdkit_radical_electrons"] == count
        assert effects["standardized_rdkit_radical_electrons"] == count
    assert _state(source) == before
    assert _state(copied) == copied_before


@pytest.mark.parametrize("count", [0, 1, 2])
def test_sdf_source_assignment_is_decoded_before_comparison(count: int) -> None:
    source = _mol("F[B-]F")
    boron = source.GetAtomWithIdx(1)
    boron.SetNumRadicalElectrons(count)
    boron.SetIntProp(UNPAIRED, count)
    Chem.CreateAtomIntPropertyList(source, UNPAIRED)
    buffer = io.StringIO()
    with Chem.SDWriter(buffer) as writer:
        writer.write(source)
    restored = next(Chem.ForwardSDMolSupplier(
        io.BytesIO(buffer.getvalue().encode()), sanitize=True, removeHs=False,
    ))
    assert restored is not None
    before = (_state(source), _state(restored))

    result = equivalence.evaluate_equivalence(source, restored, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.checks.radical_electrons.mol1 == count
    assert result.checks.radical_electrons.mol2 == count
    assert result.raw_mol2.molgr_unpaired_electrons == ((1, count),)
    assert (_state(source), _state(restored)) == before


def test_normalization_keeps_working_electron_move_not_original_source_count() -> None:
    source = _mol("C=[Fe]")
    source.GetAtomWithIdx(0).SetIntProp(UNPAIRED, 0)
    other = _mol("[CH2]->[Fe]")
    other.GetAtomWithIdx(0).SetIntProp(UNPAIRED, 2)
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.normalization_electronic_effects["mol1"]["standardized_rdkit_radical_electrons"] == 2
    assert result.checks.radical_electrons.mol1 == result.checks.radical_electrons.mol2 == 2
    assert (_state(source), _state(other)) == before


@pytest.mark.parametrize("smiles", ["F[B-]F", "CCO", "[CH2]C=CO"])
@pytest.mark.parametrize("reverse", [False, True])
def test_missing_nonmetal_unpaired_annotation_remains_unknown_without_veto(
    smiles: str, reverse: bool,
) -> None:
    baseline = _mol(smiles)
    annotated = Chem.Mol(baseline)
    for atom in annotated.GetAtoms():
        atom.SetIntProp(UNPAIRED, atom.GetNumRadicalElectrons())
    source, other = (baseline, annotated) if reverse else (annotated, baseline)
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert all(not atom.HasProp(UNPAIRED) for atom in baseline.GetAtoms())
    assert (_state(source), _state(other)) == before


@pytest.mark.parametrize("reverse", [False, True])
def test_known_nonmetal_unpaired_provenance_allows_elementary_radical_resonance(
    reverse: bool,
) -> None:
    source = _mol("[CH2]C=CO")
    other = _mol("C=C[CH]O")
    _set_counts(source, UNPAIRED, (1, 0, 0, 0))
    _set_counts(other, UNPAIRED, (0, 0, 1, 0))
    if reverse:
        source, other = other, source
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=False)

    assert result.decision == equivalence.EquivalenceDecision.EQUIVALENT
    assert result.relation == equivalence.EquivalenceRelation.RESONANCE_EQUIVALENCE
    assert result.raw_mol1 is not None
    assert result.raw_mol2 is not None
    assert result.raw_mol1.rdkit_radical_electrons == 1
    assert result.raw_mol2.rdkit_radical_electrons == 1
    assert result.checks is not None
    assert result.checks.radical_electrons.mol1 == 1
    assert result.checks.radical_electrons.mol2 == 1
    assert (_state(source), _state(other)) == before


@pytest.mark.parametrize("compatible", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_asymmetric_charge_resonance_requires_compatible_active_lone_pairs(
    compatible: bool, reverse: bool,
) -> None:
    source = _mol("[CH2-]C=O")
    other = _mol("C=C[O-]")
    _set_counts(source, LONE_PAIRS, (1, 0, 0))
    _set_counts(other, LONE_PAIRS, (0, 0, 1) if compatible else (0, 1, 0))
    _set_counts(source, UNPAIRED, (0, 0, 0))
    _set_counts(other, UNPAIRED, (0, 0, 0))
    for idx, atom in enumerate(source.GetAtoms()):
        atom.SetAtomMapNum(idx + 11)
    for idx, atom in enumerate(other.GetAtoms()):
        atom.SetAtomMapNum(idx + 61)
    if reverse:
        source, other = other, source
    before = (_state(source), _state(other))

    result = equivalence.evaluate_equivalence(source, other, use_chirality=False)

    expected = (
        equivalence.EquivalenceDecision.EQUIVALENT
        if compatible
        else equivalence.EquivalenceDecision.INCONCLUSIVE
    )
    assert result.decision == expected
    if compatible:
        assert result.invariants["electronic_properties"].status == equivalence.InvariantStatus.PASSED
    assert (_state(source), _state(other)) == before
