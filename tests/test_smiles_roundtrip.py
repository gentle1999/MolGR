from __future__ import annotations

import csv
from pathlib import Path

import pytest
from rdkit import Chem

from molgr import (
    MolGRConfig,
    ReconstructionBatchRequest,
    ReconstructionError,
    ReconstructionFailureCode,
    SmilesRoundtripError,
    _core,
    iter_xyz_to_rdmol_batch,
    mol_from_smiles,
    validate_smiles_roundtrip,
)
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import _sanitize_preserving_explicit_bonds, refresh_rdkit_aromaticity
from molgr.utils.smiles import _compare_mapped_graph, normalize_smiles_hydrogens


DATA = Path(__file__).parent / "data"
with (DATA / "xyz_stress_manifest.csv").open() as handle:
    CASES = list(csv.DictReader(handle))

# Standalone synthetic water geometry, independent of external input datasets.
WATER_XYZ = "3\nwater\nO 0 0 0\nH 0.9572 0 0\nH -0.2390 0.9270 0\n"


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_existing_public_inputs_preserve_the_explicit_h_graph(case, backend):
    xyz = (DATA / case["xyz_file"]).read_text()
    mol = xyz_to_rdmol(xyz, int(case["charge"]), int(case["spin_multiplicity"]), backend=backend)
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    smiles = validate_smiles_roundtrip(mol)
    assert smiles == Chem.MolToSmiles(Chem.Mol(mol))
    params = Chem.SmilesParserParams()
    params.removeHs = False
    parsed = Chem.MolFromSmiles(smiles, params)
    assert parsed is not None
    assert parsed.GetNumAtoms() == mol.GetNumAtoms() == int(xyz.splitlines()[0])
    assert parsed.GetNumBonds() == mol.GetNumBonds()
    assert all(a.GetNoImplicit() and a.GetNumImplicitHs() == 0 for a in mol.GetAtoms())
    assert all(a.GetNoImplicit() for a in parsed.GetAtoms())
    assert sum(a.GetAtomicNum() == 1 for a in parsed.GetAtoms()) == sum(
        a.GetAtomicNum() == 1 for a in mol.GetAtoms()
    )
    assert all(a.GetNumImplicitHs() == 0 and a.GetNumExplicitHs() == 0 for a in parsed.GetAtoms())
    # Further native readbacks must remain stable without restoring MolGR
    # electron annotations or relying on the reader's inferred radical count.
    for _ in range(2):
        parsed = Chem.MolFromSmiles(validate_smiles_roundtrip(parsed), params)
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_stale_aromatic_flags_do_not_change_explicit_double_bonds():
    mol = Chem.MolFromSmiles("C1=CC=CC1")
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(True)
    for bond in mol.GetBonds():
        bond.SetIsAromatic(True)
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    expected = [b.GetBondType() for b in mol.GetBonds()]
    result = _sanitize_preserving_explicit_bonds(mol, kekulize=True)
    assert [b.GetBondType() for b in result.GetBonds()] == expected
    assert not any(b.GetIsAromatic() for b in result.GetBonds())
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_failed_aromatic_conversion_does_not_modify_the_source():
    mol = Chem.MolFromSmiles("c1cccc1", sanitize=False)
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    with pytest.raises(Chem.KekulizeException):
        _sanitize_preserving_explicit_bonds(mol, kekulize=True)
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.parametrize("integer_bonds", [True, False])
def test_aromaticity_refresh_preserves_electron_state(integer_bonds):
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1.[C]"))
    if integer_bonds:
        Chem.Kekulize(mol)
    center = mol.GetAtomWithIdx(6)
    center.SetNumRadicalElectrons(0)
    center.SetIntProp("MOLGR_UNPAIRED_ELECTRON_COUNT", 0)
    center.SetIntProp("MOLGR_LONE_PAIR_COUNT", 1)
    before = [atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()]
    refresh_rdkit_aromaticity(mol)
    assert [atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()] == before
    assert center.GetIntProp("MOLGR_UNPAIRED_ELECTRON_COUNT") == 0
    assert center.GetIntProp("MOLGR_LONE_PAIR_COUNT") == 1
    assert all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(6))
    normalize_smiles_hydrogens(mol)
    validate_smiles_roundtrip(mol)


def test_aromaticity_refresh_rolls_back_graph_and_radicals_on_failure(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1.[C]"))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol.GetAtomWithIdx(6).SetNumRadicalElectrons(0)

    def state():
        return (
            [(a.GetNumRadicalElectrons(), a.GetIsAromatic()) for a in mol.GetAtoms()],
            [(b.GetBondType(), b.GetIsAromatic(), b.GetIsConjugated()) for b in mol.GetBonds()],
        )

    before = state()
    sanitize = Chem.SanitizeMol

    def fail_after_perception(*args, **kwargs):
        sanitize(*args, **kwargs)
        assert state() != before
        raise RuntimeError("perception failed after changing the graph")

    monkeypatch.setattr(Chem, "SanitizeMol", fail_after_perception)
    with pytest.raises(RuntimeError, match="perception failed"):
        refresh_rdkit_aromaticity(mol)
    assert state() == before


def test_aromaticity_refresh_failed_kekulization_preserves_source():
    mol = Chem.MolFromSmiles("c1cccc1", sanitize=False)
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    with pytest.raises(Chem.KekulizeException):
        refresh_rdkit_aromaticity(mol)
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.parametrize("mapping", [[], [0, 0], [-1, 1], [0, 2]])
def test_native_stereo_comparison_rejects_invalid_mappings(mapping):
    from molgr.utils.smiles import _stereo_data

    data = _stereo_data(Chem.MolFromSmiles("CO"))
    with pytest.raises(ValueError, match="mapping"):
        _core.equivalence.compare_mapped_stereo(data, data, mapping)


@pytest.mark.parametrize("smiles", ["[C]", "[CH2]", "[CH2]C", "[NH]", "[O]"])
def test_zero_hydrogen_normalization_preserves_actual_electron_assignments(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(0)
        atom.SetIntProp("MOLGR_UNPAIRED_ELECTRON_COUNT", 0)
    expected_h = sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
    normalize_smiles_hydrogens(mol)
    normalized = validate_smiles_roundtrip(mol)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    parsed = Chem.MolFromSmiles(normalized, params)
    assert sum(a.GetAtomicNum() == 1 for a in parsed.GetAtoms()) == expected_h
    assert all(a.GetTotalNumHs() == 0 for a in parsed.GetAtoms())
    assert all(a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms())
    assert all(a.GetIntProp("MOLGR_UNPAIRED_ELECTRON_COUNT") == 0 for a in mol.GetAtoms())
    assert any(a.GetNumRadicalElectrons() != 0 for a in parsed.GetAtoms())


def test_validation_does_not_ignore_hydrogen_changes():
    mol = Chem.MolFromSmiles("[C]")
    mol.GetAtomWithIdx(0).SetNumRadicalElectrons(0)
    with pytest.raises(SmilesRoundtripError) as raised:
        validate_smiles_roundtrip(mol)
    assert raised.value.details["field"] == "hydrogens"
    assert raised.value.details["before"] == 0
    assert raised.value.details["after"] == 4


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_default_output_checks_aromaticity_without_stereo_perception(backend):
    xyz = (DATA / "xyz" / "aromatic_sulfur_oxide.xyz").read_text()
    mol = xyz_to_rdmol(xyz, 0, 1, backend=backend, make_stereochemistry=False)
    validate_smiles_roundtrip(mol)
    assert any(bond.GetIsAromatic() for bond in mol.GetBonds())


def test_aromatic_triple_bond_retains_order_after_refresh_and_readback():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc#cc1"))
    Chem.Kekulize(mol)
    refresh_rdkit_aromaticity(mol)
    normalize_smiles_hydrogens(mol)
    smiles = validate_smiles_roundtrip(mol)
    assert "#" in smiles
    for graph in (mol, mol_from_smiles(smiles)):
        triples = [bond for bond in graph.GetBonds() if bond.GetBondType() == Chem.BondType.TRIPLE]
        assert len(triples) == 1
        assert triples[0].GetIsAromatic()


@pytest.mark.parametrize(
    "smiles",
    [
        "F/C=C/F",
        "F/C=C\\F",
        "N[C@@H](C)O",
        "[2H][C@H](F)Cl",
        "c1ccc#cc1",
        "[NH3]->[Cu+2]<-[NH3]",
        "[CH3]",
        "[13CH4]",
    ],
)
def test_valid_graphs_and_stereo_round_trip(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    # Existing labels must survive, without using them to change the traversal.
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(100 + i)
    validate_smiles_roundtrip(mol, check_stereochemistry=True)


@pytest.mark.parametrize("smiles", ["C", "[CH4]"])
def test_validation_requires_hydrogen_atom_nodes(smiles):
    with pytest.raises(SmilesRoundtripError, match="explicit atom nodes"):
        validate_smiles_roundtrip(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize(
    "count,template",
    [
        (3, "F[Pt@SP{n}](Cl)(Br)I"),
        (20, "F[As@TB{n}](Cl)(Br)(I)N"),
        (30, "F[Co@OH{n}](Cl)(Br)(I)(N)P"),
    ],
)
def test_all_non_tetrahedral_permutations_and_fixed_mapping(count, template):
    for i in range(1, count + 1):
        mol = Chem.AddHs(Chem.MolFromSmiles(template.format(n=i)))
        validate_smiles_roundtrip(mol, check_stereochemistry=True)
        other = Chem.AddHs(Chem.MolFromSmiles(template.format(n=i % count + 1)))
        with pytest.raises(SmilesRoundtripError, match="stereochemistry"):
            _compare_mapped_graph(mol, other, list(range(mol.GetNumAtoms())))


def test_unserializable_small_ring_stereo_is_not_silently_removed():
    mol = Chem.AddHs(Chem.MolFromSmiles("C1=CCCCC1"))
    bond = mol.GetBondBetweenAtoms(0, 1)
    bond.SetStereoAtoms(5, 2)
    bond.SetStereo(Chem.BondStereo.STEREOE)
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    validate_smiles_roundtrip(mol)
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    with pytest.raises(SmilesRoundtripError, match="stereochemistry"):
        validate_smiles_roundtrip(mol, check_stereochemistry=True)


def test_enhanced_relative_stereo_is_not_silently_projected_as_absolute():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](F)[C@@H](Cl)Br |&1:1,3|"))
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    validate_smiles_roundtrip(mol)
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    with pytest.raises(SmilesRoundtripError, match="stereochemistry"):
        validate_smiles_roundtrip(mol, check_stereochemistry=True)


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_lossy_outputs_follow_failure_policy_and_preserve_opt_out(backend, monkeypatch):
    bad = Chem.AddHs(Chem.MolFromSmiles("CC"))
    bad.GetBondBetweenAtoms(0, 1).SetBondType(Chem.BondType.ZERO)
    monkeypatch.setattr("molgr.interface.restore_stereochemistry", lambda mol: Chem.Mol(bad))
    xyz = WATER_XYZ
    config = MolGRConfig()
    with pytest.raises(ReconstructionError) as raised:
        xyz_to_rdmol(xyz, 0, 1, backend=backend, config=config)
    assert raised.value.diagnostics.code == ReconstructionFailureCode.SMILES_ROUNDTRIP_FAILED
    assert raised.value.diagnostics.stage == "rdkit.smiles_roundtrip"
    config.interface.reconstruction_failure_policy = "return_suspicious"
    mol = xyz_to_rdmol(xyz, 0, 1, backend=backend, config=config)
    assert mol.GetProp("_MolGRReconstructionStatus") == "suspicious_fallback"
    assert mol.GetProp("_MolGRReconstructionFailureCode") == "SMILES_ROUNDTRIP_FAILED"
    config.interface.require_smiles_roundtrip = False
    mol = xyz_to_rdmol(xyz, 0, 1, backend=backend, config=config)
    assert not mol.HasProp("_MolGRReconstructionStatus")
    assert mol.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.ZERO
    with pytest.raises(SmilesRoundtripError):
        validate_smiles_roundtrip(mol)


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_batch_reports_lossy_items_and_continues(backend, monkeypatch):
    from molgr.utils.post_process import make_stereochemistry

    calls = iter([True, False])

    def perceive(mol):
        if not next(calls):
            return make_stereochemistry(mol)
        bad = Chem.AddHs(Chem.MolFromSmiles("CC"))
        bad.GetBondBetweenAtoms(0, 1).SetBondType(Chem.BondType.ZERO)
        return bad

    monkeypatch.setattr("molgr.interface.restore_stereochemistry", perceive)
    requests = [ReconstructionBatchRequest(WATER_XYZ, 0, 1) for _ in range(2)]
    results = list(iter_xyz_to_rdmol_batch(requests, backend=backend, max_workers=1))
    assert results[0].result is None
    assert results[0].status.code == ReconstructionFailureCode.SMILES_ROUNDTRIP_FAILED
    assert results[1].result is not None
    assert results[1].status is None


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_aluminum_covalent_representation_roundtrips_with_native_rdkit(backend):
    path = DATA / "xyz" / "geometry-01a03dd8-a5ef-7c30-901a-3c8c4442b2f9.xyz"
    mol = xyz_to_rdmol(path.read_text(), 1, 2, backend=backend)
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    parsed = mol
    for _ in range(3):
        parsed = Chem.MolFromSmiles(validate_smiles_roundtrip(parsed), params)
        assert parsed.GetNumAtoms() == 19
        assert parsed.GetNumBonds() == mol.GetNumBonds()
        assert Chem.GetFormalCharge(parsed) == 1
        assert all(a.GetTotalNumHs() == 0 for a in parsed.GetAtoms())
        (aluminum,) = [a for a in parsed.GetAtoms() if a.GetAtomicNum() == 13]
        assert aluminum.GetFormalCharge() == 0
        assert aluminum.GetDegree() == 3
        assert all(b.GetBondType() == Chem.BondType.SINGLE for b in aluminum.GetBonds())
        radicals = [a for a in parsed.GetAtoms() if a.GetNumRadicalElectrons()]
        assert len(radicals) == 1
        assert radicals[0].GetSymbol() == "O"
        assert radicals[0].GetFormalCharge() == 0
        assert radicals[0].GetNumRadicalElectrons() == 1
        assert parsed.GetBondBetweenAtoms(radicals[0].GetIdx(), aluminum.GetIdx()) is None
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.parametrize("ion", ["Fe+3", "Cu+2", "Zn+2", "Pt+2"])
@pytest.mark.parametrize("coordination", [1, 4, 6])
def test_transition_metals_keep_native_dative_representation(ion, coordination):
    smiles = "[" + ion + "]" + "(<-[O]([H])[H])" * coordination
    mol = mol_from_smiles(smiles)
    assert mol.GetNumAtoms() == 1 + coordination * 3
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == int(ion[-1])
    assert sum(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds()) == coordination
    validate_smiles_roundtrip(mol)


@pytest.mark.parametrize(
    "invalid",
    [
        "[C](F)(F)(F)(F)F",  # carbon retains all five covalent bonds
        "[N](F)(F)(F)F",  # neutral nitrogen remains strict
        "[Al+3](F)(F)(F)(F)<-[O]",  # incoming dative cannot excuse covalent overvalence
        "[Al+3](F)(F)(F)F",  # no incoming dative on this failing metal
        "[C](F)(F)(F)(F)<-[O]",  # incoming dative into a nonmetal is not exempt
        "[Al+3](F)(F)(F)(F)->[O]",  # outgoing dative is not an exemption
        "c1cccc1",  # failed Kekulization is still fatal
    ],
)
def test_native_reader_rejects_invalid_valence_and_aromaticity(invalid):
    with pytest.raises(SmilesRoundtripError):
        mol_from_smiles(invalid)


def test_coordination_reader_rejects_malformed_smiles():
    with pytest.raises(SmilesRoundtripError):
        mol_from_smiles("[Al+3](<-[O]")


@pytest.mark.parametrize(
    "fragment",
    [
        "[2H][C@]([F])([Cl])[Br]",
        "[F]/[C]([H])=[C]([H])/[Cl]",
        "[c:1]1([H])[c]([H])[c]([H])[c]([H])[c]([H])[c]1[H]",
    ],
)
def test_native_reader_perceives_organic_stereo_and_aromaticity(fragment):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    reference = Chem.MolFromSmiles(fragment, params)
    mol = mol_from_smiles(fragment + ".[Fe+3]" + "(<-[O]([H])[H])" * 4)
    restored = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)[0]
    _compare_mapped_graph(reference, restored, list(range(reference.GetNumAtoms())))
    native = mol_from_smiles(fragment)
    _compare_mapped_graph(reference, native, list(range(reference.GetNumAtoms())))
