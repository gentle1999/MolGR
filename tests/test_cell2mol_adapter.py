"""The optional cell2mol dependency is represented by its public result shape."""

from types import SimpleNamespace

import pytest
from rdkit import Chem

from benchmarks.smiles_xyz_benchmark.methods.cell2mol_v2 import _extract_rdmol


def test_complex_without_top_level_rdkit_object_preserves_assigned_graph():
    # v2 only puts rdkit_obj on the two monatomic ligands.
    atoms = [
        SimpleNamespace(label=s, charge=q, coord=(float(i), 0, 0), bonds=[])
        for i, (s, q) in enumerate(zip(("Cl", "Cu", "Cl"), (-1, 2, -1)))
    ]
    for i in (0, 2):
        bond = SimpleNamespace(atom1=atoms[1], atom2=atoms[i], order=0)
        atoms[1].bonds.append(bond)
        atoms[i].bonds.append(bond)
    molecule = SimpleNamespace(
        atoms=atoms,
        # A stale top-level vector must not override final atom assignments.
        atomic_charges=[0, 2, 0],
        metals=[atoms[1]],
        ligands=[
            SimpleNamespace(atoms=[atoms[i]], rdkit_obj=Chem.MolFromSmiles("[Cl-]")) for i in (0, 2)
        ],
    )
    mol = _extract_rdmol(molecule)
    assert Chem.MolToSmiles(mol) == "[Cl-]->[Cu+2]<-[Cl-]"
    assert mol.GetNumAtoms() == 3
    assert mol.GetNumBonds() == 2
    assert mol.GetConformer().GetAtomPosition(2).x == 2
    assert all(a.GetNumImplicitHs() == 0 for a in mol.GetAtoms())


def test_failed_assignment_is_not_promoted_to_a_reconstruction():
    molecule = SimpleNamespace(error_multiple_distrib=True, rdkit_obj=Chem.MolFromSmiles("O"))
    with pytest.raises(ValueError, match="error_multiple_distrib"):
        _extract_rdmol(molecule)


def test_determine_bonds_failure_retains_graph_and_error_after_postprocessing(monkeypatch):
    from benchmarks.smiles_xyz_benchmark.methods import rdkit_determine_bonds as adapter

    def fail(mol, **kwargs):
        mol.GetAtomWithIdx(0).SetFormalCharge(2)
        raise ValueError("unsupported test metal")

    def postprocess(mol):
        assert mol.GetAtomWithIdx(0).GetFormalCharge() == 2
        return mol, Chem.MolToSmiles(mol)

    monkeypatch.setattr(adapter.rdDetermineBonds, "DetermineBonds", fail)
    monkeypatch.setattr(adapter, "finalize_rdmol_with_dative_bonds", postprocess)
    output = adapter.RDKitDetermineBondsMethod().run(
        {"xyz_block": "1\n\nFe 0 0 0\n", "total_charge": 0}
    )
    assert output.status == "error"
    assert output.rdkit_mol.GetAtomWithIdx(0).GetFormalCharge() == 2
    assert output.predicted_smiles
    assert "unsupported test metal" in output.error
    assert "postprocess_ms" in output.timing_ms_breakdown


def test_error_graph_is_compared_even_without_serialized_smiles():
    from benchmarks.dataset_xyz_benchmark.comparison import compare_output
    from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput

    reference = {"reference_status": "valid", "ground_truth_rdmol": Chem.MolFromSmiles("O")}
    output = MethodRunOutput(status="error", error="API failed", rdkit_mol=Chem.MolFromSmiles("O"))
    assert compare_output(reference, output)["decision"] == "equivalent"
    assert output.status == "error" and output.error == "API failed"
    assert compare_output(reference, MethodRunOutput(status="error")) == {}
    assert compare_output({**reference, "reference_status": "excluded"}, output) == {}


def test_direct_organic_exact_element_query_preserves_graph_and_coordinates():
    from rdkit.Chem import rdqueries

    mol = Chem.MolFromSmiles("[13C@H](F)(Cl)Br")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (i, i / 2, i / 3))
    mol.AddConformer(conf)
    query = Chem.RWMol(mol)
    atom = rdqueries.AtomNumEqualsQueryAtom(6)
    atom.SetAtomicNum(6)
    source = mol.GetAtomWithIdx(0)
    atom.SetIsotope(source.GetIsotope())
    atom.SetNumExplicitHs(source.GetNumExplicitHs())
    atom.SetChiralTag(source.GetChiralTag())
    atom.SetNoImplicit(source.GetNoImplicit())
    query.ReplaceAtom(0, atom)
    result = _extract_rdmol(SimpleNamespace(rdkit_obj=query))
    assert not any(a.HasQuery() for a in result.GetAtoms())
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(mol)
    assert result.GetConformer().GetPositions() == pytest.approx(mol.GetConformer().GetPositions())
    assert query.GetAtomWithIdx(0).HasQuery()  # Leave the upstream object intact.


def test_direct_organic_ambiguous_query_is_not_fabricated_into_a_graph():
    with pytest.raises(ValueError, match="unresolved atom query"):
        _extract_rdmol(SimpleNamespace(rdkit_obj=Chem.MolFromSmarts("[C,N]")))
