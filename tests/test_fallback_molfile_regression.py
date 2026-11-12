# pyright: reportMissingImports=false

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem


pytest.importorskip("openbabel")

from molgr.config import MolGRConfig
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import check_equivalence


_MOLFILE_CASES_SPEC = importlib.util.spec_from_file_location(
    "molgr_cases_molfile",
    Path("scripts/molgr_cases_molfile.py").resolve(),
)
_MOLFILE_CASES_MODULE = importlib.util.module_from_spec(_MOLFILE_CASES_SPEC)
assert _MOLFILE_CASES_SPEC.loader is not None
_MOLFILE_CASES_SPEC.loader.exec_module(_MOLFILE_CASES_MODULE)
load_molfile_cases = _MOLFILE_CASES_MODULE.load_molfile_cases


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_monnmo_molfile_regression(backend: str) -> None:
    case = load_molfile_cases(Path("tests/data/sdf/MoNNMo.sdf"), limit=1)[0]
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    result = xyz_to_rdmol(
        case["xyz_block"],
        total_charge=case["total_charge"],
        spin_multiplicity=case["total_radical_electrons"] + 1,
        backend=backend,
        config=config,
    )

    equivalent, info = check_equivalence(
        case["ground_truth_rdmol"], result, ignore_metal_coordination_bonds=False
    )
    assert equivalent, info.reason
    assert Chem.GetFormalCharge(result) == 0
    metals = [atom for atom in result.GetAtoms() if atom.GetSymbol() == "Mo"]
    assert len(metals) == 2
    assert all(atom.GetFormalCharge() == 3 for atom in metals)
    assert all(atom.GetIntProp("MOLGR_METAL_UNPAIRED_ELECTRONS") == 1 for atom in metals)
    assert all(
        atom.GetNumRadicalElectrons() == 0 for atom in result.GetAtoms() if atom.GetSymbol() != "Mo"
    )
    assert sum(bond.GetBondType() == Chem.BondType.DATIVE for bond in result.GetBonds()) == 10
    assert np.allclose(
        result.GetConformer().GetPositions(),
        case["ground_truth_rdmol"].GetConformer().GetPositions(),
        atol=1e-5,  # The Python backend's Open Babel conversion rounds coordinates.
        rtol=0,
    )


@pytest.mark.parametrize("backend,threads", [("python", 1), ("cpp", 1), ("cpp", 2)])
def test_338_s_reconstruction_matches_reference(backend: str, threads: int) -> None:
    xyz = (Path(__file__).parent / "data" / "xyz/338-S.xyz").read_text()
    fixture = next(
        iter(
            Chem.SDMolSupplier(
                str(Path(__file__).parent / "data" / "sdf/338-S.sdf"),
                removeHs=False,
                sanitize=False,
            )
        )
    )
    assert fixture is not None
    reference = Chem.AddHs(Chem.MolFromSmiles(fixture.GetProp("SOURCE_SMILES")))
    config = MolGRConfig()
    config.cpp_backend.max_threads = threads
    result = xyz_to_rdmol(xyz, 0, 1, backend=backend, config=config)
    equivalent, info = check_equivalence(
        reference, result, use_chirality=True, ignore_metal_coordination_bonds=False
    )
    assert equivalent, info.reason
    assert result.GetNumAtoms() == 67
    assert result.GetNumBonds() == 89
    assert Chem.GetFormalCharge(result) == 0
    assert sum(b.GetBondType() == Chem.BondType.DATIVE for b in result.GetBonds()) == 24
    metals = [a for a in result.GetAtoms() if a.GetSymbol() in ("Fe", "Pd")]
    assert [(a.GetSymbol(), a.GetFormalCharge()) for a in metals] == [
        ("Fe", 2),
        ("Pd", 2),
        ("Fe", 2),
    ]
    assert [a.GetDegree() for a in metals] == [10, 4, 10]
    # Remove coordination edges before identifying the four carbon-only Cp rings.
    organic = Chem.RWMol(result)
    for atom in reversed(metals):
        organic.RemoveAtom(atom.GetIdx())
    cp_rings = [
        ring
        for ring in Chem.GetSymmSSSR(organic)
        if len(ring) == 5 and all(organic.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring)
    ]
    assert len(cp_rings) == 4
    assert sorted(
        sum(organic.GetAtomWithIdx(i).GetFormalCharge() for i in ring) for ring in cp_rings
    ) == [-2, -1, -1, -1]
    assert [a.GetFormalCharge() for a in organic.GetAtoms() if a.GetSymbol() == "O"].count(-1) == 1
    cp_atoms = {i for ring in cp_rings for i in ring}
    assert (
        sum(
            a.GetFormalCharge()
            for a in organic.GetAtoms()
            if a.GetIdx() not in cp_atoms and a.GetSymbol() != "O"
        )
        == 0
    )
    assert all(a.GetIntProp("MOLGR_METAL_UNPAIRED_ELECTRONS") == 0 for a in metals)
    assert all(a.GetNumRadicalElectrons() == 0 for a in result.GetAtoms())
