from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils._molecule_stereo import apply_rdkit_stereo
from molgr.utils.converter import refresh_rdkit_aromaticity
from molgr.utils.equivalence import check_equivalence


DATA = Path(__file__).parent / "data"
FIXTURES = DATA / "reviewed/reconstruction"
CASES = json.loads((FIXTURES / "manifest.json").read_text())["fixtures"]


def _approved_graph(case):
    mol = next(
        iter(
            Chem.SDMolSupplier(
                str(FIXTURES / case["structure_file"]), sanitize=False, removeHs=False
            )
        )
    )
    assert mol is not None
    mol.UpdatePropertyCache(strict=False)
    refresh_rdkit_aromaticity(mol)
    # SDF/SMILES readers can reperceive or lose coordination stereochemistry.
    # Restore the frozen reference frames, without running today's 3D perception
    # on both the reference and candidate (which could hide a stereo regression).
    mol = Chem.RWMol(mol)
    Chem.RemoveStereochemistry(mol)
    # RDKit 2024 can infer atropisomer tags from SDF wedges and leave them
    # behind in RemoveStereochemistry. Replace all reader-derived bond stereo.
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    frames = []
    for record in json.loads(mol.GetProp("APPROVED_STEREO")):
        frame = _core.utils.StereoData()
        frame.kind = getattr(_core.utils.StereoKind, record["kind"])
        frame.center, frame.end = record["center"], record["end"]
        frame.refs, frame.configuration = record["refs"], record["configuration"]
        frames.append(frame)
    apply_rdkit_stereo(mol, SimpleNamespace(stereo=frames, stereo_groups=[]))
    return mol.GetMol()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_reviewed_graph_is_bound_to_its_input_and_electronic_state(case):
    assert hashlib.sha256((DATA / case["xyz_file"]).read_bytes()).hexdigest() == case["xyz_sha256"]
    reference = _approved_graph(case)
    assert reference.GetProp("CASE_ID") == case["case_id"]
    assert reference.GetProp("REVIEW_STATUS") == "accept_candidate"
    assert Chem.GetFormalCharge(reference) == case["total_charge"]
    assert reference.GetIntProp("TOTAL_CHARGE") == case["total_charge"]
    assert reference.GetIntProp("SPIN_MULTIPLICITY") == case["spin_multiplicity"]
    assert reference.HasProp("APPROVED_SMILES") == case["smiles_stereo_roundtrip"]
    if reference.HasProp("APPROVED_SMILES"):
        params = Chem.SmilesParserParams()
        params.removeHs = False
        parsed = Chem.MolFromSmiles(reference.GetProp("APPROVED_SMILES"), params)
        equivalent, info = check_equivalence(
            reference, parsed, use_chirality=True, ignore_metal_coordination_bonds=False
        )
        assert equivalent, info.reason


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_reconstruction_matches_user_approved_graph(case, backend):
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    result = xyz_to_rdmol(
        (DATA / case["xyz_file"]).read_text(),
        case["total_charge"],
        case["spin_multiplicity"],
        backend=backend,
        config=config,
    )
    reference = _approved_graph(case)
    equivalent, info = check_equivalence(
        reference, result, use_chirality=True, ignore_metal_coordination_bonds=False
    )
    assert equivalent, info.reason
    assert result.GetNumAtoms() == reference.GetNumAtoms()
    assert result.GetNumBonds() == reference.GetNumBonds()
