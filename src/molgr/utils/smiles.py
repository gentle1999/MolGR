"""Strict validation of RDKit's SMILES projection with explicit H retained.

Atom indices may change; explicit hydrogen nodes must survive. Element,
isotope, formal charge, connectivity, bond order,
aromaticity and dative direction must not change. Stereo round-trip checking
is optional: reconstruction assigns stereo from coordinates, not SMILES readback.
Coordinates and MolGR's custom electronic annotations are outside ordinary SMILES.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from rdkit import Chem

from molgr import _core as core
from molgr.utils._molecule_stereo import extract_rdkit_stereo


class SmilesRoundtripError(ValueError):
    """SMILES reading or round-trip graph validation failed."""

    def __init__(self, message: str, *, details: Optional[Dict[str, object]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


def _stereo_data(mol: Chem.Mol) -> core.utils.MoleculeData:
    """Project an RDKit graph onto the atom references needed to compare stereo frames."""
    # Only topology and reference frames are needed by the native stereo
    # comparison. Avoid Kekulization, radical inference or stereo cleanup here:
    # they could erase precisely the source information we need to validate.
    atoms = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atoms.append(
            (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), 0, 0, False, 0)
        )
    bonds = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        bonds.append(
            (
                bond.GetBeginAtomIdx() + 1,
                bond.GetEndAtomIdx() + 1,
                2 if bond.GetBondType() == Chem.BondType.DOUBLE else 1,
                False,
                0,
            )
        )
    data = core.utils._molecule_data_from_rows(atoms, bonds, None)
    data.stereo, data.stereo_groups = extract_rdkit_stereo(mol)
    return data


def _compare_mapped_graph(
    source: Chem.Mol, restored: Chem.Mol, mapping: List[int], *, check_stereochemistry: bool = True
) -> None:
    """Check the round trip under the writer's zero-based source-to-restored atom mapping."""
    if source.GetNumAtoms() != restored.GetNumAtoms():
        raise SmilesRoundtripError("SMILES round trip changed the atom count")
    if source.GetNumBonds() != restored.GetNumBonds():
        raise SmilesRoundtripError("SMILES round trip changed the bond count")
    fields = {
        "atomic_number": Chem.Atom.GetAtomicNum,
        "isotope": Chem.Atom.GetIsotope,
        "formal_charge": Chem.Atom.GetFormalCharge,
        "aromatic": Chem.Atom.GetIsAromatic,
        "hydrogens": Chem.Atom.GetTotalNumHs,
        "atom_map": Chem.Atom.GetAtomMapNum,
    }
    for i, j in enumerate(mapping):
        before, after = source.GetAtomWithIdx(i), restored.GetAtomWithIdx(j)
        for field, getter in fields.items():
            a, b = getter(before), getter(after)
            if a != b:
                raise SmilesRoundtripError(
                    f"SMILES round trip changed atom {i + 1} {field}: {a} -> {b}",
                    details={"source_atom_number": i + 1, "field": field, "before": a, "after": b},
                )
    # GetBonds()/GetAtoms() use Python iterator wrappers that repeatedly query
    # the molecule size. These graphs are fixed, so direct indexed access avoids
    # that overhead without changing the fields or mapping being checked.
    for bond_idx in range(source.GetNumBonds()):
        bond = source.GetBondWithIdx(bond_idx)
        i, j = mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
        other = restored.GetBondBetweenAtoms(i, j)
        if (
            other is None
            or bond.GetBondType() != other.GetBondType()
            or bond.GetIsAromatic() != other.GetIsAromatic()
            or (bond.GetBondType() == Chem.BondType.DATIVE and other.GetBeginAtomIdx() != i)
        ):
            raise SmilesRoundtripError(
                "SMILES round trip changed a bond",
                details={"source_atoms": [bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1]},
            )
    if check_stereochemistry:
        comparison = core.equivalence.compare_mapped_stereo(
            _stereo_data(source), _stereo_data(restored), mapping
        )
        if comparison != 1:
            raise SmilesRoundtripError(
                "SMILES round trip changed or lost stereochemistry",
                details={"stereo_comparison": comparison},
            )


def normalize_smiles_hydrogens(rdmol: Chem.Mol) -> None:
    """Encode the explicit-H model on every atom in the default SMILES writer.

    SetNoImplicit(True) alone is not a SMILES bracket request: an uncharged
    singlet carbon with no H can be written as bare C, which the reader fills
    with H. RDKit's smilesSymbol property requests bracketed output while
    retaining the actual element, isotope, charge, stereo and H count.
    Apply it to every atom, including saturated and aromatic atoms, so the
    reader also disables implicit H everywhere. RDKit retains aromatic casing.
    This changes neither bonds nor MolGR's real electron assignments. In particular
    we do not invent two radical electrons to make a singlet carbene print [C].
    """
    for i in range(rdmol.GetNumAtoms()):
        atom = rdmol.GetAtomWithIdx(i)
        atom.SetNoImplicit(True)
        atom.SetProp("smilesSymbol", atom.GetSymbol())
    rdmol.UpdatePropertyCache(strict=False)


def _retain_explicit_h_model(mol: Chem.Mol) -> Chem.Mol:
    # A subsequent native write may omit brackets on saturated atoms. Restore
    # the no-implicit-H convention only when all H are already atom nodes;
    # do not erase hydrogen counts in arbitrary user-supplied SMILES.
    if all(mol.GetAtomWithIdx(i).GetTotalNumHs() == 0 for i in range(mol.GetNumAtoms())):
        normalize_smiles_hydrogens(mol)
    return mol


def _read_smiles(smiles: str, *, kekulize: bool, prepare_for_writing: bool) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = not kekulize
    try:
        restored = Chem.MolFromSmiles(smiles, params)
        if restored is None:
            raise SmilesRoundtripError("RDKit could not read the SMILES under its native rules")
        if kekulize:
            # SmilesParserParams only accepts a boolean. Run the selected
            # sanitization steps and finish the parser's stereo assignment.
            flags = Chem.SanitizeFlags
            ops = flags.SANITIZE_ALL & ~flags.SANITIZE_SETAROMATICITY
            Chem.SanitizeMol(restored, sanitizeOps=ops)
            Chem.SetBondStereoFromDirections(restored)
            Chem.AssignStereochemistry(restored, cleanIt=True, force=True)
        return _retain_explicit_h_model(restored) if prepare_for_writing else restored
    except SmilesRoundtripError:
        raise
    except Exception as exc:
        raise SmilesRoundtripError(f"SMILES reading failed: {exc}") from exc


def mol_from_smiles(smiles: str, *, kekulize: bool = True) -> Chem.Mol:
    """Read an RDKit molecule while retaining explicit hydrogen nodes.

    Parameters
    ----------
    smiles : str
        SMILES to parse. Hydrogen atoms written as separate nodes are retained.
    kekulize : bool, default True
        Retain localized bonds and skip aromaticity reassignment during
        sanitization. False uses RDKit's native aromatic SMILES reading.

    Returns
    -------
    Chem.Mol
        Parsed graph. For molecules with all H as atom nodes, MolGR restores
        its no-implicit-H and bracket-writing convention for subsequent writes.

    Raises
    ------
    SmilesRoundtripError
        RDKit could not parse or sanitize the supplied SMILES.
    """
    return _read_smiles(smiles, kekulize=kekulize, prepare_for_writing=True)


def validate_smiles_roundtrip(
    rdmol: Chem.Mol, *, kekulize: bool = True, check_stereochemistry: bool = False
) -> str:
    """Write canonical SMILES and verify that RDKit can recover the explicit-H graph.

    Parameters
    ----------
    rdmol : Chem.Mol
        Source graph, copied for validation. Using the writer's atom order,
        compare atom labels (including isotope, charge and hydrogen count),
        connectivity, bond type, aromaticity and dative-bond direction.
    kekulize : bool, default True
        Read back localized integer bonds. Use False to validate a source
        graph carrying RDKit aromatic bond types.
    check_stereochemistry : bool, default False
        Also compare mapped stereo records. Enable this when the serialized
        SMILES must preserve the stereo assigned from the conformer.

    Returns
    -------
    str
        Canonical SMILES verified by the installed RDKit version, with
        explicit hydrogen nodes retained during readback.

    Raises
    ------
    SmilesRoundtripError
        Writing, reading or graph comparison failed. The exception's
        ``details`` can identify a mismatched field or atom.

    Validation uses the source graph directly and a single SMILES traversal.
    RDKit can infer radical counts when reading SMILES, so those counts are
    outside the graph comparison. The input retains its MolGR electron
    assignments and coordinates.
    """
    source = Chem.Mol(rdmol)
    try:
        atom_count = source.GetNumAtoms()
        for atom_idx in range(atom_count):
            atom = source.GetAtomWithIdx(atom_idx)
            if atom.GetNumImplicitHs() or atom.GetNumExplicitHs():
                raise SmilesRoundtripError(
                    "The source must represent all hydrogens as explicit atom nodes",
                    details={"source_atom_number": atom.GetIdx() + 1},
                )
        smiles = Chem.MolToSmiles(source)
        # Fetch only the writer's order, not every computed/private molecule
        # property. Older RDKit vectors have a trailing comma: accept that
        # empty token rather than treating the representation as JSON.
        order = [
            int(index)
            for index in source.GetProp("_smilesAtomOutputOrder")[1:-1].split(",")
            if index.strip()
        ]
        # This temporary graph is only compared, never written again. Keep the
        # native parser result instead of reapplying bracket/NoImplicit output
        # properties as the public convenience reader does for its callers.
        restored = _read_smiles(smiles, kekulize=kekulize, prepare_for_writing=False)
        if sorted(order) != list(range(atom_count)):
            raise SmilesRoundtripError("SMILES writer did not preserve all source atoms")
        mapping = [0] * len(order)
        for target, original in enumerate(order):
            mapping[original] = target
        _compare_mapped_graph(
            source, restored, mapping, check_stereochemistry=check_stereochemistry
        )
    except SmilesRoundtripError:
        raise
    except Exception as exc:
        raise SmilesRoundtripError(f"SMILES round-trip validation failed: {exc}") from exc
    return smiles


__all__ = ["SmilesRoundtripError", "mol_from_smiles", "validate_smiles_roundtrip"]
