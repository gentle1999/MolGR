"""Strict validation of RDKit's SMILES projection with explicit H retained.

Atom indices may change; explicit hydrogen nodes must survive. Element,
isotope, formal charge, connectivity, bond order,
aromaticity and dative direction must not change. Stereo round-trip checking
is optional: reconstruction assigns stereo from coordinates, not SMILES readback.
Coordinates and MolGR's custom electronic annotations are outside ordinary SMILES.
"""

from __future__ import annotations

from rdkit import Chem

from molgr import _core as core
from molgr.utils._molecule_stereo import extract_rdkit_stereo


class SmilesRoundtripError(ValueError):
    """SMILES reading or round-trip graph validation failed."""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def _stereo_data(mol):
    # Only topology and reference frames are needed by the native stereo
    # comparison. Avoid Kekulization, radical inference or stereo cleanup here:
    # they could erase precisely the source information we need to validate.
    atoms = [
        (a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(), 0, 0, False, 0)
        for a in mol.GetAtoms()
    ]
    bonds = [
        (
            b.GetBeginAtomIdx() + 1,
            b.GetEndAtomIdx() + 1,
            2 if b.GetBondType() == Chem.BondType.DOUBLE else 1,
            False,
            0,
        )
        for b in mol.GetBonds()
    ]
    data = core.utils._molecule_data_from_rows(atoms, bonds, None)
    data.stereo, data.stereo_groups = extract_rdkit_stereo(mol)
    return data


def _compare_mapped_graph(source, restored, mapping, *, check_stereochemistry=True):
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
    for atom in rdmol.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetProp("smilesSymbol", atom.GetSymbol())
    rdmol.UpdatePropertyCache(strict=False)


def _retain_explicit_h_model(mol: Chem.Mol) -> Chem.Mol:
    # A subsequent native write may omit brackets on saturated atoms. Restore
    # the no-implicit-H convention only when all H are already atom nodes;
    # do not erase hydrogen counts in arbitrary user-supplied SMILES.
    if all(atom.GetTotalNumHs() == 0 for atom in mol.GetAtoms()):
        normalize_smiles_hydrogens(mol)
    return mol


def _read_smiles(smiles: str, *, prepare_for_writing: bool) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    try:
        restored = Chem.MolFromSmiles(smiles, params)
        if restored is None:
            raise SmilesRoundtripError("RDKit could not read the SMILES under its native rules")
        return _retain_explicit_h_model(restored) if prepare_for_writing else restored
    except SmilesRoundtripError:
        raise
    except Exception as exc:
        raise SmilesRoundtripError(f"SMILES reading failed: {exc}") from exc


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Read with native RDKit sanitization and explicit hydrogen nodes retained.

    No sanitization rules are bypassed. Valence-limited metal coordination is normalized
    during bond generation, before writing. For graphs with all H as atom nodes,
    retain MolGR's NoImplicit/bracket convention for subsequent writes.
    Failure raises SmilesRoundtripError.
    """
    return _read_smiles(smiles, prepare_for_writing=True)


def validate_smiles_roundtrip(rdmol: Chem.Mol, *, check_stereochemistry: bool = False) -> str:
    """Return canonical SMILES only if its explicit-H graph survives.

    Accept an RDKit Mol and use the currently imported RDKit for both writing
    and native sanitized reading. Success returns the validated SMILES; failure raises
    SmilesRoundtripError, with mismatch information in its details attribute.
    This checks the installed version only, not cross-version compatibility.

    This does not modify the input, substitute a resonance form, change electron
    assignments, or search for a different SMILES traversal. Radical counts
    inferred by the SMILES reader are deliberately not compared. The reader
    uses removeHs=False: removing H is outside MolGR's explicit-H contract.
    Reading uses unmodified native RDKit sanitization.
    Set check_stereochemistry=True to additionally diagnose stereo loss in
    RDKit's writer/reader. Reconstruction instead keeps coordinate-derived
    stereo on the returned Mol, even when a toolkit version cannot serialize it.
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
        restored = _read_smiles(smiles, prepare_for_writing=False)
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
