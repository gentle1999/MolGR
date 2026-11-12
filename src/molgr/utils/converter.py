from typing import List, cast

from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from molgr import _core as core
from molgr.fallback.utils.consts import NON_METAL_DICT
from molgr.fallback.utils.electrons import (
    LONE_PAIR_COUNT_PROP,
    UNPAIRED_ELECTRON_COUNT_PROP,
    UNRESOLVED_TWO_ELECTRON_CENTER_PROP,
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.utils._molecule_stereo import (
    apply_rdkit_stereo,
    extract_rdkit_stereo,
    stereo_ordered_bonds,
)


OB_RDKIT_BOND_ORDER_MAPPING = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.QUADRUPLE,
    5: Chem.BondType.AROMATIC,
}

METAL_UNPAIRED_ELECTRONS_PROP = "MOLGR_METAL_UNPAIRED_ELECTRONS"


def _explicit_bond_copy(rdmol: Chem.Mol) -> Chem.Mol:
    """Discard cached aromaticity while retaining authoritative integer bonds.

    Open Babel's aromatic flags can outlive charge and bond edits. Asking
    RDKit to Kekulize those flags can change double bonds *before* raising.
    Only actual AROMATIC bond types require a Kekule assignment; perform that
    operation on a disposable copy and propagate failures instead of returning
    a partially modified graph.
    """
    explicit = Chem.Mol(rdmol)
    if any(b.GetBondType() == Chem.BondType.AROMATIC for b in explicit.GetBonds()):
        Chem.Kekulize(explicit, clearAromaticFlags=True)
    for atom in explicit.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in explicit.GetBonds():
        bond.SetIsAromatic(False)
    explicit.ClearComputedProps()
    explicit.UpdatePropertyCache(strict=False)
    return explicit


def _sanitize_preserving_explicit_bonds(rdmol: Chem.Mol, *, kekulize: bool) -> Chem.Mol:
    """Return a sanitized copy; never reuse a failed Kekulization's mutations."""
    sanitized = _explicit_bond_copy(rdmol)
    Chem.SanitizeMol(sanitized)
    if kekulize:
        Chem.Kekulize(sanitized)
    return sanitized


def _perceive_rdkit_aromaticity(rdmol: Chem.Mol) -> None:
    Chem.SanitizeMol(
        rdmol,
        sanitizeOps=(
            Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            | Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        ),
    )


def refresh_rdkit_aromaticity(rdmol: Chem.Mol) -> None:
    """Refresh serialization-facing aromaticity, preserving MolGR electron states.

    Reconstruction supplies integer bond orders, so its normal path needs no
    molecule copy or Kekulization. Save only the fields perception can change;
    restore radicals in all cases and roll back graph fields on failure.
    Actual AROMATIC bond inputs still require a disposable Kekulization copy.
    """
    bonds = [rdmol.GetBondWithIdx(i) for i in range(rdmol.GetNumBonds())]
    if any(bond.GetBondType() == Chem.BondType.AROMATIC for bond in bonds):
        perceived = _explicit_bond_copy(rdmol)
        _perceive_rdkit_aromaticity(perceived)
        for atom, normalized in zip(rdmol.GetAtoms(), perceived.GetAtoms()):
            atom.SetIsAromatic(normalized.GetIsAromatic())
        for bond, normalized in zip(bonds, perceived.GetBonds()):
            bond.SetBondType(normalized.GetBondType())
            bond.SetIsAromatic(normalized.GetIsAromatic())
            bond.SetIsConjugated(normalized.GetIsConjugated())
        rdmol.UpdatePropertyCache(strict=False)
        return

    atoms = [rdmol.GetAtomWithIdx(i) for i in range(rdmol.GetNumAtoms())]
    atom_state = [(atom.GetNumRadicalElectrons(), atom.GetIsAromatic()) for atom in atoms]
    bond_state = [
        (bond.GetBondType(), bond.GetIsAromatic(), bond.GetIsConjugated()) for bond in bonds
    ]
    try:
        for atom in atoms:
            atom.SetIsAromatic(False)
        for bond in bonds:
            bond.SetIsAromatic(False)
        rdmol.ClearComputedProps()
        rdmol.UpdatePropertyCache(strict=False)
        _perceive_rdkit_aromaticity(rdmol)
    except Exception:
        for atom, (_, aromatic) in zip(atoms, atom_state):
            atom.SetIsAromatic(aromatic)
        for bond, (kind, aromatic, conjugated) in zip(bonds, bond_state):
            bond.SetBondType(kind)
            bond.SetIsAromatic(aromatic)
            bond.SetIsConjugated(conjugated)
        # Perception may have left intermediate ring/computed caches. Discard
        # those derived values after restoring the authoritative graph fields.
        rdmol.ClearComputedProps()
        raise
    finally:
        for atom, (radicals, _) in zip(atoms, atom_state):
            atom.SetNumRadicalElectrons(radicals)
        rdmol.UpdatePropertyCache(strict=False)


def _is_metal_atomic_num(atomic_num: int) -> bool:
    return atomic_num not in NON_METAL_DICT


def _set_rdkit_unpaired_electrons(
    atom: Chem.Atom, unpaired_electrons: int, *, known: bool = True
) -> None:
    """Map unpaired electrons to RDKit and annotate only explicitly known counts."""

    if _is_metal_atomic_num(int(atom.GetAtomicNum())):
        if known:
            atom.SetIntProp(METAL_UNPAIRED_ELECTRONS_PROP, int(unpaired_electrons))
        atom.SetNumRadicalElectrons(0)
        return
    atom.SetNumRadicalElectrons(int(unpaired_electrons))
    if known:
        atom.SetIntProp(UNPAIRED_ELECTRON_COUNT_PROP, int(unpaired_electrons))


def _set_rdkit_lone_pair_count(atom: Chem.Atom, lone_pair_count: int) -> None:
    """Persist a known lone-pair count, including zero, without changing spin."""

    atom.SetIntProp(LONE_PAIR_COUNT_PROP, int(lone_pair_count))


def _set_rdkit_unresolved_two_electron_center(atom: Chem.Atom, value: bool) -> None:
    """Persist deferred two-electron occupancy without selecting singlet/triplet."""

    atom.SetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP, bool(value))


def refresh_atom_electron_property_lists(rdmol: Chem.Mol) -> None:
    """Rebuild MolGR's SDF property lists in place from current atom annotations.

    Call after editing electron properties or atom order. Remove old lists even
    when no atom retains the property, so SDF readers cannot resurrect a consumed
    electron annotation. Atom properties, formal charges and radicals are unchanged.
    """

    fields = (
        (METAL_UNPAIRED_ELECTRONS_PROP, "iprop", Chem.CreateAtomIntPropertyList),
        (UNPAIRED_ELECTRON_COUNT_PROP, "iprop", Chem.CreateAtomIntPropertyList),
        (LONE_PAIR_COUNT_PROP, "iprop", Chem.CreateAtomIntPropertyList),
        (UNRESOLVED_TWO_ELECTRON_CENTER_PROP, "bprop", Chem.CreateAtomBoolPropertyList),
    )
    for name, kind, create_list in fields:
        list_name = f"atom.{kind}.{name}"
        if rdmol.HasProp(list_name):
            rdmol.ClearProp(list_name)
        if any(
            atom.HasProp(name)
            for atom in rdmol.GetAtoms()  # pyright: ignore[reportCallIssue]
        ):
            create_list(rdmol, name)


def _finalize_metal_unpaired_electrons(rdmol: Chem.Mol) -> None:
    for atom_idx in range(rdmol.GetNumAtoms()):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        if _is_metal_atomic_num(int(atom.GetAtomicNum())):
            atom.SetNumRadicalElectrons(0)
    refresh_atom_electron_property_lists(rdmol)
    rdmol.UpdatePropertyCache(strict=False)


def _restore_rdkit_unpaired_electrons(
    rdmol: Chem.Mol,
    unpaired_electrons: List[int],
) -> None:
    """Restore MolGR electron assignments overwritten by RDKit sanitization."""

    for atom, count in zip(
        rdmol.GetAtoms(),  # pyright: ignore[reportCallIssue]
        unpaired_electrons,
    ):
        prop = (
            METAL_UNPAIRED_ELECTRONS_PROP
            if _is_metal_atomic_num(int(atom.GetAtomicNum()))
            else UNPAIRED_ELECTRON_COUNT_PROP
        )
        _set_rdkit_unpaired_electrons(atom, count, known=bool(atom.HasProp(prop)))


def get_atom_unpaired_electrons(atom: Chem.Atom) -> int:
    """Return MolGR's unpaired-electron count for an RDKit atom."""

    if _is_metal_atomic_num(int(atom.GetAtomicNum())) and atom.HasProp(
        METAL_UNPAIRED_ELECTRONS_PROP
    ):
        try:
            return int(atom.GetIntProp(METAL_UNPAIRED_ELECTRONS_PROP))
        except (RuntimeError, TypeError, ValueError):
            return int(atom.GetProp(METAL_UNPAIRED_ELECTRONS_PROP))
    return int(atom.GetNumRadicalElectrons())


def get_atom_lone_pair_count(atom: Chem.Atom) -> int:
    """Return MolGR's explicit lone-pair count for an RDKit atom."""

    if not atom.HasProp(LONE_PAIR_COUNT_PROP):
        return 0
    try:
        return int(atom.GetIntProp(LONE_PAIR_COUNT_PROP))
    except (RuntimeError, TypeError, ValueError):
        return int(atom.GetProp(LONE_PAIR_COUNT_PROP))


def has_atom_unresolved_two_electron_center(atom: Chem.Atom) -> bool:
    """Return whether an RDKit atom carries MolGR's unresolved-center marker."""

    if not atom.HasProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP):
        return False
    try:
        return bool(atom.GetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP))
    except (RuntimeError, TypeError, ValueError):
        return atom.GetProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP).lower() in {
            "1",
            "true",
        }


def rdkit_to_mol_data(
    rdmol: Chem.Mol,
    *,
    perceive_radicals: bool = False,
    include_stereo: bool = True,
    include_coordinates: bool = True,
) -> core.utils.MoleculeData:
    """Build explicit-H MoleculeData without modifying the RDKit source.

    Primitive atom/bond rows cross the binding once. Coordination direction,
    isotope and annotation presence retain their existing semantics. Stereo
    and coordinates are preserved by default; comparison callers can omit
    unused fields. Radical perception follows Kekulization and explicit MolGR
    assignments override perceived counts, including known zero.
    """
    for i in range(rdmol.GetNumAtoms()):
        atom = rdmol.GetAtomWithIdx(i)
        if atom.HasQuery() or atom.GetAtomicNum() == 0:
            raise ValueError("MoleculeData does not support query or dummy atoms")
    return _convert_rdkit_graph(
        rdmol,
        perceive_radicals=perceive_radicals,
        include_stereo=include_stereo,
        include_coordinates=include_coordinates,
    )


def _convert_rdkit_graph(
    rdmol,
    *,
    perceive_radicals,
    include_stereo,
    include_coordinates,
    atom_rows=None,
    ignore_metal_stereo=False,
):
    """Convert an atom-validated input, optionally consuming decoded atom rows.

    Only the evaluator supplies rows, from its source diagnostics scan. They
    follow the native seven-field layout; a None radical means to read RDKit's
    post-Kekulization value. AddHs preserves original atom indices and appends
    H nodes. Element, isotope, charge and MolGR metadata survive these chemical
    operations; bond orders and unannotated radicals must be read afterwards.
    """
    covalent_orders = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.QUADRUPLE: 4,
    }
    for i in range(rdmol.GetNumBonds()):
        bond = rdmol.GetBondWithIdx(i)
        if bond.HasQuery():
            raise ValueError("MoleculeData does not support query bonds")
        bond_type = bond.GetBondType()
        if bond_type not in covalent_orders and bond_type not in (
            Chem.BondType.AROMATIC,
            Chem.BondType.DATIVE,
        ):
            raise ValueError(f"Unsupported MoleculeData bond type: {bond_type}")

    keep_coordinates = include_coordinates and bool(rdmol.GetNumConformers())
    explicit = Chem.AddHs(rdmol, addCoords=keep_coordinates)
    if include_stereo:
        Chem.AssignChiralTypesFromBondDirs(explicit, replaceExistingTags=False)
        Chem.SetBondStereoFromDirections(explicit)
    else:
        # Stereo tags can affect RDKit's choice of Kekule assignment. Clear
        # them on the AddHs copy so disabling stereo preserves the previous
        # integer graph and bounded-search behavior without another Mol copy.
        Chem.RemoveStereochemistry(explicit)
    Chem.Kekulize(explicit, clearAromaticFlags=False)
    # FINDRADICALS on aromatic 1.5-order bonds can invent P/S radicals, so
    # perception must follow integer Kekule assignment.
    if perceive_radicals:
        Chem.SanitizeMol(explicit, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
    atoms = [explicit.GetAtomWithIdx(i) for i in range(explicit.GetNumAtoms())]
    bonds = [explicit.GetBondWithIdx(i) for i in range(explicit.GetNumBonds())]
    cached_rows = atom_rows
    atom_rows = []
    for i, atom in enumerate(atoms):
        if atom.GetNumImplicitHs() or atom.GetNumExplicitHs():
            raise ValueError("MoleculeData requires all hydrogens to be atom nodes")
        if cached_rows is not None:
            if i < len(cached_rows):
                row = cached_rows[i]
                if row[3] is None:
                    row = (*row[:3], atom.GetNumRadicalElectrons(), *row[4:])
            else:
                # AddHs may restore isotopic H retained by RemoveHs. Read its
                # actual isotope instead of assuming every new node is 1H.
                row = (
                    atom.GetAtomicNum(),
                    atom.GetIsotope(),
                    atom.GetFormalCharge(),
                    atom.GetNumRadicalElectrons(),
                    0,
                    False,
                    0,
                )
            atom_rows.append(row)
            continue
        atomic_num = atom.GetAtomicNum()
        metal = atomic_num not in NON_METAL_DICT
        unpaired_prop = METAL_UNPAIRED_ELECTRONS_PROP if metal else UNPAIRED_ELECTRON_COUNT_PROP
        known_unpaired = bool(atom.HasProp(unpaired_prop))
        known_pair = bool(atom.HasProp(LONE_PAIR_COUNT_PROP))
        known_unresolved = bool(atom.HasProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP))
        radical = (
            int(atom.GetProp(unpaired_prop))
            if known_unpaired
            else (0 if metal else atom.GetNumRadicalElectrons())
        )
        atom_rows.append(
            (
                atomic_num,
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                radical,
                get_atom_lone_pair_count(atom) if known_pair else 0,
                has_atom_unresolved_two_electron_center(atom) if known_unresolved else False,
                int(known_unpaired) | (2 if known_pair else 0) | (4 if known_unresolved else 0),
            )
        )

    bond_rows = []
    for bond in bonds:
        bond_type = bond.GetBondType()
        dative = bond_type == Chem.BondType.DATIVE
        bond_rows.append(
            (
                bond.GetBeginAtomIdx() + 1,
                bond.GetEndAtomIdx() + 1,
                0 if dative else covalent_orders[bond_type],
                False if dative else bond.GetIsAromatic(),
                1 if dative else 0,
            )
        )
    coordinates = explicit.GetConformer().GetPositions() if keep_coordinates else None
    data = core.utils._molecule_data_from_rows(atom_rows, bond_rows, coordinates)
    if include_stereo:
        data.stereo, data.stereo_groups = extract_rdkit_stereo(
            explicit,
            atoms=atoms,
            bonds=bonds,
            ignore_metal_coordination=ignore_metal_stereo,
        )
    return data


def mol_data_to_pybel(mol_data: core.utils.MoleculeData) -> pybel.Molecule:
    """Copy all three electron classifications from MoleculeData to Open Babel.

    ``radical_num`` remains real unpaired electrons; lone-pair and unresolved
    fields are independent. Presence bits retain known zero/false versus unknown.
    Coordination edges are omitted; retain the source for coordination checks.
    Hydrogens must already be explicit atom nodes.
    """
    obmol = ob.OBMol()
    obmol.BeginModify()
    for atom in mol_data.atoms:
        obatom: ob.OBAtom = obmol.NewAtom()
        obatom.SetAtomicNum(atom.atomic_num)
        obatom.SetIsotope(atom.isotope)
        obatom.SetImplicitHCount(0)
        obatom.SetFormalCharge(atom.formal_charge)
        presence = atom.electron_property_presence
        if presence & 1:
            set_unpaired_electron_count(obatom, atom.radical_num)
        if presence & 2:
            set_lone_pair_count(obatom, atom.lone_pair_count)
        if presence & 4:
            set_unresolved_two_electron_center(obatom, atom.unresolved_two_electron_center)
        obatom.SetVector(atom.x, atom.y, atom.z)
    for bond in mol_data.bonds:
        if bond.kind != core.utils.BondKind.Covalent:
            continue
        obmol.AddBond(bond.begin_atom_idx, bond.end_atom_idx, bond.order)
        if bond.aromatic or bond.order == 5:
            obmol.GetBond(bond.begin_atom_idx, bond.end_atom_idx).SetAromatic()
    obmol.EndModify()
    if mol_data.stereo:
        core.utils.apply_molecule_data_stereo_to_obmol(mol_data, int(obmol.this))
    return pybel.Molecule(obmol)


def mol_data_to_rdkit(
    mol_data: core.utils.MoleculeData,
    sanitize: bool = True,
    kekulize: bool = True,
) -> Chem.Mol:
    """Convert MoleculeData while preserving independent electron classifications.

    Nonmetal unpaired electrons use RDKit radicals plus explicit-count provenance;
    metals use ``MOLGR_METAL_UNPAIRED_ELECTRONS``. The presence mask distinguishes
    known zero/false from unknown for all three independent classifications.
    Sanitization is not allowed to redefine stored MolGR assignments.
    """
    unpaired_electrons = [int(atom.radical_num) for atom in mol_data.atoms]
    rwmol = Chem.RWMol()
    for atom in mol_data.atoms:
        atom_id = rwmol.AddAtom(Chem.Atom(atom.atomic_num))
        rd_atom = rwmol.GetAtomWithIdx(atom_id)
        rd_atom.SetNoImplicit(True)
        rd_atom.SetIsotope(atom.isotope)
        rd_atom.SetFormalCharge(atom.formal_charge)
        presence = atom.electron_property_presence
        _set_rdkit_unpaired_electrons(rd_atom, atom.radical_num, known=bool(presence & 1))
        if presence & 2:
            _set_rdkit_lone_pair_count(rd_atom, atom.lone_pair_count)
        if presence & 4:
            _set_rdkit_unresolved_two_electron_center(
                rd_atom,
                atom.unresolved_two_electron_center,
            )
    for bond_data in stereo_ordered_bonds(mol_data):
        if bond_data.kind == core.utils.BondKind.Dative:
            bond_type = Chem.BondType.DATIVE
        elif bond_data.kind == core.utils.BondKind.Covalent:
            bond_type = OB_RDKIT_BOND_ORDER_MAPPING.get(bond_data.order, Chem.BondType.ZERO)
        else:
            raise ValueError("RDKit conversion requires a direction for coordination bonds")
        rwmol.AddBond(
            bond_data.begin_atom_idx - 1,
            bond_data.end_atom_idx - 1,
            bond_type,
        )
        if bond_data.kind == core.utils.BondKind.Covalent and bond_data.aromatic:
            rd_bond = rwmol.GetBondBetweenAtoms(
                bond_data.begin_atom_idx - 1, bond_data.end_atom_idx - 1
            )
            rd_bond.SetIsAromatic(True)
            rd_bond.GetBeginAtom().SetIsAromatic(True)
            rd_bond.GetEndAtom().SetIsAromatic(True)
    if mol_data.stereo:
        apply_rdkit_stereo(rwmol, mol_data)
    rdmol = rwmol.GetMol()
    conf = Chem.Conformer(rdmol.GetNumAtoms())
    for atom_idx, atom in enumerate(mol_data.atoms):
        conf.SetAtomPosition(atom_idx, (atom.x, atom.y, atom.z))
    rdmol.RemoveAllConformers()
    rdmol.AddConformer(conf)

    if sanitize:
        rdmol = _sanitize_preserving_explicit_bonds(rdmol, kekulize=kekulize)
    elif kekulize:
        Chem.KekulizeIfPossible(rdmol)
    _restore_rdkit_unpaired_electrons(rdmol, unpaired_electrons)
    _finalize_metal_unpaired_electrons(rdmol)
    if mol_data.stereo:
        Chem.AssignStereochemistry(rdmol, cleanIt=False, force=True)
    return rdmol


def pybel_to_rdmol(
    omol: pybel.Molecule,
    sanitize: bool = True,
    kekulize: bool = True,
) -> Chem.Mol:
    """Convert Open Babel topology while preserving all MolGR electron fields.

    Real unpaired electrons, active lone pairs, and unresolved centers are read
    before RDKit sanitization and restored independently afterwards.
    """
    bonds = [
        (
            cast(ob.OBBond, bond).GetBeginAtomIdx() - 1,
            cast(ob.OBBond, bond).GetEndAtomIdx() - 1,
            cast(ob.OBBond, bond).GetBondOrder(),
        )
        for bond in ob.OBMolBondIter(omol.OBMol)
    ]
    formal_charges: List[int] = [
        cast(ob.OBAtom, atom).GetFormalCharge() for atom in ob.OBMolAtomIter(omol.OBMol)
    ]
    formal_radicals: List[int] = [
        get_unpaired_electron_count(cast(ob.OBAtom, atom)) for atom in ob.OBMolAtomIter(omol.OBMol)
    ]
    rwmol = Chem.RWMol(Chem.MolFromXYZBlock(omol.write("xyz")))
    for bond in bonds:
        rwmol.AddBond(
            bond[0], bond[1], OB_RDKIT_BOND_ORDER_MAPPING.get(bond[2], Chem.BondType.ZERO)
        )
    for atom_id, (charge, radical) in enumerate(zip(formal_charges, formal_radicals)):
        atom = rwmol.GetAtomWithIdx(atom_id)
        atom.SetIsotope(omol.OBMol.GetAtom(atom_id + 1).GetIsotope())
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(charge)
        _set_rdkit_unpaired_electrons(atom, radical)
    rdmol = rwmol.GetMol()
    rdmol.UpdatePropertyCache(strict=False)
    if sanitize:
        rdmol = _sanitize_preserving_explicit_bonds(rdmol, kekulize=kekulize)
    elif kekulize:
        Chem.KekulizeIfPossible(rdmol)
    _restore_rdkit_unpaired_electrons(rdmol, formal_radicals)
    for atom_id, source in enumerate(ob.OBMolAtomIter(omol.OBMol)):
        source = cast(ob.OBAtom, source)
        atom = rdmol.GetAtomWithIdx(atom_id)
        if source.HasData(LONE_PAIR_COUNT_PROP):
            _set_rdkit_lone_pair_count(atom, get_lone_pair_count(source))
        if source.HasData(UNRESOLVED_TWO_ELECTRON_CENTER_PROP):
            _set_rdkit_unresolved_two_electron_center(
                atom, has_unresolved_two_electron_center(source)
            )
        if not source.HasData(UNPAIRED_ELECTRON_COUNT_PROP):
            prop = (
                METAL_UNPAIRED_ELECTRONS_PROP
                if _is_metal_atomic_num(int(atom.GetAtomicNum()))
                else UNPAIRED_ELECTRON_COUNT_PROP
            )
            atom.ClearProp(prop)
    _finalize_metal_unpaired_electrons(rdmol)
    if omol.OBMol.HasData(ob.StereoData):
        data = core.utils.extract_molecule_data(int(omol.OBMol.this))
        rwmol = Chem.RWMol(rdmol)
        apply_rdkit_stereo(rwmol, data)
        rdmol = rwmol.GetMol()
        Chem.AssignStereochemistry(rdmol, cleanIt=False, force=True)
    return rdmol


__all__ = [
    "METAL_UNPAIRED_ELECTRONS_PROP",
    "get_atom_lone_pair_count",
    "get_atom_unpaired_electrons",
    "has_atom_unresolved_two_electron_center",
    "mol_data_to_pybel",
    "mol_data_to_rdkit",
    "pybel_to_rdmol",
    "rdkit_to_mol_data",
    "refresh_atom_electron_property_lists",
]
