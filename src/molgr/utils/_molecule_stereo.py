"""Reference-based stereo adapters for the compact MoleculeData graph.

No SMILES roundtrip or coordinate perception is used. Non-tetrahedral
permutations use the standard SMILES SP/TB/OH numbering, not CIP labels.
"""

import heapq

from rdkit import Chem

from molgr import _core as core
from molgr.fallback.utils.consts import NON_METAL_DICT


def extract_rdkit_stereo(mol, *, atoms=None, bonds=None, ignore_metal_coordination=False):
    # Reuse adapter traversal; RDKit's Python GetAtoms/GetBonds iterators check
    # the container size repeatedly for every yielded item.
    if atoms is None:
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
    if bonds is None:
        bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())]
    metals = (
        [atom.GetAtomicNum() not in NON_METAL_DICT for atom in atoms]
        if ignore_metal_coordination
        else None
    )

    def ignored(atom):
        # A frame that loses a metal neighbor is not a frame on the covalent
        # projection. Exclude it before enforcing the frame's reference count.
        return metals is not None and (
            metals[atom.GetIdx()] or any(metals[a.GetIdx()] for a in atom.GetNeighbors())
        )

    records = []
    kind = core.utils.StereoKind
    atom_kinds = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: (kind.Tetrahedral, 1, 4),
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: (kind.Tetrahedral, 2, 4),
        Chem.ChiralType.CHI_SQUAREPLANAR: (kind.SquarePlanar, None, 4),
        Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL: (kind.TrigonalBipyramidal, None, 5),
        Chem.ChiralType.CHI_OCTAHEDRAL: (kind.Octahedral, None, 6),
    }
    unknown_centers = {b.GetBeginAtomIdx() for b in bonds if b.GetBondDir() == Chem.BondDir.UNKNOWN}
    for atom in atoms:
        tag = atom.GetChiralTag()
        unknown = atom.GetIdx() in unknown_centers or (
            atom.HasProp("_UnknownStereo") and atom.GetBoolProp("_UnknownStereo")
        )
        if tag == Chem.ChiralType.CHI_UNSPECIFIED and not unknown:
            continue
        if ignored(atom):
            continue
        if unknown:
            stereo_kind, configuration, size = kind.Tetrahedral, 0, 4
        elif tag in atom_kinds:
            stereo_kind, configuration, size = atom_kinds[tag]
            if configuration is None:
                configuration = (
                    atom.GetIntProp("_chiralPermutation")
                    if atom.HasProp("_chiralPermutation")
                    else 0
                )
        else:
            raise ValueError(f"Unsupported atom stereochemistry: {tag}")
        refs = [a.GetIdx() + 1 for a in atom.GetNeighbors()]
        if len(refs) > size or (stereo_kind == kind.Tetrahedral and len(refs) < 3):
            raise ValueError("Invalid stereo reference count")
        record = core.utils.StereoData()
        record.kind, record.center = stereo_kind, atom.GetIdx() + 1
        record.refs = refs + [0] * (size - len(refs))
        record.configuration = configuration
        records.append(record)

    cis = (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS)
    trans = (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS)
    for bond in bonds:
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREONONE:
            continue
        if ignored(bond.GetBeginAtom()) or ignored(bond.GetEndAtom()):
            continue
        if stereo not in (*cis, *trans, Chem.BondStereo.STEREOANY):
            raise ValueError(f"Unsupported bond stereochemistry: {stereo}")
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError("Cis/trans stereo requires a double bond")
        record = core.utils.StereoData()
        record.kind = kind.DoubleBond
        record.center, record.end = bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1
        selected = list(bond.GetStereoAtoms())
        refs = []
        for side, other, offset in (
            (bond.GetBeginAtom(), bond.GetEndAtomIdx(), 0),
            (bond.GetEndAtom(), bond.GetBeginAtomIdx(), 1),
        ):
            neighbors = [a.GetIdx() for a in side.GetNeighbors() if a.GetIdx() != other]
            first = selected[offset] if len(selected) == 2 else neighbors[0] if neighbors else None
            if first not in neighbors or len(neighbors) > 2:
                raise ValueError("Invalid double-bond stereo references")
            remaining = [idx + 1 for idx in neighbors if idx != first]
            refs.extend([first + 1, remaining[0] if remaining else 0])
        record.refs = refs
        record.configuration = 1 if stereo in cis else 2 if stereo in trans else 0
        records.append(record)

    group_kinds = {
        Chem.StereoGroupType.STEREO_ABSOLUTE: core.utils.StereoGroupKind.Absolute,
        Chem.StereoGroupType.STEREO_AND: core.utils.StereoGroupKind.And,
        Chem.StereoGroupType.STEREO_OR: core.utils.StereoGroupKind.Or,
    }
    groups = []
    for source in mol.GetStereoGroups():
        if hasattr(source, "GetBonds") and source.GetBonds():
            raise ValueError("Bond members in enhanced stereo groups are not yet supported")
        group = core.utils.StereoGroupData()
        group.kind = group_kinds[source.GetGroupType()]
        group.atoms = [atom.GetIdx() + 1 for atom in source.GetAtoms() if not ignored(atom)]
        if not group.atoms:
            continue
        group.read_id = source.GetReadId()
        groups.append(group)
    return records, groups


def stereo_ordered_bonds(data):
    """Keep TB/OH neighbor order without rewriting their permutation numbers.

    Only their incident edges impose ordering constraints. Tetrahedral and SP
    records are transformed algebraically below. Data extracted from a graph
    has consistent constraints even if the bond array is subsequently shuffled.
    """
    bonds = data.bonds
    records = [
        s
        for s in data.stereo
        if s.kind in (core.utils.StereoKind.TrigonalBipyramidal, core.utils.StereoKind.Octahedral)
    ]
    if not records:
        return bonds
    indices = {frozenset((b.begin_atom_idx, b.end_atom_idx)): i for i, b in enumerate(bonds)}
    successors = [set() for _ in bonds]
    indegree = [0] * len(bonds)
    for record in records:
        try:
            sequence = [indices[frozenset((record.center, ref))] for ref in record.refs if ref]
        except KeyError as exc:
            raise ValueError("Stereo reference edge is missing") from exc
        for first, second in zip(sequence, sequence[1:]):
            if second not in successors[first]:
                successors[first].add(second)
                indegree[second] += 1
    queue = [i for i, degree in enumerate(indegree) if not degree]
    heapq.heapify(queue)
    ordered = []
    while queue:
        index = heapq.heappop(queue)
        ordered.append(bonds[index])
        for successor in successors[index]:
            indegree[successor] -= 1
            if not indegree[successor]:
                heapq.heappush(queue, successor)
    if len(ordered) != len(bonds):
        raise ValueError("Inconsistent non-tetrahedral stereo reference ordering")
    return ordered


def _odd_permutation(before, after):
    permutation = [before.index(ref) for ref in after]
    return sum(a > b for i, a in enumerate(permutation) for b in permutation[i + 1 :]) % 2


def _square_pairs(refs, configuration):
    order = {1: (0, 1, 2, 3), 2: (0, 2, 1, 3), 3: (0, 1, 3, 2)}[configuration]
    return {
        frozenset((refs[order[0]], refs[order[2]])),
        frozenset((refs[order[1]], refs[order[3]])),
    }


def apply_rdkit_stereo(mol, data):
    kind = core.utils.StereoKind
    for record in data.stereo:
        refs = list(record.refs)
        configuration = record.configuration
        if record.kind == kind.DoubleBond:
            bond = mol.GetBondBetweenAtoms(record.center - 1, record.end - 1)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE or len(refs) != 4:
                raise ValueError("Double-bond stereo no longer matches the graph")
            if not configuration:
                bond.SetStereo(Chem.BondStereo.STEREOANY)
                continue
            selected = []
            for offset in (0, 2):
                if not refs[offset]:
                    configuration = 3 - configuration
                    selected.append(refs[offset + 1] - 1)
                else:
                    selected.append(refs[offset] - 1)
            if bond.GetBeginAtomIdx() != record.center - 1:
                selected.reverse()
            bond.SetStereoAtoms(*selected)
            bond.SetStereo(
                Chem.BondStereo.STEREOCIS if configuration == 1 else Chem.BondStereo.STEREOTRANS
            )
            continue

        atom = mol.GetAtomWithIdx(record.center - 1)
        current = [a.GetIdx() + 1 for a in atom.GetNeighbors()]
        current += [0] * (len(refs) - len(current))
        if sorted(current) != sorted(refs):
            raise ValueError("Atom stereo references no longer match the graph")
        if record.kind == kind.Tetrahedral:
            if not configuration:
                atom.SetBoolProp("_UnknownStereo", True)
                continue
            if _odd_permutation(refs, current):
                configuration = 3 - configuration
            atom.SetChiralTag(
                Chem.ChiralType.CHI_TETRAHEDRAL_CW
                if configuration == 1
                else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            )
        else:
            tags = {
                kind.SquarePlanar: Chem.ChiralType.CHI_SQUAREPLANAR,
                kind.TrigonalBipyramidal: Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
                kind.Octahedral: Chem.ChiralType.CHI_OCTAHEDRAL,
            }
            if record.kind == kind.SquarePlanar and configuration:
                pairs = _square_pairs(refs, configuration)
                configuration = next(i for i in (1, 2, 3) if _square_pairs(current, i) == pairs)
            elif record.kind != kind.SquarePlanar and current != refs:
                raise ValueError("Non-tetrahedral stereo reference ordering changed")
            atom.SetChiralTag(tags[record.kind])
            atom.SetIntProp("_chiralPermutation", configuration)
    groups = []
    group_kinds = {
        core.utils.StereoGroupKind.Absolute: Chem.StereoGroupType.STEREO_ABSOLUTE,
        core.utils.StereoGroupKind.And: Chem.StereoGroupType.STEREO_AND,
        core.utils.StereoGroupKind.Or: Chem.StereoGroupType.STEREO_OR,
    }
    for group in data.stereo_groups:
        groups.append(
            Chem.CreateStereoGroup(
                group_kinds[group.kind], mol, [i - 1 for i in group.atoms], readId=group.read_id
            )
        )
    mol.SetStereoGroups(groups)
    Chem.SetDoubleBondNeighborDirections(mol)
