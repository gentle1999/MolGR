"""Typed access to atoms, bonds, and SSSR paths in an Open Babel molecular graph."""

from typing import Callable, Iterable, Iterator, Sequence, cast

from openbabel import openbabel as ob


# SWIG returns borrowed atom and bond proxies. Use the original iterators so
# traversing a ligand neither copies the graph nor changes neighbor order.
mol_atoms = cast("Callable[[ob.OBMol], Iterator[ob.OBAtom]]", ob.OBMolAtomIter)
mol_bonds = cast("Callable[[ob.OBMol], Iterator[ob.OBBond]]", ob.OBMolBondIter)
atom_neighbors = cast("Callable[[ob.OBAtom], Iterator[ob.OBAtom]]", ob.OBAtomAtomIter)
atom_bonds = cast("Callable[[ob.OBAtom], Iterator[ob.OBBond]]", ob.OBAtomBondIter)


def indexed_atoms(mol: ob.OBMol) -> Iterator[ob.OBAtom]:
    """Visit one-based atom indices in OBMol order while membership stays fixed."""
    return map(mol.GetAtom, range(1, mol.NumAtoms() + 1))


def indexed_bonds(mol: ob.OBMol) -> Iterator[ob.OBBond]:
    """Visit zero-based bond indices in OBMol order while membership stays fixed."""
    return map(mol.GetBond, range(mol.NumBonds()))


def mol_rings(mol: ob.OBMol) -> Iterable[ob.OBRing]:
    """Return the perceived smallest set of smallest rings from the original molecule."""
    return cast("Iterable[ob.OBRing]", mol.GetSSSR())


def ring_paths(mol: ob.OBMol) -> Iterator[Sequence[int]]:
    """Yield each SSSR perimeter as consecutive, one-based OBMol atom indices."""
    for ring in mol_rings(mol):
        yield cast("Sequence[int]", ring._path)


def neighbor_bond_order(atom: ob.OBAtom, neighbor: ob.OBAtom) -> int:
    """Read the stored bond order of an edge already established by atom traversal."""
    bond = cast(ob.OBBond, atom.GetBond(neighbor))
    return cast(int, bond.GetBondOrder())


def indexed_bond_order(mol: ob.OBMol, begin: int, end: int) -> int:
    """Read the stored order of a known edge using one-based OBMol indices."""
    bond = cast(ob.OBBond, mol.GetBond(begin, end))
    return cast(int, bond.GetBondOrder())
