"""
Utilities and Data Structures
"""

from __future__ import annotations

import collections.abc
import typing

__all__: list[str] = [
    "AtomData",
    "BondData",
    "BondKind",
    "MoleculeData",
    "StereoData",
    "StereoKind",
    "StereoGroupData",
    "StereoGroupKind",
    "apply_molecule_data_stereo_to_obmol",
    "calculate_shape_quality",
    "calculate_tetrahedron_volume",
    "extract_molecule_data",
    "get_possible_metal_radicals",
    "molecule_data_to_obmol_ptr",
    "xyz_seed_molecule_data",
]

def xyz_seed_molecule_data(xyz_block: str) -> MoleculeData:
    """Read XYZ with the same initial bond perception as the C++ pipeline."""

class AtomData:
    """Element, isotope, Cartesian coordinates and explicit electron labels for one graph atom."""

    isotope: int
    """Isotope mass number, or zero for the default isotope."""
    def __init__(self) -> None: ...
    def __repr__(self) -> str:
        """
        Return a concise debug representation of AtomData.
        """
    @property
    def atomic_num(self) -> int:
        """Atomic number of the element."""
    @atomic_num.setter
    def atomic_num(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def formal_charge(self) -> int:
        """Signed formal charge in elementary-charge units."""
    @formal_charge.setter
    def formal_charge(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def hybridization(self) -> int:
        """Hybridization code carried by the Open Babel atom."""
    @hybridization.setter
    def hybridization(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def radical_num(self) -> int:
        """Explicit count of unpaired electrons on this atom."""
    @radical_num.setter
    def radical_num(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def lone_pair_count(self) -> int:
        """Number of reconstruction-active lone pairs tracked on this atom."""
    @lone_pair_count.setter
    def lone_pair_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def unresolved_two_electron_center(self) -> bool:
        """Whether two electrons still require a singlet or triplet assignment."""
    @unresolved_two_electron_center.setter
    def unresolved_two_electron_center(self, arg0: bool) -> None: ...
    @property
    def electron_property_presence(self) -> int:
        """Known electron fields: unpaired=1, active lone pairs=2, unresolved=4."""
    @electron_property_presence.setter
    def electron_property_presence(
        self, arg0: typing.SupportsInt | typing.SupportsIndex
    ) -> None: ...
    @property
    def x(self) -> float:
        """Cartesian x coordinate in angstroms."""
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def y(self) -> float:
        """Cartesian y coordinate in angstroms."""
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def z(self) -> float:
        """Cartesian z coordinate in angstroms."""
    @z.setter
    def z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...

class BondKind:
    """Distinguishes covalent order, directed dative donation and unspecified coordination."""

    Covalent: typing.ClassVar[BondKind]
    Dative: typing.ClassVar[BondKind]
    CoordinationUnspecified: typing.ClassVar[BondKind]

class BondData:
    """An edge with one-based atom indices. A dative edge runs from donor to acceptor."""

    aromatic: bool
    """Whether the source graph marked this bond aromatic."""
    kind: BondKind
    """Covalent, dative or unspecified coordination edge."""
    def __init__(self) -> None: ...
    def __repr__(self) -> str:
        """
        Return a concise debug representation of BondData.
        """
    @property
    def begin_atom_idx(self) -> int:
        """One-based start atom index, the donor for a dative edge."""
    @begin_atom_idx.setter
    def begin_atom_idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def end_atom_idx(self) -> int:
        """One-based end atom index, the acceptor for a dative edge."""
    @end_atom_idx.setter
    def end_atom_idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def order(self) -> int:
        """Integer covalent bond order. Coordination uses the kind field."""
    @order.setter
    def order(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...

class StereoKind:
    """Geometry that determines how a stereo frame interprets its ordered neighbor references."""

    Tetrahedral: typing.ClassVar[StereoKind]
    DoubleBond: typing.ClassVar[StereoKind]
    SquarePlanar: typing.ClassVar[StereoKind]
    TrigonalBipyramidal: typing.ClassVar[StereoKind]
    Octahedral: typing.ClassVar[StereoKind]

class StereoData:
    """One atom or double-bond stereo frame. References are one-based and zero denotes a virtual non-hydrogen ligand."""

    kind: StereoKind
    """Stereo geometry of this atom or double bond."""
    center: int
    """One-based atom center or first endpoint of a double bond."""
    end: int
    """One-based second endpoint for double-bond stereo."""
    refs: list[int]
    """Ordered one-based ligand references. Zero is a virtual non-hydrogen ligand."""
    configuration: int
    """Zero denotes unspecified stereo. Tetrahedral uses 1=CW or 2=CCW, double bonds use 1=cis or 2=trans, and SP/TB/OH use SMILES permutation numbers."""
    def __init__(self) -> None: ...

class StereoGroupKind:
    """Absolute, relative (AND) or alternative (OR) stereochemistry of a group of centers."""

    Absolute: typing.ClassVar[StereoGroupKind]
    And: typing.ClassVar[StereoGroupKind]
    Or: typing.ClassVar[StereoGroupKind]

class StereoGroupData:
    """Enhanced stereo group with one-based atom members and the group identifier read from the input."""

    kind: StereoGroupKind
    """Absolute, AND or OR relation among member centers."""
    atoms: list[int]
    """One-based indices of the atom stereocenters in this group."""
    read_id: int
    """Group identifier from the input stereo annotation."""
    def __init__(self) -> None: ...

class MoleculeData:
    """Molecular graph exchanged by the native and Python backends, with explicit atoms, bonds, electron totals and stereo frames."""

    stereo: list[StereoData]
    """Sparse, ordered stereo frames for specified or explicitly unknown centers and bonds."""
    stereo_groups: list[StereoGroupData]
    """Enhanced stereo relations between atom centers."""
    def __init__(self) -> None: ...
    @property
    def atoms(self) -> list[AtomData]:
        """Atom records in input order. Bond and stereo references count from one in this array."""
    @atoms.setter
    def atoms(self, arg0: collections.abc.Sequence[AtomData]) -> None: ...
    @property
    def bonds(self) -> list[BondData]:
        """Edge records including covalent and coordination bonds."""
    @bonds.setter
    def bonds(self, arg0: collections.abc.Sequence[BondData]) -> None: ...
    @property
    def total_charge(self) -> int:
        """Net molecular charge in elementary-charge units."""
    @total_charge.setter
    def total_charge(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def total_radical_num(self) -> int:
        """Total unpaired-electron count recorded for the molecule."""
    @total_radical_num.setter
    def total_radical_num(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...

def _molecule_data_from_rows(
    atoms: collections.abc.Sequence[tuple[int, int, int, int, int, bool, int]],
    bonds: collections.abc.Sequence[tuple[int, int, int, bool, int]],
    coordinates: typing.Any = None,
) -> MoleculeData:
    """Batch-build atom/bond vectors; optional coordinates have shape (N, 3)."""

def calculate_shape_quality(
    p1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    p2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    p3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    p4: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
) -> float:
    """
    Calculate the shape quality score of a tetrahedron.

    Parameters:
        p1 (list[float]): Coordinates of the first atom.
        p2 (list[float]): Coordinates of the second atom.
        p3 (list[float]): Coordinates of the third atom.
        p4 (list[float]): Coordinates of the fourth atom.

    Returns:
        float: Quality score between 0.0 (coplanar/bad) and 1.0 (ideal).
    """

def calculate_tetrahedron_volume(
    p1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    p2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    p3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    p4: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
) -> float:
    """
    Calculate the volume of a tetrahedron defined by 4 points.

    Parameters:
        p1 (list[float]): Coordinates of the first atom.
        p2 (list[float]): Coordinates of the second atom.
        p3 (list[float]): Coordinates of the third atom.
        p4 (list[float]): Coordinates of the fourth atom.

    Returns:
        float: The volume.
    """

def extract_molecule_data(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> MoleculeData:
    """
    Extracts OBMol content into a structured object.

    Warning:
        The pointer is not thread-safe and must remain valid for the duration
        of this call. Prefer the regular pipeline APIs when possible.
    """

def get_possible_metal_radicals(
    metal: str, valence: typing.SupportsInt | typing.SupportsIndex
) -> set[int]:
    """
    Get possible radical electron counts for a metal given its valence.

    Args:
        metal (str): The chemical symbol (e.g., "Fe").
        valence (int): The oxidation state.

    Returns:
        set[int]: A set of possible unpaired electron counts.
    """

def molecule_data_to_obmol_ptr(molecule_data: MoleculeData) -> int:
    """
    Converts MoleculeData to a newly allocated covalent OBMol projection.
    Coordination edges are omitted; retain the source for coordination checks.
    Hydrogens must be explicit atom nodes.

    Warning:
        The returned pointer has manual ownership, is not safe to share across
        threads, and must be freed exactly once with _core.free_obmol_ptr.
    """

def apply_molecule_data_stereo_to_obmol(molecule_data: MoleculeData, mol_ptr: int) -> None:
    """Replace stereo on a covalent projection; the caller owns the OBMol pointer."""
