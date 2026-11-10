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
]

class AtomData:
    isotope: int
    def __init__(self) -> None: ...
    def __repr__(self) -> str:
        """
        Return a concise debug representation of AtomData.
        """
    @property
    def atomic_num(self) -> int: ...
    @atomic_num.setter
    def atomic_num(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def formal_charge(self) -> int: ...
    @formal_charge.setter
    def formal_charge(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def hybridization(self) -> int: ...
    @hybridization.setter
    def hybridization(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def radical_num(self) -> int: ...
    @radical_num.setter
    def radical_num(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def lone_pair_count(self) -> int: ...
    @lone_pair_count.setter
    def lone_pair_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def unresolved_two_electron_center(self) -> bool: ...
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
    def x(self) -> float: ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def z(self) -> float: ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...

class BondKind:
    """Dative endpoints are donor -> acceptor; covalent order is unused."""

    Covalent: typing.ClassVar[BondKind]
    Dative: typing.ClassVar[BondKind]
    CoordinationUnspecified: typing.ClassVar[BondKind]

class BondData:
    aromatic: bool
    kind: BondKind
    def __init__(self) -> None: ...
    def __repr__(self) -> str:
        """
        Return a concise debug representation of BondData.
        """
    @property
    def begin_atom_idx(self) -> int: ...
    @begin_atom_idx.setter
    def begin_atom_idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def end_atom_idx(self) -> int: ...
    @end_atom_idx.setter
    def end_atom_idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def order(self) -> int: ...
    @order.setter
    def order(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...

class StereoKind:
    Tetrahedral: typing.ClassVar[StereoKind]
    DoubleBond: typing.ClassVar[StereoKind]
    SquarePlanar: typing.ClassVar[StereoKind]
    TrigonalBipyramidal: typing.ClassVar[StereoKind]
    Octahedral: typing.ClassVar[StereoKind]

class StereoData:
    """One-based atom references; zero is a virtual non-hydrogen reference."""

    kind: StereoKind
    center: int
    end: int
    refs: list[int]
    configuration: int
    def __init__(self) -> None: ...

class StereoGroupKind:
    Absolute: typing.ClassVar[StereoGroupKind]
    And: typing.ClassVar[StereoGroupKind]
    Or: typing.ClassVar[StereoGroupKind]

class StereoGroupData:
    kind: StereoGroupKind
    atoms: list[int]
    read_id: int
    def __init__(self) -> None: ...

class MoleculeData:
    stereo: list[StereoData]
    stereo_groups: list[StereoGroupData]
    def __init__(self) -> None: ...
    @property
    def atoms(self) -> list[AtomData]: ...
    @atoms.setter
    def atoms(self, arg0: collections.abc.Sequence[AtomData]) -> None: ...
    @property
    def bonds(self) -> list[BondData]: ...
    @bonds.setter
    def bonds(self, arg0: collections.abc.Sequence[BondData]) -> None: ...
    @property
    def total_charge(self) -> int: ...
    @total_charge.setter
    def total_charge(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def total_radical_num(self) -> int: ...
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
