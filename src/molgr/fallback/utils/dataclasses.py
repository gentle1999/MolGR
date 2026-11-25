import dataclasses


@dataclasses.dataclass(frozen=True)
class ElementInfo:
    """Outer-shell electron count and reference valence used to classify nonmetal centers."""

    symbol: str
    num_outer_electrons: int
    default_valence: int


@dataclasses.dataclass(frozen=True)
class FDSP:
    """Neutral-atom subshell occupations used when enumerating metal charge and spin."""

    f: int
    d: int
    s: int
    p: int


@dataclasses.dataclass(frozen=True)
class MetalSite:
    """Element and input coordinates of an electronically unassigned metal."""

    element_idx: int
    position_x: float
    position_y: float
    position_z: float


@dataclasses.dataclass
class MetalAtomPosition:
    """One metal charge and unpaired-electron assignment at its original XYZ position."""

    idx: int
    symbol: str
    element_idx: int
    valence: int
    radical_num: int
    position_x: float
    position_y: float
    position_z: float

    @property
    def site(self) -> MetalSite:
        """Return the same metal coordinates for geometry tests shared across charge and spin assignments."""
        return MetalSite(self.element_idx, self.position_x, self.position_y, self.position_z)


__all__ = ["ElementInfo", "FDSP", "MetalAtomPosition", "MetalSite"]
