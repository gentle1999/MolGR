"""MolGR atom-local electron bookkeeping for Open Babel atoms."""

from __future__ import annotations

from functools import lru_cache
from typing import cast

from openbabel import openbabel as ob


UNPAIRED_ELECTRON_COUNT_PROP = "MOLGR_UNPAIRED_ELECTRON_COUNT"
LONE_PAIR_COUNT_PROP = "MOLGR_LONE_PAIR_COUNT"
UNRESOLVED_TWO_ELECTRON_CENTER_PROP = "MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER"


def has_cationic_lewis_pair(
    outer_electron_count: int,
    formal_charge: int,
    covalent_valence: float,
    unpaired_electrons: int,
    unresolved: bool,
) -> bool:
    """Count a resolved cation's Lewis pairs, not reconstruction-active pairs.

    Each covalent bond-order unit uses one of the atom's electrons. Subtract
    unpaired electrons before identifying pairs; deferred occupancies and
    fractional bond-order sums cannot establish an available pair.
    """
    if formal_charge <= 0 or covalent_valence < 0 or unpaired_electrons < 0 or unresolved:
        return False
    paired_electrons = outer_electron_count - formal_charge - covalent_valence - unpaired_electrons
    return paired_electrons >= 2 and paired_electrons % 2 == 0


def _get_int_data(atom: ob.OBAtom, attribute: str) -> int:
    data = cast("ob.OBGenericData | None", atom.GetData(attribute))
    if data is None:
        return 0
    try:
        # OBPairData supplies its string value through the virtual OBGenericData accessor.
        return int(data.GetValue())
    except (TypeError, ValueError):
        return 0


@lru_cache(maxsize=32)
def _electron_label(attribute: str, value: int) -> ob.OBPairData:
    """Create a label template. Each atom receives a separate Open Babel copy."""
    data = ob.OBPairData()
    data.SetAttribute(attribute)
    data.SetValue(str(value))
    data.SetOrigin(ob.local)
    return data


def _set_int_data(atom: ob.OBAtom, attribute: str, value: int) -> None:
    """Preserve explicit zero; missing generic data continues to mean unknown."""
    atom.DeleteData(attribute)
    atom.CloneData(_electron_label(attribute, int(value)))


def get_unpaired_electron_count(atom: ob.OBAtom) -> int:
    """Read the explicit radical count used in the global spin budget."""
    return _get_int_data(atom, UNPAIRED_ELECTRON_COUNT_PROP)


def set_unpaired_electron_count(atom: ob.OBAtom, value: int) -> None:
    """Store the number of real unpaired electrons; never encode lone pairs here."""

    if value < 0:
        raise ValueError("unpaired electron count must be nonnegative")
    _set_int_data(atom, UNPAIRED_ELECTRON_COUNT_PROP, value)


def get_lone_pair_count(atom: ob.OBAtom) -> int:
    """Read the search-active lone pairs available for local bond rearrangements."""
    return _get_int_data(atom, LONE_PAIR_COUNT_PROP)


def set_lone_pair_count(atom: ob.OBAtom, value: int) -> None:
    """Store reconstruction-active lone pairs, not the atom's full Lewis count."""

    if value < 0:
        raise ValueError("lone pair count must be nonnegative")
    _set_int_data(atom, LONE_PAIR_COUNT_PROP, value)


def has_unresolved_two_electron_center(atom: ob.OBAtom) -> bool:
    """Check whether the radical budget still has to resolve a two-electron center."""
    return bool(_get_int_data(atom, UNRESOLVED_TWO_ELECTRON_CENTER_PROP))


def set_unresolved_two_electron_center(atom: ob.OBAtom, value: bool) -> None:
    """Mark a C/N/Si/P-like two-electron deficit whose spin occupancy is deferred."""

    _set_int_data(atom, UNRESOLVED_TWO_ELECTRON_CENTER_PROP, int(value))


__all__ = [
    "LONE_PAIR_COUNT_PROP",
    "UNPAIRED_ELECTRON_COUNT_PROP",
    "UNRESOLVED_TWO_ELECTRON_CENTER_PROP",
    "get_lone_pair_count",
    "get_unpaired_electron_count",
    "has_cationic_lewis_pair",
    "has_unresolved_two_electron_center",
    "set_lone_pair_count",
    "set_unpaired_electron_count",
    "set_unresolved_two_electron_center",
]
