"""MolGR atom-local electron bookkeeping for Open Babel atoms."""

from __future__ import annotations

from openbabel import openbabel as ob


UNPAIRED_ELECTRON_COUNT_PROP = "MOLGR_UNPAIRED_ELECTRON_COUNT"
LONE_PAIR_COUNT_PROP = "MOLGR_LONE_PAIR_COUNT"
UNRESOLVED_TWO_ELECTRON_CENTER_PROP = "MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER"


def _get_int_data(atom: ob.OBAtom, attribute: str) -> int:
    data = atom.GetData(attribute)
    if data is None:
        return 0
    try:
        return int(ob.toPairData(data).GetValue())
    except (TypeError, ValueError):
        return 0


def _set_int_data(atom: ob.OBAtom, attribute: str, value: int) -> None:
    """Preserve explicit zero; missing generic data continues to mean unknown."""

    atom.DeleteData(attribute)
    data = ob.OBPairData()
    data.SetAttribute(attribute)
    data.SetValue(str(int(value)))
    data.SetOrigin(ob.local)
    # The Python bindings do not expose OBBase::SetData. CloneData owns a copy.
    atom.CloneData(data)


def get_unpaired_electron_count(atom: ob.OBAtom) -> int:
    return _get_int_data(atom, UNPAIRED_ELECTRON_COUNT_PROP)


def set_unpaired_electron_count(atom: ob.OBAtom, value: int) -> None:
    """Store the number of real unpaired electrons; never encode lone pairs here."""

    if value < 0:
        raise ValueError("unpaired electron count must be nonnegative")
    _set_int_data(atom, UNPAIRED_ELECTRON_COUNT_PROP, value)


def get_lone_pair_count(atom: ob.OBAtom) -> int:
    return _get_int_data(atom, LONE_PAIR_COUNT_PROP)


def set_lone_pair_count(atom: ob.OBAtom, value: int) -> None:
    """Store reconstruction-active lone pairs, not the atom's full Lewis count."""

    if value < 0:
        raise ValueError("lone pair count must be nonnegative")
    _set_int_data(atom, LONE_PAIR_COUNT_PROP, value)


def has_unresolved_two_electron_center(atom: ob.OBAtom) -> bool:
    return bool(_get_int_data(atom, UNRESOLVED_TWO_ELECTRON_CENTER_PROP))


def set_unresolved_two_electron_center(atom: ob.OBAtom, value: bool) -> None:
    """Mark a C/N/P-like two-electron deficit whose spin occupancy is deferred."""

    _set_int_data(atom, UNRESOLVED_TWO_ELECTRON_CENTER_PROP, int(value))


__all__ = [
    "LONE_PAIR_COUNT_PROP",
    "UNPAIRED_ELECTRON_COUNT_PROP",
    "UNRESOLVED_TWO_ELECTRON_CENTER_PROP",
    "get_lone_pair_count",
    "get_unpaired_electron_count",
    "has_unresolved_two_electron_center",
    "set_lone_pair_count",
    "set_unpaired_electron_count",
    "set_unresolved_two_electron_center",
]
