"""The shared geometric seed used by both reconstruction backends."""

from __future__ import annotations

from typing import cast

from openbabel import openbabel as ob
from openbabel import pybel


def read_xyz_seed(xyz_block: str) -> pybel.Molecule:
    """Use MolGR's seed perception before Python's independent search stages.

    Open Babel's XYZ plugin can choose different initial Kekule bond orders
    from the native pipeline's vendor implementation. Those differences can
    change which electronic states are reachable, even with identical scoring.
    Transfer owned molecule data, without sharing an OBMol pointer across APIs.
    """
    from molgr import _core
    from molgr.utils.converter import mol_data_to_pybel

    data = _core.utils.xyz_seed_molecule_data(xyz_block)
    omol = mol_data_to_pybel(data)
    obmol = cast(ob.OBMol, omol.OBMol)
    obmol.SetDimension(3)
    # The native reader invalidates perception caches after assigning bonds.
    # Recompute hybridization on demand after subsequent electronic edits.
    obmol.SetHybridizationPerceived(False)
    obmol.SetTitle(xyz_block.splitlines()[1])
    return omol
