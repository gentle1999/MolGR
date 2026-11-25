#pragma once

#include "molgr/config.h"

#include <openbabel/elements.h>

namespace molgr::coordination
{
    /// Return the covalent-radius based metal-ligand coordination cutoff.
    /// @param metal_atomic_num Atomic number of the indicated bond partner or site.
    /// @param ligand_atomic_num Atomic number of the indicated bond partner or site.
    /// @param config Metal contact cutoffs, visibility radii and candidate scoring weights.
    /// @return Maximum accepted donor-metal separation in angstroms.
    inline double CoordinationDistanceCutoff(
        int metal_atomic_num,
        int ligand_atomic_num,
        const molgr::config::MetalScoringConfig &config)
    {
        const double metal_radius =
            OpenBabel::OBElements::GetCovalentRad(metal_atomic_num);
        const double ligand_radius =
            OpenBabel::OBElements::GetCovalentRad(ligand_atomic_num);
        return config.metal_access_radius_scale * (metal_radius + ligand_radius) +
               config.metal_coordination_extra_tolerance_angstrom;
    }
}
