#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace molgr
{
    // Geometry of an input metal, before charge and spin assignment.
    /// Element and Cartesian coordinates of a metal before assigning its charge and spin.
    struct MetalSite
    {
        int element_idx;
        double position_x, position_y, position_z;
    };

    /// A candidate metal charge and unpaired-electron count at the original one-based atom index and Cartesian position.
    struct MetalAtomPosition
    {
        int idx = 0;
        std::string symbol;
        int element_idx = 0;
        int valence = 0;
        int radical_num = 0;
        double position_x = 0.0;
        double position_y = 0.0;
        double position_z = 0.0;

        /// @return The same metal coordinates for geometry tests shared across charge and spin assignments.
        MetalSite Site() const { return {element_idx, position_x, position_y, position_z}; }
    };

    using ChargedAtomSnapshot = std::tuple<int, double, double, double>;
    using ChargedAtomSnapshotList = std::vector<ChargedAtomSnapshot>;

    namespace metal
    {
        using MetalAtomPosition = ::molgr::MetalAtomPosition;
    }
}
