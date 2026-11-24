#pragma once

#include "molgr/state.h"

#include <vector>

namespace molgr
{
    namespace no_metals
    {
        namespace recovery
        {
            /// Lower deformed bonds whose stage explicitly updates endpoint electrons.
            /// @param states Ligand states from which to generate or select electronic alternatives.
            /// @return Electronic states surviving this seed expansion or search stage.
            std::vector<molgr::state::ReconstructionState> EnumerateDeformedPiRecoverySeeds(
                const std::vector<molgr::state::ReconstructionState> &states);

            /// Apply bond-breaking rules whose stages explicitly update endpoint electrons.
            /// @param states Ligand states from which to generate or select electronic alternatives.
            /// @return Electronic states surviving this seed expansion or search stage.
            std::vector<molgr::state::ReconstructionState> EnumerateBondBreakRecoverySeeds(
                const std::vector<molgr::state::ReconstructionState> &states);
        }
    }
}
