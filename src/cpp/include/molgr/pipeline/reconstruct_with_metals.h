#pragma once

#include "molgr/config.h"
#include "molgr/state.h"
#include "molgr/diagnostics.h"
#include "molgr/utils/utils.h"

#include <memory>
#include <string>

namespace molgr
{
    namespace pipeline
    {
        namespace reconstruct_with_metals
        {
            /// Reconstruct a graph at the requested net charge and unpaired-electron count. Optional outputs
            /// retain failure diagnostics and metal candidates.
            /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
            ///     angstroms.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @param diagnostics Diagnostic record for the current reconstruction.
            /// @param candidate_pool Optional output for the metal candidates considered during selection.
            /// @return Reconstructed molecular records, or null if the targets cannot be satisfied.
            std::unique_ptr<molgr::utils::MoleculeData> Xyz2OmolMolData(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig(),
                molgr::diagnostics::ReconstructionDiagnostics *diagnostics = nullptr,
                std::vector<molgr::state::MetalCandidateState> *candidate_pool = nullptr);
        }
    }
}
