#pragma once

#include "molgr/config.h"
#include "molgr/state.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace molgr
{
    namespace context
    {
        /// Carry Cartesian input, charge, unpaired-electron target and settings across reconstruction
        /// phases.
        struct ReconstructionContext
        {
            std::string xyz_block;
            int total_charge = 0;
            int total_radical_electrons = 0;
            const molgr::config::MolGRConfig *config = &molgr::config::GetDefaultConfig();
            std::vector<std::string> phase_history;
            molgr::state::MetadataMap metadata;

            /// @return The selected settings, falling back to the native defaults for a null config
            ///     pointer.
            const molgr::config::MolGRConfig &Config() const
            {
                return config == nullptr ? molgr::config::GetDefaultConfig() : *config;
            }

            /// @return True when the requested number of unpaired electrons is nonnegative.
            bool HasValidRadicalTarget() const
            {
                return total_radical_electrons >= 0;
            }
        };

        /// Key metal-state assignments by the charge and unpaired electrons left for the ligand.
        struct TargetBucket
        {
            int no_metal_charge = 0;
            int no_metal_radicals = 0;

            /// @param other Other instance participating in the comparison.
            /// @return True if the ligand charge/radical pair sorts before the other bucket.
            bool operator<(const TargetBucket &other) const
            {
                return std::tie(no_metal_charge, no_metal_radicals) <
                       std::tie(other.no_metal_charge, other.no_metal_radicals);
            }

            /// @param other Other instance participating in the comparison.
            /// @return True if both ligand electronic targets match.
            bool operator==(const TargetBucket &other) const
            {
                return no_metal_charge == other.no_metal_charge &&
                       no_metal_radicals == other.no_metal_radicals;
            }

            /// @return Ligand charge followed by the ligand unpaired-electron count.
            std::pair<int, int> AsPair() const
            {
                return {no_metal_charge, no_metal_radicals};
            }
        };
    }
}
