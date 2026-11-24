#pragma once

#include "molgr/config.h"
#include "molgr/state.h"
#include "molgr/utils/organic_topology.h"

#include <tuple>
#include <vector>

namespace molgr
{
    namespace no_metals
    {
        namespace selection
        {
            using NoMetalTopologySelectionKey =
                std::tuple<int, int, int, int, int, double, double, double, double>;
            using NoMetalGraphTieBreakKey = std::vector<int>;

            /// Compute and store aromaticity and conjugation metrics on the ligand candidate.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Aromaticity, conjugation and hyperconjugation measurements of the ligand.
            molgr::organic_topology::OrganicTopologyMetrics AnnotateNoMetalCandidateTopology(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());

            /// Compute the numeric score used to rank the ligand electronic state.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Numeric ligand reconstruction score.
            double ScoreReconstructionCandidate(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config);

            /// @param candidate Candidate electronic assignment being evaluated.
            /// @return The geometry conflict count, absolute formal-charge sum and stretched-carbonyl
            ///     count.
            std::tuple<int,int,int> NoMetalCandidateGeometryKey(molgr::state::ReconstructionState &candidate);

            /// Rank the candidate by aromatic and conjugated topology under the supplied configuration.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Lexicographic ranking terms, in candidate selection order.
            NoMetalTopologySelectionKey NoMetalCandidateTopologySelectionKey(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());

            /// Rank azinium geometry, charge separation and stretched carbonyls before conjugation and energy.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Lexicographic ranking terms, in candidate selection order.
            std::tuple<int, int, int, int, int, double, double, double, double, int, int, double>
            NoMetalCandidateSelectionKey(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config);

            /// Build the structural ranking prefix before comparing force-field energies.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Lexicographic ranking terms, in candidate selection order.
            std::tuple<int,int,int,int,int,double,double,double,double,int,int> NoMetalCandidateStructuralKey(
                molgr::state::ReconstructionState &candidate, const molgr::config::MolGRConfig &config);

            /// @param candidate Candidate electronic assignment being evaluated.
            /// @return A backend-independent key for exact no-metal score ties.
            NoMetalGraphTieBreakKey NoMetalCandidateGraphTieBreakKey(
                const molgr::state::ReconstructionState &candidate);
        }
    }
}
