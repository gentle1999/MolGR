#pragma once

#include "molgr/config.h"
#include "molgr/utils/resonance.h"

#include <openbabel/mol.h>

#include <functional>
#include <vector>

namespace molgr
{
    namespace reconstruct
    {
        using molgr::resonance::BuildBondIndexMapFromStateKey;
        using molgr::resonance::BuildProcessedResonanceKey;
        using molgr::resonance::BuildResonanceSearchContext;
        using molgr::resonance::BuildResonanceStateKey;
        using molgr::resonance::EnumerateOneStepResonanceMoves;
        using molgr::resonance::EstimateRemainingResonanceScoreImprovementUpperBound;
        using molgr::resonance::GetOneStepResonance;
        using molgr::resonance::IncrementResonanceStateKey;
        using molgr::resonance::IndexedResonanceTraversalMove;
        using molgr::resonance::LimitedDiscrepancyTraversalConfig;
        using molgr::resonance::MaterializeOneStepResonance;
        using molgr::resonance::MaterializeOneStepResonancePtr;
        using molgr::resonance::ProcessedResonanceKey;
        using molgr::resonance::ProcessResonance;
        using molgr::resonance::ProcessResonanceDetailed;
        using molgr::resonance::ResonanceAtomKey;
        using molgr::resonance::ResonanceBondIndexMap;
        using molgr::resonance::ResonanceBondKey;
        using molgr::resonance::ResonanceSearchNode;
        using molgr::resonance::ResonanceStateKey;
        using molgr::resonance::ResonanceTraversalMove;
        using molgr::resonance::SelectLimitedDiscrepancyMoves;
        using molgr::resonance::SmilesFirstToken;
        using molgr::resonance::UffLiteGainBoundCache;
        using molgr::resonance::UffLiteGainBoundCacheKey;
        using molgr::resonance::UffLiteGainMetrics;

        using ResonanceVisitCallback = std::function<bool(const ResonanceSearchNode &)>;

        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param max_depth Maximum number of successive resonance moves.
        /// @return The radical-delocalization states reachable within the supplied depth.
        std::vector<OpenBabel::OBMol> GetRadicalResonances(
            const OpenBabel::OBMol &mol,
            int max_depth = 2);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param max_depth Maximum number of successive resonance moves.
        /// @param traversal_config Budgets and ordering for limited-discrepancy traversal.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return States reachable within both depth and ranked-move discrepancy budgets.
        std::vector<OpenBabel::OBMol> GetRadicalResonancesLimitedDiscrepancy(
            const OpenBabel::OBMol &mol,
            int max_depth = 2,
            const LimitedDiscrepancyTraversalConfig &traversal_config = {},
            const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());

        /// Visit reachable radical-delocalization states while the visitor permits expansion.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param max_depth Maximum number of successive resonance moves.
        /// @param visit Called for each visited state. Return false to skip expansion from that state.
        void WalkRadicalResonances(
            const OpenBabel::OBMol &mol,
            int max_depth = 2,
            ResonanceVisitCallback visit = {});
        /// Visit resonance states with ranked successors and a limit on departures from the preferred move.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param max_depth Maximum number of successive resonance moves.
        /// @param visit Called for each visited state. Return false to skip expansion from that state.
        /// @param traversal_config Budgets and ordering for limited-discrepancy traversal.
        /// @param config Settings for the current reconstruction or scoring operation.
        void WalkRadicalResonancesLimitedDiscrepancy(
            const OpenBabel::OBMol &mol,
            int max_depth = 2,
            ResonanceVisitCallback visit = {},
            const LimitedDiscrepancyTraversalConfig &traversal_config = {},
            const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());
    }
}
