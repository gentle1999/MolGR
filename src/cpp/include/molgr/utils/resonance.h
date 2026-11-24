#pragma once

#include "molgr/config.h"

#include <openbabel/mol.h>

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace molgr
{
    namespace resonance
    {
        using ResonanceAtomKey = std::tuple<int, int, int, int, bool, bool>;
        using ResonanceBondKey = std::tuple<int, int, int, bool>;
        using ResonanceBondIndexMap = std::map<std::pair<int, int>, std::size_t>;
        using UffLiteGainMetrics = std::array<double, 4>;

        /// Atom labels and sorted bond orders used to recognize identical resonance states.
        struct ResonanceStateKey
        {
            std::vector<ResonanceAtomKey> atom_keys;
            std::vector<ResonanceBondKey> bond_keys;

            /// @param other Other instance participating in the comparison.
            /// @return True when the stored atom labels and bond data agree.
            bool operator==(const ResonanceStateKey &other) const;
            /// Order resonance keys deterministically for sets and tie-breaking.
            /// @param other Other instance participating in the comparison.
            /// @return True if this resonance fingerprint precedes the other.
            bool operator<(const ResonanceStateKey &other) const;
        };

        using ProcessedResonanceKey = std::string;

        /// A one-step radical resonance move and the incremental key of its output state.
        struct ResonanceTraversalMove
        {
            std::tuple<int, int, int> path;
            ResonanceStateKey next_state_key;
        };

        /// A resonance state yielded during traversal.
        struct ResonanceSearchNode
        {
            const OpenBabel::OBMol &omol;
            const ResonanceStateKey &state_key;
            int depth;
            int discrepancy;
            std::shared_ptr<OpenBabel::OBMol> omol_owner;
        };

        /// One resonance move paired with its position in the traversal ordering.
        struct IndexedResonanceTraversalMove
        {
            std::tuple<int, int, int> idxs;
            ResonanceStateKey next_state_key;
        };

        using UffLiteGainBoundCacheKey = std::pair<ResonanceStateKey, int>;
        using UffLiteGainBoundCache = std::map<UffLiteGainBoundCacheKey, UffLiteGainMetrics>;

        /// Limit the accumulated departures from the preferred successor during resonance traversal.
        struct LimitedDiscrepancyTraversalConfig
        {
            int max_discrepancy = 1;
        };

        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return The graph token of the SMILES output without its title.
        std::string SmilesFirstToken(const OpenBabel::OBMol &mol);

        /// Build the structural key used to deduplicate raw resonance states.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Fingerprint for deduplicating the corresponding electronic assignment.
        ResonanceStateKey BuildResonanceStateKey(const OpenBabel::OBMol &mol);
        /// Build the dedup key after `process_resonance` has normalized the candidate.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Fingerprint for deduplicating the corresponding electronic assignment.
        ProcessedResonanceKey BuildProcessedResonanceKey(const OpenBabel::OBMol &mol);
        /// Prepare the initial resonance fingerprint together with its bond lookup table.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Resonance fingerprint and its endpoint-to-bond-position index.
        std::pair<ResonanceStateKey, ResonanceBondIndexMap> BuildResonanceSearchContext(
            const OpenBabel::OBMol &mol);
        /// Map bond endpoints to positions in the resonance state's bond vector.
        /// @param state_key Fingerprint of the current resonance state.
        /// @return Map from ordered atom pairs to entries in the fingerprint's bond vector.
        ResonanceBondIndexMap BuildBondIndexMapFromStateKey(
            const ResonanceStateKey &state_key);
        /// Update the parent fingerprint for the local charge, radical and bond changes of one move.
        /// @param state_key Fingerprint of the current resonance state.
        /// @param bond_index_map Map from atom endpoints to positions in the bond-order fingerprint.
        /// @param idxs Atom indices along the proposed electron-shift path.
        /// @param target_unresolved Whether the endpoint must be marked as a deferred two-electron center.
        /// @return Fingerprint after applying the local electron shift.
        ResonanceStateKey IncrementResonanceStateKey(
            const ResonanceStateKey &state_key,
            const ResonanceBondIndexMap &bond_index_map,
            const std::tuple<int, int, int> &idxs,
            bool target_unresolved = false);

        /// Enumerate one-electron radical moves across an adjacent pi bond.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param state_key Fingerprint of the current resonance state.
        /// @param bond_index_map Map from atom endpoints to positions in the bond-order fingerprint.
        /// @return Permitted one-step moves with atom paths and resulting fingerprints.
        std::vector<IndexedResonanceTraversalMove> EnumerateOneStepResonanceMoves(
            const OpenBabel::OBMol &mol,
            const ResonanceStateKey &state_key,
            const ResonanceBondIndexMap &bond_index_map);
        /// Move one real unpaired electron and the coupled pi-bond pattern.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param idxs Atom indices along the proposed electron-shift path.
        /// @return Graph with the specified radical and adjacent pi bond shifted.
        OpenBabel::OBMol MaterializeOneStepResonance(
            const OpenBabel::OBMol &mol,
            const std::tuple<int, int, int> &idxs);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param idxs Atom indices along the proposed electron-shift path.
        /// @return An owned molecular graph with the one-step resonance move applied.
        std::shared_ptr<OpenBabel::OBMol> MaterializeOneStepResonancePtr(
            const OpenBabel::OBMol &mol,
            const std::tuple<int, int, int> &idxs);
        /// Enumerate distinct immediate resonance neighbors of the supplied graph.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Distinct graphs reachable by one radical shift.
        std::vector<OpenBabel::OBMol> GetOneStepResonance(const OpenBabel::OBMol &mol);

        /// Select ranked successors subject to the remaining discrepancy budget.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param moves Candidate one-step electron shifts to select or order.
        /// @param traversal_config Budgets and ordering for limited-discrepancy traversal.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return Permitted one-step moves with atom paths and resulting fingerprints.
        std::vector<IndexedResonanceTraversalMove> SelectLimitedDiscrepancyMoves(
            const OpenBabel::OBMol &mol,
            const std::vector<IndexedResonanceTraversalMove> &moves,
            const LimitedDiscrepancyTraversalConfig &traversal_config,
            const molgr::config::MolGRConfig &config);

        /// Bound the improvement reachable by the remaining resonance steps for pruning.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param state_key Fingerprint of the current resonance state.
        /// @param remaining_steps Remaining depth budget in one-step electron shifts.
        /// @param cache Stored evaluations or bounds indexed by the molecular fingerprint.
        /// @return Upper bound on the remaining improvement in candidate score.
        double EstimateRemainingResonanceScoreImprovementUpperBound(
            const OpenBabel::OBMol &mol,
            const ResonanceStateKey &state_key,
            int remaining_steps,
            UffLiteGainBoundCache *cache = nullptr);

        /// Apply the resonance normalizations and retain the resulting graph and charge bookkeeping.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @param total_charge Target net formal charge of the full input.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @return Normalized graph, outstanding charge and whether normalization changed the state.
        std::tuple<OpenBabel::OBMol, int, bool> ProcessResonanceDetailed(
            const OpenBabel::OBMol &mol,
            int charge,
            int total_charge = 0,
            int total_radical_electrons = 0);
        /// Normalize one resonance candidate using only classified electron states.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @param total_charge Target net formal charge of the full input.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @return Normalized graph and remaining charge.
        std::pair<OpenBabel::OBMol, int> ProcessResonance(
            const OpenBabel::OBMol &mol,
            int charge,
            int total_charge = 0,
            int total_radical_electrons = 0);
    }
}
