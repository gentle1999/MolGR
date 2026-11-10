#include "molgr/pipeline/resonance.h"

#include "molgr/utils/conversions.h"
#include "molgr/vendor/openbabel_copy.h"

#include <deque>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <utility>

namespace
{
    struct QueueEntry
    {
        int discrepancy = 0;
        int depth = 0;
        int order = 0;
        std::shared_ptr<OpenBabel::OBMol> omol;
        molgr::reconstruct::ResonanceStateKey state_key;
    };

    struct QueueEntryGreater
    {
        bool operator()(const QueueEntry &lhs, const QueueEntry &rhs) const
        {
            return std::tie(lhs.discrepancy, lhs.depth, lhs.order) >
                   std::tie(rhs.discrepancy, rhs.depth, rhs.order);
        }
    };
}

namespace molgr
{
    namespace reconstruct
    {
        void WalkRadicalResonances(
            const OpenBabel::OBMol &mol,
            int max_depth,
            ResonanceVisitCallback visit)
        {
            if (!visit)
            {
                visit = [](const ResonanceSearchNode &) { return true; };
            }

            const auto [root_key, bond_index_map] = BuildResonanceSearchContext(mol);
            std::set<ResonanceStateKey> seen{root_key};
            std::deque<std::tuple<std::shared_ptr<OpenBabel::OBMol>, ResonanceStateKey, int>> frontier{
                {molgr::utils::CloneMolTopologyShared(mol),
                 root_key,
                 0}};

            while (!frontier.empty())
            {
                auto [current, current_key, depth] = std::move(frontier.front());
                frontier.pop_front();

                const bool should_expand = visit(ResonanceSearchNode{
                    *current,
                    current_key,
                    depth,
                    0,
                    current,
                });
                if (depth >= max_depth || !should_expand)
                {
                    continue;
                }

                const auto moves = EnumerateOneStepResonanceMoves(*current, current_key, bond_index_map);
                for (const auto &move : moves)
                {
                    if (seen.find(move.next_state_key) != seen.end())
                    {
                        continue;
                    }
                    seen.insert(move.next_state_key);
                    frontier.emplace_back(
                        MaterializeOneStepResonancePtr(*current, move.idxs),
                        move.next_state_key,
                        depth + 1);
                }
            }
        }

        void WalkRadicalResonancesLimitedDiscrepancy(
            const OpenBabel::OBMol &mol,
            int max_depth,
            ResonanceVisitCallback visit,
            const LimitedDiscrepancyTraversalConfig &traversal_config,
            const molgr::config::MolGRConfig &config)
        {
            if (!visit)
            {
                visit = [](const ResonanceSearchNode &) { return true; };
            }

            const auto [root_key, bond_index_map] = BuildResonanceSearchContext(mol);
            std::map<ResonanceStateKey, int> best_discrepancy_by_state{{root_key, 0}};
            std::set<ResonanceStateKey> emitted_states;
            std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> frontier;
            frontier.push(QueueEntry{
                0,
                0,
                0,
                molgr::utils::CloneMolTopologyShared(mol),
                root_key});

            int push_order = 0;
            while (!frontier.empty())
            {
                QueueEntry current_entry = std::move(frontier.top());
                frontier.pop();

                const auto best_it = best_discrepancy_by_state.find(current_entry.state_key);
                if (best_it == best_discrepancy_by_state.end() ||
                    current_entry.discrepancy != best_it->second)
                {
                    continue;
                }

                bool should_expand = true;
                if (emitted_states.find(current_entry.state_key) == emitted_states.end())
                {
                    emitted_states.insert(current_entry.state_key);
                    should_expand = visit(ResonanceSearchNode{
                        *current_entry.omol,
                        current_entry.state_key,
                        current_entry.depth,
                        current_entry.discrepancy,
                        current_entry.omol,
                    });
                }

                if (current_entry.depth >= max_depth || !should_expand)
                {
                    continue;
                }

                const auto moves = EnumerateOneStepResonanceMoves(
                    *current_entry.omol,
                    current_entry.state_key,
                    bond_index_map);
                const auto selected_moves = SelectLimitedDiscrepancyMoves(
                    *current_entry.omol,
                    moves,
                    traversal_config,
                    config);
                for (std::size_t move_rank = 0; move_rank < selected_moves.size(); ++move_rank)
                {
                    const int next_discrepancy =
                        current_entry.discrepancy + static_cast<int>(move_rank);
                    if (next_discrepancy > traversal_config.max_discrepancy)
                    {
                        break;
                    }

                    const auto &move = selected_moves[move_rank];
                    const auto best_known_it = best_discrepancy_by_state.find(move.next_state_key);
                    if (best_known_it != best_discrepancy_by_state.end() &&
                        best_known_it->second <= next_discrepancy)
                    {
                        continue;
                    }

                    best_discrepancy_by_state[move.next_state_key] = next_discrepancy;
                    frontier.push(QueueEntry{
                        next_discrepancy,
                        current_entry.depth + 1,
                        ++push_order,
                        MaterializeOneStepResonancePtr(*current_entry.omol, move.idxs),
                        move.next_state_key,
                    });
                }
            }
        }

        std::vector<OpenBabel::OBMol> GetRadicalResonances(const OpenBabel::OBMol &mol, int max_depth)
        {
            std::vector<OpenBabel::OBMol> resonances;
            WalkRadicalResonances(
                mol,
                max_depth,
                [&resonances](const ResonanceSearchNode &node)
                {
                    std::lock_guard<std::recursive_mutex> lock(
                        molgr::vendor::openbabel_threading::NativeOpenBabelMutex());
                    resonances.push_back(node.omol);
                    return true;
                });
            return resonances;
        }

        std::vector<OpenBabel::OBMol> GetRadicalResonancesLimitedDiscrepancy(
            const OpenBabel::OBMol &mol,
            int max_depth,
            const LimitedDiscrepancyTraversalConfig &traversal_config,
            const molgr::config::MolGRConfig &config)
        {
            std::vector<OpenBabel::OBMol> resonances;
            WalkRadicalResonancesLimitedDiscrepancy(
                mol,
                max_depth,
                [&resonances](const ResonanceSearchNode &node)
                {
                    std::lock_guard<std::recursive_mutex> lock(
                        molgr::vendor::openbabel_threading::NativeOpenBabelMutex());
                    resonances.push_back(node.omol);
                    return true;
                },
                traversal_config,
                config);
            return resonances;
        }
    }
}
