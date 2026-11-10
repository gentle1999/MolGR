#include "molgr/utils/no_metals/neighbor_radicals.h"

#include "molgr/stages/fresh.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/resonance.h"
#include "molgr/vendor/openbabel_copy.h"

#include "molgr/compat/openbabel_iter.h"

#include <openbabel/bond.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace
{
    constexpr std::size_t kMaxNeighborRadicalSeeds = 256;
    const std::string kChargeSeparationDiscrepancyKey =
        "neighbor_radical_charge_separation_discrepancy";
    using StateKey = std::pair<molgr::resonance::ResonanceStateKey, int>;
    using EnumerationKey = std::tuple<molgr::resonance::ResonanceStateKey, int, int>;

    StateKey BuildStateKey(const molgr::state::ReconstructionState &state)
    {
        return {molgr::resonance::BuildResonanceStateKey(state.Mol()), state.given_charge};
    }

    int ChargeSeparationDiscrepancy(const molgr::state::ReconstructionState &state)
    {
        const auto it = state.metadata.find(kChargeSeparationDiscrepancyKey);
        if (it != state.metadata.end())
        {
            if (const auto *value = std::get_if<int>(&it->second))
            {
                return *value;
            }
        }
        return 0;
    }

    // Electron bookkeeping: consume the same number of real unpaired electrons
    // at both endpoints. Bond mode converts each pair to one bond-order increment
    // and refreshes endpoints; charge mode converts them to +n/-n. The n>1 charge
    // branch is a documented high-spin heuristic. Validate before mutation.
    bool ResolveNeighborRadicalPair(
        OpenBabel::OBMol &mol,
        int begin_idx,
        int end_idx,
        const std::string &mode,
        int positive_atom_idx)
    {
        auto *begin_atom = mol.GetAtom(begin_idx);
        auto *end_atom = mol.GetAtom(end_idx);
        if (begin_atom == nullptr || end_atom == nullptr)
        {
            return false;
        }
        const int spin_to_consume = std::min(
            molgr::utils::GetUnpairedElectronCount(*begin_atom),
            molgr::utils::GetUnpairedElectronCount(*end_atom));
        if (spin_to_consume <= 0)
        {
            return false;
        }

        OpenBabel::OBBond *bond = nullptr;
        if (mode == "bond_order")
        {
            bond = mol.GetBond(begin_idx, end_idx);
            if (bond == nullptr)
            {
                return false;
            }
        }
        else if (mode != "charge_separation" ||
                 (positive_atom_idx != begin_idx && positive_atom_idx != end_idx))
        {
            return false;
        }

        molgr::utils::SetUnpairedElectronCount(*begin_atom, molgr::utils::GetUnpairedElectronCount(*begin_atom) - spin_to_consume);
        molgr::utils::SetUnpairedElectronCount(*end_atom, molgr::utils::GetUnpairedElectronCount(*end_atom) - spin_to_consume);
        if (mode == "bond_order")
        {
            bond->SetBondOrder(bond->GetBondOrder() + spin_to_consume);
            molgr::reconstruct::AssignChargeRadicalForAtom(*begin_atom);
            molgr::reconstruct::AssignChargeRadicalForAtom(*end_atom);
            return true;
        }

        auto *positive_atom = positive_atom_idx == begin_idx ? begin_atom : end_atom;
        auto *negative_atom = positive_atom_idx == begin_idx ? end_atom : begin_atom;
        positive_atom->SetFormalCharge(positive_atom->GetFormalCharge() + spin_to_consume);
        negative_atom->SetFormalCharge(negative_atom->GetFormalCharge() - spin_to_consume);
        return true;
    }

    std::optional<molgr::state::ReconstructionState> ResolveSeed(
        const molgr::state::ReconstructionState &state,
        int begin_idx,
        int end_idx,
        const std::string &mode,
        int positive_atom_idx)
    {
        auto machine = molgr::state::OmolStateMachine::FromReconstructionState(state);
        auto next_mol = molgr::utils::CloneMolTopologyShared(state.Mol());
        machine = machine.Branch(std::nullopt, std::move(next_mol));
        const std::string phase = mode == "bond_order"
                                      ? "resolve_neighbor_radicals_by_bond_order"
                                      : "resolve_neighbor_radicals_by_charge_separation";
        const bool hit = machine.RunOmolStage(
            phase,
            ResolveNeighborRadicalPair,
            begin_idx,
            end_idx,
            mode,
            positive_atom_idx);
        if (!hit)
        {
            return std::nullopt;
        }

        std::string action;
        if (mode == "bond_order")
        {
            action = "bond_order:" + std::to_string(begin_idx) + "-" + std::to_string(end_idx);
            machine.metadata.erase("positive_atom_idx");
        }
        else
        {
            const int negative_atom_idx = positive_atom_idx == begin_idx ? end_idx : begin_idx;
            action = "charge_separation:" + std::to_string(positive_atom_idx) + "+" +
                     std::to_string(negative_atom_idx) + "-";
            machine.metadata["positive_atom_idx"] = positive_atom_idx;
        }
        const auto actions_it = machine.metadata.find("neighbor_radical_actions");
        if (actions_it != machine.metadata.end())
        {
            if (const auto *previous = std::get_if<std::string>(&actions_it->second))
            {
                action = *previous + ";" + action;
            }
        }
        machine.metadata["neighbor_radical_actions"] = action;
        machine.metadata["neighbor_radical_resolution"] = mode;
        int discrepancy = ChargeSeparationDiscrepancy(state);
        if (mode == "charge_separation")
        {
            ++discrepancy;
        }
        machine.metadata[kChargeSeparationDiscrepancyKey] = discrepancy;
        return machine.FreezeLike(state);
    }
}

namespace molgr
{
    namespace no_metals
    {
        namespace neighbor_radicals
        {
            std::vector<std::pair<int, int>> NeighborRadicalBondPairs(
                const OpenBabel::OBMol &mol)
            {
                std::set<std::pair<int, int>> unique;
                auto &mutable_mol = const_cast<OpenBabel::OBMol &>(mol);
                FOR_BONDS_OF_MOL(bond_iter, mutable_mol)
                {
                    auto *begin_atom = bond_iter->GetBeginAtom();
                    auto *end_atom = bond_iter->GetEndAtom();
                    if (begin_atom != nullptr && end_atom != nullptr &&
                        molgr::utils::GetUnpairedElectronCount(*begin_atom) > 0 &&
                        molgr::utils::GetUnpairedElectronCount(*end_atom) > 0)
                    {
                        unique.emplace(
                            std::min(begin_atom->GetIdx(), end_atom->GetIdx()),
                            std::max(begin_atom->GetIdx(), end_atom->GetIdx()));
                    }
                }
                return {unique.begin(), unique.end()};
            }

            std::vector<molgr::state::ReconstructionState> EnumerateNeighborRadicalSeeds(
                const molgr::state::ReconstructionState &state,
                std::optional<int> exact_discrepancy)
            {
                if (exact_discrepancy.has_value() && *exact_discrepancy < 0)
                {
                    return {};
                }
                if (NeighborRadicalBondPairs(state.Mol()).empty())
                {
                    auto machine = molgr::state::OmolStateMachine::FromReconstructionState(state);
                    machine.Annotate("neighbor_radicals_not_present");
                    machine.metadata["neighbor_radical_resolution"] = std::string("none");
                    machine.metadata[kChargeSeparationDiscrepancyKey] = 0;
                    if (exact_discrepancy.has_value() && *exact_discrepancy != 0)
                    {
                        return {};
                    }
                    return {machine.FreezeLike(state)};
                }

                std::vector<molgr::state::ReconstructionState> pending{state};
                std::vector<molgr::state::ReconstructionState> finished;
                std::set<EnumerationKey> expanded;
                std::set<StateKey> finished_keys;
                while (!pending.empty() && expanded.size() < kMaxNeighborRadicalSeeds)
                {
                    auto current = std::move(pending.back());
                    pending.pop_back();
                    const auto current_key = BuildStateKey(current);
                    const int current_discrepancy = ChargeSeparationDiscrepancy(current);
                    const int discrepancy_key = exact_discrepancy.has_value()
                                                    ? current_discrepancy
                                                    : 0;
                    const EnumerationKey enumeration_key{
                        current_key.first,
                        current_key.second,
                        discrepancy_key,
                    };
                    if (!expanded.insert(enumeration_key).second)
                    {
                        continue;
                    }
                    if (exact_discrepancy.has_value() &&
                        current_discrepancy > *exact_discrepancy)
                    {
                        continue;
                    }
                    const auto pairs = NeighborRadicalBondPairs(current.Mol());
                    if (pairs.empty())
                    {
                        if ((!exact_discrepancy.has_value() ||
                             current_discrepancy == *exact_discrepancy) &&
                            finished_keys.insert(current_key).second)
                        {
                            finished.push_back(std::move(current));
                        }
                        continue;
                    }

                    for (const auto &[begin_idx, end_idx] : pairs)
                    {
                        const std::vector<std::pair<std::string, int>> actions{
                            {"bond_order", 0},
                            {"charge_separation", begin_idx},
                            {"charge_separation", end_idx},
                        };
                        for (const auto &[mode, positive_atom_idx] : actions)
                        {
                            auto candidate = ResolveSeed(
                                current,
                                begin_idx,
                                end_idx,
                                mode,
                                positive_atom_idx);
                            if (candidate.has_value())
                            {
                                pending.push_back(std::move(*candidate));
                            }
                        }
                    }
                }

                std::sort(
                    finished.begin(),
                    finished.end(),
                    [](const auto &left, const auto &right)
                    {
                        const auto left_it = left.metadata.find("neighbor_radical_actions");
                        const auto right_it = right.metadata.find("neighbor_radical_actions");
                        const auto left_value = left_it == left.metadata.end()
                                                    ? std::string()
                                                    : std::get<std::string>(left_it->second);
                        const auto right_value = right_it == right.metadata.end()
                                                     ? std::string()
                                                     : std::get<std::string>(right_it->second);
                        return left_value < right_value;
                    });
                return finished;
            }
        }
    }
}
