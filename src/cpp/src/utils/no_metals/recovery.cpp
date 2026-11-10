#include "molgr/utils/no_metals/recovery.h"

#include "molgr/stages/break_bond.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/resonance.h"
#include "molgr/vendor/openbabel_copy.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace
{
    using StateKey = std::pair<molgr::resonance::ResonanceStateKey, int>;

    std::vector<molgr::state::ReconstructionState> Deduplicate(
        std::vector<molgr::state::ReconstructionState> states)
    {
        std::set<StateKey> seen;
        std::vector<molgr::state::ReconstructionState> unique;
        for (auto &state : states)
        {
            StateKey key{molgr::resonance::BuildResonanceStateKey(state.Mol()), state.given_charge};
            if (seen.insert(std::move(key)).second)
            {
                unique.push_back(std::move(state));
            }
        }
        return unique;
    }
}

namespace molgr
{
    namespace no_metals
    {
        namespace recovery
        {
            std::vector<molgr::state::ReconstructionState> EnumerateDeformedPiRecoverySeeds(
                const std::vector<molgr::state::ReconstructionState> &states)
            {
                std::vector<molgr::state::ReconstructionState> recovered;
                for (const auto &state : states)
                {
                    auto machine = molgr::state::OmolStateMachine::FromReconstructionState(state);
                    auto next_mol = molgr::utils::CloneMolTopologyShared(state.Mol());
                    machine = machine.Branch(std::nullopt, std::move(next_mol));
                    const bool hit = machine.RunOmolStage(
                        "recover_deformed_pi_bonds",
                        reconstruct::BreakDeformedEne,
                        machine.given_charge,
                        state.total_radical_electrons,
                        5.0);
                    if (!hit)
                    {
                        continue;
                    }
                    machine.metadata["recovery_tier"] = 1;
                    machine.metadata["recovery_strategy"] = std::string("deformed_pi_bonds");
                    recovered.push_back(machine.FreezeLike(state));
                }
                return Deduplicate(std::move(recovered));
            }

            std::vector<molgr::state::ReconstructionState> EnumerateBondBreakRecoverySeeds(
                const std::vector<molgr::state::ReconstructionState> &states)
            {
                std::vector<molgr::state::ReconstructionState> recovered;
                for (const auto &state : states)
                {
                    auto machine = molgr::state::OmolStateMachine::FromReconstructionState(state);
                    auto next_mol = molgr::utils::CloneMolTopologyShared(state.Mol());
                    machine = machine.Branch(std::nullopt, std::move(next_mol));
                    const bool hit = machine.RunOmolChargeStage(
                        "recover_by_breaking_bonds",
                        reconstruct::BreakOneBond,
                        state.total_radical_electrons);
                    if (!hit)
                    {
                        continue;
                    }
                    machine.metadata["recovery_tier"] = 2;
                    machine.metadata["recovery_strategy"] = std::string("bond_break");
                    recovered.push_back(machine.FreezeLike(state));
                }
                return Deduplicate(std::move(recovered));
            }
        }
    }
}
