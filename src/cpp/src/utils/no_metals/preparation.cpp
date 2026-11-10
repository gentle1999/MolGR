#include "molgr/utils/no_metals/preparation.h"

#include "molgr/stages/clean.h"
#include "molgr/stages/eliminate.h"
#include "molgr/stages/fresh.h"
#include "molgr/stages/preprocess.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/xyz.h"
#include "molgr/vendor/openbabel_copy.h"

#include "molgr/compat/openbabel_iter.h"

#include <memory>
#include <utility>

namespace
{
    // Electron bookkeeping: discard XYZ/Open Babel inferred formal charges and
    // all three MolGR electron classifications before rebuilding from topology
    // and the supplied global budgets.
    void NormalizeSeedElectronicLabels(OpenBabel::OBMol &mol)
    {
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            atom_iter->SetFormalCharge(0);
            molgr::utils::SetUnpairedElectronCount(*atom_iter, 0);
            molgr::utils::SetLonePairCount(*atom_iter, 0);
            molgr::utils::SetUnresolvedTwoElectronCenter(*atom_iter, false);
        }
    }
}

namespace molgr
{
    namespace no_metals
    {
        namespace preparation
        {
            molgr::state::ReconstructionState BuildSeedState(
                std::shared_ptr<OpenBabel::OBMol> omol,
                int total_charge,
                int total_radical_electrons)
            {
                return molgr::state::ReconstructionState(
                    std::move(omol),
                    0,
                    total_charge,
                    total_radical_electrons,
                    {"read_xyz", "normalize_seed_electronic_labels"},
                    {{"source", std::string("xyz_to_omol_no_metal_state")}},
                    0);
            }

            std::shared_ptr<OpenBabel::OBMol> SeedOmolFromXyzBlock(
                const std::string &xyz_block)
            {
                auto omol = std::make_shared<OpenBabel::OBMol>();
                if (!molgr::utils::ReadXyzBlockToMol(xyz_block, omol.get()))
                {
                    return nullptr;
                }
                NormalizeSeedElectronicLabels(*omol);
                return omol;
            }

            std::shared_ptr<OpenBabel::OBMol> NormalizeSeedOmolCopy(
                const OpenBabel::OBMol &seed_omol)
            {
                auto omol = molgr::utils::CloneMolTopologyShared(seed_omol);
                NormalizeSeedElectronicLabels(*omol);
                return omol;
            }

            molgr::state::ReconstructionState SeedStateFromOmol(
                const OpenBabel::OBMol &seed_omol,
                int total_charge,
                int total_radical_electrons)
            {
                return BuildSeedState(
                    NormalizeSeedOmolCopy(seed_omol),
                    total_charge,
                    total_radical_electrons);
            }

            molgr::state::ReconstructionState SeedState(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons)
            {
                auto omol = SeedOmolFromXyzBlock(xyz_block);
                if (!omol)
                {
                    return {};
                }
                return BuildSeedState(
                    std::move(omol),
                    total_charge,
                    total_radical_electrons);
            }

            molgr::state::ReconstructionState PrepareNoMetalSeed(
                const molgr::state::ReconstructionState &state)
            {
                auto machine = molgr::state::OmolStateMachine::FromReconstructionState(state);
                machine.RunOmolStage("make_connections", reconstruct::MakeConnections, 0.15);
                machine.RunOmolStage("pre_clean", reconstruct::PreClean);
                machine.RunOmolStage(
                    "fresh_omol_charge_radical_initial",
                    reconstruct::FreshOmolChargeRadical);

                int formal_charge_sum = 0;
                FOR_ATOMS_OF_MOL(atom_iter, machine.EnsureUniqueMol())
                {
                    formal_charge_sum += atom_iter->GetFormalCharge();
                }
                machine.SetGivenCharge(
                    "initialize_charge_budget",
                    state.total_charge - formal_charge_sum);
                machine.RunOmolChargeStage(
                    "eliminate_NNN_negative",
                    reconstruct::EliminateNNN,
                    false);
                machine.RunOmolChargeStage(
                    "eliminate_high_positive_charge_atoms",
                    reconstruct::EliminateHighPositiveChargeAtoms);
                machine.RunOmolChargeStage(
                    "eliminate_NNN_positive",
                    reconstruct::EliminateNNN,
                    true);
                machine.RunOmolChargeStage(
                    "eliminate_carboxyl",
                    reconstruct::EliminateCarboxyl);
                machine.RunOmolStage(
                    "clean_carbene_neighbor_unsaturated_first",
                    reconstruct::CleanCarbeneNeighborUnsaturated);
                machine.RunOmolChargeStage(
                    "eliminate_carbene_neighbor_heteroatom",
                    reconstruct::EliminateCarbeneNeighborHeteroatom);
                machine.Annotate("prepare_no_metal_seed");
                return machine.FreezeLike(state);
            }
        }
    }
}
