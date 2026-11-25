#include "molgr/utils/no_metals/preparation.h"

#include "molgr/stages/clean.h"
#include "molgr/stages/eliminate.h"
#include "molgr/stages/fresh.h"
#include "molgr/stages/preprocess.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/bond_geometry.h"
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
    /// Clear all inferred seed charges and explicit MolGR electron classifications.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
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

    /// Recognize a closed-shell B(−1)=B(−1) pair with three neighbors per boron.
    /// @param bond Bond whose endpoints and order define the local environment.
    /// @return True for a closed-shell, doubly anionic B=B site.
    bool IsAnionicDiboreneBond(OpenBabel::OBBond &bond)
    {
        if (bond.GetBondOrder() != 2 || bond.IsAromatic()) return false;
        for (auto *atom : {bond.GetBeginAtom(), bond.GetEndAtom()})
        {
            if (atom->GetAtomicNum() != 5 || atom->GetFormalCharge() != -1 ||
                atom->GetTotalValence() != 4 || atom->GetTotalDegree() != 3 ||
                molgr::utils::GetUnpairedElectronCount(*atom) != 0 ||
                molgr::utils::GetLonePairCount(*atom) != 0 ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return false;
        }
        return true;
    }

    /// Require two charge units and two C radicals on the double-bonded end of the C₃ fragment.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param given_charge Signed charge still to assign to the ligand.
    /// @param first_idx One-based atom index in the ligand graph.
    /// @param middle_idx One-based atom index in the ligand graph.
    /// @param other_idx One-based atom index in the ligand graph.
    /// @return True if the charge budget and local C3 assignment allow the reduction.
    bool CanReduceCarbonPair(OpenBabel::OBMol &mol, int given_charge,
                             int first_idx, int middle_idx, int other_idx)
    {
        auto *first = mol.GetAtom(first_idx);
        auto *middle = mol.GetAtom(middle_idx);
        auto *other = mol.GetAtom(other_idx);
        return given_charge <= -2 && first->GetBond(middle)->GetBondOrder() == 2 &&
            middle->GetBond(other)->GetBondOrder() == 1 &&
            other->GetFormalCharge() == 0 && other->GetTotalValence() == 2 &&
            first->GetFormalCharge() == 0 && middle->GetFormalCharge() == 0 &&
            molgr::utils::GetUnpairedElectronCount(*first) == 1 &&
            molgr::utils::GetUnpairedElectronCount(*middle) == 1;
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
                const molgr::state::ReconstructionState &state,
                const std::vector<int> &metal_distances)
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
                    reconstruct::EliminateHighPositiveChargeAtoms, metal_distances);
                machine.RunOmolChargeStage(
                    "eliminate_NNN_positive",
                    reconstruct::EliminateNNN,
                    true);
                machine.RunOmolChargeStage(
                    "eliminate_carboxyl",
                    reconstruct::EliminateCarboxyl, metal_distances);
                machine.RunOmolStage(
                    "clean_carbene_neighbor_unsaturated_first",
                    reconstruct::CleanCarbeneNeighborUnsaturated);
                machine.RunOmolChargeStage(
                    "eliminate_carbene_neighbor_heteroatom",
                    reconstruct::EliminateCarbeneNeighborHeteroatom);
                machine.Annotate("prepare_no_metal_seed");
                return machine.FreezeLike(state);
            }

            std::vector<molgr::state::ReconstructionState> ExpandGeometryElectronicSeeds(
                const molgr::state::ReconstructionState &state)
            {
                // Retain the original and enumerate local electronic alternatives.
                // Charge/radical validation and candidate selection remain shared.
                std::vector<molgr::state::ReconstructionState> pool{state};
                auto &source_mol = const_cast<OpenBabel::OBMol &>(state.Mol());
                // A long oxime N-O / short C-N pair may arrive with radicals
                // on both N and O. Closing N=O first can hide the O-anion
                // branch even though that closure fails final geometry checks.
                // Spend one available negative charge on O before closure,
                // retaining the original seed and leaving C-N/pi redistribution
                // to the ordinary resonance search and validation.
                FOR_ATOMS_OF_MOL(oxygen, source_mol)
                {
                    if (oxygen->GetAtomicNum() != 8 || oxygen->GetExplicitDegree() != 1)
                        continue;
                    OpenBabel::OBAtom *nitrogen = nullptr;
                    FOR_NBORS_OF_ATOM(n, &*oxygen) nitrogen = &*n;
                    if (!nitrogen || nitrogen->GetAtomicNum() != 7 ||
                        nitrogen->GetExplicitDegree() != 2 ||
                        oxygen->GetBond(nitrogen)->GetBondOrder() != 1 ||
                        molgr::bond_geometry::NitrogenOxygenOrderAllowed(*nitrogen, *oxygen, 2))
                        continue;
                    const auto source_count = pool.size();
                    for (std::size_t i = 0; i < source_count && pool.size() < 256; ++i)
                    {
                        const auto source = pool[i];
                        if (source.given_charge >= 0) continue;
                        auto &mol = const_cast<OpenBabel::OBMol &>(source.Mol());
                        bool valid = true;
                        for (const int index : {oxygen->GetIdx(), nitrogen->GetIdx()})
                        {
                            auto *atom = mol.GetAtom(index);
                            valid &= atom->GetFormalCharge() == 0 &&
                                atom->GetImplicitHCount() == 0 &&
                                molgr::utils::GetUnpairedElectronCount(*atom) == 1 &&
                                molgr::utils::GetLonePairCount(*atom) == 0 &&
                                !molgr::utils::HasUnresolvedTwoElectronCenter(*atom);
                        }
                        if (!valid) continue;
                        auto machine = molgr::state::OmolStateMachine::FromReconstructionState(source);
                        machine.RunOmolChargeStage("explore_long_oxime_oxygen_anion",
                            [](OpenBabel::OBMol &target, int &charge, int index)
                            {
                                auto *atom = target.GetAtom(index);
                                atom->SetFormalCharge(-1);
                                molgr::utils::SetUnpairedElectronCount(*atom, 0);
                                ++charge;
                                return true;
                            }, oxygen->GetIdx());
                        pool.push_back(machine.FreezeLike(source));
                    }
                }
                // Close both C-S radical pairs in a conjugated dithiooxamide
                // beside the ordinary C-C closure. Otherwise an early valid
                // solution can consume the charge budget on this ligand and
                // hide the neutral dithione / reduced ancillary-ligand branch.
                FOR_BONDS_OF_MOL(edge, source_mol)
                {
                    auto *left = edge->GetBeginAtom();
                    auto *right = edge->GetEndAtom();
                    if (left->GetAtomicNum() != 6 || right->GetAtomicNum() != 6 ||
                        edge->GetBondOrder() != 1 || left->GetDistance(right) < 1.35 ||
                        left->GetDistance(right) > 1.60 ||
                        !molgr::bond_geometry::HasConjugatedThioamideDonor(left) ||
                        !molgr::bond_geometry::HasConjugatedThioamideDonor(right)) continue;
                    std::array<unsigned int, 4> ids{left->GetIdx(), right->GetIdx(), 0, 0};
                    for (int j = 0; j < 2; ++j)
                    {
                        auto *carbon = source_mol.GetAtom(ids[j]);
                        FOR_NBORS_OF_ATOM(atom, carbon)
                            if (atom->GetAtomicNum() == 16 && atom->GetExplicitDegree() == 1 &&
                                carbon->GetBond(&*atom)->GetBondOrder() == 1 &&
                                carbon->GetDistance(&*atom) > 1.60 &&
                                carbon->GetDistance(&*atom) < 1.82) ids[j + 2] = atom->GetIdx();
                    }
                    if (!ids[2] || !ids[3]) continue;
                    const auto source_count = pool.size();
                    for (std::size_t i = 0; i < source_count && pool.size() < 256; ++i)
                    {
                        const auto source = pool[i];
                        auto &mol = const_cast<OpenBabel::OBMol &>(source.Mol());
                        bool valid = true;
                        for (int j = 0; j < 4; ++j)
                        {
                            auto *atom = mol.GetAtom(ids[j]);
                            valid &= atom->GetFormalCharge() == 0 &&
                                atom->GetTotalValence() == (j < 2 ? 3 : 1) &&
                                atom->GetExplicitDegree() == (j < 2 ? 3 : 1) &&
                                atom->GetImplicitHCount() == 0 &&
                                molgr::utils::GetUnpairedElectronCount(*atom) == 1 &&
                                !molgr::utils::HasUnresolvedTwoElectronCenter(*atom);
                        }
                        if (!valid) continue;
                        auto machine = molgr::state::OmolStateMachine::FromReconstructionState(source);
                        machine.RunOmolStage("explore_neutral_dithiooxamide", [ids](OpenBabel::OBMol &target) {
                            for (int j = 0; j < 2; ++j)
                            {
                                auto *carbon = target.GetAtom(ids[j]);
                                auto *sulfur = target.GetAtom(ids[j + 2]);
                                auto *bond = carbon->GetBond(sulfur);
                                bond->SetBondOrder(2);
                                bond->SetAromatic(false);
                                for (auto *atom : {carbon, sulfur})
                                {
                                    atom->SetAromatic(false);
                                    molgr::utils::SetUnpairedElectronCount(*atom, 0);
                                    molgr::utils::SetLonePairCount(*atom, 0);
                                }
                            }
                            target.SetAromaticPerceived(false);
                            return true;
                        });
                        pool.push_back(machine.FreezeLike(source));
                    }
                }
                for (const auto &ids : molgr::bond_geometry::ShortOrthoQuinodiimineSites(source_mol))
                {
                    const auto source_count = pool.size();
                    for (std::size_t i = 0; i < source_count && pool.size() < 256; ++i)
                    {
                        const auto source = pool[i];
                        auto &mol = const_cast<OpenBabel::OBMol &>(source.Mol());
                        bool valid = true;
                        for (int j = 0; j < 8; ++j)
                        {
                            auto *a = mol.GetAtom(ids[j]);
                            valid &= a->GetFormalCharge() == 0 &&
                                a->GetTotalValence() == (j < 6 ? 4 : 2) &&
                                molgr::utils::GetUnpairedElectronCount(*a) == (j < 6 ? 0 : 1) &&
                                !molgr::utils::HasUnresolvedTwoElectronCenter(*a);
                        }
                        if (!valid) continue;
                        auto machine = molgr::state::OmolStateMachine::FromReconstructionState(source);
                        machine.RunOmolStage("explore_short_quinodiimine", [ids](OpenBabel::OBMol &target) {
                            for (int j = 0; j < 6; ++j)
                            {
                                auto *c = target.GetAtom(ids[j]);
                                auto *bond = c->GetBond(target.GetAtom(ids[(j + 1) % 6]));
                                bond->SetBondOrder(j == 2 || j == 4 ? 2 : 1);
                                bond->SetAromatic(false);
                                c->SetAromatic(false);
                            }
                            for (int j = 0; j < 2; ++j)
                            {
                                auto *n = target.GetAtom(ids[6 + j]);
                                n->GetBond(target.GetAtom(ids[j]))->SetBondOrder(2);
                                molgr::utils::SetUnpairedElectronCount(*n, 0);
                                molgr::utils::SetLonePairCount(*n, 0);
                            }
                            target.SetAromaticPerceived(false);
                            return true;
                        });
                        pool.push_back(machine.FreezeLike(source));
                    }
                }
                FOR_ATOMS_OF_MOL(atom, source_mol)
                {
                    if (atom->GetFormalCharge() != 1 ||
                        !molgr::bond_geometry::AllowsSeesawPhosphoranide(*atom)) continue;
                    const auto source_count = pool.size();
                    for (std::size_t i = 0; i < source_count; ++i)
                    {
                        if (pool.size() >= 256) return pool;
                        const auto source = pool[i];
                        auto machine = molgr::state::OmolStateMachine::FromReconstructionState(source);
                        machine.Annotate("branch_geometry_electronic_seed");
                        machine.RunOmolChargeStage(
                            "explore_seesaw_phosphoranide",
                            [](OpenBabel::OBMol &mol, int &given_charge, int atom_idx)
                            {
                                auto *p = mol.GetAtom(atom_idx);
                                if (p->GetFormalCharge() != 1 ||
                                    !molgr::bond_geometry::AllowsSeesawPhosphoranide(*p)) return false;
                                p->SetFormalCharge(-1);
                                molgr::utils::SetUnpairedElectronCount(*p, 0);
                                molgr::utils::SetLonePairCount(*p, 0);
                                molgr::utils::SetUnresolvedTwoElectronCenter(*p, false);
                                given_charge += 2;
                                return true;
                            },
                            atom->GetIdx());
                        pool.push_back(machine.FreezeLike(source));
                    }
                }
                // XYZ bond perception can supply the reduced [B-]=[B-] state of
                // a diborane(4). Its neutral B-B state has the same sigma skeleton.
                // Return the pi pair to the charge budget before assigning charges
                // to the remaining fragments; retain the original dianion as well.
                FOR_BONDS_OF_MOL(bond, source_mol)
                {
                    if (!IsAnionicDiboreneBond(*bond)) continue;
                    const auto count = pool.size();
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        if (pool.size() >= 256) return pool;
                        const auto source = pool[i];
                        auto machine = molgr::state::OmolStateMachine::FromReconstructionState(source);
                        if (machine.RunOmolChargeStage("explore_neutral_diboron_bond",
                            [](OpenBabel::OBMol &mol, int &charge, int a, int b)
                            {
                                auto *edge = mol.GetBond(a, b);
                                if (!edge || !IsAnionicDiboreneBond(*edge)) return false;
                                edge->SetBondOrder(1);
                                edge->GetBeginAtom()->SetFormalCharge(0);
                                edge->GetEndAtom()->SetFormalCharge(0);
                                charge -= 2;
                                return true;
                            }, bond->GetBeginAtomIdx(), bond->GetEndAtomIdx()))
                            pool.push_back(machine.FreezeLike(source));
                    }
                }
                if (std::none_of(pool.begin(), pool.end(), [](const auto &source) {
                    return source.given_charge <= -2;
                })) return pool;
                for (const auto &atoms : molgr::bond_geometry::EqualizedThreeCarbonChains(source_mol))
                {
                    for (const int end : {0, 2})
                    {
                        const int first_idx = atoms[end]->GetIdx();
                        const int middle_idx = atoms[1]->GetIdx();
                        const int other_idx = atoms[2 - end]->GetIdx();
                        const auto source_count = pool.size();
                        for (std::size_t i = 0; i < source_count; ++i)
                        {
                            if (pool.size() >= 256) return pool;
                            const auto source = pool[i];
                            if (!CanReduceCarbonPair(const_cast<OpenBabel::OBMol &>(source.Mol()),
                                source.given_charge, first_idx, middle_idx, other_idx)) continue;
                            auto machine = molgr::state::OmolStateMachine::FromReconstructionState(source);
                            machine.Annotate("branch_geometry_electronic_seed");
                            machine.RunOmolChargeStage("explore_equalized_carbon_pair",
                                [](OpenBabel::OBMol &mol, int &given_charge, int a, int b, int c)
                                {
                                    if (!CanReduceCarbonPair(mol, given_charge, a, b, c)) return false;
                                    for (const int index : {a, b})
                                    {
                                        auto *atom = mol.GetAtom(index);
                                        atom->SetFormalCharge(-1);
                                        molgr::utils::SetUnpairedElectronCount(*atom, 0);
                                        molgr::utils::SetLonePairCount(*atom, 0);
                                        molgr::utils::SetUnresolvedTwoElectronCenter(*atom, false);
                                    }
                                    given_charge += 2;
                                    return true;
                                }, first_idx, middle_idx, other_idx);
                            pool.push_back(machine.FreezeLike(source));
                        }
                    }
                }
                return pool;
            }
        }
    }
}
