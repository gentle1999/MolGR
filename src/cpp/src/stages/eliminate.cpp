#include "molgr/vendor/openbabel_threading.h"
#include "molgr/stages/eliminate.h"

#include "molgr/stages/internal_helpers.h"

#include "molgr/utils/consts.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/bond_geometry.h"
#include "molgr/utils/logger.h"
#include "molgr/utils/smarts.h"
#include "molgr/utils/utils.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace molgr
{
    namespace reconstruct
    {
        using namespace OpenBabel;

        /// A candidate charge edit and the unpaired electrons it consumes or creates.
        struct ChargeAssignmentAction
        {
            int atom_idx;
            int formal_charge;
            int spin_consumed;
            bool consume_unresolved_center;
            int charge_delta;
            std::vector<int> score_key;
            int bond_idx = -1;
            int charge_atom_idx = -1;
            int spin_created = 0;
        };

        /// All of the tier predicates are single neutral atoms selected by
        /// atomic number and total valence. Atom index precedes match order in ties.
        /// @param element Atomic number of the element.
        /// @param valence Valence assigned to the atom or metal.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @return Preference tier for this anionic assignment, or -1 when it is ineligible.
        int NegativeAssignmentTier(unsigned element, unsigned valence, int charge)
        {
            switch (element) {
            case 9: return valence == 0 ? 10 : -1;
            case 8: return valence <= 1 ? 20 + valence : -1;
            case 17: return valence == 0 ? 30 : -1;
            case 7: return valence <= 2 ? 40 + valence : -1;
            case 35: return valence == 0 ? 50 : -1;
            case 53: return valence == 0 ? 60 : -1;
            case 16: return valence <= 1 ? 70 + valence : -1;
            case 34: return valence <= 1 ? 80 + valence : -1;
            case 15: return valence <= 2 ? 90 + valence : -1;
            case 5: return valence <= 2 ? 95 + valence : -1;
            case 6: return charge < 0 && valence <= 3 ? (valence == 3 ? 100 : 110) : -1;
            case 1: return charge < 0 && valence == 0 ? 120 : -1;
            default: return -1;
            }
        }

        // Electron bookkeeping: choose how many real unpaired electrons to bundle
        // into one multivalent anion action; never count lone-pair/marker state.
        /// Choose how many real unpaired electrons a multivalent anion action consumes.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @return Number of available unpaired electrons that can absorb the negative charge.
        int NegativeChargeAssignmentAmount(OBAtom *atom, int charge)
        {
            if (atom == nullptr)
            {
                return 0;
            }
            return std::min(molgr::utils::GetUnpairedElectronCount(*atom), std::max(1, std::abs(charge)));
        }

        /// @param atom Atom whose local chemical environment is examined.
        /// @return True for neutral, fully resolved boron with valence three and no unpaired electrons.
        bool IsClosedShellTrivalentBoron(OBAtom &atom)
        {
            return atom.GetAtomicNum() == 5 && atom.GetTotalValence() == 3 &&
                   atom.GetFormalCharge() == 0 &&
                   molgr::utils::GetUnpairedElectronCount(atom) == 0 &&
                   molgr::utils::GetLonePairCount(atom) == 0 &&
                   !molgr::utils::HasUnresolvedTwoElectronCenter(atom);
        }

        // Electron bookkeeping: radical actions consume real unpaired electrons.
        // An unresolved action atomically consumes one pure deferred 2e marker as
        // +/-2 without first creating a diradical. Active lone pairs are excluded.
        // A trivalent-boron reduction instead creates one unpaired electron.
        /// Encode one explicitly identified electron source as formal charge.
        /// @param mol Molecular graph on which the operation acts.
        /// @param action Proposed local charge and electron edit.
        /// @return True when the requested edit was applied to the graph.
        bool ApplyChargeAssignmentAction(OBMol &mol, const ChargeAssignmentAction &action)
        {
            OBAtom *atom = mol.GetAtom(action.atom_idx);
            if (atom == nullptr)
            {
                return false;
            }
            if (action.bond_idx >= 0 && action.charge_atom_idx >= 0)
            {
                OBAtom *charge_atom = mol.GetAtom(action.charge_atom_idx);
                OBBond *bond = mol.GetBond(atom, charge_atom);
                if (charge_atom == nullptr || bond == nullptr ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom) ||
                    molgr::utils::GetUnpairedElectronCount(*atom) < action.spin_consumed)
                {
                    return false;
                }
                molgr::utils::SetUnpairedElectronCount(
                    *atom,
                    molgr::utils::GetUnpairedElectronCount(*atom) - action.spin_consumed);
                bond->SetBondOrder(bond->GetBondOrder() + 1);
                charge_atom->SetFormalCharge(charge_atom->GetFormalCharge() + 1);
                return true;
            }
            if (atom->GetFormalCharge() != 0)
            {
                return false;
            }
            if (action.spin_created != 0)
            {
                if (action.spin_created != 1 || action.spin_consumed != 0 ||
                    action.consume_unresolved_center || action.formal_charge != -1 ||
                    !IsClosedShellTrivalentBoron(*atom))
                {
                    return false;
                }
                molgr::utils::SetUnpairedElectronCount(*atom, 1);
            }
            else if (action.consume_unresolved_center)
            {
                if (action.spin_consumed != 0 ||
                    std::abs(action.formal_charge) != 2 ||
                    !molgr::utils::HasUnresolvedTwoElectronCenter(*atom) ||
                    molgr::utils::GetUnpairedElectronCount(*atom) != 0 ||
                    molgr::utils::GetLonePairCount(*atom) != 0)
                {
                    return false;
                }
                molgr::utils::SetUnresolvedTwoElectronCenter(*atom, false);
            }
            else
            {
                if (action.spin_consumed <= 0 ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom) ||
                    molgr::utils::GetUnpairedElectronCount(*atom) < action.spin_consumed)
                {
                    return false;
                }
                molgr::utils::SetUnpairedElectronCount(
                    *atom,
                    molgr::utils::GetUnpairedElectronCount(*atom) - action.spin_consumed);
            }
            atom->SetFormalCharge(action.formal_charge);
            return true;
        }

        /// Append a ranked charge placement unless its atom and charge change have already been considered.
        /// @param actions Destination for the ranked charge-placement actions.
        /// @param seen Previously considered assignments used for deduplication.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @param tier Chemical preference tier used when ranking charge placements.
        /// @param match_order Original match order used to resolve ties.
        /// @param amount Amount to add to the counter or assignment.
        void AppendPositiveChargeAssignmentAction(
            std::vector<ChargeAssignmentAction> &actions,
            std::set<std::tuple<int, int, int>> &seen,
            OBAtom *atom,
            int charge,
            int tier,
            int match_order,
            int amount)
        {
            if (atom == nullptr || amount <= 0 ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
            {
                return;
            }
            const int atom_idx = atom->GetIdx();
            const auto seen_key = std::make_tuple(atom_idx, tier, amount);
            if (seen.count(seen_key) != 0)
            {
                return;
            }
            seen.insert(seen_key);

            const int charge_after = charge - amount;
            const int atomic_num = atom->GetAtomicNum();
            actions.push_back(ChargeAssignmentAction{
                atom_idx,
                amount,
                amount,
                false,
                -amount,
                {
                    tier,
                    std::abs(charge_after),
                    std::max(charge_after, 0),
                    atomic_num,
                    atom_idx,
                    match_order,
                }});
        }

        /// Append a ranked charge placement unless its atom and charge change have already been considered.
        /// @param actions Destination for the ranked charge-placement actions.
        /// @param seen Previously considered assignments used for deduplication.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @param tier Chemical preference tier used when ranking charge placements.
        /// @param match_order Original match order used to resolve ties.
        /// @param amount Amount to add to the counter or assignment.
        void AppendNegativeChargeAssignmentAction(
            std::vector<ChargeAssignmentAction> &actions,
            std::set<std::tuple<int, int, int>> &seen,
            OBAtom *atom,
            int charge,
            int tier,
            int match_order,
            int amount)
        {
            if (atom == nullptr || amount <= 0 ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
            {
                return;
            }
            const int atom_idx = atom->GetIdx();
            const auto seen_key = std::make_tuple(atom_idx, tier, amount);
            if (seen.count(seen_key) != 0)
            {
                return;
            }
            seen.insert(seen_key);

            const int charge_after = charge + amount;
            const int atomic_num = atom->GetAtomicNum();
            actions.push_back(ChargeAssignmentAction{
                atom_idx,
                -amount,
                amount,
                false,
                amount,
                {
                    tier,
                    std::abs(charge_after),
                    std::max(charge_after, 0),
                    atomic_num,
                    atom_idx,
                    match_order,
                }});
        }

        /// Consume one pure unresolved center directly as a +/-2 formal-charge
        /// action. A deficit smaller than two is rejected by the caller.
        /// @param actions Destination for the ranked charge-placement actions.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @param formal_charge Signed formal charge used for this electronic assignment.
        /// @param tier Chemical preference tier used when ranking charge placements.
        /// @param match_order Original match order used to resolve ties.
        void AppendUnresolvedChargeAssignmentAction(
            std::vector<ChargeAssignmentAction> &actions,
            OBAtom *atom,
            int charge,
            int formal_charge,
            int tier,
            int match_order)
        {
            if (atom == nullptr ||
                std::abs(formal_charge) != 2 ||
                atom->GetFormalCharge() != 0 ||
                !molgr::utils::HasUnresolvedTwoElectronCenter(*atom) ||
                molgr::utils::GetUnpairedElectronCount(*atom) != 0 ||
                molgr::utils::GetLonePairCount(*atom) != 0)
            {
                return;
            }
            const int charge_after = charge - formal_charge;
            const int atom_idx = static_cast<int>(atom->GetIdx());
            const int atomic_num = static_cast<int>(atom->GetAtomicNum());
            actions.push_back(ChargeAssignmentAction{
                atom_idx,
                formal_charge,
                0,
                true,
                -formal_charge,
                {
                    tier,
                    std::abs(charge_after),
                    std::max(charge_after, 0),
                    atomic_num,
                    atom_idx,
                    match_order,
                }});
        }

        // Electron bookkeeping: rank cation actions from real radicals or, with a
        // remaining deficit of at least two, one pure unresolved 2e center.
        /// Rank cation actions from radicals or a pure unresolved 2e center.
        /// @param mol Molecular graph on which the operation acts.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @return Admissible charge edits ordered by chemical preference and tie-break criteria.
        std::vector<ChargeAssignmentAction> PositiveChargeAssignmentActions(
            OBMol &mol,
            int charge,
            int total_radical_electrons = 0)
        {
            std::vector<ChargeAssignmentAction> actions;
            std::set<std::tuple<int, int, int>> seen;
            int real_radicals = 0;
            int unresolved_electrons = 0;
            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                OBAtom *atom = &(*atom_iter);
                real_radicals += molgr::utils::GetUnpairedElectronCount(*atom);
                if (molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                {
                    unresolved_electrons += 1;
                }
            }
            const bool tier5_allowed =
                std::abs(charge - 1) <= real_radicals + unresolved_electrons * 2 - 1;
            molgr::vendor::openbabel_threading::PrepareForSmartsMatching(mol);
            if (charge > 0 && real_radicals > 0)
            {
                auto n_matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::ELIM_POSITIVE_N);
                for (std::size_t match_order = 0; match_order < n_matches.size(); ++match_order)
                {
                    const auto &idxs = n_matches[match_order];
                    if (idxs.size() < 2)
                    {
                        continue;
                    }
                    OBAtom *atom = mol.GetAtom(idxs[1]);
                    if (charge > 0 && atom != nullptr && atom->GetFormalCharge() == 0 && molgr::utils::GetUnpairedElectronCount(*atom) >= 1)
                    {
                        AppendPositiveChargeAssignmentAction(
                            actions,
                            seen,
                            atom,
                            charge,
                            0,
                            static_cast<int>(match_order),
                            1);
                    }
                }

                auto c_h_matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::ELIM_POSITIVE_C_H);
                for (std::size_t match_order = 0; match_order < c_h_matches.size(); ++match_order)
                {
                    const auto &idxs = c_h_matches[match_order];
                    if (idxs.empty())
                    {
                        continue;
                    }
                    OBAtom *atom = mol.GetAtom(idxs[0]);
                    if (charge > 0 && atom != nullptr && atom->GetFormalCharge() == 0 && molgr::utils::GetUnpairedElectronCount(*atom) >= 1)
                    {
                        AppendPositiveChargeAssignmentAction(
                            actions,
                            seen,
                            atom,
                            charge,
                            10,
                            static_cast<int>(match_order),
                            1);
                    }
                }
            }

            if (tier5_allowed && real_radicals > 0)
            {
                auto dipole_matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::ELIM_POSITIVE_DIPOLE);
                for (std::size_t match_order = 0; match_order < dipole_matches.size(); ++match_order)
                {
                    const auto &idxs = dipole_matches[match_order];
                    if (idxs.size() < 2)
                    {
                        continue;
                    }
                    OBAtom *radical_atom = mol.GetAtom(idxs[0]);
                    OBAtom *charge_atom = mol.GetAtom(idxs[1]);
                    OBBond *bond = mol.GetBond(radical_atom, charge_atom);
                    if (tier5_allowed && radical_atom != nullptr && charge_atom != nullptr && bond != nullptr &&
                        radical_atom->GetFormalCharge() == 0 &&
                        molgr::utils::GetUnpairedElectronCount(*radical_atom) == 1 &&
                        !molgr::utils::HasUnresolvedTwoElectronCenter(*radical_atom))
                    {
                        const int charge_after = charge - 1;
                        actions.push_back(ChargeAssignmentAction{
                            static_cast<int>(radical_atom->GetIdx()),
                            0,
                            1,
                            false,
                            -1,
                            {molgr::bond_geometry::RestoresAziniumRing(*charge_atom, *radical_atom) ? 4 : 5, std::abs(charge_after), std::max(charge_after, 0), static_cast<int>(radical_atom->GetIdx()), static_cast<int>(match_order)},
                            static_cast<int>(bond->GetIdx()),
                            static_cast<int>(charge_atom->GetIdx()),
                        });
                    }
                }
            }

            int match_order = 0;
            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                OBAtom *atom = &(*atom_iter);
                if (charge > 0 && atom->GetFormalCharge() == 0 && molgr::utils::GetUnpairedElectronCount(*atom) >= 1)
                {
                    AppendPositiveChargeAssignmentAction(
                        actions,
                        seen,
                        atom,
                        charge,
                        100,
                        match_order,
                        std::min(molgr::utils::GetUnpairedElectronCount(*atom), charge));
                }
                if (charge >= 2)
                {
                    AppendUnresolvedChargeAssignmentAction(
                        actions, atom, charge, 2, 100, match_order);
                }
                ++match_order;
            }

            std::sort(
                actions.begin(),
                actions.end(),
                [](const auto &left, const auto &right)
                {
                    return left.score_key < right.score_key;
                });
            return actions;
        }

        // Electron bookkeeping: rank anion actions from real radicals or, with a
        // remaining deficit of at least two, one pure unresolved 2e center.
        /// Rank anion actions from radicals or a pure unresolved 2e center.
        /// @param mol Molecular graph on which the operation acts.
        /// @param charge Signed formal charge used for this electronic assignment.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @param metal_distances Nearest eligible metal-contact distances indexed by ligand atom, in
        ///     milliangstroms.
        /// @return Admissible charge edits ordered by chemical preference and tie-break criteria.
        std::vector<ChargeAssignmentAction> NegativeChargeAssignmentActions(
            OBMol &mol,
            int charge,
            int total_radical_electrons,
            const std::vector<int> &metal_distances)
        {
            std::vector<ChargeAssignmentAction> actions;
            if (charge > 0)
            {
                return actions;
            }

            std::set<std::tuple<int, int, int>> seen;

            molgr::vendor::openbabel_threading::PrepareForSmartsMatching(mol);
            FOR_ATOMS_OF_MOL(it, mol)
            {
                if (it->GetFormalCharge() != 0 || molgr::utils::GetUnpairedElectronCount(*it) < 1) continue;
                const int tier = NegativeAssignmentTier(it->GetAtomicNum(), it->GetTotalValence(), charge);
                if (tier >= 0)
                    AppendNegativeChargeAssignmentAction(actions,seen,&*it,charge,tier,it->GetIdx()-1,
                        NegativeChargeAssignmentAmount(&*it, charge));
            }

            if (charge < 0)
            {
                int real_unpaired = 0;
                if (total_radical_electrons >= 0)
                {
                    FOR_ATOMS_OF_MOL(atom_iter, mol)
                    {
                        real_unpaired += molgr::utils::GetUnpairedElectronCount(*atom_iter);
                    }
                }
                const bool can_create_radical =
                    total_radical_electrons >= 0 && real_unpaired < total_radical_electrons;
                int match_order = 0;
                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    OBAtom *atom = &(*atom_iter);
                    if (can_create_radical && IsClosedShellTrivalentBoron(*atom))
                    {
                        const int atom_idx = atom->GetIdx();
                        const int charge_after = charge + 1;
                        ChargeAssignmentAction action{
                            atom_idx, -1, 0, false, 1,
                            {98, std::abs(charge_after), std::max(charge_after, 0),
                             5, atom_idx, match_order}};
                        action.spin_created = 1;
                        actions.push_back(std::move(action));
                    }
                    if (atom->GetFormalCharge() == 0 && molgr::utils::GetUnpairedElectronCount(*atom) >= 1)
                    {
                        AppendNegativeChargeAssignmentAction(
                            actions,
                            seen,
                            atom,
                            charge,
                            1000,
                            match_order,
                            NegativeChargeAssignmentAmount(atom, charge));
                    }
                    if (charge <= -2)
                    {
                        AppendUnresolvedChargeAssignmentAction(
                            actions, atom, charge, -2, 1000, match_order);
                    }
                    ++match_order;
                }
            }

            if (!metal_distances.empty() && actions.size() > 1)
            {
                // The last two fields are atom index and SMARTS match order.
                for (auto &action : actions)
                    action.score_key.insert(action.score_key.end() - 2, metal_distances.at(action.atom_idx));
            }
            std::sort(
                actions.begin(),
                actions.end(),
                [](const auto &left, const auto &right)
                {
                    return left.score_key < right.score_key;
                });
            return actions;
        }

        // Electron bookkeeping: the negative N-N-N closure consumes one classified
        // two-electron state at each terminal N and one real unpaired electron at
        // the middle N. The positive closure consumes one middle-N monoradical.
        // All electronic preconditions are checked before bonds or charges change.
        bool EliminateNNN(OBMol &mol, int &charge, bool positive)
        {
            /// Accept exactly one pure unresolved, triplet, or active-singlet
            /// two-electron state. Mixed marker/occupancy states are invalid here.
            const auto has_consumable_two_electron_center = [](const OBAtom &atom)
            {
                const bool unresolved =
                    molgr::utils::HasUnresolvedTwoElectronCenter(atom);
                const int unpaired = molgr::utils::GetUnpairedElectronCount(atom);
                const int lone_pairs = molgr::utils::GetLonePairCount(atom);
                return (unresolved && unpaired == 0 && lone_pairs == 0) ||
                       (!unresolved && unpaired == 2 && lone_pairs == 0) ||
                       (!unresolved && unpaired == 0 && lone_pairs >= 1);
            };
            /// Consume the prevalidated state from its own field; do not infer a
            /// lone pair from unpaired-electron parity or vice versa.
            const auto consume_two_electron_center = [](OBAtom &atom)
            {
                if (molgr::utils::HasUnresolvedTwoElectronCenter(atom))
                {
                    molgr::utils::SetUnresolvedTwoElectronCenter(atom, false);
                }
                else if (molgr::utils::GetUnpairedElectronCount(atom) == 2)
                {
                    molgr::utils::SetUnpairedElectronCount(atom, 0);
                }
                else if (molgr::utils::GetLonePairCount(atom) >= 1)
                {
                    molgr::utils::SetLonePairCount(
                        atom,
                        molgr::utils::GetLonePairCount(atom) - 1);
                }
            };
            bool hit = false;
            if (!positive)
            {
                while (true)
                {
                    auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::ELIM_NNN_NEGATIVE);
                    if (matches.empty())
                    {
                        break;
                    }

                    const auto idxs = matches[0];
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBAtom *a1 = mol.GetAtom(idxs[0]);
                    OBAtom *a2 = mol.GetAtom(idxs[1]);
                    OBAtom *a3 = mol.GetAtom(idxs[2]);
                    if (!bond1 || !bond2 || !a1 || !a2 || !a3)
                    {
                        break;
                    }
                    if (!has_consumable_two_electron_center(*a1) ||
                        molgr::utils::GetUnpairedElectronCount(*a2) != 1 ||
                        !has_consumable_two_electron_center(*a3))
                    {
                        break;
                    }

                    bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                    bond2->SetBondOrder(bond2->GetBondOrder() + 1);

                    consume_two_electron_center(*a1);
                    a1->SetFormalCharge(a1->GetFormalCharge() - 1);

                    molgr::utils::SetUnpairedElectronCount(*a2, molgr::utils::GetUnpairedElectronCount(*a2) - 1);
                    a2->SetFormalCharge(a2->GetFormalCharge() + 1);

                    consume_two_electron_center(*a3);
                    a3->SetFormalCharge(a3->GetFormalCharge() - 1);

                    charge += 1;
                    hit = true;
                    LOG_DEBUG("[EliminateNNN] Applied negative branch");
                }
                return hit;
            }

            // Preserve the middle-N radical for anionic localization when
            // the remaining charge budget is negative.
            while (charge >= 0)
            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::ELIM_NNN_POSITIVE);
                if (matches.empty())
                {
                    break;
                }

                const auto idxs = matches[0];
                OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                OBAtom *a1 = mol.GetAtom(idxs[0]);
                OBAtom *a2 = mol.GetAtom(idxs[1]);
                    if (!bond1 || !a1 || !a2)
                    {
                        break;
                    }
                    if (molgr::utils::GetUnpairedElectronCount(*a2) != 1)
                    {
                        break;
                    }

                bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                a1->SetFormalCharge(a1->GetFormalCharge() + 1);
                molgr::utils::SetUnpairedElectronCount(*a2, molgr::utils::GetUnpairedElectronCount(*a2) - 1);
                charge -= 1;
                hit = true;
                LOG_DEBUG("[EliminateNNN] Applied positive branch");
            }
            return hit;
        }

        // Neutralize each positive center from eligible neighboring
        // monoradicals in descending electronegativity order. Stop once the
        // center charge is balanced by neighboring negative formal charges.
        bool EliminateHighPositiveChargeAtoms(OBMol &mol, int &charge, const std::vector<int> &metal_distances)
        {
            bool hit = false;
            const auto matches = molgr::smarts::FindAll(
                mol,
                molgr::smarts::PatternId::ELIM_HIGH_POSITIVE);
            std::map<int, std::set<int>> neighbors_by_center;
            for (const auto &idxs : matches)
            {
                if (idxs.size() == 2)
                {
                    neighbors_by_center[idxs[0]].insert(idxs[1]);
                }
            }

            for (const auto &[center_idx, matched_neighbor_indices] : neighbors_by_center)
            {
                OBAtom *center = mol.GetAtom(center_idx);
                if (center == nullptr)
                {
                    continue;
                }
                int remaining_positive_charge = center->GetFormalCharge();
                FOR_NB_OF_ATOM(nbr, center)
                {
                    remaining_positive_charge += std::min(0, nbr->GetFormalCharge());
                }
                if (remaining_positive_charge <= 0)
                {
                    continue;
                }

                std::vector<int> neighbor_indices(
                    matched_neighbor_indices.begin(),
                    matched_neighbor_indices.end());
                std::sort(
                    neighbor_indices.begin(),
                    neighbor_indices.end(),
                    [&mol, &metal_distances](int left_idx, int right_idx)
                    {
                        const OBAtom *left = mol.GetAtom(left_idx);
                        const OBAtom *right = mol.GetAtom(right_idx);
                        const double left_electronegativity =
                            OBElements::GetElectroNeg(left->GetAtomicNum());
                        const double right_electronegativity =
                            OBElements::GetElectroNeg(right->GetAtomicNum());
                        if (left_electronegativity != right_electronegativity)
                        {
                            return left_electronegativity > right_electronegativity;
                        }
                        if (!metal_distances.empty() && metal_distances.at(left_idx) != metal_distances.at(right_idx))
                            return metal_distances[left_idx] < metal_distances[right_idx];
                        return left_idx < right_idx;
                    });

                for (int neighbor_idx : neighbor_indices)
                {
                    if (remaining_positive_charge <= 0)
                    {
                        break;
                    }
                    OBAtom *neighbor = mol.GetAtom(neighbor_idx);
                    if (neighbor == nullptr)
                    {
                        continue;
                    }

                    bool neighbor_has_pending_electrons = false;
                    FOR_NB_OF_ATOM(adjacent, neighbor)
                    {
                        if (adjacent->GetIdx() == center_idx)
                        {
                            continue;
                        }
                        if (molgr::utils::GetUnpairedElectronCount(*adjacent) > 0 ||
                            molgr::utils::HasUnresolvedTwoElectronCenter(*adjacent))
                        {
                            neighbor_has_pending_electrons = true;
                            break;
                        }
                    }

                    if (neighbor->GetFormalCharge() != 0 ||
                        molgr::utils::GetUnpairedElectronCount(*neighbor) != 1 ||
                        molgr::utils::HasUnresolvedTwoElectronCenter(*neighbor) ||
                        neighbor_has_pending_electrons)
                    {
                        continue;
                    }

                    molgr::utils::SetUnpairedElectronCount(*neighbor, 0);
                    neighbor->SetFormalCharge(-1);
                    ++charge;
                    --remaining_positive_charge;
                    hit = true;
                    LOG_DEBUG("[EliminateHighPos] Applied");
                }
            }
            return hit;
        }

        /// Check whether initial bond perception could equally put C=O on either oxygen.
        /// @param carbon Atom or site occupying this position in the local chemical motif.
        /// @param radical Unpaired-electron assignment at the site.
        /// @param carbonyl Oxygen currently assigned the carbonyl bond.
        /// @return True when either oxygen can carry the carbonyl bond under the local geometry screen.
        static bool InterchangeableCarboxylOxygens(OBAtom &carbon, OBAtom &radical, OBAtom &carbonyl)
        {
            if (carbon.GetExplicitDegree() != 3 || carbon.GetFormalCharge() != 0 ||
                radical.GetExplicitDegree() != 1 || carbonyl.GetExplicitDegree() != 1 ||
                radical.GetImplicitHCount() || carbonyl.GetImplicitHCount() ||
                carbonyl.GetFormalCharge() != 0 || molgr::utils::GetUnpairedElectronCount(carbonyl) ||
                molgr::utils::GetUnpairedElectronCount(carbon))
                return false;
            for (auto *atom : {&radical, &carbon, &carbonyl})
                if (molgr::utils::GetLonePairCount(*atom) || molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                    return false;
            const double first = carbon.GetDistance(&radical);
            const double second = carbon.GetDistance(&carbonyl);
            if (!(1.20 <= first && first <= 1.34 && 1.20 <= second && second <= 1.34 &&
                  std::abs(first - second) <= 0.03)) return false;
            FOR_NB_OF_ATOM(neighbor, &carbon)
                if (neighbor->GetAtomicNum() == 1 || neighbor->GetAtomicNum() == 6) return true;
            return false;
        }

        // Electron bookkeeping: localize exactly one oxygen monoradical as O-.
        // A missing or multi-electron radical label rejects the transformation.
        bool EliminateCarboxyl(OBMol &mol, int &charge, const std::vector<int> &metal_distances)
        {
            bool hit = false;
            while (true)
            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::ELIM_CARBOXYL);
                if (matches.empty())
                    break;
                OBAtom *a1 = mol.GetAtom(matches[0][0]);
                if (a1 == nullptr ||
                    molgr::utils::GetUnpairedElectronCount(*a1) != 1)
                {
                    break;
                }
                auto *target = a1;
                auto *carbon = mol.GetAtom(matches[0][1]);
                auto *other = mol.GetAtom(matches[0][2]);
                if (!metal_distances.empty() &&
                    metal_distances.at(other->GetIdx()) < metal_distances.at(a1->GetIdx()) &&
                    InterchangeableCarboxylOxygens(*carbon, *a1, *other))
                {
                    carbon->GetBond(a1)->SetBondOrder(2);
                    carbon->GetBond(other)->SetBondOrder(1);
                    target = other;
                }
                molgr::utils::SetUnpairedElectronCount(*a1, 0);
                target->SetFormalCharge(-1);
                charge += 1;
                hit = true;
                LOG_DEBUG("[EliminateCarboxyl] Applied");
            }
            return hit;
        }

        // Electron bookkeeping: close an unresolved two-electron center with a
        // closed-shell heteroatom donor. The center marker becomes center-/donor+;
        // consume one explicitly tracked active donor lone pair when present.
        // Real unpaired electrons cannot act as the donor in this rule.
        bool EliminateCarbeneNeighborHeteroatom(OBMol &mol, int &charge)
        {
            /// Require a deferred two-electron center before applying the heteroatom-donation rule.
            const auto possible_carbene_atom = [](const OBAtom &atom) -> bool
            {
                return molgr::utils::HasUnresolvedTwoElectronCenter(atom);
            };

            bool hit = false;
            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                OBAtom *atom = &(*atom_iter);
                if (possible_carbene_atom(*atom))
                {
                    bool neighbor_radical = false;
                    FOR_NB_OF_ATOM(nbr, atom)
                    {
                        if (molgr::utils::GetUnpairedElectronCount(*nbr) == 1)
                        {
                            neighbor_radical = true;
                            break;
                        }
                    }
                    if (neighbor_radical)
                    {
                        continue;
                    }

                    FOR_NB_OF_ATOM(nbr, atom)
                    {
                        if (kHeteroatoms.count(nbr->GetAtomicNum()) &&
                            nbr->GetFormalCharge() == 0 &&
                            molgr::utils::GetUnpairedElectronCount(*nbr) == 0 &&
                            !molgr::utils::HasUnresolvedTwoElectronCenter(*nbr) &&
                            // Conventional donor pairs must exist even when
                            // the reconstruction-active pair count is zero.
                            kNonMetalDict.at(nbr->GetAtomicNum()).num_outer_electrons -
                                    nbr->GetFormalCharge() - static_cast<int>(nbr->GetTotalValence()) >= 2)
                        {
                            OBBond *bond = mol.GetBond(atom, &*nbr);
                            bond->SetBondOrder(bond->GetBondOrder() + 1);
                            molgr::utils::SetLonePairCount(*atom, 0);
                            molgr::utils::SetUnpairedElectronCount(*atom, 0);
                            molgr::utils::SetUnresolvedTwoElectronCenter(*atom, false);
                            atom->SetFormalCharge(atom->GetFormalCharge() - 1);
                            if (molgr::utils::GetLonePairCount(*nbr) > 0)
                            {
                                molgr::utils::SetLonePairCount(
                                    *nbr,
                                    molgr::utils::GetLonePairCount(*nbr) - 1);
                            }
                            nbr->SetFormalCharge(nbr->GetFormalCharge() + 1);
                            hit = true;
                            LOG_DEBUG("[EliminateCarbeneHetero] Applied");
                            break;
                        }
                    }
                }
            }
            return hit;
        }

        // Electron bookkeeping: consume exactly one terminal unpaired electron
        // to raise the adjacent bond order and shift one formal-charge unit.
        bool Eliminate13DipolePostive(OBMol &mol, int &charge)
        {
            bool hit = false;
            auto matches = molgr::smarts::FindAll(
                mol,
                molgr::smarts::PatternId::ELIM_1_3_DIPOLE_POSTIVE);
            while (charge >= 0 && !matches.empty())
            {
                auto idxs = matches.front();
                matches.erase(matches.begin());

                OBAtom *atom2 = mol.GetAtom(idxs[1]);
                OBAtom *atom3 = mol.GetAtom(idxs[2]);
                OBBond *bond23 = mol.GetBond(idxs[1], idxs[2]);
                if (!atom2 || !atom3 || !bond23)
                {
                    continue;
                }

                const ElementInfo *info = GetElementInfo(atom2->GetAtomicNum());
                if (info != nullptr &&
                    molgr::utils::GetUnpairedElectronCount(*atom3) == 1 &&
                    (info->num_outer_electrons + atom2->GetTotalValence()) == 8)
                {
                    atom2->SetFormalCharge(atom2->GetFormalCharge() + 1);
                    bond23->SetBondOrder(static_cast<int>(bond23->GetBondOrder() + 1));
                    molgr::utils::SetUnpairedElectronCount(*atom3, molgr::utils::GetUnpairedElectronCount(*atom3) - 1);
                    charge -= 1;
                    hit = true;
                }
            }
            return hit;
        }

        // Electron bookkeeping: consume a CP-like monoradical only when it is
        // not reserved for the target radical count or pending charge budget.
        bool EliminatePossibleCPLikeRadicalAnion(
            OBMol &mol,
            int &charge,
            int total_radical_electrons, const std::vector<int> &metal_distances)
        {
            /// Count explicit unpaired electrons available after reserving the target spin and charge.
            const auto available_unpaired_electrons = [&]()
            {
                int count = 0;
                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    count += molgr::utils::GetUnpairedElectronCount(*atom_iter);
                }
                return count - total_radical_electrons - std::abs(charge);
            };

            bool hit = false;
            while (true)
            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::ELIM_NEGATIVE_CP);
                if (matches.empty() || available_unpaired_electrons() < 1)
                {
                    break;
                }

                const auto &idxs = matches.front();
                if (idxs.size() < 5)
                {
                    break;
                }
                OBAtom *atom = mol.GetAtom(idxs[4]);
                if (atom == nullptr || atom->GetFormalCharge() != 0)
                {
                    break;
                }

                if (molgr::utils::GetUnpairedElectronCount(*atom) != 1)
                {
                    break;
                }
                if (!metal_distances.empty())
                {
                    for (const auto &match : matches)
                    {
                        auto *other = mol.GetAtom(match[4]);
                        if (metal_distances.at(other->GetIdx()) < metal_distances.at(atom->GetIdx()) &&
                            other->GetFormalCharge() == 0 && molgr::utils::GetUnpairedElectronCount(*other) == 1)
                            atom = other;
                    }
                }
                molgr::utils::SetUnpairedElectronCount(*atom, 0);
                atom->SetFormalCharge(-1);
                charge += 1;
                hit = true;
            }
            return hit;
        }

        /// Skip N--N/N--O donations already excluded by final geometry validation.
        /// @param mol Molecular graph on which the operation acts.
        /// @param action Proposed local charge and electron edit.
        /// @return True when the proposed charge placement respects local nitrogen geometry.
        bool PositiveChargeGeometryAllowed(OBMol &mol, const ChargeAssignmentAction &action)
        {
            if (action.bond_idx < 0 || action.charge_atom_idx < 0)
                return true;
            const auto first = mol.GetAtom(action.atom_idx)->GetAtomicNum();
            const auto second = mol.GetAtom(action.charge_atom_idx)->GetAtomicNum();
            if (!((first == 7 && (second == 7 || second == 8)) ||
                  (first == 8 && second == 7) ||
                  (first == 15 && second == 6) ||
                  (first == 6 && second == 15)))
                return true;
            // Evaluate the same resolved endpoint as final validation without
            // consuming charge or radicals when this ranked action is rejected.
            OBMol trial(mol);
            if (!ApplyChargeAssignmentAction(trial, action))
                return false;
            auto *bond = trial.GetBond(action.atom_idx, action.charge_atom_idx);
            return molgr::bond_geometry::NitrogenNitrogenOrderAllowed(
                       *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) &&
                   molgr::bond_geometry::NitrogenOxygenOrderAllowed(
                       *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) &&
                   molgr::bond_geometry::PhosphorusCarbonOrderAllowed(
                       *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder());
        }

        // Electron bookkeeping: encode real radical actions or a pure unresolved
        // 2e center as positive charge. The latter requires at least two remaining
        // charge units, becomes +2, and clears its marker atomically.
        bool EliminatePositiveChargesWithTarget(OBMol &mol, int &charge, int total_radical_electrons)
        {
            bool hit = false;
            while (true)
            {
                const auto actions = PositiveChargeAssignmentActions(mol, charge, total_radical_electrons);
                const auto allowed = std::find_if(actions.begin(), actions.end(),
                    [&](const ChargeAssignmentAction &action)
                    { return PositiveChargeGeometryAllowed(mol, action); });
                if (allowed == actions.end())
                {
                    break;
                }
                const auto &action = *allowed;
                if (!ApplyChargeAssignmentAction(mol, action))
                {
                    break;
                }
                charge += action.charge_delta;
                hit = true;
            }
            return hit;
        }

        bool EliminatePositiveCharges(OBMol &mol, int &charge)
        {
            return EliminatePositiveChargesWithTarget(mol, charge, 0);
        }

        // Electron bookkeeping: encode real radical actions or a pure unresolved
        // 2e center as negative charge. The latter requires at least two remaining
        // charge units and becomes -2; zero-budget separation remains radical-only.
        // Neutral Bv3 can instead create a radical when both budgets allow it.
        bool EliminateNegativeCharges(OBMol &mol, int &charge, int total_radical_electrons, const std::vector<int> &metal_distances)
        {
            bool hit = false;
            while (charge <= 0)
            {
                if (total_radical_electrons >= 0 && charge == 0)
                {
                    int real_unpaired = 0;
                    FOR_ATOMS_OF_MOL(atom_iter, mol)
                    {
                        real_unpaired += molgr::utils::GetUnpairedElectronCount(*atom_iter);
                    }
                    if (real_unpaired <= total_radical_electrons)
                    {
                        break;
                    }
                }

                const auto actions = NegativeChargeAssignmentActions(mol, charge, total_radical_electrons, metal_distances);
                if (actions.empty())
                {
                    break;
                }
                const auto &action = actions.front();
                if (!ApplyChargeAssignmentAction(mol, action))
                {
                    break;
                }
                charge += action.charge_delta;
                hit = true;
            }
            return hit;
        }
    }
}
