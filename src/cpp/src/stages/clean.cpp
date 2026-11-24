#include "molgr/stages/clean.h"

#include "molgr/stages/fresh.h"
#include "molgr/stages/internal_helpers.h"

#include "molgr/utils/electrons.h"
#include "molgr/utils/bond_geometry.h"
#include "molgr/utils/smarts.h"
#include "molgr/vendor/openbabel_threading.h"
#include "molgr/vendor/openbabel_copy.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/kekulize.h>
#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <set>
#include <utility>

namespace molgr
{
    namespace reconstruct
    {
        using namespace OpenBabel;

        namespace
        {
            /// @param mol Ligand graph carrying explicit unpaired and deferred two-electron labels.
            /// @return True when two distinct atoms have unpaired electrons or unresolved two-electron
            ///     centers.
            bool HasTwoRadicalEndpoints(OBMol &mol)
            {
                unsigned sites = 0;
                FOR_ATOMS_OF_MOL(atom, mol)
                {
                    if (molgr::utils::GetUnpairedElectronCount(*atom) > 0 ||
                        molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                    {
                        if (++sites == 2) return true;
                    }
                }
                return false;
            }

            // Inspect the complete proposed bond pattern on a copy. Otherwise
            // cleanup can discard a viable hydrazone/oximate for an azo/nitroso graph
            // that the unchanged final geometry validator rejects.
            /// Check the proposed pi pattern before consuming terminal electrons.
            /// @param mol Molecular graph on which the operation acts.
            /// @param bonds Bond records involved in the local operation.
            /// @param charge_changes Proposed changes to the formal charges of the affected atoms.
            /// @return True when all affected nitrogen environments remain compatible with the proposed
            ///     bond orders.
            bool PiCleanupPreservesNitrogenGeometry(
                OBMol &mol, std::initializer_list<OBBond *> bonds,
                std::initializer_list<std::pair<unsigned, int>> charge_changes = {})
            {
                std::set<unsigned> nitrogens;
                for (auto *bond : bonds)
                {
                    for (auto *atom : {bond->GetBeginAtom(), bond->GetEndAtom()})
                        if (atom->GetAtomicNum() == 7) nitrogens.insert(atom->GetIdx());
                }
                for (const auto &[atom_idx, delta] : charge_changes)
                    if (mol.GetAtom(atom_idx)->GetAtomicNum() == 7) nitrogens.insert(atom_idx);
                bool has_nitrogen_pair = false;
                for (unsigned idx : nitrogens)
                    FOR_NBORS_OF_ATOM(neighbor, mol.GetAtom(idx))
                        if (neighbor->GetAtomicNum() == 7 || neighbor->GetAtomicNum() == 8)
                            has_nitrogen_pair = true;
                if (!has_nitrogen_pair) return true;
                auto proposed = molgr::vendor::openbabel_threading::SharedMolCopy(mol);
                for (const auto &[atom_idx, delta] : charge_changes)
                {
                    auto *atom = proposed->GetAtom(atom_idx);
                    atom->SetFormalCharge(atom->GetFormalCharge() + delta);
                }
                std::size_t index = 0;
                for (auto *bond : bonds)
                    proposed->GetBond(bond->GetBeginAtomIdx(), bond->GetEndAtomIdx())
                        ->SetBondOrder(index++ % 2 == 0 ? 2 : 1);
                // Lowering C=N can invalidate an adjacent N=O whose order
                // did not change. Check the affected nitrogens' complete
                // neighborhoods, not only bonds promoted to double bonds.
                for (unsigned idx : nitrogens)
                {
                    auto *nitrogen = proposed->GetAtom(idx);
                    FOR_BONDS_OF_ATOM(bond, nitrogen)
                    {
                        auto *other = bond->GetNbrAtom(nitrogen);
                        if ((other->GetAtomicNum() == 7 || other->GetAtomicNum() == 8) &&
                            (!molgr::bond_geometry::NitrogenNitrogenOrderAllowed(
                                *nitrogen, *other, bond->GetBondOrder()) ||
                             !molgr::bond_geometry::NitrogenOxygenOrderAllowed(
                                *nitrogen, *other, bond->GetBondOrder()))) return false;
                    }
                }
                return true;
            }
        }

        // Electron bookkeeping: move an unresolved A center through A-B=C,
        // consume its marker, and produce one real unpaired electron at A and C.
        // Active lone pairs at C are preserved.
        bool CleanCarbeneNeighborUnsaturated(OBMol &mol)
        {
            bool hit = false;
            while (true)
            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_CARBENE_NEIGHBOR_UNSAT);
                if (matches.empty())
                    break;
                bool any_applied = false;

                for (const auto &idxs : matches)
                {
                    OBAtom *a1 = mol.GetAtom(idxs[0]);
                    OBAtom *a2 = mol.GetAtom(idxs[1]);
                    OBAtom *a3 = mol.GetAtom(idxs[2]);

                    OBBond *b23 = mol.GetBond(a2, a3);
                    OBBond *b12 = mol.GetBond(a1, a2);
                    if (molgr::utils::HasUnresolvedTwoElectronCenter(*a1) &&
                        molgr::utils::GetUnpairedElectronCount(*a3) == 0 &&
                        !molgr::utils::HasUnresolvedTwoElectronCenter(*a3) &&
                        b23 && b12 && b12->GetBondOrder() == 1 &&
                        b23->GetBondOrder() == 2 &&
                        !molgr::bond_geometry::ProtectShortCarbonyl(*a2, *a3, *a1))
                    {
                        b23->SetBondOrder(b23->GetBondOrder() - 1);
                        b12->SetBondOrder(b12->GetBondOrder() + 1);
                        molgr::utils::SetUnresolvedTwoElectronCenter(*a1, false);
                        molgr::utils::SetLonePairCount(*a1, 0);
                        molgr::utils::SetUnpairedElectronCount(*a1, 1);
                        molgr::utils::SetUnpairedElectronCount(*a3, molgr::utils::GetUnpairedElectronCount(*a3) + 1);
                        hit = true;
                        any_applied = true;
                        break;
                    }
                }
                if (!any_applied)
                    break;
            }
            return hit;
        }

        // Convert two excess terminal radicals into one neutral 1,3-dipole.
        // The target radical count and pending absolute charge reserve are global
        // constraints; each accepted fragment consumes one radical at each end.
        bool CleanPossible13Dipole(
            OBMol &mol,
            int given_charge,
            int total_radical_electrons)
        {
            int available_unpaired_electrons = [&]()
            {
                int count = 0;
                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    count += molgr::utils::GetUnpairedElectronCount(*atom_iter);
                }
                return count - total_radical_electrons - std::abs(given_charge);
            }();

            bool hit = false;
            if (available_unpaired_electrons < 2)
            {
                molgr::vendor::openbabel_threading::PrepareForSmartsMatching(mol);
                return false;
            }
            const auto matches = molgr::smarts::FindAll(
                mol,
                molgr::smarts::PatternId::CLEAN_POSSIBLE_1_3_DIPOLE);
            for (const auto &idxs : matches)
            {
                if (available_unpaired_electrons < 2)
                {
                    break;
                }
                if (idxs.size() != 3)
                {
                    continue;
                }

                OBAtom *atom1 = mol.GetAtom(idxs[0]);
                OBAtom *atom2 = mol.GetAtom(idxs[1]);
                OBAtom *atom3 = mol.GetAtom(idxs[2]);
                OBBond *bond12 = mol.GetBond(idxs[0], idxs[1]);
                OBBond *bond23 = mol.GetBond(idxs[1], idxs[2]);
                if (!atom1 || !atom2 || !atom3 || !bond12 || !bond23)
                {
                    continue;
                }
                if (atom1->GetFormalCharge() != 0 ||
                    atom2->GetFormalCharge() != 0 ||
                    atom3->GetFormalCharge() != 0 ||
                    molgr::utils::GetUnpairedElectronCount(*atom1) < 1 ||
                    molgr::utils::GetUnpairedElectronCount(*atom3) < 1 ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom1) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom3))
                {
                    continue;
                }

                const int degree1 = static_cast<int>(atom1->GetExplicitDegree());
                const int degree3 = static_cast<int>(atom3->GetExplicitDegree());
                if (degree1 <= 0 || degree3 <= 0)
                {
                    continue;
                }
                const int valence1 = static_cast<int>(atom1->GetExplicitValence());
                const int valence3 = static_cast<int>(atom3->GetExplicitValence());
                const int left_average_key = valence1 * degree3;
                const int right_average_key = valence3 * degree1;
                const bool add_left_bond =
                    left_average_key < right_average_key ||
                    (left_average_key == right_average_key && atom1->GetIdx() < atom3->GetIdx());
                OBBond *bond_to_increase = add_left_bond ? bond12 : bond23;
                OBAtom *negative_atom = add_left_bond ? atom3 : atom1;
                if (bond_to_increase->GetBondOrder() >= 3)
                {
                    continue;
                }

                bond_to_increase->SetBondOrder(bond_to_increase->GetBondOrder() + 1);
                atom2->SetFormalCharge(1);
                negative_atom->SetFormalCharge(-1);
                molgr::utils::SetUnpairedElectronCount(
                    *atom1,
                    molgr::utils::GetUnpairedElectronCount(*atom1) - 1);
                molgr::utils::SetUnpairedElectronCount(
                    *atom3,
                    molgr::utils::GetUnpairedElectronCount(*atom3) - 1);
                available_unpaired_electrons -= 2;
                hit = true;
            }
            return hit;
        }

        // Electron bookkeeping: resolve adjacent radical-compatible states.
        // Each endpoint contributes either its real unpaired electrons or the
        // two electrons represented by an unresolved center. A matched pair
        // consumes one electron at each endpoint and raises the bond order by
        // one. The operation is limited by the excess-electron budget, so an
        // unresolved/unresolved pair may stop at a double bond and leave real
        // unpaired electrons to be classified later.
        bool CleanNeighborRadicals(
            OBMol &mol,
            int given_charge,
            int total_radical_electrons)
        {
            /// Count electrons available for bonding after reserving the charge and spin targets.
            const auto available_electron_budget = [&]()
            {
                int count = 0;
                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    count += molgr::utils::GetUnpairedElectronCount(*atom_iter);
                    if (molgr::utils::HasUnresolvedTwoElectronCenter(*atom_iter))
                    {
                        count += 2;
                    }
                }
                return count - total_radical_electrons - std::abs(given_charge);
            };

            bool hit = false;
            FOR_BONDS_OF_MOL(bond_iter, mol)
            {
                OBBond *bond = &(*bond_iter);
                OBAtom *a1 = bond->GetBeginAtom();
                OBAtom *a2 = bond->GetEndAtom();
                int r1 = molgr::utils::GetUnpairedElectronCount(*a1);
                int r2 = molgr::utils::GetUnpairedElectronCount(*a2);
                const bool u1 = molgr::utils::HasUnresolvedTwoElectronCenter(*a1);
                const bool u2 = molgr::utils::HasUnresolvedTwoElectronCenter(*a2);
                const int capacity1 = u1 ? 2 : r1;
                const int capacity2 = u2 ? 2 : r2;
                if (capacity1 <= 0 || capacity2 <= 0)
                {
                    continue;
                }

                const int available = available_electron_budget();
                if (available < 2)
                {
                    continue;
                }
                // A nonmetal Lewis bond has at most three bond-order units.
                // Keep excess electrons available for other search branches.
                const int to_add = std::min({capacity1, capacity2, available / 2,
                                             3 - static_cast<int>(bond->GetBondOrder())});
                if (to_add <= 0)
                {
                    continue;
                }

                bond->SetBondOrder(bond->GetBondOrder() + to_add);
                /// Subtract the electrons used for the new bond and reclassify the remaining endpoint
                /// occupancy.
                const auto consume_endpoint = [&](OBAtom *atom, bool unresolved, int capacity)
                {
                    const int remaining = capacity - to_add;
                    if (unresolved)
                    {
                        molgr::utils::SetUnresolvedTwoElectronCenter(*atom, false);
                        molgr::utils::SetLonePairCount(*atom, 0);
                        molgr::utils::SetUnpairedElectronCount(*atom, remaining);
                        // Do not reclassify a partial unresolved center: its
                        // post-bond deficit would immediately recreate the
                        // unresolved marker.
                        if (remaining == 0)
                        {
                            AssignChargeRadicalForAtom(*atom);
                        }
                    }
                    else
                    {
                        molgr::utils::SetUnpairedElectronCount(*atom, remaining);
                        AssignChargeRadicalForAtom(*atom);
                    }
                };
                consume_endpoint(a1, u1, capacity1);
                consume_endpoint(a2, u2, capacity2);
                hit = true;
            }
            return hit;
        }

        // Electron bookkeeping: resolve separated terminal radical-compatible
        // states across A-B=C-D by shifting the middle pi bond outward.
        bool Clean14Radicals(
            OBMol &mol,
            int given_charge,
            int total_radical_electrons)
        {
            if (!HasTwoRadicalEndpoints(mol))
            {
                molgr::vendor::openbabel_threading::PrepareForSmartsMatching(mol);
                return false;
            }
            /// Count explicit unpaired electrons available after reserving the target spin and charge.
            const auto available_unpaired_electrons = [&]()
            {
                int count = 0;
                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    count += molgr::utils::GetUnpairedElectronCount(*atom_iter);
                }
                return count - total_radical_electrons - std::abs(given_charge);
            };

            bool hit = false;
            const auto matches = molgr::smarts::FindAll(
                mol,
                molgr::smarts::PatternId::CLEAN_1_4_RADICALS);
            for (const auto &idxs : matches)
            {
                if (idxs.size() != 4)
                {
                    continue;
                }
                OBAtom *a1 = mol.GetAtom(idxs[0]);
                OBAtom *a2 = mol.GetAtom(idxs[1]);
                OBAtom *a3 = mol.GetAtom(idxs[2]);
                OBAtom *a4 = mol.GetAtom(idxs[3]);
                OBBond *b1 = mol.GetBond(a1, a2);
                OBBond *b2 = mol.GetBond(a2, a3);
                OBBond *b3 = mol.GetBond(a3, a4);
                if (!a1 || !a2 || !a3 || !a4 || !b1 || !b2 || !b3 ||
                    b1->GetBondOrder() != 1 || b2->GetBondOrder() != 2 ||
                    b3->GetBondOrder() != 1)
                {
                    continue;
                }

                const int r1 = molgr::utils::GetUnpairedElectronCount(*a1);
                const int r4 = molgr::utils::GetUnpairedElectronCount(*a4);
                const bool u1 = molgr::utils::HasUnresolvedTwoElectronCenter(*a1);
                const bool u4 = molgr::utils::HasUnresolvedTwoElectronCenter(*a4);
                if ((r1 <= 0 && !u1) || (r4 <= 0 && !u4))
                {
                    continue;
                }
                if (static_cast<int>(r1 > 0) + static_cast<int>(r4 > 0) == 2 &&
                    available_unpaired_electrons() < 2)
                {
                    continue;
                }

                if (!PiCleanupPreservesNitrogenGeometry(mol, {b1, b2, b3})) continue;

                b1->SetBondOrder(2);
                b2->SetBondOrder(1);
                b3->SetBondOrder(2);
                if (r1 > 0)
                {
                    molgr::utils::SetUnpairedElectronCount(*a1, r1 - 1);
                }
                if (r4 > 0)
                {
                    molgr::utils::SetUnpairedElectronCount(*a4, r4 - 1);
                }
                if (u1)
                {
                    molgr::utils::SetUnresolvedTwoElectronCenter(*a1, false);
                    molgr::utils::SetLonePairCount(*a1, 0);
                }
                if (u4)
                {
                    molgr::utils::SetUnresolvedTwoElectronCenter(*a4, false);
                    molgr::utils::SetLonePairCount(*a4, 0);
                }
                AssignChargeRadicalForAtom(*a1);
                AssignChargeRadicalForAtom(*a4);
                hit = true;
            }
            return hit;
        }

        // Electron bookkeeping: resolve terminal radical-compatible states
        // across A-B=C-D=E-F by shifting both middle pi bonds outward.
        bool Clean16Radicals(
            OBMol &mol,
            int given_charge,
            int total_radical_electrons)
        {
            if (!HasTwoRadicalEndpoints(mol))
            {
                molgr::vendor::openbabel_threading::PrepareForSmartsMatching(mol);
                return false;
            }
            /// Count explicit unpaired electrons available after reserving the target spin and charge.
            const auto available_unpaired_electrons = [&]()
            {
                int count = 0;
                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    count += molgr::utils::GetUnpairedElectronCount(*atom_iter);
                }
                return count - total_radical_electrons - std::abs(given_charge);
            };

            bool hit = false;
            const auto matches = molgr::smarts::FindAll(
                mol,
                molgr::smarts::PatternId::CLEAN_1_6_RADICALS);
            for (const auto &idxs : matches)
            {
                if (idxs.size() != 6)
                {
                    continue;
                }
                OBAtom *a1 = mol.GetAtom(idxs[0]);
                OBAtom *a2 = mol.GetAtom(idxs[1]);
                OBAtom *a3 = mol.GetAtom(idxs[2]);
                OBAtom *a4 = mol.GetAtom(idxs[3]);
                OBAtom *a5 = mol.GetAtom(idxs[4]);
                OBAtom *a6 = mol.GetAtom(idxs[5]);
                OBBond *b1 = mol.GetBond(a1, a2);
                OBBond *b2 = mol.GetBond(a2, a3);
                OBBond *b3 = mol.GetBond(a3, a4);
                OBBond *b4 = mol.GetBond(a4, a5);
                OBBond *b5 = mol.GetBond(a5, a6);
                if (!a1 || !a2 || !a3 || !a4 || !a5 || !a6 ||
                    !b1 || !b2 || !b3 || !b4 || !b5 ||
                    b1->GetBondOrder() != 1 || b2->GetBondOrder() != 2 ||
                    b3->GetBondOrder() != 1 || b4->GetBondOrder() != 2 ||
                    b5->GetBondOrder() != 1)
                {
                    continue;
                }

                const int r1 = molgr::utils::GetUnpairedElectronCount(*a1);
                const int r6 = molgr::utils::GetUnpairedElectronCount(*a6);
                const bool u1 = molgr::utils::HasUnresolvedTwoElectronCenter(*a1);
                const bool u6 = molgr::utils::HasUnresolvedTwoElectronCenter(*a6);
                if ((r1 <= 0 && !u1) || (r6 <= 0 && !u6))
                {
                    continue;
                }
                if (static_cast<int>(r1 > 0) + static_cast<int>(r6 > 0) == 2 &&
                    available_unpaired_electrons() < 2)
                {
                    continue;
                }

                if (!PiCleanupPreservesNitrogenGeometry(mol, {b1, b2, b3, b4, b5})) continue;

                b1->SetBondOrder(2);
                b2->SetBondOrder(1);
                b3->SetBondOrder(2);
                b4->SetBondOrder(1);
                b5->SetBondOrder(2);
                if (r1 > 0)
                {
                    molgr::utils::SetUnpairedElectronCount(*a1, r1 - 1);
                }
                if (r6 > 0)
                {
                    molgr::utils::SetUnpairedElectronCount(*a6, r6 - 1);
                }
                if (u1)
                {
                    molgr::utils::SetUnresolvedTwoElectronCenter(*a1, false);
                    molgr::utils::SetLonePairCount(*a1, 0);
                }
                if (u6)
                {
                    molgr::utils::SetUnresolvedTwoElectronCenter(*a6, false);
                    molgr::utils::SetLonePairCount(*a6, 0);
                }
                AssignChargeRadicalForAtom(*a1);
                AssignChargeRadicalForAtom(*a6);
                hit = true;
            }
            return hit;
        }

        namespace
        {
            /// Neutralize opposite terminal charges by shifting three consecutive bond orders.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances0(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_0);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom4 = mol.GetAtom(idxs[3]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBBond *bond3 = mol.GetBond(idxs[2], idxs[3]);
                    if (!atom1 || !atom4 || !bond1 || !bond2 || !bond3)
                    {
                        continue;
                    }

                    const ElementInfo *info1 = GetElementInfo(atom1->GetAtomicNum());
                    const ElementInfo *info4 = GetElementInfo(atom4->GetAtomicNum());
                    if (atom1->GetFormalCharge() == -1 &&
                        atom4->GetFormalCharge() == 1 &&
                        bond1->GetBondOrder() == 1 &&
                        bond2->GetBondOrder() == 2 &&
                        info1 != nullptr && info4 != nullptr &&
                        info1->default_valence > atom1->GetTotalValence() &&
                        info4->default_valence > atom4->GetTotalValence())
                    {
                        bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() - 1);
                        bond3->SetBondOrder(bond3->GetBondOrder() + 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom4->SetFormalCharge(atom4->GetFormalCharge() - 1);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Convert an anion–cation cumulene to a triple/single pair with the anion at the opposite end.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances1(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_1);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom2 = mol.GetAtom(idxs[1]);
                    OBAtom *atom3 = mol.GetAtom(idxs[2]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    if (!atom1 || !atom2 || !atom3 || !bond1 || !bond2)
                    {
                        continue;
                    }

                    if (atom1->GetFormalCharge() == -1 &&
                        atom2->GetFormalCharge() == 1 &&
                        atom3->GetFormalCharge() == 0 &&
                        bond1->GetBondOrder() == 2 &&
                        bond2->GetBondOrder() == 2)
                    {
                        bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom3->SetFormalCharge(atom3->GetFormalCharge() - 1);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Shift a conjugated terminal C or N anion onto the carbonyl oxygen through four bonds.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances2(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_2);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom6 = mol.GetAtom(idxs[5]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[3]);
                    OBBond *bond3 = mol.GetBond(idxs[3], idxs[4]);
                    OBBond *bond4 = mol.GetBond(idxs[4], idxs[5]);
                    if (!atom1 || !atom6 || !bond1 || !bond2 || !bond3 || !bond4)
                    {
                        continue;
                    }

                    const ElementInfo *info6 = GetElementInfo(atom6->GetAtomicNum());
                    const int atom6_room = info6 == nullptr
                                               ? -1
                                               : info6->default_valence -
                                                     (atom6->GetTotalValence() + 1);

                    if (bond4->GetBondOrder() == 1 &&
                        bond3->GetBondOrder() == 2 &&
                        bond2->GetBondOrder() == 1 &&
                        bond1->GetBondOrder() == 2 &&
                        atom1->GetFormalCharge() == 0 &&
                        atom6->GetFormalCharge() == -1 &&
                        atom6_room >= 0 &&
                        !molgr::bond_geometry::WouldFormBentRingAllene(*atom6, *mol.GetAtom(idxs[4])))
                    {
                        if (!PiCleanupPreservesNitrogenGeometry(mol, {bond4, bond3, bond2, bond1}, {{atom1->GetIdx(), -1}, {atom6->GetIdx(), 1}})) continue;
                        bond4->SetBondOrder(bond4->GetBondOrder() + 1);
                        bond3->SetBondOrder(bond3->GetBondOrder() - 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() + 1);
                        bond1->SetBondOrder(bond1->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() - 1);
                        atom6->SetFormalCharge(atom6->GetFormalCharge() + 1);
                        hit = true;
                    }
                }
                return hit;
            }

            // Electron bookkeeping: rule 3 changes total valence only at its two
            // charged endpoints; internal atoms exchange bond-order units with net
            // zero change. Refresh endpoints only and preserve unrelated labels.
            /// Shift the rule-3 charge/pi pattern, then rebuild its two endpoints.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances3(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_3);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom5 = mol.GetAtom(idxs[4]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBBond *bond3 = mol.GetBond(idxs[2], idxs[3]);
                    OBBond *bond4 = mol.GetBond(idxs[3], idxs[4]);
                    if (!atom1 || !atom5 || !bond1 || !bond2 || !bond3 || !bond4)
                    {
                        continue;
                    }

                    if (atom1->GetFormalCharge() == 1 &&
                        atom5->GetFormalCharge() == -1 &&
                        bond1->GetBondOrder() == 2 &&
                        bond2->GetBondOrder() == 1 &&
                        bond3->GetBondOrder() == 2 &&
                        bond4->GetBondOrder() == 1)
                    {
                        if (!PiCleanupPreservesNitrogenGeometry(
                                mol, {bond4, bond3, bond2, bond1},
                                {{atom1->GetIdx(), -1}, {atom5->GetIdx(), 1}})) continue;
                        bond1->SetBondOrder(bond1->GetBondOrder() - 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() + 1);
                        bond3->SetBondOrder(bond3->GetBondOrder() - 1);
                        bond4->SetBondOrder(bond4->GetBondOrder() + 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() - 1);
                        atom5->SetFormalCharge(atom5->GetFormalCharge() + 1);
                        AssignChargeRadicalForAtom(*atom1);
                        AssignChargeRadicalForAtom(*atom5);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Neutralize a cationic N/O and a 1,3-related anion by moving their intervening pi bond.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances4(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_4);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom3 = mol.GetAtom(idxs[2]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    if (!atom1 || !atom3 || !bond1 || !bond2)
                    {
                        continue;
                    }

                    if (atom1->GetFormalCharge() == 1 &&
                        atom3->GetFormalCharge() == -1 &&
                        bond1->GetBondOrder() == 2 &&
                        bond2->GetBondOrder() == 1)
                    {
                        bond2->SetBondOrder(bond2->GetBondOrder() + 1);
                        bond1->SetBondOrder(bond1->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() - 1);
                        atom3->SetFormalCharge(atom3->GetFormalCharge() + 1);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Move an allylic anion from C/N onto a conjugated neutral N, O or S.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances5(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_5);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom3 = mol.GetAtom(idxs[2]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    if (!atom1 || !atom3 || !bond1 || !bond2)
                    {
                        continue;
                    }

                    if (bond1->GetBondOrder() == 2 &&
                        bond2->GetBondOrder() == 1 &&
                        atom3->GetFormalCharge() == -1 &&
                        atom1->GetFormalCharge() == 0 &&
                        !molgr::bond_geometry::WouldFormBentRingAllene(*atom3, *mol.GetAtom(idxs[1])) &&
                        !molgr::bond_geometry::ProtectShortCarbonyl(*mol.GetAtom(idxs[1]), *atom1, *atom3))
                    {
                        bond2->SetBondOrder(bond2->GetBondOrder() + 1);
                        bond1->SetBondOrder(bond1->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() - 1);
                        atom3->SetFormalCharge(atom3->GetFormalCharge() + 1);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Localize an anionic cumulene as a triple bond with the opposite carbon carrying the charge.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances6(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_6);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom3 = mol.GetAtom(idxs[2]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    if (!atom1 || !atom3 || !bond1 || !bond2)
                    {
                        continue;
                    }

                    if (atom1->GetFormalCharge() == 0 &&
                        atom3->GetFormalCharge() == -1 &&
                        bond1->GetBondOrder() == 2 &&
                        bond2->GetBondOrder() == 2)
                    {
                        bond2->SetBondOrder(bond2->GetBondOrder() + 1);
                        bond1->SetBondOrder(bond1->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() - 1);
                        atom3->SetFormalCharge(atom3->GetFormalCharge() + 1);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Move a ring anion onto its exocyclic neighbor while restoring an endocyclic double bond.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances7(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_7);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom3 = mol.GetAtom(idxs[2]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    if (!atom1 || !atom3 || !bond1 || !bond2)
                    {
                        continue;
                    }
                    const ElementInfo *info1 = GetElementInfo(atom1->GetAtomicNum());
                    const int atom1_room = info1 == nullptr
                                               ? -1
                                               : info1->default_valence -
                                                     (atom1->GetTotalValence() + 1);
                    if (bond1->GetBondOrder() == 1 &&
                        bond2->GetBondOrder() == 2 &&
                        atom1->GetFormalCharge() == -1 &&
                        atom3->GetFormalCharge() == 0 &&
                        atom1_room >= 0)
                    {
                        bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom3->SetFormalCharge(atom3->GetFormalCharge() - 1);
                        hit = true;
                    }
                }
                return hit;
            }

            // Electron bookkeeping: shift charge/pi bonds and locally rebuild both
            // endpoints so newly exposed unpaired/lone-pair/unresolved states are
            // not hidden by stale fields.
            /// Shift the rule-8 charge/pi pattern and rebuild both endpoint states.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances8(OBMol &mol)
            {
                bool hit = false;
                molgr::vendor::openbabel_threading::SetAromaticPerceived(mol, false);
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_8);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom5 = mol.GetAtom(idxs[4]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBBond *bond3 = mol.GetBond(idxs[2], idxs[3]);
                    OBBond *bond4 = mol.GetBond(idxs[3], idxs[4]);
                    if (!atom1 || !atom5 || !bond1 || !bond2 || !bond3 || !bond4)
                    {
                        continue;
                    }
                    if (atom1->GetFormalCharge() == -1 &&
                        atom5->GetFormalCharge() == 0 &&
                        bond1->GetBondOrder() == 1 &&
                        bond2->GetBondOrder() == 2 &&
                        bond3->GetBondOrder() == 1 &&
                        bond4->GetBondOrder() == 2)
                    {
                        bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() - 1);
                        bond3->SetBondOrder(bond3->GetBondOrder() + 1);
                        bond4->SetBondOrder(bond4->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom5->SetFormalCharge(atom5->GetFormalCharge() - 1);
                        AssignChargeRadicalForAtom(*atom1);
                        AssignChargeRadicalForAtom(*atom5);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Increase bond order to neutralize adjacent opposite charges, up to a triple bond.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances9(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_9);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom2 = mol.GetAtom(idxs[1]);
                    OBBond *bond = mol.GetBond(idxs[0], idxs[1]);
                    if (!atom1 || !atom2 || !bond)
                    {
                        continue;
                    }

                    const ElementInfo *info1 = GetElementInfo(atom1->GetAtomicNum());
                    const ElementInfo *info2 = GetElementInfo(atom2->GetAtomicNum());
                    if (info1 == nullptr || info2 == nullptr)
                    {
                        continue;
                    }

                    int room1 = info1->default_valence - atom1->GetTotalValence();
                    int room2 = info2->default_valence - atom2->GetTotalValence();
                    if (atom1->GetFormalCharge() > 0 &&
                        atom2->GetFormalCharge() < 0 &&
                        (bond->GetBondOrder() == 1 || bond->GetBondOrder() == 2) &&
                        room1 >= 1 && room2 >= 1)
                    {
                        int bond_to_add = std::min({room1, room2,
                                                   3 - static_cast<int>(bond->GetBondOrder())});
                        bond->SetBondOrder(bond->GetBondOrder() + bond_to_add);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() - bond_to_add);
                        atom2->SetFormalCharge(atom2->GetFormalCharge() + bond_to_add);
                        hit = true;
                    }
                }
                return hit;
            }

            // Electron bookkeeping: require and consume one real unpaired electron
            // at each terminal to create two new bond-order units.
            /// Consume two terminal monoradicals while redistributing one middle pi bond.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances10(OBMol &mol)
            {
                bool hit = false;
                unsigned monoradicals = 0;
                FOR_ATOMS_OF_MOL(atom, mol)
                {
                    if (molgr::utils::GetUnpairedElectronCount(*atom) == 1 &&
                        ++monoradicals == 2) break;
                }
                if (monoradicals < 2)
                {
                    molgr::vendor::openbabel_threading::PrepareForSmartsMatching(mol);
                    return false;
                }
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_10);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom4 = mol.GetAtom(idxs[3]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBBond *bond3 = mol.GetBond(idxs[2], idxs[3]);
                    if (!atom1 || !atom4 || !bond1 || !bond2 || !bond3)
                    {
                        continue;
                    }

                    if (molgr::utils::GetUnpairedElectronCount(*atom1) == 1 &&
                        molgr::utils::GetUnpairedElectronCount(*atom4) == 1 &&
                        bond1->GetBondOrder() == 1 &&
                        (bond2->GetBondOrder() == 2 || bond2->GetBondOrder() == 3) &&
                        bond3->GetBondOrder() == 1)
                    {
                        bond2->SetBondOrder(bond2->GetBondOrder() - 1);
                        bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                        bond3->SetBondOrder(bond3->GetBondOrder() + 1);
                        molgr::utils::SetUnpairedElectronCount(*atom1, molgr::utils::GetUnpairedElectronCount(*atom1) - 1);
                        molgr::utils::SetUnpairedElectronCount(*atom4, molgr::utils::GetUnpairedElectronCount(*atom4) - 1);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Donate a heteroatom lone pair into an adjacent cation when the new bond geometry permits it.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances11(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_11);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom2 = mol.GetAtom(idxs[1]);
                    OBBond *bond = mol.GetBond(idxs[0], idxs[1]);
                    if (!atom1 || !atom2 || !bond)
                    {
                        continue;
                    }

                    const ElementInfo *info2 = GetElementInfo(atom2->GetAtomicNum());
                    const int atom2_room = info2 == nullptr
                                               ? -1
                                               : info2->default_valence - atom2->GetTotalValence();
                    if (info2 != nullptr &&
                        atom1->GetFormalCharge() == 0 &&
                        atom2->GetFormalCharge() == 1 &&
                        atom2_room >= 1 &&
                        (bond->GetBondOrder() == 1 || bond->GetBondOrder() == 2))
                    {
                        // Validate the proposed donor charge as well as the N/O bond order.
                        if (bond->GetBondOrder() == 1 && !PiCleanupPreservesNitrogenGeometry(
                                mol, {bond}, {{atom1->GetIdx(), 1}, {atom2->GetIdx(), -1}}))
                            continue;
                        bond->SetBondOrder(bond->GetBondOrder() + 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom2->SetFormalCharge(atom2->GetFormalCharge() - 1);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Relay heteroatom donation through an alternating four-atom path to a terminal cation.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances12(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_12);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom4 = mol.GetAtom(idxs[3]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBBond *bond3 = mol.GetBond(idxs[2], idxs[3]);
                    if (!atom1 || !atom4 || !bond1 || !bond2 || !bond3)
                    {
                        continue;
                    }

                    const ElementInfo *info4 = GetElementInfo(atom4->GetAtomicNum());
                    const int atom4_room = info4 == nullptr
                                               ? -1
                                               : info4->default_valence - atom4->GetTotalValence();
                    if (info4 != nullptr &&
                        atom1->GetFormalCharge() == 0 &&
                        atom4->GetFormalCharge() == 1 &&
                        atom4_room >= 1 &&
                        bond1->GetBondOrder() == 1 &&
                        bond2->GetBondOrder() == 2 &&
                        bond3->GetBondOrder() == 1)
                    {
                        bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() - 1);
                        bond3->SetBondOrder(bond3->GetBondOrder() + 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom4->SetFormalCharge(atom4->GetFormalCharge() - 1);
                        hit = true;
                    }
                }
                return hit;
            }

            /// Move a ring anion onto an exocyclic imine N or carbonyl O by shifting the pi bond.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances13(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_13);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom3 = mol.GetAtom(idxs[2]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    if (!atom1 || !atom3 || !bond1 || !bond2)
                    {
                        continue;
                    }

                    const ElementInfo *info1 = GetElementInfo(atom1->GetAtomicNum());
                    if (info1 != nullptr &&
                        atom1->GetFormalCharge() == -1 &&
                        atom3->GetFormalCharge() == 0 &&
                        info1->default_valence - atom1->GetTotalValence() >= 1 &&
                        bond1->GetBondOrder() == 1 &&
                        bond2->GetBondOrder() == 2)
                    {
                        bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom3->SetFormalCharge(atom3->GetFormalCharge() - 1);
                        hit = true;
                    }
                }
                return hit;
            }

            // Electron bookkeeping: lower the charged triple bond, neutralize both
            // endpoints, and rebuild their explicit electron classifications.
            /// Neutralize the rule-14 triple-bond ion pair and rebuild both endpoints.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances14Impl(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_14);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom2 = mol.GetAtom(idxs[1]);
                    OBBond *bond = mol.GetBond(idxs[0], idxs[1]);
                    if (!atom1 || !atom2 || !bond)
                    {
                        continue;
                    }

                    if (atom1->GetFormalCharge() == -1 &&
                        atom2->GetFormalCharge() == 1 &&
                        bond->GetBondOrder() == 3)
                    {
                        bond->SetBondOrder(bond->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom2->SetFormalCharge(atom2->GetFormalCharge() - 1);
                        AssignChargeRadicalForAtom(*atom1);
                        AssignChargeRadicalForAtom(*atom2);
                        hit = true;
                    }
                }
                return hit;
            }

            // Electron bookkeeping: neutralize the conjugated ion pair, then
            // rebuild both endpoints to expose any newly created explicit or
            // unresolved electron state.
            /// Neutralize the rule-16 conjugated ion pair and rebuild both endpoints.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances16Impl(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_16);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom5 = mol.GetAtom(idxs[4]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBBond *bond3 = mol.GetBond(idxs[2], idxs[3]);
                    OBBond *bond4 = mol.GetBond(idxs[3], idxs[4]);
                    if (!atom1 || !atom5 || !bond1 || !bond2 || !bond3 || !bond4)
                    {
                        continue;
                    }

                    if (atom1->GetFormalCharge() == -1 &&
                        molgr::utils::GetUnpairedElectronCount(*atom1) == 0 &&
                        molgr::utils::GetLonePairCount(*atom1) == 0 &&
                        atom5->GetFormalCharge() == 1 &&
                        molgr::utils::GetUnpairedElectronCount(*atom5) == 0 &&
                        molgr::utils::GetLonePairCount(*atom5) == 0 &&
                        bond1->GetBondOrder() == 1 &&
                        bond2->GetBondOrder() == 2 &&
                        bond3->GetBondOrder() == 1 &&
                        bond4->GetBondOrder() == 2)
                    {
                        if (!PiCleanupPreservesNitrogenGeometry(
                                mol, {bond1, bond2, bond3, bond4},
                                {{atom1->GetIdx(), 1}, {atom5->GetIdx(), -1}})) continue;
                        bond1->SetBondOrder(bond1->GetBondOrder() + 1);
                        bond2->SetBondOrder(bond2->GetBondOrder() - 1);
                        bond3->SetBondOrder(bond3->GetBondOrder() + 1);
                        bond4->SetBondOrder(bond4->GetBondOrder() - 1);
                        atom1->SetFormalCharge(atom1->GetFormalCharge() + 1);
                        atom5->SetFormalCharge(atom5->GetFormalCharge() - 1);
                        AssignChargeRadicalForAtom(*atom1);
                        AssignChargeRadicalForAtom(*atom5);
                        hit = true;
                    }
                }
                return hit;
            }

            // Shift A(-)-B=C=D to A=B-C(-)=D only when B, C, and D are
            // ring atoms, replacing a ring allene with conjugated pi bonds.
            /// Replace a ring allene with its charge-shifted conjugated resonance form.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances17Impl(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_17);
                for (const auto &idxs : matches)
                {

                    OBAtom *atom1 = mol.GetAtom(idxs[0]);
                    OBAtom *atom3 = mol.GetAtom(idxs[2]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBBond *bond3 = mol.GetBond(idxs[2], idxs[3]);
                    if (!atom1 || !atom3 || !bond1 || !bond2 || !bond3)
                    {
                        continue;
                    }
                    const ElementInfo *info1 = GetElementInfo(atom1->GetAtomicNum());
                    const int atom1_room = info1 == nullptr
                                               ? -1
                                               : info1->default_valence -
                                                     (atom1->GetTotalValence() + 1);
                    if (atom1->GetFormalCharge() == -1 &&
                        atom3->GetFormalCharge() == 0 &&
                        bond1->GetBondOrder() == 1 &&
                        bond2->GetBondOrder() == 2 &&
                        bond3->GetBondOrder() == 2 &&
                        atom1_room >= 0)
                    {
                        bond1->SetBondOrder(2);
                        bond2->SetBondOrder(1);
                        atom1->SetFormalCharge(0);
                        atom3->SetFormalCharge(-1);
                        hit = true;
                    }
                }
                return hit;
            }

            // Electron bookkeeping: resolve an unresolved terminal diazene as
            // the charge-separated azide resonance form.
            /// Convert an unresolved terminal diazene fragment to a charge-separated azide.
            /// @param mol Ligand graph updated in place when the motif matches.
            /// @return True if at least one bond or electronic assignment changed.
            bool CleanResonances18Impl(OBMol &mol)
            {
                bool hit = false;
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::CLEAN_RESONANCE_18);
                for (const auto &idxs : matches)
                {
                    if (idxs.size() != 4)
                    {
                        continue;
                    }

                    OBAtom *atom2 = mol.GetAtom(idxs[1]);
                    OBAtom *atom3 = mol.GetAtom(idxs[2]);
                    OBAtom *atom4 = mol.GetAtom(idxs[3]);
                    OBBond *bond1 = mol.GetBond(idxs[0], idxs[1]);
                    OBBond *bond2 = mol.GetBond(idxs[1], idxs[2]);
                    OBBond *bond3 = mol.GetBond(idxs[2], idxs[3]);
                    if (!atom2 || !atom3 || !atom4 || !bond1 || !bond2 || !bond3)
                    {
                        continue;
                    }
                    if (atom2->GetFormalCharge() != 0 ||
                        atom3->GetFormalCharge() != 0 ||
                        atom4->GetFormalCharge() != 0 ||
                        molgr::utils::GetUnpairedElectronCount(*atom2) != 0 ||
                        molgr::utils::GetUnpairedElectronCount(*atom3) != 0 ||
                        molgr::utils::GetUnpairedElectronCount(*atom4) != 0 ||
                        molgr::utils::HasUnresolvedTwoElectronCenter(*atom2) ||
                        molgr::utils::HasUnresolvedTwoElectronCenter(*atom3) ||
                        !molgr::utils::HasUnresolvedTwoElectronCenter(*atom4) ||
                        bond1->GetBondOrder() != 1 ||
                        bond2->GetBondOrder() != 2 ||
                        bond3->GetBondOrder() != 1)
                    {
                        continue;
                    }

                    bond3->SetBondOrder(bond3->GetBondOrder() + 1);
                    atom3->SetFormalCharge(1);
                    atom4->SetFormalCharge(-1);
                    molgr::utils::SetUnresolvedTwoElectronCenter(*atom4, false);
                    molgr::utils::SetLonePairCount(*atom4, 0);
                    molgr::utils::SetUnpairedElectronCount(*atom4, 0);
                    hit = true;
                }
                return hit;
            }
        }

        bool CleanResonances14(OBMol &mol)
        {
            return CleanResonances14Impl(mol);
        }

        bool CleanResonances16(OBMol &mol)
        {
            return CleanResonances16Impl(mol);
        }

        bool CleanResonances17(OBMol &mol)
        {
            return CleanResonances17Impl(mol);
        }

        bool CleanResonances18(OBMol &mol)
        {
            return CleanResonances18Impl(mol);
        }

        /// C#N+-N(2-) -> C(2-)-N+#N when the terminal N-N is geometrically short.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanShortDiazoResonance(OBMol &mol)
        {
            bool hit = false;
            FOR_ATOMS_OF_MOL(t, mol)
            {
                if (t->GetAtomicNum() != 7 || t->GetExplicitDegree() != 1 || t->GetFormalCharge() != -2) continue;
                OBAtom *middle = nullptr, *carbon = nullptr;
                FOR_NBORS_OF_ATOM(a, &*t) { middle = &*a; }
                if (!middle || middle->GetAtomicNum() != 7 || middle->GetFormalCharge() != 1 || middle->GetExplicitDegree() != 2) continue;
                FOR_NBORS_OF_ATOM(a, middle) if (a->GetIdx() != t->GetIdx()) carbon = &*a;
                if (!carbon) continue;
                auto *nn = mol.GetBond(&*t, middle), *cn = mol.GetBond(middle, carbon);
                const double dnn = t->GetDistance(middle), dcn = middle->GetDistance(carbon);
                if (carbon->GetAtomicNum() != 6 || carbon->GetFormalCharge() != 0 || carbon->GetTotalValence() != 4 ||
                    carbon->GetExplicitDegree() != 2 || nn->GetBondOrder() != 1 || cn->GetBondOrder() != 3 ||
                    !(dnn > 1.05 && dnn < 1.25 && dcn > 1.20 && dcn < 1.45)) continue;
                bool active = false;
                for (auto *a : {&*t, middle, carbon}) active = active ||
                    molgr::utils::GetUnpairedElectronCount(*a) || molgr::utils::HasUnresolvedTwoElectronCenter(*a);
                if (active) continue;
                nn->SetBondOrder(3); cn->SetBondOrder(1);
                t->SetFormalCharge(0); carbon->SetFormalCharge(-2);
                AssignChargeRadicalForAtom(*t); AssignChargeRadicalForAtom(*carbon);
                hit = true;
            }
            return hit;
        }

        /// Prefer two S- termini when both long C-S and short exocyclic C-C agree.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanStretchedDithiolateResonance(OBMol &mol)
        {
            bool hit = false;
            FOR_ATOMS_OF_MOL(sulfur, mol)
            {
                if (sulfur->GetAtomicNum() != 16 || sulfur->GetExplicitDegree() != 1 || sulfur->GetFormalCharge() != 0) continue;
                OBAtom *center = nullptr;
                FOR_NBORS_OF_ATOM(a, &*sulfur) { center = &*a; }
                if (!center || center->GetAtomicNum() != 6 || center->GetExplicitDegree() != 3 ||
                    mol.GetBond(center, &*sulfur)->GetBondOrder() != 2) continue;
                OBAtom *other_s = nullptr, *carbon = nullptr;
                FOR_NBORS_OF_ATOM(a, center)
                {
                    if (a->GetIdx() == sulfur->GetIdx()) continue;
                    if (a->GetAtomicNum() == 16) other_s = &*a;
                    if (a->GetAtomicNum() == 6) carbon = &*a;
                }
                if (!other_s || !carbon || other_s->GetFormalCharge() != -1 || other_s->GetExplicitDegree() != 1) continue;
                const double cs1 = sulfur->GetDistance(center), cs2 = other_s->GetDistance(center), cc = carbon->GetDistance(center);
                if (!(cs1 > 1.73 && cs1 < 1.90 && cs2 > 1.73 && cs2 < 1.90 && cc > 1.35 && cc < 1.43)) continue;
                std::vector<std::vector<int>> paths{{static_cast<int>(sulfur->GetIdx()), static_cast<int>(center->GetIdx())}};
                std::vector<int> selected;
                for (std::size_t cursor = 0; cursor < paths.size(); ++cursor)
                {
                    const auto path = paths[cursor];
                    auto *end = mol.GetAtom(path.back());
                    if (path.size() > 2 && path.size() % 2 == 1 && end->GetFormalCharge() == -1 && end->GetTotalValence() == 3)
                    { selected = path; break; }
                    if (path.size() >= 7 || end->GetFormalCharge() != 0) continue;
                    FOR_NBORS_OF_ATOM(a, end)
                    {
                        if (std::find(path.begin(), path.end(), a->GetIdx()) == path.end() && a->GetAtomicNum() == 6 &&
                            mol.GetBond(end, &*a)->GetBondOrder() == (path.size()%2 == 0 ? 1 : 2) &&
                            !molgr::utils::GetUnpairedElectronCount(*a) && !molgr::utils::HasUnresolvedTwoElectronCenter(*a))
                        { auto next = path; next.push_back(a->GetIdx()); paths.push_back(std::move(next)); }
                    }
                }
                if (selected.empty()) continue;
                for (std::size_t i = 0; i+1 < selected.size(); ++i)
                    mol.GetBond(selected[i], selected[i+1])->SetBondOrder(1+i%2);
                auto *start = mol.GetAtom(selected.back());
                start->SetFormalCharge(0);
                sulfur->SetFormalCharge(-1);
                AssignChargeRadicalForAtom(*start);
                AssignChargeRadicalForAtom(*sulfur);
                mol.SetAromaticPerceived(false);
                hit = true;
            }
            return hit;
        }

        /// Resolve C=P+=C-N into C=P-C=N+ when short C-N supports N donation.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanPhosphorusCumulene(OBMol &mol)
        {
            bool hit = false;
            FOR_ATOMS_OF_MOL(p, mol)
            {
                if (p->GetAtomicNum() != 15 || p->GetFormalCharge() != 1 || p->GetExplicitDegree() != 2 || p->GetTotalValence() != 4) continue;
                FOR_NBORS_OF_ATOM(c, &*p)
                {
                    auto *pc = mol.GetBond(&*p, &*c);
                    if (c->GetAtomicNum() != 6 || c->GetFormalCharge() != 0 || c->GetTotalValence() != 4 ||
                        pc->GetBondOrder() != 2 || p->GetDistance(&*c) < 1.70) continue;
                    FOR_NBORS_OF_ATOM(n, &*c)
                    {
                        auto *cn = mol.GetBond(&*c, &*n);
                        const double d = c->GetDistance(&*n);
                        if (n->GetAtomicNum() != 7 || n->GetFormalCharge() != 0 || n->GetExplicitDegree() != 3 || n->GetTotalValence() != 3 ||
                            cn->GetBondOrder() != 1 || !(d > 1.25 && d < 1.43)) continue;
                        bool active = false;
                        for (auto *a : {&*p, &*c, &*n})
                            active = active || molgr::utils::GetUnpairedElectronCount(*a) || molgr::utils::HasUnresolvedTwoElectronCenter(*a);
                        if (active) continue;
                        pc->SetBondOrder(1); cn->SetBondOrder(2);
                        p->SetFormalCharge(0); n->SetFormalCharge(1);
                        mol.SetAromaticPerceived(false);
                        hit = true;
                        break;
                    }
                    if (p->GetFormalCharge() == 0) break;
                }
            }
            return hit;
        }

        /// Shift an alternating path in a closed-shell C4N ring to move its single negative
        /// charge from trivalent carbon onto the two-coordinate pyrrolide nitrogen.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanPyrrolideResonance(OBMol &mol)
        {
            bool hit = false;
            for (const auto *ring : mol.GetSSSR())
            {
                const auto ids = ring->_path;
                if (ids.size() != 5) continue;
                std::vector<int> elements, charges;
                OBAtom *nitrogen = nullptr, *start = nullptr;
                bool active = false;
                for (const int id : ids)
                {
                    auto *atom = mol.GetAtom(id);
                    elements.push_back(atom->GetAtomicNum());
                    charges.push_back(atom->GetFormalCharge());
                    if (atom->GetAtomicNum() == 7) nitrogen = atom;
                    if (atom->GetFormalCharge() == -1) start = atom;
                    active = active || molgr::utils::GetUnpairedElectronCount(*atom) ||
                        molgr::utils::HasUnresolvedTwoElectronCenter(*atom);
                }
                std::sort(elements.begin(), elements.end());
                std::sort(charges.begin(), charges.end());
                if (elements != std::vector<int>{6,6,6,6,7} ||
                    charges != std::vector<int>{-1,0,0,0,0} || active ||
                    !start || start->GetAtomicNum() != 6 || start->GetTotalValence() != 3 ||
                    nitrogen->GetExplicitDegree() != 2 || nitrogen->GetTotalValence() != 3)
                    continue;
                const int offset = std::find(ids.begin(), ids.end(), start->GetIdx()) - ids.begin();
                for (const int direction : {1, -1})
                {
                    std::vector<int> path{ids[offset]};
                    for (int step = 1; step < 5; ++step)
                    {
                        path.push_back(ids[(offset + step * direction + 5) % 5]);
                        if (path.back() == static_cast<int>(nitrogen->GetIdx())) break;
                    }
                    if ((path.size()-1) % 2) continue;
                    bool alternating = true;
                    for (std::size_t i = 0; i+1 < path.size(); ++i)
                        alternating = alternating && mol.GetBond(path[i], path[i+1])->GetBondOrder() == 1+i%2;
                    if (!alternating) continue;
                    for (std::size_t i = 0; i+1 < path.size(); ++i)
                        mol.GetBond(path[i], path[i+1])->SetBondOrder(2-i%2);
                    start->SetFormalCharge(0);
                    nitrogen->SetFormalCharge(-1);
                    AssignChargeRadicalForAtom(*start);
                    AssignChargeRadicalForAtom(*nitrogen);
                    mol.SetAromaticPerceived(false);
                    hit = true;
                    break;
                }
            }
            return hit;
        }

        // Charge-conserving C- - O+ = Si -> C = O+ - Si- resonance.
        // Only ligand geometry is used; metal states are not consulted.
        /// Move C- through C-O+=Si when short C-O and long Si-O support C=O+-Si-.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanStretchedSilyloxoniumResonance(OBMol &mol)
        {
            bool hit = false;
            FOR_ATOMS_OF_MOL(oxygen, mol)
            {
                if (oxygen->GetAtomicNum() != 8 || oxygen->GetFormalCharge() != 1 ||
                    oxygen->GetExplicitDegree() != 2) continue;
                OBAtom *carbon = nullptr, *silicon = nullptr;
                FOR_NBORS_OF_ATOM(a, &*oxygen)
                {
                    if (a->GetAtomicNum() == 6) carbon = &*a;
                    if (a->GetAtomicNum() == 14) silicon = &*a;
                }
                if (!carbon || !silicon || carbon->GetFormalCharge() != -1 ||
                    silicon->GetFormalCharge() != 0 || carbon->GetExplicitDegree() != 3 ||
                    silicon->GetExplicitDegree() != 3 || carbon->GetTotalValence() != 3 ||
                    silicon->GetTotalValence() != 4) continue;
                bool active = false;
                for (auto *a : {carbon, &*oxygen, silicon})
                    active = active || a->GetImplicitHCount() ||
                        molgr::utils::GetUnpairedElectronCount(*a) ||
                        molgr::utils::GetLonePairCount(*a) ||
                        molgr::utils::HasUnresolvedTwoElectronCenter(*a);
                if (active) continue;
                auto *co = mol.GetBond(carbon, &*oxygen), *so = mol.GetBond(silicon, &*oxygen);
                const double dco = carbon->GetDistance(&*oxygen), dso = silicon->GetDistance(&*oxygen);
                if (co->GetBondOrder() != 1 || so->GetBondOrder() != 2 ||
                    !(dco > 1.20 && dco < 1.35 && dso > 1.75 && dso < 2.15)) continue;
                co->SetBondOrder(2); so->SetBondOrder(1);
                carbon->SetFormalCharge(0); silicon->SetFormalCharge(-1);
                AssignChargeRadicalForAtom(*carbon); AssignChargeRadicalForAtom(*silicon);
                mol.SetAromaticPerceived(false);
                hit = true;
            }
            return hit;
        }

        // Localize an allylic anion at a pyramidal terminal CH2 when the
        // alternating C-C lengths support that contributor. No metal context
        // is needed, and planar/delocalized allyls retain their existing form.
        /// Place an existing allylic anion on the pyramidal CH2 end of a long C–C bond.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanPyramidalAllylResonance(OBMol &mol)
        {
            bool hit = false;
            FOR_ATOMS_OF_MOL(first, mol)
            {
                if (first->GetAtomicNum() != 6 || first->GetFormalCharge() != 0 ||
                    first->GetExplicitDegree() != 3 || first->GetTotalValence() != 4 ||
                    first->IsInRing()) continue;
                std::vector<OBAtom *> neighbors;
                OBAtom *middle = nullptr;
                int hydrogens = 0;
                FOR_NBORS_OF_ATOM(a, &*first)
                {
                    neighbors.push_back(&*a);
                    if (a->GetAtomicNum() == 1) ++hydrogens;
                    if (a->GetAtomicNum() == 6) middle = &*a;
                }
                if (hydrogens != 2 || !middle || middle->IsInRing() ||
                    middle->GetFormalCharge() != 0 || middle->GetExplicitDegree() != 3 ||
                    middle->GetTotalValence() != 4 ||
                    mol.GetBond(&*first, middle)->GetBondOrder() != 2) continue;
                const double long_cc = first->GetDistance(middle);
                if (!(1.44 < long_cc && long_cc < 1.60)) continue;
                double angle_sum = 0;
                bool pyramid = true;
                for (std::size_t i = 0; i < neighbors.size(); ++i)
                    for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                    {
                        const double angle = mol.GetAngle(neighbors[i], &*first, neighbors[j]);
                        if (!(90 < angle && angle < 130)) pyramid = false;
                        angle_sum += angle;
                    }
                if (!pyramid || !(300 < angle_sum && angle_sum < 345)) continue;
                FOR_NBORS_OF_ATOM(last, middle)
                {
                    if (last->GetIdx() == first->GetIdx() || last->GetAtomicNum() != 6 ||
                        last->IsInRing() || last->GetFormalCharge() != -1 ||
                        last->GetExplicitDegree() != 3 || last->GetTotalValence() != 3 ||
                        mol.GetBond(middle, &*last)->GetBondOrder() != 1) continue;
                    const double short_cc = middle->GetDistance(&*last);
                    if (!(1.28 < short_cc && short_cc < 1.40 && long_cc - short_cc > 0.08))
                        continue;
                    bool resolved = true;
                    for (auto *a : {&*first, middle, &*last})
                    {
                        if (a->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*a) ||
                            molgr::utils::GetLonePairCount(*a) ||
                            molgr::utils::HasUnresolvedTwoElectronCenter(*a)) resolved = false;
                        FOR_NBORS_OF_ATOM(n, a)
                            if (n->GetAtomicNum() != 1 && n->GetAtomicNum() != 6) resolved = false;
                    }
                    if (!resolved) continue;
                    mol.GetBond(&*first, middle)->SetBondOrder(1);
                    mol.GetBond(middle, &*last)->SetBondOrder(2);
                    first->SetFormalCharge(-1);
                    last->SetFormalCharge(0);
                    AssignChargeRadicalForAtom(*first);
                    AssignChargeRadicalForAtom(*last);
                    mol.SetAromaticPerceived(false);
                    hit = true;
                    break;
                }
            }
            return hit;
        }

        bool CleanResonances(OBMol &mol)
        {
            bool hit = false;
            hit = CleanResonances11(mol) || hit;
            bool has_anion = false;
            FOR_ATOMS_OF_MOL(atom, mol)
            {
                if (atom->GetFormalCharge() < 0)
                {
                    has_anion = true;
                    break;
                }
            }
            if (has_anion)
            {
                hit = CleanResonances0(mol) || hit;
                hit = CleanResonances1(mol) || hit;
                hit = CleanResonances2(mol) || hit;
                hit = CleanResonances3(mol) || hit;
                hit = CleanResonances4(mol) || hit;
                hit = CleanResonances9(mol) || hit;
                hit = CleanResonances5(mol) || hit;
                hit = CleanResonances6(mol) || hit;
                hit = CleanResonances7(mol) || hit;
                hit = CleanResonances8(mol) || hit;
                hit = CleanResonances9(mol) || hit;
            }
            else
            {
                // Rule 8 refreshes aromatic flags even when it has no matches.
                molgr::vendor::openbabel_threading::SetAromaticPerceived(mol, false);
                molgr::vendor::openbabel_threading::PrepareForSmartsMatching(mol);
            }
            hit = CleanResonances10(mol) || hit;
            hit = CleanResonances12(mol) || hit;
            hit = CleanResonances13(mol) || hit;
            hit = CleanResonances14(mol) || hit;
            hit = CleanResonances16(mol) || hit;
            hit = CleanResonances17(mol) || hit;
            hit = CleanResonances18(mol) || hit;
            hit = CleanShortDiazoResonance(mol) || hit;
            hit = CleanStretchedDithiolateResonance(mol) || hit;
            hit = CleanStretchedSilyloxoniumResonance(mol) || hit;
            hit = CleanPhosphorusCumulene(mol) || hit;
            hit = CleanPyrrolideResonance(mol) || hit;
            hit = CleanPyramidalAllylResonance(mol) || hit;
            return hit;
        }
    }
}
