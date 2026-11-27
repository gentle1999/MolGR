#include "molgr/stages/preprocess.h"

#include "molgr/stages/internal_helpers.h"

#include "molgr/utils/logger.h"
#include "molgr/utils/bond_geometry.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/search_constraints.h"
#include "molgr/utils/smarts.h"
#include "molgr/utils/utils.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <map>
#include <utility>

namespace molgr
{
    namespace reconstruct
    {
        using namespace OpenBabel;

        std::vector<int> GetFlatAtomList(OBMol &mol, molgr::smarts::PatternId pattern_id)
        {
            std::vector<int> atom_indices;
            auto matches = molgr::smarts::FindAll(mol, pattern_id);
            for (const auto &match : matches)
            {
                for (int idx : match)
                    atom_indices.push_back(idx);
            }
            return atom_indices;
        }

        // Electron bookkeeping: fixed atoms contribute only real unpaired
        // electrons to the radical budget. Resolve deferred centers deterministically
        // as triplet (2,0) or singlet (0,1); active lone pairs never count as spin.
        // Reject an impossible budget before mutating any center.
        bool ValidateOmol(OBMol &mol, int total_charge, int total_radical, bool emit_warnings)
        {
            if (!molgr::search_constraints::Satisfied(mol)) return false;
            FOR_BONDS_OF_MOL(bond, mol)
            {
                // Electron budgets alone also admit spurious C--C quadruple
                // bonds. Reject orders outside the nonmetal Lewis model.
                if (bond->GetBondOrder() > 3 &&
                    GetElementInfo(bond->GetBeginAtom()->GetAtomicNum()) != nullptr &&
                    GetElementInfo(bond->GetEndAtom()->GetAtomicNum()) != nullptr)
                    return false;
                if (!molgr::bond_geometry::PyramidalIminiumGeometryAllowed(*bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) ||
                    !molgr::bond_geometry::BentKeteniminiumGeometryAllowed(*bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) ||
                    !molgr::bond_geometry::NitriliumGeometryAllowed(*bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) ||
                    !molgr::bond_geometry::PhosphorusCarbonOrderAllowed(
                        *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) ||
                    !molgr::bond_geometry::SiliconCarbonOrderAllowed(
                        *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) ||
                    !molgr::bond_geometry::CarbonOxygenOrderAllowed(
                        *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) ||
                    !molgr::bond_geometry::NitrogenOxygenOrderAllowed(
                        *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) ||
                    !molgr::bond_geometry::NitrogenNitrogenOrderAllowed(
                        *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()) ||
                    !molgr::bond_geometry::TwistedRingLinkOrderAllowed(
                        *bond->GetBeginAtom(), *bond->GetEndAtom(), bond->GetBondOrder()))
                    return false;
            }
            int charge_sum = 0;
            int radical_sum = 0;
            std::vector<OpenBabel::OBAtom *> unresolved_atoms;

            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                OpenBabel::OBAtom *atom = &(*atom_iter);
                if (molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                {
                    unresolved_atoms.push_back(atom);
                }
                charge_sum += atom->GetFormalCharge();
                if (!molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                {
                    radical_sum += molgr::utils::GetUnpairedElectronCount(*atom);
                }
            }

            // Resolve the specific open-shell anion form ``C-`` + one radical
            // only when both global budgets require it. Generic charge stages
            // intentionally never partially consume an unresolved marker.
            if (total_charge - charge_sum == -1 &&
                !unresolved_atoms.empty() &&
                total_radical - radical_sum == 1)
            {
                auto *atom = unresolved_atoms.front();
                molgr::utils::SetUnresolvedTwoElectronCenter(*atom, false);
                molgr::utils::SetUnpairedElectronCount(*atom, 1);
                molgr::utils::SetLonePairCount(*atom, 0);
                atom->SetFormalCharge(-1);
                --charge_sum;
                ++radical_sum;
                unresolved_atoms.erase(unresolved_atoms.begin());
            }

            if (charge_sum != total_charge)
            {
                if (emit_warnings)
                {
                    LOG_WARN("[Validate] Charge mismatch. Target: " << total_charge << ", Actual: " << charge_sum);
                }
                return false;
            }

            const int required_unpaired_electrons = total_radical - radical_sum;
            if (required_unpaired_electrons < 0 ||
                required_unpaired_electrons % 2 != 0 ||
                required_unpaired_electrons >
                    2 * static_cast<int>(unresolved_atoms.size()))
            {
                if (emit_warnings)
                {
                    LOG_WARN("[Validate] Radical budget cannot resolve "
                             << unresolved_atoms.size()
                             << " two-electron centers. Target: " << total_radical
                             << ", Fixed: " << radical_sum);
                }
                return false;
            }

            const int triplet_center_count = required_unpaired_electrons / 2;
            for (std::size_t index = 0; index < unresolved_atoms.size(); ++index)
            {
                auto &atom = *unresolved_atoms[index];
                molgr::utils::SetUnresolvedTwoElectronCenter(atom, false);
                if (static_cast<int>(index) < triplet_center_count)
                {
                    molgr::utils::SetUnpairedElectronCount(atom, 2);
                    molgr::utils::SetLonePairCount(atom, 0);
                }
                else
                {
                    molgr::utils::SetUnpairedElectronCount(atom, 0);
                    molgr::utils::SetLonePairCount(atom, 1);
                }
            }
            return true;
        }

        using ConnectionRule = std::pair<smarts::PatternId, smarts::PatternId>;

        /// Delete one SMARTS-matched bond longer than 1.15 times the summed covalent radii.
        /// @param mol Molecular graph on which the operation acts.
        /// @param pattern Substructure pattern to match.
        /// @return True when the requested edit was applied to the graph.
        static bool RemoveOneLongSeedBond(OBMol &mol, smarts::PatternId pattern)
        {
            for (const auto &match : smarts::FindAll(mol, pattern))
            {
                OBAtom *first = mol.GetAtom(match[0]);
                OBAtom *second = mol.GetAtom(match[1]);
                const double radii = OBElements::GetCovalentRad(first->GetAtomicNum()) +
                                     OBElements::GetCovalentRad(second->GetAtomicNum());
                if (first->GetDistance(second) > 1.15 * radii)
                    return mol.DeleteBond(mol.GetBond(first, second));
            }
            return false;
        }

        /// Check a prospective sixth P bond for four cis neighbors and one trans neighbor.
        /// @param mol Molecular graph on which the operation acts.
        /// @param center Atom or site occupying this position in the local chemical motif.
        /// @param partner Atom or site occupying this position in the local chemical motif.
        /// @return True for an accessible sixth position with four cis neighbors and one trans neighbor.
        static bool FillsOctahedralVacancy(OBMol &mol, OBAtom &center, OBAtom &partner)
        {
            const double distance = center.GetDistance(&partner);
            int cis = 0, trans = 0;
            bool similar_bond = false;
            FOR_NBORS_OF_ATOM(neighbor, &center)
            {
                if (mol.GetBond(&center, &*neighbor)->GetBondOrder() != 1)
                    return false;
                const double angle = mol.GetAngle(&partner, &center, &*neighbor);
                if (70.0 <= angle && angle <= 110.0) ++cis;
                if (angle >= 160.0) ++trans;
                if (neighbor->GetAtomicNum() == partner.GetAtomicNum() &&
                    std::abs(center.GetDistance(&*neighbor) - distance) <= 0.10)
                    similar_bond = true;
            }
            return cis == 4 && trans == 1 && similar_bond;
        }

        /// Reconnect the first eligible donor to its nearest compatible partner within the covalent cutoff.
        /// @param mol Molecular graph on which the operation acts.
        /// @param rules Bond or atom matching rules used at this stage.
        /// @param extra_tolerance_angstrom Additional distance allowed beyond the covalent cutoff, in
        ///     angstroms.
        /// @param geometry_allowed Predicate validating the geometry of a proposed connection.
        /// @return True when the requested edit was applied to the graph.
        static bool TryAddOneConnection(OBMol &mol, std::initializer_list<ConnectionRule> rules,
                                        double extra_tolerance_angstrom,
                                        bool (*geometry_allowed)(OBMol &, OBAtom &, OBAtom &) = nullptr)
        {
            // Each donor matches one rule per group. Atom order gives donors first choice.
            std::map<int, std::vector<int>> partners_by_donor;
            for (const auto &[donor_pattern, partner_pattern] : rules)
            {
                const auto donors = GetFlatAtomList(mol, donor_pattern);
                if (donors.empty())
                    continue;
                const auto partners = GetFlatAtomList(mol, partner_pattern);
                for (int donor : donors)
                    partners_by_donor.emplace(donor, partners);
            }

            for (auto &[donor_idx, partners] : partners_by_donor)
            {
                OBAtom *donor = mol.GetAtom(donor_idx);
                partners.erase(std::remove(partners.begin(), partners.end(), donor_idx), partners.end());
                std::sort(partners.begin(), partners.end(), [&](int a, int b)
                          { return donor->GetDistance(mol.GetAtom(a)) < donor->GetDistance(mol.GetAtom(b)); });

                const double donor_radius = OBElements::GetCovalentRad(donor->GetAtomicNum());
                for (int partner_idx : partners)
                {
                    OBAtom *partner = mol.GetAtom(partner_idx);
                    const double partner_radius = OBElements::GetCovalentRad(partner->GetAtomicNum());
                    if (donor->GetDistance(partner) >= donor_radius + partner_radius + extra_tolerance_angstrom)
                        continue;
                    OBBond *bond = mol.GetBond(donor, partner);
                    if (bond && bond->GetBondOrder() != 0)
                        continue;
                    if (geometry_allowed && !geometry_allowed(mol, *donor, *partner))
                        continue;
                    if (bond)
                        bond->SetBondOrder(1);
                    else
                        mol.AddBond(donor_idx, partner_idx, 1);
                    LOG_DEBUG("[MakeConnections] Set single bond " << donor_idx << "-" << partner_idx);
                    return true;
                }
            }
            return false;
        }

        bool MakeConnections(OBMol &mol, double extra_tolerance_angstrom)
        {
            using Id = smarts::PatternId;
            const std::initializer_list<ConnectionRule> octahedral_phosphorus = {
                {Id::CONNECTION_P_FIVE_COORDINATE, Id::CONNECTION_P_SIXTH_PARTNER},
            };
            const std::initializer_list<ConnectionRule> primary = {
                {Id::CONNECTION_GENERAL_DONOR, Id::CONNECTION_GENERAL_PARTNER},
                {Id::CONNECTION_P_V2, Id::CONNECTION_P_PARTNER},
            };
            const std::initializer_list<ConnectionRule> secondary = {
                {Id::CONNECTION_O_V1, Id::CONNECTION_C_V3},
                {Id::CONNECTION_SE_I_DONOR, Id::CONNECTION_O_PARTNER},
            };

            bool changed = false;
            // Rematch after deletion: bonds sharing an endpoint may no longer qualify.
            for (auto pattern : {Id::LONG_SI_CARBONYL_BOND, Id::LONG_TETRACOORDINATE_P_P_BOND})
                while (RemoveOneLongSeedBond(mol, pattern))
                    changed = true;

            // The geometry check permits a further 0.10 Å for the sixth P bond.
            // Rematch after each addition so both endpoints use their current valences.
            while (TryAddOneConnection(mol, octahedral_phosphorus, extra_tolerance_angstrom + 0.10,
                                       FillsOctahedralVacancy) ||
                   TryAddOneConnection(mol, primary, extra_tolerance_angstrom) ||
                   TryAddOneConnection(mol, secondary, extra_tolerance_angstrom))
                changed = true;
            return changed;
        }

        /// Remove one overlong C–N contact across a four-membered ring.
        /// @param mol Seed graph with coordinates in angstroms. The SMARTS lists
        ///     bridgehead C, outer C, bridgehead N, outer C in that order.
        /// @return True if the shared edge exceeds both the covalent radii sum
        ///     and the longest perimeter bond by 15%. The local comparison
        ///     protects stretched covalent bridges in strained bicyclic systems.
        static bool RemoveOneLongRingDiagonal(OBMol &mol)
        {
            for (const auto &match : smarts::FindAll(mol, smarts::PatternId::PRE_CLEAN_BCP_RING_4))
            {
                auto *carbon = mol.GetAtom(match[0]);
                auto *first = mol.GetAtom(match[1]);
                auto *nitrogen = mol.GetAtom(match[2]);
                auto *second = mol.GetAtom(match[3]);
                const double radii = OBElements::GetCovalentRad(carbon->GetAtomicNum()) +
                                     OBElements::GetCovalentRad(nitrogen->GetAtomicNum());
                const double longest_side = std::max({
                    carbon->GetDistance(first), carbon->GetDistance(second),
                    nitrogen->GetDistance(first), nitrogen->GetDistance(second)});
                if (carbon->GetDistance(nitrogen) > 1.15 * std::max(radii, longest_side))
                    return mol.DeleteBond(mol.GetBond(carbon, nitrogen));
            }
            return false;
        }

        bool PreClean(OBMol &mol)
        {
            bool hit = false;
            while (true)
            {
                auto matches1 = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::PRE_CLEAN_HYPERVALENT);
                if (matches1.empty())
                {
                    break;
                }
                const auto &match = matches1.front();
                OBBond *bond = mol.GetBond(match[0], match[1]);
                if (bond)
                {
                    bond->SetBondOrder(bond->GetBondOrder() - 1);
                    hit = true;
                }
            }

            /// Test for an isolated neutral carbon monoxide or monosulfide bond.
            const auto neutral_carbon_monochalcogenide = [](OBBond &bond)
            {
                auto *a = bond.GetBeginAtom();
                auto *b = bond.GetEndAtom();
                /// Select oxygen or sulfur for this two-atom motif.
                const auto chalcogen = [](int z) { return z == 8 || z == 16; };
                return ((a->GetAtomicNum() == 6 && chalcogen(b->GetAtomicNum())) ||
                        (b->GetAtomicNum() == 6 && chalcogen(a->GetAtomicNum()))) &&
                       a->GetTotalDegree() == 1 && b->GetTotalDegree() == 1 &&
                       a->GetFormalCharge() == 0 && b->GetFormalCharge() == 0;
            };
            // A single pass allows isolated CS to retain the same precursor
            // as CO without changing cleanup of substituted sulfur motifs.
            for (const auto &match : molgr::smarts::FindAll(
                     mol, molgr::smarts::PatternId::PRE_CLEAN_HYPER_PI_BOND))
            {
                OBBond *bond = mol.GetBond(match[0], match[1]);
                if (bond && !neutral_carbon_monochalcogenide(*bond))
                {
                    bond->SetBondOrder(1);
                    hit = true;
                }
            }
            FOR_BONDS_OF_MOL(bond, mol)
            {
                if (neutral_carbon_monochalcogenide(*bond) &&
                    (bond->GetBondOrder() == 1 || bond->GetBondOrder() == 3))
                {
                    // Fresh inference marks C=X as an unresolved center; the
                    // shared heteroatom donor rule subsequently yields C-#X+.
                    bond->SetBondOrder(2);
                    hit = true;
                }
            }

            while (true)
            {
                auto matches3 = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::PRE_CLEAN_BCP_RING_5);
                if (matches3.empty())
                {
                    break;
                }
                const auto &idxs = matches3.front();
                int n_idx = -1, c_idx = -1;
                for (int idx : idxs)
                {
                    OBAtom *atom = mol.GetAtom(idx);
                    int internal_degree = 0;
                    for (int other : idxs)
                        if (idx != other && mol.GetBond(idx, other))
                            internal_degree++;

                    if (internal_degree >= 3)
                    {
                        if (atom->GetAtomicNum() == 7)
                            n_idx = idx;
                        if (atom->GetAtomicNum() == 6)
                            c_idx = idx;
                    }
                }
                if (n_idx != -1 && c_idx != -1)
                {
                    OBBond *bond = mol.GetBond(n_idx, c_idx);
                    if (bond)
                    {
                        mol.DeleteBond(bond);
                        hit = true;
                    }
                }
            }

            while (RemoveOneLongRingDiagonal(mol))
                hit = true;

            return hit;
        }
    }
}
