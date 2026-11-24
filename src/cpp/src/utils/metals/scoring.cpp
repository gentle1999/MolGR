#include "molgr/utils/metals/hydrazido_geometry.h"
#include "molgr/utils/metals/scoring.h"
#include "molgr/utils/metals/conjugated_geometry.h"

#include "molgr/utils/coordination.h"

#include "molgr/utils/consts.h"
#include "molgr/utils/bond_geometry.h"
#include "molgr/utils/coordination_geometry.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/organic_topology.h"
#include "molgr/utils/no_metals/selection.h"
#include "molgr/utils/scoring.h"
#include "molgr/stages/fresh.h"
#include "molgr/vendor/openbabel_copy.h"
#include "molgr/vendor/openbabel_threading.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include <openbabel/mol.h>
#include <openbabel/obfunctions.h>
#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace
{
    using molgr::coordination::Point3D;
    using molgr::coordination::BuildCoordinationBlockers;
    using molgr::coordination::IsInnerSphereAtom;
    using molgr::coordination::IsVisibleToMetalAtom;

    /// Ligand-only aromaticity, conjugation and electron-localization terms reused across metal
    /// assignments.
    struct OrganicElectronicStateMetrics
    {
        int aromatic_atom_count = 0;
        int aromatic_ring_count = 0;
        double aromatic_stability_score = 0.0;
        int conjugated_atom_count = 0;
        int conjugated_bond_count = 0;
        int max_conjugated_component_size = 0;
        int hyperconjugative_donor_count = 0;
        int hyperconjugation_score = 0;
        int singlet_two_electron_center_count = 0;
        double radical_localization_penalty = 0.0;
        double charge_localization_penalty = 0.0;
        double charge_localization_component_cancellation = 0.0;
        double charge_localization_polarity_inversion_penalty = 0.0;
    };

    /// Negative-metal count and chemical exceptions used in the metal discordance penalty.
    struct NegativeMetalDiscordance
    {
        int count = 0;
        bool has_outer_sphere_cation_exception = false;
        bool has_positive_metal_counterion_exception = false;
    };

    constexpr double kNegativeMetalDiscordancePenalty = 0.5;
    // A retained donor pair reduces cation repulsion; it does not make it neutral.
    constexpr double kCationicDonorDiscordancePenalty = 0.5;
    constexpr double kMinimumRingAlleneAngleDegrees = 150.0;
    constexpr double kMinimumChargePolarityInversionElectronegativityGap = 0.3;
    constexpr int kMaxChargeLocalizationReferenceOxidationStateDelta = 2;
    constexpr std::uint64_t kConnectivityHashOffset = 1469598103934665603ULL;
    constexpr std::uint64_t kConnectivityHashPrime = 1099511628211ULL;

    /// @param values Element-indexed table of scoring coefficients.
    /// @param atomic_num Atomic number of the element.
    /// @return The element-specific score coefficient when the lookup table contains this atomic number.
    std::optional<double> LookupMapValue(
        const std::map<int, double> &values,
        int atomic_num)
    {
        const auto it = values.find(atomic_num);
        if (it == values.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    /// Scale the clamped value to the unit interval, using the fallback for a degenerate interval.
    /// @param value Numeric value to clamp and scale.
    /// @param lower Boundary of the normalization interval.
    /// @param upper Boundary of the normalization interval.
    /// @param fallback Value used when the requested information is unavailable.
    /// @return Normalized value in [0,1], or the fallback for invalid bounds.
    double NormalizeValue(
        const std::optional<double> &value,
        double lower,
        double upper,
        double fallback)
    {
        if (!value.has_value() || upper <= lower)
        {
            return fallback;
        }
        const double clamped = std::min(std::max(*value, lower), upper);
        return (clamped - lower) / (upper - lower);
    }

    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when an inner-sphere atom carries at least two explicit unpaired electrons.
    bool IsInnerVisibleDiradicalDiscordanceAtom(const OpenBabel::OBAtom &atom)
    {
        return static_cast<int>(molgr::utils::GetUnpairedElectronCount(atom)) >= 2;
    }

    /// Return whether a resolved C/N/Si/P lone pair fills a two-unit valence deficit.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True if the atom has the required resolved electron and bonding pattern.
    bool IsExplicitSingletTwoElectronCenter(OpenBabel::OBAtom &atom)
    {
        const int atomic_num = atom.GetAtomicNum();
        return (atomic_num == 6 || atomic_num == 7 || atomic_num == 14 || atomic_num == 15) &&
               atom.GetFormalCharge() == 0 &&
               molgr::utils::GetUnpairedElectronCount(atom) == 0 &&
               molgr::utils::GetLonePairCount(atom) == 1 &&
               !molgr::utils::HasUnresolvedTwoElectronCenter(atom) &&
               molgr::reconstruct::AssignRadicalDots(atom) == 2;
    }

    /// Recognize a neutral cumulene carbon with two double bonds to neutral carbon neighbors.
    /// @param center Atom or site occupying this position in the local chemical motif.
    /// @return True if the atom has the required resolved electron and bonding pattern.
    bool IsNeutralCarbonCumuleneCenter(OpenBabel::OBAtom &center)
    {
        if (center.GetAtomicNum() != 6 || center.GetFormalCharge() != 0) return false;
        int double_bonds = 0;
        FOR_BONDS_OF_ATOM(bond, &center)
        {
            if (bond->GetBondOrder() != 2 ||
                molgr::vendor::openbabel_threading::BondIsAromatic(*bond)) continue;
            auto *neighbor = bond->GetNbrAtom(&center);
            if (neighbor->GetAtomicNum() != 6 || neighbor->GetFormalCharge() != 0) return false;
            ++double_bonds;
        }
        return double_bonds == 2;
    }

    /// Identify ring carbons with two ring double bonds forming a bent cumulated center.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @return Indices of bent cumulated allene centers in rings.
    std::set<int> BentCumulatedRingAlleneIndices(OpenBabel::OBMol &mol)
    {
        std::set<int> indices;
        FOR_ATOMS_OF_MOL(center_iter, mol)
        {
            OpenBabel::OBAtom &center = *center_iter;
            if (!center.IsInRing()) continue;
            std::vector<OpenBabel::OBAtom *> ring_double_neighbors;
            FOR_BONDS_OF_ATOM(bond_iter, &center)
            {
                OpenBabel::OBBond &bond = *bond_iter;
                if (molgr::vendor::openbabel_threading::BondIsAromatic(bond) ||
                    bond.GetBondOrder() != 2)
                {
                    continue;
                }
                OpenBabel::OBAtom *neighbor = bond.GetNbrAtom(&center);
                if (neighbor != nullptr && neighbor->IsInRing())
                {
                    ring_double_neighbors.push_back(neighbor);
                }
            }
            bool is_bent = false;
            for (std::size_t left_index = 0;
                 left_index < ring_double_neighbors.size() && !is_bent;
                 ++left_index)
            {
                for (std::size_t right_index = left_index + 1;
                     right_index < ring_double_neighbors.size();
                     ++right_index)
                {
                    const double angle = mol.GetAngle(
                        ring_double_neighbors[left_index],
                        &center,
                        ring_double_neighbors[right_index]);
                    if (std::isfinite(angle) && angle < kMinimumRingAlleneAngleDegrees)
                    {
                        is_bent = true;
                        break;
                    }
                }
            }
            if (is_bent)
            {
                indices.insert(center.GetIdx());
            }
        }
        return indices;
    }

    /// Test whether an alternative electronic assignment recovers aromaticity at a bent ring allene.
    /// @param source Molecule supplying the initial topology and electronic labels.
    /// @param index One-based molecular atom index.
    /// @return True if an aromatic resonance partner removes the targeted bent center.
    bool AromaticRingAlleneResonanceWitness(OpenBabel::OBMol &source, int index)
    {
        /// Check that the proposed allene carbon has explicit hydrogen and resolved electron labels.
        const auto resolved_carbon = [](OpenBabel::OBAtom *atom) {
            return atom != nullptr && atom->GetAtomicNum() == 6 &&
                   atom->GetImplicitHCount() == 0 &&
                   molgr::utils::GetUnpairedElectronCount(*atom) == 0 &&
                   !molgr::utils::HasUnresolvedTwoElectronCenter(*atom);
        };
        auto *center = source.GetAtom(index);
        if (!resolved_carbon(center) || center->GetFormalCharge() != 0 ||
            center->GetTotalValence() != 4) return false;

        // A bent contributor is not a distinct electronic state when an
        // alternating carbon path can move an existing carbanion onto it and
        // restore aromaticity. Require an explicit, charge-preserving witness;
        // neither another fragment's charge nor a truncated search is evidence.
        const auto initial_bent = BentCumulatedRingAlleneIndices(source);
        if (!initial_bent.count(index)) return false;
        std::deque<std::vector<int>> pending{{index}};
        std::size_t visited = 0;
        while (!pending.empty() && ++visited <= 4096)
        {
            auto path = std::move(pending.front());
            pending.pop_front();
            if (path.size() >= 32) continue;
            auto *last = source.GetAtom(path.back());
            const unsigned int order = path.size() % 2 ? 2 : 1;
            FOR_BONDS_OF_ATOM(bond, last)
            {
                auto *other = bond->GetNbrAtom(last);
                const int next = other->GetIdx();
                if (bond->GetBondOrder() != order || !resolved_carbon(other) ||
                    std::find(path.begin(), path.end(), next) != path.end()) continue;
                const bool endpoint = order == 1 && other->GetFormalCharge() == -1 &&
                                      other->GetTotalValence() == 3;
                const bool internal = other->GetFormalCharge() == 0 &&
                                      other->GetTotalValence() == 4;
                if (!endpoint && !internal) continue;
                auto extended = path;
                extended.push_back(next);
                if (!endpoint)
                {
                    pending.push_back(std::move(extended));
                    continue;
                }
                auto witness = molgr::utils::CloneMolTopologyShared(source);
                witness->GetAtom(index)->SetFormalCharge(-1);
                witness->GetAtom(next)->SetFormalCharge(0);
                for (std::size_t i = 1; i < extended.size(); ++i)
                    witness->GetBond(extended[i - 1], extended[i])->SetBondOrder(i % 2 ? 1 : 2);
                molgr::vendor::openbabel_threading::ResetAndAssignAromaticFlags(*witness);
                if (!molgr::vendor::openbabel_threading::AtomIsAromatic(*witness->GetAtom(index)))
                    continue;
                const auto remaining_bent = BentCumulatedRingAlleneIndices(*witness);
                if (!remaining_bent.count(index) &&
                    std::includes(initial_bent.begin(), initial_bent.end(),
                                  remaining_bent.begin(), remaining_bent.end())) return true;
            }
        }
        return false;
    }

    /// @param bond Bond whose endpoints and order define the local environment.
    /// @return The bond-order encoding used to compare aromatic and localized ligand bonds.
    int BondOrder(const OpenBabel::OBBond &bond)
    {
        return molgr::vendor::openbabel_threading::BondIsAromatic(
                   const_cast<OpenBabel::OBBond &>(bond))
                   ? 2
                   : static_cast<int>(bond.GetBondOrder());
    }

    /// @param bond Bond whose endpoints and order define the local environment.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return The opposite endpoint of a bond incident to the supplied atom.
    OpenBabel::OBAtom *OtherBondAtom(OpenBabel::OBBond &bond, OpenBabel::OBAtom &atom)
    {
        return bond.GetNbrAtom(&atom);
    }

    /// @param lhs First atom.
    /// @param rhs Second atom.
    /// @return The bond joining the atoms, or null when they are not adjacent.
    OpenBabel::OBBond *BondBetweenAtoms(OpenBabel::OBAtom &lhs, OpenBabel::OBAtom &rhs)
    {
        const int rhs_idx = static_cast<int>(rhs.GetIdx());
        FOR_BONDS_OF_ATOM(bond_iter, &lhs)
        {
            OpenBabel::OBBond &bond = *bond_iter;
            OpenBabel::OBAtom *other = OtherBondAtom(bond, lhs);
            if (other != nullptr && static_cast<int>(other->GetIdx()) == rhs_idx)
            {
                return &bond;
            }
        }
        return nullptr;
    }

    /// Group molecular atom indices by connected component.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @return Atom indices grouped by covalent connectivity.
    std::vector<std::vector<int>> ComponentAtomIndexGroups(OpenBabel::OBMol &mol)
    {
        std::set<int> unseen;
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            unseen.insert(static_cast<int>(atom_iter->GetIdx()));
        }

        std::vector<std::vector<int>> components;
        while (!unseen.empty())
        {
            const int start_idx = *unseen.begin();
            unseen.erase(start_idx);
            std::vector<int> stack{start_idx};
            std::vector<int> component{start_idx};
            while (!stack.empty())
            {
                const int atom_idx = stack.back();
                stack.pop_back();
                OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                if (atom == nullptr)
                {
                    continue;
                }
                FOR_BONDS_OF_ATOM(bond_iter, atom)
                {
                    OpenBabel::OBAtom *neighbor = OtherBondAtom(*bond_iter, *atom);
                    if (neighbor == nullptr)
                    {
                        continue;
                    }
                    const int neighbor_idx = static_cast<int>(neighbor->GetIdx());
                    const auto unseen_it = unseen.find(neighbor_idx);
                    if (unseen_it == unseen.end())
                    {
                        continue;
                    }
                    unseen.erase(unseen_it);
                    stack.push_back(neighbor_idx);
                    component.push_back(neighbor_idx);
                }
            }
            std::sort(component.begin(), component.end());
            components.push_back(std::move(component));
        }
        return components;
    }

    /// Hash a sequence of connectivity labels for component comparison.
    /// @param values Ordered integer labels of the component connectivity.
    /// @return Hash used to compare the component labels.
    std::uint64_t ConnectivityHash(const std::vector<std::uint64_t> &values)
    {
        std::uint64_t hash = kConnectivityHashOffset;
        for (const std::uint64_t value : values)
        {
            hash ^= value;
            hash *= kConnectivityHashPrime;
        }
        return hash;
    }

    /// Return a connectivity fingerprint for identifying repeated ligand components.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param atom_indices Atom indices defining the selected part of the molecular graph.
    /// @return Component connectivity labels in canonical comparison order.
    std::vector<std::uint64_t> ComponentConnectivitySignature(
        OpenBabel::OBMol &mol,
        const std::vector<int> &atom_indices)
    {
        std::map<int, std::uint64_t> labels;
        for (const int atom_idx : atom_indices)
        {
            OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
            labels[atom_idx] =
                atom == nullptr ? 0U : static_cast<std::uint64_t>(atom->GetAtomicNum());
        }
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            std::map<int, std::uint64_t> next_labels;
            for (const int atom_idx : atom_indices)
            {
                OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                if (atom == nullptr)
                {
                    next_labels[atom_idx] = 0U;
                    continue;
                }
                std::vector<std::uint64_t> neighbor_labels;
                FOR_BONDS_OF_ATOM(bond_iter, atom)
                {
                    OpenBabel::OBAtom *neighbor = OtherBondAtom(*bond_iter, *atom);
                    if (neighbor != nullptr)
                    {
                        neighbor_labels.push_back(labels[static_cast<int>(neighbor->GetIdx())]);
                    }
                }
                std::sort(neighbor_labels.begin(), neighbor_labels.end());
                std::vector<std::uint64_t> values{
                    labels[atom_idx],
                    static_cast<std::uint64_t>(neighbor_labels.size())};
                values.insert(values.end(), neighbor_labels.begin(), neighbor_labels.end());
                next_labels[atom_idx] = ConnectivityHash(values);
            }
            labels = std::move(next_labels);
        }

        std::vector<std::uint64_t> signature;
        signature.reserve(atom_indices.size());
        for (const int atom_idx : atom_indices)
        {
            signature.push_back(labels[atom_idx]);
        }
        std::sort(signature.begin(), signature.end());
        return signature;
    }

    /// Count charge asymmetries among ligand components with the same connectivity fingerprint.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @return Number of matching electronic or coordination conflicts.
    int RepeatedComponentChargeAsymmetryCount(OpenBabel::OBMol &mol)
    {
        std::map<std::vector<std::uint64_t>, std::vector<int>> charges_by_connectivity;
        for (const auto &atom_indices : ComponentAtomIndexGroups(mol))
        {
            int total_charge = 0;
            for (const int atom_idx : atom_indices)
            {
                OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                if (atom != nullptr)
                {
                    total_charge += static_cast<int>(atom->GetFormalCharge());
                }
            }
            charges_by_connectivity[ComponentConnectivitySignature(mol, atom_indices)]
                .push_back(total_charge);
        }

        int count = 0;
        for (const auto &entry : charges_by_connectivity)
        {
            const auto &charges = entry.second;
            if (charges.size() > 1 &&
                *std::min_element(charges.begin(), charges.end()) !=
                    *std::max_element(charges.begin(), charges.end()))
            {
                ++count;
            }
        }
        return count;
    }

    /// Count localized anionic carbons in haptically reduced carbon rings.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param visible_atoms_by_metal Ligand atoms visible from each metal site, grouped by metal.
    /// @return Total number of anionic carbons counted on qualifying haptically bound rings.
    int HapticAreneReductionCount(
        OpenBabel::OBMol &mol,
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal)
    {
        std::vector<std::set<int>> visible_carbon_indices_by_metal;
        visible_carbon_indices_by_metal.reserve(visible_atoms_by_metal.size());
        for (const auto &visible_atoms : visible_atoms_by_metal)
        {
            std::set<int> visible_indices;
            for (OpenBabel::OBAtom *atom : visible_atoms)
            {
                if (atom != nullptr && atom->GetAtomicNum() == 6)
                {
                    visible_indices.insert(static_cast<int>(atom->GetIdx()));
                }
            }
            visible_carbon_indices_by_metal.push_back(std::move(visible_indices));
        }

        /// Check whether the localized ring bonds and charged carbons account for a complete cyclic pi
        /// system.
        const auto is_complete_kekule_pi_ring = [&](const std::vector<int> &ring_atom_indices)
        {
            std::vector<OpenBabel::OBAtom *> ring_atoms;
            std::vector<OpenBabel::OBBond *> ring_bonds;
            ring_atoms.reserve(ring_atom_indices.size());
            ring_bonds.reserve(ring_atom_indices.size());
            for (std::size_t offset = 0; offset < ring_atom_indices.size(); ++offset)
            {
                const int begin_idx = ring_atom_indices[offset];
                const int end_idx = ring_atom_indices[(offset + 1) % ring_atom_indices.size()];
                OpenBabel::OBAtom *atom = mol.GetAtom(begin_idx);
                OpenBabel::OBBond *bond = mol.GetBond(begin_idx, end_idx);
                if (atom == nullptr || bond == nullptr ||
                    (bond->GetBondOrder() != 1 && bond->GetBondOrder() != 2))
                {
                    return false;
                }
                ring_atoms.push_back(atom);
                ring_bonds.push_back(bond);
            }
            bool all_aromatic = true;
            for (std::size_t index = 0; index < ring_atom_indices.size(); ++index)
            {
                if (!molgr::vendor::openbabel_threading::AtomIsAromatic(*ring_atoms[index]) ||
                    !molgr::vendor::openbabel_threading::BondIsAromatic(*ring_bonds[index]))
                {
                    all_aromatic = false;
                    break;
                }
            }
            if (all_aromatic)
            {
                return true;
            }

            std::set<std::size_t> pi_bond_indices;
            for (std::size_t index = 0; index < ring_bonds.size(); ++index)
            {
                if (ring_bonds[index]->GetBondOrder() == 2)
                {
                    pi_bond_indices.insert(index);
                }
            }
            for (const std::size_t index : pi_bond_indices)
            {
                if (pi_bond_indices.find((index + 1) % ring_atom_indices.size()) !=
                    pi_bond_indices.end())
                {
                    return false;
                }
            }
            std::vector<int> pi_edge_count_by_atom(ring_atom_indices.size(), 0);
            for (const std::size_t index : pi_bond_indices)
            {
                ++pi_edge_count_by_atom[index];
                ++pi_edge_count_by_atom[(index + 1) % ring_atom_indices.size()];
            }
            if (std::any_of(
                    pi_edge_count_by_atom.begin(),
                    pi_edge_count_by_atom.end(),
                    [](const int count) { return count > 1; }))
            {
                return false;
            }
            std::vector<std::size_t> missing_pi_atoms;
            for (std::size_t index = 0; index < pi_edge_count_by_atom.size(); ++index)
            {
                if (pi_edge_count_by_atom[index] == 0)
                {
                    missing_pi_atoms.push_back(index);
                }
            }
            if (ring_atom_indices.size() % 2 == 0)
            {
                return missing_pi_atoms.empty() &&
                       pi_bond_indices.size() == ring_atom_indices.size() / 2;
            }
            return missing_pi_atoms.size() == 1 &&
                   pi_bond_indices.size() == ring_atom_indices.size() / 2 &&
                   ring_atoms[missing_pi_atoms.front()]->GetAtomicNum() == 6 &&
                   ring_atoms[missing_pi_atoms.front()]->GetFormalCharge() < 0;
        };

        int count = 0;
        // Match Python's GetSSSR traversal. Open Babel 3.1's ring iterator
        // can leave its current pointer uninitialized when ring data is absent.
        for (OpenBabel::OBRing *ring : mol.GetSSSR())
        {
            if (ring == nullptr || ring->_path.size() < 5 || ring->_path.size() > 6)
            {
                continue;
            }
            bool all_carbon = true;
            int negative_count = 0;
            for (const int atom_idx : ring->_path)
            {
                OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                if (atom == nullptr || atom->GetAtomicNum() != 6 ||
                    atom->GetExplicitDegree() + atom->GetImplicitHCount() > 3)
                {
                    all_carbon = false;
                    break;
                }
                if (atom->GetFormalCharge() < 0)
                {
                    ++negative_count;
                }
            }
            if (!all_carbon || negative_count == 0)
            {
                continue;
            }
            bool haptically_visible = false;
            for (const auto &visible_indices : visible_carbon_indices_by_metal)
            {
                int visible_count = 0;
                for (const int atom_idx : ring->_path)
                {
                    visible_count += visible_indices.find(atom_idx) != visible_indices.end();
                }
                if (visible_count >= 3)
                {
                    haptically_visible = true;
                    break;
                }
            }
            if (!haptically_visible || is_complete_kekule_pi_ring(ring->_path))
            {
                continue;
            }
            count += negative_count;
        }
        return count;
    }

    /// Test for a conjugated path connecting the two charged carbon centers.
    /// @param begin_atom Endpoint atom of the bond or conjugated path.
    /// @param end_atom Endpoint atom of the bond or conjugated path.
    /// @return True when conjugation links the two charged endpoints.
    bool HasConjugatedBridgeBetweenChargedCarbons(
        OpenBabel::OBAtom &begin_atom,
        OpenBabel::OBAtom &end_atom)
    {
        FOR_BONDS_OF_ATOM(begin_bond_iter, &begin_atom)
        {
            OpenBabel::OBBond &begin_bond = *begin_bond_iter;
            if (BondOrder(begin_bond) != 1)
            {
                continue;
            }
            OpenBabel::OBAtom *begin_neighbor = OtherBondAtom(begin_bond, begin_atom);
            if (begin_neighbor == nullptr || begin_neighbor->GetAtomicNum() == 1)
            {
                continue;
            }
            FOR_BONDS_OF_ATOM(middle_bond_iter, begin_neighbor)
            {
                OpenBabel::OBBond &middle_bond = *middle_bond_iter;
                if (static_cast<int>(middle_bond.GetIdx()) == static_cast<int>(begin_bond.GetIdx()) ||
                    BondOrder(middle_bond) != 2)
                {
                    continue;
                }
                OpenBabel::OBAtom *middle_neighbor = OtherBondAtom(middle_bond, *begin_neighbor);
                if (middle_neighbor == nullptr ||
                    static_cast<int>(middle_neighbor->GetIdx()) == static_cast<int>(begin_atom.GetIdx()))
                {
                    continue;
                }
                OpenBabel::OBBond *bridge_end_bond = BondBetweenAtoms(*middle_neighbor, end_atom);
                if (bridge_end_bond != nullptr && BondOrder(*bridge_end_bond) == 1)
                {
                    return true;
                }
            }
        }
        return false;
    }

    /// Test whether the metal-facing carbanion has the pyramidal geometry of a sigma donor.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param metal Metal site being considered.
    /// @return True for a pyramidal sigma donor oriented toward the metal.
    bool IsPyramidalSigmaCarbanion(
        OpenBabel::OBAtom &atom, const molgr::MetalAtomPosition &metal)
    {
        // A pyramidal, three-coordinate C- directed toward the metal uses its
        // lone pair in a sigma bond. A formal C-C=C-C path alone does not make
        // two such donors an unfavorable conjugated charge pair.
        if (atom.GetFormalCharge() != -1 || atom.GetExplicitDegree() != 3 ||
            atom.GetImplicitHCount() != 0 || molgr::utils::GetUnpairedElectronCount(atom) ||
            molgr::utils::HasUnresolvedTwoElectronCenter(atom)) return false;
        std::vector<OpenBabel::OBAtom *> neighbors;
        double vx = 0, vy = 0, vz = 0;
        FOR_NBORS_OF_ATOM(n, &atom)
        {
            if (atom.GetBond(&*n)->GetBondOrder() != 1) return false;
            const double dx = n->GetX() - atom.GetX(), dy = n->GetY() - atom.GetY(),
                         dz = n->GetZ() - atom.GetZ();
            const double norm = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (!(norm > 1e-8) || !std::isfinite(norm)) return false;
            vx -= dx/norm; vy -= dy/norm; vz -= dz/norm;
            neighbors.push_back(&*n);
        }
        double angle_sum = 0;
        for (std::size_t i = 0; i < neighbors.size(); ++i)
            for (std::size_t j = i + 1; j < neighbors.size(); ++j)
            {
                const double angle = atom.GetParent()->GetAngle(neighbors[i], &atom, neighbors[j]);
                if (!std::isfinite(angle) || !(angle > 0)) return false;
                angle_sum += angle;
            }
        if (!(angle_sum < 350)) return false;
        const double mx = metal.position_x - atom.GetX(), my = metal.position_y - atom.GetY(),
                     mz = metal.position_z - atom.GetZ();
        const double denominator = std::sqrt((vx*vx+vy*vy+vz*vz)*(mx*mx+my*my+mz*mz));
        // Use the vacant direction, allowing a strained metallacycle's small
        // M-C-C angle. Planar donors and approach from the opposite face fail.
        return denominator > 1e-8 && std::isfinite(denominator) &&
            (vx*mx+vy*my+vz*mz)/denominator > std::cos(std::acos(-1.0)/6);
    }

    /// Count visible charged-carbon pairs connected by conjugation for the metal ranking penalty.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param visible_inner_atom_indices Indices of ligand atoms accessible from the metal under the
    ///     visibility test.
    /// @param visible_atoms_by_metal Ligand atoms visible from each metal site, grouped by metal.
    /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
    /// @return Number of matching electronic or coordination conflicts.
    int InnerVisibleConjugatedChargedCarbonPairCount(
        OpenBabel::OBMol &mol,
        const std::set<int> &visible_inner_atom_indices,
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal,
        const std::vector<molgr::MetalAtomPosition> &metal_states)
    {
        std::vector<OpenBabel::OBAtom *> charged_carbons;
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom &atom = *atom_iter;
            if (visible_inner_atom_indices.find(static_cast<int>(atom.GetIdx())) ==
                    visible_inner_atom_indices.end() ||
                atom.GetAtomicNum() != 6 ||
                atom.GetFormalCharge() == 0)
            {
                continue;
            }
            charged_carbons.push_back(&atom);
        }

        int count = 0;
        for (std::size_t i = 0; i < charged_carbons.size(); ++i)
        {
            for (std::size_t j = i + 1; j < charged_carbons.size(); ++j)
            {
                if ((charged_carbons[i]->GetFormalCharge() > 0) !=
                    (charged_carbons[j]->GetFormalCharge() > 0))
                {
                    continue;
                }
                if (BondBetweenAtoms(*charged_carbons[i], *charged_carbons[j]) != nullptr)
                {
                    continue;
                }
                if (HasConjugatedBridgeBetweenChargedCarbons(*charged_carbons[i], *charged_carbons[j]))
                {
                    bool sigma_chelate = false;
                    for (std::size_t k = 0; k < std::min(visible_atoms_by_metal.size(), metal_states.size()); ++k)
                    {
                        const auto &visible = visible_atoms_by_metal[k];
                        if (std::find(visible.begin(), visible.end(), charged_carbons[i]) != visible.end() &&
                            std::find(visible.begin(), visible.end(), charged_carbons[j]) != visible.end() &&
                            IsPyramidalSigmaCarbanion(*charged_carbons[i], metal_states[k]) &&
                            IsPyramidalSigmaCarbanion(*charged_carbons[j], metal_states[k]))
                            sigma_chelate = true;
                    }
                    count += !sigma_chelate;
                }
            }
        }
        return count;
    }

    /// Estimate the cost of localizing the indicated electronic defect on this element and geometry.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param is_conjugated Whether the atom belongs to the conjugated ligand network.
    /// @return Numeric localization cost used in candidate ranking.
    double ChargeLocalizationPenaltyForAtom(
        OpenBabel::OBAtom &atom,
        bool is_conjugated)
    {
        const int formal_charge = static_cast<int>(atom.GetFormalCharge());
        if (formal_charge == 0)
        {
            return 0.0;
        }

        const double magnitude = static_cast<double>(std::abs(formal_charge));
        const int atomic_num = static_cast<int>(atom.GetAtomicNum());
        const bool is_aromatic = molgr::vendor::openbabel_threading::AtomIsAromatic(
            const_cast<OpenBabel::OBAtom &>(atom));
        const int radical_electrons = static_cast<int>(molgr::utils::GetUnpairedElectronCount(atom));

        /// Estimate localization cost from charge sign and magnitude when the element-specific rule cannot
        /// be used.
        const auto generic_penalty = [&]()
        {
            double penalty = 0.0;
            if (atomic_num == 1)
            {
                penalty = 4.0 * magnitude;
            }
            else if (formal_charge < 0)
            {
                if (atomic_num == 9 || atomic_num == 17 || atomic_num == 35 || atomic_num == 53)
                {
                    penalty = 0.2 * magnitude;
                }
                else if (atomic_num == 8 || atomic_num == 16 || atomic_num == 34 || atomic_num == 52)
                {
                    penalty = 0.3 * magnitude;
                }
                else if (atomic_num == 7 || atomic_num == 15 || atomic_num == 33 || atomic_num == 51)
                {
                    penalty = 0.6 * magnitude;
                }
                else if (atomic_num == 6)
                {
                    penalty = (is_conjugated || is_aromatic ? 1.5 : 4.0) * magnitude;
                }
                else
                {
                    penalty = (is_conjugated || is_aromatic ? 1.0 : 2.0) * magnitude;
                }
            }
            else
            {
                if (atomic_num == 7 || atomic_num == 15 || atomic_num == 33 || atomic_num == 51)
                {
                    penalty = 0.4 * magnitude;
                }
                else if (atomic_num == 8 || atomic_num == 16 || atomic_num == 34 || atomic_num == 52)
                {
                    penalty = 1.0 * magnitude;
                }
                else if (atomic_num == 6)
                {
                    penalty = (is_conjugated || is_aromatic ? 1.2 : 3.0) * magnitude;
                }
                else
                {
                    penalty = (is_conjugated || is_aromatic ? 0.8 : 1.8) * magnitude;
                }
            }
            if (!is_conjugated && !is_aromatic)
            {
                penalty += 0.5 * magnitude;
            }
            return penalty;
        };

        /// Apply the main-group charge penalty when the element and its valence admit that model.
        const auto main_group_charge_penalty = [&]() -> std::optional<double>
        {
            const auto info_it = molgr::kNonMetalDict.find(atomic_num);
            if (info_it == molgr::kNonMetalDict.end())
            {
                return std::nullopt;
            }

            const molgr::ElementInfo &element_info = info_it->second;
            const int local_electron_count =
                static_cast<int>(element_info.num_outer_electrons) - formal_charge +
                static_cast<int>(atom.GetTotalValence());
            const int total_valence = static_cast<int>(atom.GetTotalValence());
            const int default_valence = static_cast<int>(element_info.default_valence);
            const int shell_target = atomic_num == 1 ? 2 : 8;
            const int coordination_excess = std::max(total_valence - default_valence, 0);
            const int shell_deficiency = std::max(shell_target - local_electron_count, 0);
            const int shell_surplus = std::max(local_electron_count - shell_target, 0);
            const double ionization_norm = NormalizeValue(
                LookupMapValue(molgr::kNonMetalFirstIonizationEnergyEv, atomic_num),
                7.5,
                18.0,
                0.5);
            const double electronegativity_norm = NormalizeValue(
                LookupMapValue(molgr::kNonMetalPaulingElectronegativity, atomic_num),
                1.9,
                4.0,
                0.5);
            const double local_environment_penalty =
                !is_conjugated && !is_aromatic ? 0.05 * magnitude : 0.0;

            if (formal_charge > 0 && shell_deficiency > 0)
            {
                return magnitude *
                       (0.45 + 0.55 * static_cast<double>(shell_deficiency) + 0.9 * ionization_norm +
                        local_environment_penalty);
            }

            if (local_electron_count >= shell_target && coordination_excess > 0)
            {
                return magnitude *
                       (0.18 + 0.12 * static_cast<double>(coordination_excess) +
                        0.04 * static_cast<double>(shell_surplus) + local_environment_penalty);
            }

            if (formal_charge < 0 && local_electron_count >= shell_target)
            {
                return magnitude *
                       (0.10 + 0.60 * (1.0 - electronegativity_norm) + local_environment_penalty);
            }
            return std::nullopt;
        };

        double penalty = main_group_charge_penalty().value_or(generic_penalty());
        if (radical_electrons > 0 && atomic_num == 6 && !is_conjugated && !is_aromatic)
        {
            penalty += 2.0 * static_cast<double>(radical_electrons);
        }
        return penalty;
    }

    /// Estimate the cost of localizing the indicated electronic defect on this element and geometry.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param is_conjugated Whether the atom belongs to the conjugated ligand network.
    /// @return Numeric localization cost used in candidate ranking.
    double RadicalLocalizationPenaltyForAtom(
        OpenBabel::OBAtom &atom,
        bool is_conjugated)
    {
        const int radical_electrons = static_cast<int>(molgr::utils::GetUnpairedElectronCount(atom));
        // A validated neutral C/N/Si/P two-electron deficit can be represented as
        // a singlet (0 unpaired electrons, one active lone pair). It remains a
        // localized carbene-/nitrene-/phosphinidene-like state. Singlet/triplet
        // relative stability is system-dependent, so both occupations contribute
        // equally until a reliable environment-specific model is available.
        // Ordinary active lone pairs do not satisfy this strict topology predicate.
        const double localized_electron_equivalents =
            static_cast<double>(radical_electrons) +
            (IsExplicitSingletTwoElectronCenter(atom) ? 2.0 : 0.0);
        if (localized_electron_equivalents <= 0.0)
        {
            return 0.0;
        }

        const double magnitude = localized_electron_equivalents;
        const int atomic_num = static_cast<int>(atom.GetAtomicNum());
        const bool is_aromatic = molgr::vendor::openbabel_threading::AtomIsAromatic(atom);

        if (atomic_num == 1)
        {
            return 4.0 * magnitude;
        }
        if (atomic_num == 6)
        {
            return (is_conjugated || is_aromatic ? 0.6 : 2.5) * magnitude;
        }
        if (atomic_num == 7 || atomic_num == 15 || atomic_num == 33 || atomic_num == 51)
        {
            return (is_conjugated || is_aromatic ? 1.0 : 2.5) * magnitude;
        }
        if (atomic_num == 8 || atomic_num == 16 || atomic_num == 34 || atomic_num == 52)
        {
            return (is_conjugated || is_aromatic ? 1.5 : 3.0) * magnitude;
        }
        return (is_conjugated || is_aromatic ? 1.2 : 2.5) * magnitude;
    }

    /// Serialize the candidate ranking terms in selection order for diagnostics.
    /// @param discordance_count Aggregate structural and electronic conflict count.
    /// @param charge_localization_margin_exceeded Flag for crossing the tolerated charge-localization
    ///     penalty.
    /// @param conjugated_atom_deficit_count Shortfall of this topology measure relative to the competing
    ///     assignments.
    /// @param conjugated_bond_deficit_count Shortfall of this topology measure relative to the competing
    ///     assignments.
    /// @param aromatic_atom_deficit_count Shortfall of this topology measure relative to the competing
    ///     assignments.
    /// @param aromatic_ring_deficit_count Shortfall of this topology measure relative to the competing
    ///     assignments.
    /// @param aromatic_stability_deficit Shortfall of this topology measure relative to the competing
    ///     assignments.
    /// @param radical_localization_penalty Energetic preference against localized unpaired electrons.
    /// @param hyperconjugation_deficit Shortfall of this topology measure relative to the competing
    ///     assignments.
    /// @param nitrosyl_geometry_penalty Penalty for the local metal-NO assignment.
    /// @param high_zinc_valence_count Number of high-valence zinc assignments contributing to the ranking.
    /// @param score_value Numeric reconstruction score included in the final tie-break.
    /// @param combination_index Original metal combination index used for deterministic tie-breaking.
    /// @return Serialized ranking terms in tie-break order.
    std::string SelectionKeyString(
        double discordance_count,
        int charge_localization_margin_exceeded,
        int conjugated_atom_deficit_count,
        int conjugated_bond_deficit_count,
        int aromatic_atom_deficit_count,
        int aromatic_ring_deficit_count,
        double aromatic_stability_deficit,
        double radical_localization_penalty,
        int hyperconjugation_deficit,
        double nitrosyl_geometry_penalty,
        int high_zinc_valence_count,
        double score_value,
        int combination_index)
    {
        std::ostringstream out;
        out << std::setprecision(17) << discordance_count << ","
            << charge_localization_margin_exceeded << ","
            << conjugated_atom_deficit_count << "," << conjugated_bond_deficit_count << ","
            << aromatic_atom_deficit_count << "," << aromatic_ring_deficit_count << ","
            << aromatic_stability_deficit << ","
            << radical_localization_penalty << ","
            << hyperconjugation_deficit << ","
            << nitrosyl_geometry_penalty << ","
            << high_zinc_valence_count << ","
            << score_value << "," << combination_index;
        return out.str();
    }

    /// @param state Ligand graph and electronic budgets at this stage of reconstruction.
    /// @return An independent ligand state for candidate-specific annotations, or null for an absent state.
    std::shared_ptr<molgr::state::ReconstructionState> CloneNoMetalStateForCandidate(
        const std::shared_ptr<molgr::state::ReconstructionState> &state)
    {
        if (!state)
        {
            return nullptr;
        }
        auto clone = std::make_shared<molgr::state::ReconstructionState>(*state);
        if (state->omol)
        {
            clone->omol = molgr::utils::CloneMolTopologyShared(*state->omol);
        }
        return clone;
    }

    /// A resolved octet C−—N+ pair need not share a conjugated pi plane.
    /// @param bond Bond whose endpoints and order define the local environment.
    /// @return True if the bond joins a closed-shell carbon anion and nitrogen cation meeting the ylide
    ///     criteria.
    bool IsClosedShellCNYlideBond(OpenBabel::OBBond &bond)
    {
        auto *carbon = bond.GetBeginAtom();
        auto *nitrogen = bond.GetEndAtom();
        if (carbon->GetAtomicNum() != 6) std::swap(carbon, nitrogen);
        if (bond.GetBondOrder() != 1 ||
            carbon->GetAtomicNum() != 6 || carbon->GetFormalCharge() != -1 ||
            carbon->GetTotalValence() != 3 ||
            nitrogen->GetAtomicNum() != 7 || nitrogen->GetFormalCharge() != 1 ||
            nitrogen->GetTotalValence() != 4)
            return false;
        // This local octet pair does not require a common conjugated pi plane.
        for (const auto *atom : {carbon, nitrogen})
            if (atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                return false;
        return true;
    }

    /// Measure ligand charge and radical localization and the stabilizing organic topology terms.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param config Settings for the current reconstruction or scoring operation.
    /// @return Ligand-only electronic metrics used to rank alternative metal states.
    OrganicElectronicStateMetrics ComputeOrganicElectronicStateMetrics(
        const OpenBabel::OBMol &mol,
        const molgr::config::MolGRConfig &config)
    {
        try
        {
            const auto topology_metrics = molgr::organic_topology::ComputeOrganicTopologyMetrics(
                mol,
                config.organic_topology);
            std::set<int> conjugated_atom_indices(
                topology_metrics.conjugated_atom_indices.begin(),
                topology_metrics.conjugated_atom_indices.end());

            OrganicElectronicStateMetrics metrics;
            metrics.aromatic_atom_count = topology_metrics.aromatic_atom_count;
            metrics.aromatic_ring_count = topology_metrics.aromatic_ring_count;
            metrics.aromatic_stability_score = topology_metrics.aromatic_stability_score;
            metrics.conjugated_atom_count = topology_metrics.conjugated_atom_count;
            metrics.conjugated_bond_count = topology_metrics.conjugated_bond_count;
            metrics.max_conjugated_component_size = topology_metrics.max_conjugated_component_size;
            metrics.hyperconjugative_donor_count = topology_metrics.hyperconjugative_donor_count;
            metrics.hyperconjugation_score = topology_metrics.hyperconjugation_score;

            const std::size_t atom_count = static_cast<std::size_t>(mol.NumAtoms());
            std::vector<double> signed_charge_penalties(atom_count, 0.0);
            std::vector<std::vector<int>> neighbor_indices(atom_count);
            std::vector<int> atomic_numbers(atom_count, 0);
            double unsigned_charge_localization_penalty = 0.0;
            FOR_ATOMS_OF_MOL(atom_iter, const_cast<OpenBabel::OBMol &>(mol))
            {
                OpenBabel::OBAtom &atom = *atom_iter;
                const int atom_idx = static_cast<int>(atom.GetIdx()) - 1;
                atomic_numbers[static_cast<std::size_t>(atom_idx)] =
                    static_cast<int>(atom.GetAtomicNum());
                const bool is_conjugated =
                    conjugated_atom_indices.find(atom_idx) != conjugated_atom_indices.end();
                metrics.singlet_two_electron_center_count +=
                    IsExplicitSingletTwoElectronCenter(atom) ? 1 : 0;
                metrics.radical_localization_penalty +=
                    RadicalLocalizationPenaltyForAtom(atom, is_conjugated);
                const double atom_charge_penalty =
                    ChargeLocalizationPenaltyForAtom(atom, is_conjugated);
                unsigned_charge_localization_penalty += atom_charge_penalty;
                signed_charge_penalties[static_cast<std::size_t>(atom_idx)] =
                    atom.GetFormalCharge() < 0 ? -atom_charge_penalty : atom_charge_penalty;
            }

            double polarity_inversion_penalty = 0.0;
            FOR_BONDS_OF_MOL(bond_iter, const_cast<OpenBabel::OBMol &>(mol))
            {
                const int begin_idx = static_cast<int>(bond_iter->GetBeginAtomIdx()) - 1;
                const int end_idx = static_cast<int>(bond_iter->GetEndAtomIdx()) - 1;
                const double begin_penalty =
                    signed_charge_penalties[static_cast<std::size_t>(begin_idx)];
                const double end_penalty =
                    signed_charge_penalties[static_cast<std::size_t>(end_idx)];
                const bool opposite_charges = begin_penalty * end_penalty < 0.0;
                bool allows_cancellation = true;
                const bool both_conjugated =
                    conjugated_atom_indices.find(begin_idx) != conjugated_atom_indices.end() &&
                    conjugated_atom_indices.find(end_idx) != conjugated_atom_indices.end();
                if (opposite_charges && !IsClosedShellCNYlideBond(*bond_iter) &&
                    bond_iter->GetBondOrder() == 1 &&
                    !molgr::vendor::openbabel_threading::BondIsAromatic(*bond_iter) &&
                    !bond_iter->IsInRing() &&
                    !both_conjugated)
                {
                    const int positive_idx = begin_penalty > 0.0 ? begin_idx : end_idx;
                    const int negative_idx = begin_penalty > 0.0 ? end_idx : begin_idx;
                    const auto positive_electronegativity = LookupMapValue(
                        molgr::kNonMetalPaulingElectronegativity,
                        atomic_numbers[static_cast<std::size_t>(positive_idx)]);
                    const auto negative_electronegativity = LookupMapValue(
                        molgr::kNonMetalPaulingElectronegativity,
                        atomic_numbers[static_cast<std::size_t>(negative_idx)]);
                    if (positive_electronegativity.has_value() &&
                        negative_electronegativity.has_value() &&
                        *positive_electronegativity >
                            *negative_electronegativity +
                                kMinimumChargePolarityInversionElectronegativityGap)
                    {
                        allows_cancellation = false;
                        polarity_inversion_penalty += 2.0 * std::min(
                            std::abs(begin_penalty),
                            std::abs(end_penalty));
                    }
                }
                if (!allows_cancellation)
                {
                    continue;
                }
                neighbor_indices[static_cast<std::size_t>(begin_idx)].push_back(end_idx);
                neighbor_indices[static_cast<std::size_t>(end_idx)].push_back(begin_idx);
            }

            // Neutral atoms do not bridge otherwise remote charge centers in
            // one large ligand component.
            std::vector<bool> visited_charged_atoms(atom_count, false);
            for (std::size_t root_idx = 0; root_idx < atom_count; ++root_idx)
            {
                if (visited_charged_atoms[root_idx] ||
                    signed_charge_penalties[root_idx] == 0.0)
                {
                    continue;
                }
                double component_signed_penalty = 0.0;
                std::vector<int> pending_indices = {static_cast<int>(root_idx)};
                visited_charged_atoms[root_idx] = true;
                while (!pending_indices.empty())
                {
                    const int atom_idx = pending_indices.back();
                    pending_indices.pop_back();
                    component_signed_penalty +=
                        signed_charge_penalties[static_cast<std::size_t>(atom_idx)];
                    for (const int neighbor_idx : neighbor_indices[static_cast<std::size_t>(atom_idx)])
                    {
                        if (visited_charged_atoms[static_cast<std::size_t>(neighbor_idx)] ||
                            signed_charge_penalties[static_cast<std::size_t>(neighbor_idx)] == 0.0)
                        {
                            continue;
                        }
                        visited_charged_atoms[static_cast<std::size_t>(neighbor_idx)] = true;
                        pending_indices.push_back(neighbor_idx);
                    }
                }
                metrics.charge_localization_penalty += std::abs(component_signed_penalty);
            }
            metrics.charge_localization_component_cancellation = std::max(
                0.0,
                unsigned_charge_localization_penalty - metrics.charge_localization_penalty);
            metrics.charge_localization_penalty += polarity_inversion_penalty;
            metrics.charge_localization_polarity_inversion_penalty =
                polarity_inversion_penalty;
            return metrics;
        }
        catch (...)
        {
            OrganicElectronicStateMetrics metrics;
            metrics.radical_localization_penalty = std::numeric_limits<double>::infinity();
            metrics.charge_localization_penalty = std::numeric_limits<double>::infinity();
            return metrics;
        }
    }

    /// @param vector Cartesian displacement vector.
    /// @return The Euclidean length of a Cartesian vector.
    double VectorNorm(const Point3D &vector)
    {
        return std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
    }

    /// @param lhs First Cartesian vector.
    /// @param rhs Second Cartesian vector.
    /// @return The dot product of the two Cartesian vectors.
    double DotProduct(const Point3D &lhs, const Point3D &rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    /// @param lhs First Cartesian vector.
    /// @param rhs Second Cartesian vector.
    /// @return The cross product of the two Cartesian vectors.
    Point3D CrossProduct(const Point3D &lhs, const Point3D &rhs)
    {
        return {
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x};
    }

    /// Classify the coordination geometry from the visible donor positions around this metal.
    /// @param visible_atoms Donors accessible from this metal site.
    /// @param metal_state Metal site with its coordinates and proposed electronic assignment.
    /// @param config Ligand-field and coordination geometry thresholds for spin inference.
    /// @return Geometry class inferred from donor coordinates.
    std::string CoordinationGeometry(
        const std::vector<OpenBabel::OBAtom *> &visible_atoms,
        const molgr::MetalAtomPosition &metal_state,
        const molgr::config::MetalRadicalInferenceConfig &config)
    {
        std::vector<Point3D> vectors;
        vectors.reserve(visible_atoms.size());
        for (const OpenBabel::OBAtom *atom : visible_atoms)
        {
            if (atom != nullptr)
            {
                vectors.push_back(
                    Point3D{
                        atom->GetX() - metal_state.position_x,
                        atom->GetY() - metal_state.position_y,
                        atom->GetZ() - metal_state.position_z});
            }
        }
        if (vectors.size() == 2)
        {
            const double denominator = VectorNorm(vectors[0]) * VectorNorm(vectors[1]);
            if (denominator <= 1.0e-8)
            {
                return "bent";
            }
            const double cosine = std::max(
                -1.0,
                std::min(1.0, DotProduct(vectors[0], vectors[1]) / denominator));
            const double angle_degrees = std::acos(cosine) * 180.0 / std::acos(-1.0);
            return angle_degrees >= config.linear_angle_min_degrees ? "linear" : "bent";
        }
        if (vectors.size() != 4)
        {
            return "other";
        }

        std::optional<Point3D> best_normal;
        double best_norm = 0.0;
        for (std::size_t i = 0; i < vectors.size(); ++i)
        {
            for (std::size_t j = i + 1; j < vectors.size(); ++j)
            {
                const Point3D normal = CrossProduct(vectors[i], vectors[j]);
                const double normal_norm = VectorNorm(normal);
                if (normal_norm > best_norm)
                {
                    best_normal = normal;
                    best_norm = normal_norm;
                }
            }
        }
        if (!best_normal.has_value() || best_norm <= 1.0e-8)
        {
            return "other";
        }
        const Point3D unit_normal{
            best_normal->x / best_norm,
            best_normal->y / best_norm,
            best_normal->z / best_norm};
        double planarity_distance = 0.0;
        for (const Point3D &vector : vectors)
        {
            planarity_distance += std::abs(DotProduct(vector, unit_normal));
        }
        planarity_distance /= static_cast<double>(vectors.size());
        return planarity_distance <= config.square_planar_planarity_tolerance_angstrom
                   ? "square_planar"
                   : "tetrahedral";
    }

    /// Count coordination environments that disagree with the candidate's metal and ligand assignment.
    /// @param visible_atoms_by_metal Ligand atoms visible from each metal site, grouped by metal.
    /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
    /// @param config Settings for the current reconstruction or scoring operation.
    /// @return Number of detected coordination geometry conflicts.
    int CoordinationGeometryDiscordanceCount(
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal,
        const std::vector<molgr::MetalAtomPosition> &metal_states,
        const molgr::config::MolGRConfig &config)
    {
        int count = 0;
        const std::size_t pair_count = std::min(visible_atoms_by_metal.size(), metal_states.size());
        for (std::size_t idx = 0; idx < pair_count; ++idx)
        {
            const auto &metal = metal_states[idx];
            auto atoms = visible_atoms_by_metal[idx];
            if (metal.symbol == "Au")
            {
                auto shell = config.metal_scoring;
                shell.metal_coordination_extra_tolerance_angstrom = 0.20;
                atoms.erase(std::remove_if(atoms.begin(), atoms.end(), [&](auto *a) {
                    const double dx = a->GetX() - metal.position_x, dy = a->GetY() - metal.position_y,
                                 dz = a->GetZ() - metal.position_z;
                    return std::sqrt(dx*dx+dy*dy+dz*dz) > molgr::coordination::CoordinationDistanceCutoff(
                        metal.element_idx, a->GetAtomicNum(), shell);
                }), atoms.end());
            }
            const auto geometry = CoordinationGeometry(atoms, metal, config.metal_radical_inference);
            bool independent_donors = true;
            std::vector<int> elements;
            bool aryl_anions = true;
            for (auto *atom : atoms)
            {
                elements.push_back(atom->GetAtomicNum());
                if (atom->GetAtomicNum() == 6)
                {
                    aryl_anions = aryl_anions && atom->GetFormalCharge() == -1;
                    FOR_NBORS_OF_ATOM(n, atom) aryl_anions = aryl_anions && n->GetAtomicNum() == 6;
                }
            }
            std::sort(elements.begin(), elements.end());
            const bool supported_square_planar_assignment = elements == std::vector<int>{7,7,7,7} ||
                (elements == std::vector<int>{6,6,15,15} && aryl_anions);
            for (std::size_t i = 0; i < atoms.size(); ++i)
            {
                for (std::size_t j = i+1; j < atoms.size(); ++j)
                    independent_donors = independent_donors && !atoms[i]->GetBond(atoms[j]);
            }
            if (((metal.symbol == "Pd" || metal.symbol == "Pt") && metal.valence >= 4 && geometry == "square_planar") ||
                ((metal.symbol == "Ag" || metal.symbol == "Au") && metal.valence >= 3 && geometry == "linear") ||
                (metal.symbol == "Au" && metal.valence == 1 && geometry == "square_planar" && independent_donors && supported_square_planar_assignment))
                ++count;
        }
        return count;
    }

    /// @param charge Signed formal charge used for this electronic assignment.
    /// @return -1, 0 or 1 according to the formal charge.
    int ChargeSign(int charge)
    {
        if (charge > 0)
        {
            return 1;
        }
        if (charge < 0)
        {
            return -1;
        }
        return 0;
    }

    /// @param begin_atom Endpoint atom of the bond or conjugated path.
    /// @param end_atom Endpoint atom of the bond or conjugated path.
    /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
    /// @return The sign of the nearest charged metal to the bond, or zero if none is charged.
    int NearestNonzeroMetalChargeSignToBond(
        const OpenBabel::OBAtom &begin_atom,
        const OpenBabel::OBAtom &end_atom,
        const std::vector<molgr::MetalAtomPosition> &metal_states)
    {
        const double midpoint_x =
            (static_cast<double>(begin_atom.GetX()) + static_cast<double>(end_atom.GetX())) * 0.5;
        const double midpoint_y =
            (static_cast<double>(begin_atom.GetY()) + static_cast<double>(end_atom.GetY())) * 0.5;
        const double midpoint_z =
            (static_cast<double>(begin_atom.GetZ()) + static_cast<double>(end_atom.GetZ())) * 0.5;
        double best_distance_sq = std::numeric_limits<double>::infinity();
        int best_charge_sign = 0;
        for (const auto &metal_state : metal_states)
        {
            const int metal_charge_sign = ChargeSign(metal_state.valence);
            if (metal_charge_sign == 0)
            {
                continue;
            }
            const double dx = midpoint_x - metal_state.position_x;
            const double dy = midpoint_y - metal_state.position_y;
            const double dz = midpoint_z - metal_state.position_z;
            const double distance_sq = dx * dx + dy * dy + dz * dz;
            if (distance_sq < best_distance_sq)
            {
                best_distance_sq = distance_sq;
                best_charge_sign = metal_charge_sign;
            }
        }
        return best_charge_sign;
    }

    /// Check for a proton outside the inner coordination spheres of the metal sites.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
    /// @param config Metal contact cutoffs, visibility radii and candidate scoring weights.
    /// @return True when a proton is present outside the metal coordination sphere.
    bool HasOuterSphereProton(
        OpenBabel::OBMol &mol,
        const std::vector<molgr::MetalAtomPosition> &metal_states,
        const molgr::config::MetalScoringConfig &config)
    {
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom &atom = *atom_iter;
            if (atom.IsMetal() || atom.GetAtomicNum() != 1 || atom.GetFormalCharge() <= 0)
            {
                continue;
            }

            bool is_inner_sphere_to_any_metal = false;
            for (const auto &metal_state : metal_states)
            {
                if (IsInnerSphereAtom(atom, metal_state.Site(), config))
                {
                    is_inner_sphere_to_any_metal = true;
                    break;
                }
            }
            if (!is_inner_sphere_to_any_metal)
            {
                return true;
            }
        }
        return false;
    }

    /// Count negatively charged metal assignments and identify applicable chemical exceptions.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
    /// @param config Metal contact cutoffs, visibility radii and candidate scoring weights.
    /// @return Count of negative-metal conflicts and applicable exceptions.
    NegativeMetalDiscordance NegativeMetalDiscordanceCount(
        OpenBabel::OBMol &mol,
        const std::vector<molgr::MetalAtomPosition> &metal_states,
        const molgr::config::MetalScoringConfig &config)
    {
        int negative_metal_count = 0;
        bool has_positive_metal_counterion = false;
        for (const auto &metal_state : metal_states)
        {
            if (metal_state.valence < 0)
            {
                int borane_acceptors = 0;
                if (metal_state.symbol == "Au")
                {
                    FOR_ATOMS_OF_MOL(a, mol)
                        if (a->GetAtomicNum() == 5 && a->GetFormalCharge() == 0 && a->GetTotalValence() == 3 &&
                            IsInnerSphereAtom(*a, metal_state.Site(), config)) ++borane_acceptors;
                }
                if (borane_acceptors < 2) negative_metal_count += std::abs(metal_state.valence);
            }
            else if (metal_state.valence > 0)
            {
                has_positive_metal_counterion = true;
            }
        }
        if (negative_metal_count == 0)
        {
            return {};
        }

        const bool has_outer_sphere_proton = HasOuterSphereProton(mol, metal_states, config);
        if (has_outer_sphere_proton || has_positive_metal_counterion)
        {
            return NegativeMetalDiscordance{
                0,
                has_outer_sphere_proton,
                has_positive_metal_counterion};
        }
        return NegativeMetalDiscordance{negative_metal_count, false, false};
    }

    /// Test whether a positively charged nonmetal has an incomplete valence shell.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the site satisfies the specified local charge or donor condition.
    bool IsUnsaturatedOrganicCation(OpenBabel::OBAtom &atom)
    {
        if (atom.IsMetal() || static_cast<int>(atom.GetFormalCharge()) <= 0)
        {
            return false;
        }

        const auto element_info = molgr::kNonMetalDict.find(
            static_cast<int>(atom.GetAtomicNum()));
        if (element_info == molgr::kNonMetalDict.end())
        {
            return false;
        }

        // Compare assigned outer-shell electrons with the closed-shell target.
        // Degree is not a proxy for this: one triple bond gives O+ three
        // bond-order units while still completing its octet.
        const int total_valence = static_cast<int>(atom.GetTotalValence());
        const int local_electron_count =
            static_cast<int>(element_info->second.num_outer_electrons) -
            static_cast<int>(atom.GetFormalCharge()) + total_valence;
        const int shell_target = atom.GetAtomicNum() == 1 ? 2 : 8;
        return local_electron_count < shell_target;
    }

    /// A positive formal charge does not remove every Lewis lone pair.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the site satisfies the specified local charge or donor condition.
    bool IsCationicPairedElectronDonor(OpenBabel::OBAtom &atom)
    {
        const auto info = molgr::kNonMetalDict.find(static_cast<int>(atom.GetAtomicNum()));
        if (info == molgr::kNonMetalDict.end() || atom.GetFormalCharge() <= 0 ||
            molgr::utils::HasUnresolvedTwoElectronCenter(atom))
        {
            return false;
        }
        // Lewis pairs include ordinary valence pairs, not just active search pairs.
        // Electrons already assigned as unpaired cannot form a donor pair.
        const int unpaired = molgr::utils::GetUnpairedElectronCount(atom);
        const int paired = static_cast<int>(info->second.num_outer_electrons) -
            atom.GetFormalCharge() - static_cast<int>(atom.GetTotalValence()) - unpaired;
        return unpaired >= 0 && paired >= 2 && paired % 2 == 0;
    }

    /// Test whether neighboring formal charges exactly cancel the charge at this site.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the site satisfies the specified local charge or donor condition.
    bool HasAdjacentFormalChargeCancellation(OpenBabel::OBAtom &atom)
    {
        const int formal_charge = static_cast<int>(atom.GetFormalCharge());
        if (formal_charge == 0)
        {
            return false;
        }
        int adjacent_charge = 0;
        FOR_BONDS_OF_ATOM(bond_iter, &atom)
        {
            OpenBabel::OBBond &bond = *bond_iter;
            OpenBabel::OBAtom *neighbor = OtherBondAtom(bond, atom);
            if (neighbor != nullptr && !neighbor->IsMetal())
            {
                adjacent_charge += static_cast<int>(neighbor->GetFormalCharge());
            }
        }
        return formal_charge + adjacent_charge == 0;
    }

    /// Discount the formal metal charge explained by resolved terminal NO+ ligands.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
    /// @param nitrosonium_indices Indices of assigned nitrosonium sites in the ligand graph.
    /// @param penalized_negative_donor_indices Anionic donor indices contributing to the candidate penalty.
    /// @param visible_atoms_by_metal Ligand atoms visible from each metal site, grouped by metal.
    /// @param negative_metal_penalty Penalty assigned to the negative metal state before exemptions.
    /// @return Amount of the formal-charge penalty compensated by the nitrosonium assignment.
    double NitrosoniumFormalChargeExemption(
        OpenBabel::OBMol &mol,
        const std::vector<molgr::MetalAtomPosition> &metal_states,
        const std::set<int> &nitrosonium_indices,
        const std::set<int> &penalized_negative_donor_indices,
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal,
        double negative_metal_penalty)
    {
        // NO+ with resolved neutral or anionic ancillary ligands can require a negative
        // formal metal charge. Avoid penalizing that representation twice;
        // retain all geometry and other electronic safeguards.
        if (metal_states.size() != 1 || metal_states.front().valence >= 0 ||
            nitrosonium_indices.empty() || nitrosonium_indices.size() % 2 != 0)
            return 0.0;
        int ancillary_charge = 0;
        bool has_linear_carbonyl = false;
        std::set<int> anionic_component_indices;
        std::set<int> remaining;
        for (unsigned int index = 1; index <= mol.NumAtoms(); ++index)
            if (!nitrosonium_indices.count(index)) remaining.insert(index);
        while (!remaining.empty())
        {
            std::vector<int> stack{*remaining.begin()};
            remaining.erase(stack.back());
            int charge = 0;
            std::set<int> component_indices;
            while (!stack.empty())
            {
                auto *atom = mol.GetAtom(stack.back());
                stack.pop_back();
                component_indices.insert(atom->GetIdx());
                if (atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                    return 0.0;
                charge += atom->GetFormalCharge();
                FOR_NBORS_OF_ATOM(neighbor, atom)
                {
                    const int index = neighbor->GetIdx();
                    if (remaining.erase(index)) stack.push_back(index);
                }
            }
            if (charge > 0) return 0.0;
            ancillary_charge += charge;
            if (charge < 0)
                anionic_component_indices.insert(component_indices.begin(), component_indices.end());
            // An anionic carbonyl/nitrosyl complex with otherwise neutral ligands
            // also requires a negative formal metal state. Recognize terminal CO
            // from the complete two-atom component and its metal-facing geometry,
            // not merely from adjacent formal charge cancellation in a ligand.
            if (charge == 0 && component_indices.size() == 2 &&
                visible_atoms_by_metal.size() == 1)
            {
                auto *carbon = mol.GetAtom(*component_indices.begin());
                auto *oxygen = mol.GetAtom(*component_indices.rbegin());
                if (carbon->GetAtomicNum() == 8) std::swap(carbon, oxygen);
                const auto &visible = visible_atoms_by_metal.front();
                auto *bond = carbon->GetBond(oxygen);
                if (carbon->GetAtomicNum() == 6 && oxygen->GetAtomicNum() == 8 &&
                    carbon->GetFormalCharge() == -1 && oxygen->GetFormalCharge() == 1 &&
                    bond && bond->GetBondOrder() == 3 &&
                    std::find(visible.begin(), visible.end(), carbon) != visible.end() &&
                    std::find(visible.begin(), visible.end(), oxygen) == visible.end())
                {
                    const double co = carbon->GetDistance(oxygen);
                    if (co > 1.05 && co < 1.25 &&
                        molgr::coordination::MetalDonorAngle(metal_states.front().Site(), *carbon, *oxygen) > 160.0)
                        has_linear_carbonyl = true;
                }
            }
        }
        const int nitrosyl_core_charge = metal_states.front().valence +
            static_cast<int>(nitrosonium_indices.size() / 2);
        const int total_charge = nitrosyl_core_charge + ancillary_charge;
        // A neutral M(NO)n formal core also permits resolved anionic ancillary
        // ligands: its negative metal charge only compensates NO+, leaving the
        // input charge on the ancillary ligands. Extra reduction of that core
        // remains outside this branch. With neutral ancillary components,
        // cations and carbonyl-supported anions retain their existing scope.
        // Positive components and unresolved occupations are excluded above.
        if (total_charge != 0 && nitrosyl_core_charge != 0 &&
            !(ancillary_charge == 0 && (total_charge > 0 || has_linear_carbonyl))) return 0.0;
        int compensated_donors = 0;
        for (int index : penalized_negative_donor_indices)
            if (!nitrosonium_indices.count(index) &&
                (HasAdjacentFormalChargeCancellation(*mol.GetAtom(index)) ||
                 anionic_component_indices.count(index)))
                ++compensated_donors;
        return negative_metal_penalty + compensated_donors;
    }

    /// Test whether a directly bonded carbon carries negative formal charge.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the site satisfies the specified local charge or donor condition.
    bool HasAdjacentCarbanion(OpenBabel::OBAtom &atom)
    {
        FOR_NBORS_OF_ATOM(neighbor, &atom)
            if (neighbor->GetAtomicNum() == 6 && neighbor->GetFormalCharge() < 0) return true;
        return false;
    }

    /// Check whether adjacent anionic polarization compensates the cation's charge.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the site satisfies the specified local charge or donor condition.
    bool HasAdjacentAnionicPolarizationCancellation(OpenBabel::OBAtom &atom)
    {
        const int formal_charge = static_cast<int>(atom.GetFormalCharge());
        const auto central_electronegativity = molgr::kNonMetalPaulingElectronegativity.find(
            static_cast<int>(atom.GetAtomicNum()));
        if (formal_charge <= 0 ||
            central_electronegativity == molgr::kNonMetalPaulingElectronegativity.end())
        {
            return false;
        }
        int adjacent_negative_charge = 0;
        FOR_BONDS_OF_ATOM(bond_iter, &atom)
        {
            OpenBabel::OBAtom *neighbor = OtherBondAtom(*bond_iter, atom);
            if (neighbor == nullptr || neighbor->IsMetal())
            {
                continue;
            }
            const int neighbor_charge = static_cast<int>(neighbor->GetFormalCharge());
            const auto neighbor_electronegativity =
                molgr::kNonMetalPaulingElectronegativity.find(
                    static_cast<int>(neighbor->GetAtomicNum()));
            if (neighbor_charge < 0 &&
                neighbor_electronegativity != molgr::kNonMetalPaulingElectronegativity.end() &&
                neighbor_electronegativity->second > central_electronegativity->second)
            {
                adjacent_negative_charge += std::abs(neighbor_charge);
            }
        }
        return adjacent_negative_charge >= formal_charge;
    }

    /// Recognize the local N(+)-C(=S)-S(-) path of a resolved iminium group.
    /// @param nitrogen Atom or site occupying this position in the local chemical motif.
    /// @return True when the site satisfies the specified local charge or donor condition.
    bool HasIminiumDithiocarbonylCompensation(OpenBabel::OBAtom &nitrogen)
    {
        /// Recognize an open-shell or unresolved center when testing anionic compensation.
        const auto open = [](OpenBabel::OBAtom &atom) {
            return molgr::utils::GetUnpairedElectronCount(atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(atom);
        };
        if (nitrogen.GetAtomicNum() != 7 || nitrogen.GetFormalCharge() != 1 ||
            nitrogen.GetExplicitDegree() != 3 || nitrogen.GetImplicitHCount() != 0 ||
            nitrogen.GetTotalValence() != 4 || open(nitrogen)) return false;
        bool external_imine = false;
        FOR_NBORS_OF_ATOM(atom, &nitrogen)
            if (atom->GetAtomicNum() == 6 && nitrogen.GetBond(&*atom)->GetBondOrder() == 2)
                external_imine = true;
        if (!external_imine) return false;
        FOR_NBORS_OF_ATOM(carbon, &nitrogen)
        {
            if (carbon->GetAtomicNum() != 6 || carbon->GetFormalCharge() != 0 ||
                carbon->GetExplicitDegree() != 3 || carbon->GetImplicitHCount() != 0 ||
                open(*carbon) || nitrogen.GetBond(&*carbon)->GetBondOrder() != 1) continue;
            std::vector<std::pair<int, int>> sulfurs;
            bool terminal = true;
            FOR_NBORS_OF_ATOM(atom, &*carbon)
            {
                if (atom->GetAtomicNum() != 16) continue;
                if (atom->GetExplicitDegree() != 1 || atom->GetImplicitHCount() != 0 || open(*atom))
                    terminal = false;
                sulfurs.emplace_back(carbon->GetBond(&*atom)->GetBondOrder(), atom->GetFormalCharge());
            }
            std::sort(sulfurs.begin(), sulfurs.end());
            if (terminal && sulfurs == std::vector<std::pair<int, int>>{{1, -1}, {2, 0}}) return true;
        }
        return false;
    }

    /// Require an actual transferable pair to compensate an electron-deficient cation.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the site satisfies the specified local charge or donor condition.
    bool HasAdjacentAnionicDonorPair(OpenBabel::OBAtom &atom)
    {
        if (molgr::utils::GetUnpairedElectronCount(atom) != 0 ||
            molgr::utils::HasUnresolvedTwoElectronCenter(atom))
        {
            return false;
        }
        FOR_BONDS_OF_ATOM(bond_iter, &atom)
        {
            auto *neighbor = OtherBondAtom(*bond_iter, atom);
            if (neighbor == nullptr || neighbor->GetFormalCharge() >= 0 ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*neighbor) ||
                (bond_iter->GetBondOrder() != 1 && bond_iter->GetBondOrder() != 2))
            {
                continue;
            }
            const auto info = molgr::kNonMetalDict.find(neighbor->GetAtomicNum());
            if (info == molgr::kNonMetalDict.end()) continue;
            const int unpaired = molgr::utils::GetUnpairedElectronCount(*neighbor);
            const int paired = static_cast<int>(info->second.num_outer_electrons) -
                neighbor->GetFormalCharge() - static_cast<int>(neighbor->GetTotalValence()) - unpaired;
            if (unpaired >= 0 && paired >= 2 && paired % 2 == 0) return true;
        }
        return false;
    }

    /// Test whether a positive nonmetal center has a locally compensated charge assignment.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the site satisfies the specified local charge or donor condition.
    bool IsLocallyChargeCompensatedNonmetalCation(OpenBabel::OBAtom &atom)
    {
        if (atom.IsMetal() || static_cast<int>(atom.GetFormalCharge()) <= 0)
        {
            return false;
        }
        // Opposite signs alone cannot repair a sextet without a transferable pair.
        if (IsUnsaturatedOrganicCation(atom) && !HasAdjacentAnionicDonorPair(atom))
        {
            return false;
        }
        if (HasAdjacentFormalChargeCancellation(atom))
        {
            return true;
        }
        const auto element_info = molgr::kNonMetalDict.find(
            static_cast<int>(atom.GetAtomicNum()));
        if (element_info == molgr::kNonMetalDict.end() ||
            static_cast<int>(atom.GetTotalValence()) <=
                static_cast<int>(element_info->second.default_valence))
        {
            return false;
        }

        int adjacent_negative_charge = 0;
        int adjacent_positive_charge = 0;
        FOR_BONDS_OF_ATOM(bond_iter, &atom)
        {
            OpenBabel::OBBond &bond = *bond_iter;
            OpenBabel::OBAtom *neighbor = OtherBondAtom(bond, atom);
            if (neighbor == nullptr || neighbor->IsMetal())
            {
                continue;
            }
            const int neighbor_charge = static_cast<int>(neighbor->GetFormalCharge());
            if (neighbor_charge < 0)
            {
                adjacent_negative_charge += std::abs(neighbor_charge);
            }
            else if (neighbor_charge > 0)
            {
                adjacent_positive_charge += neighbor_charge;
            }
        }
        return adjacent_negative_charge > adjacent_positive_charge ||
            HasIminiumDithiocarbonylCompensation(atom);
    }

    /// Collect ligand cations that remain uncompensated after local and nitrosonium exceptions.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param coordinated_nitrosonium_indices Indices of assigned nitrosonium sites in the ligand graph.
    /// @param config Settings for the current reconstruction or scoring operation.
    /// @return Indices of ligand cations that remain uncompensated.
    std::set<int> UncompensatedOrganicCationIndices(
        OpenBabel::OBMol &mol,
        const std::set<int> &coordinated_nitrosonium_indices,
        const molgr::config::MolGRConfig &config)
    {
        std::set<int> uncompensated;
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom &atom = *atom_iter;
            if (!atom.IsMetal() &&
                atom.GetFormalCharge() > 0 &&
                coordinated_nitrosonium_indices.count(atom.GetIdx()) == 0 &&
                !IsLocallyChargeCompensatedNonmetalCation(atom))
            {
                uncompensated.insert(atom.GetIdx());
            }
        }
        if (uncompensated.empty()) return uncompensated;
        const auto topology = molgr::organic_topology::ComputeOrganicTopologyMetrics(mol, config.organic_topology);
        for (const auto &component : topology.conjugated_atom_components)
        {
            int charge = 0;
            bool only_carbanions = true;
            for (int index : component)
            {
                const auto *atom = mol.GetAtom(index + 1);
                charge += atom->GetFormalCharge();
                if (atom->GetFormalCharge() < 0 && atom->GetAtomicNum() != 6)
                    only_carbanions = false;
            }
            if (charge != 0 || !only_carbanions) continue;
            for (int index : component) uncompensated.erase(index + 1);
        }
        return uncompensated;
    }

    /// Complete bounded alternating paths; An empty optional means the search must abstain.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param source One-based index of the cation at which the path starts.
    /// @return Compensation paths, or an empty optional if the search bound is reached.
    std::optional<std::vector<std::vector<int>>> CationCompensationPaths(
        OpenBabel::OBMol &mol, int source)
    {
        constexpr std::size_t max_states = 4096, max_atoms = 32, max_paths = 32;
        std::deque<std::vector<int>> pending{{source}};
        std::vector<std::vector<int>> paths;
        std::size_t visited = 0;
        while (!pending.empty())
        {
            if (++visited > max_states) return std::nullopt;
            auto path = std::move(pending.front());
            pending.pop_front();
            auto *atom = mol.GetAtom(path.back());
            const int order = path.size() % 2 ? 2 : 1;
            FOR_BONDS_OF_ATOM(bond, atom)
            {
                auto *end = bond->GetNbrAtom(atom);
                const int idx = end->GetIdx();
                if (std::find(path.begin(), path.end(), idx) != path.end() ||
                    bond->GetBondOrder() != order || end->GetFormalCharge() != 0 ||
                    end->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*end) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*end))
                    continue;
                if (end->GetAtomicNum() == 6 && end->GetTotalValence() == 4)
                {
                    if (path.size() >= max_atoms) return std::nullopt;
                    auto next = path;
                    next.push_back(idx);
                    pending.push_back(std::move(next));
                    continue;
                }
                const int z = end->GetAtomicNum();
                const auto info = molgr::kNonMetalDict.find(z);
                if (order != 1 || (z != 7 && z != 8 && z != 15 && z != 16) ||
                    info == molgr::kNonMetalDict.end() ||
                    info->second.num_outer_electrons + static_cast<int>(end->GetTotalValence()) != 8 ||
                    info->second.num_outer_electrons - static_cast<int>(end->GetTotalValence()) < 2)
                    continue;
                bool adjacent_anion = false;
                FOR_NBORS_OF_ATOM(neighbor, end)
                    if (neighbor->GetFormalCharge() < 0) adjacent_anion = true;
                if (!adjacent_anion) continue;
                if (path.size() >= max_atoms) return std::nullopt;
                auto complete = path;
                complete.push_back(idx);
                paths.push_back(std::move(complete));
                if (paths.size() > max_paths) return std::nullopt;
            }
        }
        return paths;
    }

    /// Flag a zero-valent metal assignment with an unsupported residual organic cation.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
    /// @param total_charge Target net formal charge of the full input.
    /// @param coordinated_nitrosonium_indices Indices of assigned nitrosonium sites in the ligand graph.
    /// @param config Settings for the current reconstruction or scoring operation.
    /// @return Penalty flag (zero or one) for the specified cation mismatch.
    int ZeroValentMetalsWithOrganicCationCount(
        OpenBabel::OBMol &mol,
        const std::vector<molgr::MetalAtomPosition> &metal_states,
        int total_charge,
        const std::set<int> &coordinated_nitrosonium_indices,
        const molgr::config::MolGRConfig &config)
    {
        if (total_charge > 0 || metal_states.empty()) return 0;
        for (const auto &metal_state : metal_states)
            if (metal_state.valence != 0) return 0;
        const auto uncompensated = UncompensatedOrganicCationIndices(
            mol, coordinated_nitrosonium_indices, config);
        if (uncompensated.empty()) return 0;
        if (uncompensated.size() != 1) return 1;
        const auto *source = mol.GetAtom(*uncompensated.begin());
        const auto info = molgr::kNonMetalDict.find(source->GetAtomicNum());
        if (info == molgr::kNonMetalDict.end() || source->GetFormalCharge() != 1 ||
            source->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*source) ||
            molgr::utils::HasUnresolvedTwoElectronCenter(*source) ||
            info->second.num_outer_electrons - 1 + static_cast<int>(source->GetTotalValence()) != 8 ||
            info->second.num_outer_electrons - 1 - static_cast<int>(source->GetTotalValence()) < 0)
            return 1;
        // Enumerate completely before granting credit; bounds abstain instead
        // of returning an atom-order-dependent partial witness.
        const auto paths = CationCompensationPaths(mol, source->GetIdx());
        if (!paths.has_value()) return 1;
        for (const auto &path : *paths)
        {
            auto witness = molgr::vendor::openbabel_threading::SharedMolCopy(mol);
            witness->GetAtom(path.front())->SetFormalCharge(0);
            witness->GetAtom(path.back())->SetFormalCharge(1);
            for (std::size_t i = 1; i < path.size(); ++i)
            {
                auto *bond = witness->GetBond(path[i - 1], path[i]);
                bond->SetBondOrder(3 - bond->GetBondOrder());
            }
            if (UncompensatedOrganicCationIndices(
                    *witness, coordinated_nitrosonium_indices, config).empty())
                return 0;
        }
        return 1;
    }

    /// Count narrowly scoped C--O and coordinated O2 geometry conflicts.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param visible_atom_indices Indices of ligand atoms accessible from the metal under the visibility
    ///     test.
    /// @return Counts of short-carbonyl and long-dioxygen conflicts, respectively.
    std::pair<int, int> OxygenBondGeometryDiscordance(
        OpenBabel::OBMol &mol, const std::set<int> &visible_atom_indices)
    {
        // Strong geometry conflicts only: leave intermediate O--O lengths,
        // substituted peroxides, CO/CO2 and long activated C--O bonds alone.
        int short_carbonyl = 0;
        int long_dioxygen = 0;
        FOR_BONDS_OF_MOL(bond, mol)
        {
            auto *first = bond->GetBeginAtom();
            auto *second = bond->GetEndAtom();
            if (first->GetAtomicNum() > second->GetAtomicNum())
                std::swap(first, second);
            const double length = first->GetDistance(second);
            bool carbon_dioxide = first->GetAtomicNum() == 6 && first->GetExplicitDegree() == 2;
            if (carbon_dioxide)
            {
                FOR_NBORS_OF_ATOM(neighbor, first)
                    if (neighbor->GetAtomicNum() != 8) carbon_dioxide = false;
            }
            if (first->GetAtomicNum() == 6 && second->GetAtomicNum() == 8 &&
                first->GetExplicitDegree() >= 2 && !carbon_dioxide &&
                second->GetExplicitDegree() == 1 && second->GetImplicitHCount() == 0 &&
                length > 1.10 && length < 1.30 && bond->GetBondOrder() == 1 &&
                first->GetFormalCharge() < 0 && second->GetFormalCharge() < 0)
                ++short_carbonyl;
            if (first->GetAtomicNum() == 8 && second->GetAtomicNum() == 8 &&
                first->GetExplicitDegree() == 1 && second->GetExplicitDegree() == 1 &&
                first->GetImplicitHCount() == 0 && second->GetImplicitHCount() == 0 &&
                (visible_atom_indices.count(first->GetIdx()) || visible_atom_indices.count(second->GetIdx())) &&
                length >= 1.40 && length <= 1.65 && bond->GetBondOrder() > 1)
                ++long_dioxygen;
        }
        return {short_carbonyl, long_dioxygen};
    }

    /// Count stretched C=O only when C and O are visible to the same metal.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param visible_atoms_by_metal Ligand atoms visible from each metal site, grouped by metal.
    /// @return Number of eligible bonds that conflict with the observed geometry.
    int LongCarbonylMultipleBondCount(
        OpenBabel::OBMol &mol,
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal)
    {
        int count = 0;
        FOR_BONDS_OF_MOL(bond, mol)
        {
            auto *carbon = bond->GetBeginAtom();
            auto *oxygen = bond->GetEndAtom();
            if (bond->GetBondOrder() != 2 ||
                !molgr::bond_geometry::IsElongatedCarbonylPair(*carbon, *oxygen))
                continue;
            for (const auto &atoms : visible_atoms_by_metal)
            {
                if (std::find(atoms.begin(), atoms.end(), carbon) != atoms.end() &&
                    std::find(atoms.begin(), atoms.end(), oxygen) != atoms.end())
                {
                    ++count;
                    break;
                }
            }
        }
        return count;
    }

    /// Penalize localized sulfur pi bonds contradicted by a chelating skeleton.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param visible_atoms_by_metal Ligand atoms visible from each metal site, grouped by metal.
    /// @return Number of eligible bonds that conflict with the observed geometry.
    int SulfurBondGeometryDiscordance(
        OpenBabel::OBMol &mol,
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal)
    {
        /// Local chelate geometry, not a metal oxidation-state prior. Exclude
        /// isolated thiones, thioamides and charge-separated ring C=S sites.
        const auto visible = [&](OpenBabel::OBAtom *a, OpenBabel::OBAtom *b) {
            for (const auto &atoms : visible_atoms_by_metal)
                if (std::find(atoms.begin(), atoms.end(), a) != atoms.end() &&
                    std::find(atoms.begin(), atoms.end(), b) != atoms.end()) return true;
            return false;
        };
        /// Collect terminal chalcogen neighbors of this carbon for comparing localized bond orders.
        const auto terminal_chalcogens = [](OpenBabel::OBAtom *carbon) {
            std::vector<OpenBabel::OBAtom *> result;
            FOR_NBORS_OF_ATOM(atom, carbon)
            {
                const int element = atom->GetAtomicNum();
                if (element != 16 && element != 34) continue;
                // Compare Se with the same S window after correcting the
                // covalent-radius difference; C-C geometry stays unchanged.
                const double distance = carbon->GetDistance(&*atom) -
                    (OpenBabel::OBElements::GetCovalentRad(element) -
                     OpenBabel::OBElements::GetCovalentRad(16));
                if (atom->GetExplicitDegree() == 1 &&
                    atom->GetImplicitHCount() == 0 && atom->GetFormalCharge() == 0 &&
                    carbon->GetBond(&*atom)->GetBondOrder() == 2 &&
                    distance >= 1.72 && distance <= 1.90) result.push_back(&*atom);
            }
            return result;
        };
        int count = 0;
        FOR_BONDS_OF_MOL(bond, mol)
        {
            auto *left = bond->GetBeginAtom();
            auto *right = bond->GetEndAtom();
            const double distance = left->GetDistance(right);
            if (left->GetAtomicNum() == 6 && right->GetAtomicNum() == 6 &&
                bond->GetBondOrder() == 1 && distance >= 1.25 && distance <= 1.44)
            {
                // N-to-C=S conjugation lengthens both thione bonds. It does not
                // by itself establish reduction to a dithiolate ligand.
                if (molgr::bond_geometry::HasConjugatedThioamideDonor(left) &&
                    molgr::bond_geometry::HasConjugatedThioamideDonor(right)) continue;
                bool conflict = false;
                for (auto *a : terminal_chalcogens(left))
                    for (auto *b : terminal_chalcogens(right)) conflict = conflict || visible(a, b);
                count += conflict;
            }
            if (left->GetAtomicNum() == 16) std::swap(left, right);
            if (left->GetAtomicNum() != 6 || right->GetAtomicNum() != 16 ||
                bond->GetBondOrder() != 2 || left->GetFormalCharge() != 0 ||
                right->GetFormalCharge() <= 0 || left->GetExplicitDegree() != 3 ||
                left->GetImplicitHCount() != 0 || distance < 1.80 || distance > 2.05 ||
                !visible(left, left)) continue;
            bool all_sulfur = true;
            FOR_NBORS_OF_ATOM(atom, left) all_sulfur = all_sulfur && atom->GetAtomicNum() == 16;
            count += all_sulfur;
        }
        return count;
    }

    /// Nitrosyl ranking penalty, conflict count and indices of candidate nitrosonium sites.
    struct NitrosylGeometryMetrics
    {
        double penalty = 0.0;
        std::set<int> nitrosonium_indices;
        int count = 0;
    };

    /// Measure metal-nitrosyl geometry and charge consistency for the visible NO sites.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
    /// @param visible_atoms_by_metal Ligand atoms visible from each metal site, grouped by metal.
    /// @return Nitrosyl geometry metrics and their ranking penalty.
    NitrosylGeometryMetrics GetNitrosylGeometryMetrics(
        OpenBabel::OBMol &mol,
        const std::vector<molgr::MetalAtomPosition> &metal_states,
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal)
    {
        std::vector<molgr::MetalSite> sites;
        for (const auto &metal : metal_states) sites.push_back(metal.Site());
        NitrosylGeometryMetrics result;
        for (const auto &contact : molgr::coordination::TerminalNitrosylContacts(mol, sites, visible_atoms_by_metal))
        {
            auto &nitrogen = *contact.nitrogen;
            auto *oxygen = contact.oxygen;
            const double weight = contact.weight;
            const int charge = nitrogen.GetFormalCharge() + oxygen->GetFormalCharge();
            const double displacement = (charge - 1) / 2.0;
            result.penalty += weight * displacement * displacement;
            ++result.count;
            auto *bond = nitrogen.GetBond(oxygen);
            if (charge == 1 && bond && bond->GetBondOrder() == 3 &&
                molgr::utils::GetUnpairedElectronCount(nitrogen) == 0 &&
                molgr::utils::GetUnpairedElectronCount(*oxygen) == 0)
            {
                result.nitrosonium_indices.insert(nitrogen.GetIdx());
                result.nitrosonium_indices.insert(oxygen->GetIdx());
            }
        }
        return result;
    }

    /// @param candidate Candidate electronic assignment being evaluated.
    /// @param config Settings for the current reconstruction or scoring operation.
    /// @return The nitrosyl contribution to ligand candidate ranking.
    double LigandNitrosylRankingPenalty(molgr::state::MetalCandidateState &candidate,
                                       const molgr::config::MolGRConfig &config)
    {
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        bool potential = false;
        FOR_ATOMS_OF_MOL(atom, mol)
        {
            if (atom->GetAtomicNum() != 7 || atom->GetExplicitDegree() != 1 || atom->GetImplicitHCount() != 0) continue;
            FOR_NBORS_OF_ATOM(neighbor, &*atom)
            {
                if (neighbor->GetAtomicNum() == 8 && neighbor->GetExplicitDegree() == 1 &&
                    neighbor->GetImplicitHCount() == 0) potential = true;
            }
            if (potential) break;
        }
        if (!potential) return 0.0;
        const auto blockers = BuildCoordinationBlockers(mol, config.metal_scoring);
        std::vector<std::vector<OpenBabel::OBAtom *>> groups;
        for (const auto &metal : candidate.metal_states) {
            auto &visible = groups.emplace_back();
            FOR_ATOMS_OF_MOL(atom, mol) {
                if (!atom->IsMetal() && IsInnerSphereAtom(*atom, metal.Site(),config.metal_scoring) &&
                    IsVisibleToMetalAtom(*atom,metal.Site(),blockers)) visible.push_back(&*atom);
            }
        }
        return GetNitrosylGeometryMetrics(mol,candidate.metal_states,groups).penalty;
    }

    /// Flag an unsaturated organic cation that lacks a compensating local electronic environment.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @return Penalty flag (zero or one) for the specified cation mismatch.
    int UnsaturatedOrganicCationDiscordanceCount(OpenBabel::OBMol &mol)
    {
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom &atom = *atom_iter;
            if (IsUnsaturatedOrganicCation(atom) &&
                !molgr::vendor::openbabel_threading::AtomIsAromatic(atom) &&
                !IsLocallyChargeCompensatedNonmetalCation(atom))
            {
                return 1;
            }
        }
        return 0;
    }

    /// Encode atom and bond labels outside the excluded nitrosyl sites for spectator comparison.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param excluded Motif atom indices whose local electronic labels are excluded from the spectator
    ///     comparison.
    /// @return Fingerprint of the ligand outside the nitrosyl motif.
    std::string NonNitrosylSignature(OpenBabel::OBMol &mol, const std::set<int> &excluded)
    {
        std::ostringstream out;
        out << mol.NumAtoms() << ";";
        FOR_ATOMS_OF_MOL(atom, mol)
        {
            if (excluded.count(atom->GetIdx())) continue;
            out << atom->GetIdx() << "," << atom->GetAtomicNum() << ","
                << atom->GetFormalCharge() << "," << atom->GetImplicitHCount() << ","
                << molgr::utils::GetUnpairedElectronCount(*atom) << ","
                << molgr::utils::GetLonePairCount(*atom) << ","
                << molgr::utils::HasUnresolvedTwoElectronCenter(*atom) << ";";
        }
        std::vector<std::tuple<int, int, int>> bonds;
        FOR_BONDS_OF_MOL(bond, mol)
        {
            const int a = bond->GetBeginAtomIdx(), b = bond->GetEndAtomIdx();
            if (!excluded.count(a) && !excluded.count(b))
                bonds.emplace_back(std::min(a, b), std::max(a, b), bond->GetBondOrder());
        }
        std::sort(bonds.begin(), bonds.end());
        for (const auto &[a, b, order] : bonds) out << a << "," << b << "," << order << ";";
        return out.str();
    }

    /// @param source Molecule supplying the initial topology and electronic labels.
    /// @return Aromatic atom indices after reperceiving aromaticity on a working copy.
    std::set<int> FreshAromaticAtomIndices(const OpenBabel::OBMol &source)
    {
        auto mol = molgr::utils::CloneMolTopologyShared(source);
        molgr::vendor::openbabel_threading::ResetAndAssignAromaticFlags(*mol);
        std::set<int> indices;
        FOR_ATOMS_OF_MOL(atom, *mol)
            if (molgr::vendor::openbabel_threading::AtomIsAromatic(*atom))
                indices.insert(atom->GetIdx());
        return indices;
    }

    /// Keep the angle penalty when a viable aromatic alternative exists locally.
    /// @param candidates Candidate assignments under comparison.
    void AnnotateBentAlleneAromaticAlternatives(
        std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataInt;
        using molgr::metal::scoring::MetadataDouble;
        // Snapshot independent discordance before applying local aromatic
        // protection, so iteration order and repeated selection are immaterial.
        std::vector<double> independent_counts;
        for (auto &candidate : candidates)
        {
            const int previous = MetadataInt(candidate,
                "metal_visible_bent_allene_aromatic_alternative_count");
            if (previous)
            {
                candidate.metadata["metal_discordance_structural_count"] =
                    MetadataDouble(candidate, "metal_discordance_structural_count") - previous;
                candidate.metadata["metal_discordance_bent_cumulated_ring_allene_count"] =
                    MetadataInt(candidate, "metal_discordance_bent_cumulated_ring_allene_count") - previous;
            }
            candidate.metadata["metal_visible_bent_allene_aromatic_alternative_count"] = 0;
            independent_counts.push_back(
                MetadataDouble(candidate, "metal_discordance_structural_count") -
                MetadataInt(candidate, "metal_discordance_bent_cumulated_ring_allene_count") -
                MetadataInt(candidate, "nitrosonium_redistribution_exception"));
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            const auto it = candidate.metadata.find("metal_visible_bent_allene_indices");
            if (it == candidate.metadata.end()) continue;
            const auto *text = std::get_if<std::string>(&it->second);
            if (!text) continue;
            std::set<int> indices, protected_indices;
            std::istringstream stream(*text);
            std::string part;
            while (std::getline(stream, part, ','))
                if (!part.empty()) indices.insert(std::stoi(part));
            if (indices.empty() || !candidate.no_metal_state) continue;
            auto &source = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
            std::map<int, std::set<int>> components;
            for (const int index : indices)
            {
                auto &component = components[index];
                std::vector<int> pending{index};
                while (!pending.empty())
                {
                    const int current = pending.back();
                    pending.pop_back();
                    if (!component.insert(current).second) continue;
                    FOR_NBORS_OF_ATOM(neighbor, source.GetAtom(current))
                        if (!component.count(neighbor->GetIdx())) pending.push_back(neighbor->GetIdx());
                }
            }
            const auto own_aromatic = FreshAromaticAtomIndices(source);
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                if (!candidates[j].no_metal_state || independent_counts[j] > independent_counts[i])
                    continue;
                const auto aromatic = FreshAromaticAtomIndices(candidates[j].no_metal_state->Mol());
                for (const auto &[index, component] : components)
                {
                    int own_count = 0, partner_count = 0;
                    for (const int atom_index : component)
                    {
                        own_count += own_aromatic.count(atom_index);
                        partner_count += aromatic.count(atom_index);
                    }
                    if (aromatic.count(index) || partner_count > own_count)
                        protected_indices.insert(index);
                }
            }
            const int count = static_cast<int>(protected_indices.size());
            if (count)
            {
                candidate.metadata["metal_discordance_structural_count"] =
                    MetadataDouble(candidate, "metal_discordance_structural_count") + count;
                candidate.metadata["metal_discordance_bent_cumulated_ring_allene_count"] =
                    MetadataInt(candidate, "metal_discordance_bent_cumulated_ring_allene_count") + count;
            }
            candidate.metadata["metal_visible_bent_allene_aromatic_alternative_count"] = count;
        }
    }

    using CarbonSite = std::array<int, 3>;
    using CarbonPair = std::pair<int, int>;
    using CarbonSiteMap = std::map<CarbonSite, CarbonPair>;

    /// Compare a localized alkyne and a reduced C=C pair in the same metal-visible C₃ chain.
    /// @param candidate Candidate electronic assignment being evaluated.
    /// @return Reduced and localized electronic assignments for the same equalized C3 motif.
    std::pair<CarbonSiteMap, CarbonSiteMap> EqualizedCarbonElectronicSites(
        const molgr::state::MetalCandidateState &candidate)
    {
        CarbonSiteMap reduced, localized;
        if (!candidate.no_metal_state) return {reduced, localized};
        const auto it = candidate.metadata.find("metal_visible_atom_groups");
        if (it == candidate.metadata.end()) return {reduced, localized};
        const auto *text = std::get_if<std::string>(&it->second);
        if (!text) return {reduced, localized};
        std::vector<std::set<int>> groups;
        std::istringstream group_stream(*text);
        std::string group_text;
        while (std::getline(group_stream, group_text, ';'))
        {
            std::set<int> group;
            std::istringstream atom_stream(group_text);
            std::string index;
            while (std::getline(atom_stream, index, ','))
                if (!index.empty()) group.insert(std::stoi(index));
            groups.push_back(std::move(group));
        }
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        for (const auto &atoms : molgr::bond_geometry::EqualizedThreeCarbonChains(mol))
        {
            CarbonSite site{static_cast<int>(atoms[0]->GetIdx()),
                            static_cast<int>(atoms[1]->GetIdx()),
                            static_cast<int>(atoms[2]->GetIdx())};
            std::sort(site.begin(), site.end());
            if (!std::any_of(groups.begin(), groups.end(), [&](const auto &group) {
                return std::all_of(site.begin(), site.end(), [&](int index) { return group.count(index); });
            })) continue;
            for (const int end : {0, 2})
            {
                auto *first = atoms[end];
                auto *middle = atoms[1];
                auto *other = atoms[2 - end];
                if (middle->GetBond(other)->GetBondOrder() != 1 ||
                    !IsExplicitSingletTwoElectronCenter(*other)) continue;
                CarbonPair pair{first->GetIdx(), middle->GetIdx()};
                if (pair.first > pair.second) std::swap(pair.first, pair.second);
                const int order = first->GetBond(middle)->GetBondOrder();
                if (order == 2 && first->GetFormalCharge() == -1 && middle->GetFormalCharge() == -1)
                    reduced[site] = pair;
                if (order == 3 && first->GetFormalCharge() == 0 && middle->GetFormalCharge() == 0)
                    localized[site] = pair;
            }
        }
        return {reduced, localized};
    }

    /// Admit a reduced pair at an equalized three-carbon metal-bound site.
    /// @param candidates Candidate assignments under comparison.
    void AnnotateEqualizedCarbonAlternatives(
        std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataInt;
        using molgr::metal::scoring::MetadataDouble;
        std::vector<std::pair<CarbonSiteMap, CarbonSiteMap>> features;
        std::set<CarbonSite> supported;
        for (const auto &candidate : candidates)
        {
            features.push_back(EqualizedCarbonElectronicSites(candidate));
            for (const auto &[site, pair] : features.back().first) supported.insert(site);
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            const int previous = MetadataInt(candidate, "metal_equalized_carbon_localization_count") -
                MetadataInt(candidate, "metal_equalized_carbon_reduction_exemption_count");
            std::set<CarbonPair> exemptions, conflicts;
            for (const auto &[site, pair] : features[i].first) exemptions.insert(pair);
            for (const auto &[site, pair] : features[i].second)
                if (supported.count(site)) conflicts.insert(pair);
            const int exempt_count = static_cast<int>(exemptions.size());
            const int conflict_count = static_cast<int>(conflicts.size());
            candidate.metadata["metal_discordance_structural_count"] =
                MetadataDouble(candidate, "metal_discordance_structural_count",
                    MetadataDouble(candidate, "metal_discordance_count")) +
                conflict_count - exempt_count - previous;
            candidate.metadata["metal_equalized_carbon_reduction_exemption_count"] = exempt_count;
            candidate.metadata["metal_equalized_carbon_localization_count"] = conflict_count;
        }
    }

    using AlkoxySite = std::array<int, 5>;
    using AlkoxySites = std::set<AlkoxySite>;

    /// Find nonaromatic alkoxy/oxonium alternatives at one metal-visible C3 site.
    /// @param candidate Candidate electronic assignment being evaluated.
    /// @return Reduced and inverted assignments for the matched alkoxy-allylic sites.
    std::pair<AlkoxySites, AlkoxySites> AlkoxyAllylicElectronicSites(
        const molgr::state::MetalCandidateState &candidate)
    {
        AlkoxySites reduced, inverted;
        if (!candidate.no_metal_state) return {reduced, inverted};
        const auto it = candidate.metadata.find("metal_visible_atom_groups");
        if (it == candidate.metadata.end()) return {reduced, inverted};
        const auto *text = std::get_if<std::string>(&it->second);
        if (!text) return {reduced, inverted};
        std::vector<std::set<int>> groups;
        std::istringstream stream(*text);
        std::string group_text;
        while (std::getline(stream, group_text, ';'))
        {
            std::set<int> group;
            std::istringstream atoms(group_text);
            std::string index;
            while (std::getline(atoms, index, ','))
                if (!index.empty()) group.insert(std::stoi(index));
            groups.push_back(std::move(group));
        }
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto aromatic = FreshAromaticAtomIndices(mol);
        FOR_ATOMS_OF_MOL(mid, mol)
        {
            auto *middle = &*mid;
            if (middle->GetAtomicNum() != 6) continue;
            std::vector<OpenBabel::OBAtom *> neighbors;
            FOR_NBORS_OF_ATOM(n, middle)
                if (n->GetAtomicNum() == 6) neighbors.push_back(&*n);
            for (std::size_t i = 0; i < neighbors.size(); ++i)
                for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                {
                    std::array<OpenBabel::OBAtom *, 3> atoms{neighbors[i], middle, neighbors[j]};
                    if (std::any_of(atoms.begin(), atoms.end(), [&](auto *a) {
                        return aromatic.count(a->GetIdx()) || a->GetExplicitDegree() != 3 ||
                            a->GetImplicitHCount() != 0 || molgr::utils::GetUnpairedElectronCount(*a);
                    })) continue;
                    if (!std::any_of(groups.begin(), groups.end(), [&](const auto &group) {
                        return std::all_of(atoms.begin(), atoms.end(), [&](auto *a) {
                            return group.count(a->GetIdx());
                        });
                    })) continue;
                    auto *first = neighbors[i];
                    auto *last = neighbors[j];
                    if (!(first->GetDistance(middle) > last->GetDistance(middle))) std::swap(first, last);
                    const double longer = first->GetDistance(middle), shorter = last->GetDistance(middle);
                    if (!(longer > 1.44 && longer < 1.55 && shorter > 1.30 && shorter < 1.43 &&
                          longer - shorter > 0.04)) continue;
                    const int first_order = first->GetBond(middle)->GetBondOrder();
                    const int last_order = middle->GetBond(last)->GetBondOrder();
                    FOR_NBORS_OF_ATOM(o, last)
                    {
                        auto *oxygen = &*o;
                        const double co_length = oxygen->GetDistance(last);
                        if (oxygen->GetAtomicNum() != 8 || oxygen->GetExplicitDegree() != 2 ||
                            oxygen->GetImplicitHCount() != 0 || molgr::utils::GetUnpairedElectronCount(*oxygen) ||
                            !(co_length > 1.30 && co_length < 1.42)) continue;
                        OpenBabel::OBAtom *cap = nullptr;
                        FOR_NBORS_OF_ATOM(n, oxygen)
                            if (n->GetIdx() != last->GetIdx()) cap = &*n;
                        if (!cap || cap->GetAtomicNum() != 6 || cap->GetExplicitDegree() != 4 ||
                            oxygen->GetBond(cap)->GetBondOrder() != 1) continue;
                        AlkoxySite site{static_cast<int>(first->GetIdx()), static_cast<int>(middle->GetIdx()),
                            static_cast<int>(last->GetIdx()), static_cast<int>(oxygen->GetIdx()),
                            static_cast<int>(cap->GetIdx())};
                        const int co_order = last->GetBond(oxygen)->GetBondOrder();
                        if (middle->GetFormalCharge() != 0 || last->GetFormalCharge() != 0) continue;
                        if (first_order == 1 && last_order == 2 && first->GetFormalCharge() == -1 &&
                            oxygen->GetFormalCharge() == 0 && co_order == 1) reduced.insert(site);
                        if (first_order == 2 && last_order == 1 && first->GetFormalCharge() == 0 &&
                            oxygen->GetFormalCharge() == 1 && co_order == 2) inverted.insert(site);
                    }
                }
        }
        return {reduced, inverted};
    }

    /// Remove previously recorded conflicts for this ligand motif before comparing alternatives.
    /// @param candidates Candidate assignments under comparison.
    void ClearAlkoxyAllylicGeometryConflicts(std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataInt;
        using molgr::metal::scoring::MetadataDouble;
        for (auto &candidate : candidates)
        {
            const int previous = MetadataInt(candidate, "metal_alkoxy_allylic_geometry_conflict_count");
            for (const auto *key : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[key] = MetadataDouble(candidate, key) - previous;
            candidate.metadata["metal_alkoxy_allylic_geometry_conflict_count"] = 0;
        }
    }

    /// Prefer a supported reduced ether over its inverted oxonium assignment.
    /// @param candidates Candidate assignments under comparison.
    void AnnotateAlkoxyAllylicAlternatives(std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataDouble;
        ClearAlkoxyAllylicGeometryConflicts(candidates);
        std::vector<std::pair<AlkoxySites, AlkoxySites>> features;
        std::vector<double> independent_counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(AlkoxyAllylicElectronicSites(candidate));
            independent_counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            AlkoxySites supported;
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (candidate.metal_states.size() != 1 || partner.metal_states.size() != 1 ||
                    partner.no_metal_radical_target != candidate.no_metal_radical_target ||
                    independent_counts[j] > independent_counts[i]) continue;
                const auto &metal = candidate.metal_states[0], &other = partner.metal_states[0];
                if (metal.idx != other.idx || other.valence != metal.valence + 2 ||
                    other.radical_num != metal.radical_num) continue;
                for (const auto &site : features[i].second)
                    if (features[j].first.count(site)) supported.insert(site);
            }
            const int count = static_cast<int>(supported.size());
            candidate.metadata["metal_alkoxy_allylic_geometry_conflict_count"] = count;
            for (const auto *key : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[key] = MetadataDouble(candidate, key) + count;
        }
    }

    // A complete local formazan site is compared only with an existing,
    // charge-balanced alternative; all atoms/bonds outside it must agree.
    using FormazanElectronicSites = std::map<std::vector<int>, bool>; // true: reduced

    /// Identify formazan ligand sites and their electron assignments in a visible donor group.
    /// @param candidate Candidate electronic assignment being evaluated.
    /// @return Formazan sites and their supported ligand occupations.
    FormazanElectronicSites GetFormazanElectronicSites(const molgr::state::MetalCandidateState &candidate)
    {
        FormazanElectronicSites result;
        if (!candidate.no_metal_state) return result;
        const auto found = candidate.metadata.find("metal_visible_atom_groups");
        if (found == candidate.metadata.end()) return result;
        const auto *text = std::get_if<std::string>(&found->second);
        if (!text) return result;
        std::vector<std::set<int>> groups;
        std::istringstream stream(*text);
        std::string group_text;
        while (std::getline(stream, group_text, ';'))
        {
            std::set<int> group;
            std::istringstream atoms(group_text);
            std::string index;
            while (std::getline(atoms, index, ',')) if (!index.empty()) group.insert(std::stoi(index));
            groups.push_back(std::move(group));
        }
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        auto sites = molgr::bond_geometry::PlanarBisPhenoxyFormazanSites(mol);
        const auto mono_sites = molgr::bond_geometry::PlanarPhenoxyFormazanSites(mol, 1);
        sites.insert(sites.end(), mono_sites.begin(), mono_sites.end());
        for (const auto &site : sites)
        {
            std::set<int> donors;
            bool valid = true;
            auto *center = mol.GetAtom(site.center);
            int core_charge = center->GetFormalCharge();
            if (!((core_charge == 0 && center->GetTotalValence() == 4) ||
                  (core_charge == -1 && center->GetTotalValence() == 3))) continue;
            for (const auto &arm : site.arms)
            {
                donors.insert(arm.atoms[1]);
                if (arm.atoms[4]) donors.insert(arm.atoms[4]);
                for (int i : arm.ring)
                    if (mol.GetAtom(i)->GetFormalCharge() != 0 || mol.GetAtom(i)->GetTotalValence() != 4) valid = false;
                for (int j = 0; j < 2; ++j)
                {
                    auto *nitrogen = mol.GetAtom(arm.atoms[j]);
                    const int charge = nitrogen->GetFormalCharge(), valence = nitrogen->GetTotalValence();
                    if (!((charge == 0 && valence == 3) || (charge == -1 && valence == 2))) valid = false;
                    core_charge += charge;
                }
            }
            if (!valid || !std::any_of(groups.begin(), groups.end(), [&](const auto &group) {
                return std::includes(group.begin(), group.end(), donors.begin(), donors.end());
            })) continue;
            int reduced = 0, oxidized = 0;
            for (const auto &arm : site.arms)
            {
                const auto &a = arm.atoms;
                const std::array<int, 4> order{
                    static_cast<int>(mol.GetBond(site.center, a[0])->GetBondOrder()),
                    static_cast<int>(mol.GetBond(a[0], a[1])->GetBondOrder()),
                    static_cast<int>(mol.GetBond(a[1], a[2])->GetBondOrder()),
                    a[4] ? static_cast<int>(mol.GetBond(a[3], a[4])->GetBondOrder()) : 1};
                auto *oxygen = a[4] ? mol.GetAtom(a[4]) : nullptr;
                // Admit both central C- and terminal N- resonance contributors.
                if ((order[0] == 1 || order[0] == 2) && (order[1] == 1 || order[1] == 2) &&
                    order[0] + order[1] <= 3 && order[2] == 1 && order[3] == 1 &&
                    (!oxygen || (oxygen->GetFormalCharge() == -1 && oxygen->GetTotalValence() == 1))) ++reduced;
                if (oxygen && order == std::array<int, 4>{2, 1, 2, 2} &&
                    oxygen->GetFormalCharge() == 0 && oxygen->GetTotalValence() == 2) ++oxidized;
            }
            if (reduced == 2 && core_charge == -1)
                result[site.atom_indices] = true;
            if (reduced == 1 && oxidized == 1 && core_charge == 0)
                result[site.atom_indices] = false;
        }
        return result;
    }

    /// @param first First ligand assignment in the spectator comparison.
    /// @param second Second ligand assignment in the spectator comparison.
    /// @param site Local ligand motif under comparison.
    /// @return True when the two ligand graphs have identical electronic labels outside this formazan site.
    bool SameOutsideFormazanSite(const molgr::state::MetalCandidateState &first,
        const molgr::state::MetalCandidateState &second, const std::vector<int> &site)
    {
        const std::set<int> inside(site.begin(), site.end());
        /// Encode the ligand outside the selected formazan site so spectator changes cannot bias the
        /// comparison.
        const auto signature = [&](const molgr::state::MetalCandidateState &candidate) {
            auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
            std::vector<std::array<int, 6>> atoms;
            std::vector<std::array<int, 3>> bonds;
            FOR_ATOMS_OF_MOL(atom, mol)
            {
                if (inside.count(atom->GetIdx())) continue;
                atoms.push_back({static_cast<int>(atom->GetIdx()), static_cast<int>(atom->GetAtomicNum()),
                    atom->GetFormalCharge(), molgr::utils::GetUnpairedElectronCount(*atom),
                    static_cast<int>(molgr::utils::HasUnresolvedTwoElectronCenter(*atom)),
                    static_cast<int>(atom->GetImplicitHCount())});
            }
            FOR_BONDS_OF_MOL(bond, mol)
            {
                const int a = bond->GetBeginAtomIdx(), b = bond->GetEndAtomIdx();
                if (!inside.count(a) || !inside.count(b))
                    bonds.push_back({std::min(a, b), std::max(a, b), static_cast<int>(bond->GetBondOrder())});
            }
            std::sort(bonds.begin(), bonds.end());
            return std::make_pair(atoms, bonds);
        };
        return signature(first) == signature(second);
    }

    /// Remove previously recorded conflicts for this ligand motif before comparing alternatives.
    /// @param candidates Candidate assignments under comparison.
    void ClearFormazanGeometryConflicts(std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataInt;
        using molgr::metal::scoring::MetadataDouble;
        for (auto &candidate : candidates)
        {
            const int previous = MetadataInt(candidate, "metal_formazan_geometry_conflict_count");
            for (const auto *key : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[key] = MetadataDouble(candidate, key) - previous;
            candidate.metadata["metal_formazan_geometry_conflict_count"] = 0;
        }
    }

    /// Prefer a complete reduced site only against an existing +2-metal partner.
    /// @param candidates Candidate assignments under comparison.
    void AnnotateFormazanAlternatives(std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataDouble;
        ClearFormazanGeometryConflicts(candidates);
        std::vector<FormazanElectronicSites> features;
        std::vector<double> counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(GetFormazanElectronicSites(candidate));
            counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            std::set<std::vector<int>> conflicts;
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (candidate.metal_states.size() != 1 || partner.metal_states.size() != 1 ||
                    candidate.no_metal_radical_target != partner.no_metal_radical_target || counts[j] > counts[i] ||
                    MetadataDouble(candidate, "nitrosyl_geometry_penalty") !=
                        MetadataDouble(partner, "nitrosyl_geometry_penalty")) continue;
                const auto &metal = candidate.metal_states[0], &other = partner.metal_states[0];
                if (metal.idx != other.idx || metal.radical_num != other.radical_num ||
                    other.valence - metal.valence != 2) continue;
                for (const auto &[site, reduced] : features[i])
                {
                    const auto alternative = features[j].find(site);
                    if (!reduced && alternative != features[j].end() && alternative->second &&
                        SameOutsideFormazanSite(candidate, partner, site)) conflicts.insert(site);
                }
            }
            const int count = static_cast<int>(conflicts.size());
            candidate.metadata["metal_formazan_geometry_conflict_count"] = count;
            for (const auto *key : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[key] = MetadataDouble(candidate, key) + count;
        }
    }

    using FoldedRingSite = std::array<int, 4>;
    using FoldedRingSites = std::set<FoldedRingSite>;

    /// Split folded C₈ segments by their charge and bond-order patterns for paired redox comparison.
    /// @param candidate Candidate electronic assignment being evaluated.
    /// @return Reduced and neutral electronic forms on the folded ring.
    std::pair<FoldedRingSites, FoldedRingSites> FoldedRingElectronicSites(
        const molgr::state::MetalCandidateState &candidate)
    {
        FoldedRingSites reduced, neutral;
        if (!candidate.no_metal_state) return {reduced, neutral};
        const auto it = candidate.metadata.find("metal_visible_atom_groups");
        if (it == candidate.metadata.end()) return {reduced, neutral};
        const auto *text = std::get_if<std::string>(&it->second);
        if (!text) return {reduced, neutral};
        std::vector<std::set<int>> groups;
        std::istringstream stream(*text);
        std::string group_text;
        while (std::getline(stream, group_text, ';'))
        {
            std::set<int> group;
            std::istringstream atoms(group_text);
            std::string index;
            while (std::getline(atoms, index, ','))
                if (!index.empty()) group.insert(std::stoi(index));
            groups.push_back(std::move(group));
        }
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        for (const auto &ids : molgr::bond_geometry::FoldedEightCarbonSegments(mol))
        {
            FoldedRingSite site{ids[0], ids[1], ids[2], ids[3]};
            std::sort(site.begin(), site.end());
            if (!std::any_of(groups.begin(), groups.end(), [&](const auto &group) {
                return std::includes(group.begin(), group.end(), site.begin(), site.end());
            })) continue;
            std::array<int, 8> charge, order;
            bool valid = true;
            for (int i = 0; i < 8; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                charge[i] = atom->GetFormalCharge();
                const int valence = atom->GetTotalValence();
                if (!((charge[i] == 0 && valence == 4) || (charge[i] == -1 && valence == 3)))
                    { valid = false; break; }
                order[i] = atom->GetBond(mol.GetAtom(ids[(i + 1) % 8]))->GetBondOrder();
            }
            if (!valid || !std::all_of(charge.begin() + 4, charge.end(), [](int q) { return q == 0; }) ||
                order[3] != 1 || order[4] != 2 || order[5] != 1 || order[6] != 2 || order[7] != 1)
                continue;
            if (order[0] == 1 && order[1] == 2 && order[2] == 1 &&
                charge[0] == -1 && charge[1] == 0 && charge[2] == 0 && charge[3] == -1)
                reduced.insert(site);
            if (order[0] == 2 && order[1] == 1 && order[2] == 2 &&
                std::all_of(charge.begin(), charge.end(), [](int q) { return q == 0; }))
                neutral.insert(site);
        }
        return {reduced, neutral};
    }

    /// Remove previously recorded conflicts for this ligand motif before comparing alternatives.
    /// @param candidates Candidate assignments under comparison.
    void ClearFoldedRingGeometryConflicts(std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataInt;
        using molgr::metal::scoring::MetadataDouble;
        for (auto &candidate : candidates)
        {
            const int previous = MetadataInt(candidate, "metal_folded_ring_geometry_conflict_count");
            for (const auto *key : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[key] = MetadataDouble(candidate, key) - previous;
            candidate.metadata["metal_folded_ring_geometry_conflict_count"] = 0;
        }
    }

    /// Compare existing local C4 reductions at a fixed metal/site/spin budget.
    /// @param candidates Candidate assignments under comparison.
    void AnnotateFoldedRingAlternatives(std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataDouble;
        ClearFoldedRingGeometryConflicts(candidates);
        std::vector<std::pair<FoldedRingSites, FoldedRingSites>> features;
        std::vector<double> counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(FoldedRingElectronicSites(candidate));
            counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            const auto &[reduced, neutral] = features[i];
            FoldedRingSites conflicts, all = reduced;
            all.insert(neutral.begin(), neutral.end());
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (candidate.metal_states.size() != 1 || partner.metal_states.size() != 1 ||
                    candidate.no_metal_radical_target != partner.no_metal_radical_target || counts[j] > counts[i] ||
                    MetadataDouble(candidate, "nitrosyl_geometry_penalty") !=
                        MetadataDouble(partner, "nitrosyl_geometry_penalty")) continue;
                const auto &[pr, pn] = features[j];
                FoldedRingSites partner_all = pr, changed;
                partner_all.insert(pn.begin(), pn.end());
                for (const auto &site : neutral) if (pr.count(site)) changed.insert(site);
                if (changed.empty() || all != partner_all ||
                    !std::includes(pr.begin(), pr.end(), reduced.begin(), reduced.end())) continue;
                const auto &metal = candidate.metal_states[0], &other = partner.metal_states[0];
                if (metal.idx == other.idx && metal.radical_num == other.radical_num &&
                    other.valence - metal.valence == 2 * static_cast<int>(changed.size()))
                    conflicts.insert(changed.begin(), changed.end());
            }
            const int count = static_cast<int>(conflicts.size());
            candidate.metadata["metal_folded_ring_geometry_conflict_count"] = count;
            for (const auto *key : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[key] = MetadataDouble(candidate, key) + count;
        }
    }

    using MacrocycleSites = std::map<std::vector<int>, int>;

    // Called for a recognized C8N8 circuit with four pyrrolic N donors.
    /// Check all four donor triples of a recognized C8N8 circuit.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param donors One-based indices of the four nitrogen donors defining the coordination plane.
    /// @param metal Metal site being considered.
    /// @return True when the metal lies outside the accepted plane of the four donors.
    bool MetalDisplacedFromN4Plane(OpenBabel::OBMol &mol, const std::set<int> &donors,
                                   const molgr::MetalAtomPosition &metal)
    {
        const OpenBabel::vector3 site(metal.position_x, metal.position_y, metal.position_z);
        std::vector<OpenBabel::vector3> points;
        for (int idx : donors) points.push_back(mol.GetAtom(idx)->GetVector());
        // Each triple defines a plane. Require agreement even if the N4 site is puckered.
        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto &a = points[i], &b = points[(i + 1) % 4], &c = points[(i + 2) % 4];
            const auto normal = cross(b - a, c - a);
            const double scale = normal.length();
            if (!(scale > 0.01 && std::abs(dot(site - a, normal)) > 0.80 * scale))
                return false;
        }
        return true;
    }

    /// Count inner-circuit pi electrons only for resolved C/N valence labels.
    /// @param candidate Candidate electronic assignment being evaluated.
    /// @return Recognized macrocycle electron assignments and their ligand-site keys.
    MacrocycleSites MacrocycleElectronSites(const molgr::state::MetalCandidateState &candidate)
    {
        MacrocycleSites sites;
        if (!candidate.no_metal_state) return sites;
        const auto it = candidate.metadata.find("metal_visible_atom_groups");
        if (it == candidate.metadata.end()) return sites;
        const auto *text = std::get_if<std::string>(&it->second);
        if (!text) return sites;
        std::vector<std::set<int>> groups;
        std::istringstream stream(*text);
        std::string group_text;
        while (std::getline(stream, group_text, ';'))
        {
            std::set<int> group;
            std::istringstream atoms(group_text);
            std::string index;
            while (std::getline(atoms, index, ','))
                if (!index.empty()) group.insert(std::stoi(index));
            groups.push_back(std::move(group));
        }
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        auto circuits = molgr::bond_geometry::PlanarAzaMacrocycleCircuits(mol);
        for (auto circuit : molgr::bond_geometry::PlanarTetraazaporphyrinCircuits(mol))
        {
            if (candidate.metal_states.size() == 1 &&
                MetalDisplacedFromN4Plane(mol, circuit.donor_indices, candidate.metal_states.front()))
                continue;
            circuits.push_back(std::move(circuit));
        }
        for (auto ids : molgr::bond_geometry::PlanarTetrapyrroleCircuits(mol))
        {
            std::set<int> donors;
            for (int index : ids)
                if (mol.GetAtom(index)->GetAtomicNum() == 7) donors.insert(index);
            circuits.push_back({std::move(ids), std::move(donors)});
        }
        for (auto &circuit : circuits)
        {
            auto ids = circuit.atom_indices;
            bool valid = true;
            int charge = 0, electrons = 0;
            for (int index : ids)
            {
                auto *atom = mol.GetAtom(index);
                const int q = atom->GetFormalCharge(), valence = atom->GetTotalValence();
                const int neutral_valence = atom->GetAtomicNum() == 6 ? 4 : 3;
                if (molgr::utils::GetUnpairedElectronCount(*atom) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom) ||
                    !((q == 0 && valence == neutral_valence) || (q == -1 && valence == neutral_valence - 1)))
                    { valid = false; break; }
                charge += q;
                electrons += atom->GetAtomicNum() == 7 && atom->GetExplicitDegree() == 3 ? 2 : 1;
            }
            if (!valid || !std::any_of(groups.begin(), groups.end(), [&](const auto &group) {
                return std::includes(group.begin(), group.end(),
                                     circuit.donor_indices.begin(), circuit.donor_indices.end());
            })) continue;
            std::sort(ids.begin(), ids.end());
            sites.emplace(std::move(ids), electrons - charge);
        }
        return sites;
    }

    /// Remove previously recorded conflicts for this ligand motif before comparing alternatives.
    /// @param candidates Candidate assignments under comparison.
    void ClearMacrocycleElectronConflicts(std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataInt;
        using molgr::metal::scoring::MetadataDouble;
        for (auto &candidate : candidates)
        {
            const int previous = MetadataInt(candidate, "metal_macrocycle_electron_conflict_count");
            for (const auto *key : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[key] = MetadataDouble(candidate, key) - previous;
            candidate.metadata["metal_macrocycle_electron_conflict_count"] = 0;
        }
    }

    // A bounded inner-circuit model, not general aromaticity perception. Only
    // compare existing valid candidates; snapshot counts for order independence.
    /// Prefer an existing 4n+2 partner for a planar, closed-shell 4n circuit.
    /// @param candidates Candidate assignments under comparison.
    void AnnotateMacrocycleAlternatives(std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataDouble;
        ClearMacrocycleElectronConflicts(candidates);
        std::vector<MacrocycleSites> features;
        std::vector<double> counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(MacrocycleElectronSites(candidate));
            counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            std::set<std::vector<int>> conflicts;
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (candidate.metal_states.size() != 1 || partner.metal_states.size() != 1 ||
                    candidate.no_metal_radical_target != partner.no_metal_radical_target || counts[j] > counts[i] ||
                    MetadataDouble(candidate, "nitrosyl_geometry_penalty") !=
                        MetadataDouble(partner, "nitrosyl_geometry_penalty")) continue;
                const auto &metal = candidate.metal_states[0], &other = partner.metal_states[0];
                if (metal.idx != other.idx || metal.radical_num != other.radical_num) continue;
                for (const auto &[site, electrons] : features[i])
                {
                    const auto found = features[j].find(site);
                    if (found == features[j].end()) continue;
                    const int pe = found->second;
                    if (electrons % 4 == 0 && pe % 4 == 2 && std::abs(pe - electrons) == 2 &&
                        other.valence - metal.valence == pe - electrons) conflicts.insert(site);
                }
            }
            const int count = static_cast<int>(conflicts.size());
            candidate.metadata["metal_macrocycle_electron_conflict_count"] = count;
            for (const auto *key : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[key] = MetadataDouble(candidate, key) + count;
        }
    }

    /// Do not veto an isolated M(II)/NO- -> M(0)/NO+ redistribution.
    /// @param candidates Candidate assignments under comparison.
    void AnnotateNitrosoniumRedistributionExceptions(
        std::vector<molgr::state::MetalCandidateState> &candidates)
    {
        using molgr::metal::scoring::MetadataInt;
        using molgr::metal::scoring::MetadataDouble;
        for (auto &candidate : candidates)
        {
            const int previous = MetadataInt(candidate, "nitrosonium_redistribution_exception");
            if (previous)
                candidate.metadata["metal_discordance_structural_count"] =
                    MetadataDouble(candidate, "metal_discordance_structural_count") + previous;
            candidate.metadata["nitrosonium_redistribution_exception"] = 0;
            if (candidate.metal_states.size() != 1 || candidate.metal_states[0].valence != 0 ||
                MetadataInt(candidate, "nitrosyl_geometry_site_count") != 1 ||
                !MetadataInt(candidate, "metal_discordance_zero_valent_metals_with_organic_cation_count") ||
                !candidate.no_metal_state)
                continue;
            const auto it = candidate.metadata.find("nitrosonium_geometry_atom_indices");
            if (it == candidate.metadata.end()) continue;
            const auto *text = std::get_if<std::string>(&it->second);
            if (!text) continue;
            std::set<int> indices;
            std::istringstream stream(*text);
            std::string part;
            while (std::getline(stream, part, ','))
                if (!part.empty()) indices.insert(std::stoi(part));
            if (indices.size() != 2) continue;
            auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
            const auto signature = NonNitrosylSignature(mol, indices);
            for (const auto &partner : candidates)
            {
                if (partner.metal_states.size() != 1 || partner.metal_states[0].valence != 2 ||
                    partner.metal_states[0].idx != candidate.metal_states[0].idx ||
                    partner.metal_states[0].radical_num != candidate.metal_states[0].radical_num ||
                    partner.no_metal_radical_target != candidate.no_metal_radical_target ||
                    partner.no_metal_charge_target != candidate.no_metal_charge_target - 2 ||
                    MetadataInt(partner, "nitrosyl_geometry_site_count") != 1 || !partner.no_metal_state)
                    continue;
                auto &other = const_cast<OpenBabel::OBMol &>(partner.no_metal_state->Mol());
                if (other.NumAtoms() != mol.NumAtoms()) continue;
                auto *a = other.GetAtom(*indices.begin()), *b = other.GetAtom(*indices.rbegin());
                auto *bond = a->GetBond(b);
                if (std::min(a->GetAtomicNum(), b->GetAtomicNum()) == 7 &&
                    std::max(a->GetAtomicNum(), b->GetAtomicNum()) == 8 &&
                    a->GetExplicitDegree() == 1 && b->GetExplicitDegree() == 1 &&
                    a->GetImplicitHCount() == 0 && b->GetImplicitHCount() == 0 &&
                    molgr::utils::GetUnpairedElectronCount(*a) == 0 &&
                    molgr::utils::GetUnpairedElectronCount(*b) == 0 &&
                    a->GetFormalCharge() + b->GetFormalCharge() == -1 &&
                    bond && bond->GetBondOrder() == 2 && NonNitrosylSignature(other, indices) == signature)
                {
                    candidate.metadata["nitrosonium_redistribution_exception"] = 1;
                    break;
                }
            }
        }
    }

    /// Record the candidate's geometry and electronic conflicts and return its aggregate discordance.
    /// @param candidate Candidate electronic assignment being evaluated.
    /// @param config Settings for the current reconstruction or scoring operation.
    /// @return Aggregate discordance used to rank this candidate.
    double AnnotateCandidateDiscordanceFeatures(
        molgr::state::MetalCandidateState *candidate,
        const molgr::config::MolGRConfig &config)
    {
        if (candidate == nullptr || !candidate->no_metal_state)
        {
            throw std::runtime_error(
                "MetalCandidateState requires no_metal_state before discordance scoring");
        }

        auto &mol = const_cast<OpenBabel::OBMol &>(candidate->no_metal_state->Mol());
        const auto blockers = BuildCoordinationBlockers(mol, config.metal_scoring);

        int inner_visible_diradical_count = 0;
        int inner_visible_same_sign_charge_count = 0;
        int inner_visible_cationic_donor_count = 0;
        std::set<int> penalized_negative_donor_indices;
        int outer_or_invisible_adjacent_same_sign_double_charge_count = 0;
        int outer_or_invisible_adjacent_opposite_sign_double_charge_count = 0;
        int outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count = 0;
        int inner_visible_adjacent_carbanion_pair_count = 0;
        std::set<int> visible_inner_atom_indices;
        std::vector<std::vector<OpenBabel::OBAtom *>> visible_atoms_by_metal;

        for (const auto &metal_state : candidate->metal_states)
        {
            const int metal_charge_sign = ChargeSign(metal_state.valence);
            std::vector<OpenBabel::OBAtom *> visible_atoms;
            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                OpenBabel::OBAtom &atom = *atom_iter;
                if (atom.IsMetal())
                {
                    continue;
                }
                if (!IsInnerSphereAtom(atom, metal_state.Site(), config.metal_scoring))
                {
                    continue;
                }
                if (!IsVisibleToMetalAtom(atom, metal_state.Site(), blockers))
                {
                    continue;
                }

                const int atom_idx = static_cast<int>(atom.GetIdx());
                visible_inner_atom_indices.insert(atom_idx);
                visible_atoms.push_back(&atom);
                if (IsInnerVisibleDiradicalDiscordanceAtom(atom))
                {
                    ++inner_visible_diradical_count;
                }

                const int formal_charge = static_cast<int>(atom.GetFormalCharge());
                const int atom_charge_sign = ChargeSign(formal_charge);
                const bool has_inner_same_sign_charge =
                    (metal_charge_sign == 0 && formal_charge > 0) ||
                    (metal_charge_sign != 0 && atom_charge_sign == metal_charge_sign);
                if (has_inner_same_sign_charge &&
                    !(formal_charge > 0 &&
                      molgr::vendor::openbabel_threading::AtomIsAromatic(atom)) &&
                    !HasAdjacentAnionicPolarizationCancellation(atom) &&
                    !((atom.GetAtomicNum() == 7 || atom.GetAtomicNum() == 15) &&
                      molgr::utils::GetLonePairCount(atom) == 0 && HasAdjacentCarbanion(atom) &&
                      IsLocallyChargeCompensatedNonmetalCation(atom)))
                {
                    ++inner_visible_same_sign_charge_count;
                    if (formal_charge < 0) penalized_negative_donor_indices.insert(atom_idx);
                    if (IsCationicPairedElectronDonor(atom))
                    {
                        ++inner_visible_cationic_donor_count;
                    }
                }
            }
            visible_atoms_by_metal.push_back(std::move(visible_atoms));
        }

        std::ostringstream visible_groups_text;
        for (const auto &group : visible_atoms_by_metal)
        {
            for (const auto *atom : group) visible_groups_text << atom->GetIdx() << ',';
            visible_groups_text << ';';
        }
        candidate->metadata["metal_visible_atom_groups"] = visible_groups_text.str();
        candidate->metadata["metal_equalized_carbon_reduction_exemption_count"] = 0;
        candidate->metadata["metal_equalized_carbon_localization_count"] = 0;
        candidate->metadata["metal_alkoxy_allylic_geometry_conflict_count"] = 0;
        candidate->metadata["metal_macrocycle_electron_conflict_count"] = 0;
        candidate->metadata["metal_folded_ring_geometry_conflict_count"] = 0;
        candidate->metadata["metal_formazan_geometry_conflict_count"] = 0;

        std::ostringstream visible_indices_text;
        for (const int index : visible_inner_atom_indices)
        {
            if (visible_indices_text.tellp() > 0) visible_indices_text << ",";
            visible_indices_text << index;
        }
        candidate->metadata["metal_visible_atom_indices"] = visible_indices_text.str();
        int visible_singlet_two_electron_center_count = 0;
        int visible_singlet_carbene_count = 0;
        for (const int atom_idx : visible_inner_atom_indices)
        {
            OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
            if (atom != nullptr && IsExplicitSingletTwoElectronCenter(*atom))
            {
                ++visible_singlet_two_electron_center_count;
                visible_singlet_carbene_count += atom->GetAtomicNum() == 6 ? 1 : 0;
            }
        }
        const int excess_visible_singlet_two_electron_center_count = std::max(
            0,
            visible_singlet_two_electron_center_count -
                static_cast<int>(candidate->metal_states.size()));
        int bent_cumulated_ring_allene_count = 0;
        std::ostringstream visible_bent_indices;
        // Electronic edits can leave the source aromatic flags describing an
        // earlier bond-order assignment. Inspect freshly perceived topology on
        // a copy, so those flags cannot hide a bent ring cumulene or change the
        // stored candidate while computing its score.
        auto allene_geometry = molgr::utils::CloneMolTopologyShared(mol);
        molgr::vendor::openbabel_threading::ResetAndAssignAromaticFlags(*allene_geometry);
        for (const int index : BentCumulatedRingAlleneIndices(*allene_geometry))
        {
            if (AromaticRingAlleneResonanceWitness(*allene_geometry, index)) continue;
            if (visible_inner_atom_indices.count(index) &&
                IsNeutralCarbonCumuleneCenter(*allene_geometry->GetAtom(index)))
            {
                if (!visible_bent_indices.str().empty()) visible_bent_indices << ",";
                visible_bent_indices << index;
            }
            else ++bent_cumulated_ring_allene_count;
        }
        candidate->metadata["metal_visible_bent_allene_indices"] = visible_bent_indices.str();
        candidate->metadata["metal_visible_bent_allene_aromatic_alternative_count"] = 0;

        int outer_or_invisible_adjacent_double_charge_count = 0;
        const int inner_visible_conjugated_carbanion_pair_count =
            InnerVisibleConjugatedChargedCarbonPairCount(
                mol, visible_inner_atom_indices, visible_atoms_by_metal, candidate->metal_states);
        const auto negative_metal_discordance =
            NegativeMetalDiscordanceCount(mol, candidate->metal_states, config.metal_scoring);
        int total_charge = candidate->no_metal_state->total_charge;
        for (const auto &metal_state : candidate->metal_states)
        {
            total_charge += metal_state.valence;
        }
        const auto nitrosyl = GetNitrosylGeometryMetrics(
            mol, candidate->metal_states, visible_atoms_by_metal);
        candidate->metadata["nitrosyl_geometry_penalty"] = nitrosyl.penalty;
        candidate->metadata["nitrosyl_geometry_site_count"] = nitrosyl.count;
        candidate->metadata["nitrosonium_redistribution_exception"] = 0;
        std::ostringstream nitrosonium_indices;
        for (int idx : nitrosyl.nitrosonium_indices)
        {
            if (nitrosonium_indices.tellp() > 0) nitrosonium_indices << ",";
            nitrosonium_indices << idx;
        }
        candidate->metadata["nitrosonium_geometry_atom_indices"] = nitrosonium_indices.str();
        const int zero_valent_metals_with_organic_cation_count =
            ZeroValentMetalsWithOrganicCationCount(
                mol, candidate->metal_states, total_charge, nitrosyl.nitrosonium_indices, config);
        const int unsaturated_organic_cation_discordance_count =
            UnsaturatedOrganicCationDiscordanceCount(mol);
        FOR_BONDS_OF_MOL(bond_iter, mol)
        {
            OpenBabel::OBBond &bond = *bond_iter;
            OpenBabel::OBAtom *begin_atom = bond.GetBeginAtom();
            OpenBabel::OBAtom *end_atom = bond.GetEndAtom();
            if (begin_atom == nullptr || end_atom == nullptr)
            {
                continue;
            }
            const int begin_charge = static_cast<int>(begin_atom->GetFormalCharge());
            const int end_charge = static_cast<int>(end_atom->GetFormalCharge());
            const int begin_charge_sign = ChargeSign(begin_charge);
            const int end_charge_sign = ChargeSign(end_charge);
            if (begin_charge_sign == 0 || end_charge_sign == 0)
            {
                continue;
            }
            if (begin_charge_sign != end_charge_sign)
            {
                continue;
            }

            const int begin_idx = static_cast<int>(begin_atom->GetIdx());
            const int end_idx = static_cast<int>(end_atom->GetIdx());
            const bool pair_both_inner_visible =
                visible_inner_atom_indices.find(begin_idx) != visible_inner_atom_indices.end() &&
                visible_inner_atom_indices.find(end_idx) != visible_inner_atom_indices.end();
            const bool is_inner_visible_adjacent_carbanion_pair =
                pair_both_inner_visible &&
                begin_atom->GetAtomicNum() == 6 &&
                end_atom->GetAtomicNum() == 6 &&
                begin_charge_sign == end_charge_sign;
            const bool is_resonance_mobile_outer_ring_carbanion_pair =
                !pair_both_inner_visible &&
                visible_inner_atom_indices.find(begin_idx) == visible_inner_atom_indices.end() &&
                visible_inner_atom_indices.find(end_idx) == visible_inner_atom_indices.end() &&
                begin_atom->GetAtomicNum() == 6 && end_atom->GetAtomicNum() == 6 &&
                begin_charge < 0 && end_charge < 0 && begin_atom->IsInRing() &&
                end_atom->IsInRing();
            if (is_inner_visible_adjacent_carbanion_pair)
            {
                ++inner_visible_adjacent_carbanion_pair_count;
            }
            else if (!pair_both_inner_visible &&
                     !is_resonance_mobile_outer_ring_carbanion_pair)
            {
                ++outer_or_invisible_adjacent_double_charge_count;
                const int metal_charge_sign = NearestNonzeroMetalChargeSignToBond(
                    *begin_atom,
                    *end_atom,
                    candidate->metal_states);
                if (metal_charge_sign == 0)
                {
                    ++outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count;
                }
                else if (begin_charge_sign == metal_charge_sign)
                {
                    ++outer_or_invisible_adjacent_same_sign_double_charge_count;
                }
                else
                {
                    ++outer_or_invisible_adjacent_opposite_sign_double_charge_count;
                }
            }
        }

        const int repeated_component_charge_asymmetry_count =
            RepeatedComponentChargeAsymmetryCount(mol);
        const int haptic_arene_reduction_count =
            HapticAreneReductionCount(mol, visible_atoms_by_metal);
        const int coordination_geometry_discordance_count =
            CoordinationGeometryDiscordanceCount(
                visible_atoms_by_metal,
                candidate->metal_states,
                config);
        const auto [short_carbonyl, long_dioxygen] = OxygenBondGeometryDiscordance(
            mol, visible_inner_atom_indices);
        candidate->metadata["metal_discordance_short_carbonyl_reduction_count"] = short_carbonyl;
        candidate->metadata["metal_discordance_long_dioxygen_multiple_bond_count"] = long_dioxygen;
        const int long_carbonyl = LongCarbonylMultipleBondCount(mol, visible_atoms_by_metal);
        candidate->metadata["metal_discordance_long_carbonyl_multiple_bond_count"] = long_carbonyl;
        const int sulfur_geometry = SulfurBondGeometryDiscordance(mol, visible_atoms_by_metal);
        candidate->metadata["metal_discordance_sulfur_bond_geometry_count"] = sulfur_geometry;
        const int short_quinodiimine = molgr::bond_geometry::ShortQuinodiimineGeometryConflictCount(mol);
        candidate->metadata["metal_discordance_short_quinodiimine_geometry_count"] = short_quinodiimine;
        const double negative_metal_penalty =
            kNegativeMetalDiscordancePenalty * static_cast<double>(negative_metal_discordance.count);
        const double same_sign_charge_penalty =
            inner_visible_same_sign_charge_count -
            (1.0 - kCationicDonorDiscordancePenalty) * inner_visible_cationic_donor_count;
        const double nitrosonium_formal_charge_exemption = NitrosoniumFormalChargeExemption(
            mol, candidate->metal_states, nitrosyl.nitrosonium_indices,
            penalized_negative_donor_indices, visible_atoms_by_metal, negative_metal_penalty);
        candidate->metadata["metal_nitrosonium_formal_charge_exemption"] =
            nitrosonium_formal_charge_exemption;
        // Exceptional Zn formal states remain available, but carry a preference cost.
        int high_zinc_valence_count = 0;
        for (const auto &state : candidate->metal_states)
        {
            if (state.symbol == "Zn" && state.valence > 2) ++high_zinc_valence_count;
        }
        candidate->metadata["metal_discordance_high_zinc_valence_count"] = high_zinc_valence_count;
        const double structural_discordance_count =
            static_cast<double>(
                inner_visible_diradical_count + high_zinc_valence_count +
                outer_or_invisible_adjacent_double_charge_count +
                inner_visible_adjacent_carbanion_pair_count +
                inner_visible_conjugated_carbanion_pair_count +
                excess_visible_singlet_two_electron_center_count +
                bent_cumulated_ring_allene_count +
                zero_valent_metals_with_organic_cation_count +
                unsaturated_organic_cation_discordance_count +
                repeated_component_charge_asymmetry_count +
                haptic_arene_reduction_count +
                coordination_geometry_discordance_count + short_carbonyl + long_dioxygen + long_carbonyl + sulfur_geometry + short_quinodiimine) +
            negative_metal_penalty + same_sign_charge_penalty - nitrosonium_formal_charge_exemption;
        const double discordance_count = structural_discordance_count;
        candidate->metadata["metal_discordance_structural_count"] =
            structural_discordance_count;
        candidate->metadata["metal_discordance_conjugated_atom_deficit_count"] = 0;
        candidate->metadata["metal_discordance_conjugated_bond_deficit_count"] = 0;
        candidate->metadata["metal_discordance_aromatic_atom_deficit_count"] = 0;
        candidate->metadata["metal_discordance_aromatic_ring_deficit_count"] = 0;
        candidate->metadata["metal_discordance_count"] = discordance_count;
        candidate->metadata["metal_discordance_inner_visible_diradical_count"] =
            inner_visible_diradical_count;
        candidate->metadata[
            "metal_discordance_excess_visible_singlet_two_electron_center_count"] =
            excess_visible_singlet_two_electron_center_count;
        candidate->metadata["organic_visible_singlet_carbene_count"] = visible_singlet_carbene_count;
        candidate->metadata["metal_discordance_bent_cumulated_ring_allene_count"] =
            bent_cumulated_ring_allene_count;
        candidate->metadata["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] =
            outer_or_invisible_adjacent_double_charge_count;
        candidate->metadata["metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count"] =
            outer_or_invisible_adjacent_same_sign_double_charge_count;
        candidate->metadata["metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count"] =
            outer_or_invisible_adjacent_opposite_sign_double_charge_count;
        candidate->metadata["metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count"] =
            outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count;
        candidate->metadata["metal_discordance_inner_visible_adjacent_carbanion_pair_count"] =
            inner_visible_adjacent_carbanion_pair_count;
        candidate->metadata["metal_discordance_inner_visible_conjugated_carbanion_pair_count"] =
            inner_visible_conjugated_carbanion_pair_count;
        candidate->metadata["metal_discordance_inner_visible_cationic_donor_count"] =
            inner_visible_cationic_donor_count;
        candidate->metadata["metal_discordance_inner_visible_same_sign_charge_penalty"] =
            same_sign_charge_penalty;
        candidate->metadata["metal_discordance_inner_visible_same_sign_charge_count"] =
            inner_visible_same_sign_charge_count;
        candidate->metadata["metal_discordance_negative_metal_count"] =
            negative_metal_discordance.count;
        candidate->metadata["metal_discordance_negative_metal_penalty"] =
            negative_metal_penalty;
        candidate->metadata["metal_discordance_negative_metal_outer_sphere_cation_exception"] =
            negative_metal_discordance.has_outer_sphere_cation_exception;
        candidate->metadata["metal_discordance_negative_metal_positive_metal_counterion_exception"] =
            negative_metal_discordance.has_positive_metal_counterion_exception;
        candidate->metadata["metal_discordance_zero_valent_metals_with_organic_cation_count"] =
            zero_valent_metals_with_organic_cation_count;
        candidate->metadata["metal_discordance_unsaturated_organic_cation_count"] =
            unsaturated_organic_cation_discordance_count;
        candidate->metadata["metal_discordance_repeated_component_charge_asymmetry_count"] =
            repeated_component_charge_asymmetry_count;
        candidate->metadata["metal_discordance_haptic_arene_reduction_count"] =
            haptic_arene_reduction_count;
        candidate->metadata["metal_discordance_coordination_geometry_count"] =
            coordination_geometry_discordance_count;
        return discordance_count;
    }

    /// Record ligand charge, radical and delocalization metrics for later candidate comparison.
    /// @param candidate Candidate electronic assignment being evaluated.
    /// @param config Settings for the current reconstruction or scoring operation.
    void AnnotateOrganicElectronicStateConsistency(
        molgr::state::MetalCandidateState *candidate,
        const molgr::config::MolGRConfig &config)
    {
        if (candidate == nullptr || !candidate->no_metal_state)
        {
            throw std::runtime_error(
                "MetalCandidateState requires no_metal_state before organic-state scoring");
        }

        const auto metrics = ComputeOrganicElectronicStateMetrics(
            candidate->no_metal_state->Mol(),
            config);
        candidate->metadata["organic_aromatic_atom_count"] = metrics.aromatic_atom_count;
        candidate->metadata["organic_singlet_two_electron_center_count"] =
            metrics.singlet_two_electron_center_count;
        candidate->metadata["organic_aromatic_ring_count"] = metrics.aromatic_ring_count;
        candidate->metadata["organic_aromatic_stability_score"] =
            metrics.aromatic_stability_score;
        candidate->metadata["organic_conjugated_atom_count"] = metrics.conjugated_atom_count;
        candidate->metadata["organic_conjugated_bond_count"] = metrics.conjugated_bond_count;
        candidate->metadata["organic_max_conjugated_component_size"] =
            metrics.max_conjugated_component_size;
        candidate->metadata["organic_hyperconjugative_donor_count"] =
            metrics.hyperconjugative_donor_count;
        candidate->metadata["organic_hyperconjugation_score"] =
            metrics.hyperconjugation_score;
        candidate->metadata["organic_radical_localization_penalty"] =
            metrics.radical_localization_penalty;
        double credit = 0.0;
        const auto visible_it = candidate->metadata.find("metal_visible_atom_indices");
        const auto *visible_text = visible_it == candidate->metadata.end() ? nullptr :
            std::get_if<std::string>(&visible_it->second);
        std::istringstream visible_stream(
            visible_text && molgr::metal::scoring::MetadataInt(*candidate, "nitrosyl_geometry_site_count") > 0
                ? *visible_text : "");
        std::string visible_part;
        while (std::getline(visible_stream, visible_part, ','))
        {
            if (visible_part.empty()) continue;
            auto *atom = candidate->no_metal_state->omol->GetAtom(std::stoi(visible_part));
            const int z = atom->GetAtomicNum();
            if (atom->GetFormalCharge() == -1 && (z == 7 || z == 8 || z == 16) &&
                atom->GetTotalValence() == (z == 7 ? 2 : 1) &&
                molgr::utils::GetUnpairedElectronCount(*atom) == 0)
                credit += ChargeLocalizationPenaltyForAtom(*atom, true);
        }
        candidate->metadata["organic_coordinated_anion_charge_credit"] = credit;
        candidate->metadata["organic_charge_localization_penalty"] =
            std::max(0.0, metrics.charge_localization_penalty - credit);
        candidate->metadata["organic_charge_localization_component_cancellation"] =
            metrics.charge_localization_component_cancellation;
        candidate->metadata["organic_charge_localization_polarity_inversion_penalty"] =
            metrics.charge_localization_polarity_inversion_penalty;
    }

    /// Record the final ranking terms on the selected candidate for diagnostics.
    /// @param candidate Candidate electronic assignment being evaluated.
    /// @param config Settings for the current reconstruction or scoring operation.
    void AnnotateSelectedCandidateMetrics(
        molgr::state::MetalCandidateState *candidate,
        const molgr::config::MolGRConfig &config)
    {
        if (candidate == nullptr)
        {
            throw std::runtime_error("candidate is null");
        }
        if (!candidate->score.has_value())
        {
            const double score = candidate->CombinedScore(config);
            candidate->score = score;
            candidate->metadata["score"] = score;
        }
        if (candidate->metadata.find("organic_aromatic_atom_count") == candidate->metadata.end())
        {
            AnnotateOrganicElectronicStateConsistency(candidate, config);
        }
    }

}

namespace molgr
{
    namespace metal
    {
        namespace scoring
        {
            bool HasAromaticRingAlleneResonance(const OpenBabel::OBMol &mol, int atom_index)
            {
                if (atom_index <= 0 || atom_index > static_cast<int>(mol.NumAtoms())) return false;
                auto copy = molgr::utils::CloneMolTopologyForScoring(mol);
                molgr::vendor::openbabel_threading::ResetAndAssignAromaticFlags(copy);
                return AromaticRingAlleneResonanceWitness(copy, atom_index);
            }

            int CandidateCombinationIndex(const molgr::state::MetalCandidateState &candidate)
            {
                const auto metadata_it = candidate.metadata.find("combination_index");
                if (metadata_it == candidate.metadata.end())
                {
                    return 0;
                }
                if (const auto *value = std::get_if<int>(&metadata_it->second))
                {
                    return *value;
                }
                if (const auto *value = std::get_if<double>(&metadata_it->second))
                {
                    return static_cast<int>(*value);
                }
                return 0;
            }

            int MetadataInt(
                const molgr::state::MetalCandidateState &candidate,
                const std::string &key,
                int fallback)
            {
                const auto metadata_it = candidate.metadata.find(key);
                if (metadata_it == candidate.metadata.end())
                {
                    return fallback;
                }
                if (const auto *value = std::get_if<int>(&metadata_it->second))
                {
                    return *value;
                }
                if (const auto *value = std::get_if<double>(&metadata_it->second))
                {
                    return static_cast<int>(*value);
                }
                return fallback;
            }

            double MetadataDouble(
                const molgr::state::MetalCandidateState &candidate,
                const std::string &key,
                double fallback)
            {
                const auto metadata_it = candidate.metadata.find(key);
                if (metadata_it == candidate.metadata.end())
                {
                    return fallback;
                }
                if (const auto *value = std::get_if<double>(&metadata_it->second))
                {
                    return *value;
                }
                if (const auto *value = std::get_if<int>(&metadata_it->second))
                {
                    return static_cast<double>(*value);
                }
                return fallback;
            }

            std::vector<molgr::state::MetalCandidateState> SelectLigandAlternatives(
                std::vector<molgr::state::MetalCandidateState> candidates,
                const molgr::config::MolGRConfig &config)
            {
                // Preserve NO geometry, local geometry and minimum charge separation.
                // Then avoid an unsupported localized cation before aromaticity can
                // hide its closed-shell resonance contributor within this assignment.
                // Complete organic-score ties use metal structural conflicts before
                // the arbitrary atom-index graph key.
                using namespace molgr::no_metals::selection;
                std::map<int,std::vector<std::size_t>> groups;
                std::vector<int> order;
                for (std::size_t i = 0; i < candidates.size(); ++i)
                {
                    const int combination = CandidateCombinationIndex(candidates[i]);
                    auto &indices = groups[combination];
                    if (indices.empty()) order.push_back(combination);
                    indices.push_back(i);
                }
                std::vector<molgr::state::MetalCandidateState> retained;
                for (int combination : order)
                {
                    auto &pool = groups.at(combination);
                    /// Retain all ligand assignments tied for the minimum of this ranking term before
                    /// applying the next one.
                    const auto minimize = [&](auto key) {
                        if (pool.size() < 2) return;
                        std::vector<std::size_t> tied;
                        auto best = key(candidates[pool.front()]);
                        tied.push_back(pool.front());
                        for (std::size_t j = 1; j < pool.size(); ++j) {
                            const auto next = key(candidates[pool[j]]);
                            if (next < best) { best=next; tied.clear(); }
                            if (next == best) tied.push_back(pool[j]);
                        }
                        pool = std::move(tied);
                    };
                    minimize([&](auto &c) { return std::make_tuple(
                        LigandNitrosylRankingPenalty(c,config), NoMetalCandidateGeometryKey(*c.no_metal_state)); });
                    for (auto i : pool)
                        AnnotateCandidateDiscordanceFeatures(&candidates[i], config);
                    minimize([&](auto &c) { return std::make_tuple(
                        MetadataInt(c, "metal_discordance_sulfur_bond_geometry_count"),
                        MetadataInt(c, "metal_discordance_unsaturated_organic_cation_count")); });
                    const bool measured_structure = pool.size() > 1;
                    minimize([&](auto &c) { return NoMetalCandidateStructuralKey(*c.no_metal_state,config); });
                    minimize([&](auto &c) { return ScoreReconstructionCandidate(*c.no_metal_state,config); });
                    minimize([&](auto &c) { return MetadataDouble(c,"metal_discordance_structural_count"); });
                    minimize([&](auto &c) { return NoMetalCandidateGraphTieBreakKey(*c.no_metal_state); });
                    auto &winner = candidates[pool.front()];
                    if (!measured_structure) NoMetalCandidateStructuralKey(*winner.no_metal_state,config);
                    ScoreReconstructionCandidate(*winner.no_metal_state,config);
                    winner.CombinedScore(config);
                    retained.push_back(std::move(winner));
                }
                return retained;
            }

            std::optional<molgr::state::MetalCandidateState> SelectBestCandidate(
                const std::vector<molgr::state::MetalCandidateState> &candidates,
                const molgr::config::MolGRConfig &config)
            {
                if (candidates.empty())
                {
                    return std::nullopt;
                }

                std::vector<molgr::state::MetalCandidateState> scored_candidates = candidates;
                return SelectBestCandidateInPlace(&scored_candidates, config);
            }

            std::optional<molgr::state::MetalCandidateState> SelectBestCandidateInPlace(
                std::vector<molgr::state::MetalCandidateState> *candidates,
                const molgr::config::MolGRConfig &config)
            {
                if (candidates == nullptr || candidates->empty())
                {
                    return std::nullopt;
                }

                auto &scored_candidates = *candidates;
                molgr::metal::geometry::ClearPyramidalPSConflicts(scored_candidates);
                molgr::metal::geometry::ClearPyramidalPNConflicts(scored_candidates);
                molgr::metal::geometry::ClearOrthoDioxygenConflicts(scored_candidates);
                molgr::metal::geometry::ClearArylAmidoNHCConflicts(scored_candidates);
                molgr::metal::geometry::ClearArylPhenoxideNHCConflicts(scored_candidates);
                molgr::metal::geometry::ClearArylThiolateNHCConflicts(scored_candidates);
                molgr::metal::geometry::ClearArylBenzylNHCConflicts(scored_candidates);
                molgr::metal::geometry::ClearAminoC5Conflicts(scored_candidates);
                molgr::metal::geometry::ClearLocalCCRedoxConflicts(scored_candidates);
                molgr::metal::geometry::ClearArylCarbenoidConflicts(scored_candidates);
                molgr::metal::geometry::ClearPairedDiimineConflicts(scored_candidates);
                molgr::metal::geometry::ClearPyramidalPyridylConflicts(scored_candidates);
                molgr::metal::geometry::ClearConjugatedGeometryConflicts(scored_candidates);
                ClearFormazanGeometryConflicts(scored_candidates);
                ClearFoldedRingGeometryConflicts(scored_candidates);
                molgr::metal::hydrazido_geometry::Clear(scored_candidates);
                ClearMacrocycleElectronConflicts(scored_candidates);
                ClearAlkoxyAllylicGeometryConflicts(scored_candidates);
                AnnotateEqualizedCarbonAlternatives(scored_candidates);
                AnnotateNitrosoniumRedistributionExceptions(scored_candidates);
                AnnotateBentAlleneAromaticAlternatives(scored_candidates);
                int max_aromatic_atom_count = 0;
                int max_aromatic_ring_count = 0;
                int max_conjugated_atom_count = 0;
                int max_conjugated_bond_count = 0;
                int max_hyperconjugation_score = 0;
                for (auto &candidate : scored_candidates)
                {
                    if (candidate.metadata.find("organic_aromatic_ring_count") ==
                            candidate.metadata.end() ||
                        candidate.metadata.find("organic_aromatic_stability_score") ==
                            candidate.metadata.end())
                    {
                        AnnotateOrganicElectronicStateConsistency(&candidate, config);
                    }
                    if (candidate.metadata.find("organic_aromatic_stability_score") ==
                        candidate.metadata.end())
                    {
                        candidate.metadata["organic_aromatic_stability_score"] =
                            MetadataDouble(
                                candidate,
                                "organic_aromatic_ring_count",
                                0.0);
                    }
                    max_aromatic_ring_count = std::max(
                        max_aromatic_ring_count,
                        MetadataInt(candidate, "organic_aromatic_ring_count", 0));
                    max_aromatic_atom_count = std::max(
                        max_aromatic_atom_count,
                        MetadataInt(candidate, "organic_aromatic_atom_count", 0));
                    max_conjugated_atom_count = std::max(
                        max_conjugated_atom_count,
                        MetadataInt(candidate, "organic_conjugated_atom_count", 0));
                    max_conjugated_bond_count = std::max(
                        max_conjugated_bond_count,
                        MetadataInt(candidate, "organic_conjugated_bond_count", 0));
                    max_hyperconjugation_score = std::max(
                        max_hyperconjugation_score,
                        MetadataInt(candidate, "organic_hyperconjugation_score", 0));
                }
                double max_aromatic_stability_score = 0.0;
                for (const auto &candidate : scored_candidates)
                {
                    max_aromatic_stability_score = std::max(
                        max_aromatic_stability_score,
                        MetadataDouble(candidate, "organic_aromatic_stability_score", 0.0));
                }

                for (auto &candidate : scored_candidates)
                {
                    const double structural_discordance_count = MetadataDouble(
                        candidate,
                        "metal_discordance_structural_count",
                        MetadataDouble(candidate, "metal_discordance_count", 0.0)) -
                        MetadataInt(candidate, "nitrosonium_redistribution_exception");
                    const int conjugated_atom_deficit_count = std::max(
                        0,
                        max_conjugated_atom_count -
                            MetadataInt(candidate, "organic_conjugated_atom_count", 0));
                    const int conjugated_bond_deficit_count = std::max(
                        0,
                        max_conjugated_bond_count -
                            MetadataInt(candidate, "organic_conjugated_bond_count", 0));
                    const int aromatic_atom_deficit_count = std::max(
                        0,
                        max_aromatic_atom_count -
                            MetadataInt(candidate, "organic_aromatic_atom_count", 0));
                    const int aromatic_ring_deficit_count = std::max(
                        0,
                        max_aromatic_ring_count -
                            MetadataInt(candidate, "organic_aromatic_ring_count", 0));
                    const double aromatic_stability_deficit = std::max(
                        0.0,
                        max_aromatic_stability_score -
                            MetadataDouble(candidate, "organic_aromatic_stability_score", 0.0));
                    const int hyperconjugation_deficit = std::max(
                        0,
                        max_hyperconjugation_score -
                            MetadataInt(candidate, "organic_hyperconjugation_score", 0));
                    candidate.metadata["metal_discordance_structural_count"] =
                        structural_discordance_count;
                    candidate.metadata["metal_discordance_max_conjugated_atom_count"] =
                        max_conjugated_atom_count;
                    candidate.metadata["metal_discordance_conjugated_atom_deficit_count"] =
                        conjugated_atom_deficit_count;
                    candidate.metadata["metal_discordance_max_conjugated_bond_count"] =
                        max_conjugated_bond_count;
                    candidate.metadata["metal_discordance_conjugated_bond_deficit_count"] =
                        conjugated_bond_deficit_count;
                    candidate.metadata["metal_discordance_max_aromatic_atom_count"] =
                        max_aromatic_atom_count;
                    candidate.metadata["metal_discordance_aromatic_atom_deficit_count"] =
                        aromatic_atom_deficit_count;
                    candidate.metadata["metal_discordance_max_aromatic_ring_count"] =
                        max_aromatic_ring_count;
                    candidate.metadata["metal_discordance_max_aromatic_stability_score"] =
                        max_aromatic_stability_score;
                    candidate.metadata["metal_discordance_aromatic_ring_deficit_count"] =
                        aromatic_ring_deficit_count;
                    candidate.metadata["metal_discordance_aromatic_stability_deficit"] =
                        aromatic_stability_deficit;
                    candidate.metadata["organic_hyperconjugation_max_score"] =
                        max_hyperconjugation_score;
                    candidate.metadata["organic_hyperconjugation_deficit"] =
                        hyperconjugation_deficit;
                    candidate.metadata["metal_discordance_count"] = structural_discordance_count;
                }

                AnnotateAlkoxyAllylicAlternatives(scored_candidates);
                AnnotateMacrocycleAlternatives(scored_candidates);
                AnnotateFoldedRingAlternatives(scored_candidates);
                molgr::metal::hydrazido_geometry::Annotate(scored_candidates);
                AnnotateFormazanAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateQuinodiimineAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotatePolyenamineAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotatePairedDiimineAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotatePyramidalPyridylAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateLocalCCRedoxAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateArylCarbenoidAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotatePyramidalPSAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotatePyramidalPNAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateOrthoDioxygenAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateArylAmidoNHCAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateArylPhenoxideNHCAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateArylThiolateNHCAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateArylBenzylNHCAlternatives(scored_candidates);
                molgr::metal::geometry::AnnotateAminoC5Alternatives(scored_candidates);
                double min_discordance_count = std::numeric_limits<double>::infinity();
                for (const auto &candidate : scored_candidates)
                {
                    min_discordance_count = std::min(
                        min_discordance_count,
                        MetadataDouble(candidate, "metal_discordance_count", 0.0));
                }

                std::vector<std::size_t> discordance_filtered_candidate_indices;
                discordance_filtered_candidate_indices.reserve(scored_candidates.size());
                for (std::size_t candidate_index = 0; candidate_index < scored_candidates.size();
                     ++candidate_index)
                {
                    auto &candidate = scored_candidates[candidate_index];
                    const bool passes_discordance_filter =
                        MetadataDouble(candidate, "metal_discordance_count", 0.0) ==
                        min_discordance_count;
                    candidate.metadata["passes_metal_discordance_filter"] = passes_discordance_filter;
                    if (passes_discordance_filter)
                    {
                        discordance_filtered_candidate_indices.push_back(candidate_index);
                    }
                }
                if (discordance_filtered_candidate_indices.empty())
                {
                    return std::nullopt;
                }

                for (const std::size_t candidate_index : discordance_filtered_candidate_indices)
                {
                    AnnotateSelectedCandidateMetrics(&scored_candidates[candidate_index], config);
                }

                /// Share charge references across metal-visible carbon singlet occupations when
                /// aromatic atom/ring counts agree, preserving neutral carbenes
                /// over unnecessary ligand charge separation. Keep occupation
                /// references too, so aromatic anions remain competitive.
                const auto charge_reference_key = [&](std::size_t index) {
                    return std::make_pair(
                        MetadataDouble(scored_candidates[index],
                            "organic_charge_localization_penalty", 0.0),
                        CandidateCombinationIndex(scored_candidates[index]));
                };
                /// Group references by ligand spin, paired-center count and aromatic atom and ring counts.
                const auto aromatic_reference_key = [&](const auto &candidate) {
                    return std::make_tuple(
                        candidate.no_metal_radical_target,
                        MetadataInt(candidate, "organic_singlet_two_electron_center_count", 0) -
                            MetadataInt(candidate, "organic_visible_singlet_carbene_count", 0),
                        MetadataInt(candidate, "organic_aromatic_atom_count", 0),
                        MetadataInt(candidate, "organic_aromatic_ring_count", 0));
                };
                std::map<std::pair<int, int>, std::size_t> charge_reference_by_occupation;
                std::map<std::tuple<int, int, int, int>, std::size_t> charge_reference_by_aromaticity;
                for (const std::size_t candidate_index : discordance_filtered_candidate_indices)
                {
                    const auto &candidate = scored_candidates[candidate_index];
                    const auto target = std::make_pair(
                        candidate.no_metal_radical_target,
                        MetadataInt(candidate, "organic_singlet_two_electron_center_count", 0));
                    const auto reference = charge_reference_by_occupation.find(target);
                    const auto key = std::make_pair(
                        MetadataDouble(candidate, "organic_charge_localization_penalty", 0.0),
                        CandidateCombinationIndex(candidate));
                    if (reference == charge_reference_by_occupation.end() ||
                        key < std::make_pair(
                            MetadataDouble(scored_candidates[reference->second],
                                "organic_charge_localization_penalty", 0.0),
                            CandidateCombinationIndex(scored_candidates[reference->second])))
                    {
                        charge_reference_by_occupation[target] = candidate_index;
                    }
                    const auto aromatic_target = aromatic_reference_key(candidate);
                    const auto aromatic_reference = charge_reference_by_aromaticity.find(aromatic_target);
                    if (aromatic_reference == charge_reference_by_aromaticity.end() ||
                        key < charge_reference_key(aromatic_reference->second))
                    {
                        charge_reference_by_aromaticity[aromatic_target] = candidate_index;
                    }
                }
                const double charge_localization_margin = std::max(
                    0.0,
                    config.metal_scoring.charge_localization_selection_margin);
                std::optional<
                    std::tuple<double, int, int, int, int, int, double, double, int, double, int, double, int>>
                    best_selection_key;
                double minimum_nitrosyl_penalty = std::numeric_limits<double>::infinity();
                for (const auto index : discordance_filtered_candidate_indices)
                    minimum_nitrosyl_penalty = std::min(minimum_nitrosyl_penalty,
                        MetadataDouble(scored_candidates[index], "nitrosyl_geometry_penalty", 0.0));
                std::optional<molgr::state::MetalCandidateState> best_candidate;
                for (const std::size_t candidate_index : discordance_filtered_candidate_indices)
                {
                    auto &candidate = scored_candidates[candidate_index];
                    if (MetadataDouble(candidate, "nitrosyl_geometry_penalty", 0.0) > minimum_nitrosyl_penalty)
                        continue;
                    auto reference_index = charge_reference_by_occupation.at(std::make_pair(
                            candidate.no_metal_radical_target,
                            MetadataInt(candidate, "organic_singlet_two_electron_center_count", 0)));
                    const auto aromatic_reference_index =
                        charge_reference_by_aromaticity.at(aromatic_reference_key(candidate));
                    if (charge_reference_key(aromatic_reference_index) < charge_reference_key(reference_index))
                    {
                        reference_index = aromatic_reference_index;
                    }
                    const auto &charge_localization_reference_candidate = scored_candidates[reference_index];
                    const double minimum_charge_localization_penalty = MetadataDouble(
                        charge_localization_reference_candidate,
                        "organic_charge_localization_penalty", 0.0);
                    std::map<int, int> reference_valences;
                    for (const auto &metal_state : charge_localization_reference_candidate.metal_states)
                    {
                        reference_valences[metal_state.idx] = metal_state.valence;
                    }
                    const double score_value =
                        candidate.score.has_value() ? *candidate.score : candidate.CombinedScore(config);
                    const int aromatic_atom_deficit_count = static_cast<int>(MetadataInt(
                        candidate,
                        "metal_discordance_aromatic_atom_deficit_count",
                        0));
                    const int conjugated_atom_deficit_count = static_cast<int>(MetadataInt(
                        candidate,
                        "metal_discordance_conjugated_atom_deficit_count",
                        0));
                    const int conjugated_bond_deficit_count = static_cast<int>(MetadataInt(
                        candidate,
                        "metal_discordance_conjugated_bond_deficit_count",
                        0));
                    const double aromatic_stability_deficit = MetadataDouble(
                        candidate,
                        "metal_discordance_aromatic_stability_deficit",
                        0.0);
                    const int aromatic_ring_deficit_count = static_cast<int>(MetadataInt(
                        candidate,
                        "metal_discordance_aromatic_ring_deficit_count",
                        0));
                    const double radical_localization_penalty = MetadataDouble(
                        candidate,
                        "organic_radical_localization_penalty",
                        0.0);
                    const double charge_localization_penalty = MetadataDouble(
                        candidate,
                        "organic_charge_localization_penalty",
                        0.0);
                    const double charge_localization_difference = std::max(
                        0.0,
                        charge_localization_penalty - minimum_charge_localization_penalty);
                    int reference_valence_max_delta = 0;
                    for (const auto &metal_state : candidate.metal_states)
                    {
                        const auto reference_valence = reference_valences.find(metal_state.idx);
                        if (reference_valence == reference_valences.end())
                        {
                            continue;
                        }
                        reference_valence_max_delta = std::max(
                            reference_valence_max_delta,
                            std::abs(metal_state.valence - reference_valence->second));
                    }
                    const bool oxidation_state_jump_exceeded =
                        charge_localization_difference > 0.0 &&
                        reference_valence_max_delta >
                        kMaxChargeLocalizationReferenceOxidationStateDelta;
                    const bool charge_localization_margin_exceeded =
                        oxidation_state_jump_exceeded ||
                        (
                            charge_localization_difference > 0.0 &&
                            (charge_localization_difference > charge_localization_margin ||
                             std::abs(
                                 charge_localization_difference - charge_localization_margin) <=
                                 1e-12));
                    candidate.metadata["organic_charge_localization_reference_penalty"] =
                        minimum_charge_localization_penalty;
                    candidate.metadata["organic_charge_localization_selection_margin"] =
                        charge_localization_margin;
                    candidate.metadata["organic_charge_localization_margin_difference"] =
                        charge_localization_difference;
                    candidate.metadata["organic_charge_localization_margin_exceeded"] =
                        static_cast<int>(charge_localization_margin_exceeded);
                    candidate.metadata[
                        "organic_charge_localization_reference_metal_valence_max_delta"] =
                        reference_valence_max_delta;
                    candidate.metadata["organic_charge_localization_metal_valence_jump_exceeded"] =
                        static_cast<int>(oxidation_state_jump_exceeded);
                    const int hyperconjugation_deficit = MetadataInt(
                        candidate,
                        "organic_hyperconjugation_deficit",
                        0);
                    const double nitrosyl_geometry_penalty = MetadataDouble(
                        candidate, "nitrosyl_geometry_penalty", 0.0);
                    const auto selection_key =
                        std::make_tuple(
                            min_discordance_count,
                            static_cast<int>(charge_localization_margin_exceeded),
                            conjugated_atom_deficit_count,
                            conjugated_bond_deficit_count,
                            aromatic_atom_deficit_count,
                            aromatic_ring_deficit_count,
                            aromatic_stability_deficit,
                            radical_localization_penalty,
                            hyperconjugation_deficit,
                            nitrosyl_geometry_penalty,
                            MetadataInt(candidate, "metal_discordance_high_zinc_valence_count"),
                            score_value,
                            CandidateCombinationIndex(candidate));
                    candidate.metadata["selection_key"] =
                        SelectionKeyString(
                            min_discordance_count,
                            static_cast<int>(charge_localization_margin_exceeded),
                            conjugated_atom_deficit_count,
                            conjugated_bond_deficit_count,
                            aromatic_atom_deficit_count,
                            aromatic_ring_deficit_count,
                            aromatic_stability_deficit,
                            radical_localization_penalty,
                            hyperconjugation_deficit,
                            nitrosyl_geometry_penalty,
                            MetadataInt(candidate, "metal_discordance_high_zinc_valence_count"),
                            score_value,
                            CandidateCombinationIndex(candidate));
                    if (best_selection_key.has_value() && selection_key >= *best_selection_key)
                    {
                        continue;
                    }
                    best_selection_key = selection_key;
                    best_candidate = candidate;
                }
                return best_candidate;
            }

            molgr::state::MetalCandidateState PrepareCandidateWithNoMetalState(
                const molgr::state::MetalCandidateState &candidate,
                const std::shared_ptr<molgr::state::ReconstructionState> &no_metal_state,
                const molgr::config::MolGRConfig &config)
            {
                auto machine = molgr::state::MetalCandidateStateMachine::FromCandidateState(candidate);
                machine.SetNoMetalState(
                    "reconstruct_no_metal",
                    CloneNoMetalStateForCandidate(no_metal_state));
                machine.Annotate("score_candidate");
                auto prepared_candidate = machine.Freeze();


                return prepared_candidate;
            }

        }
    }
}
