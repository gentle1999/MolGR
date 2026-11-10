#include "molgr/utils/metals/scoring.h"

#include "molgr/utils/coordination.h"

#include "molgr/utils/consts.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/organic_topology.h"
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
    struct Point3D
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct CoordinationBlocker
    {
        int atom_idx = 0;
        Point3D coordinates;
        double radius = 0.0;
    };

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
        double radical_localization_penalty = 0.0;
        double charge_localization_penalty = 0.0;
        double charge_localization_component_cancellation = 0.0;
        double charge_localization_polarity_inversion_penalty = 0.0;
    };

    struct NegativeMetalDiscordance
    {
        int count = 0;
        bool has_outer_sphere_cation_exception = false;
        bool has_positive_metal_counterion_exception = false;
    };

    constexpr double kNegativeMetalDiscordancePenalty = 0.5;
    constexpr double kMinimumRingAlleneAngleDegrees = 150.0;
    constexpr double kMinimumChargePolarityInversionElectronegativityGap = 0.3;
    constexpr int kMaxChargeLocalizationReferenceOxidationStateDelta = 2;
    constexpr std::uint64_t kConnectivityHashOffset = 1469598103934665603ULL;
    constexpr std::uint64_t kConnectivityHashPrime = 1099511628211ULL;

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

    bool IsInnerVisibleDiradicalDiscordanceAtom(const OpenBabel::OBAtom &atom)
    {
        return static_cast<int>(molgr::utils::GetUnpairedElectronCount(atom)) >= 2;
    }

    bool IsExplicitSingletTwoElectronCenter(OpenBabel::OBAtom &atom)
    {
        const int atomic_num = atom.GetAtomicNum();
        return (atomic_num == 6 || atomic_num == 7 || atomic_num == 15) &&
               atom.GetFormalCharge() == 0 &&
               molgr::utils::GetUnpairedElectronCount(atom) == 0 &&
               molgr::utils::GetLonePairCount(atom) == 1 &&
               !molgr::utils::HasUnresolvedTwoElectronCenter(atom) &&
               molgr::reconstruct::AssignRadicalDots(atom) == 2;
    }

    int BentCumulatedRingAlleneCount(OpenBabel::OBMol &mol)
    {
        int count = 0;
        FOR_ATOMS_OF_MOL(center_iter, mol)
        {
            OpenBabel::OBAtom &center = *center_iter;
            if (!center.IsInRing())
            {
                continue;
            }
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
                ++count;
            }
        }
        return count;
    }

    int BondOrder(const OpenBabel::OBBond &bond)
    {
        return molgr::vendor::openbabel_threading::BondIsAromatic(
                   const_cast<OpenBabel::OBBond &>(bond))
                   ? 2
                   : static_cast<int>(bond.GetBondOrder());
    }

    OpenBabel::OBAtom *OtherBondAtom(OpenBabel::OBBond &bond, OpenBabel::OBAtom &atom)
    {
        return bond.GetNbrAtom(&atom);
    }

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
        FOR_RINGS_OF_MOL(ring_iter, mol)
        {
            OpenBabel::OBRing *ring = &(*ring_iter);
            if (ring == nullptr || ring->_path.size() < 5 || ring->_path.size() > 6)
            {
                continue;
            }
            bool all_carbon = true;
            int negative_count = 0;
            for (const int atom_idx : ring->_path)
            {
                OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                if (atom == nullptr || atom->GetAtomicNum() != 6)
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

    int InnerVisibleConjugatedChargedCarbonPairCount(
        OpenBabel::OBMol &mol,
        const std::set<int> &visible_inner_atom_indices)
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
                    ++count;
                }
            }
        }
        return count;
    }

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

    double RadicalLocalizationPenaltyForAtom(
        OpenBabel::OBAtom &atom,
        bool is_conjugated)
    {
        const int radical_electrons = static_cast<int>(molgr::utils::GetUnpairedElectronCount(atom));
        // A validated neutral C/N/P two-electron deficit can be represented as
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
            << score_value << "," << combination_index;
        return out.str();
    }

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
                if (opposite_charges && bond_iter->GetBondOrder() == 1 &&
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

    Point3D AtomCoordinates(const OpenBabel::OBAtom &atom)
    {
        return {atom.GetX(), atom.GetY(), atom.GetZ()};
    }

    Point3D MetalCoordinates(const molgr::MetalAtomPosition &metal_state)
    {
        return {metal_state.position_x, metal_state.position_y, metal_state.position_z};
    }

    double DistanceToMetal(
        const OpenBabel::OBAtom &atom,
        const molgr::MetalAtomPosition &metal_state)
    {
        const double dx = static_cast<double>(atom.GetX()) - metal_state.position_x;
        const double dy = static_cast<double>(atom.GetY()) - metal_state.position_y;
        const double dz = static_cast<double>(atom.GetZ()) - metal_state.position_z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    double VectorNorm(const Point3D &vector)
    {
        return std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
    }

    double DotProduct(const Point3D &lhs, const Point3D &rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    Point3D CrossProduct(const Point3D &lhs, const Point3D &rhs)
    {
        return {
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x};
    }

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

    int CoordinationGeometryDiscordanceCount(
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal,
        const std::vector<molgr::MetalAtomPosition> &metal_states,
        const molgr::config::MetalRadicalInferenceConfig &config)
    {
        int count = 0;
        const std::size_t pair_count = std::min(visible_atoms_by_metal.size(), metal_states.size());
        for (std::size_t idx = 0; idx < pair_count; ++idx)
        {
            const auto &metal_state = metal_states[idx];
            const std::string geometry =
                CoordinationGeometry(visible_atoms_by_metal[idx], metal_state, config);
            const bool high_valent_square_planar_group_ten =
                (metal_state.symbol == "Pd" || metal_state.symbol == "Pt") &&
                metal_state.valence >= 4 && geometry == "square_planar";
            const bool high_valent_linear_group_eleven =
                (metal_state.symbol == "Ag" || metal_state.symbol == "Au") &&
                metal_state.valence >= 3 && geometry == "linear";
            if (high_valent_square_planar_group_ten || high_valent_linear_group_eleven)
            {
                ++count;
            }
        }
        return count;
    }

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

    double DistancePointToSegment(
        const Point3D &point,
        const Point3D &segment_start,
        const Point3D &segment_end)
    {
        const double vx = segment_end.x - segment_start.x;
        const double vy = segment_end.y - segment_start.y;
        const double vz = segment_end.z - segment_start.z;
        const double seg_len_sq = vx * vx + vy * vy + vz * vz;
        if (seg_len_sq <= 1.0e-12)
        {
            const double dx = point.x - segment_start.x;
            const double dy = point.y - segment_start.y;
            const double dz = point.z - segment_start.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        const double wx = point.x - segment_start.x;
        const double wy = point.y - segment_start.y;
        const double wz = point.z - segment_start.z;
        const double projection = std::max(
            0.0,
            std::min(1.0, (wx * vx + wy * vy + wz * vz) / seg_len_sq));
        const Point3D closest{
            segment_start.x + projection * vx,
            segment_start.y + projection * vy,
            segment_start.z + projection * vz};
        const double dx = point.x - closest.x;
        const double dy = point.y - closest.y;
        const double dz = point.z - closest.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    std::vector<CoordinationBlocker> BuildCoordinationBlockers(
        const OpenBabel::OBMol &mol,
        const molgr::config::MetalScoringConfig &config)
    {
        std::vector<CoordinationBlocker> blockers;
        FOR_ATOMS_OF_MOL(atom_iter, const_cast<OpenBabel::OBMol &>(mol))
        {
            OpenBabel::OBAtom &blocker = *atom_iter;
            const double blocker_radius =
                config.metal_access_radius_scale *
                    OpenBabel::OBElements::GetCovalentRad(static_cast<int>(blocker.GetAtomicNum())) +
                config.metal_access_clearance_angstrom;
            if (blocker_radius <= 0.0)
            {
                continue;
            }
            blockers.push_back(
                CoordinationBlocker{
                    static_cast<int>(blocker.GetIdx()),
                    AtomCoordinates(blocker),
                    blocker_radius});
        }
        return blockers;
    }

    bool HasUnobstructedCoordinationPathFromBlockers(
        int atom_idx,
        const Point3D &atom_coordinates,
        const Point3D &segment_start,
        const std::vector<CoordinationBlocker> &blockers)
    {
        for (const auto &blocker : blockers)
        {
            if (blocker.atom_idx == atom_idx)
            {
                continue;
            }
            if (DistancePointToSegment(blocker.coordinates, segment_start, atom_coordinates) <
                blocker.radius)
            {
                return false;
            }
        }
        return true;
    }

    double CoordinationRadiusCutoff(
        const OpenBabel::OBAtom &atom,
        const molgr::MetalAtomPosition &metal_state,
        const molgr::config::MetalScoringConfig &config)
    {
        return molgr::coordination::CoordinationDistanceCutoff(
            static_cast<int>(metal_state.element_idx),
            static_cast<int>(atom.GetAtomicNum()),
            config);
    }

    bool IsInnerSphereAtom(
        const OpenBabel::OBAtom &atom,
        const molgr::MetalAtomPosition &metal_state,
        const molgr::config::MetalScoringConfig &config)
    {
        const double distance = DistanceToMetal(atom, metal_state);
        return distance > 0.0 && distance <= CoordinationRadiusCutoff(atom, metal_state, config);
    }

    bool IsVisibleToMetalAtom(
        OpenBabel::OBAtom &atom,
        const molgr::MetalAtomPosition &metal_state,
        const std::vector<CoordinationBlocker> &blockers)
    {
        const double distance = DistanceToMetal(atom, metal_state);
        if (distance <= 0.0)
        {
            return false;
        }
        return HasUnobstructedCoordinationPathFromBlockers(
            static_cast<int>(atom.GetIdx()),
            AtomCoordinates(atom),
            MetalCoordinates(metal_state),
            blockers);
    }

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
                if (IsInnerSphereAtom(atom, metal_state, config))
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
                negative_metal_count += std::abs(metal_state.valence);
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

    bool IsLocallyChargeCompensatedNonmetalCation(OpenBabel::OBAtom &atom)
    {
        if (atom.IsMetal() || static_cast<int>(atom.GetFormalCharge()) <= 0)
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
        return adjacent_negative_charge > adjacent_positive_charge;
    }

    int ZeroValentMetalsWithOrganicCationCount(
        OpenBabel::OBMol &mol,
        const std::vector<molgr::MetalAtomPosition> &metal_states,
        int total_charge)
    {
        if (total_charge > 0)
        {
            return 0;
        }
        if (metal_states.empty())
        {
            return 0;
        }
        for (const auto &metal_state : metal_states)
        {
            if (metal_state.valence != 0)
            {
                return 0;
            }
        }
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom &atom = *atom_iter;
            if (!atom.IsMetal() &&
                atom.GetFormalCharge() > 0 &&
                !IsLocallyChargeCompensatedNonmetalCation(atom))
            {
                return 1;
            }
        }
        return 0;
    }

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
                if (!IsInnerSphereAtom(atom, metal_state, config.metal_scoring))
                {
                    continue;
                }
                if (!IsVisibleToMetalAtom(atom, metal_state, blockers))
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
                    !HasAdjacentAnionicPolarizationCancellation(atom))
                {
                    ++inner_visible_same_sign_charge_count;
                }
            }
            visible_atoms_by_metal.push_back(std::move(visible_atoms));
        }

        int visible_singlet_two_electron_center_count = 0;
        for (const int atom_idx : visible_inner_atom_indices)
        {
            OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
            if (atom != nullptr && IsExplicitSingletTwoElectronCenter(*atom))
            {
                ++visible_singlet_two_electron_center_count;
            }
        }
        const int excess_visible_singlet_two_electron_center_count = std::max(
            0,
            visible_singlet_two_electron_center_count -
                static_cast<int>(candidate->metal_states.size()));
        const int bent_cumulated_ring_allene_count = BentCumulatedRingAlleneCount(mol);

        int outer_or_invisible_adjacent_double_charge_count = 0;
        const int inner_visible_conjugated_carbanion_pair_count =
            InnerVisibleConjugatedChargedCarbonPairCount(mol, visible_inner_atom_indices);
        const auto negative_metal_discordance =
            NegativeMetalDiscordanceCount(mol, candidate->metal_states, config.metal_scoring);
        int total_charge = candidate->no_metal_state->total_charge;
        for (const auto &metal_state : candidate->metal_states)
        {
            total_charge += metal_state.valence;
        }
        const int zero_valent_metals_with_organic_cation_count =
            ZeroValentMetalsWithOrganicCationCount(mol, candidate->metal_states, total_charge);
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
                config.metal_radical_inference);
        const double negative_metal_penalty =
            kNegativeMetalDiscordancePenalty * static_cast<double>(negative_metal_discordance.count);
        const double structural_discordance_count =
            static_cast<double>(
                inner_visible_diradical_count +
                outer_or_invisible_adjacent_double_charge_count +
                inner_visible_adjacent_carbanion_pair_count +
                inner_visible_conjugated_carbanion_pair_count +
                inner_visible_same_sign_charge_count +
                excess_visible_singlet_two_electron_center_count +
                bent_cumulated_ring_allene_count +
                zero_valent_metals_with_organic_cation_count +
                unsaturated_organic_cation_discordance_count +
                repeated_component_charge_asymmetry_count +
                haptic_arene_reduction_count +
                coordination_geometry_discordance_count) +
            negative_metal_penalty;
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
        candidate->metadata["organic_charge_localization_penalty"] =
            metrics.charge_localization_penalty;
        candidate->metadata["organic_charge_localization_component_cancellation"] =
            metrics.charge_localization_component_cancellation;
        candidate->metadata["organic_charge_localization_polarity_inversion_penalty"] =
            metrics.charge_localization_polarity_inversion_penalty;
    }

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
                        MetadataDouble(candidate, "metal_discordance_count", 0.0));
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

                std::size_t charge_localization_reference_candidate_index =
                    discordance_filtered_candidate_indices.front();
                double minimum_charge_localization_penalty =
                    std::numeric_limits<double>::infinity();
                for (const std::size_t candidate_index : discordance_filtered_candidate_indices)
                {
                    const auto &candidate = scored_candidates[candidate_index];
                    const double penalty = MetadataDouble(
                        candidate,
                        "organic_charge_localization_penalty",
                        0.0);
                    const auto &reference_candidate =
                        scored_candidates[charge_localization_reference_candidate_index];
                    if (
                        penalty < minimum_charge_localization_penalty ||
                        (
                            penalty == minimum_charge_localization_penalty &&
                            CandidateCombinationIndex(candidate) <
                                CandidateCombinationIndex(reference_candidate)))
                    {
                        minimum_charge_localization_penalty = penalty;
                        charge_localization_reference_candidate_index = candidate_index;
                    }
                }
                const auto &charge_localization_reference_candidate =
                    scored_candidates[charge_localization_reference_candidate_index];
                minimum_charge_localization_penalty = MetadataDouble(
                    charge_localization_reference_candidate,
                    "organic_charge_localization_penalty",
                    0.0);
                std::map<int, int> reference_valences;
                for (const auto &metal_state : charge_localization_reference_candidate.metal_states)
                {
                    reference_valences[metal_state.idx] = metal_state.valence;
                }
                const double charge_localization_margin = std::max(
                    0.0,
                    config.metal_scoring.charge_localization_selection_margin);
                std::optional<
                    std::tuple<double, int, int, int, int, int, double, double, int, double, int>>
                    best_selection_key;
                std::optional<molgr::state::MetalCandidateState> best_candidate;
                for (const std::size_t candidate_index : discordance_filtered_candidate_indices)
                {
                    auto &candidate = scored_candidates[candidate_index];
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
                const double score = prepared_candidate.CombinedScore(config);
                prepared_candidate.score = score;
                prepared_candidate.metadata["score"] = score;
                AnnotateCandidateDiscordanceFeatures(&prepared_candidate, config);
                return prepared_candidate;
            }

        }
    }
}
