#include "molgr/utils/organic_topology.h"

#include "molgr/utils/consts.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/utils.h"
#include "molgr/vendor/openbabel_threading.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include "molgr/compat/openbabel_iter.h"
#include <openbabel/ring.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <vector>

namespace
{
    constexpr int kAromaticRingFormalChargeAbsRejectionThreshold = 4;

    struct IntFlagSet
    {
        std::vector<unsigned char> present;
        std::vector<int> values;

        explicit IntFlagSet(std::size_t initial_size = 0)
            : present(initial_size, 0)
        {
        }

        bool Contains(int index) const
        {
            return index >= 0 &&
                   static_cast<std::size_t>(index) < present.size() &&
                   present[static_cast<std::size_t>(index)] != 0;
        }

        bool Add(int index)
        {
            if (index < 0)
            {
                return false;
            }
            const auto offset = static_cast<std::size_t>(index);
            if (offset >= present.size())
            {
                present.resize(offset + 1, 0);
            }
            if (present[offset] != 0)
            {
                return false;
            }
            present[offset] = 1;
            values.push_back(index);
            return true;
        }

        std::size_t Size() const
        {
            return values.size();
        }
    };

    struct HeavyBondRef
    {
        OpenBabel::OBBond *bond = nullptr;
        OpenBabel::OBAtom *begin_atom = nullptr;
        OpenBabel::OBAtom *end_atom = nullptr;
        int bond_idx = -1;
    };

    struct ValidatedConjugationTopology
    {
        IntFlagSet bond_indices;
        std::vector<IntFlagSet> cumulated_bond_indices_by_center;
    };

    OpenBabel::OBAtom *FirstMultipleBondOuterAtom(
        OpenBabel::OBMol &mol,
        OpenBabel::OBAtom &center,
        const std::vector<int> &bond_indices)
    {
        for (const int bond_idx : bond_indices)
        {
            OpenBabel::OBBond *bond = mol.GetBond(bond_idx);
            if (bond == nullptr)
            {
                continue;
            }
            OpenBabel::OBAtom *outer = bond->GetNbrAtom(&center);
            if (outer != nullptr && outer->GetAtomicNum() != 1)
            {
                return outer;
            }
        }
        return nullptr;
    }

    OpenBabel::OBAtom *FirstOuterSingleBondNeighbor(
        OpenBabel::OBAtom &atom,
        const OpenBabel::OBAtom &excluded_neighbor)
    {
        OpenBabel::OBAtom *selected = nullptr;
        FOR_BONDS_OF_ATOM(bond_iter, &atom)
        {
            OpenBabel::OBBond &bond = *bond_iter;
            if (molgr::vendor::openbabel_threading::BondIsAromatic(bond) ||
                bond.GetBondOrder() != 1)
            {
                continue;
            }
            OpenBabel::OBAtom *neighbor = bond.GetNbrAtom(&atom);
            if (neighbor == nullptr || neighbor->GetIdx() == excluded_neighbor.GetIdx())
            {
                continue;
            }
            if (selected == nullptr || neighbor->GetIdx() < selected->GetIdx())
            {
                selected = neighbor;
            }
        }
        return selected;
    }

    bool AlternatingDoubleBondsAreGeometricallyConjugated(
        OpenBabel::OBMol &mol,
        OpenBabel::OBAtom &left_inner,
        OpenBabel::OBAtom &right_inner,
        const std::vector<std::vector<int>> &multiple_bonds_by_atom,
        double volume_tolerance)
    {
        const auto left_idx = static_cast<std::size_t>(left_inner.GetIdx());
        const auto right_idx = static_cast<std::size_t>(right_inner.GetIdx());
        if (left_idx >= multiple_bonds_by_atom.size() ||
            right_idx >= multiple_bonds_by_atom.size() ||
            multiple_bonds_by_atom[left_idx].empty() ||
            multiple_bonds_by_atom[right_idx].empty())
        {
            return true;
        }

        OpenBabel::OBAtom *left_outer = FirstMultipleBondOuterAtom(
            mol,
            left_inner,
            multiple_bonds_by_atom[left_idx]);
        OpenBabel::OBAtom *right_outer = FirstMultipleBondOuterAtom(
            mol,
            right_inner,
            multiple_bonds_by_atom[right_idx]);
        if (left_outer == nullptr || right_outer == nullptr)
        {
            return true;
        }

        OpenBabel::OBAtom *left_terminal = FirstOuterSingleBondNeighbor(
            *left_outer,
            left_inner);
        OpenBabel::OBAtom *right_terminal = FirstOuterSingleBondNeighbor(
            *right_outer,
            right_inner);
        if (left_terminal == nullptr || right_terminal == nullptr)
        {
            return true;
        }

        return molgr::utils::CalculateShapeQuality(
                   left_terminal->GetVector(),
                   left_outer->GetVector(),
                   right_outer->GetVector(),
                   right_terminal->GetVector()) <= volume_tolerance;
    }

    int RingFormalChargeSum(OpenBabel::OBMol &mol, const OpenBabel::OBRing &ring)
    {
        int charge_sum = 0;
        for (int atom_idx : ring._path)
        {
            const OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
            if (atom == nullptr)
            {
                continue;
            }
            charge_sum += atom->GetFormalCharge();
        }
        return charge_sum;
    }

    bool IsChargeAcceptedAromaticRing(OpenBabel::OBMol &mol, OpenBabel::OBRing &ring)
    {
        if (!molgr::vendor::openbabel_threading::RingIsAromatic(ring))
        {
            return false;
        }
        return std::abs(RingFormalChargeSum(mol, ring)) <
               kAromaticRingFormalChargeAbsRejectionThreshold;
    }

    bool AtomHasUnpairedElectrons(const OpenBabel::OBAtom &atom)
    {
        return molgr::utils::GetUnpairedElectronCount(atom) > 0;
    }

    void PrepareTopologyWorkingMolecule(OpenBabel::OBMol &mol)
    {
        molgr::vendor::openbabel_threading::ResetAndAssignAromaticFlags(mol);
        mol.SetHybridizationPerceived(false);
        molgr::vendor::openbabel_threading::EnsureHybridizationPerceived(mol);
    }

    int AdditionalAtomPiElectrons(
        const OpenBabel::OBAtom &atom,
        bool incident_to_ring_multiple_bond)
    {
        if (incident_to_ring_multiple_bond || atom.GetAtomicNum() == 1)
        {
            return 0;
        }
        if (atom.GetHyb() != 2)
        {
            return 0;
        }
        if (atom.GetFormalCharge() < 0)
        {
            return 2;
        }
        if (atom.GetAtomicNum() != 6)
        {
            return 2;
        }
        if (AtomHasUnpairedElectrons(atom))
        {
            return 1;
        }
        return 0;
    }

    int RingPiElectronCount(OpenBabel::OBMol &mol, const OpenBabel::OBRing &ring)
    {
        if (ring._path.size() < 3)
        {
            return -1;
        }

        int pi_electron_count = 0;
        IntFlagSet atoms_incident_to_ring_multiple_bond(
            static_cast<std::size_t>(mol.NumAtoms()) + 1);
        for (std::size_t offset = 0; offset < ring._path.size(); ++offset)
        {
            const int begin_idx = ring._path[offset];
            const int end_idx = ring._path[(offset + 1) % ring._path.size()];
            OpenBabel::OBBond *bond = mol.GetBond(begin_idx, end_idx);
            if (bond == nullptr)
            {
                return -1;
            }
            if (bond->GetBondOrder() >= 2)
            {
                pi_electron_count += 2;
                atoms_incident_to_ring_multiple_bond.Add(begin_idx);
                atoms_incident_to_ring_multiple_bond.Add(end_idx);
            }
        }

        for (int atom_idx : ring._path)
        {
            const OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
            if (atom == nullptr)
            {
                continue;
            }
            pi_electron_count += AdditionalAtomPiElectrons(
                *atom,
                atoms_incident_to_ring_multiple_bond.Contains(atom_idx));
        }
        return pi_electron_count;
    }

    bool HasHuckelPiElectronCount(int pi_electron_count)
    {
        return pi_electron_count >= 2 && (pi_electron_count - 2) % 4 == 0;
    }

    bool IsHuckelAcceptedAromaticRing(OpenBabel::OBMol &mol, OpenBabel::OBRing &ring)
    {
        return IsChargeAcceptedAromaticRing(mol, ring) &&
               HasHuckelPiElectronCount(RingPiElectronCount(mol, ring));
    }

    bool RingsShareFusedBond(const OpenBabel::OBRing &lhs, const OpenBabel::OBRing &rhs)
    {
        int shared_atom_count = 0;
        for (const int lhs_atom_idx : lhs._path)
        {
            if (std::find(rhs._path.begin(), rhs._path.end(), lhs_atom_idx) == rhs._path.end())
            {
                continue;
            }
            ++shared_atom_count;
            if (shared_atom_count >= 2)
            {
                return true;
            }
        }
        return false;
    }

    std::vector<std::vector<OpenBabel::OBRing *>> AromaticRingSystems(OpenBabel::OBMol &mol)
    {
        std::vector<OpenBabel::OBRing *> aromatic_rings;
        FOR_RINGS_OF_MOL(ring_iter, mol)
        {
            OpenBabel::OBRing *ring = &(*ring_iter);
            if (ring != nullptr && molgr::vendor::openbabel_threading::RingIsAromatic(*ring))
            {
                aromatic_rings.push_back(ring);
            }
        }

        std::vector<std::vector<OpenBabel::OBRing *>> systems;
        std::vector<bool> assigned(aromatic_rings.size(), false);
        for (std::size_t ring_index = 0; ring_index < aromatic_rings.size(); ++ring_index)
        {
            if (assigned[ring_index])
            {
                continue;
            }
            assigned[ring_index] = true;
            std::vector<OpenBabel::OBRing *> system{aromatic_rings[ring_index]};
            for (std::size_t system_index = 0; system_index < system.size(); ++system_index)
            {
                for (std::size_t candidate_index = 0;
                     candidate_index < aromatic_rings.size();
                     ++candidate_index)
                {
                    if (assigned[candidate_index] ||
                        !RingsShareFusedBond(*system[system_index], *aromatic_rings[candidate_index]))
                    {
                        continue;
                    }
                    assigned[candidate_index] = true;
                    system.push_back(aromatic_rings[candidate_index]);
                }
            }
            systems.push_back(std::move(system));
        }
        return systems;
    }

    IntFlagSet RingSystemAtomIndices(
        const std::vector<OpenBabel::OBRing *> &rings,
        std::size_t atom_capacity)
    {
        IntFlagSet atom_indices(atom_capacity);
        for (const OpenBabel::OBRing *ring : rings)
        {
            if (ring == nullptr)
            {
                continue;
            }
            for (const int atom_idx : ring->_path)
            {
                atom_indices.Add(atom_idx);
            }
        }
        return atom_indices;
    }

    bool IsAcceptedAromaticRingSystem(
        OpenBabel::OBMol &mol,
        const std::vector<OpenBabel::OBRing *> &rings)
    {
        if (rings.empty())
        {
            return false;
        }
        if (rings.size() == 1)
        {
            return rings.front() != nullptr &&
                   IsHuckelAcceptedAromaticRing(mol, *rings.front());
        }

        const IntFlagSet atom_indices = RingSystemAtomIndices(
            rings,
            static_cast<std::size_t>(mol.NumAtoms()) + 1);
        int charge_sum = 0;
        for (const int atom_idx : atom_indices.values)
        {
            const OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
            if (atom != nullptr)
            {
                charge_sum += atom->GetFormalCharge();
            }
        }
        return std::abs(charge_sum) < kAromaticRingFormalChargeAbsRejectionThreshold;
    }

    double AromaticRingStabilityWeight(
        OpenBabel::OBMol &mol,
        const OpenBabel::OBRing &ring,
        const molgr::config::OrganicTopologyConfig &config)
    {
        std::vector<const OpenBabel::OBAtom *> heavy_atoms;
        for (int atom_idx : ring._path)
        {
            const OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
            if (atom == nullptr || atom->GetAtomicNum() == 1)
            {
                continue;
            }
            heavy_atoms.push_back(atom);
        }
        if (heavy_atoms.empty())
        {
            return 0.0;
        }

        const int ring_size = static_cast<int>(heavy_atoms.size());
        int hetero_count = 0;
        int charge_count = 0;
        int radical_count = 0;
        for (const OpenBabel::OBAtom *atom : heavy_atoms)
        {
            if (atom->GetAtomicNum() != 6)
            {
                ++hetero_count;
            }
            if (atom->GetFormalCharge() != 0)
            {
                ++charge_count;
            }
            if (AtomHasUnpairedElectrons(*atom))
            {
                ++radical_count;
            }
        }

        if (ring_size == 6 && hetero_count == 0 && charge_count == 0 && radical_count == 0)
        {
            return config.aromatic_stability_benzene_score;
        }

        const double size_factor =
            ring_size == 6
                ? config.aromatic_stability_ring_size_6_factor
                : (ring_size == 5
                       ? config.aromatic_stability_ring_size_5_factor
                       : config.aromatic_stability_other_ring_size_factor);
        const double hetero_factor = std::max(
            config.aromatic_stability_min_hetero_factor,
            1.0 - config.aromatic_stability_hetero_atom_penalty *
                      static_cast<double>(hetero_count));
        const double charge_factor = std::max(
            config.aromatic_stability_min_charge_factor,
            1.0 - config.aromatic_stability_formal_charge_penalty *
                      static_cast<double>(charge_count));
        const double radical_factor = std::max(
            config.aromatic_stability_min_radical_factor,
            1.0 - config.aromatic_stability_radical_penalty *
                      static_cast<double>(radical_count));
        return std::min(
            config.aromatic_stability_other_ring_max_score,
            size_factor * hetero_factor * charge_factor * radical_factor);
    }

    bool IsMultipleLikeBond(const OpenBabel::OBBond &bond)
    {
        OpenBabel::OBBond &mutable_bond = const_cast<OpenBabel::OBBond &>(bond);
        return molgr::vendor::openbabel_threading::BondIsAromatic(mutable_bond) ||
               bond.GetBondOrder() >= 2;
    }

    bool IsPiActiveElectronCenter(const OpenBabel::OBAtom &atom)
    {
        OpenBabel::OBAtom &mutable_atom = const_cast<OpenBabel::OBAtom &>(atom);
        if (atom.GetAtomicNum() == 1 || mutable_atom.IsMetal())
        {
            return false;
        }
        const int lone_pair_count = molgr::utils::GetLonePairCount(atom);
        if (atom.GetFormalCharge() == 0 &&
            molgr::utils::GetUnpairedElectronCount(atom) == 0 &&
            lone_pair_count == 0 &&
            !molgr::utils::HasUnresolvedTwoElectronCenter(atom))
        {
            return false;
        }

        const auto element_info_iter = molgr::kNonMetalDict.find(atom.GetAtomicNum());
        if (element_info_iter == molgr::kNonMetalDict.end())
        {
            return false;
        }
        const bool is_under_saturated =
            element_info_iter->second.default_valence > atom.GetTotalValence();
        if (lone_pair_count > 0)
        {
            return is_under_saturated || atom.GetHyb() == 1 || atom.GetHyb() == 2;
        }
        return is_under_saturated;
    }

    int AttachedHydrogenCount(const OpenBabel::OBAtom &atom)
    {
        return static_cast<int>(atom.GetImplicitHCount()) +
               static_cast<int>(atom.ExplicitHydrogenCount());
    }

    bool IsHyperconjugativeDonor(const OpenBabel::OBAtom &atom)
    {
        return atom.GetAtomicNum() == 6 && atom.GetFormalCharge() == 0 &&
               atom.GetHyb() == 3 && molgr::utils::GetUnpairedElectronCount(atom) == 0 &&
               molgr::utils::GetLonePairCount(atom) == 0 &&
               !molgr::utils::HasUnresolvedTwoElectronCenter(atom) &&
               AttachedHydrogenCount(atom) > 0;
    }

    bool IsHyperconjugationAcceptor(
        const OpenBabel::OBAtom &atom,
        bool incident_to_multiple_like_bond)
    {
        OpenBabel::OBAtom &mutable_atom = const_cast<OpenBabel::OBAtom &>(atom);
        if (atom.GetAtomicNum() == 1 || mutable_atom.IsMetal())
        {
            return false;
        }
        if (incident_to_multiple_like_bond)
        {
            return true;
        }
        if (atom.GetFormalCharge() <= 0 && molgr::utils::GetUnpairedElectronCount(atom) == 0)
        {
            return false;
        }
        const auto element_info_iter = molgr::kNonMetalDict.find(atom.GetAtomicNum());
        return element_info_iter != molgr::kNonMetalDict.end() &&
               element_info_iter->second.default_valence > atom.GetTotalValence();
    }

    std::pair<int, int> HyperconjugationMetrics(OpenBabel::OBMol &mol)
    {
        IntFlagSet atoms_incident_to_multiple_like_bond(
            static_cast<std::size_t>(mol.NumAtoms()) + 1);
        FOR_BONDS_OF_MOL(bond_iter, mol)
        {
            if (!IsMultipleLikeBond(*bond_iter))
            {
                continue;
            }
            atoms_incident_to_multiple_like_bond.Add(
                static_cast<int>(bond_iter->GetBeginAtomIdx()));
            atoms_incident_to_multiple_like_bond.Add(
                static_cast<int>(bond_iter->GetEndAtomIdx()));
        }

        IntFlagSet donor_atom_indices(static_cast<std::size_t>(mol.NumAtoms()) + 1);
        int score = 0;
        FOR_BONDS_OF_MOL(bond_iter, mol)
        {
            if (molgr::vendor::openbabel_threading::BondIsAromatic(*bond_iter) ||
                bond_iter->GetBondOrder() != 1)
            {
                continue;
            }
            OpenBabel::OBAtom *begin_atom = bond_iter->GetBeginAtom();
            OpenBabel::OBAtom *end_atom = bond_iter->GetEndAtom();
            if (begin_atom == nullptr || end_atom == nullptr)
            {
                continue;
            }
            for (const auto &[donor, acceptor] : {
                     std::pair<OpenBabel::OBAtom *, OpenBabel::OBAtom *>{begin_atom, end_atom},
                     std::pair<OpenBabel::OBAtom *, OpenBabel::OBAtom *>{end_atom, begin_atom}})
            {
                if (!IsHyperconjugativeDonor(*donor) ||
                    !IsHyperconjugationAcceptor(
                        *acceptor,
                        atoms_incident_to_multiple_like_bond.Contains(
                            static_cast<int>(acceptor->GetIdx()))))
                {
                    continue;
                }
                donor_atom_indices.Add(static_cast<int>(donor->GetIdx()));
                score += AttachedHydrogenCount(*donor);
            }
        }
        return {static_cast<int>(donor_atom_indices.Size()), score};
    }

    bool IsHeavyAtomBond(const OpenBabel::OBBond &bond)
    {
        const OpenBabel::OBAtom *begin_atom = bond.GetBeginAtom();
        const OpenBabel::OBAtom *end_atom = bond.GetEndAtom();
        return begin_atom != nullptr && end_atom != nullptr && begin_atom->GetAtomicNum() != 1 &&
               end_atom->GetAtomicNum() != 1;
    }

    std::vector<int> AtomComponentIds(
        const std::vector<HeavyBondRef> &heavy_bonds,
        const IntFlagSet &selected_bond_indices,
        std::size_t atom_capacity)
    {
        std::vector<std::vector<int>> neighbors(atom_capacity);
        for (const HeavyBondRef &heavy_bond : heavy_bonds)
        {
            if (!selected_bond_indices.Contains(heavy_bond.bond_idx) ||
                heavy_bond.begin_atom == nullptr || heavy_bond.end_atom == nullptr)
            {
                continue;
            }
            const auto begin_idx = static_cast<std::size_t>(heavy_bond.begin_atom->GetIdx());
            const auto end_idx = static_cast<std::size_t>(heavy_bond.end_atom->GetIdx());
            if (begin_idx >= neighbors.size() || end_idx >= neighbors.size())
            {
                continue;
            }
            neighbors[begin_idx].push_back(static_cast<int>(end_idx));
            neighbors[end_idx].push_back(static_cast<int>(begin_idx));
        }

        std::vector<int> component_ids(atom_capacity, -1);
        int next_component_id = 0;
        for (std::size_t atom_idx = 0; atom_idx < neighbors.size(); ++atom_idx)
        {
            if (neighbors[atom_idx].empty() || component_ids[atom_idx] >= 0)
            {
                continue;
            }
            std::vector<int> stack{static_cast<int>(atom_idx)};
            while (!stack.empty())
            {
                const int current_idx = stack.back();
                stack.pop_back();
                if (current_idx < 0 ||
                    static_cast<std::size_t>(current_idx) >= component_ids.size() ||
                    component_ids[static_cast<std::size_t>(current_idx)] >= 0)
                {
                    continue;
                }
                component_ids[static_cast<std::size_t>(current_idx)] = next_component_id;
                for (int neighbor_idx : neighbors[static_cast<std::size_t>(current_idx)])
                {
                    if (neighbor_idx >= 0 &&
                        static_cast<std::size_t>(neighbor_idx) < component_ids.size() &&
                        component_ids[static_cast<std::size_t>(neighbor_idx)] < 0)
                    {
                        stack.push_back(neighbor_idx);
                    }
                }
            }
            ++next_component_id;
        }
        return component_ids;
    }

    ValidatedConjugationTopology ValidatedConjugatedTopology(
        OpenBabel::OBMol &mol,
        const molgr::config::OrganicTopologyConfig &config)
    {
        std::vector<HeavyBondRef> heavy_bonds;
        heavy_bonds.reserve(static_cast<std::size_t>(mol.NumBonds()));
        std::vector<std::vector<int>> nonaromatic_multiple_bonds_by_atom(
            static_cast<std::size_t>(mol.NumAtoms()) + 1);
        IntFlagSet atom_has_adjacent_multiple_like_bond(
            static_cast<std::size_t>(mol.NumAtoms()) + 1);
        IntFlagSet atom_has_adjacent_alternating_single_bond(
            static_cast<std::size_t>(mol.NumAtoms()) + 1);
        IntFlagSet aromatic_bond_indices(
            static_cast<std::size_t>(mol.NumBonds()) + 1);
        IntFlagSet multiple_like_bond_indices(
            static_cast<std::size_t>(mol.NumBonds()) + 1);

        FOR_BONDS_OF_MOL(bond_iter, mol)
        {
            OpenBabel::OBBond &bond = *bond_iter;
            if (!IsHeavyAtomBond(bond))
            {
                continue;
            }
            OpenBabel::OBAtom *begin_atom = bond.GetBeginAtom();
            OpenBabel::OBAtom *end_atom = bond.GetEndAtom();
            const int bond_idx = static_cast<int>(bond.GetIdx());
            heavy_bonds.push_back(HeavyBondRef{
                &bond,
                begin_atom,
                end_atom,
                bond_idx,
            });
            if (molgr::vendor::openbabel_threading::BondIsAromatic(bond))
            {
                continue;
            }
            if (bond.GetBondOrder() >= 2)
            {
                nonaromatic_multiple_bonds_by_atom[begin_atom->GetIdx()].push_back(bond_idx);
                nonaromatic_multiple_bonds_by_atom[end_atom->GetIdx()].push_back(bond_idx);
            }
        }

        IntFlagSet cumulated_multiple_bond_indices(
            static_cast<std::size_t>(mol.NumBonds()) + 1);
        std::vector<IntFlagSet> cumulated_bond_indices_by_center(
            static_cast<std::size_t>(mol.NumAtoms()) + 1);
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom &atom = *atom_iter;
            const auto atom_idx = static_cast<std::size_t>(atom.GetIdx());
            if (atom.GetHyb() != 1 ||
                atom_idx >= nonaromatic_multiple_bonds_by_atom.size() ||
                nonaromatic_multiple_bonds_by_atom[atom_idx].size() < 2)
            {
                continue;
            }
            for (int bond_idx : nonaromatic_multiple_bonds_by_atom[atom_idx])
            {
                cumulated_multiple_bond_indices.Add(bond_idx);
                cumulated_bond_indices_by_center[atom_idx].Add(bond_idx);
            }
        }

        for (const HeavyBondRef &heavy_bond : heavy_bonds)
        {
            OpenBabel::OBBond *bond = heavy_bond.bond;
            OpenBabel::OBAtom *begin_atom = heavy_bond.begin_atom;
            OpenBabel::OBAtom *end_atom = heavy_bond.end_atom;
            if (bond == nullptr || begin_atom == nullptr || end_atom == nullptr ||
                !IsMultipleLikeBond(*bond))
            {
                continue;
            }
            const bool is_aromatic =
                molgr::vendor::openbabel_threading::BondIsAromatic(*bond);
            atom_has_adjacent_multiple_like_bond.Add(static_cast<int>(begin_atom->GetIdx()));
            atom_has_adjacent_multiple_like_bond.Add(static_cast<int>(end_atom->GetIdx()));
            if (is_aromatic)
            {
                aromatic_bond_indices.Add(heavy_bond.bond_idx);
            }
            else if (!cumulated_multiple_bond_indices.Contains(heavy_bond.bond_idx))
            {
                multiple_like_bond_indices.Add(heavy_bond.bond_idx);
            }
        }

        IntFlagSet conjugated_bond_indices(
            static_cast<std::size_t>(mol.NumBonds()) + 1);
        IntFlagSet alternating_single_bond_indices(
            static_cast<std::size_t>(mol.NumBonds()) + 1);
        for (const HeavyBondRef &heavy_bond : heavy_bonds)
        {
            OpenBabel::OBBond *bond = heavy_bond.bond;
            OpenBabel::OBAtom *begin_atom = heavy_bond.begin_atom;
            OpenBabel::OBAtom *end_atom = heavy_bond.end_atom;
            if (bond == nullptr || begin_atom == nullptr || end_atom == nullptr ||
                molgr::vendor::openbabel_threading::BondIsAromatic(*bond) ||
                bond->GetBondOrder() != 1)
            {
                continue;
            }
            const bool begin_has_pi_bond =
                atom_has_adjacent_multiple_like_bond.Contains(static_cast<int>(begin_atom->GetIdx()));
            const bool end_has_pi_bond =
                atom_has_adjacent_multiple_like_bond.Contains(static_cast<int>(end_atom->GetIdx()));
            if ((!begin_has_pi_bond && !end_has_pi_bond) ||
                (!begin_has_pi_bond && !IsPiActiveElectronCenter(*begin_atom)) ||
                (!end_has_pi_bond && !IsPiActiveElectronCenter(*end_atom)) ||
                (begin_has_pi_bond && end_has_pi_bond &&
                 !AlternatingDoubleBondsAreGeometricallyConjugated(
                     mol,
                     *begin_atom,
                     *end_atom,
                     nonaromatic_multiple_bonds_by_atom,
                     config.conjugation_normalized_tetrahedron_volume_tolerance)))
            {
                continue;
            }
            alternating_single_bond_indices.Add(heavy_bond.bond_idx);
            atom_has_adjacent_alternating_single_bond.Add(static_cast<int>(begin_atom->GetIdx()));
            atom_has_adjacent_alternating_single_bond.Add(static_cast<int>(end_atom->GetIdx()));
        }

        for (int bond_idx : aromatic_bond_indices.values)
        {
            conjugated_bond_indices.Add(bond_idx);
        }
        for (int bond_idx : alternating_single_bond_indices.values)
        {
            conjugated_bond_indices.Add(bond_idx);
        }
        for (const HeavyBondRef &heavy_bond : heavy_bonds)
        {
            OpenBabel::OBBond *bond = heavy_bond.bond;
            OpenBabel::OBAtom *begin_atom = heavy_bond.begin_atom;
            OpenBabel::OBAtom *end_atom = heavy_bond.end_atom;
            if (bond == nullptr || begin_atom == nullptr || end_atom == nullptr ||
                !multiple_like_bond_indices.Contains(heavy_bond.bond_idx))
            {
                continue;
            }
            if (atom_has_adjacent_alternating_single_bond.Contains(static_cast<int>(begin_atom->GetIdx())) ||
                atom_has_adjacent_alternating_single_bond.Contains(static_cast<int>(end_atom->GetIdx())))
            {
                conjugated_bond_indices.Add(heavy_bond.bond_idx);
            }
        }

        const std::vector<int> base_component_ids = AtomComponentIds(
            heavy_bonds,
            conjugated_bond_indices,
            static_cast<std::size_t>(mol.NumAtoms()) + 1);
        for (std::size_t center_idx = 0;
             center_idx < cumulated_bond_indices_by_center.size();
             ++center_idx)
        {
            const IntFlagSet &center_bond_indices =
                cumulated_bond_indices_by_center[center_idx];
            if (center_bond_indices.Size() < 2)
            {
                continue;
            }

            IntFlagSet seen_component_ids;
            bool closes_outer_conjugated_path = false;
            for (int bond_idx : center_bond_indices.values)
            {
                OpenBabel::OBBond *bond = mol.GetBond(bond_idx);
                if (bond == nullptr)
                {
                    continue;
                }
                const OpenBabel::OBAtom *begin_atom = bond->GetBeginAtom();
                const OpenBabel::OBAtom *end_atom = bond->GetEndAtom();
                if (begin_atom == nullptr || end_atom == nullptr)
                {
                    continue;
                }
                const int begin_idx = static_cast<int>(begin_atom->GetIdx());
                const int end_idx = static_cast<int>(end_atom->GetIdx());
                const int outer_idx =
                    begin_idx == static_cast<int>(center_idx) ? end_idx : begin_idx;
                if (outer_idx < 0 ||
                    static_cast<std::size_t>(outer_idx) >= base_component_ids.size())
                {
                    continue;
                }
                const int component_id = base_component_ids[static_cast<std::size_t>(outer_idx)];
                if (component_id >= 0 && !seen_component_ids.Add(component_id))
                {
                    closes_outer_conjugated_path = true;
                    break;
                }
            }
            if (closes_outer_conjugated_path)
            {
                continue;
            }
            for (int bond_idx : center_bond_indices.values)
            {
                OpenBabel::OBBond *bond = mol.GetBond(bond_idx);
                if (bond == nullptr || bond->GetBeginAtom() == nullptr || bond->GetEndAtom() == nullptr)
                {
                    continue;
                }
                const int begin_idx = static_cast<int>(bond->GetBeginAtom()->GetIdx());
                const int end_idx = static_cast<int>(bond->GetEndAtom()->GetIdx());
                const int outer_idx =
                    begin_idx == static_cast<int>(center_idx) ? end_idx : begin_idx;
                if (atom_has_adjacent_alternating_single_bond.Contains(outer_idx))
                {
                    conjugated_bond_indices.Add(bond_idx);
                }
            }
        }
        return ValidatedConjugationTopology{
            std::move(conjugated_bond_indices),
            std::move(cumulated_bond_indices_by_center),
        };
    }
}

namespace molgr
{
    namespace organic_topology
    {
        bool IsConjugatedBond(const OpenBabel::OBBond &bond)
        {
            const OpenBabel::OBMol *parent = const_cast<OpenBabel::OBBond &>(bond).GetParent();
            if (parent == nullptr)
            {
                return false;
            }
            OpenBabel::OBMol working = molgr::utils::CloneMolTopologyForScoring(*parent);
            PrepareTopologyWorkingMolecule(working);
            const ValidatedConjugationTopology conjugated_topology =
                ValidatedConjugatedTopology(
                    working,
                    molgr::config::GetDefaultConfig().organic_topology);
            return conjugated_topology.bond_indices.Contains(static_cast<int>(bond.GetIdx()));
        }

        OrganicTopologyMetrics ComputeOrganicTopologyMetrics(
            const OpenBabel::OBMol &mol,
            const molgr::config::OrganicTopologyConfig &config)
        {
            OpenBabel::OBMol working = molgr::utils::CloneMolTopologyForScoring(mol);
            PrepareTopologyWorkingMolecule(working);

            OrganicTopologyMetrics metrics;
            IntFlagSet aromatic_atom_indices(
                static_cast<std::size_t>(working.NumAtoms()) + 1);
            for (const auto &ring_system : AromaticRingSystems(working))
            {
                if (!IsAcceptedAromaticRingSystem(working, ring_system))
                {
                    continue;
                }
                metrics.aromatic_ring_count += static_cast<int>(ring_system.size());
                for (const OpenBabel::OBRing *ring : ring_system)
                {
                    if (ring == nullptr)
                    {
                        continue;
                    }
                    metrics.aromatic_stability_score += AromaticRingStabilityWeight(
                        working,
                        *ring,
                        config);
                    for (int atom_idx : ring->_path)
                    {
                        const OpenBabel::OBAtom *atom = working.GetAtom(atom_idx);
                        if (atom == nullptr || atom->GetAtomicNum() == 1)
                        {
                            continue;
                        }
                        aromatic_atom_indices.Add(atom_idx);
                    }
                }
            }
            metrics.aromatic_atom_count = static_cast<int>(aromatic_atom_indices.Size());

            const ValidatedConjugationTopology conjugated_topology =
                ValidatedConjugatedTopology(working, config);
            const IntFlagSet &conjugated_bond_indices = conjugated_topology.bond_indices;
            metrics.conjugated_bond_count = static_cast<int>(conjugated_bond_indices.Size());

            std::vector<std::vector<int>> incident_conjugated_bonds(
                static_cast<std::size_t>(working.NumAtoms()));
            std::vector<std::pair<int, int>> conjugated_bond_atoms(
                static_cast<std::size_t>(working.NumBonds()),
                std::pair<int, int>{-1, -1});
            IntFlagSet conjugated_atom_indices(
                static_cast<std::size_t>(working.NumAtoms()));
            FOR_BONDS_OF_MOL(bond_iter, working)
            {
                OpenBabel::OBBond &bond = *bond_iter;
                if (!conjugated_bond_indices.Contains(static_cast<int>(bond.GetIdx())))
                {
                    continue;
                }
                OpenBabel::OBAtom *begin_atom = bond.GetBeginAtom();
                OpenBabel::OBAtom *end_atom = bond.GetEndAtom();
                if (begin_atom == nullptr || end_atom == nullptr ||
                    begin_atom->GetAtomicNum() == 1 || end_atom->GetAtomicNum() == 1)
                {
                    continue;
                }
                const int begin_idx = static_cast<int>(begin_atom->GetIdx()) - 1;
                const int end_idx = static_cast<int>(end_atom->GetIdx()) - 1;
                if (begin_idx < 0 || end_idx < 0)
                {
                    continue;
                }
                conjugated_atom_indices.Add(begin_idx);
                conjugated_atom_indices.Add(end_idx);
                const int bond_idx = static_cast<int>(bond.GetIdx());
                if (bond_idx < 0 ||
                    static_cast<std::size_t>(bond_idx) >= conjugated_bond_atoms.size())
                {
                    continue;
                }
                conjugated_bond_atoms[static_cast<std::size_t>(bond_idx)] = {
                    begin_idx,
                    end_idx,
                };
                const auto begin_offset = static_cast<std::size_t>(begin_idx);
                const auto end_offset = static_cast<std::size_t>(end_idx);
                if (begin_offset >= incident_conjugated_bonds.size() ||
                    end_offset >= incident_conjugated_bonds.size())
                {
                    continue;
                }
                incident_conjugated_bonds[begin_offset].push_back(bond_idx);
                incident_conjugated_bonds[end_offset].push_back(bond_idx);
            }
            metrics.conjugated_atom_count = static_cast<int>(conjugated_atom_indices.Size());
            metrics.conjugated_atom_indices = conjugated_atom_indices.values;
            std::sort(metrics.conjugated_atom_indices.begin(), metrics.conjugated_atom_indices.end());

            std::vector<std::vector<int>> conjugated_bond_neighbors(
                static_cast<std::size_t>(working.NumBonds()));
            for (std::size_t atom_idx = 0;
                 atom_idx < incident_conjugated_bonds.size();
                 ++atom_idx)
            {
                const std::vector<int> &incident = incident_conjugated_bonds[atom_idx];
                const std::size_t one_based_atom_idx = atom_idx + 1;
                const IntFlagSet *cumulated_at_center =
                    one_based_atom_idx <
                            conjugated_topology.cumulated_bond_indices_by_center.size()
                        ? &conjugated_topology
                               .cumulated_bond_indices_by_center[one_based_atom_idx]
                        : nullptr;
                for (std::size_t first_offset = 0;
                     first_offset < incident.size();
                     ++first_offset)
                {
                    const int first_bond_idx = incident[first_offset];
                    for (std::size_t second_offset = first_offset + 1;
                         second_offset < incident.size();
                         ++second_offset)
                    {
                        const int second_bond_idx = incident[second_offset];
                        if (cumulated_at_center != nullptr &&
                            cumulated_at_center->Contains(first_bond_idx) &&
                            cumulated_at_center->Contains(second_bond_idx))
                        {
                            continue;
                        }
                        conjugated_bond_neighbors[static_cast<std::size_t>(first_bond_idx)]
                            .push_back(second_bond_idx);
                        conjugated_bond_neighbors[static_cast<std::size_t>(second_bond_idx)]
                            .push_back(first_bond_idx);
                    }
                }
            }

            IntFlagSet visited_bonds(static_cast<std::size_t>(working.NumBonds()));
            for (int bond_idx : conjugated_bond_indices.values)
            {
                if (visited_bonds.Contains(bond_idx))
                {
                    continue;
                }
                std::vector<int> stack{bond_idx};
                IntFlagSet component_atom_indices(
                    static_cast<std::size_t>(working.NumAtoms()));
                while (!stack.empty())
                {
                    const int current_bond_idx = stack.back();
                    stack.pop_back();
                    if (!visited_bonds.Add(current_bond_idx) || current_bond_idx < 0 ||
                        static_cast<std::size_t>(current_bond_idx) >=
                            conjugated_bond_atoms.size())
                    {
                        continue;
                    }
                    const auto [begin_idx, end_idx] =
                        conjugated_bond_atoms[static_cast<std::size_t>(current_bond_idx)];
                    component_atom_indices.Add(begin_idx);
                    component_atom_indices.Add(end_idx);
                    if (static_cast<std::size_t>(current_bond_idx) >=
                        conjugated_bond_neighbors.size())
                    {
                        continue;
                    }
                    for (int neighbor_bond_idx :
                         conjugated_bond_neighbors[static_cast<std::size_t>(current_bond_idx)])
                    {
                        if (!visited_bonds.Contains(neighbor_bond_idx))
                        {
                            stack.push_back(neighbor_bond_idx);
                        }
                    }
                }
                metrics.max_conjugated_component_size =
                    std::max(
                        metrics.max_conjugated_component_size,
                        static_cast<int>(component_atom_indices.Size()));
            }
            const auto [hyperconjugative_donor_count, hyperconjugation_score] =
                HyperconjugationMetrics(working);
            metrics.hyperconjugative_donor_count = hyperconjugative_donor_count;
            metrics.hyperconjugation_score = hyperconjugation_score;
            return metrics;
        }
    }
}
