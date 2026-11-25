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
#include <array>
#include <cmath>
#include <optional>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <vector>

namespace
{
    constexpr int kAromaticRingFormalChargeAbsRejectionThreshold = 4;

    /// Dense membership flags and an insertion list for atom or bond indices.
    struct IntFlagSet
    {
        std::vector<unsigned char> present;
        std::vector<int> values;

        /// Reserve the initial range of indices for constant-time membership tests.
        /// @param initial_size Initial range of atom or bond indices to accommodate.
        explicit IntFlagSet(std::size_t initial_size = 0)
            : present(initial_size, 0)
        {
        }

        /// @param index Index of the requested entry.
        /// @return True when the index has been inserted.
        bool Contains(int index) const
        {
            return index >= 0 &&
                   static_cast<std::size_t>(index) < present.size() &&
                   present[static_cast<std::size_t>(index)] != 0;
        }

        /// Insert a new index and report whether the membership set grew.
        /// @param index Index of the requested entry.
        /// @return True if the index was newly added.
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

        /// @return The number of distinct indices inserted.
        std::size_t Size() const
        {
            return values.size();
        }
    };

    /// Heavy-atom bond and endpoint indices used to assemble conjugated components.
    struct HeavyBondRef
    {
        OpenBabel::OBBond *bond = nullptr;
        OpenBabel::OBAtom *begin_atom = nullptr;
        OpenBabel::OBAtom *end_atom = nullptr;
        int bond_idx = -1;
    };

    /// Geometry-validated atom and bond sets for scoring ligand conjugation.
    struct ValidatedConjugationTopology
    {
        IntFlagSet bond_indices;
        std::vector<IntFlagSet> cumulated_bond_indices_by_center;
    };

    using PiPlaneTensor = std::array<double, 9>;

    /// A tetrahedral second-row center cannot supply a trigonal pi plane.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True for a strongly pyramidal atom inconsistent with a local pi plane.
    bool StronglyPyramidalPiEndpoint(OpenBabel::OBAtom &atom)
    {
        if ((atom.GetAtomicNum() != 6 && atom.GetAtomicNum() != 7) ||
            atom.GetExplicitDegree() != 3 || atom.GetImplicitHCount() != 0) return false;
        std::vector<OpenBabel::OBAtom *> neighbors;
        FOR_NBORS_OF_ATOM(neighbor, &atom)
        {
            if (atom.GetDistance(&*neighbor) < 1e-8) return false;
            neighbors.push_back(&*neighbor);
        }
        double sum = 0.0;
        for (std::size_t i = 0; i < neighbors.size(); ++i)
            for (std::size_t j = i + 1; j < neighbors.size(); ++j)
            {
                const double angle = atom.GetParent()->GetAngle(neighbors[i], &atom, neighbors[j]);
                if (!(80.0 < angle && angle < 125.0)) return false;
                sum += angle;
            }
        return sum < 330.0;
    }

    /// Average unoriented plane normals over the neighboring bonds.
    /// @param center Atom or site occupying this position in the local chemical motif.
    /// @param partner Atom or site occupying this position in the local chemical motif.
    /// @param single_bonds_only Whether to build the plane estimate from single bonds only.
    /// @return Local plane tensor, or an empty optional for insufficient geometry.
    std::optional<PiPlaneTensor> LocalPiPlaneTensor(
        OpenBabel::OBAtom &center, OpenBabel::OBAtom &partner,
        bool single_bonds_only = true)
    {
        auto axis = partner.GetVector() - center.GetVector();
        const double axis_length = axis.length();
        if (axis_length < 1e-12) return std::nullopt;
        axis /= axis_length;
        PiPlaneTensor tensor{};
        int count = 0;
        FOR_BONDS_OF_ATOM(bond_iter, &center)
        {
            auto &bond = *bond_iter;
            if (single_bonds_only &&
                (molgr::vendor::openbabel_threading::BondIsAromatic(bond) || bond.GetBondOrder() != 1))
                continue;
            auto *neighbor = bond.GetNbrAtom(&center);
            if (neighbor == nullptr || neighbor->GetIdx() == partner.GetIdx()) continue;
            auto vector = neighbor->GetVector() - center.GetVector();
            const double length = vector.length();
            if (length < 1e-12) continue;
            vector /= length;
            auto normal = OpenBabel::cross(axis, vector);
            const double normal_length = normal.length();
            if (normal_length < 1e-8) continue;
            normal /= normal_length;
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    tensor[i * 3 + j] += normal[i] * normal[j];
            ++count;
        }
        if (count == 0) return std::nullopt;
        for (auto &value : tensor) value /= count;
        return tensor;
    }

    /// Require compatible local pi planes, independent of atom/bond ordering.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param left_inner Atom at the inner end of an alternating pi bond.
    /// @param right_inner Atom at the inner end of an alternating pi bond.
    /// @param multiple_bonds_by_atom Multiple-bond neighbors indexed by atom.
    /// @param min_plane_overlap Minimum allowed overlap of the local pi planes.
    /// @return True when the alternating bonds pass the local plane-overlap screen.
    bool AlternatingDoubleBondsAreGeometricallyConjugated(
        OpenBabel::OBMol &mol,
        OpenBabel::OBAtom &left_inner,
        OpenBabel::OBAtom &right_inner,
        const std::vector<std::vector<int>> &multiple_bonds_by_atom,
        double min_plane_overlap)
    {
        const auto left_idx = static_cast<std::size_t>(left_inner.GetIdx());
        const auto right_idx = static_cast<std::size_t>(right_inner.GetIdx());
        if (left_idx >= multiple_bonds_by_atom.size() ||
            right_idx >= multiple_bonds_by_atom.size() ||
            multiple_bonds_by_atom[left_idx].empty() ||
            multiple_bonds_by_atom[right_idx].empty()) return true;

        // Enumerate pi channels. Every explicit single-bonded neighbor, including
        // H, contributes to each unoriented local plane tensor; no atom/bond
        // index selects the geometry. Cumulated component boundaries stay below.
        for (const int left_bond_idx : multiple_bonds_by_atom[left_idx])
        {
            auto *left_bond = mol.GetBond(left_bond_idx);
            auto *left_outer = left_bond == nullptr ? nullptr : left_bond->GetNbrAtom(&left_inner);
            if (left_outer == nullptr || left_outer->GetAtomicNum() == 1) continue;
            for (const int right_bond_idx : multiple_bonds_by_atom[right_idx])
            {
                auto *right_bond = mol.GetBond(right_bond_idx);
                auto *right_outer = right_bond == nullptr ? nullptr : right_bond->GetNbrAtom(&right_inner);
                if (right_outer == nullptr || right_outer->GetAtomicNum() == 1) continue;
                if (StronglyPyramidalPiEndpoint(*left_outer) || StronglyPyramidalPiEndpoint(left_inner) ||
                    StronglyPyramidalPiEndpoint(right_inner) || StronglyPyramidalPiEndpoint(*right_outer))
                    continue;
                std::vector<PiPlaneTensor> planes;
                for (const auto &[center, partner] : std::array<std::pair<OpenBabel::OBAtom *, OpenBabel::OBAtom *>, 4>{{
                         {left_outer, &left_inner}, {&left_inner, left_outer},
                         {&right_inner, right_outer}, {right_outer, &right_inner}}})
                {
                    // Triple bonds retain both perpendicular pi channels;
                    // their substituents must not select one exclusive plane.
                    if (center->GetBond(partner)->GetBondOrder() >= 3) continue;
                    const auto plane = LocalPiPlaneTensor(*center, *partner);
                    if (plane) planes.push_back(*plane);
                }
                bool compatible = true;
                for (std::size_t i = 0; i < planes.size() && compatible; ++i)
                    for (std::size_t j = i + 1; j < planes.size(); ++j)
                    {
                        double overlap = 0.0;
                        for (std::size_t k = 0; k < 9; ++k)
                            overlap += planes[i][k] * planes[j][k];
                        if (overlap + 1e-12 < min_plane_overlap)
                        {
                            compatible = false;
                            break;
                        }
                    }
                // Missing/collinear coordinates cannot disprove conjugation.
                if (compatible) return true;
            }
        }
        return false;
    }

    /// Sum the formal charges of the atoms on the ring path.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param ring Ring path in the current molecule.
    /// @return Net formal charge of the ring.
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

    /// Test the ring's net charge against the aromatic scoring cutoff.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param ring Ring path in the current molecule.
    /// @return True if the ring or fused system qualifies for aromatic scoring.
    bool IsChargeAcceptedAromaticRing(OpenBabel::OBMol &mol, OpenBabel::OBRing &ring)
    {
        if (!molgr::vendor::openbabel_threading::RingIsAromatic(ring))
        {
            return false;
        }
        return std::abs(RingFormalChargeSum(mol, ring)) <
               kAromaticRingFormalChargeAbsRejectionThreshold;
    }

    /// Test for an explicit unpaired electron on the atom.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True if the atom carries at least one explicit unpaired electron.
    bool AtomHasUnpairedElectrons(const OpenBabel::OBAtom &atom)
    {
        return molgr::utils::GetUnpairedElectronCount(atom) > 0;
    }

    /// Prepare ring, aromaticity and hybridization data used by topology scoring.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    void PrepareTopologyWorkingMolecule(OpenBabel::OBMol &mol)
    {
        molgr::vendor::openbabel_threading::ResetAndAssignAromaticFlags(mol);
        mol.SetHybridizationPerceived(false);
        molgr::vendor::openbabel_threading::EnsureHybridizationPerceived(mol);
    }

    /// Count pi electrons contributed by an atom beyond ring multiple-bond contributions.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param incident_to_ring_multiple_bond Whether the atom already contributes through a ring multiple
    ///     bond.
    /// @return Estimated pi-electron contribution under the local aromaticity model.
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

    /// Sum ring pi bonds and eligible atomic contributions without counting a pi-bonded atom twice.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param ring Ring path in the current molecule.
    /// @return Pi-electron count, or -1 when the ring path or its bonds cannot be evaluated.
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

    /// Test the Hückel 4n+2 electron-count condition.
    /// @param pi_electron_count Number of electrons assigned to the ring pi system.
    /// @return True for two or more electrons satisfying the 4n+2 rule.
    bool HasHuckelPiElectronCount(int pi_electron_count)
    {
        return pi_electron_count >= 2 && (pi_electron_count - 2) % 4 == 0;
    }

    /// Check that both net charge and pi-electron count support aromatic scoring of the ring.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param ring Ring path in the current molecule.
    /// @return True if the ring or fused system qualifies for aromatic scoring.
    bool IsHuckelAcceptedAromaticRing(OpenBabel::OBMol &mol, OpenBabel::OBRing &ring)
    {
        return IsChargeAcceptedAromaticRing(mol, ring) &&
               HasHuckelPiElectronCount(RingPiElectronCount(mol, ring));
    }

    /// @param lhs First ring path.
    /// @param rhs Second ring path.
    /// @return True when the two ring paths contain a common edge.
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

    /// Group aromatic rings into systems connected by fused bonds.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @return Groups of aromatic rings joined by shared edges.
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

    /// Collect the unique atom indices across the supplied ring system.
    /// @param rings Rings belonging to the fused system.
    /// @param atom_capacity Initial range of atom or bond indices to accommodate.
    /// @return Unique atom indices belonging to the ring system.
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

    /// Check isolated rings by electron count and fused aromatic systems by their net formal charge.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param rings Rings belonging to the fused system.
    /// @return True if the ring or fused system qualifies for aromatic scoring.
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

    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param ring Ring path in the current molecule.
    /// @param config Aromatic stability weights and conjugation geometry thresholds.
    /// @return The configured aromatic stabilization weight for the accepted ring.
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

    /// Test whether the bond is aromatic or has multiple-bond character.
    /// @param bond Bond whose endpoints and order define the local environment.
    /// @return True when the local bond or atom meets the indicated conjugation criterion.
    bool IsMultipleLikeBond(const OpenBabel::OBBond &bond)
    {
        OpenBabel::OBBond &mutable_bond = const_cast<OpenBabel::OBBond &>(bond);
        return molgr::vendor::openbabel_threading::BondIsAromatic(mutable_bond) ||
               bond.GetBondOrder() >= 2;
    }

    /// Determine whether this atom's valence and electron labels permit participation in a pi system.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the local bond or atom meets the indicated conjugation criterion.
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
            // Search-active pairs are not the full Lewis inventory. Resolved
            // neutral sp2 heteroatoms can retain a conjugating p lone pair.
            const auto info = molgr::kNonMetalDict.find(atom.GetAtomicNum());
            if (info == molgr::kNonMetalDict.end() || atom.GetAtomicNum() == 6 ||
                atom.GetHyb() != 2)
            {
                return false;
            }
            const int valence = static_cast<int>(atom.GetTotalValence());
            const int outer = static_cast<int>(info->second.num_outer_electrons);
            const int paired = outer - valence;
            return outer + valence == 8 && paired >= 2 && paired % 2 == 0;
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

    /// Count implicit hydrogens and directly bonded hydrogen atoms.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return Total hydrogen count at this atom.
    int AttachedHydrogenCount(const OpenBabel::OBAtom &atom)
    {
        return static_cast<int>(atom.GetImplicitHCount()) +
               static_cast<int>(atom.ExplicitHydrogenCount());
    }

    /// Test whether a neutral carbon can supply a sigma donor for hyperconjugation.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the local bond or atom meets the indicated conjugation criterion.
    bool IsHyperconjugativeDonor(const OpenBabel::OBAtom &atom)
    {
        return atom.GetAtomicNum() == 6 && atom.GetFormalCharge() == 0 &&
               atom.GetHyb() == 3 && molgr::utils::GetUnpairedElectronCount(atom) == 0 &&
               molgr::utils::GetLonePairCount(atom) == 0 &&
               !molgr::utils::HasUnresolvedTwoElectronCenter(atom) &&
               AttachedHydrogenCount(atom) > 0;
    }

    /// Test whether this electronic environment can accept hyperconjugative donation.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param incident_to_multiple_like_bond Whether the atom is adjacent to an aromatic or multiple bond.
    /// @return True when the local bond or atom meets the indicated conjugation criterion.
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

    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @return The number of distinct donor atoms and the hyperconjugation score.
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

    /// @param bond Bond whose endpoints and order define the local environment.
    /// @return True when both bond endpoints are elements heavier than hydrogen.
    bool IsHeavyAtomBond(const OpenBabel::OBBond &bond)
    {
        const OpenBabel::OBAtom *begin_atom = bond.GetBeginAtom();
        const OpenBabel::OBAtom *end_atom = bond.GetEndAtom();
        return begin_atom != nullptr && end_atom != nullptr && begin_atom->GetAtomicNum() != 1 &&
               end_atom->GetAtomicNum() != 1;
    }

    /// Label connected atom components using only the selected heavy-atom bonds.
    /// @param heavy_bonds Heavy-atom edges and their endpoint indices.
    /// @param selected_bond_indices Indices of heavy-atom bonds retained in the conjugation graph.
    /// @param atom_capacity Initial range of atom or bond indices to accommodate.
    /// @return Component identifiers indexed by atom.
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

    /// Check carbon electron centers and neutral heteroatom lone-pair donation.
    /// @param donor Atom or site occupying this position in the local chemical motif.
    /// @param acceptor Atom or site occupying this position in the local chemical motif.
    /// @param volume_tolerance Tolerance on the donor's local out-of-plane volume.
    /// @param min_plane_overlap Minimum allowed overlap of the local pi planes.
    /// @return True when the local bond or atom meets the indicated conjugation criterion.
    bool LonePairIsGeometricallyConjugated(
        OpenBabel::OBAtom &donor,
        OpenBabel::OBAtom &acceptor,
        double volume_tolerance,
        double min_plane_overlap)
    {
        const int z = static_cast<int>(donor.GetAtomicNum());
        if (z == 6 || (z == 7 && donor.GetFormalCharge() == 0))
        {
            // Carbon electron centers and neutral N donors share the overlap
            // threshold used for alternating double bonds. A stricter volume
            // guard for N would favor an iminium contributor at identical geometry.
            if (StronglyPyramidalPiEndpoint(donor) || StronglyPyramidalPiEndpoint(acceptor))
                return false;
            FOR_BONDS_OF_ATOM(bond, &acceptor)
            {
                // Triple bonds retain both perpendicular pi channels.
                if (bond->GetBondOrder() >= 3) return true;
            }
            const auto left = LocalPiPlaneTensor(donor, acceptor, false);
            const auto right = LocalPiPlaneTensor(acceptor, donor, false);
            /// Compare the local pi-plane tensors against the minimum overlap needed for conjugation.
            const auto compatible = [min_plane_overlap](
                const std::optional<PiPlaneTensor> &a,
                const std::optional<PiPlaneTensor> &b)
            {
                if (!a || !b) return true;
                double overlap = 0.0;
                for (std::size_t k = 0; k < 9; ++k)
                    overlap += (*a)[k] * (*b)[k];
                return overlap + 1e-12 >= min_plane_overlap;
            };
            if (!compatible(left, right)) return false;
            // Include the far endpoint of each localized pi bond; otherwise
            // a twisted iminium/carbanion contributor bypasses the N-pair guard.
            bool has_pi_partner = false;
            FOR_BONDS_OF_ATOM(bond, &acceptor)
            {
                if (bond->IsAromatic() || bond->GetBondOrder() != 2) continue;
                has_pi_partner = true;
                auto *partner = bond->GetNbrAtom(&acceptor);
                if (StronglyPyramidalPiEndpoint(*partner)) continue;
                const auto outer_plane = LocalPiPlaneTensor(*partner, acceptor);
                if (compatible(left, outer_plane) && compatible(right, outer_plane))
                    return true;
            }
            return !has_pi_partner;
        }
        if (donor.GetFormalCharge() != 0 ||
            (z != 7 && z != 8 && z != 15 && z != 16 && z != 33 && z != 34))
        {
            return true;
        }
        FOR_NBORS_OF_ATOM(neighbor, &donor)
        {
            if (neighbor->GetIdx() == acceptor.GetIdx())
            {
                continue;
            }
            FOR_BONDS_OF_ATOM(bond, &acceptor)
            {
                if (!IsMultipleLikeBond(*bond))
                {
                    continue;
                }
                const auto *pi_neighbor = bond->GetNbrAtom(&acceptor);
                if (pi_neighbor != nullptr &&
                    !(molgr::utils::CalculateShapeQuality(
                        neighbor->GetVector(), donor.GetVector(),
                        acceptor.GetVector(), pi_neighbor->GetVector()) <= volume_tolerance))
                {
                    return false;
                }
            }
        }
        return true;
    }

    /// Build conjugated components after applying local lone-pair and pi-plane geometry tests.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param config Aromatic stability weights and conjugation geometry thresholds.
    /// @return Geometry-validated conjugated topology for the supplied ligand graph.
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
                     config.conjugation_min_plane_overlap)))
            {
                continue;
            }
            if ((!begin_has_pi_bond &&
                 !LonePairIsGeometricallyConjugated(
                     *begin_atom, *end_atom,
                     config.conjugation_normalized_tetrahedron_volume_tolerance,
                     config.conjugation_min_plane_overlap)) ||
                (!end_has_pi_bond &&
                 !LonePairIsGeometricallyConjugated(
                     *end_atom, *begin_atom,
                     config.conjugation_normalized_tetrahedron_volume_tolerance,
                     config.conjugation_min_plane_overlap)))
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
            if (molgr::vendor::openbabel_threading::AtomIsAromatic(*begin_atom) ||
                molgr::vendor::openbabel_threading::AtomIsAromatic(*end_atom) ||
                atom_has_adjacent_alternating_single_bond.Contains(static_cast<int>(begin_atom->GetIdx())) ||
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
                auto component = component_atom_indices.values;
                std::sort(component.begin(), component.end());
                metrics.conjugated_atom_components.push_back(std::move(component));
            }
            std::sort(metrics.conjugated_atom_components.begin(), metrics.conjugated_atom_components.end());
            const auto [hyperconjugative_donor_count, hyperconjugation_score] =
                HyperconjugationMetrics(working);
            metrics.hyperconjugative_donor_count = hyperconjugative_donor_count;
            metrics.hyperconjugation_score = hyperconjugation_score;
            return metrics;
        }
    }
}
