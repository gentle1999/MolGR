#include "molgr/utils/resonance.h"

#include "molgr/stages/clean.h"
#include "molgr/stages/eliminate.h"
#include "molgr/state.h"
#include "molgr/utils/consts.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/smarts.h"
#include "molgr/utils/utils.h"
#include "molgr/vendor/openbabel_conversion.h"
#include "molgr/vendor/openbabel_copy.h"
#include "molgr/vendor/openbabel_threading.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include <openbabel/obfunctions.h>
#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace
{
    constexpr double kUffLiteBondStrainScoreWeight = 1.0;
    constexpr double kUffLiteAngleStrainScoreWeight = 0.35;
    constexpr double kUffLiteRadicalScoreWeight = 1.0;
    constexpr double kUffLiteConjugationScoreWeight = 0.25;
    constexpr double kUffLiteBranchBoundStepSlack = 25.0;
    constexpr double kKcalToKj = 4.184;
    constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;

    /// Check whether the new valence requires restoring an unresolved two-electron marker.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param post_target_valence Valence of the endpoint after the resonance move.
    /// @return True when a deferred two-electron label must be regenerated at the endpoint.
    bool ShouldRegenerateUnresolvedTarget(
        const OpenBabel::OBAtom &atom,
        int post_target_valence)
    {
        const int atomic_num = atom.GetAtomicNum();
        const int typical_valence = static_cast<int>(OpenBabel::GetTypicalValence(
            atomic_num,
            post_target_valence,
            atom.GetFormalCharge()));
        return molgr::utils::GetUnpairedElectronCount(atom) == 1 &&
               (atomic_num == 6 || atomic_num == 7 || atomic_num == 14 || atomic_num == 15) &&
               atom.GetFormalCharge() == 0 &&
               molgr::utils::GetLonePairCount(atom) == 0 &&
               std::max(0, typical_valence - post_target_valence) == 2;
    }

    using BondOrderOverrides = std::map<std::pair<int, int>, int>;

    /// @param atom_idx_a One-based atom index in the ligand graph.
    /// @param atom_idx_b One-based atom index in the ligand graph.
    /// @return The two atom indices in canonical order for undirected bond lookup.
    std::pair<int, int> BondPair(int atom_idx_a, int atom_idx_b)
    {
        if (atom_idx_a <= atom_idx_b)
        {
            return {atom_idx_a, atom_idx_b};
        }
        return {atom_idx_b, atom_idx_a};
    }

    /// @param bond Bond whose endpoints and order define the local environment.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return The proposed integer bond order, falling back to the molecular bond order.
    int GetBondOrderWithOverrides(
        const OpenBabel::OBBond &bond,
        const BondOrderOverrides &bond_order_overrides)
    {
        const int begin_idx = bond.GetBeginAtom()->GetIdx();
        const int end_idx = bond.GetEndAtom()->GetIdx();
        const auto it = bond_order_overrides.find(BondPair(begin_idx, end_idx));
        if (it != bond_order_overrides.end())
        {
            return it->second;
        }
        return bond.GetBondOrder();
    }

    /// @param bond Bond whose endpoints and order define the local environment.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return The effective fractional bond order used in the local strain estimate.
    double NormalizedBondOrderWithOverrides(
        const OpenBabel::OBBond &bond,
        const BondOrderOverrides &bond_order_overrides)
    {
        const int begin_idx = bond.GetBeginAtom()->GetIdx();
        const int end_idx = bond.GetEndAtom()->GetIdx();
        const auto it = bond_order_overrides.find(BondPair(begin_idx, end_idx));
        if (it != bond_order_overrides.end())
        {
            return static_cast<double>(std::max(it->second, 1));
        }
        if (molgr::vendor::openbabel_threading::BondIsAromatic(
                const_cast<OpenBabel::OBBond &>(bond)))
        {
            return 1.5;
        }
        if (const_cast<OpenBabel::OBBond &>(bond).IsAmide())
        {
            return 1.41;
        }
        return static_cast<double>(std::max(static_cast<int>(bond.GetBondOrder()), 1));
    }

    /// @param atomic_num Atomic number of the element.
    /// @return An electronegativity bounded away from zero for UFF-like length estimates.
    double ClampedElectronegativity(int atomic_num)
    {
        const double value = OpenBabel::OBElements::GetElectroNeg(atomic_num);
        return std::max(value, 0.25);
    }

    /// @param atomic_num Atomic number of the element.
    /// @return The covalent radius used in the distance estimate with invalid values guarded.
    double ClampedCovalentRadius(int atomic_num)
    {
        const double value = OpenBabel::OBElements::GetCovalentRad(atomic_num);
        return value > 0.0 ? value : 0.75;
    }

    /// Estimate the equilibrium bond length in angstroms from the elements and effective bond order.
    /// @param atomic_num_a Atomic number of the indicated bond partner or site.
    /// @param atomic_num_b Atomic number of the indicated bond partner or site.
    /// @param bond_order Bond multiplicity under consideration.
    /// @return Equilibrium interatomic distance in angstroms.
    double EstimateUffEquilibriumBondLength(
        int atomic_num_a,
        int atomic_num_b,
        double bond_order)
    {
        const double ri = ClampedCovalentRadius(atomic_num_a);
        const double rj = ClampedCovalentRadius(atomic_num_b);
        const double chi_i = ClampedElectronegativity(atomic_num_a);
        const double chi_j = ClampedElectronegativity(atomic_num_b);
        const double safe_bond_order = std::max(bond_order, 0.25);

        // UFF equations 2-4 use an order contraction term and an electronegativity
        // correction. OB covalent radii are not UFF radii, but the monotonic signal
        // is the important part for resonance move ordering.
        const double rbo = -0.1332 * (ri + rj) * std::log(safe_bond_order);
        const double ren =
            ri * rj * std::pow(std::sqrt(chi_i) - std::sqrt(chi_j), 2.0) /
            std::max(chi_i * ri + chi_j * rj, 1.0e-9);
        return std::max(ri + rj + rbo - ren, 0.4);
    }

    /// Estimate the harmonic stretching coefficient from the equilibrium bond length.
    /// @param equilibrium_distance Estimated unstretched bond length in angstroms.
    /// @return Harmonic stretching coefficient for the estimated equilibrium distance.
    double EstimateUffBondForceConstant(double equilibrium_distance)
    {
        // UFF folds the 1/2 into kb and scales approximately as 1/r0^3. We omit
        // element-specific effective charges to keep this proxy setup-free.
        return (0.5 * kKcalToKj * 664.12) /
               std::max(equilibrium_distance * equilibrium_distance * equilibrium_distance, 1.0e-6);
    }

    /// Estimate the harmonic strain of a bond under the proposed bond orders.
    /// @param bond Bond whose endpoints and order define the local environment.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return Estimated stretching energy under the proposed bond-order changes.
    double UffLiteBondStretchEnergy(
        const OpenBabel::OBBond &bond,
        const BondOrderOverrides &bond_order_overrides)
    {
        const auto *begin_atom = bond.GetBeginAtom();
        const auto *end_atom = bond.GetEndAtom();
        if (begin_atom == nullptr || end_atom == nullptr)
        {
            return 0.0;
        }
        const double bond_order = NormalizedBondOrderWithOverrides(bond, bond_order_overrides);
        const double r0 = EstimateUffEquilibriumBondLength(
            begin_atom->GetAtomicNum(),
            end_atom->GetAtomicNum(),
            bond_order);
        const double dx = begin_atom->GetX() - end_atom->GetX();
        const double dy = begin_atom->GetY() - end_atom->GetY();
        const double dz = begin_atom->GetZ() - end_atom->GetZ();
        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double delta = distance - r0;
        return EstimateUffBondForceConstant(r0) * delta * delta;
    }

    /// Sum the stretching strain over bonds affected by the local resonance shift.
    /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
    /// @param affected_bonds Bonds whose local stretching terms must be recomputed.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return Estimated stretching energy under the proposed bond-order changes.
    double LocalBondStretchEnergy(
        const OpenBabel::OBMol &mol,
        const std::set<std::pair<int, int>> &affected_bonds,
        const BondOrderOverrides &bond_order_overrides)
    {
        double energy = 0.0;
        auto &mutable_mol = const_cast<OpenBabel::OBMol &>(mol);
        for (const auto &bond_pair : affected_bonds)
        {
            OpenBabel::OBBond *bond = mutable_mol.GetBond(bond_pair.first, bond_pair.second);
            if (bond != nullptr)
            {
                energy += UffLiteBondStretchEnergy(*bond, bond_order_overrides);
            }
        }
        return energy;
    }

    /// Infer an ideal coordination angle in radians from the proposed local bond orders.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return Ideal bond angle in radians.
    double IdealAngleRadiansForAtom(
        const OpenBabel::OBAtom &atom,
        const BondOrderOverrides &bond_order_overrides)
    {
        int heavy_degree = 0;
        int multiple_bond_count = 0;
        double max_bond_order = 1.0;
        FOR_BONDS_OF_ATOM(bond_iter, const_cast<OpenBabel::OBAtom *>(&atom))
        {
            const OpenBabel::OBAtom *other =
                bond_iter->GetNbrAtom(const_cast<OpenBabel::OBAtom *>(&atom));
            if (other != nullptr && other->GetAtomicNum() != 1)
            {
                ++heavy_degree;
            }
            const double bond_order =
                NormalizedBondOrderWithOverrides(*bond_iter, bond_order_overrides);
            max_bond_order = std::max(max_bond_order, bond_order);
            if (bond_order >= 1.5)
            {
                ++multiple_bond_count;
            }
        }

        if (max_bond_order >= 2.5 || multiple_bond_count >= 2)
        {
            return 180.0 * kDegreesToRadians;
        }
        if (max_bond_order >= 1.5 || heavy_degree == 3)
        {
            return 120.0 * kDegreesToRadians;
        }
        return 109.5 * kDegreesToRadians;
    }

    /// Estimate bending strain for the specified three-atom angle under the proposed bond orders.
    /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
    /// @param left First atom.
    /// @param center Atom or site occupying this position in the local chemical motif.
    /// @param right Second atom.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return Estimated angle-bending energy under the proposed bond-order changes.
    double UffLiteAngleEnergy(
        const OpenBabel::OBMol &mol,
        const OpenBabel::OBAtom &left,
        const OpenBabel::OBAtom &center,
        const OpenBabel::OBAtom &right,
        const BondOrderOverrides &bond_order_overrides)
    {
        auto &mutable_mol = const_cast<OpenBabel::OBMol &>(mol);
        OpenBabel::OBBond *left_bond = mutable_mol.GetBond(left.GetIdx(), center.GetIdx());
        OpenBabel::OBBond *right_bond = mutable_mol.GetBond(center.GetIdx(), right.GetIdx());
        if (left_bond == nullptr || right_bond == nullptr)
        {
            return 0.0;
        }

        const double theta_degrees = mutable_mol.GetAngle(
            const_cast<OpenBabel::OBAtom *>(&left),
            const_cast<OpenBabel::OBAtom *>(&center),
            const_cast<OpenBabel::OBAtom *>(&right));
        if (!std::isfinite(theta_degrees))
        {
            return 0.0;
        }

        const double theta = theta_degrees * kDegreesToRadians;
        const double theta0 = IdealAngleRadiansForAtom(center, bond_order_overrides);
        const double left_order = NormalizedBondOrderWithOverrides(*left_bond, bond_order_overrides);
        const double right_order = NormalizedBondOrderWithOverrides(*right_bond, bond_order_overrides);
        const double left_r0 = EstimateUffEquilibriumBondLength(
            left.GetAtomicNum(),
            center.GetAtomicNum(),
            left_order);
        const double right_r0 = EstimateUffEquilibriumBondLength(
            center.GetAtomicNum(),
            right.GetAtomicNum(),
            right_order);
        const double stiffness =
            25.0 * (1.0 + 0.15 * std::max(0.0, left_order + right_order - 2.0)) /
            std::max(std::sqrt(left_r0 * right_r0), 0.5);
        const double cos_delta = std::cos(theta) - std::cos(theta0);
        return stiffness * cos_delta * cos_delta;
    }

    /// Sum local bending strain around the centers affected by a resonance move.
    /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
    /// @param affected_center_indices Centers whose local bending terms must be recomputed.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return Estimated angle-bending energy under the proposed bond-order changes.
    double LocalAngleStrainEnergy(
        const OpenBabel::OBMol &mol,
        const std::set<int> &affected_center_indices,
        const BondOrderOverrides &bond_order_overrides)
    {
        double energy = 0.0;
        auto &mutable_mol = const_cast<OpenBabel::OBMol &>(mol);
        for (int center_idx : affected_center_indices)
        {
            OpenBabel::OBAtom *center = mutable_mol.GetAtom(center_idx);
            if (center == nullptr)
            {
                continue;
            }

            std::vector<OpenBabel::OBAtom *> neighbors;
            FOR_NB_OF_ATOM(neighbor_iter, center)
            {
                neighbors.push_back(&(*neighbor_iter));
            }
            for (std::size_t i = 0; i < neighbors.size(); ++i)
            {
                for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                {
                    energy += UffLiteAngleEnergy(
                        mol,
                        *neighbors[i],
                        *center,
                        *neighbors[j],
                        bond_order_overrides);
                }
            }
        }
        return energy;
    }

    /// Estimate the penalty for radical localization from element, unpaired count and heavy-atom degree.
    /// @param atomic_num Atomic number of the element.
    /// @param radical_num Unpaired-electron assignment at the site.
    /// @param heavy_degree Number of nonhydrogen neighbors of the atom.
    /// @return Estimated radical localization penalty.
    double RadicalPenaltyForAtom(int atomic_num, int radical_num, int heavy_degree)
    {
        if (radical_num <= 0)
        {
            return 0.0;
        }
        if (molgr::kHeteroatoms.find(atomic_num) != molgr::kHeteroatoms.end())
        {
            return static_cast<double>(radical_num) * 10.0;
        }
        return static_cast<double>(radical_num) *
               std::max(0.0, 3.0 - static_cast<double>(heavy_degree)) * 1.5;
    }

    /// Classify a bond for traversing the alternating pi system under the proposed bond orders.
    /// @param bond Bond whose endpoints and order define the local environment.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return 0 for an absent bond order, 1 for a single bond and 2 for a multiple or aromatic bond.
    int ConjugatedBondKind(
        const OpenBabel::OBBond &bond,
        const BondOrderOverrides &bond_order_overrides)
    {
        if (molgr::vendor::openbabel_threading::BondIsAromatic(
                const_cast<OpenBabel::OBBond &>(bond)))
        {
            return 2;
        }
        const int bond_order = GetBondOrderWithOverrides(bond, bond_order_overrides);
        if (bond_order <= 0)
        {
            return 0;
        }
        if (bond_order == 1)
        {
            return 1;
        }
        return 2;
    }

    /// Count atoms reachable by conjugation from the proposed radical site.
    /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
    /// @param radical_atom_idx One-based atom index in the ligand graph.
    /// @param bond_order_overrides Proposed local bond orders keyed by sorted endpoint indices.
    /// @return Size of the conjugated region accessible from the radical site.
    int EstimateRadicalConjugationSize(
        const OpenBabel::OBMol &mol,
        int radical_atom_idx,
        const BondOrderOverrides &bond_order_overrides = {})
    {
        std::set<std::pair<int, int>> visited_states{{radical_atom_idx, 0}};
        std::set<int> visited_atoms{radical_atom_idx};
        std::vector<std::pair<int, int>> frontier{{radical_atom_idx, 0}};

        while (!frontier.empty())
        {
            const auto [atom_idx, previous_bond_kind] = frontier.back();
            frontier.pop_back();
            OpenBabel::OBAtom *atom = const_cast<OpenBabel::OBMol &>(mol).GetAtom(atom_idx);
            if (atom == nullptr)
            {
                continue;
            }

            FOR_BONDS_OF_ATOM(bond_iter, atom)
            {
                const int bond_kind = ConjugatedBondKind(*bond_iter, bond_order_overrides);
                if (bond_kind == 0)
                {
                    continue;
                }
                if (previous_bond_kind != 0 && bond_kind == previous_bond_kind)
                {
                    continue;
                }

                const int begin_idx = bond_iter->GetBeginAtom()->GetIdx();
                const int end_idx = bond_iter->GetEndAtom()->GetIdx();
                const int neighbor_idx = begin_idx == atom_idx ? end_idx : begin_idx;
                const std::pair<int, int> state{neighbor_idx, bond_kind};
                if (visited_states.find(state) != visited_states.end())
                {
                    continue;
                }
                visited_states.insert(state);
                visited_atoms.insert(neighbor_idx);
                frontier.push_back(state);
            }
        }

        return static_cast<int>(visited_atoms.size());
    }
    /// Estimate changes in local stretching, bending, radical and conjugation terms for one move.
    /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
    /// @param move_path Atom indices along the proposed electron-shift path.
    /// @return Local gain vector for the proposed move.
    molgr::resonance::UffLiteGainMetrics ComputeUffLiteGainResonanceMetrics(
        const OpenBabel::OBMol &mol,
        const std::tuple<int, int, int> &move_path)
    {
        const int old_radical_idx = std::get<0>(move_path);
        const int center_idx = std::get<1>(move_path);
        const int new_radical_idx = std::get<2>(move_path);

        OpenBabel::OBBond *bond_old_center =
            const_cast<OpenBabel::OBMol &>(mol).GetBond(old_radical_idx, center_idx);
        OpenBabel::OBBond *bond_center_new =
            const_cast<OpenBabel::OBMol &>(mol).GetBond(center_idx, new_radical_idx);
        OpenBabel::OBAtom *old_atom = const_cast<OpenBabel::OBMol &>(mol).GetAtom(old_radical_idx);
        OpenBabel::OBAtom *new_atom = const_cast<OpenBabel::OBMol &>(mol).GetAtom(new_radical_idx);
        if (bond_old_center == nullptr || bond_center_new == nullptr ||
            old_atom == nullptr || new_atom == nullptr)
        {
            return {0.0, 0.0, 0.0, 0.0};
        }

        const BondOrderOverrides bond_order_overrides = {
            {BondPair(old_radical_idx, center_idx), bond_old_center->GetBondOrder() + 1},
            {BondPair(center_idx, new_radical_idx), bond_center_new->GetBondOrder() - 1},
        };

        const std::set<std::pair<int, int>> affected_bonds{
            BondPair(old_radical_idx, center_idx),
            BondPair(center_idx, new_radical_idx),
        };
        const std::set<int> affected_angle_centers{
            old_radical_idx,
            center_idx,
            new_radical_idx,
        };

        const double bond_strain_gain =
            LocalBondStretchEnergy(mol, affected_bonds, {}) -
            LocalBondStretchEnergy(mol, affected_bonds, bond_order_overrides);
        const double angle_strain_gain =
            LocalAngleStrainEnergy(mol, affected_angle_centers, {}) -
            LocalAngleStrainEnergy(mol, affected_angle_centers, bond_order_overrides);
        const double radical_penalty_before =
            RadicalPenaltyForAtom(
                old_atom->GetAtomicNum(),
                molgr::utils::GetUnpairedElectronCount(*old_atom),
                old_atom->GetHvyDegree()) +
            RadicalPenaltyForAtom(
                new_atom->GetAtomicNum(),
                molgr::utils::GetUnpairedElectronCount(*new_atom),
                new_atom->GetHvyDegree());
        const double radical_penalty_after =
            RadicalPenaltyForAtom(
                old_atom->GetAtomicNum(),
                molgr::utils::GetUnpairedElectronCount(*old_atom) - 1,
                old_atom->GetHvyDegree()) +
            RadicalPenaltyForAtom(
                new_atom->GetAtomicNum(),
                molgr::utils::GetUnpairedElectronCount(*new_atom) + 1,
                new_atom->GetHvyDegree());
        const double radical_penalty_gain = radical_penalty_before - radical_penalty_after;
        const double conjugation_gain =
            static_cast<double>(
                EstimateRadicalConjugationSize(mol, new_radical_idx, bond_order_overrides) -
                EstimateRadicalConjugationSize(mol, old_radical_idx));
        return {
            bond_strain_gain,
            angle_strain_gain,
            radical_penalty_gain,
            conjugation_gain,
        };
    }

    /// @param metrics Local energy and electronic gain estimates for the move.
    /// @return True if any estimated score component improves.
    bool HasPositiveUffLiteGain(const molgr::resonance::UffLiteGainMetrics &metrics)
    {
        return std::any_of(metrics.begin(), metrics.end(), [](double value) { return value > 0.0; });
    }

    /// Add the two vectors of local resonance gains component by component.
    /// @param left First vector of local resonance gains.
    /// @param right Second vector of local resonance gains.
    /// @return Componentwise gain estimates for the indicated sum, maximum or bounded search.
    molgr::resonance::UffLiteGainMetrics AddUffLiteGainMetrics(
        const molgr::resonance::UffLiteGainMetrics &left,
        const molgr::resonance::UffLiteGainMetrics &right)
    {
        molgr::resonance::UffLiteGainMetrics result{};
        for (std::size_t idx = 0; idx < result.size(); ++idx)
        {
            result[idx] = left[idx] + right[idx];
        }
        return result;
    }

    /// Take the larger estimate in each resonance-gain component.
    /// @param left First vector of local resonance gains.
    /// @param right Second vector of local resonance gains.
    /// @return Componentwise gain estimates for the indicated sum, maximum or bounded search.
    molgr::resonance::UffLiteGainMetrics ComponentwiseMaxUffLiteGainMetrics(
        const molgr::resonance::UffLiteGainMetrics &left,
        const molgr::resonance::UffLiteGainMetrics &right)
    {
        molgr::resonance::UffLiteGainMetrics result{};
        for (std::size_t idx = 0; idx < result.size(); ++idx)
        {
            result[idx] = std::max(left[idx], right[idx]);
        }
        return result;
    }

    /// Bound attainable gains by exploring the remaining steps with memoized molecular states.
    /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
    /// @param state_key Fingerprint of the current resonance state.
    /// @param remaining_steps Remaining depth budget in one-step electron shifts.
    /// @param cache Stored evaluations or bounds indexed by the molecular fingerprint.
    /// @return Componentwise gain estimates for the indicated sum, maximum or bounded search.
    molgr::resonance::UffLiteGainMetrics ComputeNStepUffLiteGainUpperBound(
        const OpenBabel::OBMol &mol,
        const molgr::resonance::ResonanceStateKey &state_key,
        int remaining_steps,
        molgr::resonance::UffLiteGainBoundCache &cache)
    {
        if (remaining_steps <= 0)
        {
            return {0.0, 0.0, 0.0, 0.0};
        }

        const molgr::resonance::UffLiteGainBoundCacheKey cache_key{state_key, remaining_steps};
        const auto cache_it = cache.find(cache_key);
        if (cache_it != cache.end())
        {
            return cache_it->second;
        }

        molgr::resonance::UffLiteGainMetrics upper_bound{0.0, 0.0, 0.0, 0.0};
        const auto bond_index_map = molgr::resonance::BuildBondIndexMapFromStateKey(state_key);
        const auto moves =
            molgr::resonance::EnumerateOneStepResonanceMoves(mol, state_key, bond_index_map);
        for (const auto &move : moves)
        {
            const auto uff_lite_metrics = ComputeUffLiteGainResonanceMetrics(mol, move.idxs);
            if (!HasPositiveUffLiteGain(uff_lite_metrics))
            {
                continue;
            }

            auto total_metrics = uff_lite_metrics;
            if (remaining_steps > 1)
            {
                auto next_omol = molgr::resonance::MaterializeOneStepResonance(mol, move.idxs);
                const auto future_upper_bound = ComputeNStepUffLiteGainUpperBound(
                    next_omol,
                    move.next_state_key,
                    remaining_steps - 1,
                    cache);
                total_metrics = AddUffLiteGainMetrics(uff_lite_metrics, future_upper_bound);
            }

            upper_bound = ComponentwiseMaxUffLiteGainMetrics(upper_bound, total_metrics);
        }

        cache[cache_key] = upper_bound;
        return upper_bound;
    }

    /// Convert componentwise gain bounds into a score improvement bound for pruning.
    /// @param metrics Local energy and electronic gain estimates for the move.
    /// @param remaining_steps Remaining depth budget in one-step electron shifts.
    /// @return Upper bound on the score improvement over the remaining steps.
    double EstimateUffLiteGainScoreImprovementUpperBound(
        const molgr::resonance::UffLiteGainMetrics &metrics,
        int remaining_steps)
    {
        return std::max(
            0.0,
            metrics[0] * kUffLiteBondStrainScoreWeight +
                metrics[1] * kUffLiteAngleStrainScoreWeight +
                metrics[2] * kUffLiteRadicalScoreWeight +
                metrics[3] * kUffLiteConjugationScoreWeight +
                static_cast<double>(std::max(remaining_steps, 0)) * kUffLiteBranchBoundStepSlack);
    }

    /// Combine the gain estimates into the ordering score for one resonance move.
    /// @param metrics Local energy and electronic gain estimates for the move.
    /// @return Numeric move-ordering score.
    double UffLiteGainMoveScore(const molgr::resonance::UffLiteGainMetrics &metrics)
    {
        return -(
            metrics[0] * kUffLiteBondStrainScoreWeight +
            metrics[1] * kUffLiteAngleStrainScoreWeight +
            metrics[2] * kUffLiteRadicalScoreWeight +
            metrics[3] * kUffLiteConjugationScoreWeight);
    }

    /// Encode the resonance fingerprint for use as a string key.
    /// @param state_key Fingerprint of the current resonance state.
    /// @return Serialized atom and bond fingerprint.
    std::string SerializeResonanceStateKey(
        const molgr::resonance::ResonanceStateKey &state_key)
    {
        std::string key;
        key.reserve(
            32 +
            state_key.atom_keys.size() * 16 +
            state_key.bond_keys.size() * 18);

        key += "A";
        key += std::to_string(state_key.atom_keys.size());
        key.push_back(':');
        for (const auto &atom_key : state_key.atom_keys)
        {
            key += std::to_string(std::get<0>(atom_key));
            key.push_back(',');
            key += std::to_string(std::get<1>(atom_key));
            key.push_back(',');
            key += std::to_string(std::get<2>(atom_key));
            key.push_back(',');
            key += std::to_string(std::get<3>(atom_key));
            key.push_back(',');
            key.push_back(std::get<4>(atom_key) ? '1' : '0');
            key.push_back(',');
            key.push_back(std::get<5>(atom_key) ? '1' : '0');
            key.push_back(';');
        }

        key += "|B";
        key += std::to_string(state_key.bond_keys.size());
        key.push_back(':');
        for (const auto &bond_key : state_key.bond_keys)
        {
            key += std::to_string(std::get<0>(bond_key));
            key.push_back(',');
            key += std::to_string(std::get<1>(bond_key));
            key.push_back(',');
            key += std::to_string(std::get<2>(bond_key));
            key.push_back(',');
            key.push_back(std::get<3>(bond_key) ? '1' : '0');
            key.push_back(';');
        }
        return key;
    }

}

namespace molgr
{
    namespace resonance
    {
        bool ResonanceStateKey::operator==(const ResonanceStateKey &other) const
        {
            return atom_keys == other.atom_keys && bond_keys == other.bond_keys;
        }

        bool ResonanceStateKey::operator<(const ResonanceStateKey &other) const
        {
            return std::tie(atom_keys, bond_keys) < std::tie(other.atom_keys, other.bond_keys);
        }

        std::string SmilesFirstToken(const OpenBabel::OBMol &mol)
        {
            return molgr::vendor::openbabel_conversion::WriteSmilesFirstToken(mol);
        }

        ResonanceStateKey BuildResonanceStateKey(const OpenBabel::OBMol &mol)
        {
            ResonanceStateKey state_key;
            OpenBabel::OBMol &mutable_mol = const_cast<OpenBabel::OBMol &>(mol);

            state_key.atom_keys.reserve(static_cast<std::size_t>(mutable_mol.NumAtoms()));
            FOR_ATOMS_OF_MOL(atom_iter, mutable_mol)
            {
                const OpenBabel::OBAtom &atom = *atom_iter;
                state_key.atom_keys.emplace_back(
                    atom.GetAtomicNum(),
                    atom.GetFormalCharge(),
                    molgr::utils::GetUnpairedElectronCount(atom),
                    molgr::utils::GetLonePairCount(atom),
                    molgr::utils::HasUnresolvedTwoElectronCenter(atom),
                    molgr::vendor::openbabel_threading::AtomIsAromatic(
                        const_cast<OpenBabel::OBAtom &>(atom)));
            }

            state_key.bond_keys.reserve(static_cast<std::size_t>(mutable_mol.NumBonds()));
            FOR_BONDS_OF_MOL(bond_iter, mutable_mol)
            {
                int begin_idx = bond_iter->GetBeginAtom()->GetIdx();
                int end_idx = bond_iter->GetEndAtom()->GetIdx();
                if (begin_idx > end_idx)
                {
                    std::swap(begin_idx, end_idx);
                }
                state_key.bond_keys.emplace_back(
                    begin_idx,
                    end_idx,
                    bond_iter->GetBondOrder(),
                    molgr::vendor::openbabel_threading::BondIsAromatic(*bond_iter));
            }
            return state_key;
        }

        ProcessedResonanceKey BuildProcessedResonanceKey(const OpenBabel::OBMol &mol)
        {
            return SerializeResonanceStateKey(BuildResonanceStateKey(mol));
        }

        std::pair<ResonanceStateKey, ResonanceBondIndexMap> BuildResonanceSearchContext(
            const OpenBabel::OBMol &mol)
        {
            const auto state_key = BuildResonanceStateKey(mol);
            return {state_key, BuildBondIndexMapFromStateKey(state_key)};
        }

        ResonanceBondIndexMap BuildBondIndexMapFromStateKey(
            const ResonanceStateKey &state_key)
        {
            ResonanceBondIndexMap bond_index_map;
            for (std::size_t idx = 0; idx < state_key.bond_keys.size(); ++idx)
            {
                const auto &bond_key = state_key.bond_keys[idx];
                bond_index_map[{
                    std::get<0>(bond_key),
                    std::get<1>(bond_key),
                }] = idx;
            }
            return bond_index_map;
        }

        ResonanceStateKey IncrementResonanceStateKey(
            const ResonanceStateKey &state_key,
            const ResonanceBondIndexMap &bond_index_map,
            const std::tuple<int, int, int> &idxs,
            bool target_unresolved)
        {
            auto next_state_key = state_key;

            const std::size_t atom1_idx = static_cast<std::size_t>(std::get<0>(idxs) - 1);
            const std::size_t atom3_idx = static_cast<std::size_t>(std::get<2>(idxs) - 1);

            auto atom1 = next_state_key.atom_keys[atom1_idx];
            auto atom3 = next_state_key.atom_keys[atom3_idx];
            std::get<2>(atom1) -= 1;
            if (target_unresolved)
            {
                std::get<2>(atom3) = 0;
                std::get<4>(atom3) = true;
            }
            else
            {
                std::get<2>(atom3) += 1;
            }
            next_state_key.atom_keys[atom1_idx] = atom1;
            next_state_key.atom_keys[atom3_idx] = atom3;

            const auto bond1_it =
                bond_index_map.find(BondPair(std::get<0>(idxs), std::get<1>(idxs)));
            const auto bond2_it =
                bond_index_map.find(BondPair(std::get<1>(idxs), std::get<2>(idxs)));
            if (bond1_it == bond_index_map.end() || bond2_it == bond_index_map.end())
            {
                return next_state_key;
            }

            auto bond1 = next_state_key.bond_keys[bond1_it->second];
            auto bond2 = next_state_key.bond_keys[bond2_it->second];
            std::get<2>(bond1) += 1;
            std::get<2>(bond2) -= 1;
            next_state_key.bond_keys[bond1_it->second] = bond1;
            next_state_key.bond_keys[bond2_it->second] = bond2;
            return next_state_key;
        }

        // Electron bookkeeping: enumerate migration of exactly one real
        // monoradical across an adjacent pi bond. When a monoradical target becomes
        // a neutral C/N/Si/P two-electron deficit after losing its pi bond, defer its
        // resulting occupancy instead of forcing a triplet state.
        std::vector<IndexedResonanceTraversalMove> EnumerateOneStepResonanceMoves(
            const OpenBabel::OBMol &mol,
            const ResonanceStateKey &state_key,
            const ResonanceBondIndexMap &bond_index_map)
        {
            OpenBabel::OBMol &query_mol = const_cast<OpenBabel::OBMol &>(mol);
            // Exactly one unpaired electron is required at the source of a move.
            bool has_source = false;
            FOR_ATOMS_OF_MOL(atom, query_mol)
            {
                if (molgr::utils::GetUnpairedElectronCount(*atom) == 1)
                {
                    has_source = true;
                    break;
                }
            }
            if (!has_source)
            {
                molgr::vendor::openbabel_threading::PrepareForSmartsMatching(query_mol);
                return {};
            }
            const auto matches =
                molgr::smarts::FindAll(query_mol, molgr::smarts::PatternId::RESONANCE_ONE_STEP);
            std::vector<IndexedResonanceTraversalMove> result;
            result.reserve(matches.size());

            for (const auto &idxs_vec : matches)
            {
                if (idxs_vec.size() < 3)
                {
                    continue;
                }

                const std::tuple<int, int, int> idxs{idxs_vec[0], idxs_vec[1], idxs_vec[2]};
                OpenBabel::OBAtom *atom1 = query_mol.GetAtom(std::get<0>(idxs));
                OpenBabel::OBAtom *atom3 = query_mol.GetAtom(std::get<2>(idxs));
                OpenBabel::OBBond *bond1 = query_mol.GetBond(std::get<0>(idxs), std::get<1>(idxs));
                OpenBabel::OBBond *bond2 = query_mol.GetBond(std::get<1>(idxs), std::get<2>(idxs));
                if (atom1 == nullptr || atom3 == nullptr || bond1 == nullptr || bond2 == nullptr)
                {
                    continue;
                }

                if (molgr::utils::GetUnpairedElectronCount(*atom1) == 1 &&
                    molgr::utils::GetUnpairedElectronCount(*atom3) <= 1 &&
                    !molgr::utils::HasUnresolvedTwoElectronCenter(*atom3) &&
                    bond1->GetBondOrder() <= 2 &&
                    bond2->GetBondOrder() >= 2)
                {
                    const int post_target_valence =
                        static_cast<int>(atom3->GetTotalValence()) - 1;
                    const bool target_unresolved =
                        ShouldRegenerateUnresolvedTarget(*atom3, post_target_valence);
                    result.push_back(
                        IndexedResonanceTraversalMove{
                            idxs,
                            IncrementResonanceStateKey(
                                state_key,
                                bond_index_map,
                                idxs,
                                target_unresolved),
                        });
                }
            }

            return result;
        }

        // Electron bookkeeping: move one real unpaired electron from the source
        // to the target while swapping the coupled bond-order pattern by one.
        // A newly formed neutral C/N/Si/P two-electron deficit is regenerated as an
        // unresolved center so validation can choose singlet versus triplet.
        OpenBabel::OBMol MaterializeOneStepResonance(
            const OpenBabel::OBMol &mol,
            const std::tuple<int, int, int> &idxs)
        {
            OpenBabel::OBMol new_mol = molgr::utils::CloneMolTopologyOnly(mol);
            OpenBabel::OBAtom *atom1 = new_mol.GetAtom(std::get<0>(idxs));
            OpenBabel::OBAtom *atom3 = new_mol.GetAtom(std::get<2>(idxs));
            OpenBabel::OBBond *bond1 = new_mol.GetBond(std::get<0>(idxs), std::get<1>(idxs));
            OpenBabel::OBBond *bond2 = new_mol.GetBond(std::get<1>(idxs), std::get<2>(idxs));
            if (atom1 == nullptr || atom3 == nullptr || bond1 == nullptr || bond2 == nullptr)
            {
                return new_mol;
            }

            bond1->SetBondOrder(bond1->GetBondOrder() + 1);
            bond2->SetBondOrder(bond2->GetBondOrder() - 1);
            molgr::utils::SetUnpairedElectronCount(*atom1, molgr::utils::GetUnpairedElectronCount(*atom1) - 1);
            const int target_unpaired = molgr::utils::GetUnpairedElectronCount(*atom3);
            const int target_valence = static_cast<int>(atom3->GetTotalValence());
            const bool target_unresolved =
                ShouldRegenerateUnresolvedTarget(*atom3, target_valence);
            if (target_unresolved)
            {
                molgr::utils::SetUnpairedElectronCount(*atom3, 0);
                molgr::utils::SetUnresolvedTwoElectronCenter(*atom3, true);
            }
            else
            {
                molgr::utils::SetUnpairedElectronCount(*atom3, target_unpaired + 1);
            }
            return new_mol;
        }

        std::shared_ptr<OpenBabel::OBMol> MaterializeOneStepResonancePtr(
            const OpenBabel::OBMol &mol,
            const std::tuple<int, int, int> &idxs)
        {
            auto new_mol = molgr::utils::CloneMolTopologyShared(mol);
            OpenBabel::OBAtom *atom1 = new_mol->GetAtom(std::get<0>(idxs));
            OpenBabel::OBAtom *atom3 = new_mol->GetAtom(std::get<2>(idxs));
            OpenBabel::OBBond *bond1 = new_mol->GetBond(std::get<0>(idxs), std::get<1>(idxs));
            OpenBabel::OBBond *bond2 = new_mol->GetBond(std::get<1>(idxs), std::get<2>(idxs));
            if (atom1 == nullptr || atom3 == nullptr || bond1 == nullptr || bond2 == nullptr)
            {
                return new_mol;
            }

            bond1->SetBondOrder(bond1->GetBondOrder() + 1);
            bond2->SetBondOrder(bond2->GetBondOrder() - 1);
            molgr::utils::SetUnpairedElectronCount(*atom1, molgr::utils::GetUnpairedElectronCount(*atom1) - 1);
            const int target_unpaired = molgr::utils::GetUnpairedElectronCount(*atom3);
            const int target_valence = static_cast<int>(atom3->GetTotalValence());
            const bool target_unresolved =
                ShouldRegenerateUnresolvedTarget(*atom3, target_valence);
            if (target_unresolved)
            {
                molgr::utils::SetUnpairedElectronCount(*atom3, 0);
                molgr::utils::SetUnresolvedTwoElectronCenter(*atom3, true);
            }
            else
            {
                molgr::utils::SetUnpairedElectronCount(*atom3, target_unpaired + 1);
            }
            return new_mol;
        }

        std::vector<OpenBabel::OBMol> GetOneStepResonance(const OpenBabel::OBMol &mol)
        {
            const auto [state_key, bond_index_map] = BuildResonanceSearchContext(mol);
            const auto moves = EnumerateOneStepResonanceMoves(mol, state_key, bond_index_map);
            std::vector<OpenBabel::OBMol> result;
            result.reserve(moves.size());
            for (const auto &move : moves)
            {
                const auto resonance = MaterializeOneStepResonance(mol, move.idxs);
                std::lock_guard<std::recursive_mutex> lock(
                    molgr::vendor::openbabel_threading::NativeOpenBabelMutex());
                result.push_back(resonance);
            }
            return result;
        }

        /// Rank one-step moves by their estimated local gains and deterministic atom-index tie-breakers.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param moves Candidate one-step electron shifts to select or order.
        /// @return Moves paired with their scores in traversal order.
        std::vector<std::pair<IndexedResonanceTraversalMove, double>> OrderUffLiteGainMoves(
            const OpenBabel::OBMol &mol,
            const std::vector<IndexedResonanceTraversalMove> &moves)
        {
            std::vector<std::pair<IndexedResonanceTraversalMove, double>> ordered_moves;
            ordered_moves.reserve(moves.size());
            for (const auto &move : moves)
            {
                ordered_moves.emplace_back(
                    move,
                    UffLiteGainMoveScore(ComputeUffLiteGainResonanceMetrics(mol, move.idxs)));
            }
            std::sort(
                ordered_moves.begin(),
                ordered_moves.end(),
                [](const auto &lhs, const auto &rhs)
                {
                    return std::tie(lhs.second, lhs.first.idxs) <
                           std::tie(rhs.second, rhs.first.idxs);
                });
            return ordered_moves;
        }

        std::vector<IndexedResonanceTraversalMove> SelectLimitedDiscrepancyMoves(
            const OpenBabel::OBMol &mol,
            const std::vector<IndexedResonanceTraversalMove> &moves,
            const LimitedDiscrepancyTraversalConfig &traversal_config,
            const molgr::config::MolGRConfig &config)
        {
            (void)traversal_config;
            if (config.resonance.traversal_score == "input_order")
            {
                return moves;
            }

            std::vector<std::pair<IndexedResonanceTraversalMove, double>> ordered_moves;
            if (config.resonance.traversal_score == "uff_lite_gain")
            {
                if (moves.size() < 2) return moves;
                ordered_moves = OrderUffLiteGainMoves(mol, moves);
            }
            else
            {
                throw std::invalid_argument(
                    "Unsupported resonance traversal_score. Expected 'uff_lite_gain' or 'input_order'.");
            }

            std::vector<IndexedResonanceTraversalMove> selected_moves;
            selected_moves.reserve(ordered_moves.size());
            for (const auto &entry : ordered_moves)
            {
                selected_moves.push_back(entry.first);
            }
            return selected_moves;
        }

        double EstimateRemainingResonanceScoreImprovementUpperBound(
            const OpenBabel::OBMol &mol,
            const ResonanceStateKey &state_key,
            int remaining_steps,
            UffLiteGainBoundCache *cache)
        {
            if (remaining_steps <= 0)
            {
                return 0.0;
            }

            UffLiteGainBoundCache local_cache;
            UffLiteGainBoundCache &cache_ref = cache == nullptr ? local_cache : *cache;
            const auto optimistic_metrics =
                ComputeNStepUffLiteGainUpperBound(mol, state_key, remaining_steps, cache_ref);
            return EstimateUffLiteGainScoreImprovementUpperBound(
                optimistic_metrics,
                remaining_steps);
        }

        // Electron bookkeeping: normalize classified radical/active-lone-pair
        // states. Deferred 2e centers normally survive until validation, but a
        // remaining charge deficit of magnitude >=2 may consume one directly as
        // +/-2. No stage derives radical occupancy from marker parity.
        std::tuple<OpenBabel::OBMol, int, bool> ProcessResonanceDetailed(
            const OpenBabel::OBMol &mol,
            int charge,
            int total_charge,
            int total_radical_electrons)
        {
            auto machine = molgr::state::OmolStateMachine(
                molgr::utils::CloneMolTopologyShared(mol),
                charge,
                {},
                {},
                0,
                total_charge,
                total_radical_electrons);
            bool hit = false;
            hit = machine.RunOmolChargeStage(
                      std::nullopt,
                      molgr::reconstruct::Eliminate13DipolePostive) ||
                  hit;
            hit = machine.RunOmolChargeStage(
                      std::nullopt,
                      molgr::reconstruct::EliminatePossibleCPLikeRadicalAnion,
                      machine.total_radical_electrons, std::vector<int>{}) ||
                  hit;
            hit = machine.RunOmolStage(
                      std::nullopt,
                      molgr::reconstruct::CleanPossible13Dipole,
                      machine.given_charge,
                      machine.total_radical_electrons) ||
                  hit;
            hit = machine.RunOmolStage(
                      std::nullopt,
                      molgr::reconstruct::Clean14Radicals,
                      machine.given_charge,
                      machine.total_radical_electrons) ||
                  hit;
            hit = machine.RunOmolStage(
                      std::nullopt,
                      molgr::reconstruct::CleanNeighborRadicals,
                      machine.given_charge,
                      machine.total_radical_electrons) ||
                  hit;
            hit = machine.RunOmolStage(
                      std::nullopt,
                      molgr::reconstruct::Clean16Radicals,
                      machine.given_charge,
                      machine.total_radical_electrons) ||
                  hit;
            hit = machine.RunOmolChargeStage(
                      std::nullopt,
                      [total_radical_electrons](OpenBabel::OBMol &mol, int &charge)
                      {
                          return molgr::reconstruct::EliminatePositiveChargesWithTarget(
                              mol, charge, total_radical_electrons);
                      }) ||
                  hit;
            hit = machine.RunOmolChargeStage(
                      std::nullopt,
                      [total_radical_electrons](OpenBabel::OBMol &mol, int &charge)
                      {
                          return molgr::reconstruct::EliminateNegativeCharges(
                              mol,
                              charge,
                              total_radical_electrons);
                      }) ||
                  hit;
            hit = machine.RunOmolChargeStage(
                      std::nullopt,
                      [total_radical_electrons](OpenBabel::OBMol &mol, int &charge)
                      {
                          return molgr::reconstruct::EliminatePositiveChargesWithTarget(
                              mol, charge, total_radical_electrons);
                      }) ||
                  hit;
            hit = machine.RunOmolStage(std::nullopt, molgr::reconstruct::CleanResonances) || hit;
            const auto result = molgr::utils::CloneMolTopologyOnly(*machine.omol);
            std::lock_guard<std::recursive_mutex> lock(
                molgr::vendor::openbabel_threading::NativeOpenBabelMutex());
            return std::make_tuple(
                result,
                machine.given_charge,
                hit);
        }

        std::pair<OpenBabel::OBMol, int> ProcessResonance(
            const OpenBabel::OBMol &mol,
            int charge,
            int total_charge,
            int total_radical_electrons)
        {
            auto processed = ProcessResonanceDetailed(
                mol,
                charge,
                total_charge,
                total_radical_electrons);
            std::lock_guard<std::recursive_mutex> lock(
                molgr::vendor::openbabel_threading::NativeOpenBabelMutex());
            return {std::get<0>(processed), std::get<1>(processed)};
        }
    }
}
