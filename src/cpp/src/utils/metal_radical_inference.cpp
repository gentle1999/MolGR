#include "molgr/utils/metal_radical_inference.h"

#include "molgr/utils/consts.h"
#include "molgr/utils/coordination.h"

#include <openbabel/elements.h>
#include <openbabel/mol.h>
#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    using Vector3 = std::tuple<double, double, double>;

    /// Remaining metal-shell occupations after formal oxidation.
    struct ShellOccupation
    {
        int remaining_f = 0;
        int effective_d = 0;
        int residual_sp = 0;
    };

    /// A nearby coordinating atom with its donor identity, displacement and metal distance.
    struct DonorSample
    {
        int atom_idx = 0;
        int atomic_num = 0;
        double distance_angstrom = 0.0;
        Vector3 vector;
    };

    const std::set<std::string> kDBlockRearrangementMetals = {
        "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y",  "Zr", "Nb", "Mo",
        "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
        "Hg"};

    const std::map<int, double> kDonorFieldStrength = {
        {1, 1.20},  {2, 0.00},  {5, 1.30},  {6, 1.30},  {7, 1.00},  {8, 0.85},
        {9, 0.65},  {10, 0.00}, {14, 1.30}, {15, 1.50}, {16, 0.75}, {17, 0.55},
        {18, 0.00}, {32, 1.30}, {33, 1.40}, {34, 1.50}, {35, 0.50}, {36, 0.00},
        {51, 1.35}, {52, 1.45}, {53, 0.45}, {54, 0.00}, {84, 1.40}, {85, 0.40},
        {86, 0.00},
    };

    const std::map<std::string, double> kGeometryFieldAdjustment = {
        {"free_ion", 0.0},
        {"terminal", 0.0},
        {"bent", 0.0},
        {"linear", 0.20},
        {"trigonal_planar", 0.15},
        {"trigonal_pyramidal", -0.05},
        {"tetrahedral", -0.25},
        {"square_planar", 0.35},
        {"octahedral_like", 0.05},
    };

    constexpr double kMetalHydrideCovalentToleranceAngstrom = 0.45;
    constexpr double kMinMetalHydrideCutoffAngstrom = 1.25;

    /// @param metal_atom Open Babel atom representing the metal center.
    /// @param donor_atom Candidate ligand atom at the donor end of the metal contact.
    /// @return The Cartesian displacement from the metal to its donor, in angstroms.
    Vector3 VectorFromAtoms(OpenBabel::OBAtom &metal_atom, OpenBabel::OBAtom &donor_atom)
    {
        return {
            donor_atom.GetX() - metal_atom.GetX(),
            donor_atom.GetY() - metal_atom.GetY(),
            donor_atom.GetZ() - metal_atom.GetZ(),
        };
    }

    /// @param vector Cartesian displacement vector.
    /// @return The Euclidean length of a Cartesian vector.
    double VectorNorm(const Vector3 &vector)
    {
        return std::sqrt(
            std::get<0>(vector) * std::get<0>(vector) +
            std::get<1>(vector) * std::get<1>(vector) +
            std::get<2>(vector) * std::get<2>(vector));
    }

    /// @param atomic_num Atomic number of the element.
    /// @return The covalent radius used in the distance estimate with invalid values guarded.
    double ClampedCovalentRadius(int atomic_num)
    {
        const double value = OpenBabel::OBElements::GetCovalentRad(atomic_num);
        if (value <= 0.0 || !std::isfinite(value))
        {
            return 0.0;
        }
        return value;
    }

    /// @param metal_atomic_num Atomic number of the indicated bond partner or site.
    /// @return The element-dependent upper distance for a directly bound hydride, in angstroms.
    double MetalHydrideCutoffAngstrom(int metal_atomic_num)
    {
        return std::max(
            kMinMetalHydrideCutoffAngstrom,
            ClampedCovalentRadius(metal_atomic_num) + ClampedCovalentRadius(1) +
                kMetalHydrideCovalentToleranceAngstrom);
    }

    /// @param metal_atom Open Babel atom representing the metal center.
    /// @param donor_atom Candidate ligand atom at the donor end of the metal contact.
    /// @param distance Measured donor-metal distance in angstroms.
    /// @return True for a hydrogen donor within the metal-specific hydride distance cutoff.
    bool IsDirectMetalHydride(
        OpenBabel::OBAtom &metal_atom,
        OpenBabel::OBAtom &donor_atom,
        double distance)
    {
        return static_cast<int>(donor_atom.GetAtomicNum()) == 1 &&
               distance <= MetalHydrideCutoffAngstrom(static_cast<int>(metal_atom.GetAtomicNum()));
    }

    /// @param lhs First Cartesian vector.
    /// @param rhs Second Cartesian vector.
    /// @return The cross product of the two Cartesian vectors.
    Vector3 Cross(const Vector3 &lhs, const Vector3 &rhs)
    {
        return {
            std::get<1>(lhs) * std::get<2>(rhs) - std::get<2>(lhs) * std::get<1>(rhs),
            std::get<2>(lhs) * std::get<0>(rhs) - std::get<0>(lhs) * std::get<2>(rhs),
            std::get<0>(lhs) * std::get<1>(rhs) - std::get<1>(lhs) * std::get<0>(rhs),
        };
    }

    /// @param lhs First Cartesian vector.
    /// @param rhs Second Cartesian vector.
    /// @return The dot product of the two Cartesian vectors.
    double Dot(const Vector3 &lhs, const Vector3 &rhs)
    {
        return std::get<0>(lhs) * std::get<0>(rhs) +
               std::get<1>(lhs) * std::get<1>(rhs) +
               std::get<2>(lhs) * std::get<2>(rhs);
    }

    /// @param lhs First Cartesian vector.
    /// @param rhs Second Cartesian vector.
    /// @return The angle between two donor directions in degrees.
    double AngleDegrees(const Vector3 &lhs, const Vector3 &rhs)
    {
        constexpr double kPi = 3.14159265358979323846;
        const double lhs_norm = VectorNorm(lhs);
        const double rhs_norm = VectorNorm(rhs);
        if (lhs_norm <= 1e-8 || rhs_norm <= 1e-8)
        {
            return 0.0;
        }
        double cos_value = Dot(lhs, rhs) / (lhs_norm * rhs_norm);
        cos_value = std::max(-1.0, std::min(1.0, cos_value));
        return std::acos(cos_value) * 180.0 / kPi;
    }

    /// @param vectors Displacements from the metal center to its donors, in angstroms.
    /// @return The mean distance of the donor displacements from the estimated coordination plane.
    double PlanarityDistance(const std::vector<Vector3> &vectors)
    {
        if (vectors.size() < 3)
        {
            return std::numeric_limits<double>::infinity();
        }

        std::optional<Vector3> best_normal;
        double best_norm = 0.0;
        for (std::size_t i = 0; i < vectors.size(); ++i)
        {
            for (std::size_t j = i + 1; j < vectors.size(); ++j)
            {
                const Vector3 normal = Cross(vectors[i], vectors[j]);
                const double normal_norm = VectorNorm(normal);
                if (normal_norm > best_norm)
                {
                    best_normal = normal;
                    best_norm = normal_norm;
                }
            }
        }
        if (!best_normal.has_value() || best_norm <= 1e-8)
        {
            return std::numeric_limits<double>::infinity();
        }

        const Vector3 normal = {
            std::get<0>(*best_normal) / best_norm,
            std::get<1>(*best_normal) / best_norm,
            std::get<2>(*best_normal) / best_norm,
        };
        double total = 0.0;
        for (const auto &vector : vectors)
        {
            total += std::abs(Dot(vector, normal));
        }
        return total / static_cast<double>(vectors.size());
    }

    /// Collect nearby donor atoms and order their measurements by distance to the metal.
    /// @param metal_atom Open Babel atom representing the metal center.
    /// @param config Settings for the current reconstruction or scoring operation.
    /// @return Donor measurements ordered by increasing metal distance.
    std::vector<DonorSample> CollectCoordinationEnvironment(
        OpenBabel::OBAtom &metal_atom,
        const molgr::config::MolGRConfig &config)
    {
        OpenBabel::OBMol *parent = metal_atom.GetParent();
        if (parent == nullptr)
        {
            return {};
        }

        std::vector<DonorSample> donors;
        FOR_ATOMS_OF_MOL(neighbor_iter, *parent)
        {
            OpenBabel::OBAtom &neighbor = *neighbor_iter;
            if (neighbor.GetIdx() == metal_atom.GetIdx() || neighbor.IsMetal())
            {
                continue;
            }
            const int atomic_num = static_cast<int>(neighbor.GetAtomicNum());

            const Vector3 vector = VectorFromAtoms(metal_atom, neighbor);
            const double distance = VectorNorm(vector);
            if (atomic_num <= 1)
            {
                if (!IsDirectMetalHydride(metal_atom, neighbor, distance))
                {
                    continue;
                }
            }
            else
            {
                const double covalent_cutoff = molgr::coordination::CoordinationDistanceCutoff(
                    static_cast<int>(metal_atom.GetAtomicNum()),
                    atomic_num,
                    config.metal_scoring);
                if (distance > covalent_cutoff)
                {
                    continue;
                }
            }

            donors.push_back(DonorSample{
                static_cast<int>(neighbor.GetIdx()),
                atomic_num,
                distance,
                vector,
            });
        }

        std::sort(
            donors.begin(),
            donors.end(),
            [](const DonorSample &lhs, const DonorSample &rhs)
            {
                return lhs.distance_angstrom < rhs.distance_angstrom;
            });
        if (static_cast<int>(donors.size()) > config.metal_radical_inference.max_considered_donors)
        {
            donors.resize(
                static_cast<std::size_t>(config.metal_radical_inference.max_considered_donors));
        }
        return donors;
    }

    /// Classify the metal coordination geometry from the donor count, angles and planarity.
    /// @param donors Coordinating ligand sites used to characterize the metal environment.
    /// @param config Ligand-field and coordination geometry thresholds for spin inference.
    /// @return Geometry class inferred from the local arrangement.
    std::string ClassifyGeometry(
        const std::vector<DonorSample> &donors,
        const molgr::config::MetalRadicalInferenceConfig &config)
    {
        const int coordination_number = static_cast<int>(donors.size());
        std::vector<Vector3> vectors;
        vectors.reserve(donors.size());
        for (const auto &donor : donors)
        {
            vectors.push_back(donor.vector);
        }

        if (coordination_number == 0)
        {
            return "free_ion";
        }
        if (coordination_number == 1)
        {
            return "terminal";
        }
        if (coordination_number == 2)
        {
            const double angle = AngleDegrees(vectors[0], vectors[1]);
            return angle >= config.linear_angle_min_degrees ? "linear" : "bent";
        }
        if (coordination_number == 3)
        {
            return PlanarityDistance(vectors) <= config.trigonal_planar_planarity_tolerance_angstrom
                       ? "trigonal_planar"
                       : "trigonal_pyramidal";
        }
        if (coordination_number == 4)
        {
            return PlanarityDistance(vectors) <= config.square_planar_planarity_tolerance_angstrom
                       ? "square_planar"
                       : "tetrahedral";
        }
        return "octahedral_like";
    }

    /// Estimate ligand-field strength from donor identities and the assigned coordination geometry.
    /// @param donors Coordinating ligand sites used to characterize the metal environment.
    /// @param geometry Coordination geometry assigned to the donor arrangement.
    /// @return Numeric estimate of the ligand field for this geometry.
    double DonorFieldScore(
        const std::vector<DonorSample> &donors,
        const std::string &geometry)
    {
        if (donors.empty())
        {
            const auto geometry_it = kGeometryFieldAdjustment.find(geometry);
            return geometry_it == kGeometryFieldAdjustment.end() ? 0.0 : geometry_it->second;
        }

        double weighted_sum = 0.0;
        double total_weight = 0.0;
        for (const auto &donor : donors)
        {
            const auto base_strength_it = kDonorFieldStrength.find(donor.atomic_num);
            const double base_strength =
                base_strength_it == kDonorFieldStrength.end() ? 0.60 : base_strength_it->second;
            const double distance_weight =
                1.0 / std::max(donor.distance_angstrom * donor.distance_angstrom, 1.0);
            weighted_sum += base_strength * distance_weight;
            total_weight += distance_weight;
        }

        const auto geometry_it = kGeometryFieldAdjustment.find(geometry);
        const double geometry_adjustment =
            geometry_it == kGeometryFieldAdjustment.end() ? 0.0 : geometry_it->second;
        if (total_weight <= 0.0)
        {
            return geometry_adjustment;
        }
        return weighted_sum / total_weight + geometry_adjustment;
    }

    /// Classify the donor-field score as weak, strong or ambiguous under the configured thresholds.
    /// @param field_score Estimated ligand-field score for the metal environment.
    /// @param config Ligand-field and coordination geometry thresholds for spin inference.
    /// @return Weak, strong or ambiguous field class.
    std::string ClassifyFieldStrength(
        double field_score,
        const molgr::config::MetalRadicalInferenceConfig &config)
    {
        const double ambiguity_margin = std::max(0.0, config.field_ambiguity_margin);
        if (field_score >= config.strong_field_threshold + ambiguity_margin)
        {
            return "strong";
        }
        if (field_score <= config.weak_field_threshold - ambiguity_margin)
        {
            return "weak";
        }
        return "ambiguous";
    }

    /// @param metal Chemical symbol of the metal.
    /// @param valence Proposed formal oxidation state of the metal.
    /// @return Shell occupations after applying the proposed metal valence, if supported.
    std::optional<ShellOccupation> ShellOccupationAfterOxidation(
        const std::string &metal,
        int valence)
    {
        const auto fdsp_it = molgr::kMetalFDSP.find(metal);
        if (fdsp_it == molgr::kMetalFDSP.end())
        {
            return std::nullopt;
        }

        const molgr::FDSP &fdsp = fdsp_it->second;
        const int total_outer = fdsp.f + fdsp.d + fdsp.s + fdsp.p;
        if (valence > total_outer)
        {
            return std::nullopt;
        }

        const int remaining_sp = std::max(0, fdsp.s + fdsp.p - valence);
        const int removed_from_d = std::min(fdsp.d, std::max(0, valence - (fdsp.s + fdsp.p)));
        const int remaining_d = fdsp.d - removed_from_d;
        const int removed_from_f =
            std::min(fdsp.f, std::max(0, valence - (fdsp.s + fdsp.p + fdsp.d)));
        const int remaining_f = fdsp.f - removed_from_f;

        int promoted_to_d = 0;
        if (kDBlockRearrangementMetals.find(metal) != kDBlockRearrangementMetals.end())
        {
            promoted_to_d = std::min(remaining_sp, std::max(0, 10 - remaining_d));
        }

        return ShellOccupation{
            remaining_f,
            remaining_d + promoted_to_d,
            remaining_sp - promoted_to_d,
        };
    }

    /// Enumerate plausible d-shell unpaired-electron counts for the field and spin preferences.
    /// @param effective_d Effective number of metal d electrons.
    /// @param geometry Coordination geometry assigned to the donor arrangement.
    /// @param field_strength Inferred weak, strong or ambiguous ligand-field class.
    /// @param prefer_low_spin Whether to consider the low-spin choice first.
    /// @param allow_low_spin_alternative Whether a low-spin alternative is permitted.
    /// @return Ordered plausible d-shell unpaired-electron counts.
    std::vector<int> CandidateDUnpairedCounts(
        int effective_d,
        const std::string &geometry,
        const std::string &field_strength,
        bool prefer_low_spin,
        bool allow_low_spin_alternative)
    {
        if (effective_d < 0 || effective_d >= static_cast<int>(molgr::kDElectronsSpin.size()))
        {
            return {0};
        }

        std::vector<int> free_ion_candidates = molgr::kDElectronsSpin[effective_d];
        std::sort(free_ion_candidates.begin(), free_ion_candidates.end());
        free_ion_candidates.erase(
            std::unique(free_ion_candidates.begin(), free_ion_candidates.end()),
            free_ion_candidates.end());

        if (geometry == "square_planar")
        {
            if (effective_d == 8)
            {
                return {0};
            }
            if (effective_d == 7 || effective_d == 9)
            {
                return {1};
            }
            return {free_ion_candidates.front(), free_ion_candidates.back()};
        }

        if (geometry == "tetrahedral")
        {
            if (field_strength == "strong")
            {
                return {free_ion_candidates.front()};
            }
            if (field_strength == "ambiguous")
            {
                return prefer_low_spin
                           ? std::vector<int>{free_ion_candidates.front(), free_ion_candidates.back()}
                           : std::vector<int>{free_ion_candidates.back(), free_ion_candidates.front()};
            }
            if (allow_low_spin_alternative)
            {
                return {free_ion_candidates.back(), free_ion_candidates.front()};
            }
            return {free_ion_candidates.back()};
        }
        if (free_ion_candidates.size() == 1)
        {
            return {free_ion_candidates.front()};
        }
        if (field_strength == "strong")
        {
            if (allow_low_spin_alternative)
            {
                return {free_ion_candidates.front(), free_ion_candidates.back()};
            }
            return {free_ion_candidates.front()};
        }
        if (field_strength == "weak")
        {
            if (allow_low_spin_alternative)
            {
                return {free_ion_candidates.back(), free_ion_candidates.front()};
            }
            return {free_ion_candidates.back()};
        }
        return prefer_low_spin
                   ? std::vector<int>{free_ion_candidates.front(), free_ion_candidates.back()}
                   : std::vector<int>{free_ion_candidates.back(), free_ion_candidates.front()};
    }
}

namespace molgr
{
    namespace metal_radical_inference
    {
        MetalRadicalInferenceResult InferMetalRadicalState(
            OpenBabel::OBAtom &metal_atom,
            int valence,
            const molgr::config::MolGRConfig &config)
        {
            const auto &metal_radical_config = config.metal_radical_inference;
            const std::string symbol =
                OpenBabel::OBElements::GetSymbol(static_cast<int>(metal_atom.GetAtomicNum()));

            const auto occupation = ShellOccupationAfterOxidation(symbol, valence);
            if (!occupation.has_value())
            {
                return {};
            }

            const auto donors = CollectCoordinationEnvironment(metal_atom, config);
            const std::string geometry = ClassifyGeometry(donors, metal_radical_config);
            const double field_score = DonorFieldScore(donors, geometry);
            const std::string field_strength = ClassifyFieldStrength(field_score, metal_radical_config);
            const int base_unpaired = (occupation->remaining_f + occupation->residual_sp) % 2;

            const auto d_candidates =
                CandidateDUnpairedCounts(
                    occupation->effective_d,
                    geometry,
                    field_strength,
                    field_score >=
                        (metal_radical_config.weak_field_threshold +
                         metal_radical_config.strong_field_threshold) /
                            2.0,
                    true);
            std::vector<int> radical_counts;
            for (const int candidate : d_candidates)
            {
                const int radical_count = base_unpaired + candidate;
                if (std::find(radical_counts.begin(), radical_counts.end(), radical_count) ==
                    radical_counts.end())
                {
                    radical_counts.push_back(radical_count);
                }
            }

            MetalRadicalInferenceResult result;
            result.radical_counts = radical_counts;
            result.effective_d_electrons = occupation->effective_d;
            result.residual_sp_electrons = occupation->residual_sp;
            result.remaining_f_electrons = occupation->remaining_f;
            result.coordination_number = static_cast<int>(donors.size());
            result.geometry = geometry;
            result.field_score = field_score;
            result.field_strength = field_strength;
            return result;
        }

        std::vector<int> InferMetalRadicalCounts(
            OpenBabel::OBAtom &metal_atom,
            int valence,
            const molgr::config::MolGRConfig &config)
        {
            return InferMetalRadicalState(metal_atom, valence, config).radical_counts;
        }
    }
}
