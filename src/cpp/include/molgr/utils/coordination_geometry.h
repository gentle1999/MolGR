#pragma once

#include "molgr/compat/openbabel_iter.h"
#include "molgr/types.h"
#include "molgr/utils/coordination.h"
#include <openbabel/atom.h>
#include <openbabel/mol.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <vector>

namespace molgr::coordination
{
    /// Cartesian point in angstroms in the input coordinate frame.
    struct Point3D
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    /// Sphere used to test whether a ligand atom shields a donor from the metal.
    struct CoordinationBlocker
    {
        int atom_idx = 0;
        Point3D coordinates;
        double radius = 0.0;
    };

    /// @param atom Atom whose local chemical environment is examined.
    /// @return Cartesian coordinates in ångströms for the metal–donor visibility test.
    inline Point3D AtomCoordinates(const OpenBabel::OBAtom &atom)
    {
        return {atom.GetX(), atom.GetY(), atom.GetZ()};
    }

    /// @param metal_state Detached metal element and Cartesian position in the ligand coordinate frame.
    /// @return The Cartesian position of a detached metal site.
    inline Point3D MetalCoordinates(const molgr::MetalSite &metal_state)
    {
        return {metal_state.position_x, metal_state.position_y, metal_state.position_z};
    }

    /// @param atom Atom whose local chemical environment is examined.
    /// @param metal_state Detached metal element and Cartesian position in the ligand coordinate frame.
    /// @return The metal–atom distance in angstroms.
    inline double DistanceToMetal(
        const OpenBabel::OBAtom &atom,
        const molgr::MetalSite &metal_state)
    {
        const double dx = static_cast<double>(atom.GetX()) - metal_state.position_x;
        const double dy = static_cast<double>(atom.GetY()) - metal_state.position_y;
        const double dz = static_cast<double>(atom.GetZ()) - metal_state.position_z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// M–donor–terminal angle in degrees for a metal outside the ligand OBMol.
    /// @param metal Metal site being considered.
    /// @param donor Atom or site occupying this position in the local chemical motif.
    /// @param terminal Terminal ligand atom defining the donor angle.
    /// @return Angle in degrees at the donor, or zero for degenerate vectors.
    inline double MetalDonorAngle(const molgr::MetalSite &metal,
                                 const OpenBabel::OBAtom &donor,
                                 const OpenBabel::OBAtom &terminal)
    {
        const double mx = metal.position_x - donor.GetX();
        const double my = metal.position_y - donor.GetY();
        const double mz = metal.position_z - donor.GetZ();
        const double tx = terminal.GetX() - donor.GetX();
        const double ty = terminal.GetY() - donor.GetY();
        const double tz = terminal.GetZ() - donor.GetZ();
        const double denominator = std::sqrt(mx * mx + my * my + mz * mz) *
                                   std::sqrt(tx * tx + ty * ty + tz * tz);
        if (!std::isfinite(denominator) || denominator <= 0.0) return 0.0;
        const double cosine = (mx * tx + my * ty + mz * tz) / denominator;
        return std::acos(std::clamp(cosine, -1.0, 1.0)) * 180.0 / std::acos(-1.0);
    }

    /// @param point Cartesian position in the input frame, in angstroms.
    /// @param segment_start Cartesian position in the input frame, in angstroms.
    /// @param segment_end Cartesian position in the input frame, in angstroms.
    /// @return The shortest distance from a blocker center to the metal–donor segment.
    inline double DistancePointToSegment(
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

    /// Represent ligand atoms as spheres using the configured covalent-radius scale and clearance.
    /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
    /// @param config Metal contact cutoffs, visibility radii and candidate scoring weights.
    /// @return Ligand blocker spheres with atom indices in the input molecule.
    inline std::vector<CoordinationBlocker> BuildCoordinationBlockers(
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

    /// @param atom_idx One-based Open Babel atom index.
    /// @param atom_coordinates Cartesian position in the input frame, in angstroms.
    /// @param segment_start Cartesian position in the input frame, in angstroms.
    /// @param blockers Ligand atoms and shielding radii used in the line-of-sight test.
    /// @return True when the metal–donor segment clears every other blocker sphere.
    inline bool HasUnobstructedCoordinationPathFromBlockers(
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

    /// Maximum inner-sphere contact distance for the metal and donor elements.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param metal_state Detached metal element and Cartesian position in the ligand coordinate frame.
    /// @param config Metal contact cutoffs, visibility radii and candidate scoring weights.
    /// @return Maximum accepted donor-metal separation in angstroms.
    inline double CoordinationRadiusCutoff(
        const OpenBabel::OBAtom &atom,
        const molgr::MetalSite &metal_state,
        const molgr::config::MetalScoringConfig &config)
    {
        return molgr::coordination::CoordinationDistanceCutoff(
            static_cast<int>(metal_state.element_idx),
            static_cast<int>(atom.GetAtomicNum()),
            config);
    }

    /// Test whether a positive metal–donor distance lies within the element-specific cutoff.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param metal_state Detached metal element and Cartesian position in the ligand coordinate frame.
    /// @param config Metal contact cutoffs, visibility radii and candidate scoring weights.
    /// @return True if the metal contact lies within the inner-sphere window.
    inline bool IsInnerSphereAtom(
        const OpenBabel::OBAtom &atom,
        const molgr::MetalSite &metal_state,
        const molgr::config::MetalScoringConfig &config)
    {
        const double distance = DistanceToMetal(atom, metal_state);
        return distance > 0.0 && distance <= CoordinationRadiusCutoff(atom, metal_state, config);
    }

    /// Test for a nonzero metal–donor separation and an unobstructed path between the sites.
    /// @param atom Atom whose local chemical environment is examined.
    /// @param metal_state Detached metal element and Cartesian position in the ligand coordinate frame.
    /// @param blockers Ligand atoms and shielding radii used in the line-of-sight test.
    /// @return True if nearby atoms leave a clear path to the metal.
    inline bool IsVisibleToMetalAtom(
        OpenBabel::OBAtom &atom,
        const molgr::MetalSite &metal_state,
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

    /// OBMol indices are stable throughout the ligand search. Store the nearest
    /// visible inner-sphere contact in milliångströms once per preparation.
    /// @param mol Molecular graph on which the operation acts.
    /// @param sites Metal sites used to evaluate ligand coordination.
    /// @param config Metal contact cutoffs, visibility radii and candidate scoring weights.
    /// @return Distances indexed by one-based atom number, in milliangstroms. INT_MAX marks an
    ///     unmatched atom. An empty vector means no visible inner-sphere contact.
    inline std::vector<int> MetalContactDistances(OpenBabel::OBMol &mol,
        const std::vector<MetalSite> &sites, const config::MetalScoringConfig &config)
    {
        if (sites.empty()) return {};
        const auto blockers = BuildCoordinationBlockers(mol, config);
        const int missing = std::numeric_limits<int>::max();
        std::vector<int> distances(mol.NumAtoms() + 1, missing);
        bool found = false;
        for (const auto &site : sites)
        {
            FOR_ATOMS_OF_MOL(atom, mol)
            {
                if (!IsInnerSphereAtom(*atom, site, config) || !IsVisibleToMetalAtom(*atom, site, blockers))
                    continue;
                auto &nearest = distances[atom->GetIdx()];
                nearest = std::min(nearest, static_cast<int>(DistanceToMetal(*atom, site) * 1000 + 0.5));
                found = true;
            }
        }
        if (!found) distances.clear();
        return distances;
    }

    /// A terminal N–O site with its metal contact geometry.
    struct NitrosylContact
    {
        OpenBabel::OBAtom *nitrogen;
        OpenBabel::OBAtom *oxygen;
        double weight;
    };

    /// Yield short terminal N-bound contacts with their M–N–O geometry weights.
    /// @param mol Molecular graph on which the operation acts.
    /// @param metal_sites Metal sites used to evaluate ligand coordination.
    /// @param visible_atoms_by_metal Ligand atoms visible from each metal site, grouped by metal.
    /// @return Terminal N-O contacts and their metal-approach geometry.
    inline std::vector<NitrosylContact> TerminalNitrosylContacts(
        OpenBabel::OBMol &mol,
        const std::vector<molgr::MetalSite> &metal_sites,
        const std::vector<std::vector<OpenBabel::OBAtom *>> &visible_atoms_by_metal)
    {
        // Restricted representation preference for short, near-linear, terminal
        // N-bound NO. It is not a universal rule for nitrosyl oxidation states.
        std::map<int, std::vector<std::size_t>> visible_sites;
        for (std::size_t site = 0; site < visible_atoms_by_metal.size(); ++site)
        {
            for (auto *atom : visible_atoms_by_metal[site])
                visible_sites[atom->GetIdx()].push_back(site);
        }
        std::vector<NitrosylContact> result;
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            auto &nitrogen = *atom_iter;
            if (nitrogen.GetAtomicNum() != 7 || nitrogen.GetExplicitDegree() != 1 ||
                nitrogen.GetImplicitHCount() != 0 || visible_sites[nitrogen.GetIdx()].size() != 1)
                continue;
            OpenBabel::OBAtom *oxygen = nullptr;
            FOR_NBORS_OF_ATOM(neighbor, nitrogen) { oxygen = &*neighbor; }
            if (!oxygen || oxygen->GetAtomicNum() != 8 || oxygen->GetExplicitDegree() != 1 ||
                oxygen->GetImplicitHCount() != 0 || !visible_sites[oxygen->GetIdx()].empty())
                continue;
            const auto &metal = metal_sites[visible_sites[nitrogen.GetIdx()].front()];
            const double no_length = nitrogen.GetDistance(oxygen);
            if (!(no_length > 1.0 && no_length < 1.25)) continue;
            const double angle = MetalDonorAngle(metal, nitrogen, *oxygen);
            if (angle <= 160.0) continue;
            const double weight = std::min(1.0, (angle - 160.0) / 20.0) *
                                  std::min(1.0, (1.25 - no_length) / 0.10);
            result.push_back({&nitrogen, oxygen, weight});
        }
        return result;
    }
}
