#pragma once

#include "molgr/config.h"
#include "molgr/types.h"
#include "molgr/utils/organic_topology.h"

#include <openbabel/mol.h>

#include <string>
#include <utility>
#include <vector>

namespace molgr
{
    namespace scoring
    {
        using ChargedAtomSnapshot = molgr::ChargedAtomSnapshot;
        using ChargedAtomSnapshotList = molgr::ChargedAtomSnapshotList;
        using MetalAtomPosition = molgr::MetalAtomPosition;

        using OrganicTopologyMetrics = molgr::organic_topology::OrganicTopologyMetrics;

        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return The graph reconstruction score using the selected configuration and available cached
        ///     terms.
        double OmolScore(const OpenBabel::OBMol &mol);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return The graph reconstruction score using the selected configuration and available cached
        ///     terms.
        double OmolScore(const OpenBabel::OBMol &mol, const molgr::config::MolGRConfig &config);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return The organic-core contribution before metal reinsertion.
        double OrganicCoreScore(const OpenBabel::OBMol &mol);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return The organic-core contribution before metal reinsertion.
        double OrganicCoreScore(const OpenBabel::OBMol &mol, const molgr::config::MolGRConfig &config);
        /// Recompute the post-reinsertion contribution for the combined graph.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Numerical score contribution for the assignment.
        double PostReinsertionScore(const OpenBabel::OBMol &mol);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Ligand symmetry and charged-atom geometry needed for subsequent metal scoring.
        std::pair<double, ChargedAtomSnapshotList> BuildPostReinsertionBaseComponents(
            const OpenBabel::OBMol &mol);
        /// Evaluate the reinsertion contribution from metal states and cached ligand components.
        /// @param base_symmetry_penalty Ligand symmetry contribution computed before metal reinsertion.
        /// @param charged_atom_snapshots Positions and formal charges of the ligand centers used for
        ///     reinsertion scoring.
        /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
        /// @return Numerical score contribution for the assignment.
        double PostReinsertionScoreFromMetalStates(
            double base_symmetry_penalty,
            const ChargedAtomSnapshotList &charged_atom_snapshots,
            const std::vector<MetalAtomPosition> &metal_states);
        /// Combine organic-core and reinsertion terms for this metal assignment.
        /// @param organic_score Score of the detached organic graph.
        /// @param post_reinsertion_base_key Cached ligand fingerprint for post-reinsertion evaluation.
        /// @param base_symmetry_penalty Ligand symmetry contribution computed before metal reinsertion.
        /// @param charged_atom_snapshots Positions and formal charges of the ligand centers used for
        ///     reinsertion scoring.
        /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
        /// @return Numerical score contribution for the assignment.
        double CombinedCandidateScoreFromMetalStates(
            double organic_score,
            const std::string &post_reinsertion_base_key,
            double base_symmetry_penalty,
            const ChargedAtomSnapshotList &charged_atom_snapshots,
            const std::vector<MetalAtomPosition> &metal_states);

        /// Penalize symmetry-breaking electronic labels in the molecular graph.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Numerical score contribution for the assignment.
        double CalcSymmetryPenalty(const OpenBabel::OBMol &mol);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return The physicochemical penalty for the current molecular electronic assignment.
        double CalculatePhysChemPenalty(const OpenBabel::OBMol &mol);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return The metal-state penalty used in candidate ranking.
        double CalculateMetalPenalty(const OpenBabel::OBMol &mol);
        /// @param charged_atom_snapshots Positions and formal charges of the ligand centers used for
        ///     reinsertion scoring.
        /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
        /// @param cutoff Maximum metal interaction distance in angstroms.
        /// @return The metal-state penalty used in candidate ranking.
        double CalculateMetalPenaltyFromMetalStates(
            const ChargedAtomSnapshotList &charged_atom_snapshots,
            const std::vector<MetalAtomPosition> &metal_states,
            double cutoff = 2.6);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param atom Atom whose local chemical environment is examined.
        /// @return The local geometry deviation score for the specified atom.
        double GetDeviationScore(const OpenBabel::OBMol &mol, const OpenBabel::OBAtom *atom);
        /// Return aromaticity and conjugation metrics used by the molecular scoring terms.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Aromaticity, conjugation and hyperconjugation measurements of the ligand.
        OrganicTopologyMetrics ComputeOrganicTopologyMetrics(const OpenBabel::OBMol &mol);
    }
}
