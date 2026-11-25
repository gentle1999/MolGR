#pragma once

#include "molgr/config.h"
#include "molgr/types.h"

#include <openbabel/mol.h>

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

namespace molgr
{
    namespace scoring
    {
        /// Raw force-field energy, unit and converted energy for comparing reconstructed graphs.
        struct ForceFieldEvaluation
        {
            double raw_energy = 0.0;
            std::string raw_unit;
            double energy_kj_mol = 0.0;
            int atom_count = 0;
            int heavy_atom_count = 0;
            bool contains_metals = false;
        };

        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return A fingerprint of the coordinates, connectivity and electronic labels used for energy
        ///     evaluation.
        std::string BuildScoreKey(const OpenBabel::OBMol &mol);
        /// @param metal_states Proposed valence and unpaired-electron assignments for the metals.
        /// @return A fingerprint of the metal charge, unpaired-electron and position assignments.
        std::string BuildMetalStateKey(const std::vector<molgr::MetalAtomPosition> &metal_states);

        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return True if at least one atom belongs to the metal set.
        bool ContainsMetalAtoms(const OpenBabel::OBMol &mol);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return A copy with the metal atoms removed, retaining the ligand graph.
        OpenBabel::OBMol StripMetalAtoms(const OpenBabel::OBMol &mol);

        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return A cached or freshly calculated UFF evaluation at the input coordinates.
        ForceFieldEvaluation EvaluateForceField(
            const OpenBabel::OBMol &mol,
            const molgr::config::MolGRConfig &config);
        /// @return Cache hits, misses and resident evaluations.
        std::tuple<std::size_t, std::size_t, std::size_t> ForceFieldEvaluationCacheInfo();
        /// Clear the cached UFF evaluations before measuring or rescoring an independent run.
        void ForceFieldEvaluationCacheClear();
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return UFF evaluation of the supplied ligand graph.
        ForceFieldEvaluation OrganicForceFieldEvaluation(
            const OpenBabel::OBMol &mol,
            const molgr::config::MolGRConfig &config);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param score_key Molecular fingerprint for reusing an evaluation with the same electronic state.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return The organic graph's force-field evaluation, reusing a supplied fingerprint when
        ///     available.
        ForceFieldEvaluation OrganicForceFieldEvaluationWithScoreKey(
            const OpenBabel::OBMol &mol,
            const std::string &score_key,
            const molgr::config::MolGRConfig &config);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return The force-field evaluation for the assembled metal–ligand graph.
        ForceFieldEvaluation CombinedForceFieldEvaluation(
            const OpenBabel::OBMol &mol,
            const molgr::config::MolGRConfig &config);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return The force-field evaluation used for selection after projecting away metal atoms when
        ///     applicable.
        ForceFieldEvaluation SelectionForceFieldEvaluation(
            const OpenBabel::OBMol &mol,
            const molgr::config::MolGRConfig &config);

        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return Organic graph UFF energy in kJ/mol.
        double OrganicForceFieldEnergy(
            const OpenBabel::OBMol &mol,
            const molgr::config::MolGRConfig &config);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return UFF energy of the combined graph in kJ/mol.
        double CombinedForceFieldEnergy(
            const OpenBabel::OBMol &mol,
            const molgr::config::MolGRConfig &config);
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return Selection energy in kJ/mol using the ligand projection when applicable.
        double SelectionForceFieldEnergy(
            const OpenBabel::OBMol &mol,
            const molgr::config::MolGRConfig &config);
    }
}
