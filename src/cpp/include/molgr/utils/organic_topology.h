#pragma once

#include "molgr/config.h"

#include <openbabel/bond.h>
#include <openbabel/mol.h>

#include <vector>

namespace molgr
{
    namespace organic_topology
    {
        /// Aromatic ring, conjugation and donor measures used to rank an organic graph.
        struct OrganicTopologyMetrics
        {
            int aromatic_atom_count = 0;
            int aromatic_ring_count = 0;
            double aromatic_stability_score = 0.0;
            int conjugated_atom_count = 0;
            int conjugated_bond_count = 0;
            int max_conjugated_component_size = 0;
            std::vector<int> conjugated_atom_indices;
            int hyperconjugative_donor_count = 0;
            int hyperconjugation_score = 0;
            // Zero-based atom sets, with orthogonal cumulene systems separate.
            std::vector<std::vector<int>> conjugated_atom_components;
        };

        /// @param bond Bond whose endpoints and order define the local environment.
        /// @return Whether this bond belongs to the geometrically validated conjugated system.
        bool IsConjugatedBond(const OpenBabel::OBBond &bond);
        /// Measure aromatic rings, conjugated components and hyperconjugative donors on the ligand graph.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @param config Aromatic stability weights and conjugation geometry thresholds.
        /// @return Aromaticity, conjugation and hyperconjugation measurements of the ligand.
        OrganicTopologyMetrics ComputeOrganicTopologyMetrics(
            const OpenBabel::OBMol &mol,
            const molgr::config::OrganicTopologyConfig &config =
                molgr::config::GetDefaultConfig().organic_topology);
    }
}
