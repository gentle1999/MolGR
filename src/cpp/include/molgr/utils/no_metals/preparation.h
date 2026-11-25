#pragma once

#include "molgr/state.h"

#include <openbabel/mol.h>

#include <memory>
#include <string>
#include <vector>

namespace molgr
{
    namespace no_metals
    {
        namespace preparation
        {
            /// Perceive an initial Open Babel ligand graph from Cartesian coordinates.
            /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
            ///     angstroms.
            /// @return New seed molecule, or null if XYZ parsing fails.
            std::shared_ptr<OpenBabel::OBMol> SeedOmolFromXyzBlock(
                const std::string &xyz_block);

            /// Copy the seed graph and reset atom-local charge and reconstruction electron fields.
            /// @param seed_omol Starting Open Babel connectivity and coordinates.
            /// @return Independent ligand graph with its seed charges and active electron labels reset.
            std::shared_ptr<OpenBabel::OBMol> NormalizeSeedOmolCopy(
                const OpenBabel::OBMol &seed_omol);

            /// Attach the global charge and radical targets to the prepared ligand graph.
            /// @param omol Molecular graph on which the operation acts.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @return Seed snapshot carrying the ligand graph and its charge and spin targets.
            molgr::state::ReconstructionState BuildSeedState(
                std::shared_ptr<OpenBabel::OBMol> omol,
                int total_charge,
                int total_radical_electrons);

            /// Create an independent initial state by cloning a normalized seed molecule.
            /// @param seed_omol Starting Open Babel connectivity and coordinates.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @return Seed snapshot carrying the ligand graph and its charge and spin targets.
            molgr::state::ReconstructionState SeedStateFromOmol(
                const OpenBabel::OBMol &seed_omol,
                int total_charge,
                int total_radical_electrons);

            /// Create the initial reconstruction state from the input XYZ block.
            /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
            ///     angstroms.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @return Seed snapshot carrying the ligand graph and its charge and spin targets.
            molgr::state::ReconstructionState SeedState(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons);

            /// Compatibility entry point for the pipeline-owned linear reconstruction.
            /// @param state Ligand graph and electronic budgets at this stage of reconstruction.
            /// @param metal_distances Nearest eligible metal-contact distances indexed by ligand atom, in
            ///     milliangstroms.
            /// @return Seed snapshot carrying the ligand graph and its charge and spin targets.
            molgr::state::ReconstructionState PrepareNoMetalSeed(
                const molgr::state::ReconstructionState &state,
                const std::vector<int> &metal_distances = {});

            /// Retain the seed and enumerate local charge or spin assignments supported by its geometry.
            /// @param state Ligand graph and electronic budgets at this stage of reconstruction.
            /// @return Electronic states surviving this seed expansion or search stage.
            std::vector<molgr::state::ReconstructionState> ExpandGeometryElectronicSeeds(
                const molgr::state::ReconstructionState &state);
        }
    }
}
