#pragma once

#include "molgr/config.h"
#include "molgr/state.h"
#include "molgr/types.h"

#include <openbabel/atom.h>
#include <openbabel/mol.h>

#include <set>
#include <string>
#include <vector>

namespace molgr
{
    namespace metal
    {
        namespace preparation
        {
            /// @param metal_symbol Element symbol of the metal.
            /// @param valence Proposed formal oxidation state of the metal.
            /// @return Heuristic radical counts for a metal under a candidate valence.
            std::set<int> GetPossibleMetalRadicals(const std::string &metal_symbol, int valence);

            /// Enumerate candidate formal charges and spin states of the detached metal.
            /// @param obatom Open Babel atom representing the metal center.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Proposed metal states with positions, charges and spin labels.
            std::vector<molgr::metal::MetalAtomPosition> BuildMetalStates(
                OpenBabel::OBAtom &obatom,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());

            /// Restore the selected metal atoms at their original input positions in the reconstructed
            /// graph.
            /// @param mol Molecular graph on which the operation acts.
            /// @param metals Metal sites used to evaluate ligand coordination.
            void ReinsertMetalStates(
                OpenBabel::OBMol &mol,
                const std::vector<molgr::metal::MetalAtomPosition> &metals);

            /// Insert selected metal charge/spin states into the no-metal winner.
            /// @param mol Molecular graph on which the operation acts.
            /// @param metals Metal sites used to evaluate ligand coordination.
            void CombineMetalWithOmol(
                OpenBabel::OBMol &mol,
                const std::vector<molgr::metal::MetalAtomPosition> &metals);

            /// Split the input into a no-metal XYZ block plus per-metal state options.
            /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
            ///     angstroms.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Detached ligand seed, metal alternatives and residual reconstruction targets.
            molgr::state::MetalPreparationState PrepareMetalState(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());
        }
    }
}
