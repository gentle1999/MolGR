#pragma once

#include "molgr/state.h"

#include <openbabel/mol.h>

#include <optional>
#include <utility>
#include <vector>

namespace molgr
{
    namespace no_metals
    {
        namespace neighbor_radicals
        {
            /// Keep a charge-aware alternative to closing an organic N--O radical pair.
            /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
            /// @return True if the ligand contains the substituted N-O radical pair.
            bool HasSubstitutedNORadicalPair(const OpenBabel::OBMol &mol);

            /// Defer elongated O2 and carbonyl pairs to charge/spin-aware normalization.
            /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
            /// @return Distinct eligible bonds as pairs of atom indices.
            std::vector<std::pair<int, int>> NeighborRadicalBondPairs(
                const OpenBabel::OBMol &mol);

            /// Resolve eligible radical pairs before charge/spin-aware normalization.
            /// @param state Ligand graph and electronic budgets at this stage of reconstruction.
            /// @param exact_discrepancy Required charge-separation discrepancy of the generated electronic
            ///     seeds.
            /// @return Electronic states surviving this seed expansion or search stage.
            std::vector<molgr::state::ReconstructionState> EnumerateNeighborRadicalSeeds(
                const molgr::state::ReconstructionState &state,
                std::optional<int> exact_discrepancy = std::nullopt);
        }
    }
}
