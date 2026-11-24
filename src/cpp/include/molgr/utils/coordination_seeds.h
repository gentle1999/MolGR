#pragma once

#include "molgr/state.h"
#include <map>
#include <tuple>
#include <vector>

namespace molgr::coordination
{
    namespace hydrazido
    {
        using Pair = std::pair<int, int>;
        // Formal charges on terminal and substituted N, followed by N–N bond order.
        using Electronic = std::tuple<int, int, int>;
        using Sites = std::map<Pair, Electronic>;
        inline constexpr Electronic neutral{-1, 1, 2}, reduced{-2, 0, 1};

        /// Match terminal N–NR2 ligands and retain the formal charges and N–N order at each site.
        /// @param mol Molecular graph on which the operation acts.
        /// @return Matched N-N ligand sites with their electronic assignments.
        Sites LocalSites(OpenBabel::OBMol &mol);
        /// Test a short, nearly linear M–N–N approach to the terminal nitrogen.
        /// @param first First or second participant of the local motif, in argument order.
        /// @param second First or second participant of the local motif, in argument order.
        /// @param metal Metal site being considered.
        /// @return True if the metal approaches the terminal nitrogen within the distance and angle
        ///     windows.
        bool SupportedGeometry(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second,
                               const molgr::MetalSite &metal);
    }

    /// Remove a stretched C–H seed bond when the same metal can bind hydride and carbonyl.
    /// @param ligands Ligand graph after detaching the metal atoms.
    /// @param metals Metal sites used to evaluate ligand coordination.
    void DetachCarbonylHydrideContacts(OpenBabel::OBMol &ligands, const std::vector<molgr::MetalSite> &metals);

    /// Keep the original ligand occupation and branch on geometry-supported NO and N–N states.
    /// @param seed Initial ligand state for this search.
    /// @param metals Metal sites used to evaluate ligand coordination.
    /// @param config Metal contact cutoffs, visibility radii and candidate scoring weights.
    /// @return Original seed and the admissible electronic alternatives at coordinating sites.
    std::vector<molgr::state::ReconstructionState> ExpandCoordinationSeeds(
        const molgr::state::ReconstructionState &seed,
        const std::vector<molgr::MetalSite> &metals,
        const molgr::config::MetalScoringConfig &config);
}
