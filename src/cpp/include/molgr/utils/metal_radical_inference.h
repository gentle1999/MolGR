#pragma once

#include "molgr/config.h"

#include <openbabel/atom.h>

#include <string>
#include <vector>

namespace molgr
{
    namespace metal_radical_inference
    {
        /// Candidate unpaired-electron assignments and ligand-field evidence for a metal charge state.
        struct MetalRadicalInferenceResult
        {
            std::vector<int> radical_counts;
            int effective_d_electrons = 0;
            int residual_sp_electrons = 0;
            int remaining_f_electrons = 0;
            int coordination_number = 0;
            std::string geometry = "free_ion";
            double field_score = 0.0;
            std::string field_strength = "weak";
        };

        /// Infer accessible metal spin assignments from valence and the local coordination environment.
        /// @param metal_atom Open Babel atom representing the metal center.
        /// @param valence Proposed formal oxidation state of the metal.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return Candidate unpaired-electron counts and the ligand-field evidence behind them.
        MetalRadicalInferenceResult InferMetalRadicalState(
            OpenBabel::OBAtom &metal_atom,
            int valence,
            const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());

        /// @param metal_atom Open Babel atom representing the metal center.
        /// @param valence Proposed formal oxidation state of the metal.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return The possible unpaired-electron counts for the specified metal assignment.
        std::vector<int> InferMetalRadicalCounts(
            OpenBabel::OBAtom &metal_atom,
            int valence,
            const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());
    }
}
