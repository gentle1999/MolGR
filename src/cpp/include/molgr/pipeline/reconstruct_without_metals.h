#pragma once

#include "molgr/config.h"
#include "molgr/state.h"
#include "molgr/utils/perf.h"
#include "molgr/utils/utils.h"

#include <openbabel/mol.h>

#include <optional>
#include <string>
#include <vector>

namespace molgr
{
    namespace pipeline
    {
        namespace reconstruct_without_metals
        {
            /// Electronic, aromatic and conjugation scores used to compare nonmetal reconstruction candidates.
            struct DebugNoMetalCandidateSummary
            {
                std::string smiles;
                int resonance_index = -1;
                double score = 0.0;
                double aromatic_stability_score = 0.0;
                int aromatic_atom_count = 0;
                int aromatic_ring_count = 0;
                int max_conjugated_component_size = 0;
                int conjugated_atom_count = 0;
                int conjugated_bond_count = 0;
                int hyperconjugative_donor_count = 0;
                int hyperconjugation_score = 0;
                int azinium_geometry_conflict_count = 0;
                int formal_charge_absolute_sum = 0;
                int long_carbonyl_multiple_bond_count = 0;
                double conjugation_charge_penalty = 0.0;
                double adjusted_max_conjugated_component_size = 0.0;
                double adjusted_conjugated_atom_count = 0.0;
                double adjusted_conjugated_bond_count = 0.0;
                int excess_radical_labels = 0;
            };

            /// Select the best ligand state compatible with the input charge and radical budgets.
            /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
            ///     angstroms.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @param timing_reducer Run-level collector for timing and work counters.
            /// @param preheat_score_bundle Whether to calculate reusable ligand scoring terms before
            ///     returning.
            /// @param metal_sites Metal sites used to evaluate ligand coordination.
            /// @return Selected ligand state, or an empty optional if no acceptable state was found.
            std::optional<molgr::state::ReconstructionState> XyzToOmolNoMetalState(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig(),
                perf::RunTimingReducer *timing_reducer = nullptr,
                bool preheat_score_bundle = true,
                const std::vector<molgr::MetalSite> &metal_sites = {});

            /// Return the best no-metal state from a shared parsed seed molecule.
            /// @param seed_omol Starting Open Babel connectivity and coordinates.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @param timing_reducer Run-level collector for timing and work counters.
            /// @param preheat_score_bundle Whether to calculate reusable ligand scoring terms before
            ///     returning.
            /// @param metal_sites Metal sites used to evaluate ligand coordination.
            /// @return Selected ligand state, or an empty optional if no acceptable state was found.
            std::optional<molgr::state::ReconstructionState> SeedOmolToOmolNoMetalState(
                const OpenBabel::OBMol &seed_omol,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig(),
                perf::RunTimingReducer *timing_reducer = nullptr,
                bool preheat_score_bundle = true,
                const std::vector<molgr::MetalSite> &metal_sites = {});

            /// Search electronic assignments starting from a copy of the supplied Open Babel graph.
            /// @param seed_omol Starting Open Babel connectivity and coordinates.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @param timing_reducer Run-level collector for timing and work counters.
            /// @param preheat_score_bundle Whether to calculate reusable ligand scoring terms before
            ///     returning.
            /// @param candidate_pool Optional destination for ligand or metal candidate states retained
            ///     during the search.
            /// @param metal_sites Metal sites used to evaluate ligand coordination.
            /// @return Selected ligand state, or an empty optional if no acceptable state was found.
            std::optional<molgr::state::ReconstructionState> SeedOmolCopyToOmolNoMetalState(
                std::shared_ptr<OpenBabel::OBMol> seed_omol,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig(),
                perf::RunTimingReducer *timing_reducer = nullptr,
                bool preheat_score_bundle = true,
                std::vector<molgr::state::ReconstructionState> *candidate_pool = nullptr,
                const std::vector<molgr::MetalSite> &metal_sites = {});

            /// Enumerate ligand assignments compatible with the detached metal sites and requested electron
            /// budgets.
            /// @param seed Initial ligand state for this search.
            /// @param charge Signed formal charge used for this electronic assignment.
            /// @param radicals Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @param timing Run-level collector for timing and work counters.
            /// @param sites Metal sites used to evaluate ligand coordination.
            /// @return Ligand candidates compatible with the requested charge and unpaired-electron
            ///     budgets.
            std::vector<molgr::state::ReconstructionState> CollectLigandCandidates(
                std::shared_ptr<OpenBabel::OBMol> seed, int charge, int radicals,
                const molgr::config::MolGRConfig &config, perf::RunTimingReducer *timing,
                const std::vector<molgr::MetalSite> &sites);

            /// Reconstruct a metal-free XYZ graph at the specified charge and unpaired-electron count.
            /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
            ///     angstroms.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Reconstructed molecular records, or null if the targets cannot be satisfied.
            std::unique_ptr<molgr::utils::MoleculeData> XyzToMolDataNoMetal(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());

            /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
            ///     angstroms.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Per-candidate scores for inspecting native ligand resonance selection.
            std::vector<DebugNoMetalCandidateSummary> DebugNoMetalResonanceCandidateSummaries(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());
        }
    }
}
