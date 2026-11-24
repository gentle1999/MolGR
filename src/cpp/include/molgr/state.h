#pragma once

#include "molgr/config.h"
#include "molgr/types.h"

#include <openbabel/mol.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace molgr
{
    namespace state
    {
        using MetadataValue = std::variant<int, double, bool, std::string>;
        using MetadataMap = std::unordered_map<std::string, MetadataValue>;
        using CombinedOmolBuilder = std::function<std::shared_ptr<OpenBabel::OBMol>(
            const OpenBabel::OBMol &,
            const std::vector<molgr::MetalAtomPosition> &)>;

        /// Cached ligand scores prepared for comparing multiple metal assignments against the same ligand state.
        struct PreheatedNoMetalScoreBundle
        {
            std::string post_reinsertion_base_key;
            double organic_core_score = 0.0;
            double base_symmetry_penalty = 0.0;
            molgr::ChargedAtomSnapshotList charged_atom_snapshots;
        };

        /// Ligand graph snapshot with its charge budget, radical target and search metadata.
        struct ReconstructionState
        {
            std::shared_ptr<OpenBabel::OBMol> omol;
            int given_charge = 0;
            int total_charge = 0;
            int total_radical_electrons = 0;
            std::vector<std::string> phase_history;
            mutable MetadataMap metadata;
            int omol_revision = 0;

            mutable std::optional<std::pair<int, std::string>> organic_score_key_cache;
            mutable std::optional<std::pair<int, double>> organic_core_score_cache;
            mutable std::optional<std::pair<int, double>> full_score_cache;
            mutable std::optional<std::pair<int, double>> post_reinsertion_base_symmetry_penalty_cache;
            mutable std::optional<std::pair<int, molgr::ChargedAtomSnapshotList>>
                post_reinsertion_charged_atom_snapshots_cache;
            mutable std::shared_ptr<const PreheatedNoMetalScoreBundle> preheated_score_bundle;

            /// Construct an empty reconstruction state ready to receive its graph or electronic targets.
            ReconstructionState() = default;

            /// Retain the supplied molecular graph, electron budgets, history and cache revision as a
            /// snapshot.
            /// @param omol_ Molecular graph on which the operation acts.
            /// @param given_charge_ Signed charge still to assign to the ligand.
            /// @param total_charge_ Target net formal charge of the full input.
            /// @param total_radical_electrons_ Target number of unpaired electrons.
            /// @param phase_history_ Ordered reconstruction stages already applied.
            /// @param metadata_ Reconstruction annotations and cached scoring terms for this state.
            /// @param omol_revision_ Revision identifying the ligand graph used by derived caches.
            ReconstructionState(
                std::shared_ptr<OpenBabel::OBMol> omol_,
                int given_charge_,
                int total_charge_,
                int total_radical_electrons_,
                std::vector<std::string> phase_history_ = {},
                MetadataMap metadata_ = {},
                int omol_revision_ = 0);

            /// @return A borrowed read-only reference to the snapshot's ligand graph.
            const OpenBabel::OBMol &Mol() const;
            /// @return The ligand graph for in-place electronic or bond edits.
            OpenBabel::OBMol &MutableMol();

            /// Drop scores and topology derived from the previous graph revision.
            void InvalidateOmolDerivedCache();
            /// Build reusable score components for ranking one ligand with multiple metal assignments.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Ligand scoring terms reusable across metal assignments.
            PreheatedNoMetalScoreBundle BuildPreheatedScoreBundle(
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig()) const;
            /// Cache the ligand's reusable score components before candidate evaluation.
            /// @param config Settings for the current reconstruction or scoring operation.
            void PreheatScoreBundle(
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig());
            /// @return The prepared scoring bundle, or null until it has been populated.
            const PreheatedNoMetalScoreBundle *PreheatedScoreBundle() const;
            /// @return The ligand fingerprint used to reuse post-reinsertion score components.
            std::string PostReinsertionBaseKey() const;
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return The cached organic-core score or evaluate it for this graph revision.
            double OrganicCoreScore(
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig()) const;
            /// @return The symmetry contribution and charged-atom positions used after metal reinsertion.
            std::pair<double, molgr::ChargedAtomSnapshotList> PostReinsertionBaseComponents() const;
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return The full ligand score for the current graph revision.
            double FullScore(
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig()) const;
        };

        /// Apply ordered ligand-graph edits while carrying the remaining charge and spin targets.
        class OmolStateMachine
        {
        public:
            std::shared_ptr<OpenBabel::OBMol> omol;
            int given_charge = 0;
            int total_charge = 0;
            int total_radical_electrons = 0;
            std::vector<std::string> phase_history;
            mutable MetadataMap metadata;
            int omol_revision = 0;

            mutable std::optional<std::pair<int, std::string>> organic_score_key_cache;
            mutable std::optional<std::pair<int, double>> organic_core_score_cache;
            mutable std::optional<std::pair<int, double>> full_score_cache;
            mutable std::optional<std::pair<int, double>> post_reinsertion_base_symmetry_penalty_cache;
            mutable std::optional<std::pair<int, molgr::ChargedAtomSnapshotList>>
                post_reinsertion_charged_atom_snapshots_cache;
            mutable std::shared_ptr<const PreheatedNoMetalScoreBundle> preheated_score_bundle;

            /// Construct an empty reconstruction state ready to receive its graph or electronic targets.
            OmolStateMachine() = default;

            /// Initialize a mutable ligand search branch with its graph, remaining charge and spin target.
            /// @param omol_ Molecular graph on which the operation acts.
            /// @param given_charge_ Signed charge still to assign to the ligand.
            /// @param phase_history_ Ordered reconstruction stages already applied.
            /// @param metadata_ Reconstruction annotations and cached scoring terms for this state.
            /// @param omol_revision_ Revision identifying the ligand graph used by derived caches.
            /// @param total_charge_ Target net formal charge of the full input.
            /// @param total_radical_electrons_ Target number of unpaired electrons.
            OmolStateMachine(
                std::shared_ptr<OpenBabel::OBMol> omol_,
                int given_charge_ = 0,
                std::vector<std::string> phase_history_ = {},
                MetadataMap metadata_ = {},
                int omol_revision_ = 0,
                int total_charge_ = 0,
                int total_radical_electrons_ = 0);

            /// Restore the mutable search bookkeeping from a ligand snapshot.
            /// @param state Ligand graph and electronic budgets at this stage of reconstruction.
            /// @return Mutable search state initialized from the snapshot.
            static OmolStateMachine FromReconstructionState(const ReconstructionState &state);

            /// @return A writable graph, copying it first when another branch shares ownership.
            OpenBabel::OBMol &EnsureUniqueMol();
            /// Record a named phase in the branch's execution history.
            /// @param phase Stage name recorded in the reconstruction history.
            void Annotate(const std::optional<std::string> &phase);
            /// Record the phase and replace the charge still to be assigned.
            /// @param phase Stage name recorded in the reconstruction history.
            /// @param next_given_charge Signed charge still to assign to the ligand.
            void SetGivenCharge(const std::optional<std::string> &phase, int next_given_charge);

            /// Apply a graph-edit stage and invalidate cached quantities when it reports a change.
            /// @tparam Stage Callable taking the editable molecule and forwarded arguments.
            /// @tparam Args Additional argument types accepted by the stage.
            /// @param phase Stage name recorded in the reconstruction history.
            /// @param stage Reconstruction stage to invoke on the editable ligand graph.
            /// @param args Additional arguments forwarded to the stage.
            /// @return True when the stage reports a graph change.
            template <typename Stage, typename... Args>
            bool RunOmolStage(const std::optional<std::string> &phase, Stage &&stage, Args &&...args)
            {
                OpenBabel::OBMol &mol = EnsureUniqueMol();
                const bool hit = std::invoke(
                    std::forward<Stage>(stage),
                    mol,
                    std::forward<Args>(args)...);
                if (hit)
                {
                    mol.SetHybridizationPerceived(false);
                    mol.SetAtomTypesPerceived(false);
                    ++omol_revision;
                    InvalidateOmolDerivedCache();
                }
                Annotate(phase);
                return hit;
            }

            /// Apply a graph-and-charge stage and track changes to either the graph or remaining budget.
            /// @tparam Stage Callable taking the molecule, charge budget and forwarded arguments.
            /// @tparam Args Additional argument types accepted by the stage.
            /// @param phase Stage name recorded in the reconstruction history.
            /// @param stage Reconstruction stage to invoke on the editable ligand graph.
            /// @param args Additional arguments forwarded to the stage.
            /// @return True when either the molecule or its outstanding charge changed.
            template <typename Stage, typename... Args>
            bool RunOmolChargeStage(
                const std::optional<std::string> &phase,
                Stage &&stage,
                Args &&...args)
            {
                OpenBabel::OBMol &mol = EnsureUniqueMol();
                const int before_charge = given_charge;
                const bool molecule_hit = std::invoke(
                    std::forward<Stage>(stage),
                    mol,
                    given_charge,
                    std::forward<Args>(args)...);
                const bool charge_hit = before_charge != given_charge;
                if (molecule_hit)
                {
                    mol.SetHybridizationPerceived(false);
                    mol.SetAtomTypesPerceived(false);
                    ++omol_revision;
                    InvalidateOmolDerivedCache();
                }
                Annotate(phase);
                return molecule_hit || charge_hit;
            }

            /// Drop scores and topology derived from the previous graph revision.
            void InvalidateOmolDerivedCache();

            /// Fork this search history, optionally replacing the graph and remaining charge.
            /// @param phase Stage name recorded in the reconstruction history.
            /// @param next_omol Molecule to install on the new branch.
            /// @param next_given_charge Signed charge still to assign to the ligand.
            /// @param branch_metadata Annotations carried by the new branch.
            /// @return New branch carrying the updated history and any replacement graph or budget.
            OmolStateMachine Branch(
                const std::optional<std::string> &phase,
                std::shared_ptr<OpenBabel::OBMol> next_omol = nullptr,
                std::optional<int> next_given_charge = std::nullopt,
                MetadataMap branch_metadata = {}) const;

            /// Freeze the current graph and history with the supplied molecular electron targets.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @return Ligand snapshot carrying the specified total charge and radical target.
            ReconstructionState Freeze(int total_charge, int total_radical_electrons) const;
            /// Freeze this branch using the original snapshot's total charge and radical target.
            /// @param state Ligand graph and electronic budgets at this stage of reconstruction.
            /// @return Ligand snapshot carrying the specified total charge and radical target.
            ReconstructionState FreezeLike(const ReconstructionState &state) const;
        };

        /// Detached ligand geometry, candidate charge and spin choices for each metal, and global targets.
        struct MetalPreparationState
        {
            std::string no_metal_xyz_block;
            std::vector<std::vector<molgr::MetalAtomPosition>> available_valence_radical_states;
            int total_charge = 0;
            int total_radical_electrons = 0;
            std::vector<std::string> phase_history;
            mutable MetadataMap metadata;
            std::vector<molgr::MetalSite> metal_sites;
        };

        /// One choice of metal electronic states together with its ligand targets and candidate scores.
        struct MetalCandidateState
        {
            std::vector<molgr::MetalAtomPosition> metal_states;
            int no_metal_charge_target = 0;
            int no_metal_radical_target = 0;
            std::vector<std::string> phase_history;
            mutable MetadataMap metadata;
            std::shared_ptr<ReconstructionState> no_metal_state;
            std::shared_ptr<OpenBabel::OBMol> combined_omol;
            mutable std::optional<double> score;

            mutable std::optional<std::string> metal_state_key_cache;
            mutable std::optional<std::pair<std::string, std::string>> combined_score_key_cache;
            mutable std::optional<std::pair<std::string, double>> combined_score_cache;
            mutable std::optional<std::tuple<const ReconstructionState *, int, std::string>>
                combined_omol_dependency_cache;

            /// @return The fingerprint of the chosen metal charges, spins and positions.
            std::string MetalStateKey() const;
            /// @return The ligand and metal fingerprints used to cache the combined score.
            std::pair<std::string, std::string> CombinedScoreKey() const;
            /// Reuse the assembled graph or call the supplied builder on first access.
            /// @param builder Callback that constructs the combined ligand and metal graph.
            /// @return Shared ownership of the assembled graph.
            std::shared_ptr<OpenBabel::OBMol> MaterializeCombinedOmol(
                const CombinedOmolBuilder &builder);
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return The score for the reassembled metal–ligand candidate.
            double CombinedScore(
                const molgr::config::MolGRConfig &config = molgr::config::GetDefaultConfig()) const;
        };

        /// Track metal assignments and the ligand solution across preparation, scoring and selection.
        class MetalCandidateStateMachine
        {
        public:
            std::vector<molgr::MetalAtomPosition> metal_states;
            int no_metal_charge_target = 0;
            int no_metal_radical_target = 0;
            std::vector<std::string> phase_history;
            MetadataMap metadata;
            std::shared_ptr<ReconstructionState> no_metal_state;
            std::shared_ptr<OpenBabel::OBMol> combined_omol;
            std::optional<double> score;

            std::optional<std::string> metal_state_key_cache;
            std::optional<std::pair<std::string, std::string>> combined_score_key_cache;
            std::optional<std::pair<std::string, double>> combined_score_cache;
            std::optional<std::tuple<const ReconstructionState *, int, std::string>>
                combined_omol_dependency_cache;

            /// Construct an empty reconstruction state ready to receive its graph or electronic targets.
            MetalCandidateStateMachine() = default;

            /// Initialize a metal assignment branch and its residual ligand targets.
            /// @param metal_states_ Proposed valence and unpaired-electron assignments for the metals.
            /// @param no_metal_charge_target_ Formal charge assigned to the detached ligand portion.
            /// @param no_metal_radical_target_ Unpaired-electron budget assigned to the detached ligand
            ///     portion.
            /// @param phase_history_ Ordered reconstruction stages already applied.
            /// @param metadata_ Reconstruction annotations and cached scoring terms for this state.
            explicit MetalCandidateStateMachine(
                std::vector<molgr::MetalAtomPosition> metal_states_,
                int no_metal_charge_target_,
                int no_metal_radical_target_,
                std::vector<std::string> phase_history_ = {},
                MetadataMap metadata_ = {});

            /// Resume mutable bookkeeping from a metal candidate snapshot.
            /// @param state Metal assignment snapshot carrying the ligand solution and its targets.
            /// @return Mutable search state initialized from the snapshot.
            static MetalCandidateStateMachine FromCandidateState(const MetalCandidateState &state);

            /// Record a named phase in the branch's execution history.
            /// @param phase Stage name recorded in the reconstruction history.
            void Annotate(const std::optional<std::string> &phase);
            /// Attach the ligand reconstruction to this metal assignment and record the phase.
            /// @param phase Stage name recorded in the reconstruction history.
            /// @param next_no_metal_state Ligand graph and electronic budgets at this stage of
            ///     reconstruction.
            void SetNoMetalState(
                const std::optional<std::string> &phase,
                std::shared_ptr<ReconstructionState> next_no_metal_state);
            /// Freeze the metal assignment, ligand targets and history into a candidate snapshot.
            /// @return Candidate snapshot with the current ligand solution and metal assignments.
            MetalCandidateState Freeze() const;
        };
    }
}
