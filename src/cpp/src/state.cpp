#include "molgr/state.h"

#include "molgr/utils/conversions.h"
#include "molgr/utils/force_field.h"
#include "molgr/utils/scoring.h"
#include "molgr/vendor/openbabel_copy.h"

#include <array>
#include <stdexcept>
#include <utility>

namespace
{
    constexpr std::array<const char *, 17> kOmolDerivedMetadataKeys = {
        "force_field_energy",
        "force_field_score_key",
        "organic_core_score",
        "score",
        "organic_aromatic_atom_count",
        "organic_aromatic_ring_count",
        "organic_aromatic_stability_score",
        "organic_conjugated_atom_count",
        "organic_conjugated_bond_count",
        "organic_max_conjugated_component_size",
        "organic_hyperconjugative_donor_count",
        "organic_hyperconjugation_score",
        "organic_formal_charge_absolute_sum",
        "organic_conjugation_charge_penalty",
        "organic_adjusted_max_conjugated_component_size",
        "organic_adjusted_conjugated_atom_count",
        "organic_adjusted_conjugated_bond_count",
    };

    constexpr std::array<const char *, 2> kCandidateDerivedMetadataKeys = {
        "force_field_energy",
        "score",
    };

    void InvalidateOmolDerivedMetadata(molgr::state::MetadataMap &metadata)
    {
        for (const char *key : kOmolDerivedMetadataKeys)
        {
            metadata.erase(key);
        }
    }

    void InvalidateCandidateDerivedMetadata(molgr::state::MetadataMap &metadata)
    {
        for (const char *key : kCandidateDerivedMetadataKeys)
        {
            metadata.erase(key);
        }
    }

    void StoreNoMetalForceFieldMetadata(
        molgr::state::MetadataMap &metadata,
        const std::string &score_key,
        double score)
    {
        metadata["force_field_energy"] = score;
        metadata["force_field_score_key"] = score_key;
        metadata["organic_core_score"] = score;
        metadata["score"] = score;
    }

}

namespace molgr
{
    namespace state
    {
        ReconstructionState::ReconstructionState(
            std::shared_ptr<OpenBabel::OBMol> omol_,
            int given_charge_,
            int total_charge_,
            int total_radical_electrons_,
            std::vector<std::string> phase_history_,
            MetadataMap metadata_,
            int omol_revision_)
            : omol(std::move(omol_)),
              given_charge(given_charge_),
              total_charge(total_charge_),
              total_radical_electrons(total_radical_electrons_),
              phase_history(std::move(phase_history_)),
              metadata(std::move(metadata_)),
              omol_revision(omol_revision_)
        {
        }

        const OpenBabel::OBMol &ReconstructionState::Mol() const
        {
            if (!omol)
            {
                throw std::runtime_error("ReconstructionState has no OBMol");
            }
            return *omol;
        }

        OpenBabel::OBMol &ReconstructionState::MutableMol()
        {
            if (!omol)
            {
                throw std::runtime_error("ReconstructionState has no OBMol");
            }
            return *omol;
        }

        void ReconstructionState::InvalidateOmolDerivedCache()
        {
            organic_score_key_cache.reset();
            organic_core_score_cache.reset();
            full_score_cache.reset();
            post_reinsertion_base_symmetry_penalty_cache.reset();
            post_reinsertion_charged_atom_snapshots_cache.reset();
            preheated_score_bundle.reset();
            InvalidateOmolDerivedMetadata(metadata);
        }

        PreheatedNoMetalScoreBundle ReconstructionState::BuildPreheatedScoreBundle(
            const molgr::config::MolGRConfig &config) const
        {
            PreheatedNoMetalScoreBundle bundle;
            bundle.post_reinsertion_base_key = molgr::scoring::BuildScoreKey(Mol());
            const auto evaluation = molgr::scoring::OrganicForceFieldEvaluationWithScoreKey(
                Mol(),
                bundle.post_reinsertion_base_key,
                config);
            bundle.organic_core_score = evaluation.energy_kj_mol;
            const auto base_components = molgr::scoring::BuildPostReinsertionBaseComponents(Mol());
            bundle.base_symmetry_penalty = base_components.first;
            bundle.charged_atom_snapshots = base_components.second;
            return bundle;
        }

        void ReconstructionState::PreheatScoreBundle(const molgr::config::MolGRConfig &config)
        {
            auto bundle = std::make_shared<const PreheatedNoMetalScoreBundle>(
                BuildPreheatedScoreBundle(config));
            StoreNoMetalForceFieldMetadata(
                metadata,
                bundle->post_reinsertion_base_key,
                bundle->organic_core_score);
            preheated_score_bundle = std::move(bundle);
        }

        const PreheatedNoMetalScoreBundle *ReconstructionState::PreheatedScoreBundle() const
        {
            return preheated_score_bundle.get();
        }

        std::string ReconstructionState::PostReinsertionBaseKey() const
        {
            if (preheated_score_bundle)
            {
                return preheated_score_bundle->post_reinsertion_base_key;
            }
            if (organic_score_key_cache.has_value() && organic_score_key_cache->first == omol_revision)
            {
                return organic_score_key_cache->second;
            }
            const std::string score_key = molgr::scoring::BuildScoreKey(Mol());
            organic_score_key_cache = std::make_pair(omol_revision, score_key);
            return score_key;
        }

        double ReconstructionState::OrganicCoreScore(const molgr::config::MolGRConfig &config) const
        {
            if (preheated_score_bundle)
            {
                StoreNoMetalForceFieldMetadata(
                    metadata,
                    preheated_score_bundle->post_reinsertion_base_key,
                    preheated_score_bundle->organic_core_score);
                return preheated_score_bundle->organic_core_score;
            }
            if (organic_core_score_cache.has_value() && organic_core_score_cache->first == omol_revision)
            {
                metadata["organic_core_score"] = organic_core_score_cache->second;
                metadata["score"] = organic_core_score_cache->second;
                return organic_core_score_cache->second;
            }
            const std::string score_key = PostReinsertionBaseKey();
            const auto evaluation = molgr::scoring::OrganicForceFieldEvaluationWithScoreKey(
                Mol(),
                score_key,
                config);
            const double score = evaluation.energy_kj_mol;
            organic_core_score_cache = std::make_pair(omol_revision, score);
            StoreNoMetalForceFieldMetadata(
                metadata,
                score_key,
                score);
            return score;
        }

        std::pair<double, molgr::ChargedAtomSnapshotList>
        ReconstructionState::PostReinsertionBaseComponents() const
        {
            if (preheated_score_bundle)
            {
                return {
                    preheated_score_bundle->base_symmetry_penalty,
                    preheated_score_bundle->charged_atom_snapshots,
                };
            }
            if (post_reinsertion_base_symmetry_penalty_cache.has_value() &&
                post_reinsertion_charged_atom_snapshots_cache.has_value() &&
                post_reinsertion_base_symmetry_penalty_cache->first == omol_revision &&
                post_reinsertion_charged_atom_snapshots_cache->first == omol_revision)
            {
                return {
                    post_reinsertion_base_symmetry_penalty_cache->second,
                    post_reinsertion_charged_atom_snapshots_cache->second,
                };
            }

            auto components = molgr::scoring::BuildPostReinsertionBaseComponents(Mol());
            post_reinsertion_base_symmetry_penalty_cache = std::make_pair(omol_revision, components.first);
            post_reinsertion_charged_atom_snapshots_cache = std::make_pair(omol_revision, components.second);
            return components;
        }

        double ReconstructionState::FullScore(const molgr::config::MolGRConfig &config) const
        {
            if (full_score_cache.has_value() && full_score_cache->first == omol_revision)
            {
                metadata["score"] = full_score_cache->second;
                return full_score_cache->second;
            }
            const double score = OrganicCoreScore(config);
            full_score_cache = std::make_pair(omol_revision, score);
            metadata["score"] = score;
            return score;
        }

        OmolStateMachine::OmolStateMachine(
            std::shared_ptr<OpenBabel::OBMol> omol_,
            int given_charge_,
            std::vector<std::string> phase_history_,
            MetadataMap metadata_,
            int omol_revision_,
            int total_charge_,
            int total_radical_electrons_)
            : omol(std::move(omol_)),
              given_charge(given_charge_),
              total_charge(total_charge_),
              total_radical_electrons(total_radical_electrons_),
              phase_history(std::move(phase_history_)),
              metadata(std::move(metadata_)),
              omol_revision(omol_revision_)
        {
        }

        OmolStateMachine OmolStateMachine::FromReconstructionState(const ReconstructionState &state)
        {
            OmolStateMachine machine(
                state.omol,
                state.given_charge,
                state.phase_history,
                state.metadata,
                state.omol_revision,
                state.total_charge,
                state.total_radical_electrons);
            machine.organic_score_key_cache = state.organic_score_key_cache;
            machine.organic_core_score_cache = state.organic_core_score_cache;
            machine.full_score_cache = state.full_score_cache;
            machine.post_reinsertion_base_symmetry_penalty_cache =
                state.post_reinsertion_base_symmetry_penalty_cache;
            machine.post_reinsertion_charged_atom_snapshots_cache =
                state.post_reinsertion_charged_atom_snapshots_cache;
            machine.preheated_score_bundle = state.preheated_score_bundle;
            return machine;
        }

        OpenBabel::OBMol &OmolStateMachine::EnsureUniqueMol()
        {
            if (!omol)
            {
                omol = std::make_shared<OpenBabel::OBMol>();
            }
            else if (!omol.unique())
            {
                omol = molgr::utils::CloneMolTopologyShared(*omol);
            }
            return *omol;
        }

        void OmolStateMachine::Annotate(const std::optional<std::string> &phase)
        {
            if (phase.has_value())
            {
                phase_history.push_back(*phase);
            }
        }

        void OmolStateMachine::SetGivenCharge(
            const std::optional<std::string> &phase,
            int next_given_charge)
        {
            given_charge = next_given_charge;
            Annotate(phase);
        }

        void OmolStateMachine::InvalidateOmolDerivedCache()
        {
            organic_score_key_cache.reset();
            organic_core_score_cache.reset();
            full_score_cache.reset();
            post_reinsertion_base_symmetry_penalty_cache.reset();
            post_reinsertion_charged_atom_snapshots_cache.reset();
            preheated_score_bundle.reset();
            InvalidateOmolDerivedMetadata(metadata);
        }

        OmolStateMachine OmolStateMachine::Branch(
            const std::optional<std::string> &phase,
            std::shared_ptr<OpenBabel::OBMol> next_omol,
            std::optional<int> next_given_charge,
            MetadataMap branch_metadata) const
        {
            const bool replaces_omol = static_cast<bool>(next_omol);
            std::shared_ptr<OpenBabel::OBMol> branch_omol;
            if (replaces_omol)
            {
                branch_omol = std::move(next_omol);
            }
            else
            {
                branch_omol = omol;
            }
            OmolStateMachine machine(
                std::move(branch_omol),
                next_given_charge.has_value() ? *next_given_charge : given_charge,
                phase_history,
                metadata,
                replaces_omol ? omol_revision + 1 : omol_revision,
                total_charge,
                total_radical_electrons);

            if (!replaces_omol)
            {
                machine.organic_score_key_cache = organic_score_key_cache;
                machine.organic_core_score_cache = organic_core_score_cache;
                machine.full_score_cache = full_score_cache;
                machine.post_reinsertion_base_symmetry_penalty_cache =
                    post_reinsertion_base_symmetry_penalty_cache;
                machine.post_reinsertion_charged_atom_snapshots_cache =
                    post_reinsertion_charged_atom_snapshots_cache;
                machine.preheated_score_bundle = preheated_score_bundle;
            }
            else
            {
                auto &branch_mol = machine.EnsureUniqueMol();
                branch_mol.SetHybridizationPerceived(false);
                branch_mol.SetAtomTypesPerceived(false);
                machine.InvalidateOmolDerivedCache();
            }

            for (auto &[key, value] : branch_metadata)
            {
                machine.metadata[key] = std::move(value);
            }
            machine.Annotate(phase);
            return machine;
        }

        ReconstructionState OmolStateMachine::Freeze(
            int total_charge,
            int total_radical_electrons) const
        {
            ReconstructionState state(
                omol,
                given_charge,
                total_charge,
                total_radical_electrons,
                phase_history,
                metadata,
                omol_revision);
            state.organic_score_key_cache = organic_score_key_cache;
            state.organic_core_score_cache = organic_core_score_cache;
            state.full_score_cache = full_score_cache;
            state.post_reinsertion_base_symmetry_penalty_cache =
                post_reinsertion_base_symmetry_penalty_cache;
            state.post_reinsertion_charged_atom_snapshots_cache =
                post_reinsertion_charged_atom_snapshots_cache;
            state.preheated_score_bundle = preheated_score_bundle;
            return state;
        }

        ReconstructionState OmolStateMachine::FreezeLike(const ReconstructionState &state) const
        {
            return Freeze(state.total_charge, state.total_radical_electrons);
        }

        std::string MetalCandidateState::MetalStateKey() const
        {
            if (metal_state_key_cache.has_value())
            {
                return *metal_state_key_cache;
            }
            metal_state_key_cache = molgr::scoring::BuildMetalStateKey(metal_states);
            return *metal_state_key_cache;
        }

        std::pair<std::string, std::string> MetalCandidateState::CombinedScoreKey() const
        {
            if (!no_metal_state)
            {
                throw std::runtime_error("MetalCandidateState requires a no-metal state before scoring");
            }
            if (combined_score_key_cache.has_value())
            {
                return *combined_score_key_cache;
            }

            combined_score_key_cache = std::make_pair(
                no_metal_state->PostReinsertionBaseKey(),
                MetalStateKey());
            return *combined_score_key_cache;
        }

        std::shared_ptr<OpenBabel::OBMol> MetalCandidateState::MaterializeCombinedOmol(
            const CombinedOmolBuilder &builder)
        {
            if (!no_metal_state)
            {
                throw std::runtime_error("MetalCandidateState requires a no-metal state before materialization");
            }
            const auto dependency = std::make_tuple(
                no_metal_state.get(),
                no_metal_state->omol_revision,
                MetalStateKey());
            if (combined_omol && combined_omol_dependency_cache.has_value() &&
                *combined_omol_dependency_cache == dependency)
            {
                return combined_omol;
            }

            combined_omol = builder(no_metal_state->Mol(), metal_states);
            combined_omol_dependency_cache = dependency;
            return combined_omol;
        }

        double MetalCandidateState::CombinedScore(const molgr::config::MolGRConfig &config) const
        {
            if (!no_metal_state)
            {
                throw std::runtime_error("MetalCandidateState requires a no-metal state before scoring");
            }
            const auto combined_key = CombinedScoreKey();
            const std::string cache_key = combined_key.first + "\n--metal--\n" + combined_key.second;
            if (combined_score_cache.has_value() && combined_score_cache->first == cache_key)
            {
                metadata["score"] = combined_score_cache->second;
                return combined_score_cache->second;
            }

            std::optional<PreheatedNoMetalScoreBundle> local_bundle;
            const auto *bundle = no_metal_state->PreheatedScoreBundle();
            if (bundle == nullptr)
            {
                local_bundle = no_metal_state->BuildPreheatedScoreBundle(config);
                bundle = &*local_bundle;
            }
            const double score_value = bundle->organic_core_score;
            combined_score_cache = std::make_pair(cache_key, score_value);
            metadata["force_field_energy"] = score_value;
            metadata["score"] = score_value;
            score = score_value;
            return score_value;
        }

        MetalCandidateStateMachine::MetalCandidateStateMachine(
            std::vector<molgr::MetalAtomPosition> metal_states_,
            int no_metal_charge_target_,
            int no_metal_radical_target_,
            std::vector<std::string> phase_history_,
            MetadataMap metadata_)
            : metal_states(std::move(metal_states_)),
              no_metal_charge_target(no_metal_charge_target_),
              no_metal_radical_target(no_metal_radical_target_),
              phase_history(std::move(phase_history_)),
              metadata(std::move(metadata_))
        {
        }

        MetalCandidateStateMachine MetalCandidateStateMachine::FromCandidateState(
            const MetalCandidateState &state)
        {
            MetalCandidateStateMachine machine(
                state.metal_states,
                state.no_metal_charge_target,
                state.no_metal_radical_target,
                state.phase_history,
                state.metadata);
            machine.no_metal_state = state.no_metal_state;
            machine.combined_omol = state.combined_omol;
            machine.score = state.score;
            machine.metal_state_key_cache = state.metal_state_key_cache;
            machine.combined_score_key_cache = state.combined_score_key_cache;
            machine.combined_score_cache = state.combined_score_cache;
            machine.combined_omol_dependency_cache = state.combined_omol_dependency_cache;
            return machine;
        }

        void MetalCandidateStateMachine::Annotate(const std::optional<std::string> &phase)
        {
            if (phase.has_value())
            {
                phase_history.push_back(*phase);
            }
        }

        void MetalCandidateStateMachine::SetNoMetalState(
            const std::optional<std::string> &phase,
            std::shared_ptr<ReconstructionState> next_no_metal_state)
        {
            if (no_metal_state != next_no_metal_state)
            {
                InvalidateCandidateDerivedMetadata(metadata);
                combined_omol.reset();
                score.reset();
                combined_score_key_cache.reset();
                combined_score_cache.reset();
                combined_omol_dependency_cache.reset();
            }
            no_metal_state = std::move(next_no_metal_state);
            Annotate(phase);
        }

        MetalCandidateState MetalCandidateStateMachine::Freeze() const
        {
            MetalCandidateState state;
            state.metal_states = metal_states;
            state.no_metal_charge_target = no_metal_charge_target;
            state.no_metal_radical_target = no_metal_radical_target;
            state.phase_history = phase_history;
            state.metadata = metadata;
            state.no_metal_state = no_metal_state;
            state.combined_omol = combined_omol;
            state.score = score;
            state.metal_state_key_cache = metal_state_key_cache;
            state.combined_score_key_cache = combined_score_key_cache;
            state.combined_score_cache = combined_score_cache;
            state.combined_omol_dependency_cache = combined_omol_dependency_cache;
            return state;
        }
    }
}
