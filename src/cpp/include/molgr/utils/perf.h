#pragma once

#include <mutex>

namespace molgr
{
    namespace pipeline
    {
        namespace perf
        {
            /// Per-run durations and work counts for ligand search, metal enumeration and force-field
            /// evaluation.
            struct RunTimingBreakdown
            {
                double no_metal_pipeline_ms = 0.0;
                double no_metal_linear_pipeline_ms = 0.0;
                double no_metal_validate_ms = 0.0;
                double resonance_handling_enumeration_ms = 0.0;
                double resonance_walk_ms = 0.0;
                double resonance_prepare_ms = 0.0;
                double resonance_dedup_score_ms = 0.0;
                double resonance_score_ms = 0.0;
                double resonance_topology_ms = 0.0;
                double resonance_raw_candidates = 0.0;
                double resonance_pruned_expansions = 0.0;
                double resonance_prepared_candidates = 0.0;
                double resonance_valid_candidates = 0.0;
                double resonance_dedup_candidates = 0.0;
                double metal_enumeration_combination_ms = 0.0;
                double force_field_total_ms = 0.0;
                double force_field_cache_key_ms = 0.0;
                double force_field_prepare_ms = 0.0;
                double force_field_setup_key_ms = 0.0;
                double force_field_setup_ms = 0.0;
                double force_field_energy_ms = 0.0;
                double force_field_calls = 0.0;
            };

            /// Aggregate timing and work counters from concurrent reconstruction stages.
            class RunTimingReducer
            {
            public:
                /// Accumulate time spent in no metal pipeline.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddNoMetalPipelineMs(double delta_ms);
                /// Accumulate time spent in no metal linear pipeline.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddNoMetalLinearPipelineMs(double delta_ms);
                /// Accumulate time spent in no metal validate.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddNoMetalValidateMs(double delta_ms);
                /// Accumulate time spent in resonance handling enumeration.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddResonanceHandlingEnumerationMs(double delta_ms);
                /// Accumulate time spent in resonance walk.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddResonanceWalkMs(double delta_ms);
                /// Accumulate time spent in resonance prepare.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddResonancePrepareMs(double delta_ms);
                /// Accumulate time spent in resonance dedup score.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddResonanceDedupScoreMs(double delta_ms);
                /// Accumulate time spent in resonance score.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddResonanceScoreMs(double delta_ms);
                /// Accumulate time spent in resonance topology.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddResonanceTopologyMs(double delta_ms);
                /// Accumulate the number of resonance raw candidates.
                /// @param delta_count Number of additional events to record.
                void AddResonanceRawCandidates(double delta_count);
                /// Accumulate the number of resonance pruned expansions.
                /// @param delta_count Number of additional events to record.
                void AddResonancePrunedExpansions(double delta_count);
                /// Accumulate the number of resonance prepared candidates.
                /// @param delta_count Number of additional events to record.
                void AddResonancePreparedCandidates(double delta_count);
                /// Accumulate the number of resonance valid candidates.
                /// @param delta_count Number of additional events to record.
                void AddResonanceValidCandidates(double delta_count);
                /// Accumulate the number of resonance dedup candidates.
                /// @param delta_count Number of additional events to record.
                void AddResonanceDedupCandidates(double delta_count);
                /// Accumulate time spent in metal enumeration combination.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddMetalEnumerationCombinationMs(double delta_ms);
                /// Accumulate time spent in force field total.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddForceFieldTotalMs(double delta_ms);
                /// Accumulate time spent in force field cache key.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddForceFieldCacheKeyMs(double delta_ms);
                /// Accumulate time spent in force field prepare.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddForceFieldPrepareMs(double delta_ms);
                /// Accumulate time spent in force field setup key.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddForceFieldSetupKeyMs(double delta_ms);
                /// Accumulate time spent in force field setup.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddForceFieldSetupMs(double delta_ms);
                /// Accumulate time spent in force field energy.
                /// @param delta_ms Elapsed time to add, in milliseconds.
                void AddForceFieldEnergyMs(double delta_ms);
                /// Accumulate the number of force field calls.
                /// @param delta_count Number of additional events to record.
                void AddForceFieldCalls(double delta_count);
                /// @return A consistent snapshot of accumulated durations and counts.
                RunTimingBreakdown Snapshot() const;

            private:
                mutable std::mutex mutex_;
                RunTimingBreakdown timing_;
            };

            /// Install a run-level timing reducer for the scope's lifetime.
            class RunTimingScope
            {
            public:
                /// Install a run-level timing reducer for the scope's lifetime.
                RunTimingScope();
                /// @return The timing reducer owned by this scope.
                RunTimingReducer &Reducer();
                /// @return The timing reducer owned by this scope.
                const RunTimingReducer &Reducer() const;
                /// Restore the prior timing context on scope exit.
                ~RunTimingScope();

            private:
                RunTimingReducer reducer_;
                RunTimingReducer *previous_ = nullptr;
            };

            /// Temporarily route timing observations to a supplied reducer on this thread.
            class ActiveRunTimingReducerScope
            {
            public:
                /// Temporarily route timing observations to a supplied reducer on this thread.
                /// @param reducer Run-level collector for timing and work counters.
                explicit ActiveRunTimingReducerScope(RunTimingReducer *reducer);
                /// Restore the previously active thread-local reducer.
                ~ActiveRunTimingReducerScope();

            private:
                RunTimingReducer *previous_ = nullptr;
            };

            /// @return The current thread's reducer, or null outside an instrumented run.
            RunTimingReducer *GetActiveRunTimingReducer();
            /// @return The timing breakdown for the calling thread.
            RunTimingBreakdown GetRunTimingBreakdown();
            /// Replace the calling thread's recorded timing breakdown.
            /// @param timing Timing breakdown to store for the calling thread.
            void SetRunTimingBreakdown(const RunTimingBreakdown &timing);
        }
    }
}
