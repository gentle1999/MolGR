#pragma once

#include "molgr/config.h"
#include "molgr/diagnostics.h"
#include "molgr/utils/utils.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace molgr::pipeline::reconstruct_batch
{
    /// XYZ geometry, net charge and target unpaired-electron count for one input.
    struct ReconstructionBatchRequest
    {
        std::string xyz_block;
        int total_charge = 0;
        int total_radical_electrons = 0;
    };

    /// Input index, optional reconstructed graph and diagnostics for one request.
    struct ReconstructionBatchResult
    {
        std::size_t index = 0;
        std::unique_ptr<molgr::utils::MoleculeData> molecule;
        molgr::diagnostics::ReconstructionDiagnostics diagnostics;
    };

    /// Stream indexed reconstruction results from the native workers. Close the iterator to stop outstanding work.
    class ReconstructionBatchIterator
    {
    public:
        /// Start a bounded batch with the requested ordering and worker limits.
        /// @param requests Requests in original input order.
        /// @param config Settings for the current reconstruction or scoring operation.
        /// @param max_workers Maximum workers participating in this operation.
        /// @param queue_size Bound on outstanding batch results.
        /// @param ordered Whether to yield results in input order.
        ReconstructionBatchIterator(
            std::vector<ReconstructionBatchRequest> requests,
            const molgr::config::MolGRConfig &config,
            std::size_t max_workers,
            std::size_t queue_size,
            bool ordered);
        /// Release the batch implementation and wait for active work to finish.
        ~ReconstructionBatchIterator();

        /// Disable copying because one iterator owns the batch progress and worker lifetime.
        ReconstructionBatchIterator(const ReconstructionBatchIterator &) = delete;
        /// Disable copying because one iterator owns the batch progress and worker lifetime.
        ReconstructionBatchIterator &operator=(const ReconstructionBatchIterator &) = delete;

        /// Wait for one result, or return an empty optional when the stream is exhausted.
        /// @return Next indexed result, or an empty optional when exhausted.
        std::optional<ReconstructionBatchResult> Next();
        /// Stop scheduling requests and wait for active tasks to finish.
        void Close();

    private:
        /// Internal scheduling state for request indices, bounded results and worker synchronization.
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
