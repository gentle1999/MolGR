#pragma once

#include "molgr/config.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace molgr
{
    namespace utils
    {
        namespace parallel
        {
            /// @return The hardware worker capacity used to size native parallel jobs.
            inline std::size_t HardwareParallelism()
            {
                const unsigned int concurrency = std::thread::hardware_concurrency();
                return std::max<std::size_t>(
                    1,
                    concurrency == 0 ? 1 : static_cast<std::size_t>(concurrency));
            }

            /// @param config Settings for the current reconstruction or scoring operation.
            /// @param task_count Number of work items or occurrences.
            /// @return The worker capacity after applying the configured thread limit.
            inline std::size_t ConfiguredParallelism(
                const molgr::config::MolGRConfig &config,
                std::size_t task_count)
            {
                if (task_count == 0)
                {
                    return 1;
                }
                std::size_t worker_count = HardwareParallelism();
                if (config.cpp_backend.max_threads.has_value())
                {
                    worker_count = std::min(
                        worker_count,
                        static_cast<std::size_t>(std::max(1, *config.cpp_backend.max_threads)));
                }
                return std::max<std::size_t>(1, std::min(worker_count, task_count));
            }

            namespace detail
            {
                /// Shared native executor used by reconstruction and candidate-scoring loops.
                class ParallelExecutor
                {
                public:
                    /// @return The shared native executor used to schedule helper tasks.
                    static ParallelExecutor &Instance()
                    {
                        // Keep one fixed native worker budget for both single-
                        // molecule helper work and asynchronous batch items.
                        // A synchronous Run still counts its calling thread,
                        // so it uses at most worker_count - 1 helpers.
                        static ParallelExecutor *instance =
                            new ParallelExecutor(HardwareParallelism());
                        return *instance;
                    }

                    /// @return The number of helper threads available to a calling thread.
                    std::size_t HelperCapacity() const
                    {
                        return workers_.size();
                    }

                    /// Queue a task for execution by the native worker pool.
                    /// @param task Callable executed by the parallel worker.
                    void Submit(std::function<void()> task)
                    {
                        {
                            std::lock_guard<std::mutex> lock(queue_mutex_);
                            if (shutting_down_)
                            {
                                throw std::runtime_error(
                                    "The MolGR native executor is shutting down.");
                            }
                            queue_.push_back(std::move(task));
                        }
                        queue_cv_.notify_one();
                    }

                    /// Run the indexed work with bounded parallelism and propagate worker exceptions.
                    /// @tparam Func Callable accepting one zero-based work index.
                    /// @param count Number of work items or occurrences.
                    /// @param worker_count Maximum workers participating in this operation.
                    /// @param func Callable executed by the parallel worker.
                    template <typename Func>
                    void Run(std::size_t count, std::size_t worker_count, Func &&func)
                    {
                        if (count == 0)
                        {
                            return;
                        }

                        worker_count = std::max<std::size_t>(1, std::min(worker_count, count));
                        const std::size_t helper_count = std::min<std::size_t>(
                            worker_count > 0 ? worker_count - 1 : 0,
                            workers_.size());
                        if (helper_count == 0)
                        {
                            for (std::size_t idx = 0; idx < count; ++idx)
                            {
                                func(idx);
                            }
                            return;
                        }

                        auto job = std::make_shared<Job>(
                            count,
                            std::function<void(std::size_t)>(std::forward<Func>(func)),
                            helper_count + 1);

                        {
                            std::lock_guard<std::mutex> lock(queue_mutex_);
                            for (std::size_t worker_idx = 0; worker_idx < helper_count; ++worker_idx)
                            {
                                queue_.push_back(
                                    [job]()
                                    {
                                        ExecuteJob(job);
                                    });
                            }
                        }
                        queue_cv_.notify_all();

                        ExecuteJob(job);

                        std::unique_lock<std::mutex> lock(job->done_mutex);
                        job->done_cv.wait(
                            lock,
                            [&job]()
                            {
                                return job->done;
                            });
                        if (job->first_exception)
                        {
                            std::rethrow_exception(job->first_exception);
                        }
                    }

                private:
                    /// Work function and shared progress and failure state for a parallel range.
                    struct Job
                    {
                        /// Initialize the range, work callback and completion bookkeeping for a parallel
                        /// job.
                        /// @param count_in Number of work items or occurrences.
                        /// @param func_in Callable executed by the parallel worker.
                        /// @param worker_count_in Maximum workers participating in this operation.
                        Job(
                            std::size_t count_in,
                            std::function<void(std::size_t)> func_in,
                            std::size_t worker_count_in)
                            : count(count_in),
                              func(std::move(func_in)),
                              remaining_workers(worker_count_in)
                        {
                        }

                        std::size_t count;
                        std::function<void(std::size_t)> func;
                        std::atomic<std::size_t> next_index{0};
                        std::atomic<std::size_t> remaining_workers;
                        std::atomic<bool> cancel{false};
                        std::mutex done_mutex;
                        std::condition_variable done_cv;
                        bool done = false;
                        std::mutex exception_mutex;
                        std::exception_ptr first_exception;
                    };

                    /// Start the executor's helper threads.
                    /// @param helper_count Number of background worker threads to start.
                    explicit ParallelExecutor(std::size_t helper_count)
                    {
                        workers_.reserve(helper_count);
                        for (std::size_t worker_idx = 0; worker_idx < helper_count; ++worker_idx)
                        {
                            workers_.emplace_back(
                                [this]()
                                {
                                    WorkerLoop();
                                });
                        }
                    }

                    /// Signal shutdown and wait for all helper threads to exit.
                    ~ParallelExecutor()
                    {
                        {
                            std::lock_guard<std::mutex> lock(queue_mutex_);
                            shutting_down_ = true;
                        }
                        queue_cv_.notify_all();
                        for (auto &worker : workers_)
                        {
                            if (worker.joinable())
                            {
                                worker.join();
                            }
                        }
                    }

                    /// Consume queued work until the executor is stopped.
                    void WorkerLoop()
                    {
                        while (true)
                        {
                            std::function<void()> task;
                            {
                                std::unique_lock<std::mutex> lock(queue_mutex_);
                                queue_cv_.wait(
                                    lock,
                                    [this]()
                                    {
                                        return shutting_down_ || !queue_.empty();
                                    });
                                if (shutting_down_ && queue_.empty())
                                {
                                    return;
                                }
                                task = std::move(queue_.front());
                                queue_.pop_front();
                            }
                            // Executor-owned tasks are required to contain
                            // their own exception propagation. Keep the pool
                            // alive if an unexpected task violates that
                            // contract instead of terminating the process.
                            try
                            {
                                task();
                            }
                            catch (...)
                            {
                            }
                        }
                    }

                    /// Execute available indices and notify peers when the job completes or fails.
                    /// @param job Shared work and completion state for this indexed operation.
                    static void ExecuteJob(const std::shared_ptr<Job> &job)
                    {
                        try
                        {
                            while (!job->cancel.load(std::memory_order_relaxed))
                            {
                                const std::size_t idx =
                                    job->next_index.fetch_add(1, std::memory_order_relaxed);
                                if (idx >= job->count)
                                {
                                    break;
                                }
                                job->func(idx);
                            }
                        }
                        catch (...)
                        {
                            std::lock_guard<std::mutex> lock(job->exception_mutex);
                            if (!job->first_exception)
                            {
                                job->first_exception = std::current_exception();
                                job->cancel.store(true, std::memory_order_relaxed);
                            }
                        }

                        if (job->remaining_workers.fetch_sub(1, std::memory_order_acq_rel) == 1)
                        {
                            std::lock_guard<std::mutex> lock(job->done_mutex);
                            job->done = true;
                            job->done_cv.notify_one();
                        }
                    }

                    std::mutex queue_mutex_;
                    std::condition_variable queue_cv_;
                    std::deque<std::function<void()>> queue_;
                    std::vector<std::thread> workers_;
                    bool shutting_down_ = false;
                };
            }

            /// Apply the indexed operation with the configured parallelism limit.
            /// @tparam Func Callable accepting one zero-based work index.
            /// @param count Number of work items or occurrences.
            /// @param worker_count Maximum workers participating in this operation.
            /// @param func Callable executed by the parallel worker.
            template <typename Func>
            void ParallelForIndices(std::size_t count, std::size_t worker_count, Func &&func)
            {
                if (count == 0)
                {
                    return;
                }

                worker_count = std::max<std::size_t>(1, std::min(worker_count, count));
                if (worker_count == 1)
                {
                    for (std::size_t idx = 0; idx < count; ++idx)
                    {
                        func(idx);
                    }
                    return;
                }

                detail::ParallelExecutor::Instance().Run(
                    count,
                    worker_count,
                    std::forward<Func>(func));
            }
        }
    }
}
