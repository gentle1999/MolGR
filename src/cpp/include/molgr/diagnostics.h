#pragma once

#include <map>
#include <string>

namespace molgr
{
    namespace diagnostics
    {
        /// Failure code, stage and counters for an unsuccessful native reconstruction.
        struct ReconstructionDiagnostics
        {
            std::string code;
            std::string stage;
            std::string message;
            std::map<std::string, long long> counts;
            std::map<std::string, std::string> details;

            /// Clear the previous failure and its counters before a new reconstruction.
            void Reset()
            {
                code.clear();
                stage.clear();
                message.clear();
                counts.clear();
                details.clear();
            }

            /// Increment the named diagnostic counter for this reconstruction.
            /// @param name Name of the requested field or counter.
            /// @param amount Amount to add to the counter or assignment.
            void Count(const std::string &name, long long amount = 1)
            {
                counts[name] += amount;
            }

            /// Record the failure code, phase and message for the current reconstruction.
            /// @param code_ Machine-readable failure identifier.
            /// @param stage_ Name of the stage that failed.
            /// @param message_ Explanation of the reconstruction failure.
            void Fail(
                const std::string &code_,
                const std::string &stage_,
                const std::string &message_)
            {
                code = code_;
                stage = stage_;
                message = message_;
            }
        };
    }
}
