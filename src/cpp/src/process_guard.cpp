#include "molgr/process_guard.h"

#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace
{
    /// @return The operating-system process identifier of the caller.
    unsigned long CurrentProcessId()
    {
#if defined(_WIN32)
        return static_cast<unsigned long>(::GetCurrentProcessId());
#else
        return static_cast<unsigned long>(::getpid());
#endif
    }

    /// @return The process identifier stored when the native backend initialized.
    unsigned long &InitializedProcessId()
    {
        static unsigned long process_id = 0;
        return process_id;
    }
}

namespace molgr
{
    void InitializeProcessGuard()
    {
        auto &process_id = InitializedProcessId();
        if (process_id == 0)
        {
            process_id = CurrentProcessId();
        }
    }

    void EnsureCurrentProcess(const char *api)
    {
        if (IsCurrentProcess())
        {
            return;
        }

        throw std::runtime_error(
            std::string(api == nullptr ? "MolGR native API" : api) +
            " cannot run in a forked child after MolGR/Open Babel was initialized; "
            "use multiprocessing start method 'spawn' (or fork before importing MolGR).");
    }

    bool IsCurrentProcess()
    {
        InitializeProcessGuard();
        return CurrentProcessId() == InitializedProcessId();
    }
}
