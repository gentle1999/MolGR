#pragma once

#include <iostream>
namespace molgr
{
    /// Severity threshold for messages emitted by the native algorithms.
    enum class LogLevel
    {
        DEBUG = 0,
        INFO = 1,
        WARN = 2,
        ERROR = 3,
        OFF = 4
    };

    extern LogLevel g_current_log_level;

    /// Set C++ backend logging level.
    /// @param level Minimum severity reported by the native logger.
    void SetLogLevel(LogLevel level);

    /// Emit a process-deduplicated warning for APIs that expose raw Open Babel
    /// objects. These objects are not safe to share across threads and have
    /// manual lifetime rules.
    /// @param api Name of the calling API for the diagnostic message.
    /// @param detail Additional context for the Open Babel warning.
    void WarnUnsafeOpenBabelUse(const char *api, const char *detail);
}

#define LOG_DEBUG(msg)                                            \
    do                                                            \
    {                                                             \
        if (molgr::g_current_log_level <= molgr::LogLevel::DEBUG) \
        {                                                         \
            std::cerr << "[DEBUG] " << msg << std::endl;          \
        }                                                         \
    } while (0)

#define LOG_INFO(msg)                                            \
    do                                                           \
    {                                                            \
        if (molgr::g_current_log_level <= molgr::LogLevel::INFO) \
        {                                                        \
            std::cerr << "[INFO]  " << msg << std::endl;         \
        }                                                        \
    } while (0)

#define LOG_WARN(msg)                                            \
    do                                                           \
    {                                                            \
        if (molgr::g_current_log_level <= molgr::LogLevel::WARN) \
        {                                                        \
            std::cerr << "[WARN]  " << msg << std::endl;         \
        }                                                        \
    } while (0)

#define LOG_ERROR(msg)                                            \
    do                                                            \
    {                                                             \
        if (molgr::g_current_log_level <= molgr::LogLevel::ERROR) \
        {                                                         \
            std::cerr << "[ERROR] " << msg << std::endl;          \
        }                                                         \
    } while (0)
