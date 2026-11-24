#pragma once

#include "molgr/config.h"

#include <pybind11/pybind11.h>

namespace molgr::config
{
    /// Read the Python configuration object into a native configuration for use without the GIL.
    /// @param config Settings for the current reconstruction or scoring operation.
    /// @return Native settings copied from the Python object.
    MolGRConfig FromPython(pybind11::handle config);
}
