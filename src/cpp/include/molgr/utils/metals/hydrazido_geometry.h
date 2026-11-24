#pragma once

#include "molgr/state.h"
#include <vector>

namespace molgr::metal::hydrazido_geometry
{
    /// Remove the hydrazido contribution before comparing a new set of metal assignments.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void Clear(std::vector<molgr::state::MetalCandidateState> &candidates);
    /// Compare isodiazene and hydrazido at fixed spin. Each reduced site requires
    /// a two-unit increase in metal charge and identical labels elsewhere.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void Annotate(std::vector<molgr::state::MetalCandidateState> &candidates);
}
