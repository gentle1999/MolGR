#pragma once
#include "molgr/utils/utils.h"

namespace molgr::equivalence
{
    /// Check whether a stereo frame survives the requested coordination projection.
    /// @param data Transferable molecular graph records.
    /// @param stereo Stereo frame and its ordered ligand references.
    /// @param ignore_metal_coordination Whether to project away metal coordination before evaluating
    ///     stereo.
    /// @return True when the specified projection retains active stereochemical information.
    bool ActiveStereo(const utils::MoleculeData &data, const utils::StereoData &stereo,
                      bool ignore_metal_coordination);
    /// Check whether any stored stereo frame survives the requested coordination projection.
    /// @param data Transferable molecular graph records.
    /// @param ignore_metal_coordination Whether to project away metal coordination before evaluating
    ///     stereo.
    /// @return True when the specified projection retains active stereochemical information.
    bool HasActiveStereo(const utils::MoleculeData &data, bool ignore_metal_coordination);

    /// Compare the specified stereo under the supplied atom mapping and coordination projection.
    /// @param left First input molecular graph.
    /// @param right Second input molecular graph.
    /// @param mapping Zero-based image of each source atom in the target graph.
    /// @param ignore_metal_coordination Whether to project away metal coordination before evaluating
    ///     stereo.
    /// @return 1 for compatible stereo, 0 for a contradiction, or -1 for insufficient information.
    int CompareStereo(const utils::MoleculeData &left, const utils::MoleculeData &right,
                      const std::vector<int> &mapping, bool ignore_metal_coordination);
}
