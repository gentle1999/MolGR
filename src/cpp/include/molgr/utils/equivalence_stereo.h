#pragma once
#include "molgr/utils/utils.h"

namespace molgr::equivalence
{
    // Inputs must already have passed structural validation.
    bool ActiveStereo(const utils::MoleculeData &data, const utils::StereoData &stereo,
                      bool ignore_metal_coordination);
    bool HasActiveStereo(const utils::MoleculeData &data, bool ignore_metal_coordination);

    // 1 = compatible; 0 = specified contradiction; -1 = insufficient information.
    int CompareStereo(const utils::MoleculeData &left, const utils::MoleculeData &right,
                      const std::vector<int> &mapping, bool ignore_metal_coordination);
}
