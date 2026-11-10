#pragma once

#include "molgr/utils/utils.h"

namespace molgr::utils
{
    // Read existing annotations only; do not infer stereo from coordinates.
    std::vector<StereoData> StereoDataFromOBMol(const OpenBabel::OBMol &mol);
    // Apply the supported covalent projection. TB/OH, enhanced groups and
    // centers whose reference edges were removed remain in MoleculeData only.
    void ApplyStereoDataToOBMol(const MoleculeData &data, OpenBabel::OBMol &mol);
}
