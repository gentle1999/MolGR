#pragma once

#include "molgr/utils/utils.h"

namespace molgr::utils
{
    /// Read existing annotations only; do not infer stereo from coordinates.
    /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
    /// @return Extracted stereo frames with molecular indices in the input ordering.
    std::vector<StereoData> StereoDataFromOBMol(const OpenBabel::OBMol &mol);
    /// Apply the supported covalent projection. TB/OH, enhanced groups and
    /// centers whose reference edges were removed remain in MoleculeData only.
    /// @param data Transferable molecular graph records.
    /// @param mol Molecular graph on which the operation acts.
    void ApplyStereoDataToOBMol(const MoleculeData &data, OpenBabel::OBMol &mol);
}
