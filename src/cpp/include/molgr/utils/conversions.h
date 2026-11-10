#pragma once

#include "molgr/utils/utils.h"

#include <memory>

namespace molgr
{
    namespace utils
    {
        OpenBabel::OBMol CloneMolTopologyOnly(const OpenBabel::OBMol &mol);
        OpenBabel::OBMol CloneMolTopologyForScoring(const OpenBabel::OBMol &mol);
        std::shared_ptr<OpenBabel::OBMol> CloneMolTopologyShared(const OpenBabel::OBMol &mol);
        OpenBabel::OBMol MolFromMoleculeData(const MoleculeData &data);
        MoleculeData MoleculeDataFromOBMol(
            const OpenBabel::OBMol &mol,
            bool assume_electron_properties_known);
    }
}
