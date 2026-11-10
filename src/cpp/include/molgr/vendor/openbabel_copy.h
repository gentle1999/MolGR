#pragma once

#include <openbabel/mol.h>

#include <memory>
#include <mutex>

namespace molgr::vendor::openbabel_threading
{
    // Native copies and perception can both append to Open Babel's global error log.
    std::recursive_mutex &NativeOpenBabelMutex();

    OpenBabel::OBMol CopyMol(const OpenBabel::OBMol &source);
    void AssignMol(OpenBabel::OBMol &target, const OpenBabel::OBMol &source);
    std::shared_ptr<OpenBabel::OBMol> SharedMolCopy(const OpenBabel::OBMol &source);
}
