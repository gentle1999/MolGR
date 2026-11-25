#pragma once

#include <openbabel/mol.h>

#include <memory>
#include <mutex>

namespace molgr::vendor::openbabel_threading
{
    // Native copies and perception can both append to Open Babel's global error log.
    /// @return The shared recursive mutex protecting Open Babel operations with process-wide side effects.
    std::recursive_mutex &NativeOpenBabelMutex();

    /// @param source Molecule supplying the initial topology and electronic labels.
    /// @return An independent Open Babel copy of the source molecule under the native copy lock.
    OpenBabel::OBMol CopyMol(const OpenBabel::OBMol &source);
    /// Replace the target molecule with a copy of the source under the native copy lock.
    /// @param target Destination molecule overwritten by the copy.
    /// @param source Molecule supplying the initial topology and electronic labels.
    void AssignMol(OpenBabel::OBMol &target, const OpenBabel::OBMol &source);
    /// @param source Molecule supplying the initial topology and electronic labels.
    /// @return Shared ownership of a newly copied molecule.
    std::shared_ptr<OpenBabel::OBMol> SharedMolCopy(const OpenBabel::OBMol &source);
}
