#include "molgr/vendor/openbabel_copy.h"

namespace molgr::vendor::openbabel_threading
{
    std::recursive_mutex &NativeOpenBabelMutex()
    {
        static std::recursive_mutex mutex;
        return mutex;
    }

    OpenBabel::OBMol CopyMol(const OpenBabel::OBMol &source)
    {
        std::lock_guard<std::recursive_mutex> lock(NativeOpenBabelMutex());
        // C++17 constructs the result directly while the lock is held.
        return OpenBabel::OBMol(source);
    }

    void AssignMol(OpenBabel::OBMol &target, const OpenBabel::OBMol &source)
    {
        std::lock_guard<std::recursive_mutex> lock(NativeOpenBabelMutex());
        target = source;
    }

    std::shared_ptr<OpenBabel::OBMol> SharedMolCopy(const OpenBabel::OBMol &source)
    {
        std::lock_guard<std::recursive_mutex> lock(NativeOpenBabelMutex());
        return std::make_shared<OpenBabel::OBMol>(source);
    }
}
